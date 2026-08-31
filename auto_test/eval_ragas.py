"""
RAGAS 全体系评估

基于 LLM-as-Judge 的 RAG 质量评估，包含四个核心指标：

无参考指标（无需 ground_truth）：
- Faithfulness      — 答案是否忠实地基于检索上下文，有无幻觉
- AnswerRelevancy   — 答案与问题的相关程度

有参考指标（需要 ground_truth / reference_answer）：
- ContextPrecision  — 检索到的上下文中有多少真正有助于回答问题
- ContextRecall     — 参考答案中的信息有多少被检索上下文覆盖

用法：
    python auto_test/eval_ragas.py                          # 全量评估
    python auto_test/eval_ragas.py --max-cases 10           # 快速验证
    python auto_test/eval_ragas.py --metrics faithfulness,answer_relevancy  # 只评估指定指标
"""
import warnings
warnings.filterwarnings("ignore")
import json
import os
import sys
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import redis
from datasets import Dataset

from sentence_transformers import CrossEncoder
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate

from vector_stores import VectorStoreService
from bm25_store import BM25Store

import config_data as config


# ---- 模块级检索单例（懒加载，复用实例避免重复初始化）----
_vector_service = None
_bm25_store = None
_reranker = None


def _get_vector_service():
    global _vector_service
    if _vector_service is None:
        embedding_fn = DashScopeEmbeddings(
            dashscope_api_key=config.dashscope_api_key,
            model=config.embedding_model_name,
        )
        _vector_service = VectorStoreService(embedding=embedding_fn)
    return _vector_service


def _get_bm25_store():
    global _bm25_store
    if _bm25_store is None:
        vs = _get_vector_service()
        _bm25_store = BM25Store(vs.vector_store._collection)
    return _bm25_store


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(
            config.rerank_model_path,
            max_length=512,
            local_files_only=True,
        )
    return _reranker


def _rrf_fusion(semantic_docs, bm25_results, k):
    """RRF 融合语义检索和关键词检索结果"""
    if not bm25_results:
        return semantic_docs
    if not semantic_docs:
        return [doc for doc, _ in bm25_results]

    scores = {}
    doc_map = {}

    def _key(doc):
        return f"{doc.page_content[:100]}|{doc.metadata.get('source', '')}"

    for rank, doc in enumerate(semantic_docs, 1):
        key = _key(doc)
        scores[key] = scores.get(key, 0) + 1 / (k + rank)
        doc_map[key] = doc

    for rank, (doc, _) in enumerate(bm25_results, 1):
        key = _key(doc)
        scores[key] = scores.get(key, 0) + 1 / (k + rank)
        if key not in doc_map:
            doc_map[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked]


def _rerank(query, docs):
    """对粗排候选文档做精排，返回 top-k"""
    k = config.rerank_top_k
    if len(docs) <= k:
        return docs

    pairs = [[query, doc.page_content] for doc in docs]
    try:
        scores = _get_reranker().predict(pairs)
    except Exception:
        return docs[:k]

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:k]]


def load_testset_from_redis() -> list[dict]:
    r = redis.Redis(
        host=config.redis_host,
        port=config.redis_port,
        db=1,
        password=config.redis_password,
        decode_responses=True,
    )
    test_cases = []
    i = 1
    while True:
        raw = r.get(str(i))
        if raw is None:
            break
        test_cases.append(json.loads(raw))
        i += 1
    if not test_cases:
        raise RuntimeError("Redis (db=1) 中无测试数据，请先运行 generate_testset.py")
    return test_cases


def load_testset_from_json(path: str = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "testset.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_testset(path: str = None) -> tuple[list[dict], dict]:
    """加载测试集，优先 Redis (db=1)，回退 JSON"""
    try:
        test_cases = load_testset_from_redis()
        diff_count = {"easy": 0, "medium": 0, "hard": 0}
        for tc in test_cases:
            diff_count[tc.get("difficulty", "unknown")] += 1
        metadata = {"source": "redis", "db": 1, "difficulty_distribution": diff_count}
        print("测试集来源: Redis (db=1)")
        return test_cases, metadata
    except Exception:
        testset = load_testset_from_json(path)
        test_cases = testset["test_cases"]
        metadata = testset["metadata"]
        print("测试集来源: JSON 文件")
        return test_cases, metadata


def get_retrieved_context(query: str) -> list[str]:
    """混合检索：语义 + BM25 → RRF 融合 → Reranker 精排，与生产环境一致"""
    vs = _get_vector_service()
    bm25 = _get_bm25_store()

    semantic_docs = vs.similarity_search(query, k=config.hybrid_candidate_k)
    bm25_results = bm25.search(query, k=config.hybrid_candidate_k)
    fused_docs = _rrf_fusion(semantic_docs, bm25_results, k=config.hybrid_rrf_k)
    final_docs = _rerank(query, fused_docs)

    return [doc.page_content for doc in final_docs]


def generate_answer(question: str, contexts: list[str]) -> str:
    context_str = ""
    for i, ctx in enumerate(contexts, 1):
        context_str += f"[{i}] {ctx}\n\n"
    if not context_str:
        context_str = "无相关参考资料"

    prompt = ChatPromptTemplate.from_messages([
        ("system", "以我提供的已知参考资料为主，简洁和专业地回答用户问题。"
         "如果发现参考资料中没有相关信息，请明确说明。参考资料:\n{context}\n"
         "而且要注意，回答中不要出现'（见表X-X）'、'（见图X-X）'等图表编号引用字样，直接描述图表中的具体信息即可。"
         "生成内容时禁止引用参考资料的章节编号、页码或内部标记，如'参考XX''见第X节'等。请将信息自然地融入到正文。"),
        ("user", "请回答用户提问：{input}"),
    ])

    llm = ChatTongyi(
        api_key=config.api_key,
        model=config.chat_model_name,
        streaming=False,
        temperature=0.0,
    )

    chain = prompt | llm
    response = chain.invoke({"context": context_str, "input": question})
    return response.content


def evaluate_ragas(
    testset_path: str = None,
    output_path: str = None,
    metrics: list[str] = None,
    max_cases: int = None,
):
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "ragas_results.json")
    if metrics is None:
        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    print("=" * 60)
    print("RAGAS 全体系评估")
    print(f"指标: {metrics}")
    print("=" * 60)

    # ---- Step 1: 加载测试集 ----
    test_cases, metadata = load_testset(testset_path)
    if max_cases:
        test_cases = test_cases[:max_cases]
    print(f"\n[1/4] 加载测试集: {len(test_cases)} 条用例")

    # ---- Step 2: 执行 RAG pipeline ----
    print(f"\n[2/4] 执行 RAG pipeline ({len(test_cases)} 条)...")
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    for i, tc in enumerate(test_cases):
        q = tc["question"]
        print(f"  [{i + 1}/{len(test_cases)}] {q[:60]}...")

        try:
            ctx = get_retrieved_context(q)
            ans = generate_answer(q, ctx)
        except Exception as e:
            print(f"    [错误] {e}")
            ctx = ["(检索失败)"]
            ans = "(生成失败)"

        questions.append(q)
        answers.append(ans)
        contexts_list.append(ctx)
        ground_truths.append(tc["reference_answer"])

    # ---- Step 3: 构建 RAGAS Dataset ----
    print(f"\n[3/4] 构建 RAGAS 评估数据集...")
    # RAGAS 0.4+ 列名规范: user_input / response / retrieved_contexts / reference
    dataset_dict = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts_list,
    }

    need_gt = any(m in metrics for m in ["context_precision", "context_recall"])
    if need_gt:
        dataset_dict["reference"] = ground_truths

    eval_dataset = Dataset.from_dict(dataset_dict)
    print(f"  数据集大小: {len(eval_dataset)}")
    print(f"  字段: {list(dataset_dict.keys())}")

    # ---- Step 4: 配置 RAGAS 并评估 ----
    print(f"\n[4/4] 配置 RAGAS 并执行评估...")

    evaluator_llm = ChatTongyi(
        api_key=config.api_key,
        model=config.eval_model_name,
        streaming=False,
        temperature=0.0,
    )

    evaluator_embeddings = DashScopeEmbeddings(
        dashscope_api_key=config.dashscope_api_key,
        model=config.embedding_model_name,
    )

    try:
        from ragas.llms import LangchainLLMWrapper
        ragas_llm = LangchainLLMWrapper(evaluator_llm)
    except ImportError:
        ragas_llm = evaluator_llm

    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        ragas_embeddings = LangchainEmbeddingsWrapper(evaluator_embeddings)
    except ImportError:
        ragas_embeddings = evaluator_embeddings

    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    metric_map = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }

    ragas_metrics = []
    for m in metrics:
        if m in metric_map:
            ragas_metrics.append(metric_map[m])
        else:
            print(f"  [警告] 未知指标: {m}")

    print(f"======待评估指标: {[m.name for m in ragas_metrics]}======")

    from ragas import evaluate as ragas_evaluate

    try:
        result = ragas_evaluate(
            dataset=eval_dataset,
            metrics=ragas_metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        df = result.to_pandas()
    except Exception as e:
        print(f"  [错误] RAGAS 评估失败: {e}")
        import traceback
        traceback.print_exc()
        result = None
        df = None

    # ---- 汇总输出 ----
    print("\n" + "=" * 60)
    print("RAGAS 评估结果")
    print("=" * 60)

    summary = {}
    by_difficulty = {}
    if df is not None:
        skip_cols = {"user_input", "response", "retrieved_contexts", "reference"}
        print(f"\n{'指标':<22} {'均值':>8}")
        print("-" * 32)
        for col in df.columns:
            if col not in skip_cols:
                try:
                    mean_val = float(df[col].mean())
                    summary[col] = round(mean_val, 4)
                    print(f"{col:<22} {mean_val:>8.4f}")
                except Exception:
                    pass

        # 按难度分组
        difficulties = [tc.get("difficulty", "unknown") for tc in test_cases]
        diff_groups = defaultdict(list)
        for i, diff in enumerate(difficulties):
            diff_groups[diff].append(i)

        for diff, indices in diff_groups.items():
            entry = {"count": len(indices)}
            for col in df.columns:
                if col not in skip_cols:
                    try:
                        entry[col] = round(float(df.iloc[indices][col].mean()), 4)
                    except Exception:
                        pass
            by_difficulty[diff] = entry

        print(f"\n按难度分组:")
        metric_cols = [c for c in df.columns if c not in skip_cols]
        header = f"{'难度':<10} {'数量':>6}"
        for col in metric_cols:
            header += f" {col:>10}"
        print(header)
        print("-" * len(header))
        for diff in ["easy", "medium", "hard"]:
            if diff in by_difficulty:
                d = by_difficulty[diff]
                row = f"{diff:<10} {d['count']:>6}"
                for col in metric_cols:
                    if col in d:
                        row += f" {d[col]:>10.4f}"
                print(row)

    # 保存结果
    per_case = []
    for i, tc in enumerate(test_cases):
        case = {
            "id": tc["id"],
            "question": tc["question"],
            "reference_answer": tc["reference_answer"],
            "generated_answer": answers[i],
            "retrieved_contexts": contexts_list[i],
            "difficulty": tc.get("difficulty", "unknown"),
        }
        if df is not None:
            for col in df.columns:
                if col not in ("user_input", "response", "retrieved_contexts", "reference"):
                    try:
                        case[col] = round(float(df.iloc[i][col]), 4)
                    except Exception:
                        pass
        per_case.append(case)

    results = {
        "testset_metadata": metadata,
        "metrics_evaluated": metrics,
        "summary": summary,
        "by_difficulty": by_difficulty,
        "per_case": per_case,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存至: {output_path}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS RAG 全体系评估")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--metrics", type=str, default="faithfulness,answer_relevancy,context_precision,context_recall")
    args = parser.parse_args()
    evaluate_ragas(metrics=args.metrics.split(","), max_cases=args.max_cases)
