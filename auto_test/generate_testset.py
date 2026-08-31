"""
自动生成 RAG 评估测试集

从 Chroma 向量库中采样文档块，利用 LLM 为每个块生成问题 + 参考答案。
源文档块用于生成问题和参考答案，形成 (question, reference_answer) 测试对。
"""
import json
import os
import random
import sys
import time
from datetime import datetime

import redis

# 将项目根目录加入 path，方便导入 config 和 vector_stores
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
import config_data as config



QUESTION_GEN_PROMPT = """你是一个测试集生成专家。请根据下面提供的文档片段，生成 {num_questions} 个该片段能够回答的问题。

要求：
1. 问题应多样化，涵盖事实查询、原因分析、操作步骤等类型
2. 问题应贴近真实用户的提问方式，避免过于模板化
3. 难度分布：简单（直接可从片段找到答案）、中等（需要理解整合）、困难（需要推理）
4. 每个问题都要附带一个基于该片段的参考答案

文档片段：
{chunk_content}

请严格按照以下 JSON 格式返回（不要输出其他内容）：
```json
[
  {{
    "question": "问题内容",
    "reference_answer": "参考答案",
    "difficulty": "easy|medium|hard"
  }}
]
```"""


def load_all_chunks() -> list[dict]:
    """从 Chroma 向量库加载所有文档块，返回带 id 和内容的列表"""
    vector_store = Chroma(
        collection_name=config.collection_name,
        embedding_function=DashScopeEmbeddings(
            dashscope_api_key=config.dashscope_api_key,
            model=config.embedding_model_name,
        ),
        persist_directory=config.persist_directory,
    )

    results = vector_store.get(include=["documents", "metadatas"])
    chunks = []
    if results and results.get("ids"):
        for i, chunk_id in enumerate(results["ids"]):
            chunks.append({
                "id": chunk_id,
                "content": results["documents"][i] if results["documents"] else "",
                "metadata": results["metadatas"][i] if results["metadatas"] else {},
            })
    return chunks


def parse_llm_json_response(text: str) -> list[dict]:
    """从 LLM 回复中解析 JSON，支持 markdown 代码块包裹的情况"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # 去掉第一行和最后一行 ```
        text = "\n".join(lines[1:]) if lines[0].startswith("```") else text
        if text.endswith("```"):
            text = text[:-3]
    return json.loads(text)


def generate_questions_for_chunk(
    llm: ChatTongyi,
    chunk: dict,
    num_questions: int = 3,
    max_retries: int = 3,
) -> list[dict]:
    """为单个文档块生成问题和参考答案，失败重试"""
    prompt = QUESTION_GEN_PROMPT.format(
        num_questions=num_questions,
        chunk_content=chunk["content"],
    )

    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            items = parse_llm_json_response(response.content)
            for item in items:
                item["source_chunk_id"] = chunk["id"]
            return items
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
            else:
                print(f"  [错误] chunk {chunk['id'][:8]} 生成失败: {e}")
                return []
    return []


def _save_to_redis(test_cases: list[dict]):
    """将测试用例逐条存入 Redis (db=1), key 为序号 1/2/3..."""
    try:
        r = redis.Redis(
        host=config.redis_host,
        port=config.redis_port,
        db=1,
        password=config.redis_password,
        decode_responses=True,
        )
        r.ping()
    except redis.ConnectionError as e:
        print(f"  [警告] Redis (db=1) 连接失败: {e}，跳过 Redis 存储")
        return

    # 先清空 db=1 中的旧测试数据
    r.flushdb()

    pipe = r.pipeline()
    for tc in test_cases:
        pipe.set(str(tc["id"]), json.dumps(tc, ensure_ascii=False))
    pipe.execute()

    print(f"  已存入 Redis (db=1): {len(test_cases)} 条, key 范围 1~{len(test_cases)}")


def generate_testset(
    num_sample_chunks: int,
    questions_per_chunk: int,
    output_path: str = None,
    random_seed: int = 42):
    """
    主流程：从知识库采样文档块，生成测试集

    :param num_sample_chunks: 采样的文档块数量
    :param questions_per_chunk: 每个块生成的问题数
    :param output_path: 输出 JSON 路径，默认 auto_test/testset.json
    :param random_seed: 随机种子
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "testset.json")

    random.seed(random_seed)

    print("=" * 60)
    print("RAG 测试集自动生成")
    print("=" * 60)

    # 1. 加载所有文档块
    print("\n[1/4] 从 Chroma 加载文档块...")
    all_chunks = load_all_chunks()
    print(f"  共加载 {len(all_chunks)} 个文档块")

    if len(all_chunks) == 0:
        print("  [错误] 知识库为空，请先上传文档")
        return None

    # 2. 采样
    actual_sample = min(num_sample_chunks, len(all_chunks))
    sampled_chunks = random.sample(all_chunks, actual_sample)
    print(f"\n[2/4] 随机采样 {actual_sample} 个文档块")

    # 3. 初始化 LLM（非流式，用于批量生成）
    print(f"\n[3/4] 初始化 LLM ({config.generate_question_name}) 生成测试问题...")
    llm = ChatTongyi(
        api_key=config.api_key,
        model=config.generate_question_name,
        streaming=False,
        temperature=0.8,  # 较高温度以增加问题多样性
    )

    test_cases = []
    success_count = 0
    fail_count = 0

    for i, chunk in enumerate(sampled_chunks):
        chunk_preview = chunk["content"][:60].replace("\n", " ")
        print(f"  [{i + 1}/{actual_sample}] chunk {chunk['id'][:8]}... [{chunk_preview}...]")
        items = generate_questions_for_chunk(llm, chunk, num_questions=questions_per_chunk)

        for idx, item in enumerate(items):
            test_cases.append({
                "id": len(test_cases) + 1,
                "question": item["question"],
                "reference_answer": item["reference_answer"],
                "difficulty": item.get("difficulty", "medium"),
            })

        if items:
            success_count += 1
        else:
            fail_count += 1

        # API 限流保护
        if i < actual_sample - 1:
            time.sleep(0.5)

    print(f"\n  生成完成: 成功 {success_count} 个块, 失败 {fail_count} 个块, 共 {len(test_cases)} 条测试用例")

    # 4. 保存测试集
    print(f"\n[4/4] 保存测试集到 Redis 和 {output_path}")

    # 5a. 写入 Redis (db=1, key 为 1/2/3...)
    _save_to_redis(test_cases)

    # 5b. 写入 JSON 文件

    # 统计难度分布
    diff_count = {"easy": 0, "medium": 0, "hard": 0}
    for tc in test_cases:
        diff_count[tc["difficulty"]] = diff_count.get(tc["difficulty"], 0) + 1

    testset = {
        "metadata": {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_test_cases": len(test_cases),
            "num_sample_chunks": actual_sample,
            "questions_per_chunk": questions_per_chunk,
            "total_chunks_in_db": len(all_chunks),
            "random_seed": random_seed,
            "difficulty_distribution": diff_count,
        },
        "test_cases": test_cases,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(testset, f, ensure_ascii=False, indent=2)

    print(f"  已保存 {len(test_cases)} 条测试用例")
    print(f"  难度分布: {diff_count}")
    print("\n" + "=" * 60)
    print("测试集生成完成!")
    print("=" * 60)

    return testset


if __name__ == "__main__":
    generate_testset(
        num_sample_chunks=config.num_sample_chunks,
        questions_per_chunk=config.questions_per_chunk,
        random_seed=42,
    )
