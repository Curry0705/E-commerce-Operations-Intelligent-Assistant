from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from file_history_store import get_history
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from sentence_transformers import CrossEncoder
from bm25_store import BM25Store
import memory_manager
import sys

def print_prompt(prompt):
    prompt_str = prompt.to_string()
    sys.stderr.write("=" * 20 + "\n")
    sys.stderr.write(prompt_str + "\n")
    sys.stderr.write("=" * 20 + "\n")
    sys.stderr.flush()
    return prompt


class RagService(object):
    def __init__(self):

        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(dashscope_api_key=config.dashscope_api_key,model=config.embedding_model_name)
        )

        self.bm25_store = BM25Store(self.vector_service.vector_store._collection)

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知参考资料为主，简洁和专业地回答用户问题。"
                 "如果发现参考资料中没有相关信息，请明确说明。参考资料:\n{context}\n"
                 "而且要注意，回答中不要出现'（见表X-X）'、'（见图X-X）'等图表编号引用字样，直接描述图表中的具体信息即可。"
                 "生成内容时禁止引用参考资料的章节编号、页码或内部标记，如'参考XX''见第X节'等。"
                 "请将信息自然地融入到正文，引用参考资料时在开头体现使用什么资料即可。"),
                ("system", "并且我提供用户的对话历史记录，如下："),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问：{input}")
            ]
        )

        self.chat_model = ChatTongyi(
            api_key=config.api_key,
            model=config.chat_model_name,
            streaming=True,
        )

        # 查询改写专用模型（非流式，快速响应）
        self._rewrite_model = None
        # Reranker 模型
        self._reranker = None

        self.chain = self.__get_chain()

    @property
    def rewrite_model(self):
        """懒加载查询改写模型"""
        if self._rewrite_model is None:
            self._rewrite_model = ChatTongyi(
                api_key=config.api_key,
                model=config.chat_model_name,
                streaming=False,
            )
        return self._rewrite_model

    _REWRITE_PROMPT = ChatPromptTemplate.from_messages([
        ("system", (
            "你是一个查询改写助手。根据对话历史，将用户的当前问题改写为一个独立完整、"
            "无需依赖上下文就能理解的问题。"
            "如果问题本身已经足够清晰，直接返回原问题。"
            "只输出改写后的问题，不要加任何解释或标记。"
        )),
        ("user", (
            "对话历史：\n"
            "{history}\n\n"
            "用户当前问题：{query}\n\n"
            "改写后的问题："
        )),
    ])

    def _rewrite_query(self, query: str, history: list) -> str:
        """结合对话历史，将省略/指代型 query 改写为独立完整的问题"""
        if not history:
            return query

        # 将历史消息序列化为可读文本
        history_lines = []
        for msg in history:
            role = "用户" if msg.type == "human" else "助手"
            history_lines.append(f"[{role}]: {msg.content}")
        history_text = "\n".join(history_lines)

        try:
            chain = self._REWRITE_PROMPT | self.rewrite_model | StrOutputParser()
            rewritten = chain.invoke({"history": history_text, "query": query})
            if rewritten and rewritten.strip():
                sys.stderr.write(f"[Query Rewrite] {query!r} -> {rewritten!r}\n")
                sys.stderr.flush()
                return rewritten.strip()
        except Exception:
            pass  # 改写失败则使用原始 query

        return query

    @property
    def reranker(self):
        """懒加载 BGE Reranker 模型"""
        if self._reranker is None:
            self._reranker = CrossEncoder(
                config.rerank_model_path,
                max_length=512,
                local_files_only=True,
            )
        return self._reranker

    def _rerank(self, query: str, docs: list[Document]) -> list[Document]:
        """对粗排候选文档做精排，返回 top-k 文档"""
        k = config.rerank_top_k
        if len(docs) <= k:
            return docs

        pairs = [[query, doc.page_content] for doc in docs]
        try:
            scores = self.reranker.predict(pairs)
        except Exception:
            return docs[:k]  # rerank 失败则截断返回原始顺序

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:k]]

    def _rrf_fusion(
        self,
        semantic_docs: list[Document],
        bm25_results: list[tuple[Document, float]],
        k: int = 60,
    ) -> list[Document]:
        """RRF 融合语义检索和关键词检索结果"""
        if not bm25_results:
            return semantic_docs
        if not semantic_docs:
            return [doc for doc, _ in bm25_results]

        scores = {}
        doc_map = {}

        def _key(doc: Document) -> str:
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

    def __get_chain(self):
        """获取最终的执行链"""

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"

            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"
            return formatted_str

        def _retrieve_and_format(value: dict) -> str:
            """检索文本 + 图片，合并格式化"""
            query = value["input"]
            history = value.get("history", [])
            # 查询改写：将指代/省略问题转为独立完整问题
            query = self._rewrite_query(query, history)
            # 长期记忆：追踪用户话题偏好
            memory_manager.track_query_topics(
                str(value.get("user_id", "user_001")),
                str(value.get("session_id", "default")),
                query,
            )
            # 文本检索：混合检索（语义 + BM25）→ RRF 融合 → 精排重排序
            semantic_docs = self.vector_service.similarity_search(
                query, k=config.hybrid_candidate_k
            )
            bm25_results = self.bm25_store.search(
                query, k=config.hybrid_candidate_k
            )
            fused_docs = self._rrf_fusion(
                semantic_docs, bm25_results, k=config.hybrid_rrf_k
            )
            text_docs = self._rerank(query, fused_docs)
            # 图片检索：CLIP 语义 + 按文字来源精确匹配
            image_docs = self.vector_service.search_images(query)
            source_pages = set()
            for doc in text_docs:
                src = doc.metadata.get("source", "")
                pns_raw = doc.metadata.get("page_nums", "")
                if not pns_raw:
                    continue
                for pn in str(pns_raw).split(","):
                    pn = pn.strip()
                    if src and pn:
                        source_pages.add((src, int(pn)))
            source_pages = list(source_pages)
            if source_pages:
                matched_images = self.vector_service.search_images_by_pages(source_pages)
                seen = {img.page_content for img in image_docs}
                for img in matched_images:
                    if img.page_content not in seen:
                        image_docs.append(img)

            parts = []
            # 用户上传附件（与知识库召回平级，排在参考资料首位）
            file_context = value.get("file_context", "")
            if file_context:
                parts.append(file_context)
            # 短期记忆：超出窗口的对话摘要
            summary = memory_manager.get_or_update_summary(
                value.get("session_id", "default"), history
            )
            if summary:
                parts.append(f"【对话历史摘要】{summary}\n")
            # 长期记忆：用户话题偏好
            profile = memory_manager.get_user_profile(
                str(value.get("user_id", "user_001"))
            )
            if profile["topics"]:
                top_topics = list(profile["topics"].keys())[:5]
                parts.append(f"【用户关注话题】{', '.join(top_topics)}\n")

            parts.append(format_document(text_docs))
            if image_docs:
                parts.append("【相关图片信息】")
                for img_doc in image_docs:
                    parts.append(
                        f"图片说明：{img_doc.page_content}"
                        f"（来源：{img_doc.metadata.get('source', '未知')}，"
                        f"第 {img_doc.metadata.get('page_num', '?')} 页）"
                    )
            return "\n".join(parts)

        def format_for_prompt_template(value):
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(_retrieve_and_format),
            } | RunnableLambda(format_for_prompt_template) | self.prompt_template | print_prompt | self.chat_model | StrOutputParser()
        )
        #带有历史记录的链
        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain
