from langchain_chroma import Chroma
from langchain_core.documents import Document
from knowledge_base import _embed_text_for_image_search
import config_data as config


class VectorStoreService(object):
    def __init__(self, embedding):
        """
        :param embedding: 嵌入模型的传入
        """
        self.embedding = embedding

        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

        # 图片向量库（仅读取，不在此处写入）
        self.image_store = Chroma(
            collection_name=config.collection_name + "_images",
            embedding_function=None,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self):
        """返回向量检索器，方便加入chain"""
        return self.vector_store.as_retriever(search_kwargs={"k": config.similarity_threshold})

    def similarity_search(self, query: str, k: int = None) -> list:
        """按指定 k 做语义检索，返回 Document 列表（用于 reranking 等需要更多候选的场景）"""
        if k is None:
            k = config.similarity_threshold
        return self.vector_store.similarity_search(query, k=k)

    def search_images(self, query: str, k: int = None) -> list[Document]:
        """将查询文本映射到图片向量空间，检索相关图片"""
        if k is None:
            k = config.similarity_threshold

        try:
            query_embedding = _embed_text_for_image_search(query)
        except Exception:
            return []  # 图片检索失败时静默降级

        results = self.image_store._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        docs = []
        if results and results.get("ids") and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                docs.append(Document(
                    page_content=results["documents"][0][i] if results.get("documents") else "",
                    metadata=results["metadatas"][0][i] if results.get("metadatas") else {},
                ))
        return docs

    def search_images_by_pages(
        self, source_page_pairs: list[tuple[str, int]]
    ) -> list[Document]:
        """按 (source, page_num) 精确查询图片"""
        if not source_page_pairs:
            return []
        docs = []
        for source, page_num in source_page_pairs:
            try:
                results = self.image_store._collection.get(
                    where={"$and": [{"source": source}, {"page_num": page_num}]},
                    include=["documents", "metadatas"],
                )
            except Exception:
                continue
            if results and results.get("ids"):
                for i in range(len(results["ids"])):
                    docs.append(Document(
                        page_content=results["documents"][i],
                        metadata=results["metadatas"][i],
                    ))
        return docs




