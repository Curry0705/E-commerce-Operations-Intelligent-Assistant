"""BM25 关键词检索引擎，基于 Chroma 文档集合构建索引"""
import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


class BM25Store:
    def __init__(self, chroma_collection):
        self._chroma = chroma_collection
        self._bm25 = None
        self._documents = []
        self._doc_count = 0
        self._build()

    def _tokenize(self, text: str) -> list[str]:
        return list(jieba.cut(text))

    def _build(self):
        results = self._chroma.get(include=["documents", "metadatas"])
        if results and results.get("ids"):
            corpus = []
            for i in range(len(results["ids"])):
                doc_text = results["documents"][i]
                meta = results["metadatas"][i] if results.get("metadatas") else {}
                self._documents.append(Document(page_content=doc_text, metadata=meta))
                corpus.append(self._tokenize(doc_text))
            self._bm25 = BM25Okapi(corpus) if corpus else None
            self._doc_count = len(self._documents)

    def _auto_rebuild(self):
        """检查 Chroma 文档数是否变化，变化则自动重建"""
        current_count = self._chroma.count()
        if current_count != self._doc_count:
            self._documents = []
            self._bm25 = None
            self._doc_count = 0
            self._build()

    def search(self, query: str, k: int = 6) -> list[tuple[Document, float]]:
        """搜索并返回 [(Document, bm25_score), ...]"""
        self._auto_rebuild()
        if not self._bm25:
            return []
        tokenized = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized)
        top_k = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        return [(self._documents[i], score) for i, score in top_k if score > 0]

    def rebuild(self):
        """强制重建索引"""
        self._documents = []
        self._bm25 = None
        self._doc_count = 0
        self._build()
