from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from langsmith import traceable
from src.vector_store import FiassVectorStore
from src.config import settings
from sentence_transformers import CrossEncoder

# helper to convert the text to lowercase and split it into individual words 
def _tokenize(text: str) -> List[str]:
    return text.lower().split()

def reciprocal_rank_fusion(rankings: List[List[int]], k: int = 60) -> List[int]:
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


class HybridRetriever:
    def __init__(self, vectorstore: FiassVectorStore):
        self.vectorstore = vectorstore
        corpus = [m["text"] for m in vectorstore.metadata]
        self.bm25 = BM25Okapi([_tokenize(t) for t in corpus])
        self.reranker = CrossEncoder(settings.reranker_model)

    @traceable(run_type="retriever", name="hybrid_retrieve")
    def retrieve(self, query: str, k: int | None = None) -> List[Dict[str, Any]]:
        k = k or settings.top_k
        fetch_k = k * 4
        dense_results = self.vectorstore.query(query, top_k=fetch_k)
        dense_ids = [r["index"] for r in dense_results]
        bm25_scores = self.bm25.get_scores(_tokenize(query))
        sparse_ids = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:fetch_k]
        fused_ids = reciprocal_rank_fusion([dense_ids, sparse_ids])[:fetch_k]
        return self._rerank(query, fused_ids, k)
    
    @traceable(run_type="retriever", name="rerank")
    def _rerank(self, query: str, chunk_ids: List[int], k: int) -> List[Dict[str, Any]]:
        candidates = [self.vectorstore.metadata[i] for i in chunk_ids]
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:k]]

