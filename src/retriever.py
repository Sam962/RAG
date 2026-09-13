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


def select_relevant_chunks(
    scored_chunks: List[tuple[float, Dict[str, Any]]],
    k: int,
    min_score: float | None = None,
    score_margin: float | None = None,
) -> List[Dict[str, Any]]:
    """Keep top-k chunks that clear a floor, or a unique winner even if below it.

    ms-marco scores are often negative for specific but correct hits. An absolute
    floor of 0 would drop those and answer 'I don't know'. If the best hit is
    isolated (the rest of the list is more than `score_margin` worse), keep it.
    A tight cluster of equally weak scores is treated as no evidence.
    """
    min_score = settings.min_rerank_score if min_score is None else min_score
    score_margin = settings.rerank_score_margin if score_margin is None else score_margin
    if not scored_chunks:
        return []
    all_ranked = sorted(scored_chunks, key=lambda item: item[0], reverse=True)
    ranked = all_ranked[:k]
    best_score = float(ranked[0][0])
    isolated = any(
        float(score) < best_score - score_margin for score, _ in all_ranked
    )
    kept: List[Dict[str, Any]] = []
    for score, chunk in ranked:
        score = float(score)
        if score < best_score - score_margin:
            continue
        if score >= min_score or isolated:
            kept.append({**chunk, "rerank_score": score})
    return kept


def expand_adjacent_chunks(
    kept: List[Dict[str, Any]],
    metadata: List[Dict[str, Any]],
    window: int = 1,
) -> List[Dict[str, Any]]:
    """Add same-source neighbors so a hit on a header also brings job dates."""
    if not kept or window <= 0:
        return kept
    by_id = {m.get("chunk_id"): m for m in metadata if m.get("chunk_id") is not None}
    selected = {c.get("chunk_id") for c in kept}
    extras: List[Dict[str, Any]] = []
    for chunk in kept:
        cid = chunk.get("chunk_id")
        source = chunk.get("source")
        if cid is None:
            continue
        for nid in range(int(cid) - window, int(cid) + window + 1):
            if nid in selected or nid not in by_id:
                continue
            neighbor = by_id[nid]
            if neighbor.get("source") != source:
                continue
            extras.append({**neighbor, "rerank_score": chunk.get("rerank_score")})
            selected.add(nid)
    merged = kept + extras
    return sorted(
        merged,
        key=lambda c: (c.get("chunk_id") is None, c.get("chunk_id") or 0),
    )


class HybridRetriever:
    def __init__(self, vectorstore: FiassVectorStore):
        self.vectorstore = vectorstore
        corpus = [m["text"] for m in vectorstore.metadata]
        if not corpus:
            raise ValueError(
                "Vector store has no chunks. Ingest documents before retrieving."
            )
        self.bm25 = BM25Okapi([_tokenize(t) for t in corpus])
        self.reranker = CrossEncoder(settings.reranker_model)

    def retrieve_fused_ids(self, query: str, fetch_k: int | None = None) -> List[int]:
        k = settings.top_k
        fetch_k = fetch_k or k * 4
        n = len(self.vectorstore.metadata)
        dense_results = self.vectorstore.query(query, top_k=fetch_k)
        dense_ids = [r["index"] for r in dense_results if 0 <= r["index"] < n]
        bm25_scores = self.bm25.get_scores(_tokenize(query))
        sparse_ids = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:fetch_k]
        return [
            i for i in reciprocal_rank_fusion([dense_ids, sparse_ids])[:fetch_k]
            if 0 <= i < n
        ]

    @traceable(run_type="retriever", name="hybrid_retrieve")
    def retrieve(self, query: str, k: int | None = None) -> List[Dict[str, Any]]:
        k = k or settings.top_k
        return self._rerank(query, self.retrieve_fused_ids(query, fetch_k=k * 4), k)
    
    @traceable(run_type="retriever", name="rerank")
    def _rerank(self, query: str, chunk_ids: List[int], k: int) -> List[Dict[str, Any]]:
        n = len(self.vectorstore.metadata)
        candidates = [
            self.vectorstore.metadata[i] for i in chunk_ids if 0 <= i < n
        ]
        if not candidates:
            return []
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.reranker.predict(pairs)
        kept = select_relevant_chunks(list(zip(scores, candidates)), k)
        return expand_adjacent_chunks(kept, self.vectorstore.metadata, window=2)

