from unittest.mock import MagicMock, patch

import pytest

from src.retriever import HybridRetriever, expand_adjacent_chunks, select_relevant_chunks
from src.vector_store import FiassVectorStore


def test_hybrid_retriever_rejects_empty_store(tmp_path):
    with patch("src.vector_store.SentenceTransformer", return_value=MagicMock()):
        store = FiassVectorStore(persist_dir=str(tmp_path))
    with pytest.raises(ValueError, match="no chunks"):
        HybridRetriever(store)


def test_select_relevant_chunks_drops_weak_and_distant_scores():
    strong = {"text": "beam search", "chunk_id": 9, "page": 4}
    mid = {"text": "encoder decoder", "chunk_id": 53, "page": 47}
    weak = {"text": "positional encoding", "chunk_id": 22, "page": 16}
    kept = select_relevant_chunks(
        [(7.2, strong), (0.4, mid), (-3.1, weak)],
        k=3,
        min_score=0.0,
        score_margin=3.0,
    )
    assert [c["chunk_id"] for c in kept] == [9]
    assert kept[0]["rerank_score"] == 7.2


def test_select_relevant_chunks_keeps_close_strong_hits():
    a = {"text": "beam search", "chunk_id": 1}
    b = {"text": "greedy decoding", "chunk_id": 2}
    kept = select_relevant_chunks(
        [(6.0, a), (5.2, b)],
        k=3,
        min_score=0.0,
        score_margin=3.0,
    )
    assert [c["chunk_id"] for c in kept] == [1, 2]


def test_select_relevant_chunks_all_weak_returns_empty():
    kept = select_relevant_chunks(
        [(-1.0, {"chunk_id": 1}), (-4.0, {"chunk_id": 2})],
        k=3,
        min_score=0.0,
        score_margin=3.0,
    )
    assert kept == []


def test_select_relevant_chunks_keeps_isolated_winner_below_floor():
    """Specific queries often score the true doc below 0; drop only weak blobs."""
    hit = {"text": "role at employer", "chunk_id": 0, "source": "resume.pdf"}
    junk = {"text": "unrelated interview q", "chunk_id": 9, "source": "other.pdf"}
    kept = select_relevant_chunks(
        [(-2.5, hit), (-11.0, junk), (-11.2, {**junk, "chunk_id": 10})],
        k=3,
        min_score=0.0,
        score_margin=3.0,
    )
    assert [c["chunk_id"] for c in kept] == [0]
    assert kept[0]["rerank_score"] == -2.5


def test_select_relevant_chunks_drops_flat_negative_blob():
    kept = select_relevant_chunks(
        [(-11.14, {"chunk_id": 1}), (-11.15, {"chunk_id": 2}), (-11.16, {"chunk_id": 3})],
        k=3,
        min_score=0.0,
        score_margin=3.0,
    )
    assert kept == []


def test_expand_adjacent_chunks_adds_same_source_neighbors():
    metadata = [
        {"chunk_id": 0, "source": "resume.pdf", "text": "summary"},
        {"chunk_id": 1, "source": "resume.pdf", "text": "Aug 2025 – Present"},
        {"chunk_id": 2, "source": "resume.pdf", "text": "Jan 2022 – Feb 2024"},
        {"chunk_id": 3, "source": "other.pdf", "text": "unrelated"},
    ]
    kept = [{"chunk_id": 0, "source": "resume.pdf", "text": "summary", "rerank_score": 1.0}]
    expanded = expand_adjacent_chunks(kept, metadata, window=2)
    assert [c["chunk_id"] for c in expanded] == [0, 1, 2]
    assert all(c["source"] == "resume.pdf" for c in expanded)
