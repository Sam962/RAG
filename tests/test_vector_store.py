import pickle
from unittest.mock import MagicMock, patch

import faiss
import numpy as np
import pytest

from src.vector_store import FiassVectorStore


@pytest.fixture
def store(tmp_path):
    with patch("src.vector_store.SentenceTransformer", return_value=MagicMock()):
        vs = FiassVectorStore(persist_dir=str(tmp_path))
    return vs


def test_defaults_are_minilm_safe_and_use_faiss_store():
    import inspect

    params = inspect.signature(FiassVectorStore.__init__).parameters
    assert params["persist_dir"].default == "faiss_store"
    assert params["chunk_size"].default == 500
    assert params["chunk_overlap"].default == 50


def test_search_skips_faiss_padding_when_top_k_exceeds_index(store):
    vectors = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype="float32",
    )
    store.add_embeddings(
        vectors,
        [{"text": "first", "chunk_id": 0}, {"text": "second", "chunk_id": 1}],
    )

    query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32")
    faiss.normalize_L2(query)
    results = store.search(query, top_k=5)

    assert len(results) == 2
    assert all(r["index"] >= 0 for r in results)
    assert [r["index"] for r in results] == [0, 1]
    assert results[0]["metadata"]["text"] == "first"


def test_search_empty_index_returns_empty_list(store):
    query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32")
    assert store.search(query, top_k=5) == []


def test_build_from_documents_rejects_empty_list(store):
    with pytest.raises(ValueError, match="No documents to index"):
        store.build_from_documents([])


def test_add_embeddings_rejects_empty_matrix(store):
    with pytest.raises(ValueError, match="empty embedding matrix"):
        store.add_embeddings(np.zeros((0, 4), dtype="float32"), [])


def test_load_missing_files_raises(store):
    with pytest.raises(FileNotFoundError, match="FAISS store not found"):
        store.load()


def test_load_accepts_matching_ip_index(store):
    vectors = np.eye(2, 4, dtype="float32")
    store.add_embeddings(vectors, [{"text": "a"}, {"text": "b"}])
    store.save()
    store.model.get_sentence_embedding_dimension.return_value = 4
    store.load()
    assert store.index.ntotal == 2
    assert len(store.metadata) == 2


def test_load_rejects_index_metadata_count_mismatch(store, tmp_path):
    vectors = np.eye(2, 4, dtype="float32")
    store.add_embeddings(vectors, [{"text": "a"}, {"text": "b"}])
    store.save()
    with open(tmp_path / "metadata.pk1", "wb") as f:
        pickle.dump([{"text": "a"}], f)
    with pytest.raises(ValueError, match="Index/metadata mismatch"):
        store.load()


def test_load_rejects_l2_index(store, tmp_path):
    dim = 4
    index = faiss.IndexFlatL2(dim)
    index.add(np.eye(2, dim, dtype="float32"))
    faiss.write_index(index, str(tmp_path / "faiss.index"))
    with open(tmp_path / "metadata.pk1", "wb") as f:
        pickle.dump([{"text": "a"}, {"text": "b"}], f)
    store.model.get_sentence_embedding_dimension.return_value = dim
    with pytest.raises(ValueError, match="inner-product"):
        store.load()


def test_load_rejects_dimension_mismatch(store):
    vectors = np.eye(2, 4, dtype="float32")
    store.add_embeddings(vectors, [{"text": "a"}, {"text": "b"}])
    store.save()
    store.model.get_sentence_embedding_dimension.return_value = 384
    with pytest.raises(ValueError, match="dimension mismatch"):
        store.load()
