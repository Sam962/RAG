from pathlib import Path

from eval.open_ragbench import (
    documents_from_paper,
    examples_from_official,
    gold_doc_ids,
    hard_negative_ids,
    index_covers,
    is_text_extractive,
    retrieval_hits,
    select_query_ids,
    write_manifest,
)
from eval.run_eval import summarize
from src.graph import load_rag_components, reset_rag_components
from src import graph as graph_mod


def test_filters_text_extractive_only():
    assert is_text_extractive({"type": "extractive", "source": "text"})
    assert not is_text_extractive({"type": "abstractive", "source": "text"})
    assert not is_text_extractive({"type": "extractive", "source": "text-image"})


def test_select_query_ids_is_stable_and_limited():
    queries = {
        "b": {"type": "extractive", "source": "text"},
        "a": {"type": "extractive", "source": "text"},
        "c": {"type": "abstractive", "source": "text"},
        "d": {"type": "extractive", "source": "text-table"},
    }
    qrels = {"a": {"doc_id": "p1", "section_id": 1}, "b": {"doc_id": "p2", "section_id": 0}}
    answers = {"a": "A", "b": "B"}
    assert select_query_ids(queries, qrels, answers, limit=1) == ["a"]
    assert select_query_ids(queries, qrels, answers, limit=10) == ["a", "b"]


def test_examples_and_gold_docs_from_official():
    maps = {
        "queries": {"q1": {"query": "What is LoRA?", "type": "extractive", "source": "text"}},
        "qrels": {"q1": {"doc_id": "2401.00001v1", "section_id": 2}},
        "answers": {"q1": "Low-Rank Adaptation."},
    }
    examples = examples_from_official(maps, ["q1"])
    assert examples[0]["question"] == "What is LoRA?"
    assert examples[0]["qrel"]["doc_id"] == "2401.00001v1"
    assert gold_doc_ids(examples) == ["2401.00001v1"]


def test_hard_negative_ids_are_never_in_qrels():
    pdf_urls = {"gold-1": "http://x", "hard-a": "http://y", "hard-b": "http://z"}
    qrels = {"q1": {"doc_id": "gold-1", "section_id": 0}}
    assert hard_negative_ids(pdf_urls, qrels) == ["hard-a", "hard-b"]


def test_index_covers_requires_manifest_ids(tmp_path):
    required = {"gold-1", "hard-a"}
    assert index_covers(tmp_path, required) is False
    (tmp_path / "faiss.index").write_bytes(b"x")
    (tmp_path / "metadata.pk1").write_bytes(b"x")
    assert index_covers(tmp_path, required) is False
    write_manifest(tmp_path, {"indexed_ids": ["gold-1"]})
    assert index_covers(tmp_path, required) is False
    write_manifest(tmp_path, {"indexed_ids": ["gold-1", "hard-a", "hard-b"]})
    assert index_covers(tmp_path, required) is True


def test_documents_from_paper_keep_qrel_ids():
    paper = {
        "id": "2401.00001v1",
        "sections": [
            {"section_id": 0, "text": ""},
            {"section_id": 2, "text": "LoRA adds low-rank adapters."},
        ],
    }
    docs = documents_from_paper(paper)
    assert len(docs) == 1
    assert docs[0].metadata["doc_id"] == "2401.00001v1"
    assert docs[0].metadata["section_id"] == 2


def test_retrieval_hits_doc_and_section():
    qrel = {"doc_id": "paper-a", "section_id": 3}
    miss = retrieval_hits([{"doc_id": "paper-b", "section_id": 3}], qrel)
    assert miss == {"doc_hit": False, "section_hit": False}
    doc_only = retrieval_hits([{"doc_id": "paper-a", "section_id": 1}], qrel)
    assert doc_only == {"doc_hit": True, "section_hit": False}
    both = retrieval_hits(
        [{"source": "eval/cache/paper-a.json", "page": "3"}],
        qrel,
    )
    assert both == {"doc_hit": True, "section_hit": True}


def test_summarize_includes_qrel_recall():
    rows = [
        {"kind": "fact", "citation_count": 1, "token_recall": 1.0, "doc_hit": True, "section_hit": False},
        {"kind": "fact", "citation_count": 1, "token_recall": 1.0, "doc_hit": True, "section_hit": True},
    ]
    summary = summarize(rows, None)
    assert summary["doc_recall"] == 1.0
    assert summary["section_recall"] == 0.5


def test_missing_eval_store_does_not_read_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(graph_mod, "_active_persist_dir", str(tmp_path / "orb-store"))
    reset_rag_components()

    def fail_load(_dir: str):
        raise AssertionError("must not fall back to data/")

    monkeypatch.setattr("src.data_loader.load_all_docs", fail_load)
    try:
        from unittest.mock import MagicMock, patch

        with patch("src.vector_store.SentenceTransformer", return_value=MagicMock()):
            load_rag_components()
    except FileNotFoundError as exc:
        assert "eval index" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")
    reset_rag_components()
    monkeypatch.setattr(graph_mod, "_active_persist_dir", None)


def test_cache_paths_are_not_data():
    from eval.open_ragbench import CACHE_DIR, STORE_DIR

    root = Path(__file__).resolve().parents[1]
    assert STORE_DIR.resolve() == (root / "eval" / ".cache" / "open_ragbench" / "store").resolve()
    assert (root / "data") not in STORE_DIR.resolve().parents
