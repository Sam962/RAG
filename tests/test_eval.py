import json
from pathlib import Path
from unittest.mock import MagicMock

from eval.dataset import DATASET_PATH, contexts_from_result, load_examples, validate_example
from eval.run_eval import run_graph_examples, summarize, token_recall
from src.search import _NO_CONTEXT_ANSWER


def test_dataset_is_own_docs_sized():
    examples = load_examples()
    assert 10 <= len(examples) <= 20
    kinds = {item["kind"] for item in examples}
    assert kinds == {"fact", "followup", "unknown"}
    assert any(item["kind"] == "followup" and item["chat_history"] for item in examples)
    assert any(item["kind"] == "unknown" for item in examples)
    assert DATASET_PATH.exists()


def test_followup_requires_history():
    try:
        validate_example(
            {
                "id": "bad",
                "kind": "followup",
                "question": "how many years?",
                "ground_truth": "4",
                "chat_history": [],
            },
            0,
        )
    except ValueError as exc:
        assert "chat_history" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_contexts_from_empty_result():
    assert contexts_from_result({}) == ["No documents retrieved."]
    assert contexts_from_result({"documents": [{"text": "Beam search"}]}) == ["Beam search"]


def test_run_graph_examples_passes_history():
    graph = MagicMock()
    graph.invoke.return_value = {
        "answer": "1991",
        "citations": [{"source": "python_intro.txt"}],
        "documents": [{"text": "first released in 1991"}],
    }
    examples = [
        {
            "id": "fu-python-year",
            "kind": "followup",
            "question": "when was it first released?",
            "ground_truth": "1991",
            "chat_history": [
                {"role": "user", "content": "Who created Python?"},
                {"role": "assistant", "content": "Guido van Rossum created Python."},
            ],
        }
    ]
    rows = run_graph_examples(examples, graph=graph)
    payload = graph.invoke.call_args.args[0]
    assert payload["question"] == "when was it first released?"
    assert payload["chat_history"][0]["content"] == "Who created Python?"
    assert rows[0]["citation_count"] == 1
    assert rows[0]["retrieved_contexts"] == ["first released in 1991"]


def test_summarize_unknown_abstain():
    rows = [
        {
            "kind": "unknown",
            "citation_count": 0,
            "unknown_ok": True,
            "token_recall": 0.2,
        },
        {
            "kind": "fact",
            "citation_count": 2,
            "unknown_ok": None,
            "token_recall": 0.8,
        },
    ]
    summary = summarize(rows, {"faithfulness": 0.8})
    assert summary["n"] == 2
    assert summary["unknown_abstain_rate"] == 1.0
    assert summary["mean_token_recall"] == 0.8
    assert summary["ragas"]["faithfulness"] == 0.8
    assert _NO_CONTEXT_ANSWER.startswith("I don't know")


def test_token_recall_counts_gold_overlap():
    assert token_recall("Guido van Rossum.", "Guido van Rossum created Python.") > 0.4
    assert token_recall("I don't know.", "Python was first released in 1991.") == 0.0


def test_dataset_json_is_valid_list():
    raw = json.loads(Path(DATASET_PATH).read_text(encoding="utf-8"))
    assert isinstance(raw, list)
    ids = [item["id"] for item in raw]
    assert len(ids) == len(set(ids))
