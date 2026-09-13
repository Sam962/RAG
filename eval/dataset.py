"""Load the in-repo eval set (own docs only — not Open RAG Bench)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"
ALLOWED_KINDS = {"fact", "followup", "unknown"}


def load_examples(path: Path | None = None) -> list[dict[str, Any]]:
    data_path = path or DATASET_PATH
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Eval dataset must be a non-empty list: {data_path}")
    examples = [validate_example(item, i) for i, item in enumerate(raw)]
    n = len(examples)
    if n < 10 or n > 20:
        raise ValueError(f"First eval track must have 10–20 examples, found {n}.")
    return examples


def validate_example(item: Any, index: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"Example {index} must be an object.")
    example_id = str(item.get("id") or "").strip()
    kind = str(item.get("kind") or "").strip()
    question = str(item.get("question") or "").strip()
    ground_truth = str(item.get("ground_truth") or "").strip()
    history = item.get("chat_history") or []
    if not example_id or not question or not ground_truth:
        raise ValueError(f"Example {index} needs id, question, and ground_truth.")
    if kind not in ALLOWED_KINDS:
        raise ValueError(f"Example {example_id} has unknown kind {kind!r}.")
    if not isinstance(history, list):
        raise ValueError(f"Example {example_id} chat_history must be a list.")
    if kind == "followup" and not history:
        raise ValueError(f"Follow-up {example_id} must include chat_history.")
    turns = []
    for turn in history:
        role = str(turn.get("role") or "")
        content = str(turn.get("content") or "")
        if role not in ("user", "assistant") or not content.strip():
            raise ValueError(f"Example {example_id} has an invalid history turn.")
        turns.append({"role": role, "content": content})
    return {
        "id": example_id,
        "kind": kind,
        "question": question,
        "ground_truth": ground_truth,
        "chat_history": turns,
    }


def contexts_from_result(result: dict[str, Any]) -> list[str]:
    documents = result.get("documents") or []
    texts = [str(doc.get("text") or "").strip() for doc in documents if doc.get("text")]
    if texts:
        return texts
    return ["No documents retrieved."]
