"""Run the in-repo eval set through the graph and score with ragas."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.dataset import contexts_from_result, load_examples
from eval.ragas_compat import patch_langchain_vertexai
from src.config import configure_langsmith, settings
from src.llm import build_chat_model, llm_provider
from src.graph import build_graph
from src.search import _NO_CONTEXT_ANSWER

LANGSMITH_DATASET = "rag-pro-own-docs"
METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def run_graph_examples(
    examples: list[dict[str, Any]],
    graph=None,
    sleep_s: float = 0.0,
) -> list[dict[str, Any]]:
    app = graph or build_graph()
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        if sleep_s > 0 and index:
            time.sleep(sleep_s)
        result = app.invoke(
            {
                "question": example["question"],
                "chat_history": list(example.get("chat_history") or []),
            },
            config={"configurable": {"thread_id": f"eval-{example['id']}-{uuid4().hex[:8]}"}},
        )
        answer = result.get("answer") or ""
        contexts = contexts_from_result(result)
        unknown_ok = None
        if example["kind"] == "unknown":
            unknown_ok = _NO_CONTEXT_ANSWER in answer or answer.strip().startswith(
                "I don't know"
            )
        rows.append(
            {
                "id": example["id"],
                "kind": example["kind"],
                "question": example["question"],
                "ground_truth": example["ground_truth"],
                "answer": answer,
                "retrieved_contexts": contexts,
                "citation_count": len(result.get("citations") or []),
                "unknown_ok": unknown_ok,
                "token_recall": token_recall(answer, example["ground_truth"]),
            }
        )
    return rows


def token_recall(answer: str, ground_truth: str) -> float:
    """Share of ground-truth tokens that appear in the answer (0–1)."""
    truth = set(re.findall(r"[a-z0-9]+", ground_truth.lower()))
    if not truth:
        return 0.0
    got = set(re.findall(r"[a-z0-9]+", answer.lower()))
    return len(truth & got) / len(truth)


def _finite_mean(values: list[Any]) -> float | None:
    nums: list[float] = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isnan(number):
            continue
        nums.append(number)
    if not nums:
        return None
    return sum(nums) / len(nums)


def score_with_ragas(rows: list[dict[str, Any]]) -> dict[str, float]:
    patch_langchain_vertexai()
    from langchain_core.embeddings import Embeddings
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )
    from ragas.run_config import RunConfig
    from sentence_transformers import SentenceTransformer

    class MiniLMEmbeddings(Embeddings):
        def __init__(self, model_name: str):
            self._model = SentenceTransformer(model_name)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._model.encode(list(texts), convert_to_numpy=True).tolist()

        def embed_query(self, text: str) -> list[float]:
            return self._model.encode([text], convert_to_numpy=True)[0].tolist()

    samples = [
        SingleTurnSample(
            user_input=row["question"],
            response=row["answer"],
            retrieved_contexts=row["retrieved_contexts"],
            reference=row["ground_truth"],
        )
        for row in rows
        if row["kind"] != "unknown"
    ]
    if not samples:
        return {}
    judge = build_chat_model()
    wrapped_llm = LangchainLLMWrapper(judge)
    wrapped_emb = LangchainEmbeddingsWrapper(MiniLMEmbeddings(settings.embedding_model))
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[
            Faithfulness(),
            AnswerRelevancy(strictness=1),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=wrapped_llm,
        embeddings=wrapped_emb,
        experiment_name=LANGSMITH_DATASET,
        raise_exceptions=False,
        batch_size=1,
        run_config=RunConfig(timeout=180, max_workers=1, max_retries=3),
    )
    means: dict[str, float] = {}
    for name in METRIC_NAMES:
        mean = _finite_mean(list(result._scores_dict.get(name, [])))
        if mean is not None:
            means[name] = mean
    return means


def summarize(rows: list[dict[str, Any]], metrics: dict[str, float] | None) -> dict[str, Any]:
    unknown = [row for row in rows if row["kind"] == "unknown"]
    unknown_ok = [row for row in unknown if row.get("unknown_ok")]
    grounded = [row for row in rows if row["kind"] != "unknown"]
    return {
        "n": len(rows),
        "by_kind": {
            kind: sum(1 for row in rows if row["kind"] == kind)
            for kind in ("fact", "followup", "unknown")
        },
        "unknown_abstain_rate": (len(unknown_ok) / len(unknown)) if unknown else None,
        "mean_citations": sum(row["citation_count"] for row in rows) / len(rows),
        "mean_token_recall": _finite_mean([row.get("token_recall") for row in grounded]),
        "ragas": metrics or {},
        "note": (
            "Ragas scores fact/follow-up only. Unknowns are judged by "
            "unknown_abstain_rate. token_recall is lexical overlap with the gold answer."
        ),
    }


def sync_langsmith_dataset(examples: list[dict[str, Any]]) -> str | None:
    if not settings.langsmith_api_key or not settings.langsmith_tracing:
        return None
    configure_langsmith()
    from langsmith import Client

    client = Client()
    if not client.has_dataset(dataset_name=LANGSMITH_DATASET):
        client.create_dataset(
            dataset_name=LANGSMITH_DATASET,
            description="In-repo Q/A from data/ (not Open RAG Bench).",
        )
    existing = {
        example.inputs.get("id")
        for example in client.list_examples(dataset_name=LANGSMITH_DATASET)
    }
    to_add = [item for item in examples if item["id"] not in existing]
    if to_add:
        client.create_examples(
            dataset_name=LANGSMITH_DATASET,
            examples=[
                {
                    "inputs": {"question": item["question"], "id": item["id"]},
                    "outputs": {"ground_truth": item["ground_truth"]},
                    "metadata": {"kind": item["kind"]},
                }
                for item in to_add
            ],
        )
    return LANGSMITH_DATASET


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG Pro on the in-repo Q/A set.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to dataset JSON (default: eval/dataset.json).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Score only the first N examples.")
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Run the graph only; skip ragas LLM judges.",
    )
    parser.add_argument(
        "--no-langsmith",
        action="store_true",
        help="Do not upsert the LangSmith dataset.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "eval" / "last_results.json",
        help="Write the score report JSON here.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=None,
        help="Seconds between examples. Default 0 for Ollama, 8 for Groq.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    examples = load_examples(args.dataset)
    if args.limit and args.limit > 0:
        examples = examples[: args.limit]

    configure_langsmith()
    if not args.no_langsmith:
        try:
            dataset_name = sync_langsmith_dataset(examples)
            if dataset_name:
                print(f"[INFO] LangSmith dataset: {dataset_name}")
        except Exception as exc:
            print(f"[WARN] LangSmith dataset sync skipped: {exc}")

    if args.sleep is None:
        args.sleep = 0.0 if llm_provider() == "ollama" else 8.0
    print(f"[INFO] Running graph on {len(examples)} examples (sleep {args.sleep}s)…")
    rows = run_graph_examples(examples, sleep_s=args.sleep)
    metrics = None if args.no_score else score_with_ragas(rows)
    report = {
        "summary": summarize(rows, metrics),
        "examples": [
            {
                "id": row["id"],
                "kind": row["kind"],
                "question": row["question"],
                "answer": row["answer"],
                "citation_count": row["citation_count"],
                "token_recall": row.get("token_recall"),
                "unknown_ok": row["unknown_ok"],
            }
            for row in rows
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"[INFO] Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
