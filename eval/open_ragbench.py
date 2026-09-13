"""Second eval track: Vectara Open RAG Bench (CC-BY-NC-4.0).

Does not touch data/. Indexes gold paper *sections* for a text-only extractive
slice into eval/.cache/open_ragbench/store/. Retrieval is scored with official
qrels (doc_id + section_id).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.config import settings
from src.graph import reset_rag_components
from src.vector_store import FiassVectorStore

HF_REPO = "vectara/open_ragbench"
HF_PREFIX = "pdf/arxiv"
QUERY_TYPE = "extractive"
QUERY_SOURCE = "text"
DEFAULT_LIMIT = 30

CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "open_ragbench"
STORE_DIR = CACHE_DIR / "store"
MANIFEST_NAME = "manifest.json"


def _official_dir(cache_dir: Path) -> Path:
    return cache_dir / HF_PREFIX


def download_official_file(filename: str, cache_dir: Path = CACHE_DIR) -> Path:
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        filename=f"{HF_PREFIX}/{filename}",
        local_dir=str(cache_dir),
    )
    return Path(path)


def load_official_maps(cache_dir: Path = CACHE_DIR) -> dict[str, dict[str, Any]]:
    queries = json.loads(download_official_file("queries.json", cache_dir).read_text())
    qrels = json.loads(download_official_file("qrels.json", cache_dir).read_text())
    answers = json.loads(download_official_file("answers.json", cache_dir).read_text())
    pdf_urls = json.loads(download_official_file("pdf_urls.json", cache_dir).read_text())
    return {
        "queries": queries,
        "qrels": qrels,
        "answers": answers,
        "pdf_urls": pdf_urls,
    }


def qrel_doc_ids(qrels: dict[str, Any]) -> set[str]:
    return {str(item["doc_id"]) for item in qrels.values() if item.get("doc_id")}


def hard_negative_ids(pdf_urls: dict[str, Any], qrels: dict[str, Any]) -> list[str]:
    """Papers in the official dump that are never a gold doc for any query."""
    positives = qrel_doc_ids(qrels)
    return sorted(str(doc_id) for doc_id in pdf_urls if str(doc_id) not in positives)


def is_text_extractive(query: dict[str, Any]) -> bool:
    return query.get("type") == QUERY_TYPE and query.get("source") == QUERY_SOURCE


def select_query_ids(
    queries: dict[str, Any],
    qrels: dict[str, Any],
    answers: dict[str, Any],
    limit: int = DEFAULT_LIMIT,
) -> list[str]:
    ids = sorted(
        qid
        for qid, item in queries.items()
        if is_text_extractive(item) and qid in qrels and qid in answers
    )
    if limit and limit > 0:
        return ids[:limit]
    return ids


def examples_from_official(
    maps: dict[str, dict[str, Any]],
    query_ids: list[str],
) -> list[dict[str, Any]]:
    queries, qrels, answers = maps["queries"], maps["qrels"], maps["answers"]
    examples: list[dict[str, Any]] = []
    for qid in query_ids:
        qrel = qrels[qid]
        examples.append(
            {
                "id": qid,
                "kind": "fact",
                "question": queries[qid]["query"],
                "ground_truth": str(answers[qid]),
                "chat_history": [],
                "qrel": {
                    "doc_id": qrel["doc_id"],
                    "section_id": qrel["section_id"],
                },
            }
        )
    return examples


def gold_doc_ids(examples: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for item in examples:
        doc_id = str(item["qrel"]["doc_id"])
        if doc_id not in seen:
            seen.append(doc_id)
    return seen


def download_corpus_paper(doc_id: str, cache_dir: Path = CACHE_DIR) -> dict[str, Any]:
    path = download_official_file(f"corpus/{doc_id}.json", cache_dir)
    return json.loads(path.read_text(encoding="utf-8"))


def documents_from_paper(paper: dict[str, Any]) -> list[Document]:
    doc_id = str(paper.get("id") or "")
    docs: list[Document] = []
    for section in paper.get("sections") or []:
        text = str(section.get("text") or "").strip()
        if not text or not doc_id:
            continue
        section_id = section.get("section_id")
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": doc_id,
                    "doc_id": doc_id,
                    "section_id": section_id,
                    "page": section_id,
                },
            )
        )
    return docs


def documents_for_doc_ids(
    doc_ids: list[str],
    cache_dir: Path = CACHE_DIR,
) -> list[Document]:
    docs: list[Document] = []
    for doc_id in doc_ids:
        try:
            paper_docs = documents_from_paper(download_corpus_paper(doc_id, cache_dir))
        except Exception as exc:
            print(f"[WARN] Skip corpus {doc_id}: {exc}")
            continue
        docs.extend(paper_docs)
    if not docs:
        raise ValueError("No gold sections to index for the Open RAG Bench slice.")
    return docs


def documents_for_examples(
    examples: list[dict[str, Any]],
    cache_dir: Path = CACHE_DIR,
    extra_doc_ids: list[str] | None = None,
) -> list[Document]:
    seen: list[str] = []
    for doc_id in gold_doc_ids(examples) + list(extra_doc_ids or []):
        if doc_id not in seen:
            seen.append(doc_id)
    return documents_for_doc_ids(seen, cache_dir)


def index_ready(persist_dir: Path = STORE_DIR) -> bool:
    return (persist_dir / "faiss.index").exists() and (persist_dir / "metadata.pk1").exists()


def read_manifest(persist_dir: Path = STORE_DIR) -> dict[str, Any] | None:
    path = persist_dir / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def write_manifest(persist_dir: Path, payload: dict[str, Any]) -> None:
    persist_dir.mkdir(parents=True, exist_ok=True)
    (persist_dir / MANIFEST_NAME).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def index_covers(persist_dir: Path, required_ids: set[str]) -> bool:
    if not index_ready(persist_dir) or not required_ids:
        return False
    manifest = read_manifest(persist_dir)
    indexed = {str(item) for item in (manifest or {}).get("indexed_ids") or []}
    return required_ids <= indexed


def build_index(documents: list[Document], persist_dir: Path = STORE_DIR) -> dict[str, Any]:
    persist_dir.mkdir(parents=True, exist_ok=True)
    store = FiassVectorStore(
        str(persist_dir),
        settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    store.build_from_documents(documents)
    reset_rag_components()
    indexed_ids = sorted(
        {str(doc.metadata.get("doc_id")) for doc in documents if doc.metadata.get("doc_id")}
    )
    payload = {
        "documents": len(documents),
        "chunks": len(store.metadata),
        "persist_dir": str(persist_dir),
        "indexed_ids": indexed_ids,
        "n_papers": len(indexed_ids),
    }
    write_manifest(persist_dir, payload)
    return payload


def _same_id(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    if left == right:
        return True
    try:
        return int(left) == int(right)
    except (TypeError, ValueError):
        return str(left) == str(right)


def chunk_doc_id(chunk: dict[str, Any]) -> str | None:
    if chunk.get("doc_id"):
        return str(chunk["doc_id"])
    source = chunk.get("source")
    if source is None:
        return None
    return Path(str(source)).stem


def chunk_section_id(chunk: dict[str, Any]) -> Any:
    if chunk.get("section_id") is not None:
        return chunk["section_id"]
    return chunk.get("page")


def retrieval_hits(
    documents: list[dict[str, Any]],
    qrel: dict[str, Any],
) -> dict[str, bool]:
    gold_doc = qrel["doc_id"]
    gold_section = qrel["section_id"]
    doc_hit = False
    section_hit = False
    for chunk in documents:
        if not _same_id(chunk_doc_id(chunk), gold_doc):
            continue
        doc_hit = True
        if _same_id(chunk_section_id(chunk), gold_section):
            section_hit = True
            break
    return {"doc_hit": doc_hit, "section_hit": section_hit}


def prepare_track(
    limit: int = DEFAULT_LIMIT,
    cache_dir: Path = CACHE_DIR,
    persist_dir: Path = STORE_DIR,
    rebuild: bool = False,
    include_hard_negatives: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    maps = load_official_maps(cache_dir)
    query_ids = select_query_ids(maps["queries"], maps["qrels"], maps["answers"], limit)
    examples = examples_from_official(maps, query_ids)
    gold = gold_doc_ids(examples)
    negatives = (
        hard_negative_ids(maps.get("pdf_urls") or {}, maps["qrels"])
        if include_hard_negatives
        else []
    )
    required = set(gold) | set(negatives)
    info: dict[str, Any] = {
        "n_queries": len(examples),
        "gold_docs": gold,
        "hard_negatives": negatives,
        "n_hard_negatives": len(negatives),
        "slice": f"{QUERY_TYPE}+{QUERY_SOURCE}",
        "license": "CC-BY-NC-4.0",
        "persist_dir": str(persist_dir),
    }
    if rebuild or not index_covers(persist_dir, required):
        reason = "rebuild" if rebuild else "store missing gold or hard-negative papers"
        print(f"[INFO] Building Open RAG Bench index ({reason})…")
        docs = documents_for_examples(examples, cache_dir, extra_doc_ids=negatives)
        info["index"] = build_index(docs, persist_dir)
        info["index"]["gold_docs"] = len(gold)
        info["index"]["hard_negatives"] = len(negatives)
    else:
        info["index"] = {"reused": True, "persist_dir": str(persist_dir)}
    return examples, info
