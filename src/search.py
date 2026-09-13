import os
import re
import time
from typing import Any, Dict, List

from langsmith import traceable

from src.config import configure_langsmith, settings
from src.llm import build_chat_model
from src.retriever import HybridRetriever
from src.vector_store import FiassVectorStore

configure_langsmith()

GROUNDING_SYSTEM = (
    "You are a retrieval-augmented assistant. Answer the question using ONLY the "
    "provided context. If the context does not contain the answer, reply exactly: "
    "I don't know. Do not use outside knowledge. Do not invent facts. "
    "You may add or compare dates and numbers that appear in the context."
)

_NO_CONTEXT_ANSWER = "I don't know. No relevant documents were found."
_RETRY_IN = re.compile(r"try again in ([\d.]+)\s*(ms|s)", re.IGNORECASE)


def retry_after_seconds(exc: BaseException) -> float:
    match = _RETRY_IN.search(str(exc))
    if not match:
        return 1.0
    value = float(match.group(1))
    return value / 1000.0 if match.group(2).lower() == "ms" else value


def is_rate_limit_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name == "RateLimitError" or "RateLimit" in name:
        return True
    text = str(exc).lower()
    return "rate_limit" in text or "429" in text and "rate" in text


def invoke_llm(llm: Any, messages: Any, attempts: int = 6) -> Any:
    """Call the chat model. Rate limits retry here only if fallback did not catch them."""
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            if not is_rate_limit_error(exc):
                raise
            last = exc
            wait = min(30.0, max(retry_after_seconds(exc) + 0.25, 0.5) * (1.4**attempt))
            print(f"[WARN] Groq rate limit; sleeping {wait:.1f}s (attempt {attempt + 1}/{attempts})")
            time.sleep(wait)
    assert last is not None
    raise last


def build_qa_messages(question: str, chunks: List[Dict[str, Any]]) -> List[tuple[str, str]]:
    parts: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page")
        loc = f"{source}" if page is None else f"{source}, page {page}"
        parts.append(f"[{i}] ({loc})\n{chunk.get('text', '')}")
    human = f"Question:\n{question}\n\nContext:\n" + "\n\n".join(parts)
    return [("system", GROUNDING_SYSTEM), ("human", human)]


def citations_from_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    citations = []
    for chunk in chunks:
        citations.append(
            {
                "source": chunk.get("source", "unknown"),
                "page": chunk.get("page"),
                "chunk_id": chunk.get("chunk_id"),
                "rerank_score": chunk.get("rerank_score"),
            }
        )
    return citations


class RAGSearch:
    def __init__(self, persist_dir: str | None = None, embedding_model: str | None = None):
        persist_dir = persist_dir or settings.persist_dir
        embedding_model = embedding_model or settings.embedding_model
        self.vectorstore = FiassVectorStore(
            persist_dir,
            embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pk1")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_docs

            docs = load_all_docs("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        self.retriever = HybridRetriever(self.vectorstore)
        self.llm = build_chat_model()

    @traceable(run_type="chain", name="ask")
    def ask(self, query: str, top_k: int | None = None) -> Dict[str, Any]:
        results = self.retriever.retrieve(query, k=top_k or settings.top_k)
        if not results:
            return {
                "answer": _NO_CONTEXT_ANSWER,
                "citations": [],
                "used_context": False,
            }
        messages = build_qa_messages(query, results)
        response = invoke_llm(self.llm, messages)
        return {
            "answer": response.content,
            "citations": citations_from_chunks(results),
            "used_context": True,
        }

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        """Backward-compatible wrapper. Prefer ask()."""
        return self.ask(query, top_k=top_k)["answer"]
