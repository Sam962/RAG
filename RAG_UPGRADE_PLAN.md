# RAG Upgrade Plan (LangGraph + LangSmith)

A self-build roadmap to turn the current linear RAG script into a robust, observable,
graph-orchestrated RAG service. Follow the steps top-to-bottom; each step is
independent enough to build and test on its own.

---

## 0. Where you are today

Current flow is linear and lives in a few files:

- `src/data_loader.py` — loads PDF / TXT / CSV into LangChain documents.
- `src/embedding.py` — chunks docs + embeds with `all-MiniLM-L6-v2`.
- `src/vector_store.py` — custom FAISS `IndexFlatL2` store + pickle metadata.
- `src/search.py` — `RAGSearch`: retrieve top-k → single "summarize" prompt (Groq).
- `app.py` — hardcoded query, prints a summary.

Main weaknesses to fix:
- No orchestration (can't branch, retry, or add steps cleanly).
- No tracing/observability, no evaluation.
- Retrieval is single-vector top-k only (no hybrid search, no reranking, no query rewrite).
- L2 distance with un-normalized vectors (should be cosine).
- "Summarize" prompt instead of a grounded QA prompt with citations.
- No conversation memory; CLI-only with a hardcoded query.

---

## Target architecture

```mermaid
flowchart LR
    client[FastAPI / Streamlit] --> graph[LangGraph app]
    subgraph graph [LangGraph pipeline]
        rewrite[rewrite_query] --> retrieve[hybrid_retrieve]
        retrieve --> rerank[rerank]
        rerank --> grade[grade_relevance]
        grade -->|relevant| generate[generate_answer]
        grade -->|weak| fallback[fallback_or_clarify]
        generate --> cite[attach_citations]
    end
    graph --> ls[(LangSmith traces)]
```

---

## 1. Config + dependencies

**Goal:** one place for settings, and all new libs installed.

- Create `src/config.py` using `pydantic-settings`. Centralize:
  - model names (embedding, reranker, Groq LLM), `top_k`, chunk size/overlap.
  - `GROQ_API_KEY`.
  - LangSmith: `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`.
- Add dependencies (`pyproject.toml` + `requirements.txt`):
  - `langgraph`, `langsmith`, `rank-bm25`, `fastapi`, `uvicorn[standard]`,
    `streamlit`, `pydantic-settings`, `ragas` (eval), `pytest`.
  - Reranking reuses your existing `sentence-transformers` (via `CrossEncoder`).
- Update `.env.example` with the LangSmith keys.

**Done when:** `from src.config import settings` works and prints your values.

---

## 2. LangSmith observability

**Goal:** see every run as a trace, no heavy code changes.

- Set env vars (works automatically for any LangChain/LangGraph runnable):
  ```
  LANGSMITH_TRACING=true
  LANGSMITH_API_KEY=ls_...
  LANGSMITH_PROJECT=rag-pro
  ```
- For your custom (non-LangChain) functions — embedding, FAISS search, rerank —
  add the `@traceable` decorator from `langsmith` so they appear as spans.

**Done when:** running a query shows a trace tree in the LangSmith UI.

---

## 3. Retrieval quality upgrades

**Goal:** better candidates before the LLM ever sees them.

- In `src/vector_store.py`, switch FAISS to cosine:
  - L2-normalize embeddings and use `IndexFlatIP` instead of `IndexFlatL2`.
  - Store richer metadata (source path, page, chunk_id), not just `{"text": ...}`.
  - Note: this requires rebuilding `faiss_store/` once.
- Create `src/retriever.py` with a `HybridRetriever`:
  - Combine dense FAISS results + `rank-bm25` keyword results via
    Reciprocal Rank Fusion (RRF).
- Add cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to reorder
  fused candidates and keep the final top-k.

**Done when:** retriever returns reranked chunks with source metadata attached.

---

## 4. LangGraph orchestration

**Goal:** replace the linear `search_and_summarize` with a graph you can extend.

- Create `src/graph.py`. Define a `TypedDict` state:
  `question`, `chat_history`, `rewritten_query`, `documents`, `answer`, `citations`.
- Nodes:
  - `rewrite_query` — Groq LLM + chat history → standalone, retrieval-friendly query.
  - `hybrid_retrieve` — call `HybridRetriever`.
  - `rerank` — cross-encoder rerank.
  - `grade_relevance` — score check; **conditional edge** to `generate` or `fallback`.
  - `generate_answer` — grounded QA prompt (answer only from context, emit inline
    source markers).
  - `attach_citations` — map the answer back to source metadata.
- Compile with a `MemorySaver` checkpointer keyed by `thread_id` → conversation memory.

**Done when:** `graph.invoke({"question": ...}, config={"thread_id": ...})` returns
an answer with citations, and follow-up questions use history.

---

## 5. Robustness

**Goal:** fewer crashes, fewer hallucinations.

- Wrap LLM calls with retry/backoff + a timeout.
- Replace the bare `"No relevant documents found"` with a graceful fallback message.
- Strict grounded prompt (replaces the "summarize" prompt) to reduce hallucination.
- Return a structured result: `{answer, citations, used_context}`.
- In the load/build path: skip unreadable files, log counts, only fail loudly on
  total failure.

**Done when:** a question with no relevant docs returns a clean fallback, not an error.

---

## 6. Interfaces

**Goal:** talk to the RAG over HTTP and in a chat UI.

- `api/main.py` (FastAPI):
  - `POST /chat` — body `{question, thread_id}` → invoke the graph → return
    `{answer, citations}`.
  - `POST /ingest` — (re)build the index.
  - `GET /health`.
  - Run with `uvicorn api.main:app --reload`.
- `ui/streamlit_app.py`:
  - Chat UI that calls the FastAPI `/chat` endpoint.
  - Render answers + expandable source citations; persist `thread_id` per session.

**Done when:** you can chat in the browser and expand sources for each answer.

---

## 7. Evaluation harness

**Goal:** measure quality instead of guessing.

- Build a small LangSmith dataset (question / expected-answer pairs from your docs).
- `eval/run_eval.py` — run the graph over the dataset and score with `ragas`:
  faithfulness, answer relevancy, context precision/recall. Log results to LangSmith.

**Done when:** you get metric scores per run and can compare before/after changes.

**Later (parked):** [vectara/open_ragbench](https://huggingface.co/datasets/vectara/open_ragbench) as a second eval track after the in-repo Q/A set. Use a small text-only slice + gold PDFs only; do not swap the working `data/` corpus. CC-BY-NC-4.0.

---

## 8. Wire-up + docs

- Slim `app.py` into a thin CLI that builds/loads the index and invokes the graph
  (keep one runnable example query).
- Update `README.md`: new architecture diagram, LangSmith setup, how to run the API,
  Streamlit, and eval.

---

## Suggested build order

1. Config + deps (Step 1)
2. Cosine + hybrid + rerank retriever (Step 3)
3. LangGraph pipeline (Step 4)
4. LangSmith tracing (Step 2 — easy once the graph exists)
5. Robustness pass (Step 5)
6. FastAPI, then Streamlit (Step 6)
7. Eval harness (Step 7)
8. Wire-up + README (Step 8)

## Default decisions (change if you prefer)

- Keep the custom FAISS store (upgraded to cosine) instead of a LangChain VectorStore
  wrapper — preserves your existing code and `faiss_store/`.
- Reranker via `sentence-transformers` CrossEncoder (no heavy new deps).
- FastAPI is the source of truth; Streamlit is a thin client over the API.

## Things you'll need

- API keys: Groq + LangSmith.
- A one-time re-embed of your corpus after switching to cosine.
