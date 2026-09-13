# RAG

A Retrieval-Augmented Generation (RAG) pipeline in Python: load your own documents, embed them, store the vectors in FAISS, retrieve with hybrid search, and answer with Groq (Ollama as fallback).

## Features

- Load PDF, text, CSV, and PowerPoint (`.pptx`) files from a `data/` folder
- Chunk documents with `RecursiveCharacterTextSplitter`
- Generate embeddings with `sentence-transformers` (`all-MiniLM-L6-v2`)
- Store and search vectors with FAISS (cosine / inner product)
- Hybrid retrieval: dense FAISS + BM25, fused with RRF, then cross-encoder rerank
- Optional LangSmith tracing
- Answer questions with a config-driven LLM (`LLM_PROVIDER` + `LLM_MODEL`; Ollama fallback on rate limits)

## Tech stack

LangChain - LangGraph - LangSmith - sentence-transformers - FAISS - BM25 - Ollama - Groq - pydantic-settings

## Project structure

```
RAG_PRO/
├── app.py                 # CLI: invoke the graph with one example query
├── api/main.py            # FastAPI: /health, /chat, /ingest
├── ui/streamlit_app.py    # Browser chat over POST /chat
├── src/
│   ├── config.py          # Settings from .env (models, keys, chunking)
│   ├── data_loader.py     # Load PDF/TXT/CSV into LangChain documents
│   ├── embedding.py       # Chunk documents and create embeddings
│   ├── vector_store.py    # FAISS index: build, save, load, query
│   ├── retriever.py       # Hybrid retrieve + rerank
│   ├── llm.py             # Provider factory + Groq→Ollama fallback
│   ├── search.py          # RAGSearch: retrieve context + LLM
│   └── graph.py           # LangGraph: rewrite → retrieve → rerank → grade → generate
├── eval/                  # Own-docs Q/A + Open RAG Bench track + ragas runner
├── data/                  # Your source documents (PDF/TXT/CSV)
├── notebook/              # Exploratory notebooks
├── requirements.txt
└── pyproject.toml
```

## How it works

```mermaid
flowchart LR
    docs[Documents in data/] --> loader[data_loader.py]
    loader --> chunker[embedding.py: chunk + embed]
    chunker --> store[vector_store.py: FAISS index]
    query[User query] --> graph[graph.py]
    store --> graph
    subgraph graph [LangGraph]
        rewrite[rewrite_query] --> retrieve[hybrid_retrieve]
        retrieve --> rerank[rerank]
        rerank --> grade[grade_relevance]
        grade -->|relevant| generate[generate_answer]
        grade -->|weak| fallback[fallback]
        generate --> cite[attach_citations]
    end
```

## Setup

This project uses [uv](https://github.com/astral-sh/uv).

```bash
# Install dependencies
uv sync

# Or with pip
pip install -r requirements.txt
```

### Environment variables

Copy `.env.example` to `.env`. Model choice lives in **env** (`LLM_PROVIDER` / `LLM_MODEL`); defaults and validation live in `src/config.py`. `graph.py` and `search.py` never name a vendor.

```bash
cp .env.example .env
ollama pull qwen2.5:7b
```

```
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-120b
GROQ_API_KEY=your_groq_key
LLM_FALLBACK_PROVIDER=ollama
LLM_FALLBACK_MODEL=qwen2.5:7b
```

Switch vendors without code edits:

| Goal | Env |
| --- | --- |
| Groq only | `LLM_PROVIDER=groq`, empty `LLM_FALLBACK_PROVIDER` |
| Ollama only | `LLM_PROVIDER=ollama`, `LLM_MODEL=qwen2.5:7b` |
| OpenAI | `LLM_PROVIDER=openai`, `LLM_MODEL=gpt-4o-mini`, `OPENAI_API_KEY=...` (`uv add langchain-openai`) |
| Anthropic | `LLM_PROVIDER=anthropic`, `LLM_MODEL=claude-sonnet-4-5`, `ANTHROPIC_API_KEY=...` (`uv add langchain-anthropic`) |

Optional LangSmith tracing:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=rag-pro
```

`from src.config import settings` works without a Groq key. Groq is only required when `LLM_PROVIDER=groq` (or when it is the fallback).

## Usage

1. Put your documents in the `data/` folder (PDF, TXT, CSV, or PPTX).
2. Run the pipeline:

```bash
uv run python app.py
```

`app.py` invokes the LangGraph app (rewrite → retrieve → rerank → grade → generate) with a `thread_id` so follow-ups can use memory. Edit the `query` variable to ask your own questions.

### API

```bash
uv run uvicorn api.main:app --reload
```

- `GET /health` — liveness
- `POST /chat` — `{"question": "...", "thread_id": "t1"}` → `{answer, citations, thread_id}`
- `POST /ingest` — rebuild the FAISS index from `data/`

### Streamlit UI

Start the API first, then:

```bash
uv run streamlit run ui/streamlit_app.py
```

If the page errors with `No module named 'ui'`, run that from the project root (`RAG_PRO/`).

Uses `RAG_API_URL` if set (default `http://127.0.0.1:8000`). The browser session keeps one `thread_id` so follow-ups use graph memory.

### Evaluation (own docs)

First eval track is 18 questions in `eval/dataset.json` written from this repo’s `data/` (facts, follow-ups, and should-unknown). It does **not** download Open RAG Bench.

```bash
uv run python eval/run_eval.py
```

Needs a Groq key when `LLM_PROVIDER=groq`, plus a running Ollama daemon if fallback is Ollama. Ragas answer relevancy uses `strictness=1` (Groq only allows `n=1`). Unknown questions are scored by abstain rate, not ragas. With `LANGSMITH_TRACING=true` the runner upserts dataset `rag-pro-own-docs`. Scores write to `eval/last_results.json` (gitignored).

`uv run python eval/run_eval.py --limit 3 --no-score` runs the graph only. Sleep between examples defaults to 0 when the primary provider is Ollama, else 8s (Groq TPM).

### Evaluation (Open RAG Bench)

Second track is a **text-only extractive** slice of [vectara/open_ragbench](https://huggingface.co/datasets/vectara/open_ragbench) (CC-BY-NC-4.0, research/personal). It does **not** replace `data/`. Gold paper sections are indexed into `eval/.cache/open_ragbench/store/` (gitignored). Retrieval is scored with official `qrels` (`doc_id` + `section_id`); generation can still use ragas.

```bash
uv run python eval/run_eval.py --track open-ragbench --limit 30 --no-score
uv run python eval/run_eval.py --track open-ragbench --limit 30
```

Default slice size is 30. The store includes official **hard-negative** papers (never a gold doc) so recall is not measured in a gold-only haystack. `--no-hard-negatives` indexes gold papers only. `--rebuild-index` forces a rebuild; a store that is missing gold or hard-negative IDs is rebuilt automatically. Skip image/table queries until ingest is multimodal. Prefer `--no-score` or an Ollama judge so Groq TPM is not spent on ragas.

### Latest scores

Local runs (Sep 2026). Re-run the commands above to refresh `eval/last_results.json` and `eval/last_open_ragbench.json` (both gitignored).

**Own docs** — 18 questions (12 fact, 3 follow-up, 3 should-unknown):

| Metric | Score |
| --- | --- |
| Unknown abstain | 1.00 (3/3) |
| Token overlap vs gold | 0.82 |
| Mean citations | 2.2 |
| Faithfulness / context precision / context recall (ragas) | 1.00* |
| Answer relevancy (ragas) | 0.84* |

\*Ragas judges shared Groq’s 8k TPM with generation; treat the 1.00s as directional.

**Open RAG Bench** — 30 text-only extractive queries, official qrels:

| Index | Papers / chunks | Doc recall | Section recall | Token overlap |
| --- | --- | --- | --- | --- |
| Gold papers only | 28 / 8.7k | 1.00 | 0.97 (29/30) | 0.85 |
| Gold + hard negatives | 629 / 156k | 1.00 | 0.97 (29/30) | 0.85 |

One section miss (NVILA) was the gold paper, neighboring section. Adding 604 hard-negative papers did not change recall. Not measured: abstractive, table, or image queries.

## Notes

- The FAISS index is saved to `faiss_store/` and is git-ignored (regenerated locally).
- Never commit your `.env` — it holds your API key and is already in `.gitignore`.
