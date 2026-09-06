# RAG

A Retrieval-Augmented Generation (RAG) pipeline in Python: load your own documents, embed them, store the vectors in FAISS, retrieve with hybrid search, and answer questions with a Groq-hosted LLM.

## Features

- Load PDF, text, and CSV files from a `data/` folder (via LangChain loaders)
- Chunk documents with `RecursiveCharacterTextSplitter`
- Generate embeddings with `sentence-transformers` (`all-MiniLM-L6-v2`)
- Store and search vectors with FAISS (cosine / inner product)
- Hybrid retrieval: dense FAISS + BM25, fused with RRF, then cross-encoder rerank
- Optional LangSmith tracing
- Summarize retrieved context with a Groq LLM (`llama-3.3-70b-versatile`)

## Tech stack

LangChain - LangGraph - LangSmith - sentence-transformers - FAISS - BM25 - Groq - pydantic-settings

## Project structure

```
RAG_PRO/
├── app.py                 # Entry point: build store, query, summarize
├── src/
│   ├── config.py          # Settings from .env (models, keys, chunking)
│   ├── data_loader.py     # Load PDF/TXT/CSV into LangChain documents
│   ├── embedding.py       # Chunk documents and create embeddings
│   ├── vector_store.py    # FAISS index: build, save, load, query
│   ├── retriever.py       # Hybrid retrieve + rerank
│   ├── search.py          # RAGSearch: retrieve context + Groq LLM
│   └── graph.py           # LangGraph scaffolding (in progress)
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
    query[User query] --> retrieve[retriever.py: hybrid + rerank]
    store --> retrieve
    retrieve --> llm[search.py: Groq LLM]
    llm --> answer[Answer / summary]
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

Copy the example file and add your Groq API key (get one at https://console.groq.com/keys):

```bash
cp .env.example .env
```

Then edit `.env`:

```
GROQ_API_KEY=your_key_here
```

Optional LangSmith tracing:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=rag-pro
```

`src/config.py` requires a real `GROQ_API_KEY` (not the placeholder) before the app will start.

## Usage

1. Put your documents in the `data/` folder (PDF, TXT, or CSV).
2. Run the pipeline:

```bash
uv run python app.py
```

`app.py` builds the FAISS store from your documents on the first run, then reuses the saved index on later runs, and prints an LLM-generated summary. Edit the `query` variable in `app.py` to ask your own questions.

## Notes

- The FAISS index is saved to `faiss_store/` and is git-ignored (regenerated locally).
- Never commit your `.env` — it holds your API key and is already in `.gitignore`.
