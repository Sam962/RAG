from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.graph import build_graph, rebuild_index

app = FastAPI(title="RAG Pro", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_graph = None


def get_compiled_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


class Citation(BaseModel):
    source: str
    page: int | None = None
    chunk_id: int | None = None
    rerank_score: float | None = None


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    thread_id: str = "default"
    chat_history: list[ChatTurn] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    thread_id: str


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    persist_dir: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    result = get_compiled_graph().invoke(
        {
            "question": body.question,
            "chat_history": [
                {"role": turn.role, "content": turn.content} for turn in body.chat_history
            ],
        },
        config={"configurable": {"thread_id": body.thread_id}},
    )
    citations = [Citation(**_citation_payload(item)) for item in (result.get("citations") or [])]
    return ChatResponse(
        answer=result.get("answer") or "",
        citations=citations,
        thread_id=body.thread_id,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    counts = rebuild_index()
    return IngestResponse(**counts)


def _citation_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": item.get("source", "unknown"),
        "page": item.get("page"),
        "chunk_id": item.get("chunk_id"),
        "rerank_score": item.get("rerank_score"),
    }
