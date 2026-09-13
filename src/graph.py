import os
from typing import Any, Dict, List, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.config import configure_langsmith, settings
from src.llm import build_chat_model
from src.retriever import HybridRetriever
from src.search import _NO_CONTEXT_ANSWER, build_qa_messages, citations_from_chunks, invoke_llm
from src.vector_store import FiassVectorStore

REWRITE_SYSTEM = (
    "Rewrite the latest user question as a standalone search query. "
    "Resolve pronouns (he, she, they, his, her) and ellipsis "
    "(how many years?, what about that role?) using ONLY the chat history. "
    "Do not invent a person, name, or topic that is not in that history. "
    "If the history is missing or too ambiguous to resolve a reference, "
    "return the question unchanged. "
    "Return only the rewritten query."
)


def recent_history(
    history: List[Dict[str, str]], n: int | None = None
) -> List[Dict[str, str]]:
    """Last n messages (default 10 = about 5 user/assistant turns)."""
    limit = settings.rewrite_history_messages if n is None else n
    if limit <= 0:
        return []
    return list(history[-limit:])


class GraphState(TypedDict, total=False):
    question: str
    chat_history: List[Dict[str, str]]
    rewritten_query: str
    documents: List[Dict[str, Any]]
    answer: str
    citations: List[Dict[str, Any]]
    relevant: bool


_vectorstore = None
_retriever = None
_llm = None


def load_rag_components():
    """Load vectorstore, hybrid retriever, and LLM. Call explicitly — not at import."""
    global _vectorstore, _retriever, _llm
    if _retriever is not None and _llm is not None:
        return _vectorstore, _retriever, _llm

    configure_langsmith()
    persist_dir = settings.persist_dir
    embedding_model = settings.embedding_model

    vectorstore = FiassVectorStore(
        persist_dir,
        embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    faiss_path = os.path.join(persist_dir, "faiss.index")
    meta_path = os.path.join(persist_dir, "metadata.pk1")

    if os.path.exists(faiss_path) and os.path.exists(meta_path):
        vectorstore.load()
    else:
        from src.data_loader import load_all_docs

        docs = load_all_docs("data")
        vectorstore.build_from_documents(docs)

    retriever = HybridRetriever(vectorstore)
    llm = build_chat_model()
    _vectorstore, _retriever, _llm = vectorstore, retriever, llm
    return _vectorstore, _retriever, _llm


def reset_rag_components() -> None:
    """Drop cached store/retriever/LLM so the next request reloads from disk."""
    global _vectorstore, _retriever, _llm
    _vectorstore = None
    _retriever = None
    _llm = None


def rebuild_index(data_dir: str = "data") -> Dict[str, Any]:
    from src.data_loader import load_all_docs

    docs = load_all_docs(data_dir)
    store = FiassVectorStore(
        settings.persist_dir,
        settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    store.build_from_documents(docs)
    reset_rag_components()
    return {
        "documents": len(docs),
        "chunks": len(store.metadata),
        "persist_dir": settings.persist_dir,
    }


def _append_history(state: GraphState, answer: str) -> List[Dict[str, str]]:
    history = list(state.get("chat_history") or [])
    history.append({"role": "user", "content": state["question"]})
    history.append({"role": "assistant", "content": answer})
    return history


def rewrite_query(state: GraphState) -> GraphState:
    question = state["question"]
    history = recent_history(state.get("chat_history") or [])
    if not history:
        return {"rewritten_query": question}

    _, _, llm = load_rag_components()
    history_lines = "\n".join(
        f"{turn.get('role', 'user')}: {turn.get('content', '')}" for turn in history
    )
    response = invoke_llm(
        llm,
        [
            ("system", REWRITE_SYSTEM),
            ("human", f"Chat history:\n{history_lines}\n\nLatest question:\n{question}"),
        ],
    )
    rewritten = (response.content or "").strip() or question
    return {"rewritten_query": rewritten}


def hybrid_retrieve(state: GraphState) -> GraphState:
    _, retriever, _ = load_rag_components()
    query = state.get("rewritten_query") or state["question"]
    fused_ids = retriever.retrieve_fused_ids(query)
    metadata = retriever.vectorstore.metadata
    documents = [
        {**metadata[i], "_index": i}
        for i in fused_ids
        if 0 <= i < len(metadata)
    ]
    return {"documents": documents}


def rerank(state: GraphState) -> GraphState:
    _, retriever, _ = load_rag_components()
    query = state.get("rewritten_query") or state["question"]
    chunk_ids = [
        doc["_index"]
        for doc in (state.get("documents") or [])
        if "_index" in doc
    ]
    documents = retriever._rerank(query, chunk_ids, settings.top_k)
    return {"documents": documents}


def grade_relevance(state: GraphState) -> GraphState:
    documents = state.get("documents") or []
    return {"relevant": bool(documents)}


def route_after_grade(state: GraphState) -> Literal["generate_answer", "fallback"]:
    if state.get("relevant"):
        return "generate_answer"
    return "fallback"


def generate_answer(state: GraphState) -> GraphState:
    _, _, llm = load_rag_components()
    messages = build_qa_messages(state["question"], state.get("documents") or [])
    response = invoke_llm(llm, messages)
    return {"answer": response.content}


def attach_citations(state: GraphState) -> GraphState:
    answer = state.get("answer") or ""
    return {
        "citations": citations_from_chunks(state.get("documents") or []),
        "chat_history": _append_history(state, answer),
    }


def fallback(state: GraphState) -> GraphState:
    return {
        "answer": _NO_CONTEXT_ANSWER,
        "citations": [],
        "chat_history": _append_history(state, _NO_CONTEXT_ANSWER),
    }


def build_graph(checkpointer=None):
    graph = StateGraph(GraphState)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("hybrid_retrieve", hybrid_retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("grade_relevance", grade_relevance)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("attach_citations", attach_citations)
    graph.add_node("fallback", fallback)

    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "hybrid_retrieve")
    graph.add_edge("hybrid_retrieve", "rerank")
    graph.add_edge("rerank", "grade_relevance")
    graph.add_conditional_edges(
        "grade_relevance",
        route_after_grade,
        {
            "generate_answer": "generate_answer",
            "fallback": "fallback",
        },
    )
    graph.add_edge("generate_answer", "attach_citations")
    graph.add_edge("attach_citations", END)
    graph.add_edge("fallback", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())
