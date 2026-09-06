import os 
from langchain_groq import ChatGroq
from src.config import settings, configure_langsmith
from src.vector_store import FiassVectorStore
from src.retriever import HybridRetriever
from typing import TypedDict, List, Dict, Any
from langgraph import StateGraph, START, END


# create graph state
class GraphState(TypedDict, total = False):
    question: str
    chat_history: List[Dict[str,str]]
    rewritten_query: str
    documents: List[Dict[str, Any]]
    answer: str
    citations: List[Dict[str, Any]]

configure_langsmith()


def load_rag_components():
    """Load vectorstore, hybrid retriever, and LLM - shared by all nodes."""
    persist_dir = settings.persist_dir
    embedding_model = settings.embedding_model

    vectorstore = FiassVectorStore(persist_dir, embedding_model)
    faiss_path = os.path.join(persist_dir, "faiss.index")
    meta_path = os.path.join(persist_dir, 'metadata.pk1')

    if os.path.exists(faiss_path) and os.path.exists(meta_path):
        vectorstore.load()

    else:
        from src.data_loader import load_all_docs
        docs = load_all_docs('data')
        vectorstore.build_from_documents(docs)

    retriever = HybridRetriever(vectorstore)
    llm = ChatGroq(
        groq_api_key = settings.groq_api_key,
        model_name = settings.groq_model
        )
    return vectorstore, retriever, llm

# load once at module level
_, RETRIEVER, LLM = load_rag_components()

def hybrid_retriever(state:GraphState) -> GraphState:
    query = state.get("rewritten_query") or state['question']




