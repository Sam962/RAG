import os
from langsmith import traceable 
from src.vector_store import FiassVectorStore
from src.config import settings, configure_langsmith
from langchain_groq import ChatGroq
from src.retriever import HybridRetriever

configure_langsmith()  
class RAGSearch:
    def __init__(self, persist_dir: str | None = None , embedding_model: str | None = None):
        persist_dir = persist_dir or settings.persist_dir
        embedding_model = embedding_model or settings.embedding_model
        self.vectorstore = FiassVectorStore(persist_dir, embedding_model)

        #laod or build vectorstore
        faiss_path = os.path.join(persist_dir, 'faiss.index')
        meta_path = os.path.join(persist_dir, 'metadata.pk1')
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_docs
            docs = load_all_docs('data')
            self.vectorstore.build_from_documents(docs)

        else:
            self.vectorstore.load()
        self.retriever =HybridRetriever(self.vectorstore)
        self.llm = ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=settings.groq_model,
        )
        print(f"[INFO] Groq LLM initialize: {settings.groq_model}")
        
    @traceable(run_type='chain', name = 'search_and_summarize')
    def search_and_summarize(self, query: str, top_k: int = 5)-> str:
        results = self.retriever.retrieve(query, k = top_k)
        texts = [r.get('text', '') for r in results]
        context = '\n\n'.join(texts)

        if not context:
            return "No relevant documents found"
        prompt = f"""Summurize the following context for qury: '{query}' \n\nContext: \n '{context}"""
        response = self.llm.invoke([prompt])
        return response.content




