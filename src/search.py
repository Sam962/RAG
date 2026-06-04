import os 
from dotenv import load_dotenv
from src.vector_store import FiassVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2"):
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
        llm_model = "llama-3.3-70b-versatile"
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("Groq_API_KEY not found!!!")

        self.llm= ChatGroq(groq_api_key = groq_api_key , model_name = llm_model)
        print(f"[INFO] Groq LLM initialize: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5)-> str:
        results = self.vectorstore.query(query, top_k = top_k)
        texts = [r['metadata'].get('text', '') for r in results if r['metadata']]
        context = '\n\n'.join(texts)

        if not context:
            return "No relevant documents found"
        prompt = f"""Summurize the following context for qury: '{query}' \n\nContext: \n '{context}"""
        response = self.llm.invoke([prompt])
        return response.content




