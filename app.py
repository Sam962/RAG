import os

from src.data_loader import load_all_docs
from src.vector_store import FiassVectorStore
from src.search import RAGSearch




## return doc 
if __name__ == "__main__":
    store = FiassVectorStore("faiss_store")

    # Build the index from documents only the first time, then reuse it.
    faiss_path = os.path.join("faiss_store", "faiss.index")
    meta_path = os.path.join("faiss_store", "metadata.pk1")
    if os.path.exists(faiss_path) and os.path.exists(meta_path):
        store.load()
    else:
        docs = load_all_docs("data")
        store.build_from_documents(docs)

    print(store.query("What is beam search, and how does it differ from greedy decoding?", top_k =3))

    rag_search = RAGSearch()
    query = "What is beam search, and how does it differ from greedy decoding?" # write query based on your doc! :) 
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("summary", summary)

    