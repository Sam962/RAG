from src.data_loader import load_all_docs
from src.vector_store import FiassVectorStore
from src.search import RAGSearch




## return doc 
if __name__ == "__main__":
    # Its already initiated uncommented for the first time.

    docs    = load_all_docs("data")
    store = FiassVectorStore("faiss_store")
    store.build_from_documents(docs)  # undo it to build faiss 

    # if it already built then 
    store.load()
    print(store.query("What is beam search, and how does it differ from greedy decoding?", top_k =3))

    rag_search = RAGSearch()
    query = "What is beam search, and how does it differ from greedy decoding?" # write query based on your doc! :) 
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("summary", summary)

    