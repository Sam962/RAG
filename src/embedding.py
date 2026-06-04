from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer 
import numpy as np
from src.data_loader import load_all_docs


class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model {model_name}")

    # chunk doc and split 
    def  chunk_documents (self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter (

            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len,
            separators=["\n\n", "\n", ". ", " ", ""]

        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    # embed chunks pass through the model to convert it into Vector Embeddings
    def emb_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embedding for {len(texts)}")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embedding shape: {embeddings.shape}")

        return embeddings




