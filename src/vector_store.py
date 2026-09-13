"""
Here is what vector store will be used
"""

import os
import faiss
from langsmith import traceable
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline


class FiassVectorStore:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok= True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Loaded embedding model {embedding_model}")

    def build_from_documents( self, documents: List[Any]):
        if not documents:
            raise ValueError(
                "No documents to index. Check the data/ folder and file load errors."
            )
        print(f"[INFO] Bulding Vector store from {len(documents)} raw documents... ")
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size =self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        if not chunks:
            raise ValueError("Documents produced no chunks after splitting.")
        embeddings = emb_pipe.emb_chunks(chunks)
        metadatas = []
        for i , chunk in enumerate(chunks):
            metadatas.append({
                "text": chunk.page_content,
                "source": chunk.metadata.get('source', 'Unknown'),
                "page": chunk.metadata.get("page", None),
                "chunk_id": i
            })
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.array, metadatas: List[Any] = None):
        embeddings = np.array(embeddings, dtype="float32")
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("Cannot add an empty embedding matrix to the FAISS index.")
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        if metadatas: 
            self.metadata.extend(metadatas)
        print(f"[INFO] Add {embeddings.shape[0]} vectors to Fiass index.")

    def save(self):
        if self.index is None:
            raise ValueError("Cannot save: FAISS index has not been built.")
        faiss_path = os.path.join(self.persist_dir, 'faiss.index')
        meta_path = os.path.join(self.persist_dir, 'metadata.pk1')
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, 'faiss.index')
        meta_path = os.path.join(self.persist_dir, 'metadata.pk1')
        if not os.path.exists(faiss_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"FAISS store not found in {self.persist_dir}. "
                "Build the index from data/ first."
            )
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, 'rb') as f:
            self.metadata = pickle.load(f)
        self._validate_loaded_index()
        print(f'[INFO] Loaded faiss index and metadata from {self.persist_dir}')

    def _validate_loaded_index(self) -> None:
        if self.index is None:
            raise ValueError("FAISS index is missing after load.")
        if not isinstance(self.metadata, list):
            raise ValueError("Metadata file is corrupt; expected a list of chunk records.")
        ntotal = self.index.ntotal
        nmeta = len(self.metadata)
        if ntotal != nmeta:
            raise ValueError(
                f"Index/metadata mismatch: FAISS has {ntotal} vectors but metadata "
                f"has {nmeta} chunks. Delete faiss_store/ and rebuild the index."
            )
        if self.index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError(
                "FAISS index is not inner-product (cosine). "
                "Delete faiss_store/ and rebuild after the cosine switch."
            )
        getter = getattr(self.model, "get_sentence_embedding_dimension", None)
        model_dim = getter() if callable(getter) else None
        if isinstance(model_dim, int) and self.index.d != model_dim:
            raise ValueError(
                f"Embedding dimension mismatch: index is {self.index.d}-d but the "
                f"current model is {model_dim}-d. Delete faiss_store/ and rebuild."
            )

    def search(self, query_embedding: np.array, top_k: int =5):
        if self.index is None or self.index.ntotal == 0:
            return []
        D, I = self.index.search(query_embedding, top_k)
        n = len(self.metadata)
        results = []
        for idx, dist in zip(I[0], D[0]):
            # FAISS pads with -1 when top_k > ntotal; -1 would index the last chunk.
            if idx < 0 or idx >= n:
                continue
            results.append({
                "index": int(idx),
                "distance": float(dist),
                "metadata": self.metadata[idx],
            })
        return results
    @traceable(run_type='retriever', name = 'fiass_query')
    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Quering vector store for '{query_text}")

        query_emb = self.model.encode([query_text]).astype('float32')
        faiss.normalize_L2(query_emb)
        return self.search(query_emb, top_k=top_k)






