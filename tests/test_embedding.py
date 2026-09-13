from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.embedding import EmbeddingPipeline


def test_chunks_stay_within_chunk_size():
    with patch("src.embedding.SentenceTransformer", return_value=MagicMock()):
        pipe = EmbeddingPipeline(chunk_size=50, chunk_overlap=10)
    docs = [Document(page_content="word " * 80)]
    chunks = pipe.chunk_documents(docs)
    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 50 for chunk in chunks)
