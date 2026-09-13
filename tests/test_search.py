from unittest.mock import MagicMock, patch

from src.search import (
    GROUNDING_SYSTEM,
    _NO_CONTEXT_ANSWER,
    RAGSearch,
    build_qa_messages,
    citations_from_chunks,
    invoke_llm,
    retry_after_seconds,
)


def test_qa_prompt_is_grounded_not_summarize():
    chunks = [
        {
            "text": "Beam search keeps the top k candidates.",
            "source": "notes.txt",
            "page": 2,
            "chunk_id": 0,
        }
    ]
    messages = build_qa_messages("What is beam search?", chunks)
    assert messages[0] == ("system", GROUNDING_SYSTEM)
    role, content = messages[1]
    assert role == "human"
    assert "Question:\nWhat is beam search?" in content
    assert "Beam search keeps the top k candidates." in content
    assert "notes.txt, page 2" in content
    assert "summari" not in content.lower()
    assert "ONLY" in GROUNDING_SYSTEM
    assert "I don't know" in GROUNDING_SYSTEM
    assert "dates and numbers" in GROUNDING_SYSTEM


def test_citations_from_chunks():
    citations = citations_from_chunks(
        [{"source": "a.pdf", "page": 4, "chunk_id": 9, "text": "x"}]
    )
    assert citations == [
        {"source": "a.pdf", "page": 4, "chunk_id": 9, "rerank_score": None}
    ]


def test_ask_without_context_does_not_call_llm():
    search = RAGSearch.__new__(RAGSearch)
    search.retriever = MagicMock()
    search.retriever.retrieve.return_value = []
    search.llm = MagicMock()

    result = RAGSearch.ask(search, "What is quantum foam?")

    search.llm.invoke.assert_not_called()
    assert result["used_context"] is False
    assert result["citations"] == []
    assert result["answer"] == _NO_CONTEXT_ANSWER


def test_retry_after_parses_groq_ms():
    err = Exception(
        "Rate limit reached ... Please try again in 577.5ms. Need more tokens?"
    )
    assert abs(retry_after_seconds(err) - 0.5775) < 0.001


def test_invoke_llm_uses_fallback_without_sleeping():
    from src.llm import FallbackChatModel

    class RateLimitError(Exception):
        pass

    primary = MagicMock()
    primary.invoke.side_effect = RateLimitError("Error code: 429")
    fallback = MagicMock()
    fallback.invoke.return_value = MagicMock(content="from-ollama")
    llm = FallbackChatModel(primary, fallback_factory=lambda: fallback)
    with patch("src.search.time.sleep") as sleep:
        response = invoke_llm(llm, [("human", "hi")], attempts=6)
    assert response.content == "from-ollama"
    sleep.assert_not_called()


def test_invoke_llm_retries_rate_limit():
    class RateLimitError(Exception):
        pass

    llm = MagicMock()
    llm.invoke.side_effect = [
        RateLimitError("Please try again in 1ms"),
        MagicMock(content="ok"),
    ]
    with patch("src.search.time.sleep"):
        response = invoke_llm(llm, [("human", "hi")], attempts=3)
    assert response.content == "ok"
    assert llm.invoke.call_count == 2
