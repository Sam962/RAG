from unittest.mock import MagicMock, patch

from langgraph.graph import END, START, StateGraph

from src import graph as graph_mod
from src.graph import (
    GraphState,
    build_graph,
    fallback,
    grade_relevance,
    recent_history,
    rewrite_query,
    route_after_grade,
)
from src.search import _NO_CONTEXT_ANSWER


def test_stategraph_imports_from_langgraph_graph():
    assert StateGraph is not None
    assert START == "__start__"
    assert END == "__end__"
    assert "question" in GraphState.__annotations__


def test_build_graph_does_not_load_models():
    with patch("src.graph.FiassVectorStore") as mock_store:
        with patch("src.graph.HybridRetriever") as mock_retriever:
            with patch("src.graph.build_chat_model") as mock_llm:
                app = build_graph()
                mock_store.assert_not_called()
                mock_retriever.assert_not_called()
                mock_llm.assert_not_called()
    assert app is not None


def test_import_does_not_eagerly_load_components():
    assert graph_mod._retriever is None
    assert graph_mod._llm is None
    assert graph_mod._vectorstore is None


def test_rewrite_query_without_history_is_identity():
    result = rewrite_query({"question": "What is beam search?", "chat_history": []})
    assert result["rewritten_query"] == "What is beam search?"


def test_rewrite_without_history_leaves_pronouns_unchanged(monkeypatch):
    monkeypatch.setattr("src.graph.settings.query_identity", "Sam Aldehayyat")
    result = rewrite_query(
        {"question": "how many years of experience he has?", "chat_history": []}
    )
    assert result["rewritten_query"] == "how many years of experience he has?"


def test_rewrite_uses_history_not_configured_identity(monkeypatch):
    monkeypatch.setattr("src.graph.settings.query_identity", "Sam Aldehayyat")
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content="how many years of experience does Jane Doe have?"
    )
    history = [
        {"role": "user", "content": "Who is Jane Doe?"},
        {"role": "assistant", "content": "Jane Doe is an engineer."},
    ]
    with patch("src.graph.load_rag_components", return_value=(None, None, llm)):
        result = rewrite_query(
            {
                "question": "how many years of experience he has?",
                "chat_history": history,
            }
        )
    assert result["rewritten_query"] == "how many years of experience does Jane Doe have?"
    assert "Sam" not in result["rewritten_query"]


def test_recent_history_keeps_only_last_n_messages():
    history = [{"role": "user", "content": str(i)} for i in range(20)]
    window = recent_history(history, n=10)
    assert [turn["content"] for turn in window] == [str(i) for i in range(10, 20)]


def test_rewrite_query_sends_only_recent_window():
    history = [{"role": "user", "content": f"old-{i}"} for i in range(15)]
    history[-1] = {"role": "user", "content": "beam search"}
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="What is beam search?")

    with patch("src.graph.load_rag_components", return_value=(None, None, llm)):
        rewrite_query({"question": "how does it work?", "chat_history": history})

    prompt = llm.invoke.call_args[0][0][1][1]
    assert "old-0" not in prompt
    assert "old-4" not in prompt
    assert "beam search" in prompt
    assert "old-5" in prompt


def test_grade_routes_empty_docs_to_fallback():
    assert route_after_grade(grade_relevance({"documents": []})) == "fallback"
    assert route_after_grade(grade_relevance({"documents": [{"text": "x"}]})) == "generate_answer"


def test_fallback_returns_structured_unknown():
    result = fallback({"question": "What is quantum foam?"})
    assert result["answer"] == _NO_CONTEXT_ANSWER
    assert result["citations"] == []
    assert result["chat_history"][-1]["content"] == _NO_CONTEXT_ANSWER


def _patched_components(retriever, llm):
    return patch("src.graph.load_rag_components", return_value=(None, retriever, llm))


def test_invoke_returns_answer_and_citations():
    retriever = MagicMock()
    retriever.retrieve_fused_ids.return_value = [0]
    retriever.vectorstore.metadata = [
        {"text": "Beam search keeps the top k candidates.", "source": "a.pdf", "page": 4, "chunk_id": 0}
    ]
    retriever._rerank.return_value = [
        {
            "text": "Beam search keeps the top k candidates.",
            "source": "a.pdf",
            "page": 4,
            "chunk_id": 0,
            "rerank_score": 6.1,
        }
    ]
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="Beam search keeps k candidates.")

    with _patched_components(retriever, llm):
        app = build_graph()
        result = app.invoke(
            {"question": "What is beam search?"},
            config={"configurable": {"thread_id": "t1"}},
        )

    assert result["answer"] == "Beam search keeps k candidates."
    assert result["citations"][0]["page"] == 4
    assert result["citations"][0]["rerank_score"] == 6.1


def test_weak_retrieve_uses_fallback_without_generate():
    retriever = MagicMock()
    retriever.retrieve_fused_ids.return_value = [0]
    retriever.vectorstore.metadata = [{"text": "unrelated", "source": "a.pdf", "page": 1, "chunk_id": 1}]
    retriever._rerank.return_value = []
    llm = MagicMock()

    with _patched_components(retriever, llm):
        app = build_graph()
        result = app.invoke(
            {"question": "What is quantum foam?"},
            config={"configurable": {"thread_id": "t-weak"}},
        )

    llm.invoke.assert_not_called()
    assert result["answer"] == _NO_CONTEXT_ANSWER
    assert result["citations"] == []


def test_follow_up_rewrites_using_history():
    retriever = MagicMock()
    retriever.retrieve_fused_ids.return_value = [0]
    retriever.vectorstore.metadata = [
        {"text": "Greedy decoding picks the top token.", "source": "a.pdf", "page": 4, "chunk_id": 0}
    ]
    retriever._rerank.return_value = [
        {
            "text": "Greedy decoding picks the top token.",
            "source": "a.pdf",
            "page": 4,
            "chunk_id": 0,
            "rerank_score": 5.0,
        }
    ]
    llm = MagicMock()
    llm.invoke.side_effect = [
        MagicMock(content="How does greedy decoding work?"),
        MagicMock(content="Greedy decoding picks one token at a time."),
    ]

    history = [
        {"role": "user", "content": "What is beam search?"},
        {"role": "assistant", "content": "Beam search keeps k sequences."},
    ]
    with _patched_components(retriever, llm):
        app = build_graph()
        result = app.invoke(
            {"question": "How does it differ from greedy decoding?", "chat_history": history},
            config={"configurable": {"thread_id": "t-follow"}},
        )

    retriever.retrieve_fused_ids.assert_called()
    assert retriever.retrieve_fused_ids.call_args[0][0] == "How does greedy decoding work?"
    assert "Greedy decoding" in result["answer"]
