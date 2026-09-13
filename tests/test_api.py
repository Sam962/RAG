from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_requires_question():
    response = client.post("/chat", json={"thread_id": "t1"})
    assert response.status_code == 422


def test_chat_rejects_empty_question():
    response = client.post("/chat", json={"question": "", "thread_id": "t1"})
    assert response.status_code == 422


def test_chat_returns_answer_and_citations():
    graph = MagicMock()
    graph.invoke.return_value = {
        "answer": "Beam search keeps k candidates.",
        "citations": [
            {
                "source": "a.pdf",
                "page": 4,
                "chunk_id": 9,
                "rerank_score": 6.1,
            }
        ],
    }
    with patch("api.main.get_compiled_graph", return_value=graph):
        response = client.post(
            "/chat",
            json={"question": "What is beam search?", "thread_id": "t1"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Beam search keeps k candidates."
    assert body["thread_id"] == "t1"
    assert body["citations"][0]["page"] == 4
    graph.invoke.assert_called_once()
    assert graph.invoke.call_args.args[0] == {
        "question": "What is beam search?",
        "chat_history": [],
    }
    assert graph.invoke.call_args.kwargs["config"]["configurable"]["thread_id"] == "t1"


def test_chat_forwards_history():
    graph = MagicMock()
    graph.invoke.return_value = {"answer": "About 4 years.", "citations": []}
    history = [
        {"role": "user", "content": "Who is Jane Doe?"},
        {"role": "assistant", "content": "Jane Doe is an engineer."},
    ]
    with patch("api.main.get_compiled_graph", return_value=graph):
        response = client.post(
            "/chat",
            json={
                "question": "how many years of experience he has?",
                "thread_id": "t-hist",
                "chat_history": history,
            },
        )
    assert response.status_code == 200
    assert graph.invoke.call_args.args[0]["chat_history"] == history


def test_ingest_reports_counts():
    with patch(
        "api.main.rebuild_index",
        return_value={"documents": 3, "chunks": 12, "persist_dir": "faiss_store"},
    ):
        response = client.post("/ingest")
    assert response.status_code == 200
    assert response.json() == {
        "documents": 3,
        "chunks": 12,
        "persist_dir": "faiss_store",
    }
