from unittest.mock import MagicMock, patch

import httpx

from ui.client import health, history_payload, send_chat


def test_send_chat_posts_question_and_thread():
    response = MagicMock()
    response.json.return_value = {
        "answer": "Beam search keeps k candidates.",
        "citations": [{"source": "a.pdf", "page": 4}],
        "thread_id": "t1",
    }
    response.raise_for_status.return_value = None
    with patch("ui.client.httpx.post", return_value=response) as post:
        body = send_chat("What is beam search?", "t1", api_url="http://127.0.0.1:8000")
    post.assert_called_once()
    assert post.call_args.args[0] == "http://127.0.0.1:8000/chat"
    assert post.call_args.kwargs["json"] == {
        "question": "What is beam search?",
        "thread_id": "t1",
        "chat_history": [],
    }
    assert body["answer"].startswith("Beam search")


def test_send_chat_posts_history():
    response = MagicMock()
    response.json.return_value = {"answer": "ok", "citations": [], "thread_id": "t1"}
    response.raise_for_status.return_value = None
    history = [{"role": "user", "content": "Who is Jane Doe?"}]
    with patch("ui.client.httpx.post", return_value=response) as post:
        send_chat("how many years?", "t1", chat_history=history, api_url="http://x")
    assert post.call_args.kwargs["json"]["chat_history"] == history


def test_history_payload_keeps_last_n_text_turns():
    messages = [
        {"role": "user", "content": "old", "citations": []},
        {"role": "assistant", "content": "a1", "citations": [{"source": "x"}]},
        {"role": "user", "content": "new", "citations": []},
        {"role": "system", "content": "ignore"},
        {"role": "assistant", "content": "   "},
    ]
    assert history_payload(messages, limit=2) == [
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "new"},
    ]


def test_health_false_when_api_down():
    with patch("ui.client.httpx.get", side_effect=httpx.ConnectError("down")):
        assert health("http://127.0.0.1:8000") is False
