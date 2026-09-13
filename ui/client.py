import os
from typing import Any

import httpx

DEFAULT_API_URL = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000")


def history_payload(messages: list[dict[str, Any]], limit: int = 10) -> list[dict[str, str]]:
    """Last N user/assistant turns as {role, content} for rewrite."""
    turns: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            turns.append({"role": role, "content": content})
    return turns[-limit:] if limit > 0 else []


def send_chat(
    question: str,
    thread_id: str,
    chat_history: list[dict[str, str]] | None = None,
    api_url: str = DEFAULT_API_URL,
    timeout: float = 120.0,
) -> dict[str, Any]:
    response = httpx.post(
        f"{api_url.rstrip('/')}/chat",
        json={
            "question": question,
            "thread_id": thread_id,
            "chat_history": list(chat_history or []),
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def health(api_url: str = DEFAULT_API_URL, timeout: float = 5.0) -> bool:
    try:
        response = httpx.get(f"{api_url.rstrip('/')}/health", timeout=timeout)
        return response.status_code == 200 and response.json().get("status") == "ok"
    except httpx.HTTPError:
        return False
