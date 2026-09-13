from unittest.mock import MagicMock, patch

import pytest

from src.llm import (
    FallbackChatModel,
    build_backend,
    build_chat_model,
    llm_provider,
    should_fallback,
)


def test_llm_provider_reads_settings(monkeypatch):
    monkeypatch.setattr("src.llm.settings.llm_provider", "Ollama")
    assert llm_provider() == "ollama"


def test_build_backend_ollama(monkeypatch):
    monkeypatch.setattr("src.llm.settings.ollama_base_url", "http://127.0.0.1:11434")
    monkeypatch.setattr("src.llm.settings.ollama_timeout", 120.0)
    fake = MagicMock()
    with patch("langchain_ollama.ChatOllama", return_value=fake) as ctor:
        model = build_backend("ollama", "qwen2.5:7b")
    ctor.assert_called_once()
    assert model is fake


def test_build_backend_groq_uses_key(monkeypatch):
    monkeypatch.setattr("src.llm.require_groq_api_key", lambda: "gsk_test")
    fake = MagicMock()
    with patch("langchain_groq.ChatGroq", return_value=fake) as ctor:
        model = build_backend("groq", "openai/gpt-oss-120b")
    ctor.assert_called_once()
    assert model is fake


def test_build_backend_unknown_uses_init_chat_model(monkeypatch):
    fake = MagicMock()
    with patch("langchain.chat_models.init_chat_model", return_value=fake) as init:
        model = build_backend("openai", "gpt-4o-mini")
    init.assert_called_once()
    assert init.call_args.kwargs["model_provider"] == "openai"
    assert model is fake


def test_build_chat_model_wraps_fallback(monkeypatch):
    monkeypatch.setattr("src.llm.settings.llm_provider", "groq")
    monkeypatch.setattr("src.llm.settings.llm_model", "openai/gpt-oss-120b")
    monkeypatch.setattr("src.llm.settings.llm_fallback_provider", "ollama")
    monkeypatch.setattr("src.llm.settings.llm_fallback_model", "qwen2.5:7b")
    primary = MagicMock()
    with patch("src.llm.build_backend", return_value=primary) as backend:
        model = build_chat_model()
    assert isinstance(model, FallbackChatModel)
    assert model.primary is primary
    assert backend.call_count == 1
    assert backend.call_args.kwargs.get("max_retries") == 0
    assert model._fallback_factory is not None


def test_build_chat_model_skips_fallback_when_blank(monkeypatch):
    monkeypatch.setattr("src.llm.settings.llm_provider", "ollama")
    monkeypatch.setattr("src.llm.settings.llm_model", "qwen2.5:7b")
    monkeypatch.setattr("src.llm.settings.llm_fallback_provider", "")
    primary = MagicMock()
    with patch("src.llm.build_backend", return_value=primary) as backend:
        model = build_chat_model()
    assert model._fallback_factory is None
    assert backend.call_args.kwargs.get("max_retries") is None


def test_fallback_used_on_rate_limit():
    class RateLimitError(Exception):
        pass

    primary = MagicMock()
    primary.invoke.side_effect = RateLimitError("429 rate_limit_exceeded")
    fallback = MagicMock()
    fallback.invoke.return_value = MagicMock(content="from-ollama")
    model = FallbackChatModel(primary, fallback_factory=lambda: fallback)
    result = model.invoke([("human", "hi")])
    assert result.content == "from-ollama"
    fallback.invoke.assert_called_once()


def test_should_fallback_rate_limit_not_bad_request():
    assert should_fallback(Exception("Error code: 429 rate_limit_exceeded"))
    assert not should_fallback(ValueError("invalid model"))
