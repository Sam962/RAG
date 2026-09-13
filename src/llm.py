"""Config-driven chat models. Switch vendor/model in .env, not in graph/search."""

from __future__ import annotations

from typing import Any, Callable

from src.config import require_groq_api_key, settings


def llm_provider() -> str:
    return (settings.llm_provider or "groq").strip().lower()


def _model_for(provider: str) -> str:
    provider = (provider or "").strip().lower()
    if provider == llm_provider():
        return (settings.llm_model or "").strip()
    if provider == (settings.llm_fallback_provider or "").strip().lower():
        return (settings.llm_fallback_model or "").strip()
    raise ValueError(
        f"No model id for provider {provider!r}. Set LLM_MODEL or LLM_FALLBACK_MODEL in .env."
    )


def should_fallback(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "ratelimit" in name or "rate_limit" in text or "429" in text:
        return True
    if "timeout" in name or "timeout" in text:
        return True
    if "connection" in name or "connect" in text:
        return True
    if "503" in text or "502" in text or "unavailable" in text:
        return True
    return False


def build_backend(
    provider: str, model: str, *, max_retries: int | None = None
) -> Any:
    """One place to add a vendor. Unknown names go through LangChain init_chat_model."""
    provider = (provider or "").strip().lower()
    model = (model or "").strip()
    if not provider or not model:
        raise ValueError("LLM provider and model are required.")

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        print(f"[INFO] LLM initialize: ollama:{model}")
        return ChatOllama(
            model=model,
            base_url=settings.ollama_base_url,
            client_kwargs={"timeout": settings.ollama_timeout},
        )
    if provider == "groq":
        from langchain_groq import ChatGroq

        retries = settings.groq_max_retries if max_retries is None else max_retries
        print(f"[INFO] LLM initialize: groq:{model}")
        return ChatGroq(
            groq_api_key=require_groq_api_key(),
            model_name=model,
            timeout=settings.groq_timeout,
            max_retries=retries,
        )

    if provider == "openai" and settings.openai_api_key:
        import os

        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    if provider == "anthropic" and settings.anthropic_api_key:
        import os

        os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)

    from langchain.chat_models import init_chat_model

    print(f"[INFO] LLM initialize: {provider}:{model}")
    return init_chat_model(model, model_provider=provider)


class FallbackChatModel:
    """Primary chat model with an optional local/cloud fallback on rate limits."""

    def __init__(self, primary: Any, fallback_factory: Callable[[], Any] | None = None):
        self.primary = primary
        self._fallback_factory = fallback_factory
        self._fallback: Any | None = None

    def _fallback_or_raise(self, exc: BaseException) -> Any:
        if not self._fallback_factory or not should_fallback(exc):
            raise exc
        print(
            f"[WARN] Primary LLM failed ({type(exc).__name__}); "
            "switching to LLM_FALLBACK_PROVIDER."
        )
        if self._fallback is None:
            self._fallback = self._fallback_factory()
        return self._fallback

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        try:
            return self.primary.invoke(messages, **kwargs)
        except Exception as exc:
            return self._fallback_or_raise(exc).invoke(messages, **kwargs)

    def generate(self, messages: Any, **kwargs: Any) -> Any:
        try:
            return self.primary.generate(messages, **kwargs)
        except Exception as exc:
            return self._fallback_or_raise(exc).generate(messages, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.primary, name)


def build_chat_model() -> FallbackChatModel:
    primary_name = settings.llm_provider
    primary = build_backend(
        primary_name,
        _model_for(primary_name),
        max_retries=0 if settings.has_llm_fallback and primary_name == "groq" else None,
    )
    factory = None
    if settings.has_llm_fallback:
        name = settings.llm_fallback_provider
        factory = lambda n=name: build_backend(n, _model_for(n))
    return FallbackChatModel(primary, factory)
