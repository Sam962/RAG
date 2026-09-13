import pytest

from src import config
from src.config import Settings, require_groq_api_key, settings


def _clear_llm_env(monkeypatch):
    for name in (
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_FALLBACK_PROVIDER",
        "LLM_FALLBACK_MODEL",
        "GROQ_MODEL",
        "OLLAMA_MODEL",
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def _isolated_settings(tmp_path, monkeypatch, key: str | None):
    _clear_llm_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text("" if key is None else f"GROQ_API_KEY={key}\n")
    return Settings(_env_file=env_file)


def test_settings_import_without_groq_key():
    assert settings.embedding_model == "all-MiniLM-L6-v2"
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50
    assert settings.persist_dir == "faiss_store"


def test_placeholder_key_is_treated_as_missing(tmp_path, monkeypatch):
    isolated = _isolated_settings(tmp_path, monkeypatch, "gsk_your_key_here")
    assert isolated.groq_api_key is None


def test_missing_key_loads_other_settings(tmp_path, monkeypatch):
    isolated = _isolated_settings(tmp_path, monkeypatch, None)
    assert isolated.groq_api_key is None
    assert isolated.reranker_model.startswith("cross-encoder/")


def test_require_groq_api_key_rejects_missing(tmp_path, monkeypatch):
    isolated = _isolated_settings(tmp_path, monkeypatch, None)
    monkeypatch.setattr(config, "settings", isolated)
    with pytest.raises(ValueError, match="GROQ_API_KEY is required"):
        require_groq_api_key()


def test_require_groq_api_key_accepts_real_key(tmp_path, monkeypatch):
    isolated = _isolated_settings(tmp_path, monkeypatch, "gsk_test_not_placeholder")
    monkeypatch.setattr(config, "settings", isolated)
    assert require_groq_api_key() == "gsk_test_not_placeholder"


def test_llm_defaults_are_groq_with_ollama_fallback(tmp_path, monkeypatch):
    isolated = _isolated_settings(tmp_path, monkeypatch, None)
    assert isolated.llm_provider == "groq"
    assert isolated.llm_model == "openai/gpt-oss-120b"
    assert isolated.llm_fallback_provider == "ollama"
    assert isolated.llm_fallback_model == "qwen2.5:7b"
    assert isolated.has_llm_fallback is True


def test_llm_env_overrides_without_code_changes(tmp_path, monkeypatch):
    _clear_llm_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_PROVIDER=openai\n"
        "LLM_MODEL=gpt-4o-mini\n"
        "LLM_FALLBACK_PROVIDER=ollama\n"
        "LLM_FALLBACK_MODEL=llama3.2\n"
    )
    isolated = Settings(_env_file=env_file)
    assert isolated.llm_provider == "openai"
    assert isolated.llm_model == "gpt-4o-mini"
    assert isolated.llm_fallback_model == "llama3.2"


def test_ollama_provider_fills_default_model(tmp_path, monkeypatch):
    _clear_llm_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_PROVIDER=ollama\nLLM_FALLBACK_PROVIDER=\n")
    isolated = Settings(_env_file=env_file)
    assert isolated.llm_provider == "ollama"
    assert isolated.llm_model == "qwen2.5:7b"
    assert isolated.has_llm_fallback is False
