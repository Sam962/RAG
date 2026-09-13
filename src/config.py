from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import os


_PLACEHOLDER_GROQ_KEY = "gsk_your_key_here"
_DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "ollama": "qwen2.5:7b",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Chat LLM — override in .env. graph/search only call build_chat_model().
    llm_provider: str = Field(default="groq", validation_alias="LLM_PROVIDER")
    llm_model: str = Field(default="", validation_alias="LLM_MODEL")
    llm_fallback_provider: str = Field(
        default="ollama",
        validation_alias="LLM_FALLBACK_PROVIDER",
    )
    llm_fallback_model: str = Field(
        default="qwen2.5:7b",
        validation_alias="LLM_FALLBACK_MODEL",
    )
    groq_model: str = Field(default="", validation_alias="GROQ_MODEL")
    ollama_model: str = Field(default="", validation_alias="OLLAMA_MODEL")
    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434",
        validation_alias="OLLAMA_BASE_URL",
    )
    ollama_timeout: float = 120.0
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias="ANTHROPIC_API_KEY",
    )
    top_k: int = 5
    min_rerank_score: float = 0.0
    rerank_score_margin: float = 3.0
    chunk_size: int = 500
    chunk_overlap: int = 50
    persist_dir: str = "faiss_store"
    groq_timeout: float = 30.0
    groq_max_retries: int = 2
    rewrite_history_messages: int = 10
    query_identity: str = Field(default="", validation_alias="QUERY_IDENTITY")

    groq_api_key: str | None = Field(default=None, validation_alias="GROQ_API_KEY")
    langsmith_tracing: bool = Field(default=False, validation_alias="LANGSMITH_TRACING")
    langsmith_api_key: str | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="rag-pro", validation_alias="LANGSMITH_PROJECT")

    @model_validator(mode="after")
    def normalize_settings(self) -> "Settings":
        key = (self.groq_api_key or "").strip()
        self.groq_api_key = None if (not key or key == _PLACEHOLDER_GROQ_KEY) else key
        self.llm_provider = (self.llm_provider or "groq").strip().lower()
        self.llm_fallback_provider = (self.llm_fallback_provider or "").strip().lower()
        self.llm_model = (self.llm_model or "").strip()
        self.llm_fallback_model = (self.llm_fallback_model or "").strip()
        groq_model = (self.groq_model or "").strip()
        ollama_model = (self.ollama_model or "").strip()
        if not self.llm_model:
            if self.llm_provider == "groq" and groq_model:
                self.llm_model = groq_model
            elif self.llm_provider == "ollama" and ollama_model:
                self.llm_model = ollama_model
            else:
                self.llm_model = _DEFAULT_MODELS.get(self.llm_provider, "")
        if not self.llm_fallback_model and self.llm_fallback_provider == "ollama":
            self.llm_fallback_model = ollama_model or "qwen2.5:7b"
        if self.llm_fallback_provider == self.llm_provider:
            self.llm_fallback_provider = ""
            self.llm_fallback_model = ""
        return self

    @property
    def has_llm_fallback(self) -> bool:
        return bool(self.llm_fallback_provider and self.llm_fallback_model)


settings = Settings()


def require_groq_api_key() -> str:
    """Raise a clear error when a real Groq key is needed (LLM calls), not on import."""
    key = settings.groq_api_key
    if not key:
        raise ValueError(
            "GROQ_API_KEY is required. Copy .env.example to .env and set your Groq API key."
        )
    return key


def configure_langsmith() -> None:
    """Sync LangSmith settings from config into os.environ for LangChain auto-tracing."""
    if settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "true"
        if settings.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
