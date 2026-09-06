from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import os 


_PLACEHOLDER_GROQ_KEY = 'gsk_your_key_here'

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    embedding_model: str = "all-MiniLM-L6-v2" 
    reranker_model :str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    groq_model : str = "llama-3.3-70b-versatile"
    top_k: int = 5
    chunk_size : int = 1000
    chunk_overlap: int = 200
    persist_dir : str = "faiss_store"

    groq_api_key: str | None = Field(default=None, validation_alias="GROQ_API_KEY")
    langsmith_tracing: bool = Field(default=False, validation_alias="LANGSMITH_TRACING")
    langsmith_api_key: str | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="rag-pro", validation_alias="LANGSMITH_PROJECT")

    @model_validator(mode="after")
    def require_groq_api_key(self) -> "Settings":
        key = (self.groq_api_key or "").strip()
        if not key or key == _PLACEHOLDER_GROQ_KEY:
            raise ValueError(
                "GROQ_API_KEY is required. Copy .env.example to .env and set your Groq API key."
            )
        self.groq_api_key = key
        return self

settings = Settings()

def configure_langsmith() -> None:
    """Sync Lnagsmith settings from config into os.environ for LangChain aito-tracing"""
    if settings.langsmith_tracing:
        os.environ['LANGSMITH_TRACING'] = 'true'
        if settings.langsmith_api_key:
            os.environ['LANGSMITH_API_KEY'] = settings.langsmith_api_key
        os.environ['LANGSMITH_PROJECT'] = settings.langsmith_project
    else:
        os.environ['LANGSMITH_TRACING'] = 'false'


