from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


@dataclass(frozen=True)
class EmbeddingSettings:
    backend: str
    model: str
    base_url: str | None
    api_key: str


@dataclass(frozen=True)
class LLMSettings:
    model: str | None
    base_url: str
    api_key: str
    temperature: float


def embedding_settings() -> EmbeddingSettings:
    return EmbeddingSettings(
        backend=os.environ.get("GRAPHSUM_EMBEDDING_BACKEND", "sentence_transformers"),
        model=os.environ.get("GRAPHSUM_EMBEDDING_MODEL", "BAAI/bge-m3"),
        base_url=os.environ.get("GRAPHSUM_EMBEDDING_BASE_URL"),
        api_key=os.environ.get("GRAPHSUM_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY") or "ollama",
    )


def llm_settings() -> LLMSettings:
    return LLMSettings(
        model=os.environ.get("GRAPHSUM_LLM_MODEL"),
        base_url=(os.environ.get("GRAPHSUM_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "http://localhost:8000/v1"),
        api_key=os.environ.get("GRAPHSUM_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "ollama",
        temperature=float(os.environ.get("GRAPHSUM_LLM_TEMPERATURE", "0.0")),
    )
