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


@dataclass(frozen=True)
class ExperimentSettings:
    dataset: str | None
    data_root: str
    limit: int
    salience: str
    llm: str
    dry_embed: bool
    chunking: str
    grid: bool
    alpha: float
    beta: float
    pacsum_beta: float
    pacsum_lambda1: float
    pacsum_lambda2: float
    entity_merge_threshold: float
    no_graph: bool
    pure_llm: bool
    target_min_tokens: int
    target_max_tokens: int
    semantic_breakpoint_percentile: float
    semantic_min_chunk_tokens: int | None
    dedup_chunks: bool
    dedup_sim_threshold: float
    dedup_require_shared_phrase: bool
    duplicate_edge_factor: float
    community_dedup: bool
    max_summary_words: int | None
    max_output_tokens: int | None
    output: str
    aggregate_output: str | None


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


def experiment_settings() -> ExperimentSettings:
    return ExperimentSettings(
        dataset=_empty_to_none(os.environ.get("GRAPHSUM_DATASET")),
        data_root=os.environ.get("GRAPHSUM_DATA_ROOT", "datasets"),
        limit=_int_env("GRAPHSUM_LIMIT", 3),
        salience=os.environ.get("GRAPHSUM_SALIENCE", "e2b"),
        llm=os.environ.get("GRAPHSUM_LLM_BACKEND", "dry_run"),
        dry_embed=_bool_env("GRAPHSUM_DRY_EMBED", False),
        chunking=os.environ.get("GRAPHSUM_CHUNKING", "semantic"),
        grid=_bool_env("GRAPHSUM_GRID", False),
        alpha=_float_env("GRAPHSUM_ALPHA", 0.10),
        beta=_float_env("GRAPHSUM_BETA", 0.20),
        pacsum_beta=_float_env("GRAPHSUM_PACSUM_BETA", 0.0),
        pacsum_lambda1=_float_env("GRAPHSUM_PACSUM_LAMBDA1", 0.0),
        pacsum_lambda2=_float_env("GRAPHSUM_PACSUM_LAMBDA2", 1.0),
        entity_merge_threshold=_float_env("GRAPHSUM_ENTITY_MERGE_THRESHOLD", 0.85),
        no_graph=_bool_env("GRAPHSUM_NO_GRAPH", False),
        pure_llm=_bool_env("GRAPHSUM_PURE_LLM", False),
        target_min_tokens=_int_env("GRAPHSUM_TARGET_MIN_TOKENS", 1000),
        target_max_tokens=_int_env("GRAPHSUM_TARGET_MAX_TOKENS", 1500),
        semantic_breakpoint_percentile=_float_env("GRAPHSUM_SEMANTIC_BREAKPOINT_PERCENTILE", 65.0),
        semantic_min_chunk_tokens=_optional_int_env("GRAPHSUM_SEMANTIC_MIN_CHUNK_TOKENS"),
        dedup_chunks=_bool_env("GRAPHSUM_DEDUP_CHUNKS", True),
        dedup_sim_threshold=_float_env("GRAPHSUM_DEDUP_SIM_THRESHOLD", 0.88),
        dedup_require_shared_phrase=_bool_env("GRAPHSUM_DEDUP_REQUIRE_SHARED_PHRASE", False),
        duplicate_edge_factor=_float_env("GRAPHSUM_DUPLICATE_EDGE_FACTOR", 0.35),
        community_dedup=_bool_env("GRAPHSUM_COMMUNITY_DEDUP", True),
        max_summary_words=_optional_int_env("GRAPHSUM_MAX_SUMMARY_WORDS"),
        max_output_tokens=_optional_int_env("GRAPHSUM_MAX_OUTPUT_TOKENS"),
        output=os.environ.get("GRAPHSUM_OUTPUT", "runs/graphsum_results.csv"),
        aggregate_output=_empty_to_none(os.environ.get("GRAPHSUM_AGGREGATE_OUTPUT")),
    )


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in {None, ""} else default


def _optional_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value not in {None, ""} else None


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value not in {None, ""} else default


def _empty_to_none(value: str | None) -> str | None:
    return value if value and value.strip() else None
