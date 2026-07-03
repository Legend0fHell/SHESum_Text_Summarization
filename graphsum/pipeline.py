from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .config import embedding_settings
from .data import Sample
from .entities import canonicalize_phrase_maps, extract_phrase_map
from .graph import GraphWeights, build_weighted_edges, cluster_chunks
from .llm import BaseLLM, LLMResult, count_tokens
from .chunking import make_chunks
from .preprocess import Chunk, TriSentenceUnit, make_sentences, make_tri_units
from .prompts import render_merge_prompt, render_topic_prompt
from .salience import compact_unit_texts, select_centroid_topk, select_pacsum


@dataclass
class PipelineConfig:
    salience_method: str = "e1"
    graph_weights: GraphWeights = GraphWeights(0.1, 0.2, 0.7)
    k_neighbors: int = 8
    evidence_max_tokens: int = 400
    target_min_tokens: int = 1000
    target_max_tokens: int = 1500
    use_graph: bool = True
    chunking_method: str = "semantic"
    pacsum_beta: float = 0.0
    pacsum_lambda1: float = 0.0
    pacsum_lambda2: float = 1.0
    entity_merge_threshold: float = 0.85


@dataclass
class PipelineOutput:
    summary: str
    input_tokens: int
    output_tokens: int
    llm_calls: int
    chunk_count: int
    topic_count: int


class Embedder:
    def __init__(
        self,
        model_name: str | None = None,
        dry_run: bool = False,
        backend: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        settings = embedding_settings()
        self.backend = backend or settings.backend
        self.model_name = model_name or settings.model
        self.base_url = base_url if base_url is not None else settings.base_url
        self.api_key = api_key or settings.api_key
        self.model = None
        self.openai_embeddings = None
        if dry_run:
            self.backend = "dry_run"
        elif self.backend == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
        elif self.backend == "openai_compatible":
            if not self.base_url:
                raise ValueError("GRAPHSUM_EMBEDDING_BASE_URL or --embedding-base-url is required for openai_compatible embeddings")
            from langchain_openai import OpenAIEmbeddings

            self.openai_embeddings = OpenAIEmbeddings(
                api_key=self.api_key,
                model=self.model_name,
                base_url=self.base_url,
                check_embedding_ctx_length=False,
                tiktoken_enabled=False,
            )
        else:
            raise ValueError(f"Unknown embedding backend: {self.backend}")

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self.model is not None:
            return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
        if self.openai_embeddings is not None:
            return [_normalize(np.asarray(vector, dtype=float)).tolist() for vector in self.openai_embeddings.embed_documents(texts)]
        return [_hash_embedding(text) for text in texts]


def run_sample(sample: Sample, config: PipelineConfig, embedder: Embedder, llm: BaseLLM) -> PipelineOutput:
    units: list[TriSentenceUnit] = []
    chunks: list[Chunk] = []
    for document in sample.documents:
        sentences = make_sentences(document)
        doc_units = make_tri_units(sentences)
        units.extend(doc_units)
        chunks.extend(
            make_chunks(
                document,
                sentences,
                doc_units,
                embedder,
                method=config.chunking_method,
                target_min_tokens=config.target_min_tokens,
                target_max_tokens=config.target_max_tokens,
            )
        )
    unit_lookup = {unit.unit_id: unit for unit in units}
    _attach_embeddings(units, chunks, embedder)
    _attach_phrases(chunks, sample.language, embedder, config.entity_merge_threshold)
    support_by_chunk = _select_support(units, chunks, unit_lookup, config)
    topics = _make_topics(chunks, config)
    topic_results: list[LLMResult] = []
    topic_support_ids: list[list[str]] = []
    for topic in topics:
        topic_chunks = [chunks[idx] for idx in topic]
        support_ids = []
        for chunk in topic_chunks:
            support_ids.extend(support_by_chunk.get(chunk.chunk_id, []))
        topic_support_ids.append(support_ids)
        prompt = _topic_prompt(topic_chunks, compact_unit_texts(support_ids, unit_lookup), sample.language)
        topic_results.append(llm.summarize(prompt))
    if len(topic_results) == 1:
        final = topic_results[0]
    else:
        merged = "\n".join(result.text for result in topic_results)
        support_pool_ids = _select_higher_level_support(topic_support_ids, unit_lookup, merged, embedder, config)
        support_pool = compact_unit_texts(support_pool_ids, unit_lookup, max_tokens=config.evidence_max_tokens)
        final = llm.summarize(_merge_prompt(merged, support_pool, sample.language))
        final.input_tokens += sum(result.input_tokens for result in topic_results)
        final.output_tokens += sum(result.output_tokens for result in topic_results)
        final.calls += sum(result.calls for result in topic_results)
    return PipelineOutput(final.text, final.input_tokens, final.output_tokens, final.calls, len(chunks), len(topics))


def _attach_embeddings(units: list[TriSentenceUnit], chunks: list[Chunk], embedder: Embedder) -> None:
    vectors = embedder.encode([unit.text for unit in units] + [chunk.text for chunk in chunks])
    for unit, vector in zip(units, vectors[: len(units)]):
        unit.embedding = vector
    for chunk, vector in zip(chunks, vectors[len(units) :]):
        chunk.embedding = vector


def _attach_phrases(chunks: list[Chunk], language: str, embedder: Embedder, merge_threshold: float) -> None:
    raw_phrase_maps = [extract_phrase_map(chunk.text, language, chunk.chunk_id) for chunk in chunks]
    canonical_maps, aliases = canonicalize_phrase_maps(raw_phrase_maps, embedder, merge_threshold=merge_threshold)
    for chunk, phrase_types in zip(chunks, canonical_maps):
        chunk.phrase_types = phrase_types
        chunk.phrases = set(phrase_types)
        chunk.phrase_aliases = {phrase: aliases.get(phrase, {phrase}) for phrase in chunk.phrases}


def _select_support(
    units: list[TriSentenceUnit],
    chunks: list[Chunk],
    unit_lookup: dict[str, TriSentenceUnit],
    config: PipelineConfig,
) -> dict[str, list[str]]:
    support = {}
    for chunk in chunks:
        chunk_units = [unit_lookup[unit_id] for unit_id in chunk.unit_ids]
        if config.salience_method == "e1":
            selected = select_centroid_topk(chunk_units, chunk.embedding or [], config.evidence_max_tokens)
        elif config.salience_method == "e2a":
            selected = select_pacsum(
                chunk_units,
                config.evidence_max_tokens,
                stride=1,
                beta=config.pacsum_beta,
                lambda1=config.pacsum_lambda1,
                lambda2=config.pacsum_lambda2,
            )
        elif config.salience_method == "e2b":
            selected = select_pacsum(
                chunk_units,
                config.evidence_max_tokens,
                stride=2,
                beta=config.pacsum_beta,
                lambda1=config.pacsum_lambda1,
                lambda2=config.pacsum_lambda2,
            )
        else:
            raise ValueError(f"Unknown salience method: {config.salience_method}")
        support[chunk.chunk_id] = selected
    return support


def _make_topics(chunks: list[Chunk], config: PipelineConfig) -> list[list[int]]:
    if not config.use_graph:
        return [[idx] for idx in range(len(chunks))]
    edges = build_weighted_edges(chunks, config.graph_weights, k=config.k_neighbors)
    return cluster_chunks(chunks, edges)


def _select_higher_level_support(
    topic_support_ids: list[list[str]],
    unit_lookup: dict[str, TriSentenceUnit],
    merged_topic_summaries: str,
    embedder: Embedder,
    config: PipelineConfig,
) -> list[str]:
    support_ids = list(dict.fromkeys(unit_id for topic_ids in topic_support_ids for unit_id in topic_ids))
    support_units = [unit_lookup[unit_id] for unit_id in support_ids if unit_id in unit_lookup]
    if not support_units:
        return []
    if config.salience_method == "e1":
        merged_embedding = embedder.encode([merged_topic_summaries])[0]
        return select_centroid_topk(support_units, merged_embedding, config.evidence_max_tokens)
    if config.salience_method == "e2a":
        return select_pacsum(
            support_units,
            config.evidence_max_tokens,
            stride=1,
            beta=config.pacsum_beta,
            lambda1=config.pacsum_lambda1,
            lambda2=config.pacsum_lambda2,
        )
    if config.salience_method == "e2b":
        return select_pacsum(
            support_units,
            config.evidence_max_tokens,
            stride=2,
            beta=config.pacsum_beta,
            lambda1=config.pacsum_lambda1,
            lambda2=config.pacsum_lambda2,
        )
    raise ValueError(f"Unknown salience method: {config.salience_method}")


def _topic_prompt(chunks: list[Chunk], support: str, language: str) -> str:
    chunk_text = "\n\n".join(chunk.text for chunk in sorted(chunks, key=lambda c: (c.document_id, c.position)))
    return render_topic_prompt(chunk_text, support, language)


def _merge_prompt(summaries: str, support: str, language: str) -> str:
    return render_merge_prompt(summaries, support, language)


def _hash_embedding(text: str, dims: int = 384) -> list[float]:
    vector = np.zeros(dims, dtype=float)
    for token in text.lower().split():
        digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % dims
        sign = 1 if digest[4] % 2 else -1
        vector[idx] += sign
    norm = np.linalg.norm(vector)
    return (vector / norm if norm else vector).tolist()


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector
