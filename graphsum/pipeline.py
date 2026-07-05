from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .config import embedding_settings
from .data import Document, Sample
from .dedup import ChunkDedupResult, build_chunk_duplicate_groups
from .entities import canonicalize_phrase_maps, extract_phrase_map
from .graph import GraphWeights, build_weighted_edge_details, cluster_chunks
from .llm import BaseLLM, LLMResult, count_tokens
from .chunking import make_chunks
from .preprocess import Chunk, TriSentenceUnit, make_sentences, make_tri_units
from .prompts import render_direct_prompt, render_merge_prompt, render_topic_prompt
from .salience import compact_unit_texts, select_centroid_topk, select_pacsum


DATASET_SUMMARY_BUDGETS = {
    "vn_mds": {"max_summary_words": 180, "max_output_tokens": 300},
    "vims": {"max_summary_words": 235, "max_output_tokens": 380},
    "multi_news": {"max_summary_words": 260, "max_output_tokens": 420},
}


@dataclass
class PipelineConfig:
    salience_method: str = "e2b"
    graph_weights: GraphWeights = GraphWeights(0.1, 0.2, 0.7)
    k_neighbors: int = 8
    evidence_max_tokens: int = 400
    target_min_tokens: int = 1000
    target_max_tokens: int = 1500
    semantic_breakpoint_percentile: float = 65.0
    semantic_min_chunk_tokens: int | None = None
    use_graph: bool = True
    chunking_method: str = "semantic"
    pacsum_beta: float = 0.0
    pacsum_lambda1: float = 0.0
    pacsum_lambda2: float = 1.0
    entity_merge_threshold: float = 0.85
    dedup_chunks: bool = True
    dedup_sim_threshold: float = 0.88
    dedup_require_shared_phrase: bool = False
    duplicate_edge_factor: float = 0.35
    community_dedup: bool = True
    max_summary_words: int | None = None
    max_output_tokens: int | None = None


@dataclass
class PipelineOutput:
    summary: str
    input_tokens: int
    output_tokens: int
    llm_calls: int
    chunk_count: int
    community_count: int
    trace: "PipelineTrace | None" = None


@dataclass
class PipelineTrace:
    progress: list[dict[str, str]] = field(default_factory=list)
    splitter_outputs: list[dict[str, object]] = field(default_factory=list)
    segments: list[dict[str, object]] = field(default_factory=list)
    chunks: list[dict[str, object]] = field(default_factory=list)
    chunk_entities: list[dict[str, object]] = field(default_factory=list)
    duplicate_groups: list[dict[str, object]] = field(default_factory=list)
    community_dedup: list[dict[str, object]] = field(default_factory=list)
    support: list[dict[str, object]] = field(default_factory=list)
    graph_edges: list[dict[str, object]] = field(default_factory=list)
    communities: list[dict[str, object]] = field(default_factory=list)
    community_summaries: list[dict[str, object]] = field(default_factory=list)
    summary_steps: list[dict[str, object]] = field(default_factory=list)
    summary_graph_nodes: list[dict[str, object]] = field(default_factory=list)
    summary_graph_edges: list[dict[str, object]] = field(default_factory=list)
    final_summary: str = ""


ProgressCallback = Callable[[str, str], None]


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


def run_sample(
    sample: Sample,
    config: PipelineConfig,
    embedder: Embedder,
    llm: BaseLLM,
    trace: PipelineTrace | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PipelineOutput:
    summary_budget = resolve_summary_budget(sample.dataset, config.max_summary_words, config.max_output_tokens)
    _record_progress(trace, progress_callback, "preprocess", "Splitting documents and building segment windows")
    units: list[TriSentenceUnit] = []
    chunks: list[Chunk] = []
    for document in sample.documents:
        sentences = make_sentences(document)
        doc_units = make_tri_units(sentences)
        units.extend(doc_units)
        _trace_splitter_outputs(trace, sentences)
        _trace_segments(trace, doc_units)
        chunks.extend(
            make_chunks(
                document,
                sentences,
                doc_units,
                embedder,
                method=config.chunking_method,
                target_min_tokens=config.target_min_tokens,
                target_max_tokens=config.target_max_tokens,
                semantic_breakpoint_percentile=config.semantic_breakpoint_percentile,
                semantic_min_chunk_tokens=config.semantic_min_chunk_tokens,
            )
        )
    unit_lookup = {unit.unit_id: unit for unit in units}

    _record_progress(trace, progress_callback, "embed", "Embedding segments and semantic chunks")
    _attach_embeddings(units, chunks, embedder)

    _record_progress(trace, progress_callback, "entities", "Extracting and consolidating chunk entities and factual phrases")
    _attach_phrases(chunks, sample.language, embedder, config.entity_merge_threshold)
    dedup_result = _deduplicate_chunks(chunks, config)
    _trace_duplicate_groups(trace, chunks, dedup_result)

    _record_progress(trace, progress_callback, "support", "Selecting source support segments")
    support_by_chunk = _select_support(units, chunks, unit_lookup, config)
    _trace_chunks(trace, chunks)
    _trace_support(trace, support_by_chunk, unit_lookup)

    _record_progress(trace, progress_callback, "graph", "Building chunk graph and Leiden chunk communities")
    communities, graph_edges = _build_communities(chunks, config, dedup_result)
    _trace_graph(trace, chunks, graph_edges)
    _trace_communities(trace, chunks, communities)

    _record_progress(trace, progress_callback, "summarize", "Generating community-level summaries")
    community_results: list[LLMResult] = []
    community_support_ids: list[list[str]] = []
    for community in communities:
        community_chunks = [chunks[idx] for idx in community]
        prompt_chunks = _deduplicate_community_chunks(community_chunks, chunks, dedup_result, config, trace, len(community_results))
        support_ids = []
        for chunk in prompt_chunks:
            support_ids.extend(support_by_chunk.get(chunk.chunk_id, []))
        community_support_ids.append(support_ids)
        prompt = _community_prompt(
            prompt_chunks,
            compact_unit_texts(support_ids, unit_lookup),
            sample.language,
            sample.dataset,
            summary_budget.max_summary_words,
        )
        community_results.append(llm.summarize(prompt, max_tokens=summary_budget.max_output_tokens))
        _trace_community_summary(trace, len(community_results) - 1, prompt_chunks, prompt, community_results[-1])
    if len(community_results) == 1:
        final = community_results[0]
    else:
        _record_progress(trace, progress_callback, "merge", "Merging community summaries with source-only support")
        merged = "\n".join(result.text for result in community_results)
        support_pool_ids = _select_higher_level_support(community_support_ids, unit_lookup, merged, embedder, config)
        support_pool = compact_unit_texts(support_pool_ids, unit_lookup, max_tokens=config.evidence_max_tokens)
        merge_prompt = _merge_prompt(merged, support_pool, sample.language, sample.dataset, summary_budget.max_summary_words)
        final = llm.summarize(merge_prompt, max_tokens=summary_budget.max_output_tokens)
        final.input_tokens += sum(result.input_tokens for result in community_results)
        final.output_tokens += sum(result.output_tokens for result in community_results)
        final.calls += sum(result.calls for result in community_results)
        _trace_summary_step(trace, "merge", "final_merge", merge_prompt, final)
    final.text = trim_to_word_budget(final.text, summary_budget.max_summary_words)
    _trace_summary_graph(trace, chunks, communities, final.text)
    _record_progress(trace, progress_callback, "done", "Finished summarization")
    return PipelineOutput(final.text, final.input_tokens, final.output_tokens, final.calls, len(chunks), len(communities), trace)


def run_direct_sample(
    sample: Sample,
    llm: BaseLLM,
    max_summary_words: int | None = None,
    max_output_tokens: int | None = None,
    trace: PipelineTrace | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PipelineOutput:
    summary_budget = resolve_summary_budget(sample.dataset, max_summary_words, max_output_tokens)
    _record_progress(trace, progress_callback, "direct", "Sending all source documents directly to the LLM")
    source_text = "\n\n".join(_document_block(index, document) for index, document in enumerate(sample.documents, start=1))
    prompt = render_direct_prompt(source_text, sample.language, sample.dataset, summary_budget.max_summary_words)
    result = llm.summarize(prompt, max_tokens=summary_budget.max_output_tokens)
    result.text = trim_to_word_budget(result.text, summary_budget.max_summary_words)
    if trace is not None:
        trace.final_summary = result.text
        trace.summary_graph_nodes.append({"node_id": "direct_llm", "kind": "direct_llm", "label": "Direct LLM"})
        _trace_summary_step(trace, "direct", "direct_llm", prompt, result)
    _record_progress(trace, progress_callback, "done", "Finished direct summarization")
    return PipelineOutput(result.text, result.input_tokens, result.output_tokens, result.calls, len(sample.documents), 1, trace)


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


def _make_communities(chunks: list[Chunk], config: PipelineConfig) -> list[list[int]]:
    return _build_communities(chunks, config, ChunkDedupResult({}, {}, []))[0]


def _deduplicate_chunks(chunks: list[Chunk], config: PipelineConfig) -> ChunkDedupResult:
    if not config.dedup_chunks:
        return ChunkDedupResult({}, {}, [])
    return build_chunk_duplicate_groups(
        chunks,
        threshold=config.dedup_sim_threshold,
        require_shared_phrase=config.dedup_require_shared_phrase,
    )


def _build_communities(
    chunks: list[Chunk],
    config: PipelineConfig,
    dedup_result: ChunkDedupResult,
) -> tuple[list[list[int]], list[dict[str, float | int | str | bool]]]:
    if not config.use_graph:
        return [[idx] for idx in range(len(chunks))], []
    edge_details = build_weighted_edge_details(
        chunks,
        config.graph_weights,
        k=config.k_neighbors,
        duplicate_group_by_index=dedup_result.group_for_index,
        duplicate_edge_factor=config.duplicate_edge_factor,
    )
    edges = [(int(edge["source_index"]), int(edge["target_index"]), float(edge["weight"])) for edge in edge_details]
    return cluster_chunks(chunks, edges), edge_details


def _deduplicate_community_chunks(
    community_chunks: list[Chunk],
    all_chunks: list[Chunk],
    dedup_result: ChunkDedupResult,
    config: PipelineConfig,
    trace: PipelineTrace | None,
    community_index: int,
) -> list[Chunk]:
    if not config.community_dedup or not dedup_result.groups:
        return community_chunks
    index_by_chunk_id = {chunk.chunk_id: idx for idx, chunk in enumerate(all_chunks)}
    community_indices = [index_by_chunk_id[chunk.chunk_id] for chunk in community_chunks]
    community_index_set = set(community_indices)
    kept_indices: list[int] = []
    skipped_indices: list[int] = []
    seen_groups: set[str] = set()
    for idx in community_indices:
        group_id = dedup_result.group_id(idx)
        if not group_id.startswith("dup_"):
            kept_indices.append(idx)
            continue
        if group_id in seen_groups:
            skipped_indices.append(idx)
            continue
        seen_groups.add(group_id)
        representative = dedup_result.representative_index(idx)
        kept_indices.append(representative if representative in community_index_set else idx)
        if representative in community_index_set and representative != idx:
            skipped_indices.append(idx)
    _trace_community_dedup(trace, all_chunks, dedup_result, community_index, kept_indices, skipped_indices)
    kept_chunks = [all_chunks[idx] for idx in sorted(dict.fromkeys(kept_indices), key=lambda item: (all_chunks[item].document_id, all_chunks[item].position))]
    return kept_chunks or community_chunks


def _select_higher_level_support(
    community_support_ids: list[list[str]],
    unit_lookup: dict[str, TriSentenceUnit],
    merged_community_summaries: str,
    embedder: Embedder,
    config: PipelineConfig,
) -> list[str]:
    support_ids = list(dict.fromkeys(unit_id for community_ids in community_support_ids for unit_id in community_ids))
    support_units = [unit_lookup[unit_id] for unit_id in support_ids if unit_id in unit_lookup]
    if not support_units:
        return []
    if config.salience_method == "e1":
        merged_embedding = embedder.encode([merged_community_summaries])[0]
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


def resolve_summary_budget(dataset: str, max_summary_words: int | None = None, max_output_tokens: int | None = None) -> "SummaryBudget":
    defaults = DATASET_SUMMARY_BUDGETS.get(dataset, {})
    return SummaryBudget(
        max_summary_words=max_summary_words if max_summary_words is not None else defaults.get("max_summary_words"),
        max_output_tokens=max_output_tokens if max_output_tokens is not None else defaults.get("max_output_tokens"),
    )


@dataclass(frozen=True)
class SummaryBudget:
    max_summary_words: int | None
    max_output_tokens: int | None


def trim_to_word_budget(text: str, max_words: int | None) -> str:
    normalized = " ".join((text or "").split())
    if max_words is None or max_words <= 0:
        return normalized
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?。！？])\s+", normalized) if sentence.strip()]
    selected: list[str] = []
    used_words = 0
    for sentence in sentences:
        words = sentence.split()
        if selected and used_words + len(words) > max_words:
            break
        if not selected and len(words) > max_words:
            return " ".join(words[:max_words])
        selected.append(sentence)
        used_words += len(words)
    if selected:
        return " ".join(selected)
    return " ".join(normalized.split()[:max_words])


def _community_prompt(chunks: list[Chunk], support: str, language: str, dataset: str, max_summary_words: int | None) -> str:
    chunk_text = "\n\n".join(chunk.text for chunk in sorted(chunks, key=lambda c: (c.document_id, c.position)))
    return render_topic_prompt(chunk_text, support, language, dataset, max_summary_words)


def _merge_prompt(summaries: str, support: str, language: str, dataset: str, max_summary_words: int | None) -> str:
    return render_merge_prompt(summaries, support, language, dataset, max_summary_words)


def _document_block(index: int, document) -> str:
    lines = [f"[Document {index}: {document.doc_id}]"]
    if document.title:
        lines.append(f"Title: {document.title}")
    if document.source:
        lines.append(f"Source: {document.source}")
    lines.append(document.text)
    return "\n".join(lines)


def _record_progress(trace: PipelineTrace | None, callback: ProgressCallback | None, stage: str, message: str) -> None:
    if trace is not None:
        trace.progress.append({"stage": stage, "message": message})
    if callback is not None:
        callback(stage, message)


def _trace_splitter_outputs(trace: PipelineTrace | None, sentences) -> None:
    if trace is None:
        return
    for sentence in sentences:
        trace.splitter_outputs.append(
            {
                "splitter_output_id": sentence.sentence_id,
                "document_id": sentence.document_id,
                "paragraph_id": sentence.paragraph_id,
                "index": sentence.index,
                "text": sentence.text,
            }
        )


def _trace_segments(trace: PipelineTrace | None, units: list[TriSentenceUnit]) -> None:
    if trace is None:
        return
    for unit in units:
        trace.segments.append(
            {
                "segment_id": unit.unit_id,
                "document_id": unit.document_id,
                "center_index": unit.center_index,
                "splitter_output_ids": ", ".join(unit.sentence_ids),
                "text": unit.text,
            }
        )


def _trace_chunks(trace: PipelineTrace | None, chunks: list[Chunk]) -> None:
    if trace is None:
        return
    for chunk in chunks:
        trace.chunks.append(
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "position": chunk.position,
                "segment_count": len(chunk.unit_ids),
                "text": chunk.text,
            }
        )
        for phrase, types in sorted(chunk.phrase_types.items()):
            trace.chunk_entities.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "phrase": phrase,
                    "types": ", ".join(sorted(types)),
                    "aliases": ", ".join(sorted(chunk.phrase_aliases.get(phrase, {phrase}))),
                }
            )


def _trace_duplicate_groups(trace: PipelineTrace | None, chunks: list[Chunk], dedup_result: ChunkDedupResult) -> None:
    if trace is None:
        return
    for group in dedup_result.groups:
        representative = chunks[group.representative_index]
        for member_index in group.member_indices:
            member = chunks[member_index]
            trace.duplicate_groups.append(
                {
                    "duplicate_group": group.group_id,
                    "representative_chunk_id": representative.chunk_id,
                    "chunk_id": member.chunk_id,
                    "document_id": member.document_id,
                    "position": member.position,
                    "is_representative": member_index == group.representative_index,
                    "similarity_to_representative": group.representative_similarity.get(member_index, 0.0),
                    "text": member.text,
                }
            )


def _trace_support(trace: PipelineTrace | None, support_by_chunk: dict[str, list[str]], unit_lookup: dict[str, TriSentenceUnit]) -> None:
    if trace is None:
        return
    for chunk_id, unit_ids in sorted(support_by_chunk.items()):
        for rank, unit_id in enumerate(unit_ids, start=1):
            unit = unit_lookup.get(unit_id)
            trace.support.append(
                {
                    "chunk_id": chunk_id,
                    "rank": rank,
                    "segment_id": unit_id,
                    "text": unit.text if unit is not None else "",
                }
            )


def _trace_graph(trace: PipelineTrace | None, chunks: list[Chunk], edges: list[dict[str, float | int | str | bool]]) -> None:
    if trace is None:
        return
    for edge in edges:
        left = int(edge["source_index"])
        right = int(edge["target_index"])
        trace.graph_edges.append(
            {
                "source": chunks[left].chunk_id,
                "target": chunks[right].chunk_id,
                "source_index": left,
                "target_index": right,
                "duplicate_group": edge["duplicate_group"],
                "is_duplicate_edge": edge["is_duplicate_edge"],
                "position_similarity": edge["position_similarity"],
                "entity_similarity": edge["entity_similarity"],
                "content_similarity": edge["content_similarity"],
                "position_weighted": edge["position_weighted"],
                "entity_weighted": edge["entity_weighted"],
                "content_weighted": edge["content_weighted"],
                "weight_before_redundancy": edge["weight_before_redundancy"],
                "redundancy_penalty": edge["redundancy_penalty"],
                "weight": edge["weight"],
            }
        )


def _trace_community_dedup(
    trace: PipelineTrace | None,
    chunks: list[Chunk],
    dedup_result: ChunkDedupResult,
    community_index: int,
    kept_indices: list[int],
    skipped_indices: list[int],
) -> None:
    if trace is None:
        return
    kept_set = set(kept_indices)
    skipped_set = set(skipped_indices)
    for idx in sorted(kept_set | skipped_set, key=lambda item: (chunks[item].document_id, chunks[item].position)):
        trace.community_dedup.append(
            {
                "community_id": f"community_{community_index}",
                "chunk_id": chunks[idx].chunk_id,
                "duplicate_group": dedup_result.group_id(idx),
                "representative_chunk_id": chunks[dedup_result.representative_index(idx)].chunk_id,
                "action": "kept" if idx in kept_set else "skipped_duplicate",
                "text": chunks[idx].text,
            }
        )


def _trace_communities(trace: PipelineTrace | None, chunks: list[Chunk], communities: list[list[int]]) -> None:
    if trace is None:
        return
    for community_index, community in enumerate(communities):
        for chunk_index in community:
            chunk = chunks[chunk_index]
            trace.communities.append(
                {
                    "community_id": f"community_{community_index}",
                    "community_index": community_index,
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "chunk_position": chunk.position,
                    "chunk_index": chunk_index,
                }
            )


def _trace_community_summary(trace: PipelineTrace | None, community_index: int, chunks: list[Chunk], prompt: str, result: LLMResult) -> None:
    if trace is None:
        return
    step_id = f"community_{community_index}"
    trace.community_summaries.append(
        {
            "community_id": step_id,
            "chunk_ids": ", ".join(chunk.chunk_id for chunk in chunks),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "prompt": prompt,
            "summary": result.text,
        }
    )
    _trace_summary_step(trace, "community", step_id, prompt, result)


def _trace_summary_step(trace: PipelineTrace | None, step_type: str, step_id: str, prompt: str, result: LLMResult) -> None:
    if trace is None:
        return
    trace.summary_steps.append(
        {
            "step_id": step_id,
            "step_type": step_type,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "prompt": prompt,
            "summary": result.text,
        }
    )


def _trace_summary_graph(trace: PipelineTrace | None, chunks: list[Chunk], communities: list[list[int]], final_summary: str) -> None:
    if trace is None:
        return
    trace.final_summary = final_summary
    for chunk in chunks:
        trace.summary_graph_nodes.append({"node_id": chunk.chunk_id, "kind": "chunk", "label": chunk.chunk_id})
    for community_index, community in enumerate(communities):
        community_id = f"community_{community_index}"
        trace.summary_graph_nodes.append({"node_id": community_id, "kind": "community", "label": community_id})
        for chunk_index in community:
            trace.summary_graph_edges.append({"source": chunks[chunk_index].chunk_id, "target": community_id, "kind": "chunk_to_community"})
    trace.summary_graph_nodes.append({"node_id": "final_summary", "kind": "final", "label": "final_summary"})
    for community_index in range(len(communities)):
        trace.summary_graph_edges.append({"source": f"community_{community_index}", "target": "final_summary", "kind": "community_to_final"})


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
