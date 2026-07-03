from __future__ import annotations

from langchain_core.embeddings import Embeddings

from .data import Document
from .preprocess import Chunk, Sentence, TriSentenceUnit, chunk_units, count_tokens


class LangChainEmbedder(Embeddings):
    def __init__(self, embedder):
        self.embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.encode([text])[0]


def make_chunks(
    document: Document,
    sentences: list[Sentence],
    units: list[TriSentenceUnit],
    embedder,
    method: str = "semantic",
    target_min_tokens: int = 1000,
    target_max_tokens: int = 1500,
) -> list[Chunk]:
    if method == "simple":
        return chunk_units(document.doc_id, units, target_min_tokens, target_max_tokens)
    if method != "semantic":
        raise ValueError(f"Unknown chunking method: {method}")
    return _semantic_chunks(document, sentences, units, embedder, target_min_tokens, target_max_tokens)


def _semantic_chunks(
    document: Document,
    sentences: list[Sentence],
    units: list[TriSentenceUnit],
    embedder,
    target_min_tokens: int,
    target_max_tokens: int,
) -> list[Chunk]:
    from langchain_experimental.text_splitter import SemanticChunker

    splitter = SemanticChunker(
        LangChainEmbedder(embedder),
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,
        sentence_split_regex=r"(?<=[.!?。！？])\s+",
        min_chunk_size=max(100, target_min_tokens // 4),
    )
    raw_chunks = [text.strip() for text in splitter.split_text(document.text) if text.strip()]
    raw_chunks = _merge_small_chunks(raw_chunks, target_min_tokens, target_max_tokens)
    if not raw_chunks:
        return chunk_units(document.doc_id, units, target_min_tokens, target_max_tokens)
    sentence_to_unit_ids = _sentence_center_to_units(units)
    chunks: list[Chunk] = []
    used_units: set[str] = set()
    for idx, text in enumerate(raw_chunks):
        sentence_indices = _sentences_present(text, sentences)
        unit_ids = []
        for sent_idx in sentence_indices:
            unit_ids.extend(sentence_to_unit_ids.get(sent_idx, []))
        unit_ids = [unit_id for unit_id in dict.fromkeys(unit_ids) if unit_id not in used_units]
        used_units.update(unit_ids)
        if not unit_ids:
            unit_ids = _nearest_unassigned_units(units, used_units, idx, len(raw_chunks))
            used_units.update(unit_ids)
        chunks.append(Chunk(f"{document.doc_id}:c{idx}", document.doc_id, text, unit_ids, idx))
    remaining = [unit.unit_id for unit in units if unit.unit_id not in used_units]
    if remaining and chunks:
        chunks[-1].unit_ids.extend(remaining)
    return chunks


def _merge_small_chunks(chunks: list[str], target_min_tokens: int, target_max_tokens: int) -> list[str]:
    merged: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for chunk in chunks:
        tokens = count_tokens(chunk)
        if current and current_tokens + tokens > target_max_tokens:
            merged.append("\n".join(current))
            current = []
            current_tokens = 0
        current.append(chunk)
        current_tokens += tokens
        if current_tokens >= target_min_tokens:
            merged.append("\n".join(current))
            current = []
            current_tokens = 0
    if current:
        if merged and count_tokens("\n".join(current)) < target_min_tokens // 2:
            merged[-1] = merged[-1] + "\n" + "\n".join(current)
        else:
            merged.append("\n".join(current))
    return merged


def _sentence_center_to_units(units: list[TriSentenceUnit]) -> dict[int, list[str]]:
    output: dict[int, list[str]] = {}
    for unit in units:
        output.setdefault(unit.center_index, []).append(unit.unit_id)
    return output


def _sentences_present(chunk_text: str, sentences: list[Sentence]) -> list[int]:
    normalized = " ".join(chunk_text.split())
    indices = []
    for sentence in sentences:
        text = " ".join(sentence.text.split())
        if text and text in normalized:
            indices.append(sentence.index)
    return indices


def _nearest_unassigned_units(
    units: list[TriSentenceUnit],
    used_units: set[str],
    chunk_index: int,
    chunk_count: int,
) -> list[str]:
    unassigned = [unit for unit in units if unit.unit_id not in used_units]
    if not unassigned:
        return []
    start = int(len(unassigned) * chunk_index / max(1, chunk_count))
    end = int(len(unassigned) * (chunk_index + 1) / max(1, chunk_count))
    return [unit.unit_id for unit in unassigned[start:end] or unassigned[:1]]
