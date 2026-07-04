from __future__ import annotations

from dataclasses import dataclass

from langchain_core.embeddings import Embeddings

from .data import Document
from .preprocess import Chunk, Sentence, TriSentenceUnit, chunk_units, count_tokens


@dataclass(frozen=True)
class ChunkCandidate:
    text: str
    sentence_indices: list[int]


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
    semantic_breakpoint_percentile: float = 65.0,
    semantic_min_chunk_tokens: int | None = None,
) -> list[Chunk]:
    if method == "simple":
        return chunk_units(document.doc_id, units, target_min_tokens, target_max_tokens)
    if method != "semantic":
        raise ValueError(f"Unknown chunking method: {method}")
    return _semantic_chunks(
        document,
        sentences,
        units,
        embedder,
        target_min_tokens,
        target_max_tokens,
        semantic_breakpoint_percentile,
        semantic_min_chunk_tokens,
    )


def _semantic_chunks(
    document: Document,
    sentences: list[Sentence],
    units: list[TriSentenceUnit],
    embedder,
    target_min_tokens: int,
    target_max_tokens: int,
    semantic_breakpoint_percentile: float,
    semantic_min_chunk_tokens: int | None,
) -> list[Chunk]:
    from langchain_experimental.text_splitter import SemanticChunker

    min_chunk_size = semantic_min_chunk_tokens or max(100, target_min_tokens // 4)
    splitter = SemanticChunker(
        LangChainEmbedder(embedder),
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=semantic_breakpoint_percentile,
        sentence_split_regex=r"(?<=[.!?。！？])\s+",
        min_chunk_size=min_chunk_size,
    )
    candidates = _paragraph_candidates(sentences, splitter, target_min_tokens, target_max_tokens)
    candidates = _merge_small_candidates(candidates, target_min_tokens, target_max_tokens)
    if not candidates:
        return chunk_units(document.doc_id, units, target_min_tokens, target_max_tokens)
    sentence_to_unit_ids = _sentence_center_to_units(units)
    chunks: list[Chunk] = []
    used_units: set[str] = set()
    for idx, candidate in enumerate(candidates):
        unit_ids = []
        for sent_idx in candidate.sentence_indices:
            unit_ids.extend(sentence_to_unit_ids.get(sent_idx, []))
        unit_ids = [unit_id for unit_id in dict.fromkeys(unit_ids) if unit_id not in used_units]
        used_units.update(unit_ids)
        if not unit_ids:
            unit_ids = _nearest_unassigned_units(units, used_units, idx, len(candidates))
            used_units.update(unit_ids)
        chunks.append(Chunk(f"{document.doc_id}:c{idx}", document.doc_id, candidate.text, unit_ids, idx))
    remaining = [unit.unit_id for unit in units if unit.unit_id not in used_units]
    if remaining and chunks:
        chunks[-1].unit_ids.extend(remaining)
    return chunks


def _paragraph_candidates(sentences: list[Sentence], splitter, target_min_tokens: int, target_max_tokens: int) -> list[ChunkCandidate]:
    candidates: list[ChunkCandidate] = []
    for paragraph_sentences in _paragraph_sentence_groups(sentences):
        paragraph_text = " ".join(sentence.text for sentence in paragraph_sentences).strip()
        if not paragraph_text:
            continue
        if count_tokens(paragraph_text) <= target_min_tokens:
            candidates.append(ChunkCandidate(paragraph_text, [sentence.index for sentence in paragraph_sentences]))
            continue
        pieces = [text.strip() for text in splitter.split_text(paragraph_text) if text.strip()]
        if len(pieces) <= 1:
            candidates.extend(_split_sentence_group(paragraph_sentences, target_min_tokens, target_max_tokens))
            continue
        for piece in pieces:
            indices = _sentences_present(piece, paragraph_sentences)
            if indices:
                index_set = set(indices)
                candidates.extend(
                    _split_sentence_group(
                        [sentence for sentence in paragraph_sentences if sentence.index in index_set],
                        target_min_tokens,
                        target_max_tokens,
                    )
                )
            else:
                candidates.append(ChunkCandidate(piece, []))
    return candidates


def _paragraph_sentence_groups(sentences: list[Sentence]) -> list[list[Sentence]]:
    groups: list[list[Sentence]] = []
    current: list[Sentence] = []
    current_paragraph_id = ""
    for sentence in sentences:
        if current and sentence.paragraph_id != current_paragraph_id:
            groups.append(current)
            current = []
        current.append(sentence)
        current_paragraph_id = sentence.paragraph_id
    if current:
        groups.append(current)
    return groups


def _split_sentence_group(sentences: list[Sentence], target_min_tokens: int, target_max_tokens: int) -> list[ChunkCandidate]:
    candidates: list[ChunkCandidate] = []
    current: list[Sentence] = []
    current_tokens = 0
    for sentence in sentences:
        tokens = count_tokens(sentence.text)
        if current and current_tokens + tokens > target_max_tokens:
            candidates.append(_candidate_from_sentences(current))
            current = []
            current_tokens = 0
        current.append(sentence)
        current_tokens += tokens
        if current_tokens >= target_min_tokens:
            candidates.append(_candidate_from_sentences(current))
            current = []
            current_tokens = 0
    if current:
        candidates.append(_candidate_from_sentences(current))
    return candidates


def _candidate_from_sentences(sentences: list[Sentence]) -> ChunkCandidate:
    return ChunkCandidate(" ".join(sentence.text for sentence in sentences), [sentence.index for sentence in sentences])


def _merge_small_candidates(candidates: list[ChunkCandidate], target_min_tokens: int, target_max_tokens: int) -> list[ChunkCandidate]:
    merged: list[ChunkCandidate] = []
    current: list[ChunkCandidate] = []
    current_tokens = 0
    for candidate in candidates:
        tokens = count_tokens(candidate.text)
        if current and current_tokens + tokens > target_max_tokens:
            merged.append(_merge_candidates(current))
            current = []
            current_tokens = 0
        current.append(candidate)
        current_tokens += tokens
        if current_tokens >= target_min_tokens:
            merged.append(_merge_candidates(current))
            current = []
            current_tokens = 0
    if current:
        current_candidate = _merge_candidates(current)
        if (
            merged
            and count_tokens(current_candidate.text) < target_min_tokens // 2
            and count_tokens(merged[-1].text) + count_tokens(current_candidate.text) <= target_max_tokens
        ):
            merged[-1] = _merge_candidates([merged[-1], current_candidate])
        else:
            merged.append(current_candidate)
    return merged


def _merge_candidates(candidates: list[ChunkCandidate]) -> ChunkCandidate:
    sentence_indices = []
    for candidate in candidates:
        sentence_indices.extend(candidate.sentence_indices)
    return ChunkCandidate(
        "\n\n".join(candidate.text for candidate in candidates if candidate.text.strip()),
        list(dict.fromkeys(sentence_indices)),
    )


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
