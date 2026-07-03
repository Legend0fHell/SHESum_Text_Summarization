from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .data import Document


@dataclass
class Sentence:
    sentence_id: str
    document_id: str
    paragraph_id: str
    text: str
    index: int


@dataclass
class TriSentenceUnit:
    unit_id: str
    document_id: str
    sentence_ids: list[str]
    sentence_texts: list[str]
    text: str
    center_index: int
    embedding: list[float] | None = None
    phrases: set[str] = field(default_factory=set)
    phrase_types: dict[str, set[str]] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    unit_ids: list[str]
    position: int
    embedding: list[float] | None = None
    phrases: set[str] = field(default_factory=set)
    phrase_types: dict[str, set[str]] = field(default_factory=dict)
    phrase_aliases: dict[str, set[str]] = field(default_factory=dict)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    text = normalize_text(text)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    return paragraphs or [text]


def split_sentences(text: str) -> list[str]:
    text = normalize_text(text)
    pattern = r"(?<=[.!?。！？])\s+|(?<=[.!?。！？])(?=[A-ZÀ-Ỵ])"
    sentences = [part.strip() for part in re.split(pattern, text) if part.strip()]
    return sentences or ([text] if text else [])


def make_sentences(document: Document) -> list[Sentence]:
    output = []
    index = 0
    for para_idx, paragraph in enumerate(split_paragraphs(document.text)):
        paragraph_id = f"{document.doc_id}:p{para_idx}"
        for sentence in split_sentences(paragraph):
            sentence_id = f"{document.doc_id}:s{index}"
            output.append(Sentence(sentence_id, document.doc_id, paragraph_id, sentence, index))
            index += 1
    return output


def make_tri_units(sentences: list[Sentence]) -> list[TriSentenceUnit]:
    units = []
    for idx, sentence in enumerate(sentences):
        start = max(0, idx - 1)
        end = min(len(sentences), idx + 2)
        window = sentences[start:end]
        units.append(
            TriSentenceUnit(
                unit_id=f"{sentence.document_id}:u{idx}",
                document_id=sentence.document_id,
                sentence_ids=[item.sentence_id for item in window],
                sentence_texts=[item.text for item in window],
                text=" ".join(item.text for item in window),
                center_index=idx,
            )
        )
    return units


def chunk_units(
    document_id: str,
    units: list[TriSentenceUnit],
    target_min_tokens: int = 1000,
    target_max_tokens: int = 1500,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    current: list[TriSentenceUnit] = []
    current_tokens = 0
    for unit in units:
        unit_tokens = count_tokens(unit.text)
        if current and current_tokens + unit_tokens > target_max_tokens:
            chunks.append(_make_chunk(document_id, len(chunks), current))
            current = []
            current_tokens = 0
        current.append(unit)
        current_tokens += unit_tokens
        if current_tokens >= target_min_tokens:
            chunks.append(_make_chunk(document_id, len(chunks), current))
            current = []
            current_tokens = 0
    if current:
        chunks.append(_make_chunk(document_id, len(chunks), current))
    return chunks


def count_tokens(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _make_chunk(document_id: str, position: int, units: list[TriSentenceUnit]) -> Chunk:
    return Chunk(
        chunk_id=f"{document_id}:c{position}",
        document_id=document_id,
        text="\n".join(unit.text for unit in units),
        unit_ids=[unit.unit_id for unit in units],
        position=position,
    )
