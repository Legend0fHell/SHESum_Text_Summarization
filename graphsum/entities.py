from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class FactualPhrase:
    text: str
    phrase_type: str
    source_id: str


NUMBER_PATTERNS = [
    ("percent", re.compile(r"\b\d+(?:[,.]\d+)?\s?%")),
    ("money", re.compile(r"(?:\$|USD|VND|đồng|tỷ|triệu)\s?\d+(?:[,.]\d+)?(?:\s?(?:billion|million|trillion|tỷ|triệu))?|\d+(?:[,.]\d+)?\s?(?:billion|million|trillion|tỷ|triệu)\s?(?:đồng|USD|VND)?", re.I)),
    ("date", re.compile(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b|\b\d{4}\b")),
]


def extract_phrases(text: str, language: str, source_id: str) -> set[str]:
    return set(extract_phrase_map(text, language, source_id))


def extract_phrase_map(text: str, language: str, source_id: str) -> dict[str, set[str]]:
    phrase_map: dict[str, set[str]] = {}
    for item in extract_factual_phrases(text, language, source_id):
        phrase = normalize_phrase(item.text)
        if phrase:
            phrase_map.setdefault(phrase, set()).add(item.phrase_type)
    return phrase_map


def extract_factual_phrases(text: str, language: str, source_id: str) -> list[FactualPhrase]:
    phrases: list[FactualPhrase] = []
    phrases.extend(_extract_ner(text, language, source_id))
    for phrase_type, pattern in NUMBER_PATTERNS:
        for match in pattern.finditer(text):
            phrases.append(FactualPhrase(match.group(0), phrase_type, source_id))
    return phrases


def canonicalize_phrase_maps(
    phrase_maps: list[dict[str, set[str]]],
    embedder,
    merge_threshold: float = 0.85,
) -> tuple[list[dict[str, set[str]]], dict[str, set[str]]]:
    phrases = sorted({phrase for phrase_map in phrase_maps for phrase in phrase_map})
    if not phrases:
        return phrase_maps, {}

    type_map: dict[str, set[str]] = {phrase: set() for phrase in phrases}
    frequency = {phrase: 0 for phrase in phrases}
    for phrase_map in phrase_maps:
        for phrase, types in phrase_map.items():
            type_map[phrase].update(types)
            frequency[phrase] += 1

    vectors = np.vstack([_normalize(np.asarray(vector, dtype=float)) for vector in embedder.encode(phrases)])
    parent = list(range(len(phrases)))

    for i in range(len(phrases)):
        for j in range(i + 1, len(phrases)):
            if not _types_compatible(type_map[phrases[i]], type_map[phrases[j]]):
                continue
            if float(vectors[i] @ vectors[j]) >= merge_threshold:
                _union(parent, i, j)

    groups: dict[int, list[str]] = {}
    for idx, phrase in enumerate(phrases):
        groups.setdefault(_find(parent, idx), []).append(phrase)

    canonical_for: dict[str, str] = {}
    aliases: dict[str, set[str]] = {}
    for group in groups.values():
        canonical = max(group, key=lambda phrase: (frequency[phrase], len(phrase), phrase))
        aliases[canonical] = set(group)
        for phrase in group:
            canonical_for[phrase] = canonical

    canonical_maps = []
    for phrase_map in phrase_maps:
        canonical_map: dict[str, set[str]] = {}
        for phrase, types in phrase_map.items():
            canonical = canonical_for[phrase]
            canonical_map.setdefault(canonical, set()).update(types)
        canonical_maps.append(canonical_map)
    return canonical_maps, aliases


def normalize_phrase(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text.lower()


def _types_compatible(left: set[str], right: set[str]) -> bool:
    left_families = {_type_family(item) for item in left}
    right_families = {_type_family(item) for item in right}
    factual = {"percent", "money", "date", "code"}
    if left_families & factual or right_families & factual:
        return False
    return True


def _type_family(phrase_type: str) -> str:
    phrase_type = phrase_type.lower()
    if phrase_type in {"person", "per"}:
        return "person"
    if phrase_type in {"org", "organization"}:
        return "org"
    if phrase_type in {"gpe", "loc", "location"}:
        return "location"
    if phrase_type in {"percent", "money", "date", "code"}:
        return phrase_type
    return "entity"


def _find(parent: list[int], idx: int) -> int:
    while parent[idx] != idx:
        parent[idx] = parent[parent[idx]]
        idx = parent[idx]
    return idx


def _union(parent: list[int], left: int, right: int) -> None:
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _extract_ner(text: str, language: str, source_id: str) -> list[FactualPhrase]:
    if language == "vi":
        from underthesea import ner

        rows = ner(text)
        terms = []
        current = []
        current_type = "entity"
        for row in rows:
            token = row[0]
            label = row[-1]
            if isinstance(label, str) and label.startswith("B-"):
                if current:
                    terms.append((" ".join(current), current_type))
                current = [token]
                current_type = label[2:].lower() or "entity"
            elif isinstance(label, str) and label.startswith("I-") and current:
                current.append(token)
            else:
                if current:
                    terms.append((" ".join(current), current_type))
                    current = []
                    current_type = "entity"
        if current:
            terms.append((" ".join(current), current_type))
        return [FactualPhrase(term, phrase_type, source_id) for term, phrase_type in terms]
    if language == "en":
        nlp = _english_nlp()
        doc = nlp(text)
        return [FactualPhrase(ent.text, ent.label_.lower(), source_id) for ent in doc.ents]
    return _capitalized_phrases(text, source_id)


@lru_cache(maxsize=1)
def _english_nlp():
    import spacy

    return spacy.load("en_core_web_sm")


def _capitalized_phrases(text: str, source_id: str) -> list[FactualPhrase]:
    pattern = re.compile(r"\b(?:[A-ZÀ-Ỵ][\wÀ-ỹ.-]+(?:\s+|$)){1,5}")
    return [FactualPhrase(match.group(0).strip(), "entity", source_id) for match in pattern.finditer(text)]
