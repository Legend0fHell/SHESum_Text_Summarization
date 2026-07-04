from __future__ import annotations

import numpy as np

from .preprocess import TriSentenceUnit, count_tokens


def select_centroid_topk(
    units: list[TriSentenceUnit],
    chunk_embedding: list[float],
    evidence_max_tokens: int,
) -> list[str]:
    chunk_vec = _normalize(np.asarray(chunk_embedding, dtype=float))
    scored = []
    for unit in units:
        if unit.embedding is None:
            continue
        score = float(_normalize(np.asarray(unit.embedding, dtype=float)) @ chunk_vec)
        scored.append((score, unit))
    return _select_by_budget([unit for _, unit in sorted(scored, key=lambda x: x[0], reverse=True)], evidence_max_tokens)


def select_pacsum(
    units: list[TriSentenceUnit],
    evidence_max_tokens: int,
    stride: int = 1,
    beta: float = 0.0,
    lambda1: float = 0.0,
    lambda2: float = 1.0,
) -> list[str]:
    candidates = units if stride <= 1 else units[1::stride]
    candidates = [unit for unit in candidates if unit.embedding is not None]
    if not candidates:
        return []
    matrix = np.vstack([_normalize(np.asarray(unit.embedding, dtype=float)) for unit in candidates])
    similarity = matrix @ matrix.T
    min_score = float(similarity.min())
    max_score = float(similarity.max())
    threshold = min_score + beta * (max_score - min_score)
    adjusted = similarity - threshold
    forward = np.zeros(len(candidates))
    backward = np.zeros(len(candidates))
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            edge = adjusted[i, j]
            if edge > 0:
                forward[j] += edge
                backward[i] += edge
    forward = -forward
    scores = lambda1 * forward + lambda2 * backward
    ranked = [unit for _, unit in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]
    return _select_by_budget(ranked, evidence_max_tokens)


def compact_unit_texts(unit_ids: list[str], unit_lookup: dict[str, TriSentenceUnit], max_tokens: int | None = None) -> str:
    # Merge overlapping segment windows into splitter-output-ordered source spans.
    ordered = sorted((unit_lookup[uid] for uid in unit_ids if uid in unit_lookup), key=lambda u: (u.document_id, u.center_index))
    seen = set()
    texts = []
    for unit in ordered:
        for sentence_id, sentence_text in zip(unit.sentence_ids, unit.sentence_texts):
            if sentence_id in seen:
                continue
            if max_tokens is not None and texts and count_tokens(" ".join(texts + [sentence_text])) > max_tokens:
                return " ".join(texts)
            seen.add(sentence_id)
            texts.append(sentence_text)
    return " ".join(texts)


def _select_by_budget(units: list[TriSentenceUnit], evidence_max_tokens: int) -> list[str]:
    selected = []
    total = 0
    for unit in units:
        tokens = count_tokens(unit.text)
        if selected and total + tokens > evidence_max_tokens:
            continue
        selected.append(unit.unit_id)
        total += tokens
        if total >= evidence_max_tokens:
            break
    return selected


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector
