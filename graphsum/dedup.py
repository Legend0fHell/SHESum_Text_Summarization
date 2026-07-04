from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocess import Chunk


@dataclass(frozen=True)
class DuplicateGroup:
    group_id: str
    representative_index: int
    member_indices: list[int]
    representative_similarity: dict[int, float]


@dataclass(frozen=True)
class ChunkDedupResult:
    group_for_index: dict[int, str]
    representative_for_group: dict[str, int]
    groups: list[DuplicateGroup]

    def group_id(self, chunk_index: int) -> str:
        return self.group_for_index.get(chunk_index, f"unique_{chunk_index}")

    def representative_index(self, chunk_index: int) -> int:
        return self.representative_for_group.get(self.group_id(chunk_index), chunk_index)


def build_chunk_duplicate_groups(
    chunks: list[Chunk],
    threshold: float = 0.88,
    require_shared_phrase: bool = False,
) -> ChunkDedupResult:
    if not chunks:
        return ChunkDedupResult({}, {}, [])
    embeddings = np.vstack([_normalize(np.asarray(chunk.embedding, dtype=float)) for chunk in chunks])
    similarities: dict[tuple[int, int], float] = {}

    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            similarity = max(0.0, float(embeddings[i] @ embeddings[j]))
            similarities[(i, j)] = similarity

    raw_groups = _representative_groups(chunks, similarities, threshold, require_shared_phrase)

    groups: list[DuplicateGroup] = []
    group_for_index: dict[int, str] = {}
    representative_for_group: dict[str, int] = {}
    duplicate_group_idx = 0
    for members in sorted(raw_groups, key=lambda items: min(items)):
        if len(members) <= 1:
            continue
        group_id = f"dup_{duplicate_group_idx}"
        duplicate_group_idx += 1
        representative = members[0]
        representative_for_group[group_id] = representative
        rep_similarity = {member: _pair_similarity(similarities, representative, member) for member in members}
        for member in members:
            group_for_index[member] = group_id
        groups.append(DuplicateGroup(group_id, representative, members, rep_similarity))
    return ChunkDedupResult(group_for_index, representative_for_group, groups)


def _representative_groups(
    chunks: list[Chunk],
    similarities: dict[tuple[int, int], float],
    threshold: float,
    require_shared_phrase: bool,
) -> list[list[int]]:
    groups: list[list[int]] = []
    for idx in sorted(range(len(chunks)), key=lambda item: (chunks[item].document_id, chunks[item].position, item)):
        assigned = False
        for group in groups:
            representative = group[0]
            if _pair_similarity(similarities, representative, idx) < threshold:
                continue
            if require_shared_phrase and not (chunks[representative].phrases & chunks[idx].phrases):
                continue
            group.append(idx)
            assigned = True
            break
        if not assigned:
            groups.append([idx])
    return groups


def _pair_similarity(similarities: dict[tuple[int, int], float], left: int, right: int) -> float:
    if left == right:
        return 1.0
    return similarities.get((min(left, right), max(left, right)), 0.0)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector
