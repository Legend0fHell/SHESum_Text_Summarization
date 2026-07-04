from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocess import Chunk, count_tokens


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
    parent = list(range(len(chunks)))
    similarities: dict[tuple[int, int], float] = {}

    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            similarity = max(0.0, float(embeddings[i] @ embeddings[j]))
            similarities[(i, j)] = similarity
            if similarity < threshold:
                continue
            if require_shared_phrase and not (chunks[i].phrases & chunks[j].phrases):
                continue
            _union(parent, i, j)

    raw_groups: dict[int, list[int]] = {}
    for idx in range(len(chunks)):
        raw_groups.setdefault(_find(parent, idx), []).append(idx)

    groups: list[DuplicateGroup] = []
    group_for_index: dict[int, str] = {}
    representative_for_group: dict[str, int] = {}
    duplicate_group_idx = 0
    for members in sorted(raw_groups.values(), key=lambda items: min(items)):
        if len(members) <= 1:
            continue
        group_id = f"dup_{duplicate_group_idx}"
        duplicate_group_idx += 1
        representative = _choose_representative(chunks, members)
        representative_for_group[group_id] = representative
        rep_similarity = {member: _pair_similarity(similarities, representative, member) for member in members}
        for member in members:
            group_for_index[member] = group_id
        groups.append(DuplicateGroup(group_id, representative, members, rep_similarity))
    return ChunkDedupResult(group_for_index, representative_for_group, groups)


def _choose_representative(chunks: list[Chunk], members: list[int]) -> int:
    return min(
        members,
        key=lambda idx: (
            chunks[idx].document_id,
            chunks[idx].position,
            -len(chunks[idx].phrases),
            -count_tokens(chunks[idx].text),
        ),
    )


def _pair_similarity(similarities: dict[tuple[int, int], float], left: int, right: int) -> float:
    if left == right:
        return 1.0
    return similarities.get((min(left, right), max(left, right)), 0.0)


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
