from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocess import Chunk


PHRASE_TYPE_WEIGHTS = {
    "person": 1.15,
    "per": 1.15,
    "org": 1.15,
    "organization": 1.15,
    "gpe": 1.10,
    "loc": 1.10,
    "location": 1.10,
    "percent": 1.20,
    "money": 1.20,
    "date": 1.10,
    "code": 1.20,
}


@dataclass(frozen=True)
class GraphWeights:
    alpha: float
    beta: float
    gamma: float


def weight_grid() -> list[GraphWeights]:
    values = []
    for alpha_int in range(0, 21, 5):
        alpha = alpha_int / 100
        max_beta = round(1 - alpha, 10)
        betas = [round(i / 10, 10) for i in range(0, int(max_beta * 10) + 1)]
        if max_beta not in betas:
            betas.append(max_beta)
        for beta in betas:
            gamma = round(1 - alpha - beta, 10)
            if gamma >= -1e-9:
                values.append(GraphWeights(alpha, beta, max(0.0, gamma)))
    return values


def build_weighted_edges(chunks: list[Chunk], weights: GraphWeights, k: int = 8) -> list[tuple[int, int, float]]:
    if len(chunks) <= 1:
        return []
    embeddings = np.vstack([_normalize(np.asarray(chunk.embedding, dtype=float)) for chunk in chunks])
    content = embeddings @ embeddings.T
    phrase_idf = _phrase_idf(chunks)
    candidates = set()
    for i in range(len(chunks)):
        neighbors = np.argsort(-content[i])
        for j in neighbors[1 : k + 1]:
            candidates.add(tuple(sorted((i, int(j)))))
        if i + 1 < len(chunks) and chunks[i].document_id == chunks[i + 1].document_id:
            candidates.add((i, i + 1))
    edges = []
    for i, j in sorted(candidates):
        p = positional_similarity(chunks[i], chunks[j])
        e = phrase_similarity(chunks[i], chunks[j], phrase_idf)
        c = max(0.0, float(content[i, j]))
        score = weights.alpha * p + weights.beta * e + weights.gamma * c
        if score > 0:
            edges.append((i, j, score))
    return edges


def cluster_chunks(chunks: list[Chunk], edges: list[tuple[int, int, float]], resolution: float = 1.0) -> list[list[int]]:
    if not chunks:
        return []
    if not edges:
        return [[idx] for idx in range(len(chunks))]
    import igraph as ig
    import leidenalg

    graph = ig.Graph(n=len(chunks), edges=[(i, j) for i, j, _ in edges], directed=False)
    graph.es["weight"] = [w for _, _, w in edges]
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
    )
    return [list(comm) for comm in partition]


def positional_similarity(left: Chunk, right: Chunk, tau: float = 3.0) -> float:
    if left.document_id != right.document_id:
        return 0.0
    return float(np.exp(-abs(left.position - right.position) / tau))


def phrase_similarity(left: Chunk, right: Chunk, phrase_idf: dict[str, float]) -> float:
    left_vector = _phrase_vector(left, phrase_idf)
    right_vector = _phrase_vector(right, phrase_idf)
    if not left_vector or not right_vector:
        return 0.0
    shared = set(left_vector) & set(right_vector)
    numerator = sum(left_vector[phrase] * right_vector[phrase] for phrase in shared)
    left_norm = float(np.sqrt(sum(weight * weight for weight in left_vector.values())))
    right_norm = float(np.sqrt(sum(weight * weight for weight in right_vector.values())))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _phrase_idf(chunks: list[Chunk]) -> dict[str, float]:
    df: dict[str, int] = {}
    for chunk in chunks:
        for phrase in chunk.phrases:
            df[phrase] = df.get(phrase, 0) + 1
    n = len(chunks)
    return {phrase: float(np.log((n + 1) / (count + 1)) + 1.0) for phrase, count in df.items()}


def _phrase_vector(chunk: Chunk, phrase_idf: dict[str, float]) -> dict[str, float]:
    vector = {}
    for phrase in chunk.phrases:
        types = chunk.phrase_types.get(phrase, {"entity"})
        type_weight = max(PHRASE_TYPE_WEIGHTS.get(phrase_type, 1.0) for phrase_type in types)
        vector[phrase] = phrase_idf.get(phrase, 1.0) * type_weight
    return vector


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector
