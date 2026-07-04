from __future__ import annotations

from pathlib import Path

import pandas as pd


NUMERIC_COLUMNS = [
    "rouge1",
    "rouge2",
    "rougeL",
    "input_tokens",
    "output_tokens",
    "llm_calls",
    "runtime_seconds",
    "chunk_count",
    "topic_count",
]

GROUP_COLUMNS = [
    "dataset",
    "run_mode",
    "salience",
    "alpha",
    "beta",
    "gamma",
    "use_graph",
    "chunking",
    "target_min_tokens",
    "target_max_tokens",
    "semantic_breakpoint_percentile",
    "semantic_min_chunk_tokens",
    "dedup_chunks",
    "dedup_sim_threshold",
    "dedup_require_shared_phrase",
    "duplicate_edge_factor",
    "community_dedup",
    "embedding_backend",
    "embedding_model",
    "llm",
    "llm_model",
    "pacsum_beta",
    "pacsum_lambda1",
    "pacsum_lambda2",
    "entity_merge_threshold",
    "rouge_backend",
]


def aggregate_result_files(inputs: list[str | Path]) -> list[dict[str, object]]:
    frame = pd.concat((pd.read_csv(input_path) for input_path in inputs), ignore_index=True)
    return aggregate_result_frame(frame)


def aggregate_result_frame(frame: pd.DataFrame) -> list[dict[str, object]]:
    frame = frame.copy()
    for column in GROUP_COLUMNS:
        if column not in frame:
            frame[column] = ""
    for column in NUMERIC_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    grouped = frame.groupby(GROUP_COLUMNS, dropna=False, as_index=False)
    counts = grouped.size().rename(columns={"size": "n"})
    means = grouped[NUMERIC_COLUMNS].mean().rename(columns={column: f"mean_{column}" for column in NUMERIC_COLUMNS})
    return counts.merge(means, on=GROUP_COLUMNS, how="left").to_dict(orient="records")
