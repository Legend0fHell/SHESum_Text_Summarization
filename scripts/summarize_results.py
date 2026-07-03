from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphsum.evaluate import write_csv


NUMERIC_COLUMNS = [
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
    "salience",
    "alpha",
    "beta",
    "gamma",
    "use_graph",
    "chunking",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate graph summarization result CSVs.")
    parser.add_argument("inputs", nargs="+", help="Input result CSV files.")
    parser.add_argument("--output", default="runs/summary_results.csv")
    args = parser.parse_args()

    frame = pd.concat((pd.read_csv(input_path) for input_path in args.inputs), ignore_index=True)
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
    output_rows = counts.merge(means, on=GROUP_COLUMNS, how="left").to_dict(orient="records")

    write_csv(Path(args.output), output_rows)
    for row in output_rows:
        print(row)


if __name__ == "__main__":
    main()
