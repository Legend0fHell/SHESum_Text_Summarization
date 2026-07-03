from __future__ import annotations

from pathlib import Path

import pandas as pd
from rouge_score import rouge_scorer


ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)


def rouge_scores(prediction: str, references: list[str]) -> dict[str, object]:
    if not references:
        return {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "rouge_backend": "rouge-score",
            "reference_count": 0,
            "selected_reference_index": -1,
            "reference_selector": "max_mean_rouge2_rougeL",
        }
    scores = [_rouge_single(prediction, ref) for ref in references]
    selected_index, selected_score = max(
        enumerate(scores),
        key=lambda item: (item[1]["rouge2"] + item[1]["rougeL"]) / 2,
    )
    return {
        "rouge1": selected_score["rouge1"],
        "rouge2": selected_score["rouge2"],
        "rougeL": selected_score["rougeL"],
        "rouge_backend": selected_score["rouge_backend"],
        "reference_count": len(references),
        "selected_reference_index": selected_index,
        "reference_selector": "max_mean_rouge2_rougeL",
    }


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def _rouge_single(prediction: str, reference: str) -> dict[str, object]:
    scores = ROUGE_SCORER.score(reference, prediction)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
        "rouge_backend": "rouge-score",
    }
