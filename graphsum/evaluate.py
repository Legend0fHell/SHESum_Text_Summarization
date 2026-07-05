from __future__ import annotations

from pathlib import Path

import pandas as pd
from rouge_score import rouge_scorer


class CharacterTokenizer:
    def tokenize(self, text: str) -> list[str]:
        normalized = "".join((text or "").lower().split())
        return list(normalized)


class RegexTokenizer:
    def __init__(self, pattern: str):
        import re

        self.pattern = re.compile(pattern, flags=re.UNICODE)

    def tokenize(self, text: str) -> list[str]:
        return self.pattern.findall((text or "").lower())


DEFAULT_ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
VI_CHAR_ROUGE_SCORER = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=False,
    tokenizer=CharacterTokenizer(),
)
VI_REGEX_ROUGE_SCORER = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=False,
    tokenizer=RegexTokenizer(r"[\wÀ-ỹ]+"),
)


def rouge_scores(prediction: str, references: list[str], language: str = "en", rouge_tokenizer: str | None = None) -> dict[str, object]:
    scorer, tokenizer_name = _scorer_for(language, rouge_tokenizer)
    if not references:
        return {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "rouge_backend": "rouge-score",
            "rouge_tokenizer": tokenizer_name,
            "reference_count": 0,
            "selected_reference_index": -1,
            "reference_selector": "max_rouge_tuple",
        }
    scores = [_rouge_single(prediction, ref, scorer, tokenizer_name) for ref in references]
    selected_index, selected_score = max(
        enumerate(scores),
        key=lambda item: (item[1]["rouge1"], item[1]["rouge2"], item[1]["rougeL"]),
    )
    return {
        "rouge1": selected_score["rouge1"],
        "rouge2": selected_score["rouge2"],
        "rougeL": selected_score["rougeL"],
        "rouge_backend": selected_score["rouge_backend"],
        "rouge_tokenizer": selected_score["rouge_tokenizer"],
        "reference_count": len(references),
        "selected_reference_index": selected_index,
        "reference_selector": "max_rouge_tuple",
    }


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def _scorer_for(language: str, rouge_tokenizer: str | None) -> tuple[rouge_scorer.RougeScorer, str]:
    tokenizer_name = rouge_tokenizer or ("unicode_char" if language == "vi" else "default")
    if tokenizer_name == "unicode_char":
        return VI_CHAR_ROUGE_SCORER, tokenizer_name
    if tokenizer_name == "vietnamese_aware":
        return VI_REGEX_ROUGE_SCORER, tokenizer_name
    return DEFAULT_ROUGE_SCORER, "default"


def _rouge_single(prediction: str, reference: str, scorer: rouge_scorer.RougeScorer, tokenizer_name: str) -> dict[str, object]:
    scores = scorer.score(reference, prediction)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
        "rouge_backend": "rouge-score",
        "rouge_tokenizer": tokenizer_name,
    }
