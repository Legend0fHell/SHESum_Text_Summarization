from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphsum.aggregate import aggregate_result_frame
from graphsum.data import load_samples
from graphsum.evaluate import rouge_scores, write_csv
from graphsum.graph import GraphWeights, weight_grid
from graphsum.llm import make_llm
from graphsum.pipeline import Embedder, PipelineConfig, run_direct_sample, run_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Run graph-guided Extract-Support summarization experiments.")
    parser.add_argument("--dataset", choices=["vn_mds", "vims", "multi_news"], required=True)
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--salience", choices=["e1", "e2a", "e2b"], default="e1")
    parser.add_argument("--llm", choices=["dry_run", "openai_compatible"], default="dry_run")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--dry-embed", action="store_true", help="Use hash embeddings instead of BGE-M3.")
    parser.add_argument("--embedding-backend", choices=["sentence_transformers", "openai_compatible"], default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-base-url", default=None)
    parser.add_argument("--embedding-api-key", default=None)
    parser.add_argument("--chunking", choices=["semantic", "simple"], default="semantic")
    parser.add_argument("--grid", action="store_true", help="Run the alpha/beta/gamma graph-weight grid.")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--beta", type=float, default=0.20)
    parser.add_argument("--pacsum-beta", type=float, default=0.0)
    parser.add_argument("--pacsum-lambda1", type=float, default=0.0)
    parser.add_argument("--pacsum-lambda2", type=float, default=1.0)
    parser.add_argument("--entity-merge-threshold", type=float, default=0.85)
    parser.add_argument("--no-graph", action="store_true", help="Sequential hierarchical baseline without graph clustering.")
    parser.add_argument("--pure-llm", action="store_true", help="Direct baseline: feed all source text to the LLM without graph, chunking, support selection, or embeddings.")
    parser.add_argument("--aggregate-output", default=None, help="Optional aggregate metrics CSV. Defaults to <output>_summary.csv for openai_compatible runs.")
    parser.add_argument("--output", default="runs/graphsum_results.csv")
    args = parser.parse_args()

    samples = load_samples(args.dataset, args.data_root, limit=args.limit)
    llm = make_llm(args.llm, model=args.model, base_url=args.base_url, api_key=args.api_key, temperature=args.temperature)
    rows = []
    if args.pure_llm:
        for sample in samples:
            started = time.perf_counter()
            output = run_direct_sample(sample, llm)
            runtime_seconds = time.perf_counter() - started
            rouge = rouge_scores(output.summary, sample.references)
            rows.append(
                {
                    "dataset": sample.dataset,
                    "sample_id": sample.sample_id,
                    "run_mode": "pure_llm",
                    "salience": "none",
                    "alpha": "",
                    "beta": "",
                    "gamma": "",
                    "use_graph": False,
                    "chunking": "none",
                    "embedding_backend": "none",
                    "embedding_model": "none",
                    "llm": args.llm,
                    "llm_model": getattr(llm, "model", "dry_run"),
                    "pacsum_beta": "",
                    "pacsum_lambda1": "",
                    "pacsum_lambda2": "",
                    "entity_merge_threshold": "",
                    "rouge1": rouge["rouge1"],
                    "rouge2": rouge["rouge2"],
                    "rougeL": rouge["rougeL"],
                    "rouge_backend": rouge["rouge_backend"],
                    "reference_count": rouge["reference_count"],
                    "selected_reference_index": rouge["selected_reference_index"],
                    "reference_selector": rouge["reference_selector"],
                    "input_tokens": output.input_tokens,
                    "output_tokens": output.output_tokens,
                    "llm_calls": output.llm_calls,
                    "runtime_seconds": runtime_seconds,
                    "chunk_count": output.chunk_count,
                    "topic_count": output.topic_count,
                    "generated_summary": output.summary,
                }
            )
            _print_row(rows[-1])
        _write_outputs(Path(args.output), rows, args.aggregate_output, args.llm)
        return

    embedder = Embedder(
        dry_run=args.dry_embed,
        backend=args.embedding_backend,
        model_name=args.embedding_model,
        base_url=args.embedding_base_url,
        api_key=args.embedding_api_key,
    )
    weights_list = weight_grid() if args.grid else [GraphWeights(args.alpha, args.beta, 1 - args.alpha - args.beta)]
    for weights in weights_list:
        config = PipelineConfig(
            salience_method=args.salience,
            graph_weights=weights,
            use_graph=not args.no_graph,
            chunking_method=args.chunking,
            pacsum_beta=args.pacsum_beta,
            pacsum_lambda1=args.pacsum_lambda1,
            pacsum_lambda2=args.pacsum_lambda2,
            entity_merge_threshold=args.entity_merge_threshold,
        )
        for sample in samples:
            started = time.perf_counter()
            output = run_sample(sample, config, embedder, llm)
            runtime_seconds = time.perf_counter() - started
            rouge = rouge_scores(output.summary, sample.references)
            rows.append(
                {
                    "dataset": sample.dataset,
                    "sample_id": sample.sample_id,
                    "run_mode": "graphsum",
                    "salience": args.salience,
                    "alpha": weights.alpha,
                    "beta": weights.beta,
                    "gamma": weights.gamma,
                    "use_graph": not args.no_graph,
                    "chunking": args.chunking,
                    "embedding_backend": embedder.backend,
                    "embedding_model": embedder.model_name,
                    "llm": args.llm,
                    "llm_model": getattr(llm, "model", "dry_run"),
                    "pacsum_beta": args.pacsum_beta,
                    "pacsum_lambda1": args.pacsum_lambda1,
                    "pacsum_lambda2": args.pacsum_lambda2,
                    "entity_merge_threshold": args.entity_merge_threshold,
                    "rouge1": rouge["rouge1"],
                    "rouge2": rouge["rouge2"],
                    "rougeL": rouge["rougeL"],
                    "rouge_backend": rouge["rouge_backend"],
                    "reference_count": rouge["reference_count"],
                    "selected_reference_index": rouge["selected_reference_index"],
                    "reference_selector": rouge["reference_selector"],
                    "input_tokens": output.input_tokens,
                    "output_tokens": output.output_tokens,
                    "llm_calls": output.llm_calls,
                    "runtime_seconds": runtime_seconds,
                    "chunk_count": output.chunk_count,
                    "topic_count": output.topic_count,
                    "generated_summary": output.summary,
                }
            )
            _print_row(rows[-1])
    _write_outputs(Path(args.output), rows, args.aggregate_output, args.llm)


def _print_row(row: dict[str, object]) -> None:
    print(json.dumps(row, ensure_ascii=True))


def _write_outputs(output_path: Path, rows: list[dict[str, object]], aggregate_output: str | None, llm_kind: str) -> None:
    write_csv(output_path, rows)
    summary_path = Path(aggregate_output) if aggregate_output else _default_summary_path(output_path, llm_kind)
    if summary_path is None:
        return
    summary_rows = aggregate_result_frame(pd.DataFrame(rows))
    write_csv(summary_path, summary_rows)
    print(json.dumps({"aggregate_output": str(summary_path)}, ensure_ascii=True))


def _default_summary_path(output_path: Path, llm_kind: str) -> Path | None:
    if llm_kind != "openai_compatible":
        return None
    return output_path.with_name(f"{output_path.stem}_summary{output_path.suffix}")


if __name__ == "__main__":
    main()
