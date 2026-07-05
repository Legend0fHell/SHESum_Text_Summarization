from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphsum.aggregate import aggregate_result_frame
from graphsum.config import experiment_settings, llm_settings
from graphsum.data import Sample, load_samples
from graphsum.evaluate import rouge_scores, write_csv


REPO_ROOT = Path(__file__).resolve().parents[1]
MOA_ROOT = REPO_ROOT / "libs" / "MoA-MDS"


def main() -> None:
    settings = experiment_settings()
    env_llm = llm_settings()
    parser = argparse.ArgumentParser(description="Run libs/MoA-MDS with GraphSum dataset and LLM configuration.")
    parser.add_argument("--dataset", choices=["vn_mds", "vims", "multi_news"], required=settings.dataset is None, default=settings.dataset)
    parser.add_argument("--data-root", default=settings.data_root)
    parser.add_argument("--limit", type=int, default=settings.limit)
    parser.add_argument("--model", default=env_llm.model)
    parser.add_argument("--base-url", default=env_llm.base_url)
    parser.add_argument("--api-key", default=env_llm.api_key)
    parser.add_argument("--temperature", type=float, default=env_llm.temperature)
    parser.add_argument("--max-tokens-param", default="max_tokens", help="Use max_tokens for vLLM/Ollama-compatible endpoints; MoA release default is max_completion_tokens.")
    parser.add_argument("--output", default="runs/moa_mds_results.csv")
    parser.add_argument("--aggregate-output", default=None)
    parser.add_argument("--case-output-root", default="runs/moa_mds_case_studies")
    parser.add_argument("--no-case-studies", action="store_true", help="Disable MoA case-study JSON/text artifacts.")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not args.model:
        raise ValueError("GRAPHSUM_LLM_MODEL or --model is required to run MoA-MDS.")

    _ensure_moa_import_path()
    from src.agents import MixtureOfAgents
    from src.utils.config import load_config

    samples = load_samples(args.dataset, args.data_root, limit=args.limit)
    config = _build_moa_config(
        load_config,
        dataset=args.dataset,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens_param=args.max_tokens_param,
        output_root=args.case_output_root,
    )
    moa = MixtureOfAgents(config, output_root=args.case_output_root)

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for sample in samples:
        started = time.perf_counter()
        try:
            moa_sample = _to_moa_sample(sample)
            result = moa.process_sample(moa_sample, write_case_study=not args.no_case_studies, sample_id_suffix="_moa_graphsum")
            runtime_seconds = time.perf_counter() - started
            row = _result_row(sample, result, runtime_seconds, args, config)
            rows.append(row)
            _print_json(row)
        except Exception as exc:
            failure = {
                "dataset": sample.dataset,
                "sample_id": sample.sample_id,
                "run_mode": "moa_mds",
                "error": str(exc),
            }
            failures.append(failure)
            _print_json(failure)
            if not args.continue_on_error:
                raise

    output_path = Path(args.output)
    write_csv(output_path, rows)
    summary_path = Path(args.aggregate_output) if args.aggregate_output else output_path.with_name(f"{output_path.stem}_summary{output_path.suffix}")
    if rows:
        write_csv(summary_path, aggregate_result_frame(pd.DataFrame(rows)))
        _print_json({"aggregate_output": str(summary_path)})
    if failures:
        failure_path = output_path.with_name(f"{output_path.stem}_failures.jsonl")
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        with failure_path.open("w", encoding="utf-8") as file:
            for failure in failures:
                file.write(json.dumps(failure, ensure_ascii=False) + "\n")
        _print_json({"failure_output": str(failure_path), "failed": len(failures)})


def _ensure_moa_import_path() -> None:
    if not MOA_ROOT.exists():
        raise FileNotFoundError(f"MoA-MDS checkout not found: {MOA_ROOT}")
    sys.path.insert(0, str(MOA_ROOT))


def _build_moa_config(
    load_config,
    *,
    dataset: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
    temperature: float | None,
    max_tokens_param: str,
    output_root: str,
):
    api_key_env = "GRAPHSUM_MOA_API_KEY"
    if api_key:
        os.environ[api_key_env] = api_key
    config = load_config(
        base_config=str(MOA_ROOT / "configs" / "base.yaml"),
        storage_config=str(MOA_ROOT / "configs" / "storage.yaml"),
        overlay_configs=[
            str(MOA_ROOT / "configs" / "agents" / "default.yaml"),
            str(MOA_ROOT / "configs" / "evaluations" / f"{dataset}.yaml"),
        ],
    )
    # Keep MoA-MDS method/evaluation overlays intact. Only the LLM runtime and
    # output path are replaced so MoA uses the same vLLM-compatible endpoint as GraphSum.
    return config.merge(
        {
            "llm": {
                "model_id": model,
                "base_url": base_url,
                "api_key_env": api_key_env,
                "generation": {
                    "temperature": temperature,
                    "max_tokens_param": max_tokens_param,
                    # The release default targets OpenAI reasoning models; most
                    # OpenAI-compatible vLLM endpoints reject this extra field.
                    "reasoning_effort": None,
                },
            },
            "outputs": {"root_dir": output_root},
        }
    )


def _to_moa_sample(sample: Sample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "dataset_name": sample.dataset,
        "split": "test",
        "language": sample.language,
        "documents": [_document_text(document) for document in sample.documents],
        "reference_summary": sample.references[0] if sample.references else "",
        "metadata": {
            "reference_summaries": sample.references,
            "num_references": len(sample.references),
            "num_documents": len(sample.documents),
            "graphsum_loader": True,
        },
    }


def _document_text(document) -> str:
    lines = []
    if document.title:
        lines.append(f"Title: {document.title}")
    if document.source:
        lines.append(f"Source: {document.source}")
    lines.append(document.text)
    return "\n".join(line for line in lines if line)


def _result_row(sample: Sample, result: Mapping[str, Any], runtime_seconds: float, args: argparse.Namespace, config) -> dict[str, object]:
    final_summary = str(result.get("final_summary", ""))
    rouge = rouge_scores(final_summary, sample.references, sample.language)
    token_usage = _sum_agent_token_usage(result.get("agents", {}))
    kg_metrics = _kg_metrics(result.get("agents", {}))
    moa_values = _moa_config_values(config)
    return {
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "run_mode": "moa_mds",
        "salience": "moa_mds",
        "alpha": "",
        "beta": "",
        "gamma": "",
        "use_graph": True,
        "chunking": "moa_mds",
        "target_min_tokens": "",
        "target_max_tokens": "",
        "semantic_breakpoint_percentile": "",
        "semantic_min_chunk_tokens": "",
        "dedup_chunks": "",
        "dedup_sim_threshold": "",
        "dedup_require_shared_phrase": "",
        "duplicate_edge_factor": "",
        "community_dedup": "",
        "max_summary_words": moa_values["amf_max_summary_words"],
        "max_output_tokens": moa_values["amf_max_summary_tokens"],
        "embedding_backend": "moa_mds",
        "embedding_model": "lexical_centroid",
        "llm": "openai_compatible",
        "llm_model": args.model,
        "pacsum_beta": "",
        "pacsum_lambda1": "",
        "pacsum_lambda2": "",
        "entity_merge_threshold": "0.85",
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "rouge_backend": rouge["rouge_backend"],
        "rouge_tokenizer": rouge["rouge_tokenizer"],
        "reference_count": rouge["reference_count"],
        "selected_reference_index": rouge["selected_reference_index"],
        "reference_selector": rouge["reference_selector"],
        "input_tokens": token_usage["input_tokens"],
        "output_tokens": token_usage["output_tokens"],
        "llm_calls": _count_agent_calls(result.get("agents", {})),
        "runtime_seconds": runtime_seconds,
        "chunk_count": "",
        "community_count": kg_metrics.get("num_communities", ""),
        "moa_num_entities": kg_metrics.get("num_entities", ""),
        "moa_num_relations": kg_metrics.get("num_relations", ""),
        "moa_contrastive_relation_count": kg_metrics.get("contrastive_relation_count", ""),
        "moa_config_overlays": moa_values["config_overlays"],
        "moa_llm_base_url": args.base_url or "",
        "moa_llm_max_tokens_param": args.max_tokens_param,
        "moa_extractor_model_name": moa_values["extractor_model_name"],
        "moa_extractor_lambda": moa_values["extractor_lambda"],
        "moa_extractor_top_sentence_proportion": moa_values["extractor_top_sentence_proportion"],
        "moa_kgsum_max_summary_tokens": moa_values["kgsum_max_summary_tokens"],
        "moa_abstractor_max_summary_words": moa_values["abstractor_max_summary_words"],
        "moa_abstractor_max_summary_tokens": moa_values["abstractor_max_summary_tokens"],
        "moa_amf_max_summary_words": moa_values["amf_max_summary_words"],
        "moa_amf_max_summary_tokens": moa_values["amf_max_summary_tokens"],
        "moa_force_extractor_anchor": moa_values["force_extractor_anchor"],
        "moa_max_kgsum_weight": moa_values["max_kgsum_weight"],
        "moa_direct_anchor_output": moa_values["direct_anchor_output"],
        "moa_direct_anchor_policy": moa_values["direct_anchor_policy"],
        "case_agent_log": result.get("artifact_paths", {}).get("agent_log", ""),
        "case_metadata": result.get("artifact_paths", {}).get("metadata", ""),
        "generated_summary": final_summary,
    }


def _sum_agent_token_usage(agents: Any) -> dict[str, int | str]:
    if not isinstance(agents, Mapping):
        return {"input_tokens": "", "output_tokens": ""}
    input_tokens = 0
    output_tokens = 0
    saw_input = False
    saw_output = False
    for agent in agents.values():
        if not isinstance(agent, Mapping):
            continue
        usage = agent.get("token_usage") or {}
        if usage.get("input_tokens") is not None:
            input_tokens += int(usage["input_tokens"])
            saw_input = True
        if usage.get("output_tokens") is not None:
            output_tokens += int(usage["output_tokens"])
            saw_output = True
    return {"input_tokens": input_tokens if saw_input else "", "output_tokens": output_tokens if saw_output else ""}


def _count_agent_calls(agents: Any) -> int:
    if not isinstance(agents, Mapping):
        return 0
    return sum(len(agent.get("llm_calls") or []) for agent in agents.values() if isinstance(agent, Mapping))


def _kg_metrics(agents: Any) -> dict[str, object]:
    if not isinstance(agents, Mapping):
        return {}
    kgsum = agents.get("kgsum", {})
    metadata = kgsum.get("metadata", {}) if isinstance(kgsum, Mapping) else {}
    kg_metadata = metadata.get("M_kg", {}) if isinstance(metadata, Mapping) else {}
    metrics = kg_metadata.get("graph_metrics", {}) if isinstance(kg_metadata, Mapping) else {}
    return dict(metrics) if isinstance(metrics, Mapping) else {}


def _moa_config_values(config) -> dict[str, object]:
    source_paths = [str(path) for path in getattr(config, "source_paths", [])]
    return {
        "config_overlays": ";".join(source_paths),
        "extractor_model_name": config.get("method.extractor.model_name", ""),
        "extractor_lambda": config.get("method.extractor.lambda", ""),
        "extractor_top_sentence_proportion": config.get("method.extractor.top_sentence_proportion", ""),
        "kgsum_max_summary_tokens": config.get("method.kgsum.max_summary_tokens", ""),
        "abstractor_max_summary_words": config.get("method.abstractor.max_summary_words", ""),
        "abstractor_max_summary_tokens": config.get("method.abstractor.max_summary_tokens", ""),
        "amf_max_summary_words": config.get("method.amf.max_summary_words", ""),
        "amf_max_summary_tokens": config.get("method.amf.max_summary_tokens", ""),
        "force_extractor_anchor": config.get("method.amf.force_extractor_anchor", ""),
        "max_kgsum_weight": config.get("method.amf.max_kgsum_weight", ""),
        "direct_anchor_output": config.get("method.amf.direct_anchor_output", ""),
        "direct_anchor_policy": config.get("method.amf.direct_anchor_policy", ""),
    }


def _print_json(payload: Mapping[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=True))


if __name__ == "__main__":
    main()
