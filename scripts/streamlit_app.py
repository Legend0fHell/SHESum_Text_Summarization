from __future__ import annotations

import sys
import time
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphsum.data import load_samples
from graphsum.config import embedding_settings, experiment_settings, llm_settings
from graphsum.evaluate import rouge_scores
from graphsum.graph import GraphWeights
from graphsum.llm import make_llm
from graphsum.pipeline import Embedder, PipelineConfig, PipelineTrace, run_direct_sample, run_sample


STAGES = ["preprocess", "embed", "entities", "support", "graph", "summarize", "merge", "direct", "done"]


def main() -> None:
    st.set_page_config(page_title="GraphSum Sample Viewer", layout="wide")
    st.title("GraphSum Sample Viewer")
    st.caption("Run one sample and inspect segments, chunk communities, graph edges, entities, summary graph, and generated summary.")
    env_llm = llm_settings()
    env_embedding = embedding_settings()
    env_experiment = experiment_settings()

    with st.sidebar:
        st.header("Sample")
        dataset_options = ["vn_mds", "vims", "multi_news"]
        dataset = st.selectbox(
            "Dataset",
            dataset_options,
            index=dataset_options.index(env_experiment.dataset) if env_experiment.dataset in dataset_options else 0,
        )
        data_root = st.text_input("Data root", env_experiment.data_root)
        preview_limit = st.number_input("Samples to load", min_value=1, max_value=500, value=max(1, env_experiment.limit), step=1)

        samples = load_samples(dataset, data_root, limit=int(preview_limit))
        sample_labels = [f"{sample.sample_id} ({len(sample.documents)} docs, {len(sample.references)} refs)" for sample in samples]
        sample_index = st.selectbox("Sample", range(len(samples)), format_func=lambda idx: sample_labels[idx]) if samples else None

        st.header("Run Mode")
        st.caption("Defaults are loaded from the root `.env`; non-empty sidebar fields override them for this run.")
        run_mode_options = ["graphsum", "pure_llm"]
        run_mode_default = "pure_llm" if env_experiment.pure_llm else "graphsum"
        run_mode = st.selectbox("Mode", run_mode_options, index=run_mode_options.index(run_mode_default))
        llm_options = ["dry_run", "openai_compatible"]
        llm_kind = st.selectbox("LLM", llm_options, index=llm_options.index(env_experiment.llm) if env_experiment.llm in llm_options else 0)
        model = st.text_input("LLM model", env_llm.model or "")
        base_url = st.text_input("LLM base URL", env_llm.base_url or "")
        api_key = st.text_input("LLM API key override", "", type="password")
        temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0, value=float(env_llm.temperature), step=0.1)

        if run_mode == "graphsum":
            st.header("GraphSum Config")
            salience_options = ["e1", "e2a", "e2b"]
            salience = st.selectbox("Salience", salience_options, index=salience_options.index(env_experiment.salience) if env_experiment.salience in salience_options else 0)
            use_graph = not st.checkbox("Disable graph clustering", value=env_experiment.no_graph)
            chunking_options = ["semantic", "simple"]
            chunking = st.selectbox("Chunking", chunking_options, index=chunking_options.index(env_experiment.chunking) if env_experiment.chunking in chunking_options else 0)
            target_min_tokens = st.number_input("Target min tokens", min_value=1, max_value=10000, value=env_experiment.target_min_tokens, step=50)
            target_max_tokens = st.number_input("Target max tokens", min_value=1, max_value=20000, value=env_experiment.target_max_tokens, step=50)
            semantic_breakpoint_percentile = st.number_input(
                "Semantic breakpoint percentile",
                min_value=1.0,
                max_value=99.0,
                value=env_experiment.semantic_breakpoint_percentile,
                step=1.0,
            )
            semantic_min_chunk_tokens = st.number_input(
                "Semantic min chunk tokens",
                min_value=1,
                max_value=10000,
                value=env_experiment.semantic_min_chunk_tokens or max(100, env_experiment.target_min_tokens // 4),
                step=25,
            )
            dry_embed = st.checkbox(
                "Dry embeddings",
                value=env_experiment.dry_embed,
                help="Use deterministic hash embeddings for debugging only. Leave off to use the embedding backend from `.env` or the sidebar.",
            )
            embedding_backends = ["sentence_transformers", "openai_compatible"]
            embedding_backend = st.selectbox(
                "Embedding backend",
                embedding_backends,
                index=embedding_backends.index(env_embedding.backend) if env_embedding.backend in embedding_backends else 0,
            )
            embedding_model = st.text_input("Embedding model", env_embedding.model or "")
            embedding_base_url = st.text_input("Embedding base URL", env_embedding.base_url or "")
            embedding_api_key = st.text_input("Embedding API key override", "", type="password")
            alpha = st.number_input("alpha", min_value=0.0, max_value=1.0, value=env_experiment.alpha, step=0.05)
            beta = st.number_input("beta", min_value=0.0, max_value=1.0, value=env_experiment.beta, step=0.05)
            pacsum_beta = st.number_input("PACSUM beta", value=env_experiment.pacsum_beta, step=0.1)
            pacsum_lambda1 = st.number_input("PACSUM lambda1", value=env_experiment.pacsum_lambda1, step=0.1)
            pacsum_lambda2 = st.number_input("PACSUM lambda2", value=env_experiment.pacsum_lambda2, step=0.1)
            entity_merge_threshold = st.number_input("Entity merge threshold", min_value=0.0, max_value=1.0, value=env_experiment.entity_merge_threshold, step=0.01)

        run_clicked = st.button("Run Sample", type="primary", disabled=sample_index is None)

    if not samples:
        st.warning("No samples loaded. Check dataset and data root.")
        return

    selected_sample = samples[int(sample_index)] if sample_index is not None else None
    current_selection_key = f"{dataset}:{data_root}:{selected_sample.sample_id}:{run_mode}" if selected_sample is not None else ""

    if run_clicked:
        sample = selected_sample
        progress_bar = st.progress(0)
        status = st.empty()
        trace = PipelineTrace()
        started = time.perf_counter()

        def progress_callback(stage: str, message: str) -> None:
            progress_bar.progress(_stage_progress(stage))
            status.info(f"{stage}: {message}")

        try:
            llm = make_llm(
                llm_kind,
                model=model or None,
                base_url=base_url or None,
                api_key=api_key or None,
                temperature=temperature,
            )
            if run_mode == "pure_llm":
                output = run_direct_sample(sample, llm, trace=trace, progress_callback=progress_callback)
            else:
                embedder = Embedder(
                    dry_run=dry_embed,
                    backend=embedding_backend,
                    model_name=embedding_model or None,
                    base_url=embedding_base_url or None,
                    api_key=embedding_api_key or None,
                )
                config = PipelineConfig(
                    salience_method=salience,
                    graph_weights=GraphWeights(alpha, beta, 1 - alpha - beta),
                    use_graph=use_graph,
                    chunking_method=chunking,
                    target_min_tokens=target_min_tokens,
                    target_max_tokens=target_max_tokens,
                    semantic_breakpoint_percentile=semantic_breakpoint_percentile,
                    semantic_min_chunk_tokens=semantic_min_chunk_tokens,
                    pacsum_beta=pacsum_beta,
                    pacsum_lambda1=pacsum_lambda1,
                    pacsum_lambda2=pacsum_lambda2,
                    entity_merge_threshold=entity_merge_threshold,
                )
                output = run_sample(sample, config, embedder, llm, trace=trace, progress_callback=progress_callback)
            runtime_seconds = time.perf_counter() - started
            rouge = rouge_scores(output.summary, sample.references)
            st.session_state["viewer_result"] = {
                "selection_key": current_selection_key,
                "sample_id": sample.sample_id,
                "run_mode": run_mode,
                "runtime_seconds": runtime_seconds,
                "output": output,
                "trace": trace,
                "rouge": rouge,
            }
            progress_bar.progress(1.0)
            status.success("Finished")
        except Exception as exc:  # Streamlit should show endpoint/context errors instead of hiding them.
            status.error(f"Run failed: {exc}")
            raise

    result = st.session_state.get("viewer_result")
    if result:
        if result.get("selection_key") == current_selection_key:
            _render_result(result)
        else:
            st.info("The sidebar selection changed. Click `Run Sample` to generate results for the selected sample.")


def _render_result(result: dict[str, object]) -> None:
    output = result["output"]
    trace = result["trace"]
    rouge = result["rouge"]

    st.subheader(f"Result: {result['sample_id']} / {result['run_mode']}")
    metric_cols = st.columns(8)
    metric_cols[0].metric("ROUGE-1", f"{rouge['rouge1']:.4f}")
    metric_cols[1].metric("ROUGE-2", f"{rouge['rouge2']:.4f}")
    metric_cols[2].metric("ROUGE-L", f"{rouge['rougeL']:.4f}")
    metric_cols[3].metric("Input tokens", output.input_tokens)
    metric_cols[4].metric("Output tokens", output.output_tokens)
    metric_cols[5].metric("LLM calls", output.llm_calls)
    metric_cols[6].metric("Chunks", output.chunk_count)
    metric_cols[7].metric("Seconds", f"{result['runtime_seconds']:.2f}")

    tabs = st.tabs(["Progress", "Segments", "Chunks & Entities", "Communities", "KNN Graph", "Summary Graph", "Final Summary"])
    with tabs[0]:
        _dataframe(trace.progress, "No progress events recorded.")
    with tabs[1]:
        st.markdown("**Rule-based splitter outputs**")
        _dataframe(trace.splitter_outputs, "No splitter outputs recorded.")
        st.markdown("**Segments: overlapping three-output evidence windows**")
        _dataframe(trace.segments, "No segments recorded.")
    with tabs[2]:
        st.markdown("**Semantic chunks**")
        _dataframe(trace.chunks, "No chunks recorded.")
        st.markdown("**Chunk entities and factual phrases**")
        _dataframe(trace.chunk_entities, "No entities recorded.")
        st.markdown("**Selected support segments**")
        _dataframe(trace.support, "No support segments recorded.")
    with tabs[3]:
        _dataframe(trace.communities, "No chunk communities recorded.")
        _dataframe(trace.community_summaries, "No community summaries recorded.")
    with tabs[4]:
        _dataframe(trace.graph_edges, "No graph edges recorded.")
        if trace.graph_edges:
            st.graphviz_chart(_dot_from_weighted_edges(trace.graph_edges))
    with tabs[5]:
        _dataframe(trace.summary_graph_nodes, "No summary graph nodes recorded.")
        _dataframe(trace.summary_graph_edges, "No summary graph edges recorded.")
        if trace.summary_graph_nodes:
            st.graphviz_chart(_dot_from_summary_graph(trace.summary_graph_nodes, trace.summary_graph_edges))
    with tabs[6]:
        st.text_area("Generated summary", output.summary, height=320)
        _render_summary_steps(trace.summary_steps)


def _dataframe(rows: list[dict[str, object]], empty_message: str) -> None:
    if not rows:
        st.info(empty_message)
        return
    st.markdown(_wrapped_table_html(pd.DataFrame(rows)), unsafe_allow_html=True)


def _render_summary_steps(rows: list[dict[str, object]]) -> None:
    if not rows:
        st.info("No prompts recorded.")
        return
    st.markdown("**Prompt and result trace**")
    for row in rows:
        label = f"{row['step_id']} ({row['step_type']}, input={row['input_tokens']}, output={row['output_tokens']})"
        with st.expander(label, expanded=False):
            st.text_area("Input prompt", str(row["prompt"]), height=260, key=f"prompt_{row['step_id']}")
            st.text_area("Step result", str(row["summary"]), height=180, key=f"summary_{row['step_id']}")


def _wrapped_table_html(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
    body_rows = []
    for _, row in frame.iterrows():
        cells = "".join(f"<td>{escape(str(row[column]))}</td>" for column in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return """
<style>
.wrapped-table table { width: 100%; table-layout: fixed; border-collapse: collapse; }
.wrapped-table th, .wrapped-table td {
  border: 1px solid rgba(128, 128, 128, 0.35);
  padding: 0.35rem;
  vertical-align: top;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  word-break: break-word;
}
.wrapped-table th { font-weight: 600; background: rgba(128, 128, 128, 0.12); }
</style>
<div class="wrapped-table"><table><thead><tr>
""" + header + "</tr></thead><tbody>" + "".join(body_rows) + "</tbody></table></div>"


def _stage_progress(stage: str) -> float:
    if stage not in STAGES:
        return 0.05
    return min(1.0, (STAGES.index(stage) + 1) / len(STAGES))


def _dot_from_weighted_edges(edges: list[dict[str, object]]) -> str:
    lines = ["graph G {", "  graph [overlap=false, splines=true];", "  node [shape=box, style=rounded];"]
    for edge in edges:
        label = (
            f"w={float(edge['weight']):.3f}\n"
            f"pos={float(edge['position_weighted']):.3f}\n"
            f"ent={float(edge['entity_weighted']):.3f}\n"
            f"content={float(edge['content_weighted']):.3f}"
        )
        lines.append(f"  {_quote(edge['source'])} -- {_quote(edge['target'])} [label={_quote(label)}];")
    lines.append("}")
    return "\n".join(lines)


def _dot_from_summary_graph(nodes: list[dict[str, object]], edges: list[dict[str, object]]) -> str:
    colors = {"chunk": "lightgray", "community": "lightblue", "final": "palegreen", "direct_llm": "palegreen"}
    lines = ["digraph G {", "  graph [rankdir=LR];", "  node [shape=box, style=filled];"]
    for node in nodes:
        color = colors.get(str(node.get("kind", "")), "white")
        lines.append(f"  {_quote(node['node_id'])} [label={_quote(node['label'])}, fillcolor={_quote(color)}];")
    for edge in edges:
        lines.append(f"  {_quote(edge['source'])} -> {_quote(edge['target'])};")
    lines.append("}")
    return "\n".join(lines)


def _quote(value: object) -> str:
    return '"' + str(value).replace('"', '\\"') + '"'


if __name__ == "__main__":
    main()
