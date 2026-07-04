# Graph-Guided Extract-Support Summarization

Training-free experiment scaffold for Vietnamese and English long-/multi-document summarization.

Core variants:

- `e1`: BGE-M3 chunk-centroid to segment cosine support selection.
- `e2a`: PACSUM over all segments.
- `e2b`: PACSUM over stride-2 segments.

PACSUM defaults follow the reference CLI in `libs/PacSum/code/run.py`: `--pacsum-beta 0.0 --pacsum-lambda1 0.0 --pacsum-lambda2 1.0`. This implementation uses the original PACSUM thresholding and directional centrality rule over BGE-M3 cosine edge scores, then selects under the configured evidence token budget.

Graph weights follow:

```text
w_ij = alpha * position + beta * entity_or_factual_phrase + gamma * content
gamma = 1 - alpha - beta
alpha in {0, 0.05, 0.10, 0.15, 0.20}
beta in [0, 1-alpha], step 0.10, including 1-alpha
```

Entity/factual phrase extraction runs at chunk level. Raw chunk named entities are consolidated into canonical aliases with embedding cosine similarity using `--entity-merge-threshold` (default `0.85`), while numeric/date/money factual phrases are only exact-normalized to avoid corrupting values. Graph entity similarity uses typed binary TF-IDF cosine over the canonical phrase set. Underthesea labels and regex factual phrase types are retained for Vietnamese; English uses spaCy `en_core_web_sm`.

## Model Configuration

Runtime defaults live in `graphsum/config.py`. A root `.env` file is loaded automatically if present, environment variables override code defaults, and CLI flags override both. Use `.env.example` as the template for local endpoint settings.

Embedding settings:

```powershell
$env:GRAPHSUM_EMBEDDING_BACKEND="openai_compatible"
$env:GRAPHSUM_EMBEDDING_MODEL="BAAI/bge-m3"
$env:GRAPHSUM_EMBEDDING_BASE_URL="http://localhost:7997/v1"
$env:GRAPHSUM_EMBEDDING_API_KEY="ollama"
```

Summarization LLM settings:

```powershell
$env:GRAPHSUM_LLM_MODEL="Qwen3.6-35B"
$env:GRAPHSUM_LLM_BASE_URL="http://localhost:8000/v1"
$env:GRAPHSUM_LLM_API_KEY="ollama"
$env:GRAPHSUM_LLM_TEMPERATURE="0.0"
```

Equivalent CLI overrides:

```powershell
python scripts/run_graphsum.py --dataset vn_mds --data-root datasets --limit 5 --salience e1 --embedding-backend openai_compatible --embedding-model BAAI/bge-m3 --embedding-base-url http://localhost:7997/v1 --llm openai_compatible --model Qwen3.6-35B --base-url http://localhost:8000/v1
```

Dry-run smoke test:

```powershell
python scripts/run_graphsum.py --dataset vn_mds --data-root datasets --limit 2 --salience e1 --llm dry_run --dry-embed
```

The default chunker is LangChain Experimental `SemanticChunker` with BGE-M3 embeddings. Use `--chunking simple` only when intentionally running the explicit simple-chunking ablation.

Prompt templates live in `graphsum/prompts.py`:

- `TOPIC_EXTRACT_SUPPORT_PROMPT`
- `MERGE_EXTRACT_SUPPORT_PROMPT`

OpenAI-compatible local LLM endpoint, for example vLLM serving Qwen or Gemma:

```powershell
python scripts/run_graphsum.py --dataset vn_mds --data-root datasets --limit 5 --salience e1 --llm openai_compatible --model <model-name> --base-url http://localhost:8000/v1
```

Validation grid:

```powershell
python scripts/run_graphsum.py --dataset vn_mds --data-root datasets --limit 5 --salience e1 --grid --llm dry_run --dry-embed --output runs/grid_vn_mds.csv
```

Aggregate one or more result files:

```powershell
python scripts/summarize_results.py runs/graphsum_results.csv --output runs/summary_results.csv
```

`scripts/run_graphsum.py` also accepts `--aggregate-output <path>` to write aggregate metrics immediately after an experiment. For `--llm openai_compatible`, the runner automatically writes `<output_stem>_summary.csv` when `--aggregate-output` is not provided.

Recorded metrics include ROUGE-1, ROUGE-2, ROUGE-L, ROUGE backend, input tokens, output tokens, LLM calls, runtime seconds, chunk count, community/topic count, and the generated summary text. ROUGE is computed with the required `rouge-score` package.

`dry_run` and `--dry-embed` are for pipeline debugging only and must not be reported as paper results.

Required for real Leiden and standard ROUGE experiments:

```powershell
pip install igraph leidenalg rouge-score
```

Sequential no-graph baseline:

```powershell
python scripts/run_graphsum.py --dataset vn_mds --data-root datasets --limit 5 --salience e1 --no-graph --llm openai_compatible --model <model-name> --base-url http://localhost:8000/v1
```

Pure LLM baseline:

```powershell
python scripts/run_graphsum.py --dataset vn_mds --data-root datasets --limit 5 --pure-llm --llm openai_compatible --model <model-name> --base-url http://localhost:8000/v1 --output runs/vn_mds_pure_llm.csv
```

## Streamlit Sample Viewer

Launch the single-sample visualization app:

```powershell
streamlit run scripts/streamlit_app.py
```

The viewer runs one selected sample and shows processing progress, splitter outputs, segment evidence windows, semantic chunks, chunk entities/factual phrases, selected support, Leiden chunk communities, weighted KNN graph edges, the summary graph from chunks to communities to final summary, ROUGE scores, and the generated summary.
