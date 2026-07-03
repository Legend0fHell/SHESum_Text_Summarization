# Stage 2 Experiment Checklist

Use this checklist to fill the TODO tables in `paper/draft.md`. Do not report dry-run LLM or hash-embedding outputs as paper results.

## Evaluation Rules

- VN-MDS and ViMs each provide two human gold summaries per sample.
- Score each generated summary against both gold summaries, then select one reference per sample by maximum mean of ROUGE-2 and ROUGE-L.
- Report ROUGE-2 and ROUGE-L from the same selected reference, not independently maximized references.
- VN-MDS uses raw references only (`*.ref1.txt`, `*.ref2.txt`); tokenized references (`*.tok.txt`) are excluded.
- Result CSVs include the generated summary in `generated_summary`; keep these files for qualitative inspection and error analysis.
- Result CSVs include `reference_count`, `selected_reference_index`, and `reference_selector` for auditability.

## Required Real Runs

Run each salience variant on the Vietnamese datasets with the same LLM and endpoint:

```powershell
python scripts/run_graphsum.py --dataset vn_mds --data-root datasets --limit <N> --salience e1 --llm openai_compatible --model <model-name> --base-url <base-url> --output runs/vn_mds_e1_real.csv
python scripts/run_graphsum.py --dataset vn_mds --data-root datasets --limit <N> --salience e2a --llm openai_compatible --model <model-name> --base-url <base-url> --output runs/vn_mds_e2a_real.csv
python scripts/run_graphsum.py --dataset vn_mds --data-root datasets --limit <N> --salience e2b --llm openai_compatible --model <model-name> --base-url <base-url> --output runs/vn_mds_e2b_real.csv
```

```powershell
python scripts/run_graphsum.py --dataset vims --data-root datasets --limit <N> --salience e1 --llm openai_compatible --model <model-name> --base-url <base-url> --output runs/vims_e1_real.csv
python scripts/run_graphsum.py --dataset vims --data-root datasets --limit <N> --salience e2a --llm openai_compatible --model <model-name> --base-url <base-url> --output runs/vims_e2a_real.csv
python scripts/run_graphsum.py --dataset vims --data-root datasets --limit <N> --salience e2b --llm openai_compatible --model <model-name> --base-url <base-url> --output runs/vims_e2b_real.csv
```

## No-Graph Baseline

This baseline still uses our chunking, salience, source-support propagation, and hierarchical prompting. It only disables graph clustering.

```powershell
python scripts/run_graphsum.py --dataset vn_mds --data-root datasets --limit <N> --salience e1 --no-graph --llm openai_compatible --model <model-name> --base-url <base-url> --output runs/vn_mds_e1_no_graph_real.csv
python scripts/run_graphsum.py --dataset vims --data-root datasets --limit <N> --salience e1 --no-graph --llm openai_compatible --model <model-name> --base-url <base-url> --output runs/vims_e1_no_graph_real.csv
```

## Pure LLM Baseline

This baseline bypasses all proposed modifications: no embeddings, no chunking, no graph, no entity extraction, no evidence selection, and no hierarchical Extract-Support merge. It feeds all source documents directly to the LLM and asks for a summary.

```powershell
python scripts/run_graphsum.py --dataset vn_mds --data-root datasets --limit <N> --pure-llm --llm openai_compatible --model <model-name> --base-url <base-url> --output runs/vn_mds_pure_llm_real.csv
python scripts/run_graphsum.py --dataset vims --data-root datasets --limit <N> --pure-llm --llm openai_compatible --model <model-name> --base-url <base-url> --output runs/vims_pure_llm_real.csv
```

Pure LLM runs may fail or truncate externally if a sample exceeds the served model context window. Record any failures and keep the same sample set when comparing against GraphSum runs.

## Graph-Weight Grid

Start with one dataset and one salience method before scaling:

```powershell
python scripts/run_graphsum.py --dataset vn_mds --data-root datasets --limit <N> --salience e1 --grid --llm openai_compatible --model <model-name> --base-url <base-url> --output runs/vn_mds_e1_grid_real.csv
```

## Aggregation

```powershell
python scripts/summarize_results.py runs/vn_mds_e1_real.csv runs/vn_mds_e2a_real.csv runs/vn_mds_e2b_real.csv runs/vn_mds_e1_no_graph_real.csv runs/vn_mds_pure_llm_real.csv --output runs/vn_mds_summary_real.csv
```

## Before Stage 2.5 Integrity

- Replace all `TODO` result cells in `paper/draft.md`.
- Verify citation metadata for every placeholder in `References to Verify`.
- Confirm dataset versions, splits, sample counts, and licenses.
- Confirm LLM model names, serving configuration, temperature, and decoding settings.
- Cross-check ROUGE with a standard implementation or document the exact local implementation.
- Inspect `generated_summary` examples for at least a few GraphSum, no-graph, and pure LLM runs.
- Confirm `en_core_web_sm` is installed before reporting Multi-News entity-aware results.
