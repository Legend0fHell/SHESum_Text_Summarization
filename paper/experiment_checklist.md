# Stage 2 Experiment Checklist

Use this checklist to fill the TODO tables in `paper/draft.md`. Do not report dry-run LLM or hash-embedding outputs as paper results.

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

```powershell
python scripts/run_graphsum.py --dataset vn_mds --data-root datasets --limit <N> --salience e1 --no-graph --llm openai_compatible --model <model-name> --base-url <base-url> --output runs/vn_mds_e1_no_graph_real.csv
python scripts/run_graphsum.py --dataset vims --data-root datasets --limit <N> --salience e1 --no-graph --llm openai_compatible --model <model-name> --base-url <base-url> --output runs/vims_e1_no_graph_real.csv
```

## Graph-Weight Grid

Start with one dataset and one salience method before scaling:

```powershell
python scripts/run_graphsum.py --dataset vn_mds --data-root datasets --limit <N> --salience e1 --grid --llm openai_compatible --model <model-name> --base-url <base-url> --output runs/vn_mds_e1_grid_real.csv
```

## Aggregation

```powershell
python scripts/summarize_results.py runs/vn_mds_e1_real.csv runs/vn_mds_e2a_real.csv runs/vn_mds_e2b_real.csv runs/vn_mds_e1_no_graph_real.csv --output runs/vn_mds_summary_real.csv
```

## Before Stage 2.5 Integrity

- Replace all `TODO` result cells in `paper/draft.md`.
- Verify citation metadata for every placeholder in `References to Verify`.
- Confirm dataset versions, splits, sample counts, and licenses.
- Confirm LLM model names, serving configuration, temperature, and decoding settings.
- Cross-check ROUGE with a standard implementation or document the exact local implementation.
- Confirm `en_core_web_sm` is installed before reporting Multi-News entity-aware results.
