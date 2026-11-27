---
name: graphs
description: Regenerate the latest ClaimScan graphs (retriever benchmark, recall@k curves, end-to-end confusion matrices, verifier training curves) from results already on disk into the clean repo-root graphs/ folder, and summarize what they show. Use when asked for graphs, charts, plots, or to refresh/update the figures.
---

# /graphs: rebuild the graphs/ folder

Draw every graph we have data for into `graphs/` at the repo root. Nothing is retrained,
re-retrieved or re-evaluated; it only reads finished results. Takes under a minute.

## Run

```bash
cd /mnt/d/FEVER_RAG_LLM_Model && source .venv/bin/activate && source scripts/env.sh
nice -n 19 python -m claimscan.graph.make_graphs
```

`claimscan/graph/make_graphs.py` owns the folder. It deletes the old `*.png` and `README.md` in `graphs/`,
writes each graph whose source exists, and rewrites `graphs/README.md` (each image, what it shows,
which run it came from, and a "Not generated" list with the reason for each skip).
Don't hand-edit or add ad-hoc files in `graphs/`; add a new graph as a `graph_*` function in that script.

| file | source |
|---|---|
| `benchmark.png` | latest finished `<artifacts>/eval/runs/<ts>/summary.json` |
| `retriever_recall_set.png`, `retriever_hit.png`, `retriever_recall.png` | benchmark cached hits (`<artifacts>/eval/cache`) |
| `confusion_matrices.png` | latest run's `predictions.parquet` |
| `verifier_training_curves.png` | verifier output dir, else `archive/verifier_training_curves.png` |

## Rules
- **Don't start the benchmark or any eval to fill a gap.** They take hours of CPU and the user may be training.
  If a graph is skipped because a source is missing, report the skip and the command that would produce it
  (`python -m claimscan.eval.benchmark --n 300`), and let the user decide.
- Never use the GPU; keep `nice -n 19`.
- If a graph fails, the script logs the traceback and keeps going. Report the failure with the error.

## Report back
1. Made / skipped list (from the script's output).
2. Open each new PNG with Read and look at it: check for clipped labels or overlapping text before calling it done.
3. 2-4 lines per graph on what it shows, using the numbers in `graphs/README.md`. Say which run it came from.
   Note small-sample noise (300 claims is ±~5 pts on accuracy) and the known caveat that the verifier
   curves come from an unshuffled split.
