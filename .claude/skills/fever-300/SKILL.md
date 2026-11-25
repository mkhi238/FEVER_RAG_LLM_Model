---
name: fever-300
description: Run the main 300-claim FEVER claim suite (pytest tests/test_fever_suite.py, shipped hybrid retriever, FAISS + BM25 without RM3, end to end through the verifier) and report label accuracy, FEVER score, retrieval recall and any per-claim changes against the baseline. Use when asked for the 300-claim suite, the full claim test, or to check the baseline still reproduces. For a quick check use /fever-10.
---

# /fever-300: the 300-claim FEVER suite (shipped hybrid only)

The 300 FEVER validation claims (seed 0, 100 per label) that `python -m claimscan.eval.benchmark --n 300` uses,
run as a pytest test on the shipped **hybrid** retriever only (FAISS + BM25 without RM3, RRF; the benchmark's `hybrid_norm3`; k=30, conf 0.5). It's deselected from the
normal unit tests (`-m "not fever_suite"` in pyproject), so `/tests` and the Stop hook never start it.

## Run (low priority, 2 threads)

```bash
cd /mnt/d/FEVER_RAG_LLM_Model && source .venv/bin/activate && source scripts/env.sh
OMP_NUM_THREADS=2 nice -n 19 python -m pytest -q -m fever_suite tests/test_fever_suite.py --fever-claims 300
```

Run it in the background and wait for the notification. Don't poll.

Compute, kept low on purpose:
- **No retrieval:** it reads the `faiss` and `bm25_norm3` hits cached by the benchmark (`<artifacts>/eval/cache/hits/*_367fb1deca_c100.jsonl`). The cached FAISS and BM25 hits matched a cold rerun exactly (checked 2026-11-25). Only if they're missing does it rebuild them (~15 min, FAISS reads the whole 26 GB index once).
- **Corpus:** the sentences for the hits come from a small cached subset (`<artifacts>/eval/fever_suite/corpus_*.parquet`). The first run cuts it from the 4 GB corpus (a few minutes), and later runs read it in seconds.
- **The only real work** is the verifier on CPU for 300 claims: ~1.2 s/claim, ~6 min. No GPU.
- Don't run it alongside other benchmark or verifier jobs (check `ps aux | grep claimscan`). If one is running, tell the user and ask before starting.

## Report

Read `<artifacts>/eval/fever_suite/latest_n300.json` (artifacts = `/mnt/d/crisis-claim-analysis/artifacts`):

- pytest line: `X passed, Y failed, Z skipped in T s`
- end to end: `label_accuracy`, `fever_score`, `evidence_f1` next to `baseline.e2e`
- retrieval: `recall_set@5/@10/@50` and `mrr` next to `baseline.retrieval`
- `flips_vs_baseline`: count, and for each one the claim id, gold label, baseline prediction and current prediction

Baseline: `tests/data/fever_suite_hybrid_norm3_baseline.json` (benchmark run 20261124-220309, `hybrid_norm3`):
label accuracy 0.510, FEVER score 0.380, recall_set@10 0.660. The test also requires `bm25.rm3: false` in the config. The tests allow at most 1 flipped claim. A failure means the pipeline's
behaviour changed (retrieval, evidence scoring or the verifier). Say which metric moved, and don't fix anything
inside this skill.
