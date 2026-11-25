---
name: fever-10
description: Fast 10-claim version of the FEVER claim suite (pytest tests/test_fever_suite.py --fever-claims 10, shipped hybrid retriever, FAISS + BM25 without RM3, end to end through the verifier). Checks that each claim's prediction still matches the 300-claim baseline. Use for a quick end-to-end check after a pipeline change; use /fever-300 for the full numbers.
---

# /fever-10: fast FEVER claim suite (shipped hybrid only)

A label-balanced slice of the 300-claim suite (4 TRUE, 3 FALSE, 3 NEI: the first of each label in the sample's
order), on the shipped **hybrid** retriever (FAISS + BM25 without RM3, RRF; the benchmark's `hybrid_norm3`; k=30, conf 0.5). It's the same test as `/fever-300` with
`--fever-claims 10`.

## Run (low priority, 2 threads)

```bash
cd /mnt/d/FEVER_RAG_LLM_Model && source .venv/bin/activate && source scripts/env.sh
OMP_NUM_THREADS=2 nice -n 19 python -m pytest -q -m fever_suite tests/test_fever_suite.py --fever-claims 10
```

Compute: it reuses the 300-claim sample's cached hits and corpus subset (see `/fever-300`), so the only work is
loading the verifier and running it on 10 claims on CPU. The first run ever cuts the corpus subset from the 4 GB
corpus (a few minutes); after that it takes about a minute. No GPU, no retrieval.

## Report

Read `<artifacts>/eval/fever_suite/latest_n10.json` (artifacts = `/mnt/d/crisis-claim-analysis/artifacts`):

- pytest line. Expect `1 passed, 2 skipped, 1 deselected` (the deselected one is the gold-evidence measurement, run separately), because the metric-vs-baseline tests only apply at 300 claims.
- `flips_vs_baseline`: count, and each claim id with gold / baseline / current prediction. 0 is expected, and the test fails above 1.
- `label_accuracy` and `fever_score` on the 10 claims. Too few claims to compare with the 300-claim numbers, so say so.
- `wall_clock_s`

Don't fix anything inside this skill. If it fails, offer `/fever-300` to see how big the change is.
