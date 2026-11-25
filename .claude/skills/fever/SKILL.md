---
name: fever
description: Run the FEVER claim suite (pytest tests/test_fever_suite.py) on N claims, default 10 (the 300-claim max gives the full numbers), on the shipped hybrid retriever (FAISS + BM25 without RM3) end to end through the verifier, and report label accuracy, FEVER score, evidence/retrieval recall and per-claim changes against the baseline. Use after a pipeline change ("/fever" quick, "/fever 300" full), or when asked for the claim suite or whether the baseline still reproduces.
argument-hint: "[N claims, default 10, max 300]"
---

# /fever [N]: the FEVER claim suite (shipped hybrid only)

The 300 FEVER validation claims (seed 0, 100 per label) from `python -m claimscan.eval.benchmark --n 300`, run end to
end on the shipped **hybrid** retriever (FAISS + BM25 without RM3, RRF; benchmark name `hybrid_norm3`; k=30, conf 0.5).
N < 300 runs a label-balanced slice of the same claims (the first of each label in the sample's order).

| Command | Claims | Time | Checks |
|---|---|---|---|
| `/fever` | 10 (default, same N as the hook's `/tests 10`) | ~3.5 min, mostly loading | per-claim predictions |
| `/fever 300` | 300 | ~10 min | per-claim predictions + label accuracy, FEVER score, retrieval metrics |

A plain `pytest` / `/tests` / the Stop hook never start it (tests/conftest.py).

## Run (low priority, 2 threads, in the background)

```bash
cd /mnt/d/FEVER_RAG_LLM_Model && source .venv/bin/activate && source scripts/env.sh
OMP_NUM_THREADS=2 nice -n 19 python -m pytest -q tests/test_fever_suite.py --fever-claims N
```

Wait for the notification; don't poll. Expect `1 passed, 1 deselected` (the deselected one is the gold-evidence
measurement below).

Compute, kept low on purpose:
- **No retrieval:** it reads the `faiss` and `bm25_norm3` hits cached by the benchmark (`<artifacts>/eval/cache/hits/*_367fb1deca_c100.jsonl`).
  Only if they're missing does it rebuild them (~15 min, FAISS reads the whole 26 GB index once).
- **Corpus:** the hits' sentences come from a small cached subset in `<artifacts>/eval/fever_suite/`.
- **The real work** is the verifier on CPU: ~1.3 s/claim. No GPU.
- Don't run it alongside other benchmark or verifier jobs (`ps aux | grep claimscan`). If one is running, ask first.

## Report: the numbers

Read `<artifacts>/eval/fever_suite/latest_n<N>.json` (artifacts = `/mnt/d/crisis-claim-analysis/artifacts`) and give:

- pytest line
- end to end (`e2e`): `label_accuracy`, `fever_score`, `evidence_recall`, `evidence_f1`, `ms_per_claim`
- retrieval: `recall_set@5/@10/@50` and `mrr`
- next to `baseline.e2e` / `baseline.retrieval`. Those are 300-claim numbers: at N < 300 show them but say the
  sample is too small to compare.
- `flips_vs_baseline`: count, and each claim id with gold / baseline / current prediction (more than 1 fails)
- `wall_clock_s`

Baseline: `tests/data/fever_suite_hybrid_norm3_baseline.json` (benchmark run 20261124-220309): label accuracy 0.510,
FEVER score 0.380, recall_set@10 0.660. The test also requires `bm25.rm3: false`. A failure means the pipeline's
behaviour changed (retrieval, evidence scoring or the verifier): say which number moved. Don't fix anything inside
this skill. If `/fever` fails, offer `/fever 300` to see how big the change is.

## Not part of /fever: verifier with gold evidence

`pytest tests/test_fever_suite.py -k gold_evidence -s` measures the verifier alone, given the gold sentences instead of
retrieved ones (~200 claims, options `--verifier`, `--gold-split`). Run it only when asked.
