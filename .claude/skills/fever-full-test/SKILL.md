---
name: fever-full-test
description: 'FEVER full test: answers "how well does the whole system fact-check?". Runs real FEVER validation claims end to end through the pipeline (hybrid FAISS + BM25 retrieval -> Cohere rerank -> evidence scoring -> DeBERTa verifier -> rear aggregator, the trained Gemma adapter, whose label is scored) and scores them with the official FEVER metric: label accuracy, FEVER score, evidence and retrieval recall, and per-claim flips against the saved baseline. N claims, default 10 (~3.5 min, a quick check); "/fever-full-test 300" gives the full benchmark numbers (~10 min). Needs the artifacts on D:, so it runs locally, not in CI. Use after changing retrieval, scoring, the verifier or the config, or when asked whether accuracy or the baseline still holds.'
argument-hint: "[N claims, default 10, max 300]"
---

# /fever-full-test [N]: the FEVER claim suite (shipped hybrid only)

The 300 FEVER validation claims (seed 0, 100 per label) from `python -m claimscan.eval.benchmark --n 300`, run end to
end on the shipped **hybrid** retriever (FAISS + BM25 without RM3, RRF; benchmark name `hybrid_norm3`; k=30, conf 0.5).
N < 300 runs a label-balanced slice of the same claims (the first of each label in the sample's order).

On top of that: the reranker from the config (`rerank.backend`, Cohere: one API call per claim, 6 s apart; trial key =
1,000 calls/month) reorders the 100 fused candidates, and the **final label is the rear aggregator's**
(`--aggregator aggregator_dpo` = `<artifacts>/models/aggregator_dpo`, Gemma-3-4B 4-bit on the GPU; it can only turn
TRUE/FALSE into NEI). `--aggregator none` scores the verifier's label instead (files get `_noagg`). The aggregator
needs the GPU free (check `nvidia-smi`; ask first if training is running). Time: ~10 s/claim, so 300 claims ≈ 50 min.

| Command | Claims | Time | Checks |
|---|---|---|---|
| `/fever-full-test` | 10 (default, same N as the hook's `/unit-tests 10`) | ~3.5 min, mostly loading | per-claim predictions |
| `/fever-full-test 300` | 300 | ~10 min | per-claim predictions + label accuracy, FEVER score, retrieval metrics |

A plain `pytest` / `/unit-tests` / the Stop hook never start it (tests/conftest.py).

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
- **FAISS hits are cached per index file** (`faiss_*_c100_<index>.jsonl` when the config's index isn't the baseline's), so the first run after an index change rebuilds them (~15-20 min).
- **The real work** is the verifier on CPU (~0.5 s/claim), the Cohere call (6 s gap) and Gemma on the GPU.
- Don't run it alongside other benchmark or verifier jobs (`ps aux | grep claimscan`). If one is running, ask first.

## Report: the numbers

Read `<artifacts>/eval/fever_suite/latest_n<N>.json` (artifacts = `/mnt/d/crisis-claim-analysis/artifacts`) and give:

- pytest line
- end to end (`e2e`): `label_accuracy`, `fever_score`, `evidence_recall`, `evidence_f1`, `ms_per_claim`
- retrieval: `recall_set@5/@10/@50` and `mrr`
- next to `baseline.e2e` / `baseline.retrieval`. Those are 300-claim numbers: at N < 300 show them but say the
  sample is too small to compare.
- `flips_vs_baseline`: count, and each claim id with gold / baseline / current prediction (more than 1 fails)
- `vetoes` / `aggregator_errors` (errors = the LM reply couldn't be parsed; the verifier's label was kept)
- `wall_clock_s`
- If asked for LM samples: `<artifacts>/eval/fever_suite/agg_responses_n<N>.jsonl` (claim, gold, verifier label,
  final label, veto, llm_confidence, explanation, evidence per claim).

Baseline: `tests/data/fever_suite_hybrid_norm3_baseline.json` (re-baselined 2026-11-25 for the ready-made verifier
`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`): label accuracy 0.640, FEVER score 0.483, recall_set@10 0.660.
`--verifier <HF NLI model>` runs another verifier into its own latest_n<N>_<model>.json, without the baseline check. The test also requires `bm25.rm3: false`. The baseline is verifier-only on the old 175-shard index with no reranker,
so with the reranker/aggregator/full index on, flips are expected: report the numbers next to the baseline. Otherwise a failure means the pipeline's
behaviour changed (retrieval, evidence scoring or the verifier): say which number moved. Don't fix anything inside
this skill. If `/fever-full-test` fails, offer `/fever-full-test 300` to see how big the change is.

## Not part of /fever-full-test: verifier with gold evidence

`pytest tests/test_fever_suite.py -k gold_evidence -s` measures the verifier alone, given the gold sentences instead of
retrieved ones (~200 claims, options `--verifier`, `--gold-split`). Run it only when asked.
