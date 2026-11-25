# LIAR: tried and shelved (2026-11-13)

LIAR (Wang 2017, ~12.8k PolitiFact statements, 6 labels, no evidence) was tried as
(1) an out-of-domain eval set for the retrieve -> score -> verify pipeline and
(2) a claim pool for the research agent. It was shelved: on this system the LIAR numbers
measure label priors and noise, not fact-checking.

## Files here
| file | was | what |
|---|---|---|
| `liar.py` | `claimscan/data/liar.py` | loader for the original TSVs (the HF `UKPLab/liar` script loader is refused by `datasets` 5.x), label-map presets, LIAR-PLUS join, `load_claims_for_agent()`, `--download`, writes `liar_cleaned.csv` |
| `liar_eval.py` | `claimscan/eval/liar_eval.py` | `--evidence retrieved` (pipeline, decision logic unchanged; memory-light corpus lookup) or `--evidence justification` (LIAR-PLUS text as oracle evidence) |
| `test_liar.py` | `tests/test_liar.py` | 10 tests on synthetic TSVs (passed before archiving) |
| `liar_config.yaml` | the `liar:` section of `configs/default.yaml` | `dir`, `label_map` |

To revive: move the three .py files back, append `liar_config.yaml` to `configs/default.yaml`,
re-add `"claimscan.data.liar"` to `tests/test_imports.py`. The code imports
`claimscan.neurosymbolic_layer` (the replacement of `claimscan.evidence.scoring`).

Data (14 MB, kept): `/mnt/d/crisis-claim-analysis/artifacts/liar/` (`raw/` LIAR TSVs + zip,
`liar_plus/` LIAR-PLUS TSVs, `liar_cleaned.csv`, `eval/` per-claim results).

## Label mapping (default `binary`)
true, mostly-true -> TRUE; false, pants-fire -> FALSE; half-true, barely-true dropped (~37% of rows).
Also `binary_strict` (only true / false / pants-fire) and `three_way` (middle labels -> NOT ENOUGH INFO,
a stretch: PolitiFact means "misleading", not "no evidence").

## Results (test split, seed 0, CPU, verifier deberta_v1, conf threshold 0.5)
| mode | n | accuracy | macro-F1 | coverage (not NEI) | acc. on decided | majority baseline |
|---|---|---|---|---|---|---|
| justification (LIAR-PLUS), all rows | 200 | 0.185 | 0.257 | 0.36 | 0.514 | 0.56 |
| justification, aligned rows only | 92 | 0.337 | 0.340 | 0.67 | 0.500 | 0.543 |
| retrieved, BM25+RM3, k=30 | 50 | 0.460 | 0.448 | 0.88 | 0.523 | 0.60 |

Retrieved mode: every claim got evidence, and 96% share >= 2 content words with it, but that lexical
test is meaningless here. By hand, of the first 20 claims only 1 had evidence bearing on the claim
(#4, a DHS illegal-immigration estimate), ~5 hit the right entity/topic without the fact
(McDonnell, Putin, Perry, Tsarnaev), and the rest were noise ("46th Ohio Infantry" for
"Ohio ranks 46th in classroom spending", "Residence time" for an Atlanta commuting-cost claim).
FAISS / hybrid were not run (memory was taken by other jobs at the time).

## Why it was shelved
1. **The verifier is at chance on LIAR even with on-topic evidence.** With PolitiFact's own
   justification as evidence, accuracy on the claims it decides is 0.50, below the majority baseline.
   Any LIAR number is therefore uninformative about retrieval or the agent.
2. **The 2017 Wikipedia corpus has almost no decisive evidence** for PolitiFact claims (state budgets,
   job statistics, who-said-what). ~1 in 20 claims had evidence that bears on it.
3. **The label mapping is lossy and the semantics differ:** PolitiFact grades accuracy/misleadingness on a
   6-point scale; FEVER labels mean "Wikipedia supports / refutes / is silent". 37% of rows are dropped.
4. **LIAR-PLUS justifications are misaligned** for many rows (another statement's text): justification shares
   < 20% of the claim's content words for 87% of train, 18% of validation, 52% of test rows.
5. **As agent training claims** they have the same problem: with no evidence in the corpus and no gold
   evidence, there is no verifiable reward.

Worth revisiting only with a web/news retriever (the plan's research agent with live search) and a
verifier trained on real-world claims (e.g. VitaminC, or LIAR-style data with evidence such as
MultiFC / AVeriTeC).
