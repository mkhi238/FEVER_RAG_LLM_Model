# Natural logic + FOL layer (branch `natlog`, 2026-11-27)

## The product
Highlight text, pick an action. Five tasks: **Define** (a term), **Explain** (a sentence / paragraph),
**Verify** (claim -> TRUE / FALSE / NOT ENOUGH INFO), **Question** (ask about the sentence), **Example** (reach goal).

## Core idea: the FOL form is the checklist
1. **Parse.** The decomposer (the local Qwen3-4B) structures the claim and writes its FOL form plus search queries:
   "Hinton is an American fantasy writer born in 1950" ->
   `writer(hinton, fantasy) ∧ nationality(hinton, American) ∧ born_in(hinton, 1950)`.
2. **Check.** Each unit (atom, or quantified subformula) becomes a short NL hypothesis; DeBERTa NLI checks it against
   the evidence -> entailed / contradicted / not found.
3. **Compose.** The logic, not DeBERTa, combines them (Kleene three-valued logic): ∧ needs every part, ¬ flips,
   → is ¬A ∨ B, quantifiers per the quantifier mode. The verdict carries the reason: which unit decided it, on which
   sentence.

Natural logic sits on top: the operators (≡ ⊑ ⊒ ¬ #) say how a unit relates to its evidence. The expected operator per
task: Define = term ⊑ genus (`∀x term(x) → genus(x) ∧ feature(x)`); Example = instance ⊑ concept; Explain = ≡ / ⊑
(each atom of the original holds in the explanation, nothing contradicts); Question = answer ≡ / ⊑ evidence, # = not
in the evidence; Verify = ≡ / ⊑ support, ¬ refute.
Prior work: ProgramFC (Pan et al. 2023), FOLK (Wang & Shu 2023), ProoFVer, QA-NatVer, FOLIO, MALLS / LogicLLaMA.

## File map
| file | what |
|---|---|
| `claimscan/neurosymbolic_layer/fol/formula.py` | AST, JSON schema (for constrained decoding), validator (`FOLError` messages feed the repair prompt), FOLIO / MALLS string parser |
| `fol/decomposition.py` | decomposer JSON (`claims[]: claim, entities, fol, queries`), `parse_output`, `fallback` (whole claim = one atom) |
| `fol/hypothesis.py` | unit -> NL hypothesis (node `text`, else a template); quantifier modes |
| `fol/check.py` | `VerifierChecker` (the Verify decision as is, plus sentence attribution) and `NatlogChecker` (spans -> align -> operators -> automaton) |
| `fol/compose.py` | Kleene composition -> `FOLVerdict(label, reason, units, deciding, formula)` |
| `fol/verify.py` | `verify_fol(...)`, `settings(cfg)`, `build_checker(...)` |
| `fol/reward.py` | GRPO reward proposal for the decomposer |
| `fol/sft_data.py` | decomposer SFT data: MALLS + FOLIO -> chat records; FEVER silver via a teacher + rejection sampling |
| `claimscan/instruct/front_decomposer.py`, `prompts.py` | decomposer: prompt + 4 few-shots + schema; pluggable client (`langchain_client`, `FakeClient`); repair loop; fallback |
| `claimscan/verifier/tasks.py` | the 5 task specs (premise, template, labels, success ops) and `TaskVerifiers` (shared base + LoRA per task, or a separate model per task) |
| `claimscan/verifier/task_data.py` | adapter training pairs per task (WordNet, PAWS, SQuAD 2, MED) |
| `claimscan/verifier/train_task_adapter.py` | LoRA (or `--separate` full) fine-tune of one task |
| `natlog/automaton.py`, `natlog/verify.py` | (earlier commits) + `table_with(forward_supports)`, `verify_claim(min_sim=, forward_supports=)` |
| `configs/default.yaml` `natlog:` | checker, quantifier_mode, min_sim, forward_supports, decomposer_attempts, per-task model / adapter / min_sim, data dirs |
| tests | `test_fol_formula.py`, `test_fol_verify.py`, `test_front_decomposer.py`, `test_task_verifiers.py`, `test_fol_sft_reward.py` |

## Decisions made (and why)
- **JSON, not FOL strings, from the model.** A JSON schema turns into a llama-server grammar, so the output always
  parses; the validator then checks the logic (bound variables, ops, sizes). FOL strings are only read for MALLS / FOLIO.
- **Every unit carries its own `text`.** Templates from predicate names read badly ("Hinton born_in 1950"), so the
  prompt asks for a plain sentence per atom / quantifier; templates are only the fallback.
- **Verify entailment logic unchanged.** `VerifierChecker` calls `pipeline.verify` and applies the same rule
  (argmax, NEI if max < 0.5 or P(NEI) >= `verifier.nei_prob_threshold`). With the fallback decomposition the label
  equals the pipeline's (tested). Sentence attribution is extra NLI passes and never changes a label.
- **Fail open.** Invalid decomposer output gets one repair turn with the validator's message
  (`natlog.decomposer_attempts`), then the fallback (whole claim = one atom).
- **One shared DeBERTa, per-task LoRA by default.** `natlog.tasks.<task>.adapter` (LoRA on the shared base, adapters
  switched per call) or `.model` (a fully separate model); null both = zero-shot with the task template. Fits
  15 GB RAM / 6 GB VRAM: one ~0.7 GB base plus a few MB per adapter.

## Done
- FOL schema, validator, parser, hypotheses, checkers, composition, reasons (tests: no models).
- Decomposer prompt + few-shot + schema-constrained client + repair + fallback (tests: fake client).
- Per-task zero-shot verifiers (templates) and adapter switching (tests: fake models, a tiny random DeBERTa + LoRA).
- Data prep for define / example / explain / question / operators; LoRA training script; decomposer SFT data prep.
- Smoke runs: see the final section.

## Smoke runs (2026-11-27; they show the code runs, they are not evals)
- Data prep (`--limit 2000`): define 1800 / 200, example 1800 / 200 (balanced 3-way), explain 2000 / 200 (no
  contradiction class), question 2000 / 200 (825 / 819 / 356), operators (MED) 1800 / 200 (no contradiction class).
- Decomposer SFT: MALLS 26,969 kept of 28,284 (1,249 dropped for variables used outside their quantifier's scope,
  49 for equality); FOLIO 2,966 kept (4,006 duplicate premises, 625 dropped, mostly equality); 28,439 train / 1,496 dev.
- Define LoRA, 200 steps, batch 8, 128 tokens, RTX 2060: 270 s, 0.44M trainable params, loss 0.75 -> 0.18.
- Real DeBERTa on hand-written evidence: the Hinton claim decomposes into 3 checks; the FOL verdict is FALSE and names
  "fantasy writer" (contradicted by "writes young adult fiction, not fantasy") and the birth year; the whole-claim
  verifier is also FALSE but only cites the year. The five zero-shot task checks give the expected label on one example each.

## Running (from the worktree)
The shared `.venv` (main checkout) has `claimscan` installed editable from the main checkout, so in this worktree use
`python -m ...` from the worktree root (or set `PYTHONPATH` to it); a plain `python script.py` imports the overhaul code.
```bash
source /mnt/d/FEVER_RAG_LLM_Model/.venv/bin/activate && source scripts/env.sh
python -m claimscan.verifier.task_data --task all --limit 2000           # -> <artifacts>/natlog/task_data/
python -m claimscan.verifier.train_task_adapter --task define --max-steps 200 --max-length 128
python -m claimscan.neurosymbolic_layer.fol.sft_data                    # -> <artifacts>/natlog/fol_sft/
python -m pytest -q tests/test_fol_*.py tests/test_front_decomposer.py tests/test_task_verifiers.py
```

## Next
1. Run the decomposer against the real Qwen3-4B on llama-server (the user's llama.cpp build) on ~20 FEVER claims and
   look at validity / repair rates.
2. Wire `verify_fol` into the natlog suite (`tests/test_natlog_suite.py`) as a second mode once the evals below are OK'd.
3. FEVER silver SFT records with a teacher (choice of teacher open), then LoRA SFT of the decomposer, then GRPO with
   `fol_reward` (after Dec 22).
4. Train the task adapters past smoke size once the dev evals below are approved.

## Proposed evals (waiting for the user's OK; none were run)
- **FOL Verify vs plain Verify:** the FEVER suite's 300 claims (same retrieval, rerank, k), label accuracy and FEVER
  score with the vendored metric, predicted evidence = `FOLVerdict.used`; plus how often the decomposer falls back.
- **Decomposer validity:** share of valid parses on first try / after repair, on 100 FEVER dev claims.
- **Per-task adapters:** 3-way accuracy and per-class recall on each task's dev.jsonl, zero-shot vs adapter.
- **Gold-evidence check:** the existing gold_evidence set (200 TRUE / FALSE claims), FOL vs single-hypothesis.

## Open questions (the user's; each is a config option with a default)
- **Quantifiers** (`natlog.quantifier_mode`, default `unit`): a forall / exists is one hypothesis checked by its text;
  `distribute` splits ∀x (R → B1 ∧ B2) into two checks (Define: genus and feature apart); `nei` never checks them.
  Also: the natlog `project()` hook (monotonicity through "all" / "no") is still a no-op.
- **MIN_SIM** (`natlog.min_sim` 0.6, per task `natlog.tasks.<task>.min_sim`): only used by the natlog checker.
- **Hinton ⊑ case** (`natlog.forward_supports`, default true): a more specific evidence span ("American fantasy writer")
  supports a more general claim span ("fantasy writer"). false = SUPPORT + ⊑ -> NEI.
- **Reward shape** (`fol/reward.py`): format bonus 0.2 + 1 for the right verdict; a one-atom decomposition scores like
  the plain verifier. Add a split bonus or an evidence-recall term?
- **Adapter labels:** WordNet siblings as contradictions and PAWS non-paraphrases as neutral are heuristics.
- **Teacher for FEVER silver FOL**: which hosted model (front_facer's?).
