# ClaimScan overhaul (branch `overhaul`, started 2026-11-26)

## Goal
Move from the single-pass FEVER fact-checker to the ClaimScan pipeline, with **semantics and logic** at its core
instead of a black-box TRUE/FALSE/NEI verifier.

## Pipeline
```
prompt
 → Qwen3-4B decomposer   (the "shepherd")   claim → FOL conjunction of predicates (FOLK-style)
 → retriever             hybrid FAISS + BM25 over the sentence store (the facts live here)
 → reranker              Cohere (no DeBERTa here)
 → natural-logic layer   DeBERTa + a fixed logic table:
                           - which retrieved sentences matter (aligned to claim spans)
                           - is the evidence enough? (if not: change k or search again, max 3 loops)
                           - verdict per predicate, AND across predicates
 → Gemma-3-4b rear       writes the response (and can show the proof)
```
A standard RAG loop: scaling up = more text in the store + more training pairs.

## The two core ideas
### 1. Natural logic, layered with FOL (Claude builds, user supervises)
- **FOL layer:** the decomposer turns a claim into `P1 ∧ P2 ∧ ...`. TRUE only if every predicate holds, FALSE if any is refuted.
  Until the Qwen decomposer exists, the whole claim is one predicate.
- **Natural-logic layer**, per predicate (ProoFVer / QA-NatVer):
  1. split the predicate into short spans (who / did what / to whom / when);
  2. align each span to the best-matching evidence span (bge-small embeddings, already loaded);
  3. tag each aligned pair with an operator: ≡ same, ⊑ more specific, ⊒ more general, ¬ negation, ⇃ alternative, # unrelated
     (the existing DeBERTa NLI model zero-shot first, an SFT'd DeBERTa later);
  4. a fixed automaton (ProoFVer's table) turns the operator sequence into SUPPORT / REFUTE / NEI.
- **Outputs:** label, the proof (pairs + operators), the sentences that matter, and whether the evidence is enough.
- Upgrades the current neurosymbolic layer (keyword rules) to a real semantic one, and is explainable by construction.

### 2. RL librarian (user builds, Claude assists)
- Qwen3-4B writes the search queries for each predicate. Queries can be **natural-logic edits** of the claim
  (broader word, synonym, negation), i.e. inference as search (NaturalLI).
- Reward = gold-evidence recall (FEVER gold sets, vendored metric). The bet: by being rewarded for what finds evidence, it
  learns real word meaning (which edits preserve meaning, which don't).
- SFT first, GRPO at the hackathon (more compute). Prior work: DeepRetrieval, Search-R1, Nogueira & Cho 2017.
- Needs first: llama.cpp CUDA build, and a fast CPU index (IVF-PQ in RAM, <1 s/query) since RL calls retrieval thousands of times.

## Hardware (RTX 2060 6 GB, ~5.1 GB usable; 15 GB RAM)
- Qwen3-4B Q4 on the GPU (~3.3 GB with a 4k context). Gemma-3-4b split GPU/CPU (it runs once per request).
- DeBERTa and bge-small on the CPU. Reranker hosted (Cohere).
- Cold FAISS queries on the current on-disk IVF-Flat index take 30–110 s. The IVF-PQ CPU index fixes that.

## Build order
1. Clean up this branch (remove the old verifier path and the metrics we no longer need), see the list agreed in chat.
2. Write the new unit tests and the full-model test (natlog suite) first.
3. Build the natural-logic layer step by step: automaton → spans → align → operators → verify → config switch.
4. Librarian prerequisites (CPU index, CUDA build), then the librarian itself.

## Sources
ProoFVer https://arxiv.org/abs/2108.11357 · QA-NatVer https://arxiv.org/abs/2310.14198 ·
FOLK https://arxiv.org/abs/2310.05253 · NaturalLI https://aclanthology.org/D14-1059/ ·
DeepRetrieval https://arxiv.org/abs/2503.00223 · Search-R1 https://arxiv.org/abs/2503.09516 ·
Nogueira & Cho 2017 https://arxiv.org/abs/1704.04572 · Sufficient context https://arxiv.org/abs/2411.06037
