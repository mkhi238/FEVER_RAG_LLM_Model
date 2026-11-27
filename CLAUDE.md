# ClaimScan / FEVER_RAG_LLM_Model: notes for Claude

## What this is
A FEVER fact-checker (FAISS bge-small retrieval → heuristic evidence scoring → DeBERTa-v3 verifier)
being turned into **ClaimScan (RAG + RL)**. Plan PDF: `claimscan-plan-summary.pdf` in the repo root (untracked, not committed; dated Oct 12, 2026, so some parts are stale).
The plan: front facer → LangGraph per-claim loop (decomposer = research agent, one local Qwen3-4B on llama.cpp:
splits claims, writes queries, decides search-again/stop → retrieve → rerank → DeBERTa verify, max 3 hops; decided
2026-11-25). Only that Qwen3-4B gets trained (LoRA SFT, then GRPO after Dec 22). The verifier is frozen and used as the reward. Branches (2026-11-27): `main` = shipped; `ship` (GitHub only, PR to main: README, /readme skill, docs/architecture LaTeX); `overhaul` = main checkout (cleanup + new plan, `docs/OVERHAUL.md`); `natlog` = natural-logic layer, worktree `../FEVER_RAG_LLM_Model-natlog`. `refactor` was deleted 2026-11-27 (all its commits are in `overhaul`).

## Storage rule (hard): nothing on C:
- Repo: `/mnt/d/FEVER_RAG_LLM_Model` (the WSL home `~` is a disk image ON C:, so never put project files under `~`)
- Artifacts (data, FAISS, verifier): `/mnt/d/crisis-claim-analysis/artifacts` (`configs/default.yaml` → `artifacts_root`)
- llama.cpp source (tag b11800) + cmake + GGUF models: `/mnt/e/llm`
- venv: `.venv` (self-contained Python 3.12, torch 2.14.1+cu126; nothing borrowed from `~/.local`)
- Caches/temp: `source scripts/env.sh` (uv, pip, torch caches + TMPDIR on D:, tools on E:)
- D: has ~348 GB free, so new artifacts go on D:. E: is for llama.cpp/models.

## Run
```bash
source .venv/bin/activate && source scripts/env.sh
python -m pytest -q             # slow: D: is WSL drvfs
```
- pytest uses `--capture=sys` (pyproject). The default tmpfile capture breaks on drvfs.
- Skills: `/unit-tests [N|file|-k]` = is the code correct (pass/fail); `/fever-full-test [N]` = how accurate is the system (FEVER metric vs baseline).
- CI: `.github/workflows/unit-tests.yml` runs the unit tests on every push (no artifacts, offline HF), so artifact tests must skip themselves.
- Machine: 15 GB RAM, RTX 2060 6 GB (sm_75), 6 cores. Design for that.
- `HF_HOME` in the user's shell is `/mnt/d/whisper_v2/hf_cache` (on D:, fine; it wins over `hf_home` in config).
- Git remote is SSH and works: `git push` is fine when the user asks.

## Serving + graphs (2026-11-24)
- API serves `pipeline.fact_check_detailed` (same logic as `fact_check_cleaned`, k=30, 0.5), so benchmark numbers = API behaviour. Keep it that way.
- All FEVER data is in one SQLite file, `<artifacts>/data/processed/fever_sentences.sqlite` (tables sentences, claims, evidence;
  `claimscan/data/fever_db.py` builds and reads it; moved off parquet 2026-11-25). Pipeline, eval and API
  look sentences up there, never loading the 4 GB corpus. `docker/README.md` for the image (artifacts + HF cache mounted read-only).
- `graphs/` is generated: `/graphs` skill or `python -m claimscan.graph.make_graphs`; never hand-edit it. A Stop hook (`.claude/hooks/graphs_if_stale.sh`) rebuilds it in the background when an eval result is newer.
- Publishing: only working end-to-end parts go to `main` (merged from a separate `ship` branch built in its own worktree,
  never by switching branches in the shared checkout). Shipped 2026-11-25: API + SQLite store + Docker, hybrid retriever,
  rear aggregator (`aggregation.py`, `instruct/{schemas,chat_clients,rear_aggregator}.py`, off by default, fails open), graph, `.claude/`.
  Shipped 2026-11-26: Cohere reranker, full FAISS index. Still `overhaul`-only (not shipped): agents, llm, serve, `instruct/{front_decomposer,prompts}.py`, most of `llamacpp/`, `scripts/e2e_cpu_eval.py`.
- Shipped retriever default: `retriever.kind: hybrid` with `bm25.rm3: false` (benchmark name `hybrid_norm3`). 300 claims, k=30:
  label acc 0.64, FEVER 0.48 (with the ready-made verifier, 2026-11-25; 0.51 / 0.38 with the old deberta_v1). The FEVER claim suite (`/fever-full-test` for 10 claims, `/fever-full-test 300` for the full numbers) checks it per claim. The user's reference number is
  64.5% (an N=30 trial, not saved); the recall/testing method is being redone.
- 2026-11-26 `/fever-full-test 300` (full index + Cohere rerank + trained Gemma DPO aggregator; the suite now scores the aggregator's label):
  verifier label acc 0.65 / FEVER 0.467; aggregator 0.463 / 0.373 (122 vetoes, 84 of them on correct verdicts). The DPO adapter
  copies its SFT targets (llm_confidence always 0.95/0.05, so the >= 0.9 gate never filters). Replies: `<artifacts>/eval/fever_suite/agg_responses_n300.jsonl`.
  Why 0.65 (same run): per gold label the verifier gets TRUE 91%, FALSE 76%, NEI 28%; 72 of the 105 errors are NEI claims called
  TRUE/FALSE (50 with conf >= 0.9), and in samples several are really supported by the retrieved sentences (FEVER NEI label noise).
  NEI rule (default since 2026-11-26): `verifier.nei_prob_threshold: 0.2` (P(NEI) >= 0.2 -> NEI) gives 0.670 on the same run
  (chosen on these 300 claims; median P(NEI) is only 0.08 on gold-NEI claims). User's view: the rest is data (retriever + verifier); scale up at the hackathon.
  Retrieval isn't the limit: full gold set sent for 131/200 TRUE/FALSE claims, verifier right on 85% of those.
  The full index is slow cold through the API (30-110 s per FAISS query from D:); the eval's batch search is fine.
- Rear aggregator veto: OFF since 2026-11-26 (`rear_aggregator.VETO_ENABLED = False`; the LLM only explains). When on, the gate in
  `rear_aggregator.final_label` is verifier conf < 0.6 or LLM conf >= 0.9. Gate tests switch it on with the `veto_on` fixture.
  Hosted LLM = Together AI (`ChatTogether` in `aggregation.build_llm`): setting `aggregator.api_key_env` (the env var NAME, never the key) switches to it; local llama-server goes through `instruct/chat_clients.build_chat`.
- Future direction (settled): a paper-reading helper. Highlight a term, get a grounded 1-2 sentence RAG definition.

## Working style
- The user writes the **llama.cpp parts themselves** (`llamacpp/`, `claimscan/llm/`) with Claude reviewing/assisting.
  Don't implement those unless asked. Walkthrough: `llamacpp/README.md`.
- The user wants plans shown before big changes. Keep retrieval/scoring/decision logic unchanged unless asked.
- Commit only files you changed (the user may have uncommitted edits).

## Open jobs
1. **Hybrid BM25 + FAISS retriever (RRF k=60): code done and Lucene index built (2026-11-12).**
   - Lucene index BUILT 2026-11-12: 25,246,092 docs, 4.3 GB, 23 min (`-Xmx2g`, `--memory-buffer 512`; the 4 GB default OOMs).
     Cold BM25+RM3 queries take 6–11 s. DECIDED 2026-11-25: RM3 off (better and ~17x faster), default `hybrid`.
   - The stored bge FAISS index is **L2** (lower = better). Hybrid `score` = 1/rrf to match. Evidence scoring now uses rank, not score
     (`claimscan/neurosymbolic_layer/`, which replaced `evidence/scoring.py` on 2026-11-12).
     But `faiss_indexer.py` builds `IndexFlatIP`, so a rebuild would flip the score direction.
2. Learning notes moved out of the code to `notes/ml_notes.txt`; ask before committing `notes/`.
3. **Per-file check:** make every file run, be efficient for 15 GB RAM / 6 GB VRAM, and be clear. Known issues:
   - The 25.2M FAISS docid list (~2.5 GB) is loaded into RAM (peak RSS ~7 GB with the full index). (The corpus is no longer loaded: SQLite lookups, 2026-11-25.)
4. **Next (when the user's RL training is off the machine):** `python -m claimscan.eval.eval_retriever --retriever faiss bm25 hybrid`,
   then `python -m claimscan.eval.fever_eval --max-claims 200`.
   `claimscan/instruct/` (front decomposer / rear aggregator, LangChain) is the user's; help only if asked.
   Decomposer = the research agent (local Qwen3-4B, RL-trained): splits claims, writes queries, decides search-again/stop.

## Eval (`claimscan/eval/`, 2026-11-12)
- `metrics/fever/` = the user's FEVER metric from huggingface/evaluate PR #710 (merged), vendored unchanged and loaded with
  `evaluate.load(<dir>)`. The adapter adds the official conventions: Page#line ids, first 5 predicted sentences, NEI gold = [[]].
  The later RLVR reward for the agent should import from here (per-claim compute() gives a 0/1 FEVER score).

## Known data/model issues (for the plan)
- **Default verifier (2026-11-25): `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`** (`verifier.inference_model`,
  `claimscan/verifier/hf_nli.py`, now the only verifier; the old self-trained `deberta_v1` code was removed on `overhaul`). With gold evidence on the suite's 200 TRUE/FALSE
  claims: ready-made 82%, old 57% (`pytest tests/test_fever_suite.py -k gold_evidence`). Leak check: 88% on FEVER train claims
  it saw vs 82% on our dev claims, so no sign of leakage (FEVER-NLI trains on FEVER train only). Weak spot: NEI recall 39%.
  The user decided not to fine-tune it for now; options if needed: confidence cut-off tuning, or fine-tune on NEI (VitaminC).
- **FEVER line 0 was stored as -1** (HF loader, falsy 0): 46% of gold pairs. The SQLite evidence table stores 0 now, but the
  old `deberta_v1` verifier was trained on data from before the fix.
- FAISS: `faiss_index_full.bin` + `.ivfdata` (on-disk, 39 GB) has all 253 shards (25.2M), built 2026-11-26 by
  `retrieval/faiss_extend.py` from the old `faiss_index_175.bin` (17.1M, shards 4–174), which is kept.
- The old verifier's 94–95% figure came from the old split: its validation set was all 1,500 NEI examples, so TRUE/FALSE was never checked.
- The llama.cpp CUDA build is blocked: CUDA 12.4 nvcc needs gcc ≤ 13, and WSL has gcc 15 (options in `llamacpp/README.md`).
- The big front-facer MoE won't fit in 15 GB RAM, so route front_facer to a hosted API (`llamacpp/roles.yaml`). The decomposer is now the local Qwen3-4B
  (2026-11-25); `roles.yaml` still has decomposer sharing front_facer, which is the user's to change.

## User to-dos (not Claude's)
- Make the cache settings permanent: `cat scripts/env.sh >> ~/.bashrc`
- Free C: (96% full): PowerShell `wsl --shutdown` then `wsl --manage Ubuntu --move E:\WSL\Ubuntu`
- Fine-tuning dataset shortlist (WordNet 2023, WiC, VitaminC, Search-R1 data, WiCE, ...) was given in chat; not saved yet.
