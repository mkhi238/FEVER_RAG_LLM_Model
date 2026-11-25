# ClaimScan / FEVER_RAG_LLM_Model: notes for Claude

## What this is
A FEVER fact-checker (FAISS bge-small retrieval → heuristic evidence scoring → DeBERTa-v3 verifier)
being turned into **ClaimScan (RAG + RL)**. Plan PDF: `claimscan-plan-summary.pdf` in the repo root (untracked, not committed; dated Oct 12, 2026, so some parts are stale).
The plan: front facer + decomposer LLM → LangGraph per-claim loop (research agent Qwen3-4B on llama.cpp
→ retrieve → rerank → DeBERTa verify, max 3 hops). Only the research agent gets trained (LoRA SFT, then GRPO
after Dec 22). The verifier is frozen and used as the reward. Branch `refactor` = setup/cleanup only; no plan features are built yet.

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
python scripts/verifier_smoke.py
```
- pytest uses `--capture=sys` (pyproject). The default tmpfile capture breaks on drvfs.
- Machine: 15 GB RAM, RTX 2060 6 GB (sm_75), 6 cores. Design for that.
- `HF_HOME` in the user's shell is `/mnt/d/whisper_v2/hf_cache` (on D:, fine; it wins over `hf_home` in config).
- Git remote is SSH and works: `git push` is fine when the user asks.

## Serving + graphs (2026-11-24)
- API serves `pipeline.fact_check_detailed` (same logic as `fact_check_cleaned`, k=30, 0.5), so benchmark numbers = API behaviour. Keep it that way.
- API reads sentence text from the SQLite store (`python -m claimscan.evidence.store build`), never the 4 GB corpus. `docker/README.md` for the image (artifacts + HF cache mounted read-only).
- `graphs/` is generated: `/graphs` skill or `python -m claimscan.graph.make_graphs`; never hand-edit it. A Stop hook (`.claude/hooks/graphs_if_stale.sh`) rebuilds it in the background when an eval result is newer.
- Publishing: only working end-to-end parts go to `main` (merged from a separate `ship` branch built in its own worktree,
  never by switching branches in the shared checkout). Shipped 2026-11-25: API + SQLite store + Docker, hybrid retriever,
  rear aggregator (`aggregation.py`, `instruct/{schemas,chat_clients,rear_aggregator}.py`, off by default, fails open), graph, `.claude/`.
  Still `refactor`-only: agents, rerank, llm, serve, `instruct/{front_decomposer,prompts}.py`, most of `llamacpp/`, `scripts/e2e_cpu_eval.py`.
- Shipped retriever default: `retriever.kind: hybrid` with `bm25.rm3: false` (benchmark name `hybrid_norm3`). 300 claims, k=30:
  label acc 0.51, FEVER 0.38. The FEVER claim suite (`/fever` for 10 claims, `/fever 300` for the full numbers) checks it per claim. The user's reference number is
  64.5% (an N=30 trial, not saved); the recall/testing method is being redone.
- Rear aggregator veto: fixed in `rear_aggregator.final_label` (verifier conf < 0.6 or LLM conf >= 0.9), not configurable.
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
   - `verifier/prepare.py` scans all 25M corpus rows per claim (75k claims); use `evidence.select.index_corpus`.
   - Loads the full corpus (~4 GB) and the 17.1M docid list (~1.5 GB) into RAM.
4. **Next (when the user's RL training is off the machine):** `python -m claimscan.eval.eval_retriever --retriever faiss bm25 hybrid`,
   then `python -m claimscan.eval.fever_eval --max-claims 200`; then fix line-0 in `verifier/prepare.py` and retrain.
   `claimscan/instruct/` (front decomposer / rear aggregator, LangChain) is the user's; help only if asked.
   Front decomposer: keep it to splitting text into claims; query writing is the RL-trained research agent's job.

## Eval (`claimscan/eval/`, 2026-11-12)
- `metrics/fever/` = the user's FEVER metric from huggingface/evaluate PR #710 (merged), vendored unchanged and loaded with
  `evaluate.load(<dir>)`. The adapter adds the official conventions: Page#line ids, first 5 predicted sentences, NEI gold = [[]].
  The later RLVR reward for the agent should import from here (per-claim compute() gives a 0/1 FEVER score).

## Known data/model issues (for the plan)
- **FEVER line 0 is stored as -1** (HF loader, falsy 0) in the evidence parquet and fever_gold.json: 46% of gold pairs.
  `eval/gold.py` fixes it; `verifier/prepare.py` does NOT, so verifier training lost that evidence (retrain candidate).
- FAISS indexes cover only shards 0–174 of 253 (17.1M of 25.2M sentences). The corpus is sorted by title
  (bge index ends at `Panel_of_Chairs#1`), and the training shards 0–3 were never added.
- The verifier's 94–95% figure came from an unshuffled split. It says NEI 0.995 on "Obama was born in Hawaii".
- The llama.cpp CUDA build is blocked: CUDA 12.4 nvcc needs gcc ≤ 13, and WSL has gcc 15 (options in `llamacpp/README.md`).
- The big front-facer MoE won't fit in 15 GB RAM, so route front_facer/decomposer to a hosted API (`llamacpp/roles.yaml`).

## User to-dos (not Claude's)
- Make the cache settings permanent: `cat scripts/env.sh >> ~/.bashrc`
- Free C: (96% full): PowerShell `wsl --shutdown` then `wsl --manage Ubuntu --move E:\WSL\Ubuntu`
- Fine-tuning dataset shortlist (WordNet 2023, WiC, VitaminC, Search-R1 data, WiCE, ...) was given in chat; not saved yet.
