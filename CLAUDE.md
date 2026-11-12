# ClaimScan / FEVER_RAG_LLM_Model: notes for Claude

## What this is
A FEVER fact-checker (FAISS bge-small retrieval → heuristic evidence scoring → DeBERTa-v3 verifier)
being turned into **ClaimScan (RAG + RL)**. Plan PDF: `C:\Users\mukun\Downloads\claimscan-plan-summary.pdf`.
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
python -m pytest -q             # 41 tests, ~2.5 min (D: is slow over WSL drvfs)
python scripts/verifier_smoke.py
```
- pytest uses `--capture=sys` (pyproject). The default tmpfile capture breaks on drvfs.
- Machine: 15 GB RAM, RTX 2060 6 GB (sm_75), 6 cores. Design for that.
- `HF_HOME` in the user's shell is `/mnt/d/whisper_v2/hf_cache` (on D:, fine; it wins over `hf_home` in config).
- Git remote is SSH and works: `git push` is fine when the user asks.

## Working style
- The user writes the **llama.cpp parts themselves** (`llamacpp/`, `claimscan/llm/`) with Claude reviewing/assisting.
  Don't implement those unless asked. Walkthrough: `llamacpp/README.md`.
- The user wants plans shown before big changes. Keep retrieval/scoring/decision logic unchanged unless asked.
- Commit only files you changed (the user may have uncommitted edits).

## Open jobs (requested, not started)
1. **Hybrid BM25 + FAISS retriever:** reuse existing code, don't rewrite.
   - BM25: `archive/bm25_sentence_retriever.py` (pyserini/Lucene, returns `{id,line,score}` like FAISS) and
     `archive/bm25_page_constructor.py` (sentence JSONL with title boost; fix its C: paths).
   - The RRF hybrid (k=60) exists only on C: at `C:\Users\mukun\crisis-claim-analysis\src\rag\archive\hybrid_retriever\`.
     It's unfinished (imports a missing `fever_v2.bm25_retriever`; last line is a broken `np.argsort`). Copy and finish it.
   - **The Lucene index no longer exists**, so it has to be rebuilt: pyserini + Java 21 (installed), 25.2M-sentence JSONL (~5 GB)
     + index (~5 GB), 1–3 h. Put it on D:. Ask the user before starting the build.
   - FAISS works locally with `faiss.read_index(path, faiss.IO_FLAG_MMAP)`: 0.85 GB RAM, ~0.1 s warm query,
     ~25 s first query. Add this to `claimscan/retrieval/faiss_retriever.py`.
   - Compare BM25 / FAISS / hybrid with `claimscan/retrieval/eval_recall.py` (recall@k).
2. **Notes folder:** move the user's long learning comments/docstrings out of the code into `notes/`
   (one markdown per topic, each linking to the code it explains), then leave short, clear comments in the code.
   Ask whether `notes/` should be committed or kept outside the repo.
3. **Per-file check:** make every file run, be efficient for 15 GB RAM / 6 GB VRAM, and be clear. Known issues:
   - `evidence/select.py` `get_evidence` scans all 25M corpus rows per hit → index the corpus by (id, line).
   - `verifier/prepare.py` does the same full scan per claim (75k claims).
   - Loads the full corpus (~4 GB) and the 17.1M docid list (~1.5 GB) into RAM.

## Known data/model issues (for the plan)
- FAISS indexes cover only shards 0–174 of 253 (17.1M of 25.2M sentences). The corpus is sorted by title
  (bge index ends at `Panel_of_Chairs#1`), and the training shards 0–3 were never added.
- The verifier's 94–95% figure came from an unshuffled split. It says NEI 0.995 on "Obama was born in Hawaii".
- The llama.cpp CUDA build is blocked: CUDA 12.4 nvcc needs gcc ≤ 13, and WSL has gcc 15 (options in `llamacpp/README.md`).
- The big front-facer MoE won't fit in 15 GB RAM, so route front_facer/decomposer to a hosted API (`llamacpp/roles.yaml`).

## User to-dos (not Claude's)
- Make the cache settings permanent: `cat scripts/env.sh >> ~/.bashrc`
- Free C: (96% full): PowerShell `wsl --shutdown` then `wsl --manage Ubuntu --move E:\WSL\Ubuntu`
- Fine-tuning dataset shortlist (WordNet 2023, WiC, VitaminC, Search-R1 data, WiCE, ...) was given in chat; not saved yet.
