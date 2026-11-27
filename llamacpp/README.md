# llama.cpp serving for ClaimScan

Local serving for the LLM roles in the ClaimScan plan. Everything lives on D:/E:, never C:.

| What | Where |
|---|---|
| llama.cpp source (tag `b11800`, 2026-10-22) | `/mnt/e/llm/llama.cpp` |
| build output (`llama-server`, `llama-quantize`, ...) | `/mnt/e/llm/llama.cpp/build/bin` |
| cmake 4.4.4 (uv tool) | `/mnt/e/llm/bin/cmake` |
| GGUF models | `/mnt/e/llm/models` |
| role -> model / port / flags | `llamacpp/roles.yaml` |
| agent action grammar | `llamacpp/grammars/research_action.gbnf` |
| Python client | `claimscan/llm/client.py` |

Status legend: **ready** = set up already. **you** = we code it together.

## Step 1: Build (ready to run, one decision first)

CUDA 12.4's `nvcc` only accepts host gcc ≤ 13, and this WSL has gcc 15, so a CUDA build
fails at configure. Pick one:

- **A. CPU build now.** It works immediately, but the 4B agent will be slow (seconds per action).
  `llamacpp/scripts/build.sh cpu`
- **B. Install a newer CUDA toolkit (12.8+ supports gcc 15).** Your driver (610.60) supports it.
  Put the toolkit on E: with the runfile installer (`--toolkitpath=/mnt/e/cuda-12.x`, no driver).
  Then run: `CUDA_HOME=/mnt/e/cuda-12.x llamacpp/scripts/build.sh cuda`
- **C. Install gcc-13 alongside (apt).** This is the smallest change, but apt installs to the WSL disk on C:.
  `CUDAHOSTCXX=g++-13 llamacpp/scripts/build.sh cuda`

The RTX 2060 is compute capability 7.5, so the script builds `-DCMAKE_CUDA_ARCHITECTURES=75` only.

## Step 2: Get models (you)

Download into `/mnt/e/llm/models`. Check each model card first; the PDF warns that the names come from roundups.
- `research_agent`: Qwen3-4B (non-thinking / instruct), Q4_K_M, about 2.5 GB.
- `reranker`: Qwen3-Reranker-0.6B, Q8_0, under 1 GB.
- `front_facer` / `decomposer`: Qwen3-30B-A3B-class MoE, Q4_K_M, about 17 GB.
  **Doesn't fit here.** It needs about 16 GB+ of free RAM with `--cpu-moe`, and this machine has 15 GB total.
  Route these two roles to a hosted API instead (`roles.yaml` `backend: hosted`).

Tip: `huggingface-cli download <repo> <file>.gguf --local-dir /mnt/e/llm/models`.

## Step 3: Sanity checks before trusting anything (you)

- [ ] **Reranker scores.** Some Qwen3-Reranker GGUFs return near-zero for everything. Score one relevant
      pair and one irrelevant pair through `/v1/rerank`. Expect about 1 vs about 0.
- [ ] **Grammar.** With `research_action.gbnf` loaded, 20 sampled actions should all parse.
- [ ] **VRAM.** The 4B Q4 agent (all layers on GPU) and the 0.6B reranker together should stay under 6 GB.
      Check with `nvidia-smi`. Never load the 4B reranker beside the 4B agent.
- [ ] **LoRA → GGUF (plan checkpoint, Oct 25).** Converting a dummy LoRA adapter with
      `convert_lora_to_gguf.py` and loading it with `--lora` should work before SFT depends on it.

## Step 4: Launch script (you)

`llamacpp/scripts/serve.sh <role>` reads `roles.yaml` and starts `llama-server` for that role.
The skeleton marks what to fill in.

## Step 5: Python client (you)

`claimscan/llm/client.py` has one client per role behind the same config, so a role can move between
local llama.cpp and a hosted API by editing `roles.yaml` only (the plan's "one config per LLM role").
The skeleton has the function signatures and docstrings.

## Useful llama-server flags

- `-m` model, `--port`, `-c` context, `-ngl 99` (all layers on GPU)
- `--grammar-file` constrains output
- `--reranking` exposes `/v1/rerank` (reranker models)
- `--lora` loads an adapter
- `--cpu-moe` / `--n-cpu-moe N` keeps MoE experts in RAM
- `--fit` sizes the model to the available memory
- OpenAI-compatible endpoints: `/v1/chat/completions`, `/v1/completions`. Health check: `/health`.
