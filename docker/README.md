# Running the API in Docker

The image holds code and Python packages only (CPU torch, plus Java 21 + pyserini for BM25). Everything large is
mounted read-only at run time.

**Retriever:** the shipped default is `hybrid` (FAISS + BM25 without RM3, fused with RRF; `retriever.kind` /
`bm25.rm3` in `configs/default.yaml`). It needs about 2.5 GB more RAM than FAISS alone for BM25's JVM.
`CLAIMSCAN_RETRIEVER=faiss` is the low-memory option. Building with `WITH_BM25=0` drops Java and pyserini and is
only for faiss-only images (they must run with `CLAIMSCAN_RETRIEVER=faiss`).


| mount | host path (default) | what's used |
|---|---|---|
| `/artifacts` (ro) | `/mnt/d/crisis-claim-analysis/artifacts` | FAISS index `dense/faiss_bge_small_en_v1_5/` (26 GB, memory-mapped), BM25 Lucene index `sparse/bm25_sentences/index/` (4.3 GB), `data/processed/fever_sentences.sqlite`, verifier `model.pt` |
| `/hf` (ro) | `/mnt/d/huggingface_cache` | `BAAI/bge-small-en-v1.5`, `microsoft/deberta-v3-base` (offline; nothing is downloaded) |
| `/models` (ro, `llama` only) | `/mnt/e/llm/models` | the Gemma GGUF for the rear aggregator |

Files: `Dockerfile` and `docker-compose.yml` (repo root), `docker/constraints.txt` (exact versions for the
image; `requirements.txt` stays the list of what the code needs; `WITH_BM25=0` drops its pyserini line).

## One-time setup

1. **Docker in WSL:** Docker Desktop → Settings → Resources → WSL Integration → enable Ubuntu.
2. **Keep Docker off C:** Docker Desktop → Settings → Resources → Advanced → *Disk image location* → a
   folder on D: (e.g. `D:\DockerDesktop`). The default is under `%LOCALAPPDATA%` on C:, and the image plus
   build cache is a few GB. Check with `docker info --format '{{.DockerRootDir}}'` and Docker Desktop's setting.
3. **Sentence store** (replaces loading the 4 GB corpus into RAM):
   ```bash
   source .venv/bin/activate && source scripts/env.sh
   python -m claimscan.evidence.store build    # ~20-40 min, writes <artifacts>/data/processed/fever_sentences.sqlite
   python -m claimscan.evidence.store check
   ```

## Run with compose (recommended)

```bash
docker compose up --build -d                  # api only: hybrid retriever, rear aggregator OFF
CLAIMSCAN_RETRIEVER=faiss docker compose up -d   # low-memory: FAISS only
docker compose logs -f api                    # wait for "Load Complete"
curl localhost:8000/health
curl -X POST localhost:8000/fact-check -H 'content-type: application/json' \
  -d '{"claim": "Tokyo is the capital of Japan"}'
docker compose down
```

The `llama` service is in the `aggregator` profile, so plain `docker compose up` never starts it and needs
no GGUF file. To turn the rear aggregator on (an LLM veto that can only turn TRUE/FALSE into NOT ENOUGH INFO):

```bash
CLAIMSCAN_AGGREGATOR_ENABLED=1 AGGREGATOR_MODEL=gemma-3-4b-it-Q4_K_M.gguf \
  docker compose --profile aggregator up --build -d
curl localhost:8083/health                    # llama-server, from the host
```

Both settings are needed: the profile starts `llama`, the env var makes the api call it. If `llama` is down
or slow, the api returns the verifier's verdict with `aggregator_error` set (it fails open).

**Hosted model instead of llama** (no `aggregator` profile, no GGUF):

```bash
CLAIMSCAN_AGGREGATOR_ENABLED=1 CLAIMSCAN_AGGREGATOR_URL=https://api.openai.com/v1 CLAIMSCAN_AGGREGATOR_MODEL=gpt-5-nano CLAIMSCAN_AGGREGATOR_API_KEY_ENV=OPENAI_API_KEY OPENAI_API_KEY=... docker compose up api
```

`CLAIMSCAN_AGGREGATOR_API_KEY_ENV` holds the *name* of the variable with the key; compose passes
`OPENAI_API_KEY` into the container. The key comes from your shell or an untracked `.env` file next to
`docker-compose.yml` (git-ignored, and excluded from the build context by `.dockerignore`); never put it in a
committed file such as `configs/default.yaml` or `docker-compose.yml`.

Settings (shell env or a `.env` file next to `docker-compose.yml`; `.env` is git-ignored):

| variable | default | |
|---|---|---|
| `CLAIMSCAN_ARTIFACTS_HOST` | `/mnt/d/crisis-claim-analysis/artifacts` | mounted at `/artifacts` |
| `HF_CACHE_HOST` | `/mnt/d/huggingface_cache` | mounted at `/hf` (`/mnt/d/whisper_v2/hf_cache` also has both models) |
| `CLAIMSCAN_RETRIEVER` | `hybrid` | shipped default (FAISS + BM25, no RM3); `faiss` = low-memory (~2.5 GB less); `bm25` also works |
| `WITH_BM25` | `1` | build arg: Java 21 + pyserini; `0` only for a faiss-only image |
| `API_PORT` | `8000` | |
| `CLAIMSCAN_AGGREGATOR_ENABLED` | `0` | `1` = call the rear aggregator |
| `CLAIMSCAN_AGGREGATOR_URL` | `http://llama:8080/v1` | the compose `llama` service; point elsewhere to use another server |
| `CLAIMSCAN_AGGREGATOR_TIMEOUT` | `30` | seconds per request, no retries |
| `CLAIMSCAN_AGGREGATOR_MODEL` | unset (config's `aggregator.model`) | e.g. `gpt-5-nano` for a hosted API |
| `CLAIMSCAN_AGGREGATOR_API_KEY_ENV` | unset (config's `aggregator.api_key_env`) | name of the env var with the key, e.g. `OPENAI_API_KEY` |
| `OPENAI_API_KEY` | unset | the key; shell or untracked `.env` only |
| `AGGREGATOR_MODEL` | `gemma-3-4b-it-Q4_K_M.gguf` | placeholder: file name under `LLAMA_MODELS_DIR` |
| `LLAMA_MODELS_DIR` | `/mnt/e/llm/models` | mounted at `/models` |
| `LLAMA_IMAGE` | `ghcr.io/ggml-org/llama.cpp:server` | `:server-cuda` for the GPU (see the comments in `docker-compose.yml`) |
| `LLAMA_PORT` / `LLAMA_CTX` / `LLAMA_THREADS` | `8083` / `8192` / `4` | |

**GPU for llama (optional):** use `LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda`, add `-ngl 99` and
uncomment the `deploy:` block in `docker-compose.yml`. Needs the Windows NVIDIA driver with GPU support in
Docker Desktop's WSL2 backend (NVIDIA Container Toolkit). The api itself stays on CPU.

## Run without compose

```bash
docker build -t claimscan-api .                          # default: hybrid-capable (Java 21 + pyserini)
docker build -t claimscan-api-faiss --build-arg WITH_BM25=0 .  # faiss-only image

docker run --rm -p 8000:8000 \
  -v /mnt/d/crisis-claim-analysis/artifacts:/artifacts:ro \
  -v /mnt/d/huggingface_cache:/hf:ro \
  -e CLAIMSCAN_RETRIEVER=hybrid \
  claimscan-api

# low memory: add -e CLAIMSCAN_RETRIEVER=faiss (works with either image; required for claimscan-api-faiss)
```

## The endpoint

`/fact-check` returns the label, a short response, the confidence, the three class probabilities,
the evidence sentences with their `Page#line` source, a `low_confidence` flag and the latency
(plus `verifier_prediction`, the veto fields and `aggregator_error` when the aggregator is on).
With the aggregator off it serves the evaluated pipeline as is (`fact_check_detailed`, k=30, threshold 0.5),
so the benchmark numbers in `graphs/` describe this endpoint. Interactive docs: http://localhost:8000/docs

## Resources

- **RAM:** ~4.5 GB with the default `hybrid` (~2 GB for the memory-mapped FAISS index + DeBERTa + bge-small,
  plus ~2.5 GB for BM25's JVM); ~2 GB with `faiss`. Add ~3 GB for `llama` with a 4B Q4_K_M model at ctx 8192. Docker Desktop's WSL VM
  must be allowed that much (the machine has 15 GB).
- **First request is slow** (~25 s for FAISS; cold BM25 queries took 6-11 s with RM3, less without) while the
  indexes page into the OS cache; later ones take a few seconds.
  `/health` reports `models_loaded: true` once startup finishes; the container healthcheck waits for it.
- **One claim at a time** (`--workers 1` plus a lock): inference is CPU-bound, and the Lucene searcher isn't thread-safe.
  Scale out with more containers behind a load balancer, not more threads.
- Reading the 26 GB index from `/mnt/d` through a bind mount is slower than from a Linux filesystem. For a real
  deployment, copy the artifacts the API needs onto the server's local disk.
