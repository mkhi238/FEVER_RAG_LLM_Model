---
name: dockerfile
version: 1.0.0
description: Build, run, health-check and smoke-test the ClaimScan fact-check API Docker image (Dockerfile + docker-compose.yml in the FEVER/ClaimScan repo). Use when asked to "build the docker image", "run the API in docker", "docker compose up", "is the container healthy", "test the image", or to rebuild after changing requirements, the Dockerfile or the shipped code.
---

# /dockerfile: build and verify the ClaimScan API image

Adapted from [wrsmith108/docker-claude-skill](https://github.com/wrsmith108/docker-claude-skill) (pre-flight check,
quick reference, `docker exec` without `-it`, wait-for-healthy script), rewritten for this Python/FastAPI project.
The image holds code + Python packages only (CPU torch, Java 21 + pyserini). The artifacts (FAISS, BM25, sentence
store) and the HF cache are mounted read-only. Details: `docker/README.md`.

**Ship only what works.** Build from `main` or the `ship` branch (a clean checkout or the ship worktree), never from
a branch with work in progress. `.dockerignore` is a whitelist (Dockerfile, requirements.txt,
docker/constraints.txt, configs/, claimscan/), so notes, PNGs, `.venv`, `.env` and keys never enter the build
context, but any half-written file under `claimscan/` would.

Use `docker compose` service names (`api`, `llama`), not container names: compose names containers after the
folder, which differs between checkouts and worktrees.

## Pre-flight (MANDATORY, stop and report on any failure)

```bash
docker info --format '{{.ServerVersion}} root={{.DockerRootDir}} mem={{.MemTotal}}'   # daemon up (Docker Desktop + WSL integration)
ls /mnt/c/Users/*/AppData/Local/Docker/wsl/disk/docker_data.vhdx 2>/dev/null && echo "STOP: Docker's disk image is on C:"
df -h /mnt/c | tail -1                                                                  # C: must not be the build target
git branch --show-current; git status --short claimscan configs requirements.txt Dockerfile docker   # main/ship, clean
A=$(sed -n 's/^artifacts_root:[[:space:]]*//p' configs/default.yaml | head -1)
ls -d "$A/data/processed/fever_sentences.sqlite" "$A/dense/faiss_bge_small_en_v1_5" "$A/sparse/bm25_sentences/index"
ls -d /mnt/d/huggingface_cache/hub/models--BAAI--bge-small-en-v1.5 /mnt/d/huggingface_cache/hub/models--MoritzLaurer--DeBERTa-v3-base-mnli-fever-anli
```

- **Docker disk on C:** stop. The repo's hard rule is nothing on C:. Docker Desktop → Settings → Resources →
  Advanced → *Disk image location* → `E:\DockerDesktop`.
- **No sentence store:** build `fever_sentences.sqlite` as described in `claimscan/data/fever_db.py` (on D:),
  then check it with `python -m claimscan.evidence.store`. The image can be built without it, but the API can't start.
- **Uncommitted changes under `claimscan/` or `configs/`:** they would be baked into the image. Build from a clean
  ship/main checkout instead.

## Build

```bash
time docker compose build api                        # = docker build -t claimscan-api . (WITH_BM25=1: Java 21 + pyserini)
WITH_BM25=0 docker compose build api                 # faiss-only image; must run with CLAIMSCAN_RETRIEVER=faiss
docker image ls claimscan-api --format '{{.Repository}}:{{.Tag}} {{.Size}}'
docker history --no-trunc claimscan-api | grep -iE "key|secret|token|password" | grep -v "GPG_KEY=" && echo "STOP: secret in a layer"   # GPG_KEY = python base image's public signing key, not a secret
```

Rebuild after changing `requirements.txt`, `docker/constraints.txt`, the `Dockerfile` or any shipped code.
Deps are installed before the code is copied, so code-only rebuilds reuse the pip layer.

## Run + health

```bash
docker compose up -d api                             # hybrid retriever, rear aggregator OFF
.claude/skills/dockerfile/docker-health.sh           # waits for healthy (<=30 min; cold start from D: measured ~19 min), 3 smoke claims, uid, ro mount, RAM
docker compose down
```

`docker-health.sh` waits on the container's HEALTHCHECK (`/health` → `models_loaded`, start period 1500 s). Run it in the background and wait for the notification. It
falls back to probing `/health` directly, then POSTs a likely-TRUE, a likely-FALSE and a likely-NEI claim. It
checks the HTTP status and response shape, not the verdict: the verifier is imperfect, and the benchmark covers
accuracy (`/fever-full-test 300`).

## Quick reference

| Task | Command |
|---|---|
| Status | `docker compose ps` |
| Logs | `docker compose logs --tail 50 api` (`-f` to follow; wait for "Load Complete") |
| Health | `docker inspect --format '{{.State.Health.Status}}' $(docker compose ps -q api)` |
| One request | `curl -s -X POST localhost:8000/fact-check -H 'content-type: application/json' -d '{"claim": "..."}'` |
| Command inside | `docker compose exec -T api <cmd>` (`-T`, never `-it`, in Claude Code) |
| Low-memory run | `CLAIMSCAN_RETRIEVER=faiss docker compose up -d api` |
| Aggregator + local llama | `CLAIMSCAN_AGGREGATOR_ENABLED=1 AGGREGATOR_MODEL=<file>.gguf docker compose --profile aggregator up -d` |
| Aggregator, hosted model | `CLAIMSCAN_AGGREGATOR_ENABLED=1 CLAIMSCAN_AGGREGATOR_URL=https://api.openai.com/v1 CLAIMSCAN_AGGREGATOR_MODEL=gpt-5-nano CLAIMSCAN_AGGREGATOR_API_KEY_ENV=OPENAI_API_KEY docker compose up -d api` (the key itself comes from your shell / an untracked `.env`) |
| Restart | `docker compose restart api` |
| Rebuild + run | `docker compose up -d --build api` |
| Stop | `docker compose down` |
| Disk used | `docker system df` |

## Rules

- Mounts stay read-only (`:ro`). The API never writes to the artifacts or the HF cache.
- Never put keys in the image, `configs/default.yaml` or `docker-compose.yml`. The config holds the *name* of the
  env var (`api_key_env`), and the value comes from the shell or an untracked `.env`.
- Never build onto C: (see the pre-flight).
- One uvicorn worker. Inference is CPU-bound and the Lucene searcher isn't thread-safe; scale with more containers.
- Ship code only: build from main/ship, not from a WIP branch.
- Clean up with `docker compose down` and `docker image prune`. Don't run `docker system prune -a` without
  asking: it deletes every image, including the llama.cpp one.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `error during connect ... dockerDesktopLinuxEngine` | Docker Desktop isn't running. Start it. |
| `500 Internal Server Error ... dockerDesktopLinuxEngine`, or `docker` missing in WSL while Docker Desktop runs | The engine is stuck (seen 2026-11-25: a WSL-integration error dialog left open, the `docker-desktop` distro stopped). Restart it: `powershell.exe -NoProfile -Command "Stop-Process -Name 'Docker Desktop','com.docker.backend','com.docker.build' -Force; Start-Sleep 5; wsl.exe --terminate docker-desktop; Start-Process 'C:\Program Files\Docker\Docker\Docker Desktop.exe'"`, then poll `docker info` (~1 min). Never `wsl --shutdown` from inside WSL. |
| `docker: command not found` in WSL | Docker Desktop → Settings → Resources → WSL Integration → enable Ubuntu. Don't use `docker.exe` from WSL: it can't read `/mnt/d` bind mounts. |
| Container exits or is OOM-killed while loading | The WSL VM needs ~4.5 GB for `hybrid` (BM25's JVM ~2.5 GB). Raise the VM memory, or run `CLAIMSCAN_RETRIEVER=faiss`. |
| `fever_sentences.sqlite missing` in the logs | Build the sentence store (pre-flight). |
| `OSError ... offline mode` / model not found | The HF cache mount is missing a model. Check `HF_CACHE_HOST` has bge-small-en-v1.5 and DeBERTa-v3-base-mnli-fever-anli under `hub/`. |
| `JavaException` / pyserini import error | The image was built with `WITH_BM25=0` but is running hybrid/bm25. Rebuild with the default, or run faiss. |
| First request takes 10-30 s | Normal: the FAISS/BM25 indexes are paging in from D:. Later requests take a few seconds. |
| Port 8000 in use | `API_PORT=8001 docker compose up -d api` (and `docker-health.sh 420 8001`) |
| Unhealthy but `/health` responds | `models_loaded` is false: read `docker compose logs api` for the load error. |

## Report

Image tag + size, build time, the health result and time to healthy, each smoke claim's status, label, confidence
and first evidence `Page#line`, the uid, the read-only check, and RAM. On a failure: the step, the error lines
from the logs, and the fix from the table above.
