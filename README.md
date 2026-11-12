# FEVER Fact-Checking System with RAG

An end-to-end fact-checking pipeline using Retrieval-Augmented Generation (RAG) to verify claims against a 17.1M sentence Wikipedia corpus. Combines FAISS dense retrieval with a fine-tuned DeBERTa-v3 verifier model.

This is the base for **ClaimScan** (RAG + RL): a LangGraph research agent, reranker, LoRA SFT and GRPO. Those parts are not built yet. Their packages exist as empty slots (`claimscan/agents`, `graph`, `rerank`, `serve`).

## Overview

This system verifies factual claims by:
1. Retrieving relevant evidence from Wikipedia using dense vector search (FAISS)
2. Scoring and filtering evidence based on claim-type patterns
3. Classifying claims as TRUE, FALSE, or NOT ENOUGH INFO using a neural verifier (deBERTa-v3)

Built for the FEVER (Fact Extraction and VERification) dataset with modifications for real-world claim types.

## Architecture
Input Claim - FAISS Retriever (BGE embeddings) - Evidence Quality Scoring (pattern-based filters, using claim type, similarity scores, e.t.c) - DeBERTa-v3 Verifier
↓
Prediction: TRUE/FALSE/NOT ENOUGH INFO

## Features

- Dense Retrieval: FAISS indexing with BGE embeddings for semantic search across 17.1M sentences
- Pattern-Based Filtering: Claim-type detection (authorship, temporal, comparative, geographic) for evidence relevance, along with line similariy scoring and penalization based on sentence text
- FastAPI service with health monitoring and structured request/response
- MLflow integration for hyperparameter evaluation and metrics logging

## Layout

```
claimscan/
  config.py        single config root (reads configs/default.yaml)
  data/            build FEVER claims / evidence / corpus parquet + gold map
  retrieval/       FAISS retriever, embedding + index builders, recall@k eval
  evidence/        scoring.py (claim-type heuristics), select.py (get_evidence)
  verifier/        DeBERTa model, training-data prep, training
  pipeline.py      load_services(), fact_check(), fact_check_cleaned()
  api/             FastAPI service (to be replaced by LangServe)
  agents/ graph/ rerank/ serve/   empty, filled by the ClaimScan plan
configs/default.yaml
scripts/           verifier_smoke, retriever_smoke, rag_demo, run_experiments, check_label_balance
tests/
archive/           old TF-IDF / BM25 / HNSW experiments (not imported)
```

## Installation

```bash
# Everything on D: (see "Location" below)
cd /mnt/d && git clone https://github.com/mkhi238/FEVER_RAG_LLM_Model.git && cd FEVER_RAG_LLM_Model
git config core.fileMode false          # NTFS can't store exec bits
export UV_PYTHON_INSTALL_DIR=/mnt/d/uv/python UV_CACHE_DIR=/mnt/d/uv/cache TMPDIR=/mnt/d/uv/tmp UV_LINK_MODE=copy
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt pytest httpx
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -m pytest
```

## Artifacts

Data, FAISS indexes and the verifier checkpoint stay outside the repo, under `artifacts_root` in `configs/default.yaml` (default `/mnt/d/crisis-claim-analysis/artifacts`). To point somewhere else, set `CLAIMSCAN_ARTIFACTS=/path`, or use your own YAML file with `CLAIMSCAN_CONFIG=/path/to.yaml`.

| Path (under artifacts_root) | What | Built by |
|---|---|---|
| `data/processed/fever_{claims,corpus,evidence}_parquet`, `fever_gold.json` | FEVER claims, corpus, gold evidence | `python -m claimscan.data.unify` |
| `dense/<index dir>/embeddings_shard_*.npy` | sentence embeddings | `python -m claimscan.retrieval.faiss_embeddings --index <name>` |
| `dense/<index dir>/faiss_index_175.bin` | IVFFlat index, 4096 lists (bge: **26 GB**) | `python -m claimscan.retrieval.faiss_indexer --index <name>` |
| `verifier/deberta_v1/training_data.json/training_data.json` | verifier training pairs | `python -m claimscan.verifier.prepare` |
| `verifier/deberta_v1/trained_model/model.pt` | DeBERTa-v3-base verifier | `python -m claimscan.verifier.train` |

### Dense indexes

The FAISS index builders are `claimscan/retrieval/faiss_embeddings.py` (corpus to embedding shards) and `claimscan/retrieval/faiss_indexer.py` (shards to IVFFlat index). These are the only FAISS builders in this project. `archive/build_hnsw_ann_indexer.py` builds hnswlib indexes, not FAISS. Both scripts take `--index`, which picks an entry from `dense_indexes` in `configs/default.yaml`:

| `--index` | Model | Dir under `dense/` | Dim | State on D: |
|---|---|---|---|---|
| `bge_small` (default, used by the retriever) | BAAI/bge-small-en-v1.5 | `faiss_bge_small_en_v1_5` | 384 | 253 shards + index |
| `minilm` | sentence-transformers/all-MiniLM-L6-v2 | `faiss_all_minilm_l6_v2` | 384 | 253 shards + index |
| `e5_base` | intfloat/e5-base-v2 | `faiss_e5_base_v2` | 768 | shards 1–56 only, no index |

## Location

The repo, its `.venv`, the Python interpreter (`/mnt/d/uv/python`), the uv cache, the artifacts and the HF cache are all on D:. Nothing in the project lives on C:. WSL's own disk (`~`) sits on C:, so don't clone or create a venv under `~`. If you need a temp folder, use `TMPDIR=/mnt/d/uv/tmp`.

## Usage

```bash
python scripts/verifier_smoke.py          # verifier only; runs on 16 GB RAM
python scripts/rag_demo.py                # full pipeline; needs RAM for the 26 GB index
python -m claimscan.retrieval.eval_recall # recall@k on FEVER dev
python scripts/run_experiments.py         # k / threshold sweep logged to MLflow
uvicorn claimscan.api.main:app            # POST /fact-check {"claim": "..."}, GET /health
```

From Python:

```python
from claimscan.pipeline import fact_check, load_services
services = load_services()
prediction, response = fact_check("Tokyo is the capital of Japan", services)
```

## Known limits

- The FAISS index is a 26 GB uncompressed `IndexIVFFlat`, so it doesn't fit on a 16 GB machine. The ClaimScan plan rebuilds it in compressed form.
- The verifier's 94–95% validation accuracy came from an unshuffled split. It returns NEI with high confidence on some easy supported claims (see `scripts/verifier_smoke.py`). The plan re-evaluates it.
- `fact_check` uses k=30 and a 0.5 confidence floor. The API uses k=10 and 0.55.
