# ClaimScan fact-check API (CPU). Code and Python deps only: the artifacts (FAISS index, BM25
# index, sentence store) and the HF model cache are mounted read-only at run time.
#
#   docker compose up --build                                # recommended, see docker-compose.yml
#   docker build -t claimscan-api .                          # default: Java 21 + pyserini, for the hybrid retriever
#   docker build -t claimscan-api --build-arg WITH_BM25=0 .  # faiss-only image (run with CLAIMSCAN_RETRIEVER=faiss)
#
# CPU torch on purpose: the API runs one DeBERTa-base forward pass per claim (~1-2 s on CPU), and the
# 6 GB RTX 2060 is kept for llama-server and RL training. The CPU wheel also keeps the image a few GB
# smaller than with CUDA. See docker/README.md for memory and file requirements.

# trixie: has openjdk-21 (bookworm only has 17, and pyserini 2.x needs 21)
FROM python:3.12-slim-trixie

# 1 (default): Java 21 + pyserini, needed by the shipped retriever.kind hybrid (and bm25).
# 0: ~0.5 GB smaller faiss-only image; it must run with CLAIMSCAN_RETRIEVER=faiss.
ARG WITH_BM25=1
ARG TORCH_VERSION=2.14.1
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    CLAIMSCAN_ARTIFACTS=/artifacts \
    HF_HOME=/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    OMP_NUM_THREADS=4 \
    JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

RUN if [ "$WITH_BM25" = "1" ]; then \
      apt-get update && apt-get install -y --no-install-recommends openjdk-21-jre-headless \
      && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app

# deps first so code edits don't reinstall them. WITH_BM25=0 drops pyserini from requirements.txt.
# CPU torch first: requirements.txt's torch>=2.14 is
# then already satisfied, so pip never fetches the CUDA build from PyPI.
COPY requirements.txt ./
COPY docker/constraints.txt docker/
RUN pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu \
 && if [ "$WITH_BM25" = "1" ]; then cp requirements.txt /tmp/req.txt; \
    else grep -v '^pyserini' requirements.txt > /tmp/req.txt; fi \
 && pip install -r /tmp/req.txt -c docker/constraints.txt && rm /tmp/req.txt \
 && python -c "import torch; assert not torch.cuda.is_available() and '+cpu' in torch.__version__, torch.__version__"

COPY configs/ configs/
COPY claimscan/ claimscan/

# code stays root-owned (read-only for the app); the process runs unprivileged
RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=1500s --retries=3 \
  CMD python -c "import json,urllib.request,sys; sys.exit(0 if json.load(urllib.request.urlopen('http://127.0.0.1:8000/health'))['models_loaded'] else 1)"

# one worker: inference is CPU-bound and serialised by a lock (the Lucene searcher isn't thread-safe)
CMD ["uvicorn", "claimscan.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
