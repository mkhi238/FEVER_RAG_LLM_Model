#!/usr/bin/env bash
# Build llama.cpp on E: (never C:). Usage:
#   llamacpp/scripts/build.sh cpu
#   CUDA_HOME=/mnt/e/cuda-12.x llamacpp/scripts/build.sh cuda   # newer CUDA toolkit on E:
#   CUDAHOSTCXX=g++-13 llamacpp/scripts/build.sh cuda             # older gcc for CUDA 12.4
# See llamacpp/README.md step 1 for why the CUDA build needs one of those.
set -euo pipefail

MODE="${1:-cpu}"
SRC="${LLAMA_CPP_DIR:-/mnt/e/llm/llama.cpp}"
CMAKE="${CMAKE:-/mnt/e/llm/bin/cmake}"
export TMPDIR="${TMPDIR:-/mnt/e/llm/tmp}"
mkdir -p "$TMPDIR"

case "$SRC" in /mnt/c/*|/home/*|/tmp/*) echo "refusing to build on C: ($SRC)"; exit 1;; esac

ARGS=(-S "$SRC" -B "$SRC/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF)

if [[ "$MODE" == "cuda" ]]; then
  if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    ARGS+=(-DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc")
  fi
  if [[ -n "${CUDAHOSTCXX:-}" ]]; then
    ARGS+=(-DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX")
  fi
  # RTX 2060 = compute capability 7.5
  ARGS+=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75)
elif [[ "$MODE" != "cpu" ]]; then
  echo "usage: $0 cpu|cuda"; exit 2
fi

"$CMAKE" "${ARGS[@]}"
"$CMAKE" --build "$SRC/build" --config Release -j "$(nproc)" \
  --target llama-server llama-cli llama-quantize llama-bench

echo
echo "built: $SRC/build/bin"
ls "$SRC/build/bin" | grep -E '^llama-(server|cli|quantize|bench)$'
