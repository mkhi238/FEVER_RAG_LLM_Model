# Keep every cache, tool and temp file for this project off C:.
# Use:  source scripts/env.sh   (or add these lines to ~/.bashrc to make them permanent)
export UV_CACHE_DIR=/mnt/d/uv/cache
export UV_PYTHON_INSTALL_DIR=/mnt/d/uv/python
export UV_TOOL_DIR=/mnt/e/llm/tools
export UV_TOOL_BIN_DIR=/mnt/e/llm/bin
export UV_LINK_MODE=copy
export PIP_CACHE_DIR=/mnt/d/pip_cache
export TORCH_HOME=/mnt/d/torch_cache
export TMPDIR=/mnt/d/uv/tmp
export PATH="/mnt/e/llm/bin:$PATH"   # cmake, later llama.cpp tools
mkdir -p "$TMPDIR"
