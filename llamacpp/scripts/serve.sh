#!/usr/bin/env bash
# Start llama-server for one role from llamacpp/roles.yaml.   (YOU: fill in the TODOs)
# Usage: llamacpp/scripts/serve.sh research_agent|reranker
set -euo pipefail

ROLE="${1:?usage: $0 <role>}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ROLES="$REPO/llamacpp/roles.yaml"

# TODO 1: read the role's fields from $ROLES (python3 -c 'import yaml ...' is fine; pyyaml is installed).
#         Fail with a clear message if backend != llamacpp or the model file doesn't exist.

# TODO 2: build the llama-server command:
#         -m <models_dir>/<model> --port <port> -c <ctx> -ngl <n_gpu_layers> --host 127.0.0.1
#         + --grammar-file <grammar>  if grammar is set
#         + --reranking               if reranking: true
#         + --lora <path>             if lora is set

# TODO 3: exec it, and print the base URL (http://127.0.0.1:<port>/v1) the client will use.

echo "serve.sh: not implemented yet (role=$ROLE)"; exit 1
