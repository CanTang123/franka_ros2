#!/usr/bin/env bash
set -euo pipefail
FR3_ENTRY_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FR3_REPO_DIR=$(cd "$FR3_ENTRY_DIR/../.." && pwd)
FR3_PAYLOAD_DIR=${FR3_BUNDLE_DIR:-$FR3_REPO_DIR/local_models}
if [[ ! -f "$FR3_PAYLOAD_DIR/bundle.json" && -f "$FR3_REPO_DIR/comparison_data/local_models/bundle.json" ]]; then
  FR3_PAYLOAD_DIR="$FR3_REPO_DIR/comparison_data/local_models"
fi
if [[ ! -f "$FR3_PAYLOAD_DIR/bundle.json" ]]; then
  echo "Missing local_models/bundle.json. Extract the complete package with weights." >&2
  exit 1
fi
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
if [[ "$JAX_PLATFORMS" == cuda || "$JAX_PLATFORMS" == gpu ]]; then
  export JAX_PLATFORMS=cuda,cpu
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec python "$FR3_ENTRY_DIR/inference_server.py" --bundle "$FR3_PAYLOAD_DIR" \
  --trial "${FR3_TRIAL:-$FR3_PAYLOAD_DIR/trials/A_000_s0.json}" "$@"
