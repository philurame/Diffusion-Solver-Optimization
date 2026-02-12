#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."

MLFLOW_DIR="/workspace-SR008.fs2/philurame/MLFLOW"
MLFLOW_PORT="${MLFLOW_PORT:-5013}"
export MLFLOW_GUNICORN_CMD_ARGS="--timeout 600 --workers 1 --threads 2 --graceful-timeout 120 --keep-alive 5 --max-requests 512 --max-requests-jitter 50"

maybe_run_mlflow_server() {
  local uri="sqlite:///${MLFLOW_DIR}/mlflow.db"
  local artifact_root="file:${MLFLOW_DIR}/mlruns"   # make it an explicit file: URI

  local port="$1"
  if ! (ss -ltn 2>/dev/null | grep -q ":${port} "); then
    outp_f="$ROOT/logs/.mlflow_server_${port}.log"
    > "$outp_f"
    nohup mlflow server \
      --backend-store-uri "$uri" \
      --default-artifact-root "$artifact_root" \
      --host 0.0.0.0 \
      --port "$port" \
      >> "$outp_f" 2>&1 &
    echo $! >> "$outp_f"
    echo "Started central MLflow server on 0.0.0.0:${port}"
  else
    echo "MLflow server already listening on 0.0.0.0:${port}"
  fi
}

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u && conda deactivate && set -u
conda activate "$ROOT/env"

maybe_run_mlflow_server "$MLFLOW_PORT"