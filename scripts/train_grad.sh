#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
cd "$ROOT"

DEVICE="$1"

# snapshot config before conda dead slow activation
ORIG_CONFIG="$ROOT/configs/grad_config.yaml"
SNAP_CONFIG="$ROOT/logs/grad_config_${DEVICE}_$(date +%Y%m%d_%H%M%S).yaml"
cp "$ORIG_CONFIG" "$SNAP_CONFIG"

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u && conda deactivate && set -u
conda activate "$ROOT/env"

outp_f="$ROOT/logs/train_grad_$DEVICE.txt"
> $outp_f

nohup python -m "train.grad.main" \
  --device "$DEVICE" \
  --config "$SNAP_CONFIG" \
  --main_server true \
  >> $outp_f 2>&1 &

echo $! | tee -a $outp_f