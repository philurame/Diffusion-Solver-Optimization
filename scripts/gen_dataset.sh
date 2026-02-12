#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u && conda deactivate && set -u
conda activate "$SCRIPT_DIR/../env"

DEVICE=$1
SEED_FROM=$((DEVICE * 1000))
SEED_TO=$((SEED_FROM + 1000))


outp_f="$SCRIPT_DIR/../logs/gen_dataset_${DEVICE}.log"
> "$outp_f"
nohup python "$SCRIPT_DIR/../lib/gen_dataset.py" $DEVICE $SEED_FROM $SEED_TO >> "$outp_f" 2>&1 &
echo $! >> "$outp_f"
