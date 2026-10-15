#!/bin/bash
# Full-epoch joint EASA + HTM training on physical GPU 0; global batch 32.
# Uses the validated microbatch 1 with accumulation 32, independently of smoke.
set -euo pipefail
QVIC_ROOT="${QVIC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export QVIC_ROOT
export CUDA_VISIBLE_DEVICES=0 TRAINING_NUM_GPUS=1 MODE=full
export BATCH_SIZE=1
export MASTER_PORT="${MASTER_PORT:-29530}"
export OUTPUT_DIR="${OUTPUT_DIR:-/data/jwb/ckpt/easa_htm_gpu0_full_$(date +%Y%m%d_%H%M%S)}"
export RUN_NAME="${RUN_NAME:-$(basename "$OUTPUT_DIR")}"
echo "GPU-0 full training: 1 * 1 * 32 = global batch 32; 1 epoch; default report_to=wandb"
exec bash "$QVIC_ROOT/scripts/train/train_easa_htm_4xh100.sh"
