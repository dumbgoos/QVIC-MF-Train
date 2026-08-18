#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=2
#PJM -L elapse=1:00:00
#PJM -g YOUR_GROUP
#PJM -N qvic_train_smoke
#PJM -j
#PJM -o logs/cluster/job_train_smoke.log
#
# Fall back to the single-GPU ZeRO-2 path with:
#   pjsub -x QVIC_ROOT=$PWD,...,SMOKE_DS=zero2,SMOKE_GPUS=1 ...
set -euo pipefail

source "${QVIC_ROOT:?set QVIC_ROOT, e.g. pjsub -x QVIC_ROOT=\$PWD ...}/scripts/cluster/env.sh"
cd "$QVIC_ROOT"
mkdir -p logs/cluster
qvic_require_dataset

echo "host=$(hostname)"; nvidia-smi -L

SMOKE_DS="${SMOKE_DS:-zero3}"
SMOKE_GPUS="${SMOKE_GPUS:-2}"
# Mirror the real job's micro-batch so the smoke exercises the same code path
# (batch > 1 changes the QMSA mask shape and the question-padding handling).
SMOKE_BATCH="${SMOKE_BATCH:-2}"
SMOKE_ACCUM="${SMOKE_ACCUM:-2}"

# No wandb: a smoke run would just litter the project with 8-step runs.
export WANDB_MODE=disabled

OUT="$QVIC_ROOT/ckpt/QViC-MF-7B-smoke-${SMOKE_DS}"
rm -rf "$OUT"
echo "ds=$SMOKE_DS gpus=$SMOKE_GPUS out=$OUT"

NUM_GPUS="$SMOKE_GPUS" \
QVIC_PY="$QVIC_PY" \
OUTPUT_DIR="$OUT" \
DS_CONFIG="$QVIC_ROOT/scripts/train/${SMOKE_DS}.json" \
NUM_WORKERS=2 \
BATCH_SIZE="$SMOKE_BATCH" \
GRAD_ACCUM="$SMOKE_ACCUM" \
SAVE_STEPS=8 \
LOGGING_STEPS=1 \
EXTRA_ARGS="--max_records 32 --max_steps 8 --qa_pick first" \
    bash scripts/train/train_qvic_83k.sh

echo "== checkpoint contents =="
ls -l "$OUT"

echo "== reload check (the builder path lmms-eval uses) =="
"$QVIC_PY" - <<PY
from qvic.model.builder import load_pretrained_model
tok, model, proc, ctx = load_pretrained_model(
    "$OUT", "$QVIC_BASE", "QViC-MF-7B-smoke",
    torch_dtype="bfloat16", device_map="cuda:0", attn_implementation="sdpa")
print("reloaded:", model.__class__.__name__, "| context tokens:", model.context_embed_tokens)
print("SMOKE OK")
PY
