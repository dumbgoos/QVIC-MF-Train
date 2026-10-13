#!/bin/bash
# Full-data Stage B after gradient fix; preserve original pooled/256 strategy.
set -euo pipefail

QVIC_ROOT="${QVIC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$QVIC_ROOT"
export QVIC_ROOT
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_b_fresh_$(date +%Y%m%d_%H%M%S)}"
export HTM_INIT_CHECKPOINT="${HTM_INIT_CHECKPOINT:-/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_a_2xh100/checkpoint-5000}"
export DATASET_ROOT="${DATASET_ROOT:-/mnt/sdb1/wenbin/luoling/LVDU/LLaVA-Video-83K}"
export ANNOTATION_PATH="${ANNOTATION_PATH:-/mnt/sdb1/wenbin/luoling/LLaVA-Video-83K/annotations/all_sampled.json}"
export MODEL_BASE=lmms-lab/LLaVA-Video-7B-Qwen2
export NUM_GPUS=2 BATCH_SIZE=1 GRAD_ACCUM=8 EPOCHS=1
export LR=1e-4 WARMUP_RATIO=0.03 WEIGHT_DECAY=0
export CONTEXT_TOKENS=16 CONTEXT_MEMORY_LENGTH=256 CLIP_FRAMES=64
export HTM_RECENT_WINDOW=4 HTM_QUANTILE=0.9 HTM_LAMBDA=0.1
export LORA_R=64 LORA_ALPHA=16 LORA_DROPOUT=0.05
export DS_CONFIG="$QVIC_ROOT/scripts/train/zero3.json"
export MEDIA_BACKEND=shards QA_MODE=single
export SAVE_STEPS=500 SAVE_TOTAL_LIMIT=3 LOGGING_STEPS=10 REPORT_TO=none
# Explicitly discard limits inherited from the ten-step smoke run.
export MAX_RECORDS="" EXTRA_ARGS=""
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
RESUME="${RESUME:-0}"

if [ "$RESUME" != "0" ] && [ "$RESUME" != "1" ]; then
    echo "ERROR: RESUME must be 0 or 1" >&2
    exit 2
fi
if [ "$RESUME" = "0" ] && [ -e "$OUTPUT_DIR" ]; then
    echo "ERROR: fresh training requires a new OUTPUT_DIR: $OUTPUT_DIR" >&2
    exit 2
fi
if [ "$RESUME" = "1" ]; then
    if ! compgen -G "$OUTPUT_DIR/checkpoint-*" >/dev/null; then
        echo "ERROR: RESUME=1 requires checkpoint-* under OUTPUT_DIR" >&2
        exit 2
    fi
fi

echo "Stage B full training: fresh=$((1-RESUME)) out=$OUTPUT_DIR"
echo "Init: base LLaVA-Video; fresh compressor LoRA/context seed; Stage A predictor=$HTM_INIT_CHECKPOINT"
echo "HTM: pooled events, 256 write tokens, W=4, q=0.9; K_r=0 during train"
echo "Data: all records, 1 epoch; global batch=2*1*8=16; max_steps unset"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY RUN: configuration only; training was not started"
    exit 0
fi

if [ "$RESUME" = "0" ] && [ ! -f "$HTM_INIT_CHECKPOINT/non_lora_trainables.bin" ]; then
    echo "ERROR: Stage A predictor checkpoint missing non_lora_trainables.bin" >&2
    exit 2
fi
if [ ! -f "$ANNOTATION_PATH" ] || [ ! -f "$DATASET_ROOT/metadata/shard_index.parquet" ]; then
    echo "ERROR: training annotations or shard_index.parquet missing" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"
if [ "$RESUME" = "0" ]; then
    "${QVIC_PY:-python}" - <<'PY'
import json, os, subprocess
from pathlib import Path
keys = ("OUTPUT_DIR", "HTM_INIT_CHECKPOINT", "MODEL_BASE", "DATASET_ROOT", "ANNOTATION_PATH", "NUM_GPUS", "BATCH_SIZE", "GRAD_ACCUM", "EPOCHS", "LR", "WARMUP_RATIO", "WEIGHT_DECAY", "CONTEXT_TOKENS", "CONTEXT_MEMORY_LENGTH", "CLIP_FRAMES", "HTM_RECENT_WINDOW", "HTM_QUANTILE", "HTM_LAMBDA", "LORA_R", "LORA_ALPHA", "LORA_DROPOUT", "DS_CONFIG", "MEDIA_BACKEND", "QA_MODE", "MAX_RECORDS", "EXTRA_ARGS", "SKIP_LIST_PATH", "CUDA_VISIBLE_DEVICES")
config = {key: os.environ.get(key) for key in keys}
config.update(event_representation="pooled", memory_layout="legacy", train_feedback=False,
              git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()))
(Path(os.environ["OUTPUT_DIR"]) / "launch_config.json").write_text(json.dumps(config, indent=2) + "\n")
PY
fi
exec bash "$QVIC_ROOT/scripts/train/train_htm_stage_b_2xh100.sh"
