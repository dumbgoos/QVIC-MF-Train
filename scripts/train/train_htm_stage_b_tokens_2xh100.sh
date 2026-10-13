#!/bin/bash
# Stage B with C-token events; default pilot before a full epoch.
set -euo pipefail
QVIC_ROOT="${QVIC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$QVIC_ROOT"
export QVIC_ROOT
MODE="${MODE:-pilot}"
RESUME="${RESUME:-0}"
if [[ "$MODE" != pilot && "$MODE" != full ]]; then
    echo "ERROR: MODE must be pilot or full" >&2; exit 2
fi
if [[ "$RESUME" != 0 && "$RESUME" != 1 ]]; then
    echo "ERROR: RESUME must be 0 or 1" >&2; exit 2
fi
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_tokens_${MODE}_$(date +%Y%m%d_%H%M%S)}"
if [[ "$RESUME" = 0 && -e "$OUTPUT_DIR" ]]; then
    echo "ERROR: fresh run requires a new OUTPUT_DIR: $OUTPUT_DIR" >&2; exit 2
fi
if [[ "$RESUME" = 1 ]] && ! compgen -G "$OUTPUT_DIR/checkpoint-*" >/dev/null; then
    echo "ERROR: resume requires checkpoint-* under OUTPUT_DIR" >&2; exit 2
fi
export HTM_INIT_CHECKPOINT="${HTM_INIT_CHECKPOINT:-/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_a_2xh100/checkpoint-5000}"
export MODEL_BASE=lmms-lab/LLaVA-Video-7B-Qwen2
export DATASET_ROOT="${DATASET_ROOT:-/mnt/sdb1/wenbin/luoling/LVDU/LLaVA-Video-83K}"
export ANNOTATION_PATH="${ANNOTATION_PATH:-/mnt/sdb1/wenbin/luoling/LLaVA-Video-83K/annotations/all_sampled.json}"
export NUM_GPUS=2 BATCH_SIZE=1 EPOCHS=1
export CONTEXT_TOKENS=16 CONTEXT_MEMORY_LENGTH=256 CLIP_FRAMES=64
export HTM_RECENT_WINDOW=4 HTM_QUANTILE=0.9 HTM_LAMBDA=0.1
export LR=1e-4 WARMUP_RATIO=0.03 WEIGHT_DECAY=0
export LORA_R=64 LORA_ALPHA=16 LORA_DROPOUT=0.05
export DS_CONFIG="$QVIC_ROOT/scripts/train/zero3.json"
export MEDIA_BACKEND=shards QA_MODE=single REPORT_TO=none
export EXTRA_ARGS="--htm_token_budget 4096 --htm_event_representation tokens --htm_memory_layout chronological --htm_group_tokens True --htm_reset_thresholds_per_video True"
if [[ "$MODE" = pilot ]]; then
    export MAX_RECORDS=128 GRAD_ACCUM=4 SAVE_STEPS=25 SAVE_TOTAL_LIMIT=4 LOGGING_STEPS=5
    export EXTRA_ARGS="$EXTRA_ARGS --max_steps 100"
else
    export MAX_RECORDS="" GRAD_ACCUM=8 SAVE_STEPS=500 SAVE_TOTAL_LIMIT=3 LOGGING_STEPS=10
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "Stage B tokens: mode=$MODE resume=$RESUME out=$OUTPUT_DIR"
echo "C=16 write_budget=4096 relevance_capacity=256 chronological grouped=True reset_thresholds_per_video=True"
echo "Stage A predictor=$HTM_INIT_CHECKPOINT; fresh compressor LoRA/context seed"
echo "max_records=${MAX_RECORDS:-all} grad_accum=$GRAD_ACCUM extra_args=$EXTRA_ARGS"
if [[ "${DRY_RUN:-0}" = 1 ]]; then
    echo "DRY RUN: training not started"; exit 0
fi
export HTM_TOKENS_RUN_MODE="$MODE"
if [[ "$RESUME" = 1 ]]; then
    "${QVIC_PY:-python}" - <<'PY'
import json, os
from pathlib import Path
config = json.loads((Path(os.environ["OUTPUT_DIR"]) / "launch_config.json").read_text())
if config.get("HTM_TOKENS_RUN_MODE") != os.environ.get("HTM_TOKENS_RUN_MODE", ""):
    raise SystemExit("ERROR: resume mode does not match launch_config.json")
PY
fi
if [[ "$RESUME" = 0 && ! -f "$HTM_INIT_CHECKPOINT/non_lora_trainables.bin" ]]; then
    echo "ERROR: missing Stage A predictor weights" >&2; exit 2
fi
if [[ ! -f "$ANNOTATION_PATH" || ! -f "$DATASET_ROOT/metadata/shard_index.parquet" ]]; then
    echo "ERROR: missing training annotations or shard index" >&2; exit 2
fi
mkdir -p "$OUTPUT_DIR"
if [[ "$RESUME" = 0 ]]; then
    "${QVIC_PY:-python}" - <<'PY'
import json, os, subprocess
from pathlib import Path
keys = ("HTM_TOKENS_RUN_MODE", "OUTPUT_DIR", "HTM_INIT_CHECKPOINT", "MODEL_BASE", "DATASET_ROOT", "ANNOTATION_PATH", "NUM_GPUS", "BATCH_SIZE", "GRAD_ACCUM", "LR", "MAX_RECORDS", "EXTRA_ARGS", "DS_CONFIG", "SKIP_LIST_PATH")
config = {key: os.environ.get(key) for key in keys}
config["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
(Path(os.environ["OUTPUT_DIR"]) / "launch_config.json").write_text(json.dumps(config, indent=2) + "\n")
PY
fi
exec bash "$QVIC_ROOT/scripts/train/train_htm_stage_b_2xh100.sh"
