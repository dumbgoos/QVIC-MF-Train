#!/bin/bash
# #@HTM — HTM Stage A launcher for the user's 2×H100 server.
#
# Stage A (see docs/htm-design.md §7):
#   - Freezes compressor / VLM / LoRA / vision tower
#   - Trains the GRU predictor (+ write/consolidation path online)
#   - Read path stays off (K_r = 0); loss defaults to L_pred only
#
# -----------------------------------------------------------------------------
# How to run (on the server)
# -----------------------------------------------------------------------------
#   1. Activate the training env (conda or venv) with the README stack:
#        Python 3.10, torch 2.5.1+cu121, deepspeed, `pip install -e ".[train]"`
#   2. cd to the code root, then launch:
#
#        cd /mnt/data_nvme1/wenbin/luoling/cvpr/code/QVIC-MF-Train
#        bash scripts/train/train_htm_stage_a_2xh100.sh
#
#   Optional overrides (env vars):
#        OUTPUT_DIR=...  GRAD_ACCUM=8  DS_CONFIG=...  MAX_RECORDS=64  EXTRA_ARGS=...
#
# Required env: same as scripts/train/train_qvic_83k.sh (CUDA + DeepSpeed).
# Re-running resumes from the newest checkpoint-* under OUTPUT_DIR.
#
# -----------------------------------------------------------------------------
# Data path mapping (server defaults)
# -----------------------------------------------------------------------------
#   Annotations (HF LLMasterLL/LLaVA-Video-83K):
#     /mnt/sdb1/wenbin/luoling/LLaVA-Video-83K
#       → annotations/all_sampled.json (+ metadata/shard_index.parquet on this tree)
#
#   Media / tar shards (HF bucket LLMasterLL/LVDU):
#     /mnt/sdb1/wenbin/luoling/LVDU/LLaVA-Video-83K
#       → shards/{video,frame}/*.tar , videos_large/
#
#   Wire-up:
#     --annotation_path  → Data B annotations JSON
#     --dataset_root     → Data A media root (ShardReader needs metadata/ + shards/)
#
#   If DATASET_ROOT is missing metadata/shard_index.parquet, symlink it once:
#     ln -sfn /mnt/sdb1/wenbin/luoling/LLaVA-Video-83K/metadata \
#             /mnt/sdb1/wenbin/luoling/LVDU/LLaVA-Video-83K/metadata
# -----------------------------------------------------------------------------

set -euo pipefail

# #@HTM — server code checkout (header/docs only; runtime root is this repo)
# /mnt/data_nvme1/wenbin/luoling/cvpr/code/QVIC-MF-Train

QVIC_ROOT="${QVIC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$QVIC_ROOT"

PYTHON="${QVIC_PY:-python}"
# #@HTM — 2× H100
NUM_GPUS="${NUM_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-29517}"

# #@HTM — absolute server data defaults
# Media / shards (LLMasterLL/LVDU → .../LVDU/LLaVA-Video-83K)
DATASET_ROOT="${DATASET_ROOT:-/mnt/sdb1/wenbin/luoling/LVDU/LLaVA-Video-83K}"
# Annotations (LLMasterLL/LLaVA-Video-83K)
ANNOTATION_ROOT="${ANNOTATION_ROOT:-/mnt/sdb1/wenbin/luoling/LLaVA-Video-83K}"
ANNOTATION_PATH="${ANNOTATION_PATH:-$ANNOTATION_ROOT/annotations/all_sampled.json}"

MODEL_BASE="${MODEL_BASE:-lmms-lab/LLaVA-Video-7B-Qwen2}"
OUTPUT_DIR="${OUTPUT_DIR:-$QVIC_ROOT/checkpoints/htm_stage_a_2xh100}"
# #@HTM — keep ZeRO-3 like train_qvic_83k.sh (full frozen 7B+7B still resident)
DS_CONFIG="${DS_CONFIG:-$QVIC_ROOT/scripts/train/zero3.json}"
RUN_NAME="${RUN_NAME:-$(basename "$OUTPUT_DIR")}"

MEDIA_BACKEND="${MEDIA_BACKEND:-shards}"
VIDEO_FOLDER="${VIDEO_FOLDER:-}"

BATCH_SIZE="${BATCH_SIZE:-1}"
# #@HTM — 83k script uses GRAD_ACCUM=4 on 4 GPUs (global 16); scale to 2 GPUs → 8
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
EPOCHS="${EPOCHS:-1}"
CONTEXT_TOKENS="${CONTEXT_TOKENS:-16}"
CONTEXT_MEMORY_LENGTH="${CONTEXT_MEMORY_LENGTH:-256}"
CLIP_FRAMES="${CLIP_FRAMES:-64}"

# #@HTM Stage A knobs
HTM_RECENT_WINDOW="${HTM_RECENT_WINDOW:-4}"
HTM_QUANTILE="${HTM_QUANTILE:-0.75}"
HTM_LAMBDA_LM="${HTM_LAMBDA_LM:-0}"

QA_MODE="${QA_MODE:-single}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
REPORT_TO="${REPORT_TO:-none}"
MAX_RECORDS="${MAX_RECORDS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

OPTIONAL_ARGS=()
[ -n "$VIDEO_FOLDER" ] && OPTIONAL_ARGS+=(--video_folder "$VIDEO_FOLDER")
[ -n "$ANNOTATION_PATH" ] && OPTIONAL_ARGS+=(--annotation_path "$ANNOTATION_PATH")
[ -n "$MAX_RECORDS" ] && OPTIONAL_ARGS+=(--max_records "$MAX_RECORDS")

if [ ! -f "$ANNOTATION_PATH" ]; then
    echo "ERROR: annotation JSON not found: $ANNOTATION_PATH" >&2
    echo "       Expected under ANNOTATION_ROOT=$ANNOTATION_ROOT" >&2
    exit 1
fi
if [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: media dataset_root not found: $DATASET_ROOT" >&2
    exit 1
fi
if [ ! -f "$DATASET_ROOT/metadata/shard_index.parquet" ]; then
    echo "WARNING: missing $DATASET_ROOT/metadata/shard_index.parquet" >&2
    echo "         ShardReader needs metadata/ beside shards/. Symlink from annotations tree:" >&2
    echo "         ln -sfn $ANNOTATION_ROOT/metadata $DATASET_ROOT/metadata" >&2
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$QVIC_ROOT:${PYTHONPATH:-}"
echo "root=$QVIC_ROOT  gpus=$NUM_GPUS  out=$OUTPUT_DIR  ds=$(basename "$DS_CONFIG")"
echo "dataset_root=$DATASET_ROOT  annotation_path=$ANNOTATION_PATH"
echo "HTM Stage A: --use_htm --htm_stage_a (freeze compressor; train GRU predictor)"

# shellcheck disable=SC2086
"$PYTHON" -m torch.distributed.run \
    --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    -m qvic.train.train_qvic \
    --deepspeed "$DS_CONFIG" \
    --model_base "$MODEL_BASE" \
    --dataset_root "$DATASET_ROOT" \
    --media_backend "$MEDIA_BACKEND" \
    "${OPTIONAL_ARGS[@]}" \
    --num_frames "$CLIP_FRAMES" \
    --qa_mode "$QA_MODE" \
    --conv_template qwen_1_5 \
    --context_embed_tokens "$CONTEXT_TOKENS" \
    --context_memory_length "$CONTEXT_MEMORY_LENGTH" \
    --max_frame_num_encoder "$CLIP_FRAMES" \
    --question_guided_selective_attention True \
    --guiding_context2vision True \
    --ctx_attn_mask_type framewise \
    --fill_context_memory False \
    --attn_implementation sdpa \
    --encoder_attn_implementation sdpa \
    --use_htm True \
    --htm_stage_a True \
    --htm_recent_window "$HTM_RECENT_WINDOW" \
    --htm_quantile "$HTM_QUANTILE" \
    --htm_lambda_lm "$HTM_LAMBDA_LM" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --lr_scheduler_type cosine \
    --warmup_ratio "$WARMUP_RATIO" \
    --weight_decay "$WEIGHT_DECAY" \
    --optim adamw_torch \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --model_max_length 32768 \
    --dataloader_num_workers "$NUM_WORKERS" \
    --dataloader_persistent_workers True \
    --logging_steps "$LOGGING_STEPS" \
    --save_strategy steps \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --report_to "$REPORT_TO" \
    $EXTRA_ARGS
