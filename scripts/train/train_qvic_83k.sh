#!/bin/bash

set -euo pipefail

QVIC_ROOT="${QVIC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$QVIC_ROOT"

PYTHON="${QVIC_PY:-python}"
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29517}"

DATASET_ROOT="${DATASET_ROOT:?set DATASET_ROOT to the dataset directory (see the README)}"
MODEL_BASE="${MODEL_BASE:-lmms-lab/LLaVA-Video-7B-Qwen2}"
OUTPUT_DIR="${OUTPUT_DIR:-$QVIC_ROOT/ckpt/QViC-MF-7B-repro}"
DS_CONFIG="${DS_CONFIG:-$QVIC_ROOT/scripts/train/zero3.json}"
RUN_NAME="${RUN_NAME:-$(basename "$OUTPUT_DIR")}"

MEDIA_BACKEND="${MEDIA_BACKEND:-shards}"
VIDEO_FOLDER="${VIDEO_FOLDER:-}"

ANNOTATION_PATH="${ANNOTATION_PATH:-}"

BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
EPOCHS="${EPOCHS:-1}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
CONTEXT_TOKENS="${CONTEXT_TOKENS:-16}"
CONTEXT_MEMORY_LENGTH="${CONTEXT_MEMORY_LENGTH:-256}"
CLIP_FRAMES="${CLIP_FRAMES:-64}"
# -----------------------------------------------------------------------------

QA_MODE="${QA_MODE:-single}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
REPORT_TO="${REPORT_TO:-none}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

OPTIONAL_ARGS=()
[ -n "$VIDEO_FOLDER" ] && OPTIONAL_ARGS+=(--video_folder "$VIDEO_FOLDER")
[ -n "$ANNOTATION_PATH" ] && OPTIONAL_ARGS+=(--annotation_path "$ANNOTATION_PATH")

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$QVIC_ROOT:${PYTHONPATH:-}"
echo "root=$QVIC_ROOT  gpus=$NUM_GPUS  out=$OUTPUT_DIR  ds=$(basename "$DS_CONFIG")"

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
    --fill_context_memory True \
    --attn_implementation sdpa \
    --encoder_attn_implementation sdpa \
    --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
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
