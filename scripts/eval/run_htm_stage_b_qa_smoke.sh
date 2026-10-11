#!/bin/bash
# #@HTM — Stage B QA smoke (thin wrapper over existing LVBench eval; no new harness).
#
# Loads Stage B ckpt via builder QViC branch:
#   initialize_htm + htm_predictor (non_lora) + context_embed + Peft LoRA
# Write-path fair compare: K_r=0 (--compress_with_relevance False), QUANTILE=0.9
# Stage B joint: --htm_stage_a False
#
# Requires VIDEO_DIR + QUESTION_FP (Leo's small LVBench / QA subset).
# Does NOT run full LVBench. Success = HTM active + preds written; vs MF needs MF ckpt.
#
set -euo pipefail

QVIC_ROOT="${QVIC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$QVIC_ROOT"

HTM_CKPT="${HTM_CKPT:-/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_b_2xh100/checkpoint-5000}"
MODEL_BASE="${MODEL_BASE:-lmms-lab/LLaVA-Video-7B-Qwen2}"
OUT_PARENT="${OUT_PARENT:-/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_b_2xh100}"
QA_OUT="${QA_OUT:-$OUT_PARENT/qa_smoke}"
QUANTILE="${QUANTILE:-0.9}"
FRAMES_NUM="${FRAMES_NUM:-64}"
CONTEXT_MEMORY_LENGTH="${CONTEXT_MEMORY_LENGTH:-256}"
OUTPUT_NAME="${OUTPUT_NAME:-htm_stage_b_qa_smoke}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

# Optional: set to attempt a real smoke with existing eval.
VIDEO_DIR="${VIDEO_DIR:-}"
QUESTION_FP="${QUESTION_FP:-}"
RUN_EXISTING="${RUN_EXISTING:-0}"

cat <<EOF
[htm-qa-b] HTM Stage B QA smoke
  checkpoint : $HTM_CKPT
  model_base : $MODEL_BASE
  out        : $QA_OUT
  quantile   : $QUANTILE
  L          : $CONTEXT_MEMORY_LENGTH
  K_r        : off (compress_with_relevance False)
  htm_stage_a: False (Stage B joint)

Wiring (this branch):
  1. model_name with htm_* → QViC LoRA builder branch (even without "qvic"/"lora" in path)
  2. builder: initialize_htm(W=4, q=$QUANTILE) + load htm_predictor + context_embed + Peft LoRA
  3. eval: set_htm_stage_a(False); sync controller L; fill_context_memory=False

Exact command:
  mkdir -p "$QA_OUT"
  python -m qvic.eval.model_video_lvbench \\
    --model-path "$HTM_CKPT" \\
    --model-base "$MODEL_BASE" \\
    --video_dir "\$VIDEO_DIR" \\
    --question_fp "\$QUESTION_FP" \\
    --output_dir "$QA_OUT" \\
    --output_name "$OUTPUT_NAME" \\
    --frames_num $FRAMES_NUM \\
    --context_memory_length $CONTEXT_MEMORY_LENGTH \\
    --compress_with_relevance False \\
    --htm_stage_a False \\
    --htm_quantile $QUANTILE \\
    --torch_dtype $TORCH_DTYPE

MF contrast (same QUESTION_FP / L / K_r off): point --model-path at Leo's QViC-MF SFT ckpt
  (not wired as a default path in-repo).

Set RUN_EXISTING=1 VIDEO_DIR=... QUESTION_FP=... to execute.
EOF

if [ "$RUN_EXISTING" != "1" ]; then
  echo "[htm-qa-b] documentation only (set RUN_EXISTING=1 VIDEO_DIR=... QUESTION_FP=... to run)."
  exit 0
fi

if [ -z "$VIDEO_DIR" ] || [ -z "$QUESTION_FP" ]; then
  echo "[htm-qa-b] ERROR: RUN_EXISTING=1 requires VIDEO_DIR and QUESTION_FP" >&2
  exit 2
fi

if [ ! -f "$HTM_CKPT/non_lora_trainables.bin" ] || [ ! -f "$HTM_CKPT/adapter_model.bin" ]; then
  echo "[htm-qa-b] ERROR: Stage B ckpt missing non_lora_trainables.bin or adapter_model.bin under $HTM_CKPT" >&2
  exit 2
fi

mkdir -p "$QA_OUT"
export PYTHONPATH="$QVIC_ROOT:${PYTHONPATH:-}"
python -m qvic.eval.model_video_lvbench \
  --model-path "$HTM_CKPT" \
  --model-base "$MODEL_BASE" \
  --video_dir "$VIDEO_DIR" \
  --question_fp "$QUESTION_FP" \
  --output_dir "$QA_OUT" \
  --output_name "$OUTPUT_NAME" \
  --frames_num "$FRAMES_NUM" \
  --context_memory_length "$CONTEXT_MEMORY_LENGTH" \
  --compress_with_relevance False \
  --htm_stage_a False \
  --htm_quantile "$QUANTILE" \
  --torch_dtype "$TORCH_DTYPE"

echo "[htm-qa-b] wrote $QA_OUT/${OUTPUT_NAME}.json"
echo "[htm-qa-b] success smoke: file non-empty; logs show (HTM) initialize_htm / loaded htm_predictor / eval runtime stage_a=False"
