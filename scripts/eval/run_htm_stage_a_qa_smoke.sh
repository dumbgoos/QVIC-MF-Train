#!/bin/bash
# #@HTM — QA smoke hook for HTM Stage A (thin wrapper; no new benchmark harness).
#
# HTM load path is now wired in builder + model_video_lvbench (see Stage B wrapper
# for the primary smoke Leo should run on checkpoint-5000 Stage B).
#
# Stage A success (design §7–§8): token usage ↓ ~30–50% (stats) AND QA not
# significantly hurt. Beating MF at same L is Stage B.
#
set -euo pipefail

QVIC_ROOT="${QVIC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$QVIC_ROOT"

HTM_CKPT="${HTM_CKPT:-/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_a_2xh100/checkpoint-5000}"
MODEL_BASE="${MODEL_BASE:-lmms-lab/LLaVA-Video-7B-Qwen2}"
OUT_PARENT="${OUT_PARENT:-/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_a_2xh100}"
QA_OUT="${QA_OUT:-$OUT_PARENT/qa_smoke}"
QUANTILE="${QUANTILE:-0.9}"

VIDEO_DIR="${VIDEO_DIR:-}"
QUESTION_FP="${QUESTION_FP:-}"
RUN_EXISTING="${RUN_EXISTING:-0}"

cat <<EOF
[htm-qa] HTM Stage A QA smoke hook
  checkpoint : $HTM_CKPT
  model_base : $MODEL_BASE
  out        : $QA_OUT
  quantile   : $QUANTILE

Note: Stage A ckpts may lack adapter_model.bin (predictor-only). The QViC LoRA
builder branch asserts adapter — for full compressor+HTM QA prefer Stage B:
  bash scripts/eval/run_htm_stage_b_qa_smoke.sh

If Stage A dir has adapter + non_lora (unusual), LVBench-style command:
  mkdir -p "$QA_OUT"
  python -m qvic.eval.model_video_lvbench \\
    --model-path "$HTM_CKPT" \\
    --model-base "$MODEL_BASE" \\
    --video_dir "\$VIDEO_DIR" \\
    --question_fp "\$QUESTION_FP" \\
    --output_dir "$QA_OUT" \\
    --output_name htm_stage_a_qa_smoke \\
    --frames_num 64 \\
    --context_memory_length 256 \\
    --compress_with_relevance False \\
    --htm_stage_a True \\
    --htm_quantile $QUANTILE \\
    --torch_dtype bfloat16

Write-path fair compare: K_r=0 (--compress_with_relevance False).
EOF

if [ "$RUN_EXISTING" != "1" ]; then
  echo "[htm-qa] exiting 0 with documentation only (set RUN_EXISTING=1 to call legacy eval)."
  exit 0
fi

if [ -z "$VIDEO_DIR" ] || [ -z "$QUESTION_FP" ]; then
  echo "[htm-qa] ERROR: RUN_EXISTING=1 requires VIDEO_DIR and QUESTION_FP" >&2
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
  --output_name htm_stage_a_qa_smoke \
  --frames_num 64 \
  --context_memory_length 256 \
  --compress_with_relevance False \
  --htm_stage_a True \
  --htm_quantile "$QUANTILE" \
  --torch_dtype bfloat16
