#!/bin/bash
# #@HTM — QA smoke *hook* for HTM Stage A (does NOT invent a new benchmark harness).
#
# Status (2026-10): existing ``qvic/eval/model_video_lvbench.py`` / ``model_vqa.py``
# load QViC via ``builder.load_pretrained_model`` but do **not** yet call
# ``initialize_htm`` / load ``htm_predictor`` from Stage A ``non_lora_trainables.bin``.
# Wiring that into the eval entrypoints is a small follow-up; until then this
# script documents the exact next commands and exits non-zero if HTM is not wired.
#
# Stage A success (design §7–§8): token usage ↓ ~30–50% (see stats script) AND
# QA not significantly hurt. Beating MF at same L is Stage B — not required here.
#
set -euo pipefail

QVIC_ROOT="${QVIC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$QVIC_ROOT"

HTM_CKPT="${HTM_CKPT:-/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_a_2xh100/checkpoint-5000}"
MODEL_BASE="${MODEL_BASE:-lmms-lab/LLaVA-Video-7B-Qwen2}"
OUT_PARENT="${OUT_PARENT:-/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_a_2xh100}"
QA_OUT="${QA_OUT:-$OUT_PARENT/qa_smoke}"

# Optional: set these to attempt a real LVBench-style smoke with existing eval.
VIDEO_DIR="${VIDEO_DIR:-}"
QUESTION_FP="${QUESTION_FP:-}"
RUN_EXISTING="${RUN_EXISTING:-0}"

cat <<EOF
[htm-qa] HTM Stage A QA smoke hook
  checkpoint : $HTM_CKPT
  model_base : $MODEL_BASE
  out        : $QA_OUT

Covered by stats tooling (not this script):
  - token budget |M|, merges, S_t/D_t  →  scripts/eval/run_htm_stage_a_stats.sh

NOT covered yet (needs eval entrypoint HTM load):
  - full VQA accuracy vs MF on the same clips / same L

Exact next command once HTM is wired into qvic/eval (LVBench-style):
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
    --torch_dtype bfloat16

Required wiring (follow-up, minimal):
  1. After load_pretrained_model(...), call:
       model.initialize_htm(recent_window_W=4, quantile=0.75)
       model.set_use_htm(True); model.set_htm_stage_a(True)
  2. Load htm_predictor.* from \$HTM_CKPT/non_lora_trainables.bin
     (same key stripping as scripts/tools/htm_stage_a_stats.py).
  3. Keep K_r off for an apples-to-apples Stage A write-path check, or enable
     compress_with_relevance only when comparing read-path behavior.

Until that lands, use token stats as the primary Stage A gate and treat QA as
blocked on eval HTM load — do not invent a parallel benchmark harness.
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
# Warning: without HTM wiring this runs MF/base write path only.
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
  --torch_dtype bfloat16
