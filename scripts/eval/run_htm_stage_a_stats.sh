#!/bin/bash
# #@HTM — thin launcher for Stage A memory / consolidation stats.
#
# Defaults match the user's 2×H100 box. Override with env vars.
#
# Smoke (CPU, no VLM — always safe):
#   bash scripts/eval/run_htm_stage_a_stats.sh
#
# Real videos + Stage A ckpt (1 GPU recommended):
#   ENCODE_VIDEOS=1 LIMIT=32 DEVICE=cuda:0 bash scripts/eval/run_htm_stage_a_stats.sh
#
set -euo pipefail

QVIC_ROOT="${QVIC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$QVIC_ROOT"

# #@HTM — server defaults
CODE_HINT="${CODE_HINT:-/mnt/data_nvme1/wenbin/luoling/cvpr/code/ours/QVIC-MF-Train}"
HTM_CKPT="${HTM_CKPT:-/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_a_2xh100/checkpoint-5000}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/mnt/sdb1/wenbin/luoling/LLaVA-Video-83K/annotations/all_sampled.json}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/sdb1/wenbin/luoling/LVDU/LLaVA-Video-83K}"
OUT_PARENT="${OUT_PARENT:-/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_a_2xh100}"
HTM_STATS_OUT="${HTM_STATS_OUT:-$OUT_PARENT/htm_stage_a_stats.json}"

MODEL_BASE="${MODEL_BASE:-lmms-lab/LLaVA-Video-7B-Qwen2}"
LIMIT="${LIMIT:-8}"
L="${L:-256}"
C="${C:-16}"
W="${W:-4}"
DEVICE="${DEVICE:-cpu}"
ENCODE_VIDEOS="${ENCODE_VIDEOS:-0}"
PYTHON="${QVIC_PY:-python}"

if ! mkdir -p "$(dirname "$HTM_STATS_OUT")" 2>/dev/null; then
  HTM_STATS_OUT="${TMPDIR:-/tmp}/htm_stage_a_stats.json"
  mkdir -p "$(dirname "$HTM_STATS_OUT")"
  echo "[htm-stats] WARNING: default OUT_PARENT not writable; using $HTM_STATS_OUT" >&2
fi
export PYTHONPATH="$QVIC_ROOT:${PYTHONPATH:-}"

echo "[htm-stats] root=$QVIC_ROOT (server code often at $CODE_HINT)"
echo "[htm-stats] ckpt=$HTM_CKPT  out=$HTM_STATS_OUT  limit=$LIMIT device=$DEVICE"

EXTRA=()
if [ "$ENCODE_VIDEOS" = "1" ] || [ "$ENCODE_VIDEOS" = "true" ]; then
  EXTRA+=(--encode-videos)
else
  # #@HTM — CPU-friendly default when not encoding real videos
  EXTRA+=(--smoke)
fi

# Prefer python3 when QVIC_PY unset and python is missing (dev boxes).
if [ -z "${QVIC_PY:-}" ] && ! command -v "$PYTHON" >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
fi

# shellcheck disable=SC2086
"$PYTHON" "$QVIC_ROOT/scripts/tools/htm_stage_a_stats.py" \
  --checkpoint "$HTM_CKPT" \
  --model-base "$MODEL_BASE" \
  --annotation-path "$ANNOTATION_PATH" \
  --dataset-root "$DATASET_ROOT" \
  --output "$HTM_STATS_OUT" \
  --limit "$LIMIT" \
  --token-budget-L "$L" \
  --context-tokens "$C" \
  --recent-window "$W" \
  --device "$DEVICE" \
  "${EXTRA[@]}" \
  "$@"
