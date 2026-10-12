#!/bin/bash
# #@HTM — Full official LVBench for Stage B HTM (not the 16-q smoke).
#
# Matches QViC-MF / Leo baseline protocol for 1fps≈50.3% / 2fps≈50.2%:
#   --frames_num 2048  --video_fps {1|2}  --min_frames_num 64
#   --context_memory_length 256  --context_condition_frame_num 32
# HTM Stage B write-path fair settings (same as smoke):
#   --compress_with_relevance False  --htm_stage_a False  --htm_quantile 0.9
#
# Requires device fix in tree (PR #2 / cursor/htm-device-fix-53f1 merged or checked out).
# Does NOT invent MF ckpt paths.
#
# Usage (Leo):
#   conda activate video
#   cd /mnt/data_nvme1/wenbin/luoling/cvpr/code/ours/QVIC-MF-Train
#   git fetch origin && git checkout cursor/htm-lvbench-full-eval-8ef4   # or HTM + device fix
#
#   # 1) Format meta once
#   python scripts/lvbench_format_annotation.py \
#     --input  /mnt/sdb1/wenbin/luoling/LVDU/eval/LVBench/video_info.meta.jsonl \
#     --output /mnt/sdb1/wenbin/luoling/LVDU/eval/LVBench/data/lvbench_qa_full.jsonl
#
#   # 2) Run (1fps or 2fps). Multi-GPU: CUDA_VISIBLE_DEVICES=0,1,...
#   RUN_EXISTING=1 VIDEO_FPS=1 \
#     VIDEO_DIR=/mnt/sdb1/wenbin/luoling/LVDU/eval/LVBench/data/video/all_videos \
#     QUESTION_FP=/mnt/sdb1/wenbin/luoling/LVDU/eval/LVBench/data/lvbench_qa_full.jsonl \
#     META_FP=/mnt/sdb1/wenbin/luoling/LVDU/eval/LVBench/video_info.meta.jsonl \
#     bash scripts/eval/run_htm_stage_b_lvbench_full.sh
#
set -euo pipefail

QVIC_ROOT="${QVIC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$QVIC_ROOT"

HTM_CKPT="${HTM_CKPT:-/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_b_2xh100/checkpoint-5000}"
MODEL_BASE="${MODEL_BASE:-lmms-lab/LLaVA-Video-7B-Qwen2}"
OUT_PARENT="${OUT_PARENT:-/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_b_2xh100}"
QUANTILE="${QUANTILE:-0.9}"
CONTEXT_MEMORY_LENGTH="${CONTEXT_MEMORY_LENGTH:-256}"
CONTEXT_CONDITION_FRAME_NUM="${CONTEXT_CONDITION_FRAME_NUM:-32}"
FRAMES_NUM="${FRAMES_NUM:-2048}"
MIN_FRAMES_NUM="${MIN_FRAMES_NUM:-64}"
VIDEO_FPS="${VIDEO_FPS:-1}"
FORCE_SAMPLE="${FORCE_SAMPLE:-False}"
COMPRESS_WITH_RELEVANCE="${COMPRESS_WITH_RELEVANCE:-False}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
OUTPUT_NAME="${OUTPUT_NAME:-htm_stage_b_lvbench_full_${VIDEO_FPS}fps}"

VIDEO_DIR="${VIDEO_DIR:-}"
QUESTION_FP="${QUESTION_FP:-}"
META_FP="${META_FP:-}"
RUN_EXISTING="${RUN_EXISTING:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

QA_OUT="${QA_OUT:-$OUT_PARENT/lvbench_full_${VIDEO_FPS}fps}"

cat <<EOF
[htm-lvbench-full] Stage B HTM full LVBench
  checkpoint : $HTM_CKPT
  model_base : $MODEL_BASE
  out        : $QA_OUT
  fps        : $VIDEO_FPS   (Leo baseline: 1 → ~50.3%, 2 → ~50.2% on QViC-MF)
  frames_num : $FRAMES_NUM  (official QViC-MF lvbench_eval_qvic.sh default)
  min_frames : $MIN_FRAMES_NUM
  L / q      : $CONTEXT_MEMORY_LENGTH / $QUANTILE
  K_r        : compress_with_relevance=$COMPRESS_WITH_RELEVANCE  (True=paper-matched; False=write-fair)
  htm_stage_a: False
  GPUs       : $CUDA_VISIBLE_DEVICES

Prep (once):
  python scripts/lvbench_format_annotation.py \\
    --input  \$META_FP \\
    --output \$QUESTION_FP

Score after run:
  python scripts/eval/score_lvbench_preds.py --preds \$QA_OUT/merge.jsonl --outdir \$QA_OUT

Set RUN_EXISTING=1 VIDEO_DIR=... QUESTION_FP=... VIDEO_FPS=1|2 to execute.
EOF

if [ "$RUN_EXISTING" != "1" ]; then
  echo "[htm-lvbench-full] documentation only (set RUN_EXISTING=1 to run)."
  exit 0
fi

if [ -z "$VIDEO_DIR" ] || [ -z "$QUESTION_FP" ]; then
  echo "[htm-lvbench-full] ERROR: RUN_EXISTING=1 requires VIDEO_DIR and QUESTION_FP" >&2
  exit 2
fi
if [ ! -f "$QUESTION_FP" ]; then
  echo "[htm-lvbench-full] ERROR: QUESTION_FP not found: $QUESTION_FP" >&2
  echo "  Format meta first with scripts/lvbench_format_annotation.py" >&2
  exit 2
fi
if [ ! -f "$HTM_CKPT/non_lora_trainables.bin" ] || [ ! -f "$HTM_CKPT/adapter_model.bin" ]; then
  echo "[htm-lvbench-full] ERROR: Stage B ckpt missing files under $HTM_CKPT" >&2
  exit 2
fi

# Device-fix sanity: GRU must be movable (PR #2). Soft warn only.
if ! grep -q 'p0.device != context_frames.device' "$QVIC_ROOT/qvic/model/htm.py" 2>/dev/null; then
  echo "[htm-lvbench-full] WARNING: htm.py may lack device-fix (PR #2). Expect CUDA device mismatch on smoke/full." >&2
fi

mkdir -p "$QA_OUT"
export PYTHONPATH="$QVIC_ROOT:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
CHUNKS=${#GPULIST[@]}
echo "[htm-lvbench-full] Using $CHUNKS GPU chunk(s): ${GPULIST[*]}"

PIDS=()
for IDX in $(seq 0 $((CHUNKS - 1))); do
  GPU_ID=${GPULIST[$IDX]}
  if [ "$IDX" -eq 0 ]; then
    set_verbose=True
  else
    set_verbose=False
  fi
  echo "[htm-lvbench-full] chunk $IDX / $CHUNKS on GPU $GPU_ID"
  CUDA_VISIBLE_DEVICES=$GPU_ID python -m qvic.eval.model_video_lvbench \
    --model-path "$HTM_CKPT" \
    --model-base "$MODEL_BASE" \
    --video_dir "$VIDEO_DIR" \
    --question_fp "$QUESTION_FP" \
    --output_dir "$QA_OUT" \
    --output_name "$OUTPUT_NAME" \
    --num-chunks "$CHUNKS" \
    --chunk-idx "$IDX" \
    --frames_num "$FRAMES_NUM" \
    --video_fps "$VIDEO_FPS" \
    --min_frames_num "$MIN_FRAMES_NUM" \
    --force_sample "$FORCE_SAMPLE" \
    --context_memory_length "$CONTEXT_MEMORY_LENGTH" \
    --context_condition_frame_num "$CONTEXT_CONDITION_FRAME_NUM" \
    --compress_with_relevance "$COMPRESS_WITH_RELEVANCE" \
    --htm_stage_a False \
    --htm_quantile "$QUANTILE" \
    --torch_dtype "$TORCH_DTYPE" \
    --set_verbose "$set_verbose" \
    >"$QA_OUT/stdout_chunk${IDX}.log" 2>&1 &
  PIDS+=($!)
done

ec=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    ec=1
  fi
done
if [ "$ec" -ne 0 ]; then
  echo "[htm-lvbench-full] ERROR: one or more chunks failed; see $QA_OUT/stdout_chunk*.log" >&2
  exit 1
fi

MERGE="$QA_OUT/merge.jsonl"
: >"$MERGE"
if [ "$CHUNKS" -eq 1 ]; then
  # Single-chunk writes ${OUTPUT_NAME}.json (JSONL lines despite .json suffix).
  SRC="$QA_OUT/${OUTPUT_NAME}.json"
  if [ ! -f "$SRC" ]; then
    echo "[htm-lvbench-full] ERROR: missing $SRC" >&2
    exit 1
  fi
  cp "$SRC" "$MERGE"
else
  for IDX in $(seq 0 $((CHUNKS - 1))); do
    PART="$QA_OUT/${CHUNKS}_${IDX}.json"
    if [ ! -f "$PART" ]; then
      echo "[htm-lvbench-full] ERROR: missing chunk file $PART" >&2
      exit 1
    fi
    cat "$PART" >>"$MERGE"
  done
fi

echo "[htm-lvbench-full] merged → $MERGE ($(wc -l <"$MERGE") lines)"
python scripts/eval/score_lvbench_preds.py --preds "$MERGE" --outdir "$QA_OUT" \
  --write-official-answers
echo "[htm-lvbench-full] score → $QA_OUT/score.json"

if [ -n "$META_FP" ] && [ -f "$META_FP" ]; then
  echo "[htm-lvbench-full] META_FP set ($META_FP). Optional official LVBench scorer:"
  echo "  # if zai-org/LVBench is cloned:"
  echo "  # python -c \"from lvbench.metrics.compute_accuracy import compute_accuracy; print(compute_accuracy('$QA_OUT/official_answers.json', '$META_FP'))\""
fi
