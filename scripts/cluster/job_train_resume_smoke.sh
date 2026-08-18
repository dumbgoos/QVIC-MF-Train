#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=2
#PJM -L elapse=1:00:00
#PJM -g YOUR_GROUP
#PJM -N qvic_resume_smoke
#PJM -j
#PJM -o logs/cluster/job_train_resume_smoke.log
#
#   pjsub -x QVIC_ROOT=$PWD,DATASET_ROOT=/data/LLaVA-Video-83K \
#         scripts/cluster/job_train_resume_smoke.sh
set -euo pipefail

source "${QVIC_ROOT:?set QVIC_ROOT, e.g. pjsub -x QVIC_ROOT=\$PWD ...}/scripts/cluster/env.sh"
cd "$QVIC_ROOT"
mkdir -p logs/cluster
qvic_require_dataset

echo "host=$(hostname)"; nvidia-smi -L

export WANDB_MODE=disabled
OUT="$QVIC_ROOT/ckpt/QViC-MF-7B-resume-smoke"
rm -rf "$OUT"

run_phase() {   # $1 = max_steps
    NUM_GPUS=2 \
    QVIC_PY="$QVIC_PY" \
    OUTPUT_DIR="$OUT" \
    DS_CONFIG="$QVIC_ROOT/scripts/train/zero3.json" \
    NUM_WORKERS=2 \
    SAVE_STEPS=2 \
    SAVE_TOTAL_LIMIT=4 \
    LOGGING_STEPS=1 \
    EXTRA_ARGS="--max_records 32 --max_steps $1 --qa_pick first" \
        bash scripts/train/train_qvic_83k.sh
}

echo "===== phase 1: fresh start, stop at step 4 ====="
run_phase 4
"$QVIC_PY" -c "
import json; s=json.load(open('$OUT/trainer_state.json'))
assert s['global_step'] == 4, s['global_step']
print('phase 1 reached global_step', s['global_step'])"
ls -d "$OUT"/checkpoint-*

MARK_BEFORE="$(stat -c %Y "$OUT/checkpoint-4")"

echo "===== phase 2: same output dir, continue to step 8 ====="
run_phase 8
MARK_AFTER="$(stat -c %Y "$OUT/checkpoint-4")"

echo "===== verdict ====="
MARK_BEFORE="$MARK_BEFORE" MARK_AFTER="$MARK_AFTER" "$QVIC_PY" -c "
import json, os
s = json.load(open('$OUT/trainer_state.json'))
before, after = os.environ['MARK_BEFORE'], os.environ['MARK_AFTER']
print('global_step        :', s['global_step'])
print('checkpoint-4 mtime :', before, '->', after)
assert s['global_step'] == 8, f'expected 8, got {s[\"global_step\"]}'
# If phase 2 had restarted from zero it would have re-executed steps 1-4 and
# rewritten checkpoint-4 (save_steps=2). An untouched checkpoint-4 means the
# first step phase 2 actually ran was step 5.
assert before == after, 'phase 2 rewrote checkpoint-4, i.e. it restarted instead of resuming'
runtime = [l for l in s['log_history'] if 'train_runtime' in l][-1]['train_runtime']
print('phase 2 train_runtime:', round(runtime, 1), 's for 4 steps')
print('RESUME OK: phase 2 continued from step 4 instead of restarting')"
