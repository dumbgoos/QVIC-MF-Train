#!/bin/bash
#PJM -L rscgrp=a-batch
#PJM -L node=1
#PJM -L elapse=6:00:00
#PJM -g YOUR_GROUP
#PJM -N qvic_scan_media
#PJM -j
#PJM -o logs/cluster/job_scan_media.log
#
#   pjsub -x QVIC_ROOT=$PWD,DATASET_ROOT=/data/LLaVA-Video-83K \
#         scripts/cluster/job_scan_media.sh
set -euo pipefail

source "${QVIC_ROOT:?set QVIC_ROOT, e.g. pjsub -x QVIC_ROOT=\$PWD ...}/scripts/cluster/env.sh"
cd "$QVIC_ROOT"
mkdir -p logs/cluster
qvic_require_dataset

echo "host=$(hostname)  cores=$(nproc)"

OUT="${OUT:-$QVIC_ROOT/work_dirs/media_scan}"
WORKERS="${WORKERS:-48}"
# Fold the result straight into the live training run's skip list. Writes are
# atomic (os.replace) and the trainer only reads it when an attempt starts, so
# this takes effect from the next attempt/resubmission onward.
MERGE_INTO="${MERGE_INTO:-$OUTPUT_DIR/bad_media.json}"

mkdir -p "$OUT" "$(dirname "$MERGE_INTO")"

"$QVIC_PY" -u scripts/tools/scan_media.py \
    --root "$DATASET_ROOT" \
    --backend "${MEDIA_BACKEND:-shards}" \
    --out "$OUT" \
    --workers "$WORKERS" \
    --num-frames 64 \
    --merge-into "$MERGE_INTO"

echo "=== bad media ==="
"$QVIC_PY" -c "
import json
b = json.load(open('$OUT/bad_media.json'))
print(len(b), 'bad media keys')
for k in b[:40]: print(' ', k)
if len(b) > 40: print('  ...')
"
