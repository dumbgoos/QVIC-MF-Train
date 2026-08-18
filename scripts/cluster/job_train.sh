#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=4
#PJM -L elapse=24:00:00
#PJM -g YOUR_GROUP
#PJM -N qvic_train
#PJM -j
#PJM -o logs/cluster/job_train.log

set -euo pipefail

source "${QVIC_ROOT:?set QVIC_ROOT, e.g. pjsub -x QVIC_ROOT=\$PWD ...}/scripts/cluster/env.sh"
cd "$QVIC_ROOT"
mkdir -p logs/cluster
qvic_require_dataset

echo "host=$(hostname)"; nvidia-smi -L

export NUM_GPUS="${NUM_GPUS:-4}"
export QVIC_PY DATASET_ROOT OUTPUT_DIR
export MODEL_BASE="${MODEL_BASE:-$QVIC_BASE}"
export DS_CONFIG="${DS_CONFIG:-$QVIC_ROOT/scripts/train/zero3.json}"

# --- Weights & Biases (optional) --------------------------------------------
# Export WANDB_API_KEY in your shell (or run `wandb login` once) and submit with
# `pjsub -x WANDB_API_KEY=...`; leave it unset to disable reporting entirely.
# The compute nodes need outbound network for online mode -- WANDB_MODE=offline
# plus a later `wandb sync` works otherwise.
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-qvic-mf}"
export WANDB_DIR="${WANDB_DIR:-$QVIC_ROOT/logs/wandb}"
# A stable id so the resubmissions that finish one epoch land in a single run.
# NOTE: wandb permanently retires a deleted run id ("was previously created and
# deleted; try a new run id", HTTP 410) and that is fatal at init -- if you ever
# delete this run on the server, change the id.
export WANDB_RUN_ID="${WANDB_RUN_ID:-qvic-mf-7b-repro}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
mkdir -p "$WANDB_DIR"
if [ -n "${WANDB_API_KEY:-}" ]; then
    export REPORT_TO="${REPORT_TO:-wandb}"
    echo "wandb: mode=$WANDB_MODE project=$WANDB_PROJECT run_id=$WANDB_RUN_ID"
else
    export REPORT_TO="${REPORT_TO:-none}"
    echo "wandb: WANDB_API_KEY unset -- reporting disabled"
fi
# ----------------------------------------------------------------------------

# decord plus the SigLIP preprocessing are the dataloader bottleneck; 4 workers
# per rank saturate a 4-GPU node without starving the compute threads.
export NUM_WORKERS="${NUM_WORKERS:-4}"

# Table 1: batch 1 / grad-accum 4. This leaves an H100 94 GiB at ~38 GiB, but do
# NOT spend that headroom on a bigger micro-batch -- measured, same 16-sample
# effective batch on 4 GPUs:
#
#   batch 1 x accum 4 -> 35.7 s/step, 38.4 GiB peak
#   batch 2 x accum 2 -> 36.0 s/step, 65.9 GiB peak
#
# The step is bound by the QMSA attention bias, a [B, 28, 14.6k, 14.6k] bf16
# tensor materialised per encoder layer -- bandwidth work that scales linearly
# with B, so there is nothing for a larger batch to amortise.
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"

# ~5187 optimizer steps at ~36 s. Checkpoint every 100 steps (~1 h) so that no
# single failure costs more than that; each one is ~2.5 GB / ~30 s, i.e. ~1 %
# overhead, and save_total_limit keeps the directory bounded.
export SAVE_STEPS="${SAVE_STEPS:-100}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"

# --- crash-tolerant launch ---------------------------------------------------
# decord segfaults on a handful of the corpus. That kills the dataloader worker
# process, which no in-process try/except can catch, and torch.distributed then
# tears the whole job down. So: blacklist whatever was being decoded, resume from
# the last checkpoint, and go again. Bounded, so a genuine bug still stops.
export MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"

mkdir -p "$OUTPUT_DIR/inflight"
attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
    attempt=$((attempt + 1))

    # Fold breadcrumbs from a killed worker into the skip list. Single process,
    # before the ranks start, so it cannot race.
    "$QVIC_PY" -c "
from qvic.train.data import harvest_inflight
import logging; logging.basicConfig(level=logging.INFO)
harvest_inflight('$OUTPUT_DIR/inflight', '$OUTPUT_DIR/bad_media.json')
"

    # A fresh data order per attempt. Combined with --ignore_data_skip this is
    # what makes resuming cheap: replaying the dataloader to the checkpoint step
    # would re-decode every sample already trained on (~1.7 h at step 2400, and
    # growing), whereas a new shuffle just carries on over the same corpus.
    export DATA_SEED=$((42 + attempt))

    if compgen -G "$OUTPUT_DIR/checkpoint-*" > /dev/null; then
        echo "== attempt $attempt/$MAX_ATTEMPTS, resuming from $(ls -dt "$OUTPUT_DIR"/checkpoint-* | head -1)"
    else
        echo "== attempt $attempt/$MAX_ATTEMPTS, starting from scratch"
    fi

    set +e
    EXTRA_ARGS="--ignore_data_skip True --data_seed $DATA_SEED ${EXTRA_ARGS:-}" \
        bash scripts/train/train_qvic_83k.sh
    rc=$?
    set -e

    if [ "$rc" -eq 0 ]; then
        echo "== training finished cleanly on attempt $attempt"
        exit 0
    fi
    echo "== attempt $attempt exited $rc; harvesting and retrying" >&2
done

echo "ERROR: gave up after $MAX_ATTEMPTS attempts" >&2
exit 1
