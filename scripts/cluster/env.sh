#!/bin/bash
#   pjsub -x QVIC_ROOT=$PWD scripts/cluster/job_train.sh

: "${QVIC_ROOT:?set QVIC_ROOT to the repository root, e.g. pjsub -x QVIC_ROOT=\$PWD ...}"

export QVIC_ROOT
export QVIC_PY="${QVIC_PY:-$QVIC_ROOT/env/bin/python}"   # conda prefix, Python 3.10
export QVIC_BIN="${QVIC_BIN:-$(dirname "$QVIC_PY")}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Weights are pre-fetched on a login node; pinning avoids a surprise download
# stalling a batch job. Unset these if a job genuinely needs egress.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

export PATH="$QVIC_BIN:$PATH"
export PYTHONPATH="$QVIC_ROOT:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"


module load cuda/12.2.2 2>/dev/null || true

# Dataset and base model. DATASET_ROOT has no sensible default -- see the README.
export QVIC_BASE="${QVIC_BASE:-lmms-lab/LLaVA-Video-7B-Qwen2}"
export DATASET_ROOT="${DATASET_ROOT:-}"
export OUTPUT_DIR="${OUTPUT_DIR:-$QVIC_ROOT/ckpt/QViC-MF-7B-repro}"

qvic_require_dataset() {
    if [ -z "${DATASET_ROOT:-}" ]; then
        echo "ERROR: DATASET_ROOT is unset. Point it at the training corpus," >&2
        echo "       e.g. pjsub -x QVIC_ROOT=\$PWD,DATASET_ROOT=/data/LLaVA-Video-83K ..." >&2
        return 1
    fi
}
