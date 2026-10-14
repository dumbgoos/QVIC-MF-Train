#!/bin/bash
# EASA on top of the HTM C-token / chronological / grouped Stage B recipe.
set -euo pipefail
QVIC_ROOT="${QVIC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export QVIC_ROOT
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/easa_${MODE:-pilot}_$(date +%Y%m%d_%H%M%S)}"
export EASA_LAMBDA="${EASA_LAMBDA:-1.0}"
export EASA_TEMPERATURE="${EASA_TEMPERATURE:-0.1}"
export EASA_EPSILON="${EASA_EPSILON:-1e-6}"
export EASA_OWNER_BIAS="${EASA_OWNER_BIAS:-2.0}"
export EXTRA_ARGS="${EXTRA_ARGS:-} --use_easa True --easa_lambda $EASA_LAMBDA --easa_temperature $EASA_TEMPERATURE --easa_epsilon $EASA_EPSILON --easa_owner_bias $EASA_OWNER_BIAS"
echo "EASA decoder memory attention: lambda=$EASA_LAMBDA temperature=$EASA_TEMPERATURE epsilon=$EASA_EPSILON"
exec bash "$QVIC_ROOT/scripts/train/train_htm_stage_b_tokens_2xh100.sh"
