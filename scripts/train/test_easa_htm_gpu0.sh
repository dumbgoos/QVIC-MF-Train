#!/bin/bash
# Real pretrained GPU-0 smoke: four optimizer steps + checkpoint update checks.
# Uses the full 64-frame / C=16 / 4096-token EASA+HTM recipe, with global batch 1.
set -euo pipefail
QVIC_ROOT="${QVIC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export QVIC_ROOT
cd "$QVIC_ROOT"
export CUDA_VISIBLE_DEVICES=0
export MODE=smoke RESUME=0
export MASTER_PORT="${MASTER_PORT:-29529}"
export OUTPUT_DIR="${OUTPUT_DIR:-/data/jwb/ckpt/easa_htm_smoke_gpu0_$(date +%Y%m%d_%H%M%S)}"
export RUN_NAME="$(basename "$OUTPUT_DIR")"
PYTHON="${QVIC_PY:-python}"
echo "GPU-0 smoke: 16 records, 4 optimizer steps, global batch 1, no W&B"
echo "Checks: real forward/backward, finite loss, saved checkpoints and updates in all three trainable groups"
bash "$QVIC_ROOT/scripts/train/train_easa_htm_4xh100.sh"
if [[ "${DRY_RUN:-0}" = 1 ]]; then exit 0; fi

"$PYTHON" - "$OUTPUT_DIR" <<'PY'
import json, math, sys
from pathlib import Path
out = Path(sys.argv[1])
for step in (2, 4):
    checkpoint = out / f"checkpoint-{step}"
    for filename in ("config.json", "adapter_config.json", "adapter_model.bin", "non_lora_trainables.bin", "trainer_state.json"):
        if not (checkpoint / filename).is_file():
            raise SystemExit(f"FAIL: missing {checkpoint / filename}")
config = json.loads((out / "checkpoint-4/config.json").read_text())
expected = {"use_easa": True, "htm_token_budget": 4096, "htm_event_representation": "tokens", "htm_group_tokens": True}
if any(config.get(key) != value for key, value in expected.items()):
    raise SystemExit("FAIL: saved checkpoint does not match EASA + HTM recipe")
state = json.loads((out / "checkpoint-4/trainer_state.json").read_text())
losses = [row["loss"] for row in state.get("log_history", []) if "loss" in row]
if state.get("global_step") != 4 or not losses or not all(math.isfinite(loss) for loss in losses):
    raise SystemExit("FAIL: expected four optimizer steps with finite logged losses")
print("PASS: four updates, finite losses and both EASA + HTM checkpoints saved")
PY
"$PYTHON" "$QVIC_ROOT/scripts/tools/check_htm_checkpoint_updates.py" \
    --before "$OUTPUT_DIR/checkpoint-2" --after "$OUTPUT_DIR/checkpoint-4" \
    --out "$OUTPUT_DIR/smoke_parameter_updates.json"
echo "PASS: GPU-0 EASA + HTM training smoke; output=$OUTPUT_DIR"
echo "This verifies training/saving/parameter changes; it does not measure accuracy or four-GPU memory."
