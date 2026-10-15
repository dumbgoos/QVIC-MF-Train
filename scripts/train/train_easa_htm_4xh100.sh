#!/bin/bash
# Joint EASA + HTM Stage B, four H100s, effective global batch fixed at 32.
# Full epoch by default. MODE=pilot uses 128 records / 100 updates, also batch 32.
# MODE=smoke is used by the GPU-0 wrapper: one GPU / batch 1 / four updates.
# DRY_RUN=1 prints the command without creating directories or loading models.
set -euo pipefail
QVIC_ROOT="${QVIC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$QVIC_ROOT"
PYTHON="${QVIC_PY:-python}"
MODE="${MODE:-full}"
RESUME="${RESUME:-0}"
if [[ "$MODE" != full && "$MODE" != pilot && "$MODE" != smoke ]]; then
    echo "ERROR: MODE must be full, pilot or smoke" >&2; exit 2
fi
if [[ "$RESUME" != 0 && "$RESUME" != 1 ]]; then
    echo "ERROR: RESUME must be 0 or 1" >&2; exit 2
fi
if [[ "$MODE" = smoke ]]; then
    # Explicitly isolate the requested physical GPU, ignoring an inherited list.
    export CUDA_VISIBLE_DEVICES=0
    NUM_GPUS=1 GLOBAL_BATCH_SIZE=1 BATCH_SIZE=1 GRAD_ACCUM=1
else
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
    IFS=',' read -r -a DEVICES <<< "$CUDA_VISIBLE_DEVICES"
    if [[ ${#DEVICES[@]} != 4 ]]; then
        echo "ERROR: exactly four CUDA_VISIBLE_DEVICES are required" >&2; exit 2
    fi
    for ((i=0; i<4; i++)); do
        if [[ ! "${DEVICES[i]}" =~ ^[0-9]+$ ]]; then
            echo "ERROR: use four distinct numeric GPU IDs" >&2; exit 2
        fi
        for ((j=0; j<i; j++)); do
            if [[ "${DEVICES[i]}" = "${DEVICES[j]}" ]]; then
                echo "ERROR: duplicate GPU ID" >&2; exit 2
            fi
        done
    done
    NUM_GPUS=4
    GLOBAL_BATCH_SIZE=32
    BATCH_SIZE="${BATCH_SIZE:-1}"
    if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || ((BATCH_SIZE > 8)); then
        echo "ERROR: BATCH_SIZE must be a positive divisor of 8" >&2; exit 2
    fi
    if (( GLOBAL_BATCH_SIZE % (NUM_GPUS * BATCH_SIZE) )); then
        echo "ERROR: 4 * BATCH_SIZE must divide global batch 32" >&2; exit 2
    fi
    # Derived rather than inherited: BATCH_SIZE=2 automatically changes accumulation to 4.
    GRAD_ACCUM=$(( GLOBAL_BATCH_SIZE / NUM_GPUS / BATCH_SIZE ))
fi
MASTER_PORT="${MASTER_PORT:-29528}"
DATASET_ROOT="${DATASET_ROOT:-/data/jwb/data/Prot/ProteinAnn}"
ANNOTATION_ROOT="${ANNOTATION_ROOT:-/data/jwb/data/ProteinAnn}"
ANNOTATION_PATH="${ANNOTATION_PATH:-$ANNOTATION_ROOT/annotations/all_sampled.json}"
MODEL_BASE="${MODEL_BASE:-/data/jwb/ckpt/LLaVA-Video-7B-Qwen2}"
VISION_TOWER="${VISION_TOWER:-/data/jwb/ckpt/siglip-so400m-patch14-384}"
HTM_INIT_CHECKPOINT="${HTM_INIT_CHECKPOINT:-/data/jwb/ckpt/htm_stage_a_2xh100}"
SKIP_LIST_PATH="${SKIP_LIST_PATH:-/data2/jwb/QVIC-MF-Train/bad_media.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/jwb/ckpt/easa_htm_4xh100_${MODE}_$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-$(basename "$OUTPUT_DIR")}"
DS_CONFIG="$QVIC_ROOT/scripts/train/zero3.json"
LR="${LR:-1e-4}"
EASA_LAMBDA="${EASA_LAMBDA:-1.0}"
EASA_TEMPERATURE="${EASA_TEMPERATURE:-0.1}"
EASA_EPSILON="${EASA_EPSILON:-1e-6}"
EASA_OWNER_BIAS="${EASA_OWNER_BIAS:-2.0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
if [[ "$MODE" = smoke ]]; then
    REPORT_TO=none SAVE_STEPS=2 LOGGING_STEPS=1 SAVE_TOTAL_LIMIT=2
    NUM_WORKERS=1
elif [[ "$MODE" = full ]]; then
    REPORT_TO="${REPORT_TO:-wandb}"
    SAVE_STEPS="${SAVE_STEPS:-250}"
    LOGGING_STEPS="${LOGGING_STEPS:-5}"
    SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
else
    REPORT_TO="${REPORT_TO:-none}"
    SAVE_STEPS="${SAVE_STEPS:-25}"
    LOGGING_STEPS="${LOGGING_STEPS:-5}"
    SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$QVIC_ROOT:${PYTHONPATH:-}"
if [[ "$REPORT_TO" = wandb ]]; then
    export WANDB_PROJECT="${WANDB_PROJECT:-qvic-easa-htm}"
fi

# Preflight only on a real launch; dry runs also work away from the training server.
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    for file in "$ANNOTATION_PATH" "$DATASET_ROOT/metadata/shard_index.parquet" \
                "$MODEL_BASE/config.json" "$VISION_TOWER/config.json" "$SKIP_LIST_PATH" "$DS_CONFIG"; do
        if [[ ! -f "$file" ]]; then
            echo "ERROR: missing required file: $file" >&2
            if [[ "$file" = "$DATASET_ROOT/metadata/shard_index.parquet" ]]; then
                echo "Media root needs metadata/shard_index.parquet; annotation metadata may be under $ANNOTATION_ROOT/metadata." >&2
            fi
            exit 2
        fi
    done
    if [[ "$RESUME" = 0 && -e "$OUTPUT_DIR" ]]; then
        echo "ERROR: fresh training requires a new OUTPUT_DIR; do not mkdir it beforehand" >&2; exit 2
    fi
    if [[ "$RESUME" = 1 ]]; then
        if [[ ! -f "$OUTPUT_DIR/launch_config.json" ]] || ! compgen -G "$OUTPUT_DIR/checkpoint-*" >/dev/null; then
            echo "ERROR: resume requires launch_config.json and checkpoint-* under OUTPUT_DIR" >&2; exit 2
        fi
    else
        # Accept a checkpoint directory, a .bin, or the Stage A run root supplied
        # by the user. Prefer the established checkpoint-5000 over other steps.
        HTM_INIT_CHECKPOINT="$("$PYTHON" - "$HTM_INIT_CHECKPOINT" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
if path.is_file():
    print(path)
elif (path / "non_lora_trainables.bin").is_file():
    print(path)
elif (path / "checkpoint-5000/non_lora_trainables.bin").is_file():
    print(path / "checkpoint-5000")
else:
    candidates = [p for p in path.glob("checkpoint-*")
                  if p.name.removeprefix("checkpoint-").isdigit() and (p / "non_lora_trainables.bin").is_file()]
    if not candidates:
        raise SystemExit(f"ERROR: no Stage A predictor checkpoint under {path}")
    print(max(candidates, key=lambda p: int(p.name.split("-")[-1])))
PY
)"
    fi
fi

CMD=("$PYTHON" -m torch.distributed.run --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT"
    -m qvic.train.train_qvic
    --deepspeed "$DS_CONFIG" --model_base "$MODEL_BASE" --vision_tower "$VISION_TOWER"
    --dataset_root "$DATASET_ROOT" --annotation_path "$ANNOTATION_PATH" --media_backend shards
    --skip_list_path "$SKIP_LIST_PATH" --num_frames 64 --qa_mode single --qa_pick random --qa_seed 42 --seed 42
    --conv_template qwen_1_5 --context_embed_tokens 16 --context_memory_length 256 --max_frame_num_encoder 64
    --question_guided_selective_attention True --guiding_context2vision True --ctx_attn_mask_type framewise
    --fill_context_memory False --attn_implementation sdpa --encoder_attn_implementation sdpa
    --lora_r 64 --lora_alpha 16 --lora_dropout 0.05
    --use_htm True --htm_stage_a False --htm_recent_window 4 --htm_quantile 0.9 --htm_lambda 0.1
    --htm_token_budget 4096 --htm_event_representation tokens --htm_memory_layout chronological
    --htm_group_tokens True --htm_reset_thresholds_per_video True
    --use_easa True --easa_lambda "$EASA_LAMBDA" --easa_temperature "$EASA_TEMPERATURE"
    --easa_epsilon "$EASA_EPSILON" --easa_owner_bias "$EASA_OWNER_BIAS"
    --output_dir "$OUTPUT_DIR" --run_name "$RUN_NAME" --num_train_epochs 1
    --per_device_train_batch_size "$BATCH_SIZE" --gradient_accumulation_steps "$GRAD_ACCUM"
    --learning_rate "$LR" --lr_scheduler_type cosine --warmup_ratio 0.03 --weight_decay 0 --optim adamw_torch
    --bf16 True --tf32 True --gradient_checkpointing True --model_max_length 32768
    --dataloader_num_workers "$NUM_WORKERS" --dataloader_persistent_workers True
    --logging_steps "$LOGGING_STEPS" --save_strategy steps --save_steps "$SAVE_STEPS"
    --save_total_limit "$SAVE_TOTAL_LIMIT" --report_to "$REPORT_TO")
if [[ "$RESUME" = 0 ]]; then CMD+=(--htm_init_checkpoint "$HTM_INIT_CHECKPOINT"); fi
if [[ "$MODE" = pilot ]]; then CMD+=(--max_records 128 --max_steps 100); fi
if [[ "$MODE" = smoke ]]; then CMD+=(--max_records 16 --max_steps 4 --logging_nan_inf_filter False); fi
echo "EASA + HTM: mode=$MODE resume=$RESUME devices=$CUDA_VISIBLE_DEVICES port=$MASTER_PORT"
TRAINING_LENGTH="1 epoch"
if [[ "$MODE" = pilot ]]; then TRAINING_LENGTH="100 optimizer steps"; fi
if [[ "$MODE" = smoke ]]; then TRAINING_LENGTH="4 optimizer steps"; fi
echo "global batch=$NUM_GPUS * $BATCH_SIZE * $GRAD_ACCUM = $GLOBAL_BATCH_SIZE; lr=$LR; $TRAINING_LENGTH"
echo "Stage A=$HTM_INIT_CHECKPOINT (run roots resolve to checkpoint-5000, or latest available)"
echo "output=$OUTPUT_DIR; report_to=$REPORT_TO; run_name=$RUN_NAME"
if [[ "$REPORT_TO" = wandb ]]; then
    echo "W&B project=$WANDB_PROJECT entity=${WANDB_ENTITY:-account default} mode=${WANDB_MODE:-online}"
fi
printf 'Command: '; printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" = 1 ]]; then echo "DRY RUN: no files written, training not started"; exit 0; fi

# Store the exact command, data exclusion hash and semantic arguments. A resume
# must keep the data, model and training recipe; logging destinations may change.
"$PYTHON" - "$OUTPUT_DIR" "$RESUME" "$MODE" "$SKIP_LIST_PATH" "$CUDA_VISIBLE_DEVICES" "${CMD[@]}" <<'PY'
import hashlib, json, os, subprocess, sys
from pathlib import Path
out, resume, mode, skip, devices = sys.argv[1:6]
command = sys.argv[6:]
args = command[command.index("qvic.train.train_qvic") + 1:]
options = dict(zip(args[::2], args[1::2]))
signature = {k: v for k, v in options.items() if k not in ("--htm_init_checkpoint", "--run_name", "--report_to")}
signature["skip_list_sha256"] = hashlib.sha256(Path(skip).read_bytes()).hexdigest()
path = Path(out) / "launch_config.json"
if resume == "1":
    saved = json.loads(path.read_text())
    if saved.get("mode") != mode or saved.get("signature") != signature:
        raise SystemExit("ERROR: resume data/model/training settings differ from launch_config.json; use the original settings")
else:
    Path(out).mkdir(parents=True, exist_ok=False)
record = dict(mode=mode, command=command, signature=signature, cuda_visible_devices=devices,
              git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
              environment={k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_MODE")})
target = Path(out) / ("resume_config.json" if resume == "1" else "launch_config.json")
target.write_text(json.dumps(record, indent=2) + "\n")
PY
set +e
"${CMD[@]}" 2>&1 | tee -a "$OUTPUT_DIR/train.log"
RESULTS=("${PIPESTATUS[@]}")
set -e
echo "training_exit=${RESULTS[0]} output_dir=$OUTPUT_DIR"
if [[ "${RESULTS[0]}" != 0 ]]; then exit "${RESULTS[0]}"; fi
exit "${RESULTS[1]}"
