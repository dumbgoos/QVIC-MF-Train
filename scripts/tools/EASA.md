# Event-Aware Selective Attention

The first implementation applies the paper's event prior to **decoder visual
memory self-attention**, after HTM writes the observed video. It does not alter
the visual encoder, compressor QMSA, event boundaries, or eviction policy.
The prior is shared across decoder heads/layers and introduces no new weights.

## Soft event membership

The manuscript defines `R = P P^T` but leaves the construction of `P` unspecified.
This implementation makes that choice explicit:

```
logit[i, e] = cosine(mean(memory_unit_i), event_prototype_e) / temperature
              + owner_bias * 1[e == HTM_owner(i)]
P[i] = softmax(logit[i])
R = P @ P.T
bias = easa_lambda * log(R + easa_epsilon)
```

All discovered events, including the open event, are prototypes. HTM's temporal
intervals identify ownership for both event units and recent observations. The
finite ownership prior distinguishes visually identical but temporally separate
events; it allows cross-event membership rather than imposing a hard mask.
Setting `owner_bias=0` ablates this prior. Defaults (lambda=1, temperature=0.1,
epsilon=1e-6, owner_bias=2) are implementation defaults, not tuned paper results.

One membership vector is repeated across a unit's retained tokens. Pooling for
membership does not replace or detach the readable memory values. Assignments,
HTM prototypes and routing are stop-gradient; QA gradients still flow through
decoder attention into Stage B compressor LoRA/context seeds. The GRU retains
its existing prediction-loss supervision.

Memberships follow exactly the same eviction, chronological sorting, token
truncation and grouping as readable HTM memory. Text, padding and inserted
newline tokens have zero membership and receive no EASA bias. Both endpoints
must be visual tokens for the bias to apply. The original causal/padding mask
is retained; EASA does not open future-token attention. Cached answer queries
receive no direct prior, but attend to memory states computed with EASA during
prefill. Greedy and beam generation propagate memberships explicitly.

## Training and inference

EASA is off by default. Training requires HTM and a Qwen2 decoder using `sdpa`
or `eager`; FlashAttention-2 does not accept this dense structural bias.

To use an existing HTM launcher, append:

```bash
EXTRA_ARGS="--use_easa True --easa_lambda 1.0 --easa_temperature 0.1 --easa_epsilon 1e-6 --easa_owner_bias 2.0" \
bash scripts/train/train_htm_stage_b_2xh100.sh
```

For the chronological C=16 / 4096-token / grouped recipe (default pilot uses
128 records and 100 optimizer steps), with training-machine paths configured:

```bash
DRY_RUN=1 bash scripts/train/train_easa_stage_b_2xh100.sh

DATASET_ROOT=/data/LLaVA-Video-83K \
ANNOTATION_PATH=/data/LLaVA-Video-83K/annotations/all_sampled.json \
HTM_INIT_CHECKPOINT=/checkpoints/htm_stage_a/checkpoint-5000 \
OUTPUT_DIR=/checkpoints/easa_pilot \
bash scripts/train/train_easa_stage_b_2xh100.sh
```

`MODE=full` runs the full epoch; `RESUME=1` resumes the same output directory.
The EASA wrapper initializes only the predictor from Stage A, with fresh
compressor LoRA/context seeds, following the existing grouped recipe. Set
`EASA_LAMBDA`, `EASA_TEMPERATURE`, `EASA_EPSILON`, and `EASA_OWNER_BIAS` to override
defaults. Use a new output directory for each ablation.

The enable flag and all four numeric parameters are saved in `config.json` and
restored by the existing QViC checkpoint loader. Old checkpoints stay disabled.
LVBench accepts `--use_easa True/False`, `--easa_lambda`, `--easa_temperature`,
`--easa_epsilon`, and `--easa_owner_bias`; omitted flags retain checkpoint values.
An old HTM checkpoint can enable EASA at inference without extra weights.
The checkpoint inference diagnostic also accepts `--easa` / `--no-easa` and
hyphenated numeric options, and includes their settings in its report.

For a controlled comparison, keep checkpoint, questions, frame sampling,
memory representation/budget and feedback fixed; compare disabled, lambda=0,
and positive lambda. The first two should produce identical logits. Changing
the prior after training is an inference ablation, not joint EASA training.

## Validation and limits

### Four-H100 local-path launcher

`scripts/train/train_easa_htm_4xh100.sh` is an independent launcher, so it avoids
the hard-coded two-GPU settings in the older wrapper. It defaults to a full
epoch, GPUs `4,5,6,7`, per-GPU batch 1 and accumulation 8 (global batch 32).
`BATCH_SIZE=2` automatically uses accumulation 4. The learning rate stays
`1e-4`; saving every 250 updates and logging every 5 updates approximately
preserve the old batch-16 recipe's sample-based frequencies. Pilot mode also
uses global batch 32, with 128 records / 100 updates.

Default server paths:

| Input | Path |
| --- | --- |
| Media and shard index | `/data/jwb/data/Prot/ProteinAnn` |
| Annotations | `/data/jwb/data/ProteinAnn/annotations/all_sampled.json` |
| Base model | `/data/jwb/ckpt/LLaVA-Video-7B-Qwen2` |
| Vision encoder | `/data/jwb/ckpt/siglip-so400m-patch14-384` |
| Stage A run | `/data/jwb/ckpt/htm_stage_a_2xh100` |
| Bad-media list | `/data2/jwb/QVIC-MF-Train/bad_media.json` |

The Stage A run root resolves to its own `non_lora_trainables.bin` if present,
otherwise `checkpoint-5000`, otherwise the highest numbered checkpoint with
that file. The resolved path is printed and recorded. Override
`HTM_INIT_CHECKPOINT` to select a specific checkpoint explicitly.

Local vision weights are selected through `--vision_tower`, overriding
`mm_vision_tower` before constructing the model; this value is also saved in
the checkpoint config. The vision factory checks SigLIP names before the
generic existing-directory CLIP fallback. Offline mode is on by default.

```bash
conda activate video
cd /data2/jwb/QVIC-MF-Train

# Preview only; does not check server files, create outputs or start training.
CUDA_VISIBLE_DEVICES=4,5,6,7 DRY_RUN=1 \
  bash scripts/train/train_easa_htm_4xh100.sh

# New pilot, no W&B reporting.
CUDA_VISIBLE_DEVICES=4,5,6,7 MODE=pilot REPORT_TO=none \
  bash scripts/train/train_easa_htm_4xh100.sh

# New full epoch, also reported to W&B.
CUDA_VISIBLE_DEVICES=4,5,6,7 MODE=full REPORT_TO=wandb \
  bash scripts/train/train_easa_htm_4xh100.sh

# Resume this launcher's run, keeping its original mode and batch settings.
CUDA_VISIBLE_DEVICES=4,5,6,7 MODE=full RESUME=1 \
OUTPUT_DIR=/data/jwb/ckpt/ACTUAL_EASA_RUN_DIRECTORY \
  bash scripts/train/train_easa_htm_4xh100.sh
```

Do not pre-create a fresh `OUTPUT_DIR`. The launcher creates it and writes
`train.log` internally. Use a new output directory for pilot versus full runs;
an inherited `OUTPUT_DIR`/`RUN_NAME` can be cleared with `unset OUTPUT_DIR RUN_NAME`.
It validates inputs before training, uses master port 29528, records the full
command plus skip-list SHA-256, and rejects resumes with changed data, batch,
model or recipe settings. It does not resume old two-GPU launcher runs.
Logging backend and run name may change on resume. Do not change the bad-media
list mid-run when exact data-order continuation matters.

Full mode defaults to `REPORT_TO=wandb` and project `qvic-easa-htm`;
pilot defaults to `REPORT_TO=none`. Run `wandb login` in the training environment first, and set
`WANDB_MODE=online` to override a previously inherited offline setting. A
previously exported `REPORT_TO=none` still takes precedence over the default;
the full-training command above explicitly enables W&B. `WANDB_ENTITY` can select an account
or team, and `RUN_NAME` controls the displayed run name.

The media directory must include `metadata/shard_index.parquet`; merely having
the index under the annotation directory is insufficient for the current
ShardReader. W&B authentication is separate from offline model loading.
Run `python scripts/tools/test_easa_4gpu_launcher.py` for temporary-directory
tests with a fake torchrun (no model/GPU execution), including local SigLIP
dispatch, path wiring, checkpoint selection, batch derivation and resume guards.

### GPU-0 full-epoch training

`scripts/train/train_easa_htm_gpu0.sh` runs the full EASA + HTM recipe on
physical GPU 0 with microbatch 1, accumulation 32 and effective batch 32.
It explicitly selects full mode with no record/step limit, keeps 64 frames,
all model/memory/optimizer settings and server paths, reports to W&B by
default, uses port 29530 and writes to a new `easa_htm_gpu0_full_*` directory.
It starts from Stage A predictor weights with fresh compressor LoRA/context
seeds; it does not continue the four-step smoke checkpoint.

```bash
conda activate proteinnovo
cd /data2/jwb/QVIC-MF-Train
unset OUTPUT_DIR RUN_NAME
DRY_RUN=1 bash scripts/train/train_easa_htm_gpu0.sh
REPORT_TO=wandb WANDB_MODE=online bash scripts/train/train_easa_htm_gpu0.sh

# Resume this same single-GPU run, using its actual output path:
RESUME=1 OUTPUT_DIR=/data/jwb/ckpt/ACTUAL_GPU0_FULL_RUN \
REPORT_TO=wandb WANDB_MODE=online bash scripts/train/train_easa_htm_gpu0.sh
```

Keep the working proxy environment from the successful W&B connectivity test.
Saving remains every 250 optimizer updates, logging every 5, and training is
one epoch (approximately 2593 updates for the 82,991-sample filtered corpus).
Each optimizer step processes 32 microbatches on one GPU, so the interval
between progress updates is much longer than the four-step batch-1 smoke.
`LOGGING_STEPS=1` can make full-training metric updates more frequent, but
cannot shorten a single optimizer step. Gradient accumulation does not keep
all 32 microbatch forward graphs alive simultaneously.

### GPU-0 pretrained training smoke

```bash
conda activate video
cd /data2/jwb/QVIC-MF-Train
unset OUTPUT_DIR RUN_NAME
DRY_RUN=1 bash scripts/train/test_easa_htm_gpu0.sh
bash scripts/train/test_easa_htm_gpu0.sh
```

This wrapper forces physical GPU 0, one process, microbatch/accumulation 1,
16 annotation records and four optimizer updates. It keeps the full recipe's
64 frames, C=16, 4096-token grouped HTM memory, EASA, LoRA, BF16, SDPA and ZeRO-3.
It uses the same local data/model/Stage A/skip-list paths as the four-GPU launcher,
with default port 29529, a separate `easa_htm_smoke_gpu0_*` output directory and
no W&B reporting. It does not resume or touch the full training output.

The wrapper requires checkpoints at steps 2 and 4, finite logged training
losses, saved EASA/HTM configuration, and stored parameter updates in compressor
LoRA, context seed and predictor between those checkpoints. Any training or
postflight failure returns nonzero. Read `train.log` and
`smoke_parameter_updates.json` in its output directory. No accuracy threshold
or loss-decrease requirement is imposed on a four-step smoke.

A single GPU holds more model state than each rank of a four-GPU ZeRO-3 job;
single-GPU OOM therefore does not establish that four-GPU training cannot fit.
This smoke does not test checkpoint reload, distributed execution or accuracy.
Local launcher tests simulate GPU execution only; run this command on the H100
server to validate actual pretrained forward/backward and checkpoint writing.

### Controller and decoder tests

```bash
python scripts/tools/test_easa.py
python scripts/tools/test_easa_decoder.py
python scripts/tools/check_htm_gradients.py
```

The first script needs PyTorch only. It checks the affinity formula, soft
ownership of identical prototypes, SDPA/eager math, causal/padding preservation,
zero-strength identity, reduced precision, cached offsets, newline/truncation/
batch padding alignment, and HTM metadata across layouts and eviction.

The decoder test additionally needs Transformers. It uses the real local
Qwen2 and LlavaQwen forward/generate code, actual QViC packing/newline methods,
and synthetic compressed vision features. It checks eager/SDPA agreement,
lambda=0 parity, cached/full agreement, reentrant/non-reentrant checkpoint
gradients, QA gradients to an upstream projection, greedy/beam generation,
inference cleanup and config round-trip. It downloads no model weights.

The prior is calculated over already observed HTM memory. It does not implement
event-aware compressor attention or a second video pass. The affinity/mask has
quadratic decoder-sequence storage, and event assignment cost grows with the
number of discovered events. GPU peak memory, the full pretrained stack,
distributed training/resume and LVBench accuracy require a training-machine
pilot; CPU success does not establish an accuracy improvement.

Local validation: Python 3.12.14, PyTorch 2.5.1 (CPU on macOS), Transformers
4.40.0 release. Both EASA tests, all eight `test_htm_*.py` scripts and
`check_htm_gradients.py` passed. Shell syntax and the pilot launcher dry run
also passed. This is not the README's Python 3.10 / pinned Transformers Git
commit / CUDA environment. Full GPU/pretrained execution has not been run.
