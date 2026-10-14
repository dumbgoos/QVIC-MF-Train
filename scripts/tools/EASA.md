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
