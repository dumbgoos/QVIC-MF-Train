# QViC-MF-Train

**An unofficial, from-scratch reconstruction of the training code for
QViC-MF (CVPR 2026).**

The [official release](https://github.com/FujitsuResearch/QViC-MF) of
*Question-guided Visual Compression with Memory Feedback for Long-Term Video
Understanding* states that "Training code is **not** included in this release".
This repository reconstructs the supervised fine-tuning stage from the paper's
supplementary material, on top of the released model code — **no released file
was modified**, and everything the model does at train time is the same code
path the official evaluator uses.

> **Status.** The pipeline runs end to end on GPUs and the checkpoints it
> produces reload through the released `qvic.model.builder.load_pretrained_model`.
> A full 83k-sample epoch has **not** finished, and **no benchmark evaluation has
> been run**, so this is a faithful reconstruction of the described procedure —
> not a confirmed replication of the paper's numbers.

> **Not affiliated with Fujitsu Research, Macquarie University, or the paper's
> authors.** Any error here is this repository's, not theirs.

---

## What is trained

| Component | State |
| --- | --- |
| Visual encoder (SigLIP) + `mm_projector` | frozen |
| Visual compressor (encoder LLM, Qwen2-7B) | **LoRA** r=64, α=16, dropout=0.05 |
| Context seed embedding | **trainable** |
| Decoder LLM (Qwen2-7B) | frozen, no LoRA |

~1.6 % of the 14 B parameters. Hyper-parameters follow Table 1 of the
supplementary material: batch 1 × grad-accum 4, lr 1e-4 cosine, warm-up 0.03,
weight decay 0, 1 epoch, AdamW, DeepSpeed ZeRO-3, `C=16` context tokens,
`L=256` memory capacity, `K=64` clip frames, `K_r=0`.

Every knob is on the command line — `python -m qvic.train.train_qvic --help`,
or the environment variables at the top of `scripts/train/train_qvic_83k.sh`.

## Repository contents

```
qvic/
├── model/, constants.py, conversation.py, mm_utils.py, utils.py, eval/
│                        # verbatim copy of the upstream QViC-MF package
└── train/               # NEW — the reconstruction
    ├── train_qvic.py    #   entry point: assemble, freeze, drive
    ├── data.py          #   corpus -> single-question samples + collator
    ├── shard_reader.py  #   packed tar-shard media reader
    └── qvic_trainer.py  #   Trainer subclass writing QViC-shaped checkpoints
scripts/
├── train/               # launcher, DeepSpeed configs, CPU self-test
├── tools/scan_media.py  # find media that crashes the decoder
└── cluster/             # batch-scheduler job examples (Fujitsu TCS / pjsub)
```

## Installation

Python 3.10, CUDA 12.1, PyTorch 2.5.1 — the same stack as upstream, plus
DeepSpeed.

```bash
git clone https://github.com/dumbgoos/QVIC-MF-Train.git
cd QVIC-MF-Train
conda create -y -p ./env python=3.10.14 pip && conda activate ./env

pip install "setuptools==75.8.0"
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[train]"
```

Two pins that upstream does not give you, and without which the install breaks:

* **`setuptools==75.8.0`**, installed first — pip now pulls setuptools 84+,
  which has dropped `pkg_resources`; several transitive dependencies still
  import it. This one has to be done by hand, before anything else.
* **`scipy<1.15`** — `qvic/model/qvic_meta_mixin.py` imports
  `scipy.ndimage.filters`, removed in SciPy 1.15. Upstream's `pyproject.toml`
  leaves scipy unbounded, so a fresh install lands on 1.15.x and fails at
  import; the bound is applied in this repository's `pyproject.toml`.

Reference environment: Python 3.10.14 · torch 2.5.1+cu121 · transformers
4.40.0.dev0 (the pinned upstream commit) · peft 0.10.0 · accelerate 0.29.3 ·
deepspeed 0.14.4 · numpy 1.26.4 · scipy 1.14.1 · tokenizers 0.15.2.

Nothing on the default path JIT-compiles a CUDA extension, so no compiler is
needed on the compute node. The one exception is
`scripts/train/zero3_offload.json`, which pulls in `DeepSpeedCPUAdam`.

## Quick start

**1. Check the wiring without a GPU** (~1 min — builds a 2-layer model with the
real SigLIP tower and runs one step end to end, asserting what is trainable,
what gets gradient, and what lands in the checkpoint):

```bash
PYTHONPATH=. python scripts/train/selftest_train_cpu.py
```

**2. Prepare the corpus** — an 83 000-record subset of
[LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K),
in the **upstream annotation schema, unchanged**: a JSON list of records with a
`video` field and alternating `human` / `gpt` turns under `conversations`. The
paper does not say which 83 000, so a uniform random draw is a reasonable
choice.

Two media backends, selected with `MEDIA_BACKEND`:

* `loose` — an ordinary directory tree. `VIDEO_FOLDER` is prepended to each
  record's `video` field; a `video` that resolves to a *directory* is read as a
  frame folder, which is what the `llava_hound` split needs (its media lives in
  [`ShareGPTVideo/train_video_and_instruction`](https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction),
  not in LLaVA-Video-178K).
* `shards` (default) — uncompressed tar shards plus
  `metadata/shard_index.parquet` mapping `member → (shard, offset, nbytes,
  kind)`, so any file is one `pread` away and nothing is unpacked before
  training. Members over 64 MiB stay loose under `videos_large/` with an empty
  `shard`. Reader: `qvic/train/shard_reader.py`.

A handful of files crash `decord` outright, which kills the dataloader worker
and takes the job with it. `scripts/tools/scan_media.py` decodes every medium in
a forked child to find them (8 bad out of 71 224 on our subset, 2 of them
`SIGSEGV`) and writes the skip list training consumes.

**3. Train:**

```bash
DATASET_ROOT=/data/LLaVA-Video-83K NUM_GPUS=8 \
OUTPUT_DIR=ckpt/QViC-MF-7B-repro \
bash scripts/train/train_qvic_83k.sh
```

Re-running the same command resumes from the newest `checkpoint-*`. One epoch is
~51 h on 4 × H100 or ~25 h on 8, so expect to submit it more than once against a
wall clock; [`scripts/cluster/`](scripts/cluster) has batch-scheduler
wrappers that add automatic resume and tolerance for decoder crashes.

## Licence

CC-BY-NC-SA-4.0, inherited from the upstream release. See [`LICENSE`](LICENSE)
and [`NOTICE`](NOTICE).

The base model `lmms-lab/LLaVA-Video-7B-Qwen2` and the LLaVA-Video-178K dataset
carry their own terms; check them before any downstream use of a trained
adapter.
