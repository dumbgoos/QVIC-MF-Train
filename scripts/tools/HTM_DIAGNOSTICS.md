# HTM diagnostics

Run from the repository root in a Python environment with PyTorch. The four
default checks use CPU tensors and load `htm.py` directly; they need no model
weights or video data. Assertions fail with a nonzero exit status. Budget and
policy scripts print observations rather than treating known design issues as
test failures.

## 1. Training gradients

```bash
python scripts/tools/check_htm_gradients.py
PYTHONPATH=. python scripts/tools/check_htm_gradients.py --full-model
```

The CPU regression checks QA gradients into a small upstream projection and
context seed, prediction gradients into the GRU, frozen decoder parameters,
merged/unmerged events, and identical forward values with/without gradient
retention. It also checks Stage A keeps upstream features detached. The optional
full-model check runs the existing tiny Qwen/SigLIP training and checkpoint
self-test in HTM Stage B mode. It needs the README training dependencies and
downloaded tokenizer/SigLIP weights. Synthetic success alone does not establish
that a real training job or historical checkpoint is correct.

## 2. Memory budget

```bash
python scripts/tools/check_htm_budget.py
python scripts/tools/check_htm_budget.py --frames 2048 --htm-tokens 4096
```

Reports actual materialized tokens for the current cap and a cap equal to
`mf_entries * context_tokens`. MF's `L=256` counts frame entries; HTM's `L=256`
counts individual tokens. Enlarging the cap may leave it underfilled because
events have already been pooled. Use matched question IDs, sampling, weights,
and feedback settings for any subsequent LVBench accuracy comparison. Changing
the shared `context_memory_length` also changes the relevance-tracking cap;
do not treat that alone as an isolated write-budget ablation.

## 3. Event representation

```bash
python scripts/tools/check_htm_representation.py
```

Constructs different token sets with identical means and reversed frame orders
with identical event vectors. This demonstrates information loss, not a measured
accuracy drop. A learned one-token compressor is different from averaging a
compressor trained to emit sixteen tokens. A QA ablation retaining the token
axis is still needed before choosing a replacement representation.

## 4. Memory policy

```bash
python scripts/tools/check_htm_policy.py
```

Reports temporal ordering, overlap between recent frames and events, a tight
budget case that evicts the newest recent, threshold-buffer persistence after
reset, and the default relevance weight. Temporal markers are synthetic and
do not rely on the visual features themselves encoding timestamps.

## Implemented fix and limits

The controller accepts `retain_memory_grad`; the model enables it only during
Stage B training. Readable event/recent values keep their graph, while predictor
inputs/targets and routing statistics stay detached. Stage A and inference
retain the old behavior. This increases training graph memory; validate GPU
memory on a small run before a long job. Discrete merge/eviction decisions remain
outside QA autograd, so the predictor still learns from prediction loss.

The default budget, event pooling, eviction priorities, and ordering are unchanged.
Evaluation now resets threshold history for each new video/question; training
keeps rolling threshold history. `check_htm_policy.py` reports the controller's
default reset behavior, which intentionally retains training history. Run
`python scripts/tools/test_htm_eval_threshold_reset.py` to check the eval reset
and synthetic order independence. This inference fix applies to existing
checkpoints, but fixing gradients cannot repair their past training;
validate and retrain Stage B before drawing accuracy conclusions.

## Local validation

Validated on CPU with Python 3.14.6 / PyTorch 2.14.1+cpu in a temporary
`/tmp/htm-diagnostic-deps` installation. All four diagnostics and the existing
`test_htm_stage_a.py` / `test_htm_kr_relevance_buffers.py` completed successfully.
The original controller was separately reproduced with detached readable memory,
no upstream gradient, and nonzero predictor gradients. With identical synthetic
inputs (512 frames, C=16), caps of 256 and 4096 both materialized 182 tokens:
pooling had already reduced the representation below either cap.

The full-model check, GPU execution, pinned PyTorch 2.5.1 environment, and
LVBench accuracy were not validated locally. CPU diagnostics establish controller
behavior; they do not establish accuracy recovery for the trained model.
