# Repository Guidelines

## Project Structure & Module Organization

This repository reconstructs QViC-MF training and adds HTM memory functionality. `qvic/model/` contains model construction, multimodal components, and the HTM controller (`htm.py`). `qvic/train/` contains the training entry point, dataset/collator, tar-shard reader, and checkpoint-saving trainer. `qvic/eval/` contains benchmark inference and HTM evaluation helpers.

`scripts/train/` holds training launchers, DeepSpeed JSON configurations, and the CPU self-test; `scripts/tools/` contains media scanning and HTM checks. Evaluation wrappers live in `scripts/eval/`, and scheduler examples live in `scripts/cluster/`. Model configuration assets are bundled under the vision encoder directories. Keep datasets and generated checkpoints outside tracked source files.

## Build, Test, and Development Commands

Use Python 3.10 and the pinned CUDA/PyTorch dependency stack documented in `README.md`.

- `pip install -e ".[train]"`: install the editable package with training dependencies; follow the README's prerequisite installation steps first.
- `python -m qvic.train.train_qvic --help`: inspect training arguments.
- `PYTHONPATH=. python scripts/train/selftest_train_cpu.py`: validate training, gradients, and checkpoint serialization on a small model; requires the model dependencies and SigLIP weights.
- `python scripts/tools/test_htm_stage_a.py`: run the CPU HTM write-path smoke test without VLM weights.
- `python scripts/tools/test_htm_kr_relevance_buffers.py`: check HTM relevance-buffer bookkeeping.
- `DATASET_ROOT=/data/LLaVA-Video-83K NUM_GPUS=8 OUTPUT_DIR=ckpt/repro bash scripts/train/train_qvic_83k.sh`: launch GPU training with automatic checkpoint resume.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions and variables, and `PascalCase` classes. Match surrounding code and add type annotations to new training utilities. Black is configured in `pyproject.toml` with a 240-character line length; no lint runner is configured. Use uppercase environment variables in shell launchers and keep configuration overrides explicit.

## Testing Guidelines

Tests are executable Python scripts using assertions, rather than a configured pytest suite. Name focused checks `test_<feature>.py` under `scripts/tools/`. Run the relevant checks for changes to memory behavior, training, or checkpoint compatibility. No coverage threshold is configured. For GPU changes, document the launcher, hardware, and smoke-run outcome.

## Commit & Pull Request Guidelines

Recent commits use scoped subjects such as `fix(htm): ...` and `feat(htm): ...`. Keep commits focused and use imperative descriptions. PRs should explain the behavior change, affected training/evaluation settings, and validation commands/results; link related issues when applicable. Avoid committing datasets, weights, logs, credentials, or machine-specific paths. Preserve the licensing notices in `LICENSE` and `NOTICE`.
