"""Exercise the real four-GPU launcher with a fake torchrun and temporary paths."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import types

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts/train/train_easa_htm_4xh100.sh"


def test_vision_dispatch(tmp: Path):
    # Exercise the actual factory without importing the VLM backends.
    source = ROOT / "qvic/model/multimodal_encoder/builder.py"
    tree = ast.parse(source.read_text())
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_vision_tower")
    namespace = {"os": os}
    for name in ("SigLipVisionTower", "CLIPVisionTower", "CLIPVisionTowerS2", "HFVisionTower", "ImageBindWrapper", "OpenCLIPVisionTower"):
        namespace[name] = lambda *a, _name=name, **kw: (_name, a, kw)
    exec(compile(ast.Module(body=[factory], type_ignores=[]), str(source), "exec"), namespace)
    local_siglip = tmp / "siglip-so400m-patch14-384"
    local_clip = tmp / "clip-local"
    local_siglip.mkdir(); local_clip.mkdir()
    for name in (str(local_siglip), "google/siglip-so400m-patch14-384"):
        result = namespace["build_vision_tower"](types.SimpleNamespace(mm_vision_tower=name))
        assert result[0] == "SigLipVisionTower"
    assert namespace["build_vision_tower"](types.SimpleNamespace(mm_vision_tower=str(local_clip)))[0] == "CLIPVisionTower"


def main():
    with tempfile.TemporaryDirectory(prefix="easa-4gpu-") as temporary:
        tmp = Path(temporary)
        test_vision_dispatch(tmp)
        base, vision, media, annotation, stage_a = [tmp / n for n in ("base", "vision", "media", "annotation", "stage_a")]
        for path in (base, vision, media / "metadata", annotation / "annotations", stage_a / "checkpoint-5000", stage_a / "checkpoint-6000"):
            path.mkdir(parents=True)
        for model in (base, vision):
            (model / "config.json").write_text("{}")
        (media / "metadata/shard_index.parquet").touch()
        (annotation / "annotations/all_sampled.json").write_text("[]")
        for step in (5000, 6000):
            (stage_a / f"checkpoint-{step}/non_lora_trainables.bin").touch()
        skip = tmp / "bad_media.json"
        skip.write_text("[]")
        # Delegate all stdlib preflight Python to the current interpreter. Only
        # torchrun is replaced: capture the command instead of touching GPUs.
        shim = tmp / "python-shim"
        shim.write_text("#!/bin/bash\nif [[ \"${1:-}\" = -m && \"${2:-}\" = torch.distributed.run ]]; then\n"
                        "  printf '%s\\n' \"$@\" > \"$CAPTURE_COMMAND\"\n  exit 0\nfi\n"
                        f"exec {shlex.quote(sys.executable)} \"$@\"\n")
        shim.chmod(0o755)
        env = os.environ.copy()
        env.update(QVIC_ROOT=str(ROOT), QVIC_PY=str(shim), CUDA_VISIBLE_DEVICES="4,5,6,7", BATCH_SIZE="1",
                   DATASET_ROOT=str(media), ANNOTATION_ROOT=str(annotation),
                   ANNOTATION_PATH=str(annotation / "annotations/all_sampled.json"),
                   MODEL_BASE=str(base), VISION_TOWER=str(vision), HTM_INIT_CHECKPOINT=str(stage_a),
                   SKIP_LIST_PATH=str(skip), OUTPUT_DIR=str(tmp / "full-run"), RUN_NAME="easa-test",
                   REPORT_TO="none", CAPTURE_COMMAND=str(tmp / "command.txt"), MODE="full", RESUME="0", DRY_RUN="0")

        def run(**overrides):
            result = subprocess.run(["bash", str(LAUNCHER)], env={**env, **overrides}, text=True, capture_output=True)
            return result

        result = run()
        assert result.returncode == 0, result.stdout + result.stderr
        launch = json.loads((tmp / "full-run/launch_config.json").read_text())
        args = launch["signature"]
        assert args["--per_device_train_batch_size"] == "1" and args["--gradient_accumulation_steps"] == "8"
        assert args["--use_easa"] == args["--use_htm"] == "True"
        assert args["--model_base"] == str(base) and args["--vision_tower"] == str(vision)
        assert args["--dataset_root"] == str(media) and args["--skip_list_path"] == str(skip)
        assert args["--save_steps"] == "250" and "--max_steps" not in args
        assert launch["command"][-1] == str(stage_a / "checkpoint-5000")
        assert "--nproc_per_node=4" in launch["command"] and "--master_port=29528" in launch["command"]
        assert launch["cuda_visible_devices"] == "4,5,6,7"
        assert (tmp / "full-run/train.log").is_file()
        assert run().returncode != 0  # fresh output must not overwrite an old run
        (tmp / "full-run/checkpoint-250").mkdir()
        assert run(RESUME="1").returncode == 0
        assert run(RESUME="1", LR="2e-4").returncode != 0
        skip.write_text('["new_bad_media"]')
        assert run(RESUME="1").returncode != 0
        skip.write_text("[]")
        pilot = tmp / "pilot-run"
        result = run(MODE="pilot", OUTPUT_DIR=str(pilot), BATCH_SIZE="2")
        assert result.returncode == 0, result.stdout + result.stderr
        args = json.loads((pilot / "launch_config.json").read_text())["signature"]
        assert args["--per_device_train_batch_size"] == "2" and args["--gradient_accumulation_steps"] == "4"
        assert args["--max_records"] == "128" and args["--max_steps"] == "100"
        dry = tmp / "dry"
        assert run(DRY_RUN="1", OUTPUT_DIR=str(dry)).returncode == 0 and not dry.exists()
        assert run(DRY_RUN="1", BATCH_SIZE="3").returncode != 0
        assert run(DRY_RUN="1", CUDA_VISIBLE_DEVICES="4,5").returncode != 0
        assert run(DRY_RUN="1", CUDA_VISIBLE_DEVICES="4,5,6,6").returncode != 0
        assert run(OUTPUT_DIR=str(tmp / "missing-file"), SKIP_LIST_PATH=str(tmp / "absent")).returncode != 0
    print("PASS local SigLIP dispatch, four-GPU batch=32 commands, Stage A selection, full/pilot and local paths")
    print("PASS fresh/resume guards, skip-list hash, dry-run isolation and invalid GPU/batch rejection")


if __name__ == "__main__":
    main()
