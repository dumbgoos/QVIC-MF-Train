"""Reload an HTM checkpoint and validate a small real LVBench inference run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def read_rows(path: Path) -> list[dict]:
    raw = path.read_text().strip()
    return json.loads(raw) if raw.startswith("[") else [json.loads(line) for line in raw.splitlines() if line.strip()]


def select_rows(rows: list[dict], video_dir: Path, count: int) -> list[dict]:
    candidates = []
    seen_keys = set()
    for row in rows:
        if not all(key in row for key in ("key", "question", "answer", "question_type")):
            raise ValueError("Expected formatted LVBench QA rows; use scripts/lvbench_format_annotation.py for raw metadata")
        key = row["key"]
        if not isinstance(key, str) or len(key) < 13 or key[11] != "_":
            raise ValueError(f"Invalid formatted LVBench key: {key!r}")
        if key not in seen_keys and (video_dir / f"{key[:11]}.mp4").is_file():
            candidates.append(row)
            seen_keys.add(key)
    # Prefer different videos to exercise state cleanup between streams.
    selected, videos = [], set()
    for row in candidates:
        if row["key"][:11] not in videos:
            selected.append(row)
            videos.add(row["key"][:11])
            if len(selected) == count:
                return selected
    keys = {row["key"] for row in selected}
    selected.extend(row for row in candidates if row["key"] not in keys)
    if len(selected) < count:
        raise ValueError(f"Need {count} QA rows with existing videos, found {len(selected)}")
    return selected[:count]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--model-base", default="lmms-lab/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--num-questions", type=int, default=2)
    parser.add_argument("--frames", type=int, default=64, help="Uniformly sampled frames per video, not full-video accuracy evaluation")
    parser.add_argument("--feedback", choices=("on", "off"), default="on")
    parser.add_argument("--memory-tokens", type=int, default=None)
    parser.add_argument("--event-representation", choices=("pooled", "tokens"), default=None)
    parser.add_argument("--memory-layout", choices=("legacy", "chronological", "frames"), default=None)
    parser.add_argument("--group-tokens", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--quantile", type=float, default=0.9)
    parser.add_argument("--prepare-only", action="store_true", help="Validate paths and write input/command without loading the model")
    args = parser.parse_args()
    if min(args.num_questions, args.frames) <= 0 or (args.memory_tokens is not None and args.memory_tokens <= 0) or not 0 <= args.quantile <= 1:
        parser.error("Counts must be positive and quantile must be in [0, 1]")
    args.outdir = args.outdir.resolve()
    try:
        for filename in ("config.json", "adapter_config.json", "adapter_model.bin", "non_lora_trainables.bin"):
            if not (args.checkpoint / filename).is_file():
                raise ValueError(f"Missing checkpoint file: {args.checkpoint / filename}")
        config = json.loads((args.checkpoint / "config.json").read_text())
        args.memory_tokens = args.memory_tokens if args.memory_tokens is not None else config.get("htm_token_budget", 256)
        args.event_representation = args.event_representation or config.get("htm_event_representation", "pooled")
        args.memory_layout = args.memory_layout or config.get("htm_memory_layout", "legacy")
        args.group_tokens = args.group_tokens if args.group_tokens is not None else config.get("htm_group_tokens", False)
        selected = select_rows(read_rows(args.questions), args.video_dir, args.num_questions)
        artifacts = ("questions.jsonl", "predictions.json", "inference.log", "report.json", "command.json")
        if any((args.outdir / name).exists() for name in artifacts):
            raise ValueError("Output contains prior diagnostic artifacts; choose a new --outdir")
        args.outdir.mkdir(parents=True, exist_ok=True)
        subset = args.outdir / "questions.jsonl"
        subset.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
        command = [sys.executable, "-u", "-m", "qvic.eval.model_video_lvbench",
                   "--model-path", str(args.checkpoint.resolve()), "--model-base", args.model_base,
                   "--video_dir", str(args.video_dir.resolve()), "--question_fp", str(subset),
                   "--output_dir", str(args.outdir), "--output_name", "predictions",
                   "--frames_num", str(args.frames), "--min_frames_num", str(args.frames),
                   "--force_sample", "True", "--video_fps", "1",
                   "--context_memory_length", "256", "--htm_token_budget", str(args.memory_tokens),
                   "--htm_event_representation", args.event_representation,
                   "--htm_memory_layout", args.memory_layout,
                   "--htm_group_tokens", str(args.group_tokens),
                   "--context_condition_frame_num", "32", "--compress_with_relevance", str(args.feedback == "on"),
                   "--htm_stage_a", "False", "--htm_quantile", str(args.quantile),
                   "--torch_dtype", "bfloat16", "--set_verbose", "False"]
        (args.outdir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        print(f"Selected {len(selected)} questions: {[row['key'] for row in selected]}", flush=True)
        if args.prepare_only:
            print(f"PREPARED ONLY: {args.outdir}; model loading and inference were not tested")
            return 0
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["HTM_EVAL_LOG_TAU"] = "1"
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        log_path = args.outdir / "inference.log"
        with log_path.open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            exit_code = process.wait()
        log_text = log_path.read_text()
        predictions_path = args.outdir / "predictions.json"
        predictions = read_rows(predictions_path) if predictions_path.is_file() else []
        expected = {r["key"]: r for r in selected}
        actual = [r.get("key") for r in predictions]
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("lvbench_score", ROOT / "scripts/eval/score_lvbench_preds.py")
        scorer = module_from_spec(spec)
        spec.loader.exec_module(scorer)
        checks = {
            "process_success": exit_code == 0,
            "predictor_loaded": bool(re.search(r"loaded 6 htm_predictor tensors", log_text)),
            "htm_runtime_active": "[HTM] eval runtime: stage_a=False" in log_text,
            "requested_event_representation_active": f"event_representation={args.event_representation}" in log_text,
            "requested_memory_layout_active": f"memory_layout={args.memory_layout}" in log_text,
            "requested_decoder_grouping_active": f"grouped={args.group_tokens}" in log_text,
            "all_selected_questions_predicted_once": len(actual) == len(expected) and set(actual) == set(expected),
            "nonempty_valid_predictions": bool(predictions) and all(str(r.get("pred", "")).strip() and scorer.normalize_letter(r.get("pred")) in "ABCD" for r in predictions),
            "ground_truth_matches_subset": all(r.get("gt") == expected.get(r.get("key"), {}).get("answer") for r in predictions),
            "telemetry_for_each_question": all(f"[HTM τ] key={key} " in log_text for key in expected),
        }
        report = {"checkpoint": str(args.checkpoint.resolve()), "feedback": args.feedback,
                  "event_representation": args.event_representation, "relevance_memory_entries": 256,
                  "memory_layout": args.memory_layout,
                  "group_tokens": args.group_tokens,
                  "frames": args.frames, "memory_tokens": args.memory_tokens, "exit_code": exit_code,
                  "checks": checks, "passed": all(checks.values()),
                  "score": scorer.score(predictions), "note": "Loading/inference smoke only; tiny uniformly sampled subset cannot measure accuracy recovery."}
        (args.outdir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        print("PASS: checkpoint reload and HTM inference" if report["passed"] else "CHECK FAILED: inspect report.json and inference.log")
        return 0 if report["passed"] else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
