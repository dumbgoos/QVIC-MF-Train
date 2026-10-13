"""Randomly select two videos and compare solo/forward/reverse HTM inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import subprocess
import sys

from check_htm_checkpoint_inference import ROOT, read_rows, select_rows


TAU_RE = re.compile(
    r"\[HTM τ\] key=(\S+) merge_rate=([\d.eE+-]+) n_merges=(\d+)/(\d+) "
    r"\|M\|=(\d+) L=(\d+) tau_s=([\d.eE+-]+) tau_d=([\d.eE+-]+) ready=(True|False)"
)


def sample_pair(rows: list[dict], video_dir: Path, seed: int) -> list[dict]:
    # Uniformly sample an available QA first, then a QA from another video.
    select_rows(rows, video_dir, 1)
    # select_rows validates schema; request all unique available keys to avoid
    # duplicate annotation rows influencing random selection.
    count = len({r["key"] for r in rows if (video_dir / f"{r['key'][:11]}.mp4").is_file()})
    available = select_rows(rows, video_dir, count)
    rng = random.Random(seed)
    first = rng.choice(available)
    alternatives = [r for r in available if r["key"][:11] != first["key"][:11]]
    if not alternatives:
        raise ValueError("Need at least two different videos with available media")
    return [first, rng.choice(alternatives)]


def parse_observations(directory: Path) -> dict:
    telemetry = {}
    for match in TAU_RE.finditer((directory / "inference.log").read_text()):
        key, merge, merges, steps, memory, budget, tau_s, tau_d, ready = match.groups()
        telemetry[key] = {"merge_rate": float(merge), "n_merges": int(merges), "n_steps": int(steps),
                          "memory_tokens": int(memory), "budget": int(budget),
                          "tau_s": float(tau_s), "tau_d": float(tau_d), "ready": ready == "True"}
    result = {}
    for row in read_rows(directory / "predictions.json"):
        key = row["key"]
        if key not in telemetry:
            raise ValueError(f"Missing telemetry for {key} in {directory}")
        result[key] = {"prediction": row["pred"], "gt": row["gt"], **telemetry[key]}
    return result


def compare_orders(pair: list[dict], runs: dict) -> dict:
    comparisons = {}
    for index, row in enumerate(pair):
        key = row["key"]
        solo = runs[f"solo_{index + 1}"][key]
        observations = {"solo": solo, "forward": runs["forward"][key], "reverse": runs["reverse"][key]}
        differences = {}
        for mode in ("forward", "reverse"):
            other = observations[mode]
            differences[mode] = {
                "prediction_changed": other["prediction"] != solo["prediction"],
                "merge_count_delta": other["n_merges"] - solo["n_merges"],
                "memory_tokens_delta": other["memory_tokens"] - solo["memory_tokens"],
                "tau_s_delta": other["tau_s"] - solo["tau_s"],
                "tau_d_delta": other["tau_d"] - solo["tau_d"],
            }
        comparisons[key] = {"observations": observations, "vs_solo": differences,
                            "logged_behavior_changed": any(observations[m] != solo for m in ("forward", "reverse"))}
    return comparisons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--feedback", choices=("on", "off"), default="on")
    parser.add_argument("--memory-tokens", type=int, default=256)
    parser.add_argument("--model-base", default="lmms-lab/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.frames <= 0 or args.memory_tokens <= 0:
            raise ValueError("Frames and memory tokens must be positive")
        pair = sample_pair(read_rows(args.questions), args.video_dir, args.seed)
        out = args.outdir.resolve()
        if out.exists() and any(out.iterdir()):
            raise ValueError("Choose a new empty --outdir; prior experiment files will not be overwritten")
        out.mkdir(parents=True, exist_ok=True)
        plan = {"solo_1": pair[:1], "solo_2": pair[1:], "forward": pair, "reverse": pair[::-1]}
        manifest = {"seed": args.seed, "source": str(args.questions), "checkpoint": str(args.checkpoint),
                    "frames": args.frames, "feedback": args.feedback, "memory_tokens": args.memory_tokens,
                    "selected": pair, "orders": {name: [r["key"] for r in rows] for name, rows in plan.items()}}
        (out / "selection.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(manifest["orders"], indent=2), flush=True)
        runs = {}
        for name, rows in plan.items():
            subset = out / f"{name}.jsonl"
            subset.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            command = [sys.executable, "-u", str(ROOT / "scripts/tools/check_htm_checkpoint_inference.py"),
                       "--checkpoint", str(args.checkpoint.resolve()), "--questions", str(subset),
                       "--video-dir", str(args.video_dir.resolve()), "--outdir", str(out / name),
                       "--num-questions", str(len(rows)), "--frames", str(args.frames),
                       "--feedback", args.feedback, "--memory-tokens", str(args.memory_tokens),
                       "--model-base", args.model_base]
            if args.prepare_only:
                command.append("--prepare-only")
            print(f"RUN {name}: fresh model process", flush=True)
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode:
                (out / "order_report.json").write_text(json.dumps({"complete": False, "failed_run": name, "exit_code": result.returncode}, indent=2) + "\n")
                return result.returncode
            if not args.prepare_only:
                run_report = json.loads((out / name / "report.json").read_text())
                if not run_report["passed"]:
                    raise ValueError(f"Inference check failed in {name}")
                runs[name] = parse_observations(out / name)
        if args.prepare_only:
            print(f"PREPARED ONLY: {out}; GPU inference was not run")
            return 0
        comparisons = compare_orders(pair, runs)
        report = {"complete": True, "seed": args.seed, "comparisons": comparisons,
                  "order_sensitive_logged_behavior": any(r["logged_behavior_changed"] for r in comparisons.values()),
                  "note": "Differences are diagnostic, not failures. Current eval resets threshold history per question. Tau logs are rounded to 4 decimals. This is not an accuracy evaluation."}
        (out / "order_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"COMPLETE: order diagnostic saved to {out / 'order_report.json'}")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
