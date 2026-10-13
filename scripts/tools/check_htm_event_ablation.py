"""Compare pooled/token events at fixed budgets on the same random LVBench subset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys

from check_htm_checkpoint_inference import ROOT, read_rows, select_rows
from check_htm_question_order import parse_observations


def variant_settings(ablation: str) -> dict:
    if ablation == "structure":
        return {"tokens_legacy_4096": ("tokens", 4096, "legacy"),
                "tokens_chronological_4096": ("tokens", 4096, "chronological"),
                "frames_4096": ("tokens", 4096, "frames")}
    return {"pooled_256": ("pooled", 256, "legacy"), "tokens_256": ("tokens", 256, "legacy"), "tokens_4096": ("tokens", 4096, "legacy")}


def summarize(out: Path, ablation: str) -> dict:
    selection = json.loads((out / "selection.json").read_text())
    selected = selection["selected"]
    expected = {row["key"]: row["answer"] for row in selected}
    if not expected or len(expected) != len(selected):
        raise ValueError("Selection must contain nonempty, unique question keys")
    variants = variant_settings(ablation)
    runs, summary, settings = {}, {}, None
    for name, (representation, budget, layout) in variants.items():
        report = json.loads((out / name / "report.json").read_text())
        if not report.get("passed") or report.get("exit_code") != 0:
            raise ValueError(f"{name}: inference did not pass; cannot summarize as complete")
        if (report.get("event_representation") != representation or report.get("memory_tokens") != budget
                or report.get("memory_layout", "legacy") != layout
                or report.get("relevance_memory_entries") != 256):
            raise ValueError(f"{name}: settings differ from the requested ablation")
        current = (report["checkpoint"], report["frames"], report["feedback"])
        if settings is not None and current != settings:
            raise ValueError(f"{name}: checkpoint, sampling, or feedback differs across groups")
        settings = current
        predictions = read_rows(out / name / "predictions.json")
        runs[name] = parse_observations(out / name)
        if len(predictions) != len(expected) or set(runs[name]) != set(expected):
            raise ValueError(f"{name}: prediction keys do not match selection")
        if any(row["gt"] != expected[row["key"]] for row in predictions):
            raise ValueError(f"{name}: ground truth differs from selection")
        summary[name] = {"score": report["score"], "event_representation": representation,
                         "write_budget": budget, "relevance_memory_entries": 256,
                         "memory_layout": layout,
                         "mean_memory_tokens": sum(r["memory_tokens"] for r in runs[name].values()) / len(selected)}
    per_question = {key: {name: runs[name][key] for name in variants} for key in expected}
    event_variants = [name for name, (_, _, layout) in variants.items() if layout != "frames"]
    report = {"complete": True, "checkpoint": settings[0], "frames": settings[1], "feedback": settings[2], "seed": selection["seed"],
              "ablation": ablation, "summary": summary, "per_question": per_question,
              "merge_counts_equal": all(len({observations[name]["n_merges"] for name in event_variants}) == 1 for observations in per_question.values()),
              "note": "Relevance capacity fixed at 256. Chronological events retain overlap with recent frames. Frame baseline disables merges/recent duplication and restores decoder grouping. Actual decoder lengths differ. Small subset/64-frame sampling is diagnostic; checkpoint training is unchanged."}
    (out / "ablation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"COMPLETE: {out / 'ablation_report.json'}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    for name in ("checkpoint", "questions", "video-dir"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--num-questions", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--feedback", choices=("on", "off"), default="on")
    parser.add_argument("--ablation", choices=("representation", "structure"), default="representation")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true", help="Read existing child results and write summary without loading models")
    args = parser.parse_args()
    try:
        if args.summarize_only:
            if args.prepare_only:
                raise ValueError("--summarize-only and --prepare-only are mutually exclusive")
            summarize(args.outdir.resolve(), args.ablation)
            return 0
        if any(value is None for value in (args.checkpoint, args.questions, args.video_dir)):
            raise ValueError("Inference requires --checkpoint, --questions and --video-dir")
        if args.num_questions <= 0 or args.frames <= 0:
            raise ValueError("Question and frame counts must be positive")
        rows = read_rows(args.questions)
        select_rows(rows, args.video_dir, 1)
        available_count = len({r["key"] for r in rows if (args.video_dir / f"{r['key'][:11]}.mp4").is_file()})
        available = select_rows(rows, args.video_dir, available_count)
        if len(available) < args.num_questions:
            raise ValueError("Not enough questions with available videos")
        selected = random.Random(args.seed).sample(available, args.num_questions)
        out = args.outdir.resolve()
        if out.exists() and any(out.iterdir()):
            raise ValueError("Choose a new empty output directory")
        out.mkdir(parents=True, exist_ok=True)
        subset = out / "questions.jsonl"
        subset.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
        (out / "selection.json").write_text(json.dumps({"seed": args.seed, "source": str(args.questions), "selected": selected}, indent=2, ensure_ascii=False) + "\n")
        variants = variant_settings(args.ablation)
        for name, (representation, budget, layout) in variants.items():
            command = [sys.executable, "-u", str(ROOT / "scripts/tools/check_htm_checkpoint_inference.py"),
                       "--checkpoint", str(args.checkpoint.resolve()), "--questions", str(subset),
                       "--video-dir", str(args.video_dir.resolve()), "--outdir", str(out / name),
                       "--num-questions", str(len(selected)), "--frames", str(args.frames),
                       "--feedback", args.feedback, "--memory-tokens", str(budget),
                       "--event-representation", representation, "--memory-layout", layout, "--no-group-tokens"]
            if args.prepare_only:
                command.append("--prepare-only")
            print(f"RUN {name}", flush=True)
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode:
                (out / "ablation_report.json").write_text(json.dumps({"complete": False, "failed_run": name, "exit_code": result.returncode}, indent=2) + "\n")
                return result.returncode
        if args.prepare_only:
            print(f"PREPARED ONLY: {out}")
            return 0
        summarize(out, args.ablation)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
