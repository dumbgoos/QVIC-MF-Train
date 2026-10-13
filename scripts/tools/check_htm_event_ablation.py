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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "questions", "video-dir", "outdir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--num-questions", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--feedback", choices=("on", "off"), default="on")
    parser.add_argument("--ablation", choices=("representation", "structure"), default="representation")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    try:
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
        variants = {"pooled_256": ("pooled", 256, "legacy"), "tokens_256": ("tokens", 256, "legacy"), "tokens_4096": ("tokens", 4096, "legacy")}
        if args.ablation == "structure":
            variants = {"tokens_legacy_4096": ("tokens", 4096, "legacy"),
                        "tokens_chronological_4096": ("tokens", 4096, "chronological"),
                        "frames_4096": ("tokens", 4096, "frames")}
        runs, summary = {}, {}
        for name, (representation, budget, layout) in variants.items():
            command = [sys.executable, "-u", str(ROOT / "scripts/tools/check_htm_checkpoint_inference.py"),
                       "--checkpoint", str(args.checkpoint.resolve()), "--questions", str(subset),
                       "--video-dir", str(args.video_dir.resolve()), "--outdir", str(out / name),
                       "--num-questions", str(len(selected)), "--frames", str(args.frames),
                       "--feedback", args.feedback, "--memory-tokens", str(budget),
                       "--event-representation", representation, "--memory-layout", layout]
            if args.prepare_only:
                command.append("--prepare-only")
            print(f"RUN {name}", flush=True)
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode:
                (out / "ablation_report.json").write_text(json.dumps({"complete": False, "failed_run": name, "exit_code": result.returncode}, indent=2) + "\n")
                return result.returncode
            if not args.prepare_only:
                runs[name] = parse_observations(out / name)
                report = json.loads((out / name / "report.json").read_text())
                summary[name] = {"score": report["score"], "event_representation": representation,
                                 "write_budget": budget, "relevance_memory_entries": 256,
                                 "memory_layout": layout,
                                 "mean_memory_tokens": sum(r["memory_tokens"] for r in runs[name].values()) / len(selected)}
        if args.prepare_only:
            print(f"PREPARED ONLY: {out}")
            return 0
        per_question = {key: {name: runs[name][key] for name in variants} for key in runs["pooled_256"]}
        event_variants = [name for name, (_, _, layout) in variants.items() if layout != "frames"]
        report = {"complete": True, "checkpoint": str(args.checkpoint), "frames": args.frames, "seed": args.seed,
                  "ablation": args.ablation,
                  "summary": summary, "per_question": per_question,
                  "merge_counts_equal": all(len({observations[name]["n_merges"] for name in event_variants}) == 1 for observations in per_question.values()),
                  "note": "Relevance capacity fixed at 256. Chronological events sort retained units by start/end time and retain overlap with recent frames. Frame baseline disables merges and recent duplication. Actual decoder lengths differ. Small subset/64-frame sampling is diagnostic; checkpoint training is unchanged."}
        (out / "ablation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(summary, indent=2))
        print(f"COMPLETE: {out / 'ablation_report.json'}")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
