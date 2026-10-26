#!/usr/bin/env python3
"""Format MLVU/MVLU task JSONs → JSONL for model_video_lvbench (patched).

MC only by default. Writes letter GT + relative video path under video/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# (json_name, question_type, video_subdir)
MC_TASKS = [
    ("1_plotQA.json", "plotQA", "1_plotQA"),
    ("2_needle.json", "needle", "2_needle"),
    ("3_ego.json", "ego", "3_ego"),
    ("4_count.json", "count", "4_count"),
    ("5_order.json", "order", "5_order"),
    ("6_anomaly_reco.json", "anomaly_reco", "6_anomaly_reco"),
    ("7_topic_reasoning.json", "topic_reasoning", "7_topic_reasoning"),
]


def letter_for(answer: str, candidates: list[str]) -> str:
    for i, c in enumerate(candidates):
        if c == answer:
            return chr(ord("A") + i)
    raise ValueError(f"answer not in candidates: {answer!r} vs {candidates!r}")


def format_question(question: str, candidates: list[str]) -> str:
    lines = [f"Question: {question}", "Options:"]
    for i, c in enumerate(candidates):
        lines.append(f"({chr(ord('A') + i)}) {c}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json_dir", required=True, help=".../MLVU/json")
    p.add_argument("--output", required=True, help="output JSONL")
    p.add_argument(
        "--tasks",
        default="mc",
        choices=("mc",),
        help="mc = 7 multiple-choice tasks (M-Avg)",
    )
    args = p.parse_args()
    json_dir = Path(args.json_dir)
    n = 0
    with open(args.output, "w") as out:
        for fname, qtype, vsubdir in MC_TASKS:
            path = json_dir / fname
            data = json.loads(path.read_text())
            for i, row in enumerate(data):
                cands = row["candidates"]
                letter = letter_for(row["answer"], cands)
                video_rel = f"{vsubdir}/{row['video']}"
                entry = {
                    "key": f"{qtype}_{Path(row['video']).stem}_{i}",
                    "video": video_rel,
                    "question": format_question(row["question"], cands),
                    "answer": letter,
                    "question_type": row.get("question_type", qtype),
                    "candidates": cands,
                    "answer_text": row["answer"],
                    "duration": row.get("duration"),
                }
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                n += 1
    print(f"Wrote {n} MC lines → {args.output}")


if __name__ == "__main__":
    main()
