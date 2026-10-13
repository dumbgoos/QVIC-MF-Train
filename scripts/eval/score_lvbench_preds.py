#!/usr/bin/env python3
"""Score model_video_lvbench JSON/JSONL preds → overall + per-type accuracy.

Matches the reporting style Leo used for QViC-MF LVBench (~50.x% overall):
letter match after light normalize (strip, take first A–D if present).

Also can emit an official-LVBench-style answers dict keyed by uid for
zai-org/LVBench ``compute_accuracy`` if needed.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


LETTER_RE = re.compile(r"[ABCD]", re.IGNORECASE)


def normalize_letter(text: str | None) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""
    # Prefer first standalone A–D; else first letter char.
    m = re.search(r"\b([ABCD])\b", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = LETTER_RE.search(s)
    return m.group(0).upper() if m else s[0].upper()


def load_preds(path: Path) -> list[dict]:
    raw = path.read_text().strip()
    if not raw:
        return []
    # Single JSON list, or JSONL (possibly named .json by the eval script).
    if raw[0] == "[":
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"Expected list in {path}")
        return data
    rows = []
    for i, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"Bad JSON on line {i} of {path}: {e}") from e
    return rows


def score(rows: list[dict]) -> dict:
    total = 0
    correct = 0
    type_total: dict[str, int] = defaultdict(int)
    type_correct: dict[str, int] = defaultdict(int)

    for row in rows:
        gt = normalize_letter(row.get("gt") or row.get("answer"))
        pred = normalize_letter(row.get("pred") or row.get("prediction"))
        if not gt:
            continue
        total += 1
        hit = int(pred == gt)
        correct += hit
        qtypes = row.get("task_type") or row.get("question_type") or []
        if isinstance(qtypes, str):
            qtypes = [qtypes]
        for t in qtypes:
            type_total[t] += 1
            type_correct[t] += hit

    out = {
        "n": total,
        "correct": correct,
        "overall_acc": (correct / total) if total else 0.0,
        "overall_pct": (100.0 * correct / total) if total else 0.0,
        "per_type": {},
    }
    for t in sorted(type_total):
        n = type_total[t]
        c = type_correct[t]
        out["per_type"][t] = {
            "n": n,
            "correct": c,
            "acc": (c / n) if n else 0.0,
            "pct": (100.0 * c / n) if n else 0.0,
        }
    return out


def to_official_answers(rows: list[dict]) -> dict[str, str]:
    """Map uid → predicted letter for zai-org/LVBench compute_accuracy."""
    answers: dict[str, str] = {}
    for row in rows:
        key = str(row.get("key", ""))
        # Formatted keys are "{youtube_id}_{uid}"; youtube ids are 11 chars.
        uid = key[12:] if len(key) > 12 and key[11] == "_" else key.rsplit("_", 1)[-1]
        answers[str(uid)] = normalize_letter(row.get("pred") or row.get("prediction"))
    return answers


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--preds",
        required=True,
        help="Prediction JSON/JSONL from model_video_lvbench (or merged chunks).",
    )
    p.add_argument(
        "--outdir",
        default=None,
        help="Optional dir to write score.json (+ optional official answers).",
    )
    p.add_argument(
        "--write-official-answers",
        action="store_true",
        help="Also write uid→letter JSON for LVBench compute_accuracy.",
    )
    args = p.parse_args()

    rows = load_preds(Path(args.preds))
    result = score(rows)
    print(json.dumps(result, indent=2))
    print(
        f"\nOverall: {result['correct']}/{result['n']} = "
        f"{result['overall_pct']:.1f}%"
    )
    for t, st in result["per_type"].items():
        print(f"  {t}: {st['correct']}/{st['n']} = {st['pct']:.1f}%")

    if args.outdir:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "score.json").write_text(json.dumps(result, indent=2) + "\n")
        if args.write_official_answers:
            ans = to_official_answers(rows)
            (outdir / "official_answers.json").write_text(
                json.dumps(ans, indent=2) + "\n"
            )
            print(f"Wrote {outdir / 'official_answers.json'} ({len(ans)} uids)")


if __name__ == "__main__":
    main()
