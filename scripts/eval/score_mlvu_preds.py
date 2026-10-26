#!/usr/bin/env python3
"""Score MLVU MC preds → per-task acc + M-Avg (macro over 7 tasks)."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

LETTER_RE = re.compile(r"[A-D]", re.IGNORECASE)
MC_TASKS = [
    "plotQA", "needle", "ego", "count", "order", "anomaly_reco", "topic_reasoning",
]


def normalize_letter(text) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""
    m = re.search(r"\b([A-D])\b", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = LETTER_RE.search(s)
    return m.group(0).upper() if m else s[0].upper()


def load_preds(path: Path) -> list[dict]:
    raw = path.read_text().strip()
    if not raw:
        return []
    if raw[0] == "[":
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    rows = load_preds(Path(args.preds))

    type_tot: dict[str, int] = defaultdict(int)
    type_hit: dict[str, int] = defaultdict(int)
    micro_n = micro_c = 0
    for row in rows:
        gt = normalize_letter(row.get("gt") or row.get("answer"))
        pred = normalize_letter(row.get("pred") or row.get("prediction"))
        if not gt:
            continue
        qtypes = row.get("task_type") or row.get("question_type") or "unknown"
        if isinstance(qtypes, list):
            qtypes = qtypes[0] if qtypes else "unknown"
        hit = int(pred == gt)
        micro_n += 1
        micro_c += hit
        type_tot[qtypes] += 1
        type_hit[qtypes] += hit

    per = {}
    for t in MC_TASKS + sorted(set(type_tot) - set(MC_TASKS)):
        if t not in type_tot:
            continue
        n, c = type_tot[t], type_hit[t]
        per[t] = {"n": n, "correct": c, "acc": c / n if n else 0.0, "pct": 100.0 * c / n if n else 0.0}

    m_tasks = [t for t in MC_TASKS if t in per]
    m_avg = sum(per[t]["acc"] for t in m_tasks) / len(m_tasks) if m_tasks else 0.0
    out = {
        "n": micro_n,
        "correct": micro_c,
        "micro_acc": (micro_c / micro_n) if micro_n else 0.0,
        "micro_pct": (100.0 * micro_c / micro_n) if micro_n else 0.0,
        "M_Avg": m_avg,
        "M_Avg_pct": 100.0 * m_avg,
        "M_Avg_tasks": m_tasks,
        "per_type": per,
        "missing_mc_tasks": [t for t in MC_TASKS if t not in per],
    }
    print(json.dumps(out, indent=2))
    print(f"\nM-Avg (macro/{len(m_tasks)}): {out['M_Avg_pct']:.1f}%")
    print(f"Micro: {micro_c}/{micro_n} = {out['micro_pct']:.1f}%")
    for t in m_tasks:
        st = per[t]
        print(f"  {t}: {st['correct']}/{st['n']} = {st['pct']:.1f}%")

    if args.outdir:
        od = Path(args.outdir)
        od.mkdir(parents=True, exist_ok=True)
        (od / "score.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
