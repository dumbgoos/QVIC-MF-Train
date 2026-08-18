#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from typing import Dict, List

# Importable without installing the package: the repository root is two levels up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def decode_in_child(loader, key: str, num_frames: int, timeout: int) -> Dict:
    """Decode `key` in a fork. Returns a status dict; never raises."""
    pid = os.fork()
    if pid == 0:
        try:
            signal.alarm(timeout)
            loader(key, num_frames)
            os._exit(0)
        except BaseException:
            os._exit(2)
    _, status = os.waitpid(pid, 0)
    if os.WIFSIGNALED(status):
        sig = os.WTERMSIG(status)
        return {"ok": False, "how": f"signal:{signal.Signals(sig).name}"}
    code = os.WEXITSTATUS(status)
    if code == 0:
        return {"ok": True, "how": "ok"}
    return {"ok": False, "how": "exception" if code == 2 else f"exit:{code}"}


def run_shard(shard_id: int, records: List[Dict], args, out_dir: str, loader) -> None:
    """`loader` is built once in the parent and inherited through the fork --
    rebuilding the shard index per worker would cost 30 s and ~300 MB each."""
    path = os.path.join(out_dir, f"shard-{shard_id:03d}.jsonl")

    done = set()
    if os.path.exists(path):                      # resume a killed scan
        with open(path) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["video"])
                except Exception:
                    pass

    t0, n_bad = time.time(), 0
    with open(path, "a", buffering=1) as fh:      # line buffered: survives a kill
        for i, rec in enumerate(records):
            key = rec["video"]
            if key in done:
                continue
            res = decode_in_child(loader, key, args.num_frames, args.timeout)
            if not res["ok"]:
                n_bad += 1
            fh.write(json.dumps({
                "video": key, "id": rec.get("id", ""),
                "source": rec.get("data_source", ""), **res,
            }) + "\n")
            if shard_id == 0 and (i + 1) % 100 == 0:
                rate = (time.time() - t0) / (i + 1)
                print(f"[shard 0] {i+1}/{len(records)} {rate:.2f}s/rec "
                      f"{n_bad} bad; eta shard {(len(records)-i-1)*rate/60:.0f} min",
                      flush=True)
    print(f"[shard {shard_id}] done: {len(records)} records, {n_bad} bad, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset directory (see the README)")
    ap.add_argument("--backend", default="shards", choices=("shards", "loose"),
                    help="how media is stored; must match the training run")
    ap.add_argument("--video-folder", default=None, help="media root for --backend loose")
    ap.add_argument("--annotations", default=None,
                    help="annotation JSON (default: <root>/annotations/all_sampled.json)")
    ap.add_argument("--out", default="work_dirs/media_scan")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--num-frames", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--limit", type=int, default=-1, help="debug: scan only the first N records")
    ap.add_argument("--merge-into", default=None,
                    help="also fold the bad keys into this training skip list")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    from qvic.train.data import MediaLoader
    loader = MediaLoader(backend=args.backend, dataset_root=args.root,
                         video_folder=args.video_folder, quiet_decoder=True)

    annotations = args.annotations or os.path.join(args.root, "annotations", "all_sampled.json")
    with open(annotations) as fh:
        records = json.load(fh)
    if args.limit > 0:
        records = records[: args.limit]

    by_key: Dict[str, Dict] = {}
    for r in records:
        by_key.setdefault(r["video"], r)
    unique = list(by_key.values())
    print(f"{len(records)} records -> {len(unique)} distinct media, "
          f"{args.workers} workers", flush=True)

    shards = [unique[i::args.workers] for i in range(args.workers)]
    t0 = time.time()
    pids = []
    for sid, shard in enumerate(shards):
        if not shard:
            continue
        pid = os.fork()
        if pid == 0:
            try:
                run_shard(sid, shard, args, args.out, loader)
                os._exit(0)
            except BaseException as exc:  # noqa: BLE001
                print(f"[shard {sid}] FAILED: {type(exc).__name__}: {exc}", flush=True)
                os._exit(1)
        pids.append(pid)
    for pid in pids:
        os.waitpid(pid, 0)

    rows: List[Dict] = []
    for name in sorted(os.listdir(args.out)):
        if not name.startswith("shard-"):
            continue
        with open(os.path.join(args.out, name)) as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    bad = [r for r in rows if not r["ok"]]
    bad_keys = sorted({r["video"] for r in bad})

    by_how: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    for r in bad:
        by_how[r["how"]] = by_how.get(r["how"], 0) + 1
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1

    n_affected = sum(1 for r in records if r["video"] in set(bad_keys))
    report = {
        "scanned_media": len(rows),
        "bad_media": len(bad_keys),
        "bad_media_pct": round(100.0 * len(bad_keys) / max(len(rows), 1), 4),
        "affected_records": n_affected,
        "affected_records_pct": round(100.0 * n_affected / max(len(records), 1), 4),
        "by_failure": by_how,
        "by_source": by_source,
        "minutes": round((time.time() - t0) / 60, 1),
    }
    with open(os.path.join(args.out, "bad_media.json"), "w") as fh:
        json.dump(bad_keys, fh, indent=1)
    with open(os.path.join(args.out, "bad_detail.json"), "w") as fh:
        json.dump(bad, fh, indent=1)
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print("\n=== REPORT ===")
    print(json.dumps(report, indent=1))

    if args.merge_into and bad_keys:
        existing: List[str] = []
        if os.path.exists(args.merge_into):
            with open(args.merge_into) as fh:
                existing = json.load(fh)
        merged = sorted(set(existing) | set(bad_keys))
        tmp = args.merge_into + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(merged, fh, indent=1)
        os.replace(tmp, args.merge_into)
        print(f"merged into {args.merge_into}: {len(existing)} -> {len(merged)} keys")

    print("SCAN OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
