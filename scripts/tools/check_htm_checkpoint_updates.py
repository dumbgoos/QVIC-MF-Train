"""Compare HTM checkpoint parameter updates on CPU, without loading a VLM."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def load_group(path: Path, filename: str) -> dict[str, torch.Tensor]:
    state = torch.load(path / filename, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError(f"{path / filename}: expected a tensor state dict")
    return state


def group_for(key: str) -> str:
    if "lora_" in key and "encoder." in key:
        return "compressor_lora"
    if "context_embed." in key:
        return "context_seed"
    if "htm_predictor." in key:
        return "predictor"
    return "other"


def compare(before: Path, after: Path, tolerance: float) -> dict:
    groups = {name: [] for name in ("compressor_lora", "context_seed", "predictor", "other")}
    for filename in ("adapter_model.bin", "non_lora_trainables.bin"):
        old, new = load_group(before, filename), load_group(after, filename)
        if old.keys() != new.keys():
            raise ValueError(f"{filename}: key mismatch; missing after={sorted(old.keys() - new.keys())}; added after={sorted(new.keys() - old.keys())}")
        for key in sorted(old):
            a, b = old[key], new[key]
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError(f"{key}: shape/dtype mismatch: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
            if not a.is_floating_point():
                raise ValueError(f"{key}: expected floating-point parameters")
            finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            row = {"key": key, "numel": a.numel(), "dtype": str(a.dtype), "finite": finite}
            if finite:
                # Subtract in double precision rather than the checkpoint dtype.
                a64, b64 = a.double(), b.double()
                delta = b64 - a64
                row.update(changed_elements=int((delta.abs() > tolerance).sum()),
                           delta_l2=float(delta.norm()), before_l2=float(a64.norm()),
                           max_abs_delta=float(delta.abs().max()) if delta.numel() else 0.0)
            groups[group_for(key)].append(row)
    report = {"before": str(before), "after": str(after), "atol": tolerance, "groups": {}}
    for name, rows in groups.items():
        finite_rows = [r for r in rows if r["finite"]]
        changed = [r for r in finite_rows if r["changed_elements"] > 0]
        delta_l2 = math.sqrt(sum(r["delta_l2"] ** 2 for r in finite_rows))
        before_l2 = math.sqrt(sum(r["before_l2"] ** 2 for r in finite_rows))
        nonfinite = [r["key"] for r in rows if not r["finite"]]
        report["groups"][name] = {
            "status": "NONFINITE" if nonfinite else "MISSING" if not rows else "UPDATED" if changed else "UNCHANGED",
            "tensors": len(rows), "changed_tensors": len(changed),
            "elements": sum(r["numel"] for r in rows),
            "changed_elements": sum(r["changed_elements"] for r in finite_rows),
            "delta_l2": delta_l2,
            "relative_delta_l2": delta_l2 / before_l2 if before_l2 else None,
            "max_abs_delta": max((r["max_abs_delta"] for r in finite_rows), default=0.0),
            "nonfinite_keys": nonfinite,
            "unchanged_keys": [r["key"] for r in finite_rows if not r["changed_elements"]],
            "largest_changes": sorted(changed, key=lambda r: r["delta_l2"], reverse=True)[:5],
            "per_tensor": rows,
        }
    required = ("compressor_lora", "context_seed", "predictor")
    report["all_required_groups_updated"] = all(report["groups"][g]["status"] == "UPDATED" for g in required)
    report["all_parameters_finite"] = all(r["finite"] for rows in groups.values() for r in rows)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="Write detailed per-tensor JSON report")
    parser.add_argument("--atol", type=float, default=0.0, help="Absolute change threshold; default checks any stored change")
    args = parser.parse_args()
    if not math.isfinite(args.atol) or args.atol < 0:
        parser.error("--atol must be finite and nonnegative")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    try:
        report = compare(args.before, args.after, args.atol)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"Before: {args.before}\nAfter:  {args.after}\nChange threshold: {args.atol}")
    for name, stats in report["groups"].items():
        if name == "other" and not stats["tensors"]:
            continue
        print(f"{name}: {stats['status']} changed={stats['changed_tensors']}/{stats['tensors']} tensors, "
              f"elements={stats['changed_elements']}/{stats['elements']}, "
              f"delta_L2={stats['delta_l2']:.6e}, max_abs_delta={stats['max_abs_delta']:.6e}")
        for key in stats["nonfinite_keys"]:
            print(f"  NONFINITE: {key}")
        for key in stats["unchanged_keys"][:5]:
            print(f"  UNCHANGED: {key}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Detailed report: {args.out}")
    passed = report["all_required_groups_updated"] and report["all_parameters_finite"]
    print("PASS: all three groups have stored parameter changes" if passed else "CHECK FAILED: a required group is missing, unchanged, or nonfinite")
    print("Stored changes alone do not prove QA-gradient attribution or accuracy recovery; zero stored changes may reflect low-precision rounding.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
