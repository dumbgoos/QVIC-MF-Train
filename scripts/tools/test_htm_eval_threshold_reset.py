"""Check evaluation resets threshold history while training keeps its behavior."""
from __future__ import annotations

import ast
from pathlib import Path

import torch

from htm_diagnostic_utils import load_htm


def main() -> None:
    module = load_htm()
    ctrl = module.HTMMemoryController(256, quantile=0.8)
    ctrl.thresholds.warmup = 2
    for _ in range(3):
        ctrl.thresholds.update(s=3.0, d=4.0)
    assert ctrl.thresholds.ready()
    ctrl.reset(1)  # Stage A/B training default: keep rolling history.
    assert len(ctrl.thresholds._s) == 3 and ctrl.thresholds.ready()
    ctrl.reset(1, reset_thresholds=True)  # New independent eval question.
    assert not ctrl.thresholds.ready()
    assert not ctrl.thresholds._s and not ctrl.thresholds._d
    assert ctrl.thresholds.tau_s == ctrl.thresholds.tau_d == 1.0
    assert ctrl.thresholds.quantile == 0.8 and ctrl.thresholds.warmup == 2
    assert len(ctrl.states) == 1 and ctrl._eval_n_steps == 0

    source = Path(__file__).resolve().parents[2] / "qvic/model/qvic_meta_mixin.py"
    tree = ast.parse(source.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "reset"
             and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "htm_controller"]
    assert len(calls) == 1
    reset_flag = next(kw.value for kw in calls[0].keywords if kw.arg == "reset_thresholds")
    assert ast.unparse(reset_flag) == "not self.training"

    torch.manual_seed(42)
    predictor = module.HTMPredictor(8, 8).eval()
    streams = {"A": torch.randn(64, 1, 1, 4, 8), "B": torch.randn(64, 1, 1, 4, 8) * 3}

    def run(order):
        controller = module.HTMMemoryController(256)
        results = {}
        with torch.no_grad():
            for key in order:
                controller.reset(1, reset_thresholds=True)
                for frames in streams[key]:
                    controller.write_batch(frames, predictor, collect_threshold_stats=True)
                results[key] = (controller.materialize_context_memory(frames).clone(),
                                controller.snapshot_eval_write_stats())
        return results

    solo = {key: run([key])[key] for key in streams}
    for order in (("A", "B"), ("B", "A")):
        observed = run(order)
        for key in order:
            assert torch.equal(observed[key][0], solo[key][0]), f"memory changed with order for {key}"
            assert observed[key][1] == solo[key][1], f"telemetry changed with order for {key}"
    print("PASS eval clears threshold history at each video; training retains rolling history")
    print("PASS synthetic solo/forward/reverse memory values and telemetry are identical")


if __name__ == "__main__":
    main()
