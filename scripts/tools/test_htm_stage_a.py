#!/usr/bin/env python3
# #@HTM
"""CPU smoke test for HTM Stage A write path (no VLM weights required)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

# Load htm.py directly to avoid pulling the full VLM dependency stack.
_HTM_PATH = Path(__file__).resolve().parents[2] / "qvic" / "model" / "htm.py"
_spec = importlib.util.spec_from_file_location("qvic_htm_stage_a", _HTM_PATH)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
HTMMemoryController = _mod.HTMMemoryController
HTMPredictor = _mod.HTMPredictor
htm_distance = _mod.htm_distance


def main() -> None:
    torch.manual_seed(0)
    B, T, C, D, L = 2, 12, 4, 32, 24
    pred = HTMPredictor(dim=D, hidden_size=16)
    ctrl = HTMMemoryController(token_budget_L=L, recent_window_W=3, quantile=0.75)
    ctrl.reset(B)

    opt = torch.optim.Adam(pred.parameters(), lr=1e-2)
    total_loss = None
    for t in range(T):
        frames = torch.randn(B, 1, C, D)
        # mild temporal continuity
        if t > 0:
            frames = 0.8 * prev + 0.2 * frames
        prev = frames
        step = ctrl.write_batch(frames, pred, collect_threshold_stats=True)
        if step is not None:
            total_loss = step if total_loss is None else total_loss + step

    assert total_loss is not None and total_loss.ndim == 0
    opt.zero_grad()
    total_loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in pred.parameters() if p.grad is not None)
    assert grad_norm > 0, "expected predictor gradients"

    mem = ctrl.materialize_context_memory(frames)
    assert mem.ndim == 4 and mem.shape[0] == B and mem.shape[2] == 1
    assert mem.shape[1] <= L, f"token budget violated: {mem.shape[1]} > {L}"
    # Event tokens cost 1; with merges we should use fewer than T*C raw tokens.
    assert mem.shape[1] < T * C

    a = torch.randn(4, D)
    b = torch.randn(4, D)
    dist = htm_distance(a, b)
    assert dist.shape == (4,)

    print(
        "ok",
        {
            "loss": float(total_loss.detach()),
            "grad_norm": grad_norm,
            "mem_shape": tuple(mem.shape),
            "tau_s": ctrl.thresholds.tau_s,
            "tau_d": ctrl.thresholds.tau_d,
            "events": [len(s.events) + (1 if s.open_event else 0) for s in ctrl.states],
        },
    )


if __name__ == "__main__":
    main()
