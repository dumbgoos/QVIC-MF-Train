"""Audit chronological order, recent protection, and cross-video thresholds."""
import json

import torch
from htm_diagnostic_utils import load_htm


def main():
    htm = load_htm()
    ctrl = htm.HTMMemoryController(16, recent_window_W=2)
    ctrl.reset(1)
    state = ctrl.states[0]
    state.age_counter = 4
    state.recent.extend([torch.full((2, 2), 20.), torch.full((2, 2), 30.)])
    state.recent_t.extend([2, 3])
    state.events = [htm.HTMEvent(torch.full((2,), float(i)), i, i, 1., 1) for i in range(3)]
    state.open_event = htm.HTMEvent(torch.full((2,), 3.), 3, 3, 1., 1)
    markers = ctrl.build_memory_tokens(state)[:, 0].tolist()
    order = [int(x / 10) if x >= 20 else int(x) for x in markers]
    # With a tight budget, high-surprise events beat even the newest recent.
    ctrl.L = 2
    for event in state.events + [state.open_event]:
        event.S = 100.
    tight = ctrl.build_memory_tokens(state)[:, 0].tolist()
    ctrl.thresholds.update(s=2., d=3.)
    before = len(ctrl.thresholds._s)
    ctrl.reset(1)
    report = {
        "memory_time_order": order,
        "chronological": order == sorted(order),
        "duplicate_frame_indices": sorted(set(state.recent_t) & {ev.t_s for ev in state.events + [state.open_event]}),
        "tight_budget_memory_markers": tight,
        "newest_recent_preserved_in_example": 30. in tight,
        "threshold_samples_before_reset": before,
        "threshold_samples_after_reset": len(ctrl.thresholds._s),
        "default_relevance_weight_beta": ctrl.beta,
    }
    print(json.dumps(report, indent=2))
    print("Diagnostics describe current policy; they do not assert it improves QA accuracy.")


if __name__ == "__main__":
    main()
