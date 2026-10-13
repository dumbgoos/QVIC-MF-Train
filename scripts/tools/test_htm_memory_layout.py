"""Check frame baseline values and chronological sorting without selection changes."""
import torch
from htm_diagnostic_utils import load_htm


def main():
    htm = load_htm()
    torch.manual_seed(0)
    predictor = htm.HTMPredictor(8, 8)
    frames = torch.arange(8.).view(8, 1, 1, 1, 1).expand(8, 1, 1, 4, 8).clone().requires_grad_()

    def run(layout, budget):
        ctrl = htm.HTMMemoryController(budget, event_representation="tokens", memory_layout=layout)
        ctrl.thresholds.tau_s = ctrl.thresholds.tau_d = -1.0
        for feature in frames:
            ctrl.write_batch(feature, predictor, False, retain_memory_grad=True)
        return ctrl, ctrl.materialize_context_memory(frames[-1])[0].flatten(0, 1)

    raw, memory = run("frames", 4096)
    assert torch.equal(memory, frames[:, 0, 0].reshape(-1, 8))
    assert raw.snapshot_eval_write_stats()["n_merges"] == 0
    assert memory.shape[0] == 8 * 4  # Exactly once per frame/token, no recent duplicate.
    assert raw.materialize_context_memory(frames[-1]).shape == (1, 8, 4, 8)
    assert torch.autograd.grad(memory.sum(), frames)[0].abs().sum() > 0
    for budget in (4096, 16):
        legacy, a = run("legacy", budget)
        chronological, b = run("chronological", budget)
        assert a.shape == b.shape and b.shape[0] <= budget
        assert sorted(map(tuple, a.detach().tolist())) == sorted(map(tuple, b.detach().tolist()))
        assert b[:, 0].tolist() == sorted(b[:, 0].tolist())
        assert legacy.snapshot_eval_write_stats() == chronological.snapshot_eval_write_stats()
    print("PASS frame baseline is exact and chronological sort preserves selected tokens and gating")


if __name__ == "__main__":
    main()
