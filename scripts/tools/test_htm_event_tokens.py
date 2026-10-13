"""Check token-preserving events, gradient flow, and actual token accounting."""
import torch
from htm_diagnostic_utils import load_htm


def main():
    htm = load_htm()
    torch.manual_seed(0)
    predictor = htm.HTMPredictor(8, 8)
    frames = torch.randn(6, 1, 1, 4, 8, requires_grad=True)
    results = {}
    for mode in ("pooled", "tokens"):
        ctrl = htm.HTMMemoryController(256, recent_window_W=0, event_representation=mode)
        ctrl.thresholds.tau_s = ctrl.thresholds.tau_d = float("inf")
        for feature in frames:
            ctrl.write_batch(feature, predictor, False, retain_memory_grad=True)
        memory = ctrl.materialize_context_memory(frames[-1])
        assert ctrl.states[0].open_event.n == 6
        expected = frames[:, 0, 0].mean(0)
        if mode == "pooled":
            expected = expected.mean(0, keepdim=True)
        assert torch.allclose(memory[0, :, 0], expected, atol=1e-6)
        assert ctrl.memory_composition(ctrl.states[0])["event_tokens"] == memory.shape[1]
        grad = torch.autograd.grad(memory.square().sum(), frames, retain_graph=True)[0]
        assert torch.isfinite(grad).all() and grad.norm() > 0
        results[mode] = ctrl.snapshot_eval_write_stats()
    assert results["pooled"]["n_merges"] == results["tokens"]["n_merges"] == 5
    assert results["pooled"]["final_M"] == 1 and results["tokens"]["final_M"] == 4
    tight = htm.HTMMemoryController(8, event_representation="tokens")
    tight.thresholds.tau_s = tight.thresholds.tau_d = -1.0
    with torch.no_grad():
        for feature in frames:
            tight.write_batch(feature, predictor, False)
        assert tight.materialize_context_memory(frames[-1]).shape[1] <= 8
    print("PASS pooled/token event values, gradients, gating parity, and budget enforcement")


if __name__ == "__main__":
    main()
