"""Show information collisions caused by token pooling and event averaging."""
import json

import torch
from htm_diagnostic_utils import load_htm


def main():
    htm = load_htm()
    torch.manual_seed(0)
    frame_a = torch.tensor([[1., 0.], [0., 1.]])
    frame_b = torch.tensor([[2., -1.], [-1., 2.]])
    assert not torch.equal(frame_a, frame_b)
    assert torch.equal(frame_a.mean(0), frame_b.mean(0))
    vectors = []
    with torch.no_grad():
        for order in ((frame_a, frame_b), (frame_b, frame_a)):
            ctrl = htm.HTMMemoryController(64)
            predictor = htm.HTMPredictor(2, 2)
            ctrl.thresholds.tau_s = ctrl.thresholds.tau_d = float("inf")
            for frame in order:
                ctrl.write_batch(frame[None, None], predictor, False)
            vectors.append(ctrl.states[0].open_event.z)
            assert ctrl.states[0].open_event.n == 2
    assert torch.equal(*vectors)
    print(json.dumps({"distinct_frames_same_pooled_token": True,
                      "reversed_event_same_vector": True,
                      "event_vector": vectors[0].tolist()}, indent=2))
    print("This is a constructed information-loss example, not a benchmark accuracy measurement.")


if __name__ == "__main__":
    main()
