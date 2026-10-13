"""Compare actual HTM token use with the MF frame-entry budget (CPU)."""
import argparse
import json

import torch
from htm_diagnostic_utils import load_htm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--context-tokens", type=int, default=16)
    parser.add_argument("--mf-entries", type=int, default=256)
    parser.add_argument("--htm-tokens", type=int, default=256)
    args = parser.parse_args()
    assert min(vars(args).values()) > 0
    htm = load_htm()
    torch.manual_seed(0)
    report = []
    with torch.no_grad():
        for budget in sorted({args.htm_tokens, args.mf_entries * args.context_tokens}):
            # Same predictor weights and input frames: vary only the cap.
            torch.manual_seed(0)
            ctrl = htm.HTMMemoryController(budget)
            predictor = htm.HTMPredictor(16, 8)
            for _ in range(args.frames):
                features = torch.randn(1, 1, args.context_tokens, 16)
                ctrl.write_batch(features, predictor, True)
            stats = ctrl.memory_composition(ctrl.states[0])
            assert stats["final_M"] <= budget
            report.append(stats)
    print(json.dumps({"mf_max_tokens": args.mf_entries * args.context_tokens,
                      "mf_tokens_for_input": min(args.frames, args.mf_entries) * args.context_tokens,
                      "htm": report}, indent=2))
    print("A larger cap alone cannot restore detail already pooled into event vectors.")


if __name__ == "__main__":
    main()
