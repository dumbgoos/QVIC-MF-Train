"""CPU backward regression for Stage A/B; --full-model runs the VLM self-test."""
import argparse
import subprocess
import sys
from pathlib import Path

import torch
from htm_diagnostic_utils import load_htm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-model", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(7)
    htm = load_htm()
    baseline_memories = {}
    for retain in (False, True):
        for force_merge in (False, True):
            torch.manual_seed(7)
            compressor = torch.nn.Linear(8, 8)
            seed = torch.nn.Parameter(torch.randn(4, 8))
            predictor = htm.HTMPredictor(8, 8)
            decoder = torch.nn.Linear(8, 3).requires_grad_(False)
            ctrl = htm.HTMMemoryController(64, recent_window_W=1)
            ctrl.thresholds.tau_s = ctrl.thresholds.tau_d = float("inf") if force_merge else -1.0
            losses = []
            for _ in range(6):
                features = compressor(torch.randn(1, 1, 4, 8)) + seed
                pred_loss = ctrl.write_batch(features, predictor, False, retain_memory_grad=retain)
                if pred_loss is not None:
                    losses.append(pred_loss)
            memory = ctrl.materialize_context_memory(features)
            if not retain:
                baseline_memories[force_merge] = memory.detach().clone()
            else:
                assert torch.equal(memory.detach(), baseline_memories[force_merge]), "gradient fix changed forward values"
            # Historical event gradients must survive even without recent tokens.
            if retain:
                event_loss = torch.stack([ev.z.square().mean() for ev in ctrl.states[0].events + [ctrl.states[0].open_event]]).sum()
                event_grad = torch.autograd.grad(event_loss, compressor.weight, retain_graph=True)[0]
                assert event_grad.norm() > 0
                qa_loss = torch.nn.functional.cross_entropy(decoder(memory.mean((1, 2))), torch.tensor([1]))
                qa_grads = torch.autograd.grad(qa_loss, (compressor.weight, seed), retain_graph=True)
                assert all(torch.isfinite(g).all() and g.norm() > 0 for g in qa_grads)
                (qa_loss + 0.1 * torch.stack(losses).mean()).backward()
                assert seed.grad.norm() > 0 and compressor.weight.grad.norm() > 0
            else:
                assert not memory.requires_grad
                torch.stack(losses).mean().backward()
                assert seed.grad is None and compressor.weight.grad is None
            assert any(p.grad is not None and p.grad.norm() > 0 for p in predictor.parameters())
            assert all(p.grad is None for p in decoder.parameters())
            print(f"PASS stage={'B' if retain else 'A'} merge={force_merge}")
    if args.full_model:
        root = Path(__file__).resolve().parents[2]
        subprocess.run([sys.executable, "scripts/train/selftest_train_cpu.py", "--htm", "--memory-length", "64"], cwd=root, check=True)


if __name__ == "__main__":
    main()
