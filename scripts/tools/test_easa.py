"""CPU EASA regressions without video data, VLM imports or model weights."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]


def load_modules():
    # Avoid qvic.__init__ eagerly importing every VLM backend.
    for name, path in (("qvic", ROOT / "qvic"), ("qvic.model", ROOT / "qvic/model")):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    from qvic.model import easa, htm
    return easa, htm


def test_formula(easa):
    P = torch.tensor([[[0., 0.], [1., 0.], [0.9, 0.1], [0., 1.], [0., 0.]]])
    mask = easa.apply_easa_attention_mask(None, P, query_length=5, key_length=5, past_length=0,
                                          dtype=torch.float32, device=P.device, strength=0.7, epsilon=1e-6)
    R = P @ P.transpose(-1, -2)
    expected = 0.7 * (R + 1e-6).log()
    valid = P.sum(-1) > 0
    expected = torch.where(valid[:, :, None] & valid[:, None, :], expected, 0.)
    causal = torch.ones(5, 5, dtype=torch.bool).tril()
    assert torch.allclose(mask[:, 0][:, causal], expected[:, causal])
    q = torch.randn(1, 2, 5, 4, requires_grad=True)
    k, v = torch.randn_like(q), torch.randn_like(q)
    logits = q @ k.transpose(-1, -2) / 2
    probabilities = (logits + mask).softmax(-1)
    assert (probabilities[..., ~causal] == 0).all()
    actual = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(actual, probabilities @ v, atol=1e-6)
    actual.square().sum().backward()
    assert q.grad.isfinite().all() and q.grad.norm() > 0
    assert easa.apply_easa_attention_mask(mask, P, query_length=5, key_length=5, past_length=0,
                                          dtype=mask.dtype, device=P.device, strength=0, epsilon=1e-6) is mask
    cached = easa.apply_easa_attention_mask(None, P, query_length=1, key_length=6, past_length=5,
                                           dtype=mask.dtype, device=P.device, strength=1, epsilon=1e-6)
    assert cached is None
    partial = easa.apply_easa_attention_mask(None, P, query_length=2, key_length=4, past_length=2,
                                            dtype=mask.dtype, device=P.device, strength=0.7, epsilon=1e-6)
    assert torch.equal(partial, mask[:, :, 2:4, :4])
    padded = mask.clone()
    padded[..., 0] = torch.finfo(mask.dtype).min
    result = easa.apply_easa_attention_mask(padded, P, query_length=5, key_length=5, past_length=0,
                                           dtype=mask.dtype, device=P.device, strength=1, epsilon=1e-6)
    assert torch.equal(result[..., 0], padded[..., 0])
    for dtype in (torch.float16, torch.bfloat16):
        mixed = easa.apply_easa_attention_mask(None, P, query_length=5, key_length=5, past_length=0,
                                              dtype=dtype, device=P.device, strength=1, epsilon=1e-6)
        assert mixed[:, 0][:, causal].isfinite().all()
    # Identical content: same-event predecessors receive greater probability.
    equal = mask[0, 0, 3, :4].softmax(-1)
    assert equal[3] > equal[1]


def test_alignment(easa):
    P = torch.tensor([[[1., 0.]] * 4, [[0., 1.]] * 4])
    grid = easa.add_assignment_newlines(P, "grid", "spatial_unpad")
    assert grid.shape == (12, 2)
    assert torch.equal(grid[[0, 1, 3, 4]], P[0])
    assert (grid[[2, 5, 8, 11]] == 0).all()
    assert easa.add_assignment_newlines(P, "frame", "flat").shape == (10, 2)
    assert easa.add_assignment_newlines(P, "one_token", "spatial_unpad").shape == (9, 2)
    assert torch.equal(easa.add_assignment_newlines(P, "no_token", "flat"), P.flatten(0, 1))
    ids = torch.tensor([[11, -200, 12, 13], [21, -200, 22, 0]])
    padding = ids != 0
    memories = [grid, torch.tensor([[1., 0., 0.], [0., 1., 0.]])]
    right = easa.align_decoder_assignments(ids, padding, memories, 15, -200, "right")
    left = easa.align_decoder_assignments(ids, padding, memories, 15, -200, "left")
    assert torch.equal(right[0, 1:13, :2], grid)
    assert (right[0, [0, 13, 14]] == 0).all()
    assert torch.equal(left[1, 12:14], memories[1])
    truncated = easa.align_decoder_assignments(ids[:1], padding[:1], memories[:1], 4, -200, "right")
    assert torch.equal(truncated[0, 1:], grid[:3])


def test_htm(easa, htm):
    for layout, representation, grouped in (("legacy", "pooled", False), ("chronological", "tokens", True), ("frames", "tokens", False)):
        ctrl = htm.HTMMemoryController(8, recent_window_W=2, memory_layout=layout,
                                       event_representation=representation, group_tokens=grouped)
        ctrl.reset(2)
        reference = torch.zeros(2, 1, 4, 3)
        for b, state in enumerate(ctrl.states):
            vectors = torch.eye(3)[:3 - b]
            for i, z in enumerate(vectors):
                event = htm.HTMEvent(z, i, i, float(i), 1, tokens=z.repeat(4, 1))
                state.events.append(event)
            state.recent.append(vectors[-1].repeat(4, 1))
            state.recent_t.append(len(vectors) - 1)
        before = ctrl.materialize_context_memory(reference).clone()
        P = ctrl.materialize_event_assignments(reference, temperature=0.1)
        after = ctrl.materialize_context_memory(reference)
        assert torch.equal(before, after) and P.shape[:3] == after.shape[:3]
        for b in range(2):
            values = after[b].flatten(0, 1)
            memberships = P[b].flatten(0, 1)
            active = values.norm(dim=-1) > 0
            prototypes = torch.stack([ev.z for ev in ctrl.states[b].events])
            expected = easa.soft_event_assignments(values[active], prototypes, 0.1, owners=values[active].argmax(-1))
            assert torch.allclose(memberships[active, :len(prototypes)], expected)
            assert torch.allclose(memberships[active].sum(-1), torch.ones(int(active.sum())))
            assert (memberships[~active] == 0).all()
        assert not P.requires_grad


def main():
    torch.manual_seed(42)
    easa, htm = load_modules()
    test_formula(easa)
    test_alignment(easa)
    test_htm(easa, htm)
    identical = torch.ones(2, 3)
    membership = easa.soft_event_assignments(identical, identical, 0.1, owners=torch.tensor([0, 1]))
    assert membership[0, 0] > membership[0, 1] and membership[1, 1] > membership[1, 0]
    assert (membership > 0).all()  # finite ownership prior is not a rigid event mask
    for values in ((-1., .1, 1e-6), (1., 0., 1e-6), (1., .1, 0.), (float("nan"), .1, 1e-6)):
        try:
            easa.validate_easa_config(*values)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid EASA configuration accepted")
    print("PASS EASA formula, SDPA equivalence, causal/padding masks, mixed precision and gradients")
    print("PASS cached-query offsets, newline/truncation/batch alignment and HTM eviction/order metadata")


if __name__ == "__main__":
    main()
