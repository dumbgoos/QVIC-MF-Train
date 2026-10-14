"""Event priors for decoder memory self-attention (EASA).

Assignments are a detached structural prior. Zero assignment rows mark text,
padding and newline tokens, whose attention is left unchanged.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F


def validate_easa_config(strength: float, temperature: float, epsilon: float, owner_bias: float = 2.0) -> None:
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("EASA strength must be finite and nonnegative")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("EASA assignment temperature must be finite and positive")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("EASA epsilon must be finite and positive")
    if not math.isfinite(owner_bias) or owner_bias < 0:
        raise ValueError("EASA owner bias must be finite and nonnegative")


@torch.no_grad()
def soft_event_assignments(observations: torch.Tensor, prototypes: torch.Tensor, temperature: float,
                           owners: Optional[torch.Tensor] = None, owner_bias: float = 2.0) -> torch.Tensor:
    """Cosine-softmax membership, with a finite prior on actual HTM ownership.

    logits = cosine / temperature + owner_bias * one_hot(HTM event).
    The prior disambiguates temporally distinct events with identical prototypes.
    """
    validate_easa_config(1.0, temperature, 1e-6, owner_bias)
    if observations.ndim != 2 or prototypes.ndim != 2 or observations.shape[-1] != prototypes.shape[-1]:
        raise ValueError("EASA observations/prototypes must be matrices with matching hidden dimensions")
    if prototypes.shape[0] == 0:
        raise ValueError("EASA requires at least one HTM event")
    logits = F.normalize(observations.float(), dim=-1) @ F.normalize(prototypes.float(), dim=-1).T
    logits = logits / temperature
    if owners is not None:
        if owners.shape != (observations.shape[0],):
            raise ValueError("EASA ownership must have one event index per observation")
        logits = logits + owner_bias * F.one_hot(owners.to(device=logits.device), prototypes.shape[0])
    return logits.softmax(dim=-1)


def add_assignment_newlines(assignments: torch.Tensor, position: str, merge_type: str, faster_video: bool = False) -> torch.Tensor:
    """Mirror the decoder's [groups, C, events] flatten/grid/frame transforms.

    Newline tokens receive zero membership: they are not visual observations.
    """
    groups, C, E = assignments.shape
    if position == "grid":
        side = math.isqrt(C)
        if side * side != C:
            raise ValueError("EASA grid memory must have square C-token groups")
        if faster_video:
            raise ValueError("EASA does not support add_faster_video grid interleaving")
        grid = assignments.reshape(groups, side, side, E)
        return torch.cat((grid, assignments.new_zeros(groups, side, 1, E)), dim=2).reshape(-1, E)
    if position == "frame":
        return torch.cat((assignments, assignments.new_zeros(groups, 1, E)), dim=1).reshape(-1, E)
    if position in ("one_token", "no_token"):
        flat = assignments.reshape(-1, E)
        if position == "one_token" and "unpad" in merge_type:
            flat = torch.cat((flat, flat.new_zeros(1, E)))
        return flat
    raise ValueError(f"Unknown memory newline position: {position}")


def align_decoder_assignments(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    memory_assignments: Sequence[torch.Tensor],
    sequence_length: int,
    image_token_index: int,
    padding_side: str,
) -> torch.Tensor:
    """Mirror single-image placeholder replacement, truncation and batch padding."""
    E = max(p.shape[-1] for p in memory_assignments)
    out = memory_assignments[0].new_zeros(input_ids.shape[0], sequence_length, E)
    for b, memberships in enumerate(memory_assignments):
        ids = input_ids[b] if attention_mask is None else input_ids[b][attention_mask[b].bool()]
        indices = torch.where(ids == image_token_index)[0]
        if indices.numel() != 1:
            raise ValueError("EASA requires exactly one video placeholder per sample")
        start = int(indices[0])
        expanded_length = min(ids.numel() - 1 + memberships.shape[0], sequence_length)
        count = min(memberships.shape[0], max(expanded_length - start, 0))
        offset = sequence_length - expanded_length if padding_side == "left" else 0
        out[b, offset + start:offset + start + count, :memberships.shape[-1]] = memberships[:count]
    return out


def apply_easa_attention_mask(
    attention_mask: Optional[torch.Tensor],
    assignments: torch.Tensor,
    *,
    query_length: int,
    key_length: int,
    past_length: int,
    dtype: torch.dtype,
    device: torch.device,
    strength: float,
    epsilon: float,
) -> Optional[torch.Tensor]:
    """Add lambda*log(P P^T + epsilon), only on visual-memory pairs.

    The original additive causal/padding mask is preserved. Cached answer
    queries lie beyond P and have no event prior. Build once per model forward,
    then share the head-independent mask across layers and checkpoint replay.
    """
    validate_easa_config(strength, 1.0, epsilon)
    if strength == 0:
        return attention_mask
    if assignments.ndim != 3:
        raise ValueError("EASA assignments must have shape [batch, sequence, events]")
    B, N, _ = assignments.shape
    end = min(past_length + query_length, N)
    if past_length >= end:
        return attention_mask
    P = assignments.detach().to(device=device, dtype=torch.float32)
    valid = P.sum(dim=-1) > 0
    queries = P[:, past_length:end]
    keys = P[:, :min(N, key_length)]
    pairs = valid[:, past_length:end, None] & valid[:, None, :keys.shape[1]]
    if not pairs.any():
        return attention_mask
    bias = strength * torch.log(queries @ keys.transpose(-1, -2) + epsilon)
    bias = torch.where(pairs, bias, 0.0).to(dtype=dtype)
    if attention_mask is None:
        q = torch.arange(past_length, past_length + query_length, device=device)
        k = torch.arange(key_length, device=device)
        mask = torch.zeros(B, 1, query_length, key_length, device=device, dtype=dtype)
        mask.masked_fill_(k[None, :] > q[:, None], torch.finfo(dtype).min)
    else:
        if attention_mask.dtype == torch.bool or attention_mask.shape != (B, 1, query_length, key_length):
            raise ValueError("EASA expects a head-independent additive 4D attention mask")
        mask = attention_mask.to(device=device, dtype=dtype).clone()
    mask[:, 0, :end - past_length, :keys.shape[1]] += bias
    return mask
