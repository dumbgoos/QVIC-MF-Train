# Copyright (c) 2026 Fujitsu Limited and the QViC-MF authors.
#
# This file is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).
# https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# This module contains the core QMSA (Question-guided Mask Selective Attention)
# algorithm implementation for QViC. It is intentionally separated from the
# Apache-2.0-licensed `qvic/model/language_model/modeling_qwen2.py` (derived from
# HuggingFace Transformers v4.40.0) so that the upstream attention code stays
# close to its original form and the proprietary attention biasing logic is
# clearly attributed.
#
# Paper references (eq.X) below correspond to the QViC paper.

"""QMSA: Question-guided Mask Selective Attention.

Provides four small functions that the Qwen2 attention layers call as hooks
when ``token_ranges`` is supplied. They implement the QMSA biasing described
in the QViC paper (eq.9, eq.10):

    apply_qmsa_eager(attn_weights, token_ranges)
        In-place additive bias on the eager-attention pre-softmax weights so
        context-token rows are guided by the user-query mean attention vector
        on the image-token columns.

    extract_qmsa_attn_output_eager(attn_weights, token_ranges)
        Slice the post-softmax eager attention map to the (query_user, image)
        block for ``output_attentions`` use.

    apply_qmsa_sdpa(attention_mask, query_states, key_states, *, num_heads,
                    head_dim, bsz, q_len, kv_seq_len, token_ranges, dtype, device)
        Build an ``attn_bias`` tensor (replacing the upstream ``attn_mask``
        argument of ``F.scaled_dot_product_attention``) that bakes in the QMSA
        guide vector before SDPA runs.

    extract_qmsa_attn_output_sdpa(query_states, key_states, attn_bias,
                                  token_ranges, head_dim)
        Reconstruct the [q2v ; c2v] attention map post-hoc from the SDPA inputs
        for ``output_attentions`` (since SDPA does not return weights).
"""

from typing import List, Dict, Optional
import math

import torch
import torch.nn.functional as F


__all__ = [
    "apply_qmsa_eager",
    "extract_qmsa_attn_output_eager",
    "apply_qmsa_sdpa",
    "extract_qmsa_attn_output_sdpa",
]


def apply_qmsa_eager(attn_weights: torch.Tensor, token_ranges: List[Dict]) -> torch.Tensor:
    """Apply QMSA bias on eager-attention pre-softmax ``attn_weights`` in place.

    For each batch element, computes the mean ``query_user -> image`` attention
    vector and adds it to the ``context -> image`` rows (paper eq.9, eq.10
    expressed directly on ``attn_weights``).

    Args:
        attn_weights: ``[B, H, Q, K]`` pre-softmax attention scores.
        token_ranges: per-batch list of dicts with keys ``range_image``,
            ``range_query_user``, ``range_context`` (each ``[start, end]``).

    Returns:
        The same ``attn_weights`` tensor, mutated in place.
    """
    bsz = attn_weights.size(0)
    for batch_idx in range(bsz):
        range_image = token_ranges[batch_idx]["range_image"]
        range_query_user = token_ranges[batch_idx]["range_query_user"]
        range_context = token_ranges[batch_idx]["range_context"]

        attn_weights_query2vision = attn_weights[
            batch_idx, :, range_query_user[0]:range_query_user[1], range_image[0]:range_image[1]
        ].to(torch.float32)
        guide_vector = attn_weights_query2vision.mean(dim=1, keepdim=True)  # [H, 1, K_i]
        attn_weights_context2vision = attn_weights[
            batch_idx, :, range_context[0]:range_context[1], range_image[0]:range_image[1]
        ].to(torch.float32)
        attn_weights_context2vision_guided = attn_weights_context2vision + guide_vector
        attn_weights[
            batch_idx, :, range_context[0]:range_context[1], range_image[0]:range_image[1]
        ] = attn_weights_context2vision_guided.to(attn_weights.dtype)

    return attn_weights


def extract_qmsa_attn_output_eager(attn_weights: torch.Tensor, token_ranges: List[Dict]) -> torch.Tensor:
    """Slice eager ``attn_weights`` to the QMSA query-user -> image block.

    Args:
        attn_weights: ``[B, H, Q, K]`` post-softmax attention map.
        token_ranges: per-batch list of dicts; only batch index 0 is consulted
            here, matching the existing single-batch debug behavior.

    Returns:
        ``attn_weights[:, :, q0:q1, i0:i1]``.
    """
    range_image = token_ranges[0]["range_image"]
    range_query_user = token_ranges[0]["range_query_user"]
    return attn_weights[:, :, range_query_user[0]:range_query_user[1], range_image[0]:range_image[1]]


def apply_qmsa_sdpa(
    attention_mask: Optional[torch.Tensor],
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    bsz: int,
    q_len: int,
    kv_seq_len: int,
    token_ranges: List[Dict],
) -> torch.Tensor:
    """Build the SDPA ``attn_bias`` tensor with QMSA guidance baked in.

    The SDPA backend does not expose pre-softmax weights, so we instead inject
    the QMSA bias into the additive attention mask (paper eq.9). The
    ``query_user -> image`` logits are recomputed from ``q``/``k`` projections
    (paper eq.10) and the per-head mean is added to the ``context -> image``
    bias rows.

    Args:
        attention_mask: original 4D causal/padding mask, or ``None``.
        query_states / key_states: ``[B, H, *, D]`` projected Q/K (after
            rotary + KV repeat).
        num_heads / head_dim / bsz / q_len / kv_seq_len: shape metadata from
            the calling attention layer.
        token_ranges: per-batch list of dicts with ``range_image``,
            ``range_query_user``, ``range_context``, and optionally
            ``guiding_context2vision`` (default ``True``).

    Returns:
        ``attn_bias`` of shape ``[B, H, q_len, kv_seq_len]`` to be passed as
        ``attn_mask`` to ``F.scaled_dot_product_attention``.
    """
    # 1. Mirror the original SDPA mask preparation (without token_ranges).
    attn_bias = None
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            attn_bias = attention_mask
        else:
            attn_bias = attention_mask.to(dtype=query_states.dtype)

        if attn_bias.shape[1] == 1 and num_heads > 1:
            attn_bias = attn_bias.expand(bsz, num_heads, q_len, kv_seq_len).clone()

    # 2. Promote bool bias to a float bias so we can add the QMSA guide vector.
    if attn_bias is None or attn_bias.dtype == torch.bool:
        float_bias = torch.zeros(
            (bsz, 1, q_len, kv_seq_len), dtype=query_states.dtype, device=query_states.device
        )
        if attn_bias is not None and attn_bias.dtype == torch.bool:
            float_bias = float_bias.masked_fill(attn_bias, torch.finfo(float_bias.dtype).min)  # -inf
        attn_bias = float_bias

    # 3. Add per-batch QMSA guide vector to the context -> image rows.
    for b in range(bsz):
        range_image = token_ranges[b]["range_image"]
        range_query_user = token_ranges[b]["range_query_user"]
        range_context = token_ranges[b]["range_context"]
        guiding_context2vision = token_ranges[b].get("guiding_context2vision", True)

        if not guiding_context2vision:
            # No guidance for this batch element; leave the bias untouched.
            continue

        i0, i1 = range_image
        q0, q1 = range_query_user
        c0, c1 = range_context

        with torch.no_grad():
            q_user = query_states[b, :, q0:q1, :].detach()  # [H, Q_u, D]
            k_image = key_states[b, :, i0:i1, :].detach()   # [H, K_i, D]
            # paper: eq.(10)
            logits_q2v = torch.matmul(q_user, k_image.transpose(-1, -2)) / math.sqrt(head_dim)  # [H, Q_u, K_i]
            guide_vec = logits_q2v.mean(dim=1, keepdim=True)  # [H, 1, K_i]
            # paper: eq.(9)
            attn_bias[b, :, c0:c1, i0:i1] += guide_vec  # broadcast on Q_c

    return attn_bias


def extract_qmsa_attn_output_sdpa(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    attn_bias: torch.Tensor,
    token_ranges: List[Dict],
    *,
    head_dim: int,
) -> torch.Tensor:
    """Reconstruct the [q2v ; c2v] attention map for SDPA ``output_attentions``.

    Since ``F.scaled_dot_product_attention`` does not return attention weights,
    we recompute the relevant slices on demand using the same Q/K projections
    that were fed to SDPA, then softmax them with the actual ``attn_bias`` so
    the reported map matches the executed attention.

    Args:
        query_states / key_states: ``[B, H, *, D]`` projected Q/K.
        attn_bias: ``[B, H, q_len, kv_seq_len]`` bias used by SDPA.
        token_ranges: per-batch list of dicts with ``range_image``,
            ``range_query_user``, ``range_context``, ``num_frames``,
            ``num_contexts``.
        head_dim: per-head dimension.

    Returns:
        ``[B, H, Q_u + Q_c, K_i]`` attention map (concatenation of q2v and c2v).
    """
    bsz = query_states.size(0)
    out_dtype = query_states.dtype
    attn_weights_out = []
    for b in range(bsz):
        range_image = token_ranges[b]["range_image"]
        range_query_user = token_ranges[b]["range_query_user"]
        range_context = token_ranges[b]["range_context"]
        num_frames = token_ranges[b]["num_frames"]
        num_contexts = token_ranges[b]["num_contexts"]
        i0, i1 = range_image
        c0, c1 = range_context
        q0, q1 = range_query_user
        Ni = (i1 - i0) // num_frames    # num image tokens per frame
        Nc = (c1 - c0) // num_contexts  # num context tokens per frame
        assert (i1 - i0) % num_frames == 0
        assert (c1 - c0) % num_contexts == 0
        assert num_frames == num_contexts

        # q2v
        q_user = query_states[b, :, q0:q1, :]  # [H, Q_u, D]
        k_all = key_states[b, :, :, :]         # [H, K, D]
        logits_q2v = torch.matmul(q_user, k_all.transpose(-1, -2)) / math.sqrt(head_dim)  # [H, Q_u, K]
        logits_q2v += attn_bias[b, :, q0:q1, :].detach()
        # NOTE: Query rows q0:q1 are intentionally NOT QMSA-guided.
        attn_weight_q2v = F.softmax(logits_q2v, dim=-1).to(out_dtype)  # [H, Q_u, K]
        attn_weight_q2v = attn_weight_q2v[:, :, i0:i1]                  # [H, Q_u, K_i]

        # c2v (per-frame block-wise softmax then concatenated)
        q_ctx = query_states[b, :, c0:c1, :]   # [H, Q_c, D]
        k_all = key_states[b, :, :, :]         # [H, K, D]
        logits_c2v = torch.matmul(q_ctx, k_all.transpose(-1, -2)) / math.sqrt(head_dim)  # [H, Q_c, K]
        logits_c2v += attn_bias[b, :, c0:c1, :].detach()
        logits_c2v_concat = []
        for f in range(num_frames):
            i0_tar, i1_tar = (i0 + f * Ni, i0 + (f + 1) * Ni)
            c0_tar, c1_tar = f * Nc, (f + 1) * Nc
            logits_c2v_f = logits_c2v[:, c0_tar:c1_tar, i0_tar:i1_tar]  # [H, Nc, Ni]
            logits_c2v_concat.append(logits_c2v_f)
        logits_c2v_concat = torch.cat(logits_c2v_concat, dim=-1)  # [H, Nc, K_i]
        attn_weight_c2v = F.softmax(logits_c2v_concat, dim=-1).to(out_dtype)

        attn_weights_out_b = torch.cat([attn_weight_q2v, attn_weight_c2v], dim=-2)
        attn_weights_out.append(attn_weights_out_b)

    return torch.stack(attn_weights_out, dim=0)  # [B, H, Q_u+Q_c, K_i]
