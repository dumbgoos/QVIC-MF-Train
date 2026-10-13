# #@HTM — tiny helpers for QA smoke / LVBench (no new harness).
"""Shared HTM eval wiring used by ``model_video_lvbench`` / smoke wrappers."""

from __future__ import annotations

import os

from qvic.mm_utils import get_model_name_from_path
from qvic.utils import rank0_print


def resolve_htm_model_name(model_path: str, model_base: str | None = None) -> str:
    """Ensure builder enters the QViC LoRA branch for oddly named HTM ckpts.

    ``get_model_name_from_path(.../htm_stage_b_.../checkpoint-5000)`` already
    contains ``htm``. If a future dir drops that token, append ``-qvic`` when
    adapter + non_lora markers are present.
    """
    name = get_model_name_from_path(model_path)
    lower = name.lower()
    if "qvic" in lower or "htm" in lower or "llava" in lower:
        return name
    if model_base is None:
        return name
    adapter = os.path.join(model_path, "adapter_model.bin")
    adapter_cfg = os.path.join(model_path, "adapter_config.json")
    non_lora = os.path.join(model_path, "non_lora_trainables.bin")
    if os.path.isfile(adapter) and os.path.isfile(adapter_cfg) and os.path.isfile(non_lora):
        forced = f"{name}-qvic"
        rank0_print(f"[HTM] model_name override for QViC builder branch: {name} -> {forced}")
        return forced
    return name


def apply_htm_eval_runtime(
    model,
    *,
    htm_stage_a: bool = False,
    htm_quantile: float = 0.9,
    context_memory_length: int | None = None,
    htm_token_budget: int | None = None,
    htm_event_representation: str | None = None,
    htm_memory_layout: str | None = None,
    htm_group_tokens: bool | None = None,
) -> bool:
    """Finish HTM runtime knobs after ``load_pretrained_model``.

    Builder already ``initialize_htm`` + loads ``htm_predictor`` when non_lora
    contains those keys. Here we set Stage A/B mode, sync L / quantile, and
    disable MF fill (HTM owns the write budget).

    Returns True if HTM write path is active.
    """
    active = bool(getattr(model, "use_htm", False) and getattr(model, "htm_predictor", None) is not None)
    if not active:
        rank0_print("[HTM] eval: HTM not active on loaded model (MF / base write path)")
        return False

    model.set_htm_stage_a(bool(htm_stage_a))
    model.set_use_htm(True)
    # K_r / compress_with_relevance is caller-controlled (False = write-fair;
    # True = paper-matched read feedback). Do not force it here.
    model.fill_context_memory = False

    ctrl = getattr(model, "htm_controller", None)
    kr_on = bool(getattr(model, "compress_with_relevance", False))
    kr_label = "on" if kr_on else "off"
    if ctrl is not None:
        config = getattr(model, "config", None)
        htm_event_representation = htm_event_representation if htm_event_representation is not None else getattr(config, "htm_event_representation", "pooled")
        htm_memory_layout = htm_memory_layout if htm_memory_layout is not None else getattr(config, "htm_memory_layout", "legacy")
        htm_group_tokens = htm_group_tokens if htm_group_tokens is not None else getattr(config, "htm_group_tokens", False)
        if htm_token_budget is None:
            htm_token_budget = getattr(config, "htm_token_budget", None)
        if context_memory_length is not None:
            ctrl.L = int(context_memory_length)
        if htm_token_budget is not None:
            if htm_token_budget <= 0:
                raise ValueError("HTM token budget must be positive")
            ctrl.L = int(htm_token_budget)
        if htm_event_representation not in ("pooled", "tokens"):
            raise ValueError("Unknown HTM event representation")
        ctrl.event_representation = htm_event_representation
        if htm_memory_layout not in ("legacy", "chronological", "frames"):
            raise ValueError("Unknown HTM memory layout")
        if htm_memory_layout == "frames" and htm_event_representation != "tokens":
            raise ValueError("Frame baseline requires token representation")
        ctrl.memory_layout = htm_memory_layout
        if htm_group_tokens and htm_event_representation != "tokens":
            raise ValueError("Grouped decoder memory requires token events")
        C = getattr(config, "context_embed_tokens", 16)
        if htm_group_tokens and (ctrl.L <= 0 or ctrl.L % C):
            raise ValueError("Grouped HTM budget must be a positive multiple of C")
        ctrl.group_tokens = bool(htm_group_tokens)
        if hasattr(ctrl, "thresholds") and hasattr(ctrl.thresholds, "quantile"):
            # Quantile alone is inert until write_batch feeds S_t/D_t (must keep
            # collect_threshold_stats=True at eval — see qvic_meta_mixin HTM write).
            ctrl.thresholds.quantile = float(htm_quantile)
        rank0_print(
            f"[HTM] eval runtime: stage_a={htm_stage_a} L={getattr(ctrl, 'L', '?')} "
            f"quantile={htm_quantile} K_r={kr_label} (compress_with_relevance={kr_on}) "
            f"event_representation={htm_event_representation} memory_layout={htm_memory_layout} grouped={htm_group_tokens}"
        )
    else:
        rank0_print(
            f"[HTM] eval runtime: stage_a={htm_stage_a} K_r={kr_label} "
            f"(compress_with_relevance={kr_on}; no controller?)"
        )
    return True
