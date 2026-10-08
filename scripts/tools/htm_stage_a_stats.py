#!/usr/bin/env python3
# #@HTM
"""HTM Stage A memory / consolidation statistics.

Primary gate tooling (design §7–§8): report token-budget usage under the HTM
write path vs an MF-style append/FIFO approximation on the same clip stream.

Modes
-----
* ``--smoke`` (default when no annotation path / encode): synthetic temporally
  coherent context tokens — CPU-friendly, no VLM weights.
* ``--annotation-path`` + ``--encode-videos``: load compressor + Stage A
  ``htm_predictor`` checkpoint and encode a LIMIT subset of real videos.
* ``--context-cache DIR``: load precomputed ``*.pt`` tensors shaped ``[T, C, D]``.

Does **not** run VQA. See ``scripts/eval/run_htm_stage_a_qa_smoke.sh`` / docs
for the existing ``qvic/eval`` next command.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

# #@HTM — load htm.py directly so --smoke works without the full VLM stack.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_HTM_PATH = _REPO_ROOT / "qvic" / "model" / "htm.py"
_spec = importlib.util.spec_from_file_location("qvic_htm_stage_a_stats", _HTM_PATH)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
HTMMemoryController = _mod.HTMMemoryController
HTMPredictor = _mod.HTMPredictor

# #@HTM — user server defaults (see docs/htm-stage-a-eval.md)
_DEFAULT_CODE = "/mnt/data_nvme1/wenbin/luoling/cvpr/code/ours/QVIC-MF-Train"
_DEFAULT_CKPT = (
    "/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_a_2xh100/checkpoint-5000"
)
_DEFAULT_ANN = "/mnt/sdb1/wenbin/luoling/LLaVA-Video-83K/annotations/all_sampled.json"
_DEFAULT_MEDIA = "/mnt/sdb1/wenbin/luoling/LVDU/LLaVA-Video-83K"
_DEFAULT_OUT_PARENT = (
    "/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_a_2xh100"
)


def _mean(xs: Sequence[float]) -> Optional[float]:
    return float(statistics.mean(xs)) if xs else None


def _median(xs: Sequence[float]) -> Optional[float]:
    return float(statistics.median(xs)) if xs else None


def _summarize_scores(xs: Sequence[float]) -> Dict[str, Any]:
    return {"mean": _mean(xs), "median": _median(xs), "n": len(xs)}


# #@HTM
def load_htm_predictor_weights(
    predictor: HTMPredictor, checkpoint: str, device: torch.device
) -> List[str]:
    """Load ``htm_predictor.*`` tensors from a Stage A checkpoint directory or .bin."""
    ckpt = Path(checkpoint)
    if ckpt.is_dir():
        bin_path = ckpt / "non_lora_trainables.bin"
    else:
        bin_path = ckpt
    if not bin_path.is_file():
        raise FileNotFoundError(f"HTM weights not found: {bin_path}")

    raw = torch.load(str(bin_path), map_location="cpu")
    if not isinstance(raw, dict):
        raise ValueError(f"expected state-dict dict in {bin_path}")

    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        key = k
        for prefix in (
            "base_model.model.htm_predictor.",
            "base_model.htm_predictor.",
            "model.htm_predictor.",
            "htm_predictor.",
        ):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        if key.startswith("gru.") or key.startswith("proj."):
            cleaned[key] = v

    if not cleaned:
        raise KeyError(
            f"no htm_predictor.* tensors in {bin_path}; keys sample={list(raw)[:12]}"
        )
    missing, unexpected = predictor.load_state_dict(cleaned, strict=False)
    predictor.to(device)
    predictor.eval()
    notes = [f"loaded {len(cleaned)} tensors from {bin_path}"]
    if missing:
        notes.append(f"missing_keys={list(missing)}")
    if unexpected:
        notes.append(f"unexpected_keys={list(unexpected)}")
    return notes


# #@HTM
def run_htm_on_clips(
    clips: torch.Tensor,  # [T, C, D]
    predictor: HTMPredictor,
    token_budget_L: int,
    recent_window_W: int,
    quantile: float,
    collect_threshold_stats: bool = True,
) -> Dict[str, Any]:
    """Online HTM write over a clip stream; returns composition + score stats."""
    T, C, D = clips.shape
    ctrl = HTMMemoryController(
        token_budget_L=token_budget_L,
        recent_window_W=recent_window_W,
        quantile=quantile,
    )
    ctrl.reset(1)
    step_stats: List[Dict] = []
    n_merges = 0
    with torch.no_grad():
        for t in range(T):
            c_tokens = clips[t]  # [C, D]
            ctrl.step_sample(
                ctrl.states[0],
                c_tokens,
                predictor,
                collect_threshold_stats=collect_threshold_stats,
                step_stats=step_stats,
            )
            if step_stats and step_stats[-1].get("merged"):
                n_merges += 1

    state = ctrl.states[0]
    comp = ctrl.memory_composition(state)
    s_vals = [float(s["S_t"]) for s in step_stats if s.get("S_t") is not None]
    d_vals = [float(s["D_t"]) for s in step_stats if s.get("D_t") is not None]
    # merge_rate over decisions that could merge (have pred + open event) ≈ T-1
    decidable = max(T - 1, 1)
    return {
        **comp,
        "n_clips": T,
        "C": C,
        "D": D,
        "n_merges": n_merges,
        "merge_rate": n_merges / decidable,
        "S_t": _summarize_scores(s_vals),
        "D_t": _summarize_scores(d_vals),
        "tau_s": ctrl.thresholds.tau_s,
        "tau_d": ctrl.thresholds.tau_d,
        "thresholds_ready": ctrl.thresholds.ready(),
        "steps": step_stats,
    }


# #@HTM
def run_mf_append_fifo(clips: torch.Tensor, token_budget_L: int) -> Dict[str, Any]:
    """MF-style baseline: append raw C tokens per clip; FIFO-drop oldest until |M|≤L.

    Approximation note: training MF without ``compress_with_relevance`` merges
    adjacent high-similarity entries; inference MF prunes by question relevance.
    Without questions here we use FIFO token eviction — fair for *capacity*
    comparison (same L, same C), not for relevance-ranking quality.
    """
    T, C, _D = clips.shape
    kept: List[torch.Tensor] = []
    n_evicted_clips = 0
    for t in range(T):
        kept.append(clips[t])
        while sum(int(x.shape[0]) for x in kept) > token_budget_L and kept:
            kept.pop(0)
            n_evicted_clips += 1
    final_M = int(sum(int(x.shape[0]) for x in kept))
    raw_uncapped = T * C
    return {
        "mode": "append_fifo",
        "approximation": (
            "FIFO token eviction under |M|≤L; not relevance-prune and not "
            "MF adjacent-similarity merge. Use for token-budget comparison only."
        ),
        "n_clips": T,
        "C": C,
        "raw_uncapped_tokens": raw_uncapped,
        "final_M": final_M,
        "token_budget_L": token_budget_L,
        "final_M_over_L": (final_M / token_budget_L) if token_budget_L > 0 else None,
        "n_evicted_clips": n_evicted_clips,
        "n_kept_clips": len(kept),
    }


def _reduction(htm_M: int, mf_M: int) -> Optional[float]:
    if mf_M <= 0:
        return None
    return (mf_M - htm_M) / mf_M


def make_synthetic_clips(
    n_clips: int, C: int, D: int, seed: int = 0
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    clips = []
    prev = torch.randn(C, D, generator=g)
    for t in range(n_clips):
        # Piecewise-stationary blocks → merges should fire inside a block.
        if t % 8 == 0:
            prev = torch.randn(C, D, generator=g)
        noise = torch.randn(C, D, generator=g) * 0.15
        cur = 0.85 * prev + noise
        prev = cur
        clips.append(cur)
    return torch.stack(clips, dim=0)


def load_annotation_subset(
    annotation_path: str,
    limit: int,
    video_list: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    with open(annotation_path, "r") as fh:
        records = json.load(fh)
    if video_list:
        want = set(video_list)
        records = [r for r in records if r.get("video") in want or r.get("id") in want]
    if limit and limit > 0:
        records = records[:limit]
    return records


def load_context_cache(cache_dir: str, limit: int) -> List[Tuple[str, torch.Tensor]]:
    paths = sorted(Path(cache_dir).glob("*.pt"))
    if limit and limit > 0:
        paths = paths[:limit]
    out: List[Tuple[str, torch.Tensor]] = []
    for p in paths:
        obj = torch.load(str(p), map_location="cpu")
        if isinstance(obj, dict) and "clips" in obj:
            clips = obj["clips"]
            vid = str(obj.get("video_id", p.stem))
        else:
            clips = obj
            vid = p.stem
        if not torch.is_tensor(clips) or clips.ndim != 3:
            raise ValueError(f"{p}: expected [T,C,D] tensor, got {type(clips)}")
        out.append((vid, clips.float()))
    return out


# #@HTM
def encode_videos_to_clips(
    records: List[Dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> List[Tuple[str, torch.Tensor]]:
    """Encode videos with the frozen compressor; return per-video [T,C,D] clips."""
    # Import heavy stack only when encoding.
    sys.path.insert(0, str(_REPO_ROOT))
    from transformers import AutoTokenizer

    from qvic.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM
    from qvic.model.language_model.modeling_qwen2 import Qwen2ForCausalLM
    from qvic.train.data import MediaLoader

    model_base = args.model_base
    print(f"[htm-stats] loading compressor base from {model_base}", flush=True)
    config = LlavaQwenConfig.from_pretrained(model_base)
    config.context_embed_tokens = args.context_tokens
    config.context_memory_length = args.token_budget_L

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = LlavaQwenForCausalLM.from_pretrained(
        model_base, torch_dtype=dtype, low_cpu_mem_usage=True, attn_implementation="sdpa"
    )
    model.encoder = Qwen2ForCausalLM.from_pretrained(
        model_base, torch_dtype=dtype, low_cpu_mem_usage=True, attn_implementation="sdpa"
    )

    class _MA:
        context_embed_tokens = args.context_tokens
        question_guided_selective_attention = True
        guiding_context2vision = True
        ctx_attn_mask_type = "framewise"

    model.initialize_context_embed(_MA)
    model.set_context_memory_length(args.token_budget_L)
    model.max_frame_num_encoder = args.num_frames
    model.context_condition_frame_num = 0
    model.compress_with_relevance = False
    model.fill_context_memory = False
    # #@HTM — stats encode path only needs the frozen compressor; HTM write runs later.
    model.use_htm = False

    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 151643
    model.pad_token_id = tokenizer.pad_token_id

    # Vision processor from the multimodal tower.
    vision_tower = model.get_vision_tower()
    if hasattr(vision_tower, "image_processor"):
        image_processor = vision_tower.image_processor
    else:
        from transformers import AutoProcessor

        image_processor = AutoProcessor.from_pretrained(model_base).image_processor

    media = MediaLoader(
        backend=args.media_backend,
        dataset_root=args.dataset_root,
        video_folder=args.video_folder or None,
        quiet_decoder=True,
    )

    # Minimal question ids so prepare_inputs_labels_for_multimodal can run.
    from qvic.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from qvic.conversation import conv_templates
    from qvic.mm_utils import tokenizer_image_token

    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\nDescribe the video.")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    q_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX)
    input_ids_q = torch.tensor([q_ids], dtype=torch.long, device=device)

    results: List[Tuple[str, torch.Tensor]] = []
    for rec in records:
        vid = str(rec.get("id") or rec.get("video"))
        print(f"[htm-stats] encode {vid}", flush=True)
        frames = media(rec["video"], args.num_frames)  # [K,H,W,3]
        pixel_values = image_processor.preprocess(frames, return_tensors="pt")[
            "pixel_values"
        ]
        pixel_values = pixel_values.to(device=device, dtype=dtype)
        clips = _reencode_context_clips(model, pixel_values, input_ids_q)
        results.append((vid, clips.cpu().float()))
        if args.dump_context_cache:
            os.makedirs(args.dump_context_cache, exist_ok=True)
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in vid)[:180]
            torch.save(
                {"video_id": vid, "clips": clips.cpu()},
                os.path.join(args.dump_context_cache, f"{safe}.pt"),
            )
    return results


def _reencode_context_clips(model, pixel_values, input_ids_q) -> torch.Tensor:
    """Collect compressor outputs ``[T, C, D]`` (HTM write disabled)."""
    T = pixel_values.shape[0]
    num_frame_clip = model.max_frame_num_encoder
    clip_tensors: List[torch.Tensor] = []
    model.context_memory = None
    num_clips = (T + num_frame_clip - 1) // num_frame_clip
    for i_clip in range(num_clips):
        start = i_clip * num_frame_clip
        end = min((i_clip + 1) * num_frame_clip, T)
        images_encode = [pixel_values[start:end]]
        modalities = ["video"]
        (
            _,
            _,
            attention_mask_encoder,
            _,
            inputs_embeds_,
            _,
            token_ranges,
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids_q,
            None,
            None,
            None,
            None,
            images_encode,
            modalities,
            None,
        )
        with torch.no_grad():
            encoder_outputs = model.forward_encoder(
                inputs_embeds_,
                attention_mask_encoder,
                token_ranges
                if model.question_guided_selective_attention
                or model.ctx_attn_mask_type == "single_frame"
                else None,
                output_attentions=False,
            )
            context_embeds = model.get_context_embeds_from_encoder_outputs(
                encoder_outputs, token_ranges=token_ranges
            )  # [B, T_clip, C, D]
        clip_tensors.append(context_embeds[0].detach().float().cpu())
    return torch.cat(clip_tensors, dim=0)  # [T, C, D]


def aggregate_video_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n_videos": 0}

    def _gather(path: str) -> List[float]:
        out = []
        for r in rows:
            cur: Any = r
            for p in path.split("."):
                cur = cur.get(p) if isinstance(cur, dict) else None
                if cur is None:
                    break
            if isinstance(cur, (int, float)) and not isinstance(cur, bool):
                out.append(float(cur))
        return out

    htm_M = _gather("htm.final_M")
    mf_M = _gather("mf_baseline.final_M")
    reductions = [
        _reduction(int(h), int(m))
        for h, m in zip(htm_M, mf_M)
        if m > 0
    ]
    reductions_f = [x for x in reductions if x is not None]
    mean_red = _mean(reductions_f)
    # Stage A ballpark 30–50% fewer tokens
    token_gate = None
    if mean_red is not None:
        token_gate = 0.30 <= mean_red <= 0.80  # allow headroom above 50%

    s_means = _gather("htm.S_t.mean")
    d_means = _gather("htm.D_t.mean")
    merge_rates = _gather("htm.merge_rate")
    m_over_l = _gather("htm.final_M_over_L")

    return {
        "n_videos": len(rows),
        "htm": {
            "final_M": {"mean": _mean(htm_M), "median": _median(htm_M)},
            "final_M_over_L": {"mean": _mean(m_over_l), "median": _median(m_over_l)},
            "merge_rate": {"mean": _mean(merge_rates), "median": _median(merge_rates)},
            "S_t_mean": {"mean": _mean(s_means), "median": _median(s_means)},
            "D_t_mean": {"mean": _mean(d_means), "median": _median(d_means)},
            "n_merges": {
                "mean": _mean(_gather("htm.n_merges")),
                "median": _median(_gather("htm.n_merges")),
            },
            "n_events": {
                "mean": _mean(_gather("htm.n_events")),
                "median": _median(_gather("htm.n_events")),
            },
        },
        "mf_baseline": {
            "final_M": {"mean": _mean(mf_M), "median": _median(mf_M)},
            "mode": rows[0].get("mf_baseline", {}).get("mode"),
        },
        "mean_token_reduction_vs_mf": mean_red,
        "median_token_reduction_vs_mf": _median(reductions_f),
        "stage_a_gate": {
            "token_reduction_target": "≈30–50% fewer tokens vs MF append/prune",
            "token_reduction_observed": mean_red,
            "token_gate_ballpark_pass": token_gate,
            "qa": "not_run",
            "qa_note": (
                "Stage A also requires QA not significantly hurt; run existing "
                "qvic/eval after HTM load is wired (see run_htm_stage_a_qa_smoke.sh)."
            ),
            "surprise_vs_boundaries": "analysis_only_not_run",
        },
    }


def format_human_summary(payload: Dict[str, Any]) -> str:
    agg = payload.get("aggregate", {})
    gate = agg.get("stage_a_gate", {})
    lines = [
        "=== HTM Stage A stats ===",
        f"videos: {agg.get('n_videos')}",
        f"HTM |M| mean/median: {agg.get('htm', {}).get('final_M')}",
        f"MF  |M| mean/median: {agg.get('mf_baseline', {}).get('final_M')}",
        f"token reduction vs MF: mean={gate.get('token_reduction_observed')} "
        f"(target ≈30–50%; ballpark_pass={gate.get('token_gate_ballpark_pass')})",
        f"merge_rate mean: {agg.get('htm', {}).get('merge_rate')}",
        f"S_t mean-of-means: {agg.get('htm', {}).get('S_t_mean')}",
        f"D_t mean-of-means: {agg.get('htm', {}).get('D_t_mean')}",
        f"QA: {gate.get('qa')} — {gate.get('qa_note')}",
        "Surprise vs boundaries: analysis-only (not computed here).",
    ]
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HTM Stage A memory statistics (#@HTM)")
    p.add_argument(
        "--checkpoint",
        default=os.environ.get("HTM_CKPT", _DEFAULT_CKPT),
        help="Stage A checkpoint dir (non_lora_trainables.bin) or .bin path",
    )
    p.add_argument("--model-base", default=os.environ.get(
        "MODEL_BASE", "lmms-lab/LLaVA-Video-7B-Qwen2"
    ))
    p.add_argument(
        "--annotation-path",
        default=os.environ.get("ANNOTATION_PATH", _DEFAULT_ANN),
    )
    p.add_argument(
        "--dataset-root",
        default=os.environ.get("DATASET_ROOT", _DEFAULT_MEDIA),
    )
    p.add_argument("--video-folder", default=os.environ.get("VIDEO_FOLDER", ""))
    p.add_argument("--media-backend", default=os.environ.get("MEDIA_BACKEND", "shards"))
    p.add_argument(
        "--output",
        default=os.environ.get(
            "HTM_STATS_OUT",
            os.path.join(_DEFAULT_OUT_PARENT, "htm_stage_a_stats.json"),
        ),
    )
    p.add_argument("--limit", type=int, default=int(os.environ.get("LIMIT", "8")))
    p.add_argument(
        "--video-list",
        default="",
        help="Comma-separated video ids/keys to keep (optional)",
    )
    p.add_argument("--token-budget-L", type=int, default=int(os.environ.get("L", "256")))
    p.add_argument("--context-tokens", type=int, default=int(os.environ.get("C", "16")))
    p.add_argument("--recent-window", type=int, default=int(os.environ.get("W", "4")))
    p.add_argument("--quantile", type=float, default=float(os.environ.get("QUANTILE", "0.75")))
    p.add_argument("--htm-hidden-size", type=int, default=-1)
    p.add_argument("--num-frames", type=int, default=int(os.environ.get("NUM_FRAMES", "64")))
    p.add_argument("--dim", type=int, default=32, help="Synthetic / smoke embedding dim")
    p.add_argument("--smoke-clips", type=int, default=48)
    p.add_argument("--smoke", action="store_true", help="Force synthetic smoke mode")
    p.add_argument(
        "--encode-videos",
        action="store_true",
        help="Load compressor + encode real videos (needs GPU-friendly box)",
    )
    p.add_argument("--context-cache", default="", help="Dir of precomputed [T,C,D].pt")
    p.add_argument(
        "--dump-context-cache",
        default="",
        help="When encoding, also write [T,C,D].pt for reuse",
    )
    p.add_argument("--mf-baseline", action="store_true", default=True)
    p.add_argument("--no-mf-baseline", action="store_true")
    p.add_argument("--device", default=os.environ.get("DEVICE", "cpu"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--keep-steps",
        action="store_true",
        help="Keep per-clip step records in JSON (larger)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.no_mf_baseline:
        args.mf_baseline = False

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[htm-stats] CUDA unavailable; falling back to cpu", flush=True)
        device = torch.device("cpu")

    video_list = [x.strip() for x in args.video_list.split(",") if x.strip()] or None
    streams: List[Tuple[str, torch.Tensor]] = []
    load_notes: List[str] = []
    mode = "smoke"

    if args.context_cache:
        mode = "context_cache"
        streams = load_context_cache(args.context_cache, args.limit)
    elif args.encode_videos and not args.smoke:
        mode = "encode_videos"
        if not os.path.isfile(args.annotation_path):
            print(
                f"[htm-stats] annotation missing: {args.annotation_path}; "
                "falling back to --smoke",
                flush=True,
            )
            mode = "smoke"
        else:
            records = load_annotation_subset(
                args.annotation_path, args.limit, video_list
            )
            streams = encode_videos_to_clips(records, args, device)
    else:
        # Auto-smoke when encode not requested (CPU-friendly default).
        mode = "smoke"

    if mode == "smoke":
        C = args.context_tokens
        D = args.dim
        for i in range(max(args.limit, 1)):
            clips = make_synthetic_clips(args.smoke_clips, C, D, seed=args.seed + i)
            streams.append((f"synthetic_{i}", clips))
        load_notes.append(
            "smoke mode: synthetic piecewise-stationary clips "
            f"(T={args.smoke_clips}, C={C}, D={D})"
        )

    # Predictor dim from first stream
    _T0, C0, D0 = streams[0][1].shape
    hidden = None if args.htm_hidden_size < 0 else args.htm_hidden_size
    predictor = HTMPredictor(dim=D0, hidden_size=hidden)
    predictor.to(device)
    predictor.eval()

    ckpt_loaded = False
    if args.checkpoint and os.path.exists(args.checkpoint):
        try:
            # Skip weight load in smoke when dims won't match 7B (≈3584).
            if mode == "smoke" and D0 < 512:
                load_notes.append(
                    f"skip checkpoint load in smoke (D={D0}); "
                    f"path kept for schema: {args.checkpoint}"
                )
            else:
                load_notes.extend(
                    load_htm_predictor_weights(predictor, args.checkpoint, device)
                )
                ckpt_loaded = True
        except Exception as exc:  # noqa: BLE001 — tooling: report and continue
            load_notes.append(f"checkpoint load failed: {exc}")
    else:
        load_notes.append(f"checkpoint not found (random predictor): {args.checkpoint}")

    rows: List[Dict[str, Any]] = []
    for vid, clips in streams:
        clips = clips.to(device)
        htm = run_htm_on_clips(
            clips,
            predictor,
            token_budget_L=args.token_budget_L,
            recent_window_W=args.recent_window,
            quantile=args.quantile,
            collect_threshold_stats=True,
        )
        row: Dict[str, Any] = {"video_id": vid, "htm": htm}
        if args.mf_baseline:
            mf = run_mf_append_fifo(clips.cpu(), args.token_budget_L)
            row["mf_baseline"] = mf
            row["token_reduction_vs_mf"] = _reduction(htm["final_M"], mf["final_M"])
        if not args.keep_steps:
            htm.pop("steps", None)
        rows.append(row)

    payload = {
        "config": {
            "mode": mode,
            "checkpoint": args.checkpoint,
            "checkpoint_loaded": ckpt_loaded,
            "model_base": args.model_base,
            "annotation_path": args.annotation_path,
            "dataset_root": args.dataset_root,
            "token_budget_L": args.token_budget_L,
            "context_tokens_C": args.context_tokens,
            "recent_window_W": args.recent_window,
            "quantile": args.quantile,
            "limit": args.limit,
            "device": str(device),
            "mf_baseline": (
                "append_fifo" if args.mf_baseline else None
            ),
            "notes": load_notes,
            "code_default": _DEFAULT_CODE,
            "design_ref": "docs/htm-design.md §7–§8 Stage A",
        },
        "videos": rows,
        "aggregate": aggregate_video_rows(rows),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    summary = format_human_summary(payload)
    summary_path = out_path.with_suffix(".txt")
    with open(summary_path, "w") as fh:
        fh.write(summary + "\n")
    print(summary)
    print(f"[htm-stats] wrote {out_path}")
    print(f"[htm-stats] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
