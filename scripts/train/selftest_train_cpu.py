#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch

from qvic.constants import IGNORE_INDEX
from qvic.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM
from qvic.model.language_model.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM
from qvic.train.data import DataCollatorForQViC, build_prompt_and_target
from qvic.train.qvic_trainer import save_qvic_checkpoint


def tiny_config(vocab_size: int, model_max_length: int, memory_length: int) -> LlavaQwenConfig:
    cfg = LlavaQwenConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=model_max_length,
        rms_norm_eps=1e-6,
        attn_implementation="sdpa",
    )
    # LLaVA-Video-7B-Qwen2 multimodal settings, verbatim.
    cfg.mm_vision_tower = "google/siglip-so400m-patch14-384"
    cfg.mm_hidden_size = 1152
    cfg.mm_projector_type = "mlp2x_gelu"
    cfg.mm_patch_merge_type = "spatial_unpad"
    cfg.mm_newline_position = "grid"
    cfg.mm_spatial_pool_mode = "bilinear"
    cfg.mm_spatial_pool_stride = 2
    cfg.mm_vision_select_layer = -2
    cfg.mm_vision_select_feature = "patch"
    cfg.image_aspect_ratio = "anyres_max_9"
    cfg.mm_use_im_start_end = False
    cfg.mm_use_im_patch_token = False
    cfg.add_faster_video = False
    cfg.tokenizer_model_max_length = model_max_length
    cfg.tokenizer_padding_side = "right"
    cfg.use_cache = False
    # `set_context_memory_length()` only touches the model instance; the value
    # has to be on the config too or it will not survive into config.json.
    cfg.context_memory_length = memory_length
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=4, help="K. Kept small so SigLIP stays cheap on CPU.")
    ap.add_argument("--context-tokens", type=int, default=16, help="C")
    ap.add_argument("--memory-length", type=int, default=16, help="L (>= K to exercise the interpolation)")
    ap.add_argument("--model-base", default="lmms-lab/LLaVA-Video-7B-Qwen2")
    ap.add_argument("--htm", action="store_true", help="Check Stage B HTM through the real compressor and decoder.")
    ap.add_argument("--htm-grouped", action="store_true", help="Use 16-token events and native decoder groups.")
    args = ap.parse_args()
    if args.htm_grouped and not args.htm:
        ap.error("--htm-grouped requires --htm")

    torch.manual_seed(0)
    torch.set_num_threads(min(8, os.cpu_count() or 8))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 151643

    model_max_length = 8192
    cfg = tiny_config(152064, model_max_length, args.memory_length)

    print("== building tiny LlavaQwen (loads the real SigLIP tower) ==")
    model = LlavaQwenForCausalLM(cfg)
    model.encoder = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size, num_hidden_layers=cfg.num_hidden_layers,
            num_attention_heads=cfg.num_attention_heads, num_key_value_heads=cfg.num_key_value_heads,
            max_position_embeddings=model_max_length, attn_implementation="sdpa",
        )
    )
    with torch.no_grad():
        model.get_model().image_newline.normal_(std=0.02)  # from_pretrained supplies this in a real run

    class _ModelArgs:
        context_embed_tokens = args.context_tokens
        question_guided_selective_attention = True
        guiding_context2vision = True
        ctx_attn_mask_type = "framewise"

    model.initialize_context_embed(_ModelArgs)
    with torch.no_grad():
        model.context_embed.weight.normal_(std=1.0 / cfg.hidden_size ** 0.5)
    model.pad_token_id = tokenizer.pad_token_id
    model.set_context_memory_length(args.memory_length)
    model.max_frame_num_encoder = args.frames
    model.context_condition_frame_num = max(1, args.frames // 2)
    model.compress_with_relevance = False   # training: one-way, K_r = 0
    model.fill_context_memory = True
    model.train_qvic_freeze_encoder = False
    if args.htm:
        model.initialize_htm(hidden_size=32, recent_window_W=2,
                             token_budget=args.memory_length,
                             event_representation="tokens" if args.htm_grouped else "pooled",
                             memory_layout="chronological" if args.htm_grouped else "legacy",
                             group_tokens=args.htm_grouped,
                             reset_thresholds_per_video=args.htm_grouped)
        model.set_htm_stage_a(False)
        model.fill_context_memory = False

    print("== applying LoRA to the compressor only ==")
    from peft import LoraConfig, get_peft_model
    target = r"encoder\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
    model.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                             bias="none", target_modules=target,
                                             task_type="CAUSAL_LM"))
    for name, param in model.named_parameters():
        if "context_embed" in name or (args.htm and "htm_predictor" in name):
            param.requires_grad = True

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any("context_embed" in n for n in trainable), "context seed embedding is frozen"
    assert all("lora_" in n or "context_embed" in n or (args.htm and "htm_predictor" in n) for n in trainable), \
        f"unexpected trainable params: {[n for n in trainable if 'lora_' not in n and 'context_embed' not in n]}"
    assert not any(n.startswith("base_model.model.model.layers") and "lora_" in n for n in trainable), \
        "LoRA leaked into the decoder"
    print(f"   trainable tensors: {len(trainable)} "
          f"({sum('lora_' in n for n in trainable)} LoRA + "
          f"{sum('context_embed' in n for n in trainable)} context_embed)")

    print("== building one batch ==")
    frames = np.random.randint(0, 255, (args.frames, 240, 320, 3), dtype=np.uint8)
    pixel_values = model.get_vision_tower().image_processor.preprocess(
        frames, return_tensors="pt")["pixel_values"]
    input_ids, labels, input_ids_q = build_prompt_and_target(
        "What happens in the video?", "A cat knocks a glass off a table.", tokenizer)
    collate = DataCollatorForQViC(tokenizer=tokenizer, pad_token_id=tokenizer.pad_token_id,
                                  frame_dtype=torch.float32)
    batch = collate([{
        "input_ids": input_ids, "labels": labels, "input_ids_q": input_ids_q,
        "image": pixel_values, "modality": "video", "record_id": "selftest",
    }])
    print(f"   images {tuple(batch['images'][0].shape)}  input_ids {tuple(batch['input_ids'].shape)}  "
          f"input_ids_q {tuple(batch['input_ids_q'].shape)}  "
          f"supervised tokens {(batch['labels'] != IGNORE_INDEX).sum().item()}")

    print("== forward ==")
    model.train()
    out = model(**batch)
    loss = out.loss
    print(f"   loss = {loss.item():.4f}")
    assert torch.isfinite(loss), "loss is not finite"

    # MF stretches frame entries; HTM materializes a flat token-budget memory.
    base = model.get_base_model()
    mem = base.memory_for_each_batch
    print(f"   context memory: {tuple(mem.shape)}  (K={args.frames}, C={args.context_tokens}, L={args.memory_length})")
    assert mem.shape[0] == 1
    if args.htm:
        assert mem.shape[2] == (args.context_tokens if args.htm_grouped else 1)
        assert 0 < mem.shape[1] * mem.shape[2] <= args.memory_length
    else:
        assert mem.shape[2] == args.context_tokens
        assert args.frames <= mem.shape[1] <= args.memory_length

    print("== backward ==")
    loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    missing = [n for n, g in grads.items() if g is None]
    assert not missing, f"no gradient reached: {missing[:5]}"
    dead = [n for n, g in grads.items() if not torch.isfinite(g).all()]
    assert not dead, f"non-finite gradient: {dead[:5]}"
    ce_grad = next(g for n, g in grads.items() if "context_embed" in n)
    lora_grad_norm = max(g.norm().item() for n, g in grads.items() if "lora_" in n)
    print(f"   |grad context_embed| = {ce_grad.norm().item():.3e}   "
          f"max |grad LoRA| = {lora_grad_norm:.3e}")
    assert ce_grad.norm().item() > 0, "context seed embedding received a zero gradient"
    assert lora_grad_norm > 0, "no LoRA parameter received a gradient"
    if args.htm:
        pred_norm = max(g.norm().item() for n, g in grads.items() if "htm_predictor" in n)
        assert pred_norm > 0, "predictor received no nonzero gradient"
        print(f"   max |grad predictor| = {pred_norm:.3e}")

    print("== checkpoint ==")
    with tempfile.TemporaryDirectory() as tmp:
        model.config.use_cache = True
        save_qvic_checkpoint(model, tmp, lora_bias="none", is_main_process=True)
        files = sorted(os.listdir(tmp))
        print(f"   {files}")
        for required in ("config.json", "adapter_config.json", "adapter_model.bin",
                         "non_lora_trainables.bin"):
            assert required in files, f"builder.py needs {required}, got {files}"
        conf = json.load(open(os.path.join(tmp, "config.json")))
        assert conf["model_type"] == "llava_qwen", conf["model_type"]
        for key, want in (("context_embed_tokens", args.context_tokens),
                          ("context_memory_length", args.memory_length),
                          ("question_guided_selective_attention", True),
                          ("guiding_context2vision", True),
                          ("ctx_attn_mask_type", "framewise")):
            assert conf.get(key) == want, f"config.json[{key}] = {conf.get(key)!r}, expected {want!r}"
        non_lora = torch.load(os.path.join(tmp, "non_lora_trainables.bin"), map_location="cpu")
        assert "base_model.model.context_embed.weight" in non_lora
        assert all("context_embed" in k or (args.htm and "htm_predictor" in k) for k in non_lora)
        if args.htm:
            assert any("htm_predictor" in k for k in non_lora)
            assert conf["htm_token_budget"] == args.memory_length
            assert conf["htm_group_tokens"] == args.htm_grouped
            assert conf["htm_event_representation"] == ("tokens" if args.htm_grouped else "pooled")
        lora_sd = torch.load(os.path.join(tmp, "adapter_model.bin"), map_location="cpu")
        assert lora_sd and all("encoder." in k for k in lora_sd), \
            f"LoRA weights outside the compressor: {[k for k in lora_sd if 'encoder.' not in k][:3]}"
        print(f"   {len(lora_sd)} LoRA tensors, all under model.encoder; "
              f"non_lora_trainables key = {list(non_lora)[0]}")

    print("\nSELFTEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
