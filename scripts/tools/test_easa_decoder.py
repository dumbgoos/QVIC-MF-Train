"""Random tiny Qwen2 integration: real attention/cache/checkpoint and QViC packing.

Requires torch + the project's Transformers version; no pretrained weights.
Only the vision backend is replaced by synthetic compressed HTM features.
"""
from __future__ import annotations

import ast
import importlib
import math
from pathlib import Path
import sys
import types
from typing import Dict, List, Optional, Tuple

import torch
from transformers import Qwen2Config

from test_easa import ROOT, load_modules


def load_decoder(easa):
    utils = types.ModuleType("qvic.utils")
    utils.rank0_print = utils.rank_print = lambda *args: None
    sys.modules["qvic.utils"] = utils
    qwen = importlib.import_module("qvic.model.language_model.modeling_qwen2")
    # Load actual packing/newline methods without importing vision backends.
    namespace = dict(torch=torch, nn=torch.nn, math=math, random=__import__("random"),
                     Optional=Optional, List=List, Tuple=Tuple, Dict=Dict,
                     IGNORE_INDEX=-100, IMAGE_TOKEN_INDEX=-200, rank0_print=lambda *a: None,
                     rank_print=lambda *a: None, add_assignment_newlines=easa.add_assignment_newlines,
                     align_decoder_assignments=easa.align_decoder_assignments)
    methods = {}
    wanted = {"prepare_inputs_labels_for_qvic", "prepare_inputs_for_decoder", "add_token_per_grid", "add_token_per_frame"}
    for path in (ROOT / "qvic/model/qvic_meta_mixin.py", ROOT / "qvic/model/llava_arch.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
                methods[node.name] = namespace[node.name]
    arch = types.ModuleType("qvic.model.llava_arch")
    arch.LlavaMetaForCausalLM = type("SyntheticVisionMeta", (), methods)
    arch.LlavaMetaModel = type("SyntheticVisionModelMeta", (), {})
    sys.modules[arch.__name__] = arch
    llava = importlib.import_module("qvic.model.language_model.llava_qwen")
    return qwen, llava


def config(backend):
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      pad_token_id=0, bos_token_id=1, eos_token_id=63)
    cfg._attn_implementation = backend
    cfg.use_easa, cfg.easa_lambda, cfg.easa_temperature, cfg.easa_epsilon = True, 0.7, 0.1, 1e-6
    cfg.easa_owner_bias = 2.0
    return cfg


def test_qwen(qwen):
    sdpa = qwen.Qwen2ForCausalLM(config("sdpa"))
    eager = qwen.Qwen2ForCausalLM(config("eager"))
    eager.load_state_dict(sdpa.state_dict())
    sdpa.eval(); eager.eval()
    x = torch.randn(1, 8, 32)
    P = torch.zeros(1, 8, 2)
    P[:, 1:3, 0], P[:, 3:6, 1] = 1, 1
    guided = sdpa(inputs_embeds=x, easa_assignments=P, use_cache=False).logits
    baseline = sdpa(inputs_embeds=x, use_cache=False).logits
    assert not torch.allclose(guided, baseline)
    assert torch.allclose(guided, eager(inputs_embeds=x, easa_assignments=P, use_cache=False).logits, atol=1e-6)
    sdpa.config.easa_lambda = 0
    assert torch.equal(baseline, sdpa(inputs_embeds=x, easa_assignments=P, use_cache=False).logits)
    sdpa.config.easa_lambda = .7
    prefix = sdpa(inputs_embeds=x, easa_assignments=P, use_cache=True)
    answer = torch.randn(1, 1, 32)
    cached = sdpa(inputs_embeds=answer, easa_assignments=P, past_key_values=prefix.past_key_values, use_cache=True).logits
    full = sdpa(inputs_embeds=torch.cat((x, answer), dim=1), easa_assignments=P, use_cache=False).logits
    assert torch.allclose(cached[:, -1], full[:, -1], atol=1e-6)
    # Real non-reentrant/reentrant checkpoint replay shares an immutable mask.
    for backend in (sdpa, eager):
        for reentrant in (False, True):
            backend.train()
            backend.requires_grad_(False)
            backend.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": reentrant})
            features = x.clone().requires_grad_(True)
            loss = backend(inputs_embeds=features, easa_assignments=P, labels=torch.arange(8).unsqueeze(0), use_cache=False).loss
            loss.backward()
            assert torch.isfinite(features.grad).all() and features.grad.norm() > 0
            backend.gradient_checkpointing_disable()
            backend.eval()


def test_qvic_packing(llava, htm):
    cfg = config("sdpa")
    cfg.tokenizer_model_max_length, cfg.tokenizer_padding_side = 64, "right"
    cfg.mm_newline_position, cfg.mm_patch_merge_type, cfg.add_faster_video = "grid", "spatial_unpad", False
    model = llava.LlavaQwenForCausalLM(cfg)
    model.model.image_newline = torch.nn.Parameter(torch.zeros(32), requires_grad=False)
    model.use_htm, model.htm_stage_a, model.has_memory_constructed = True, False, False
    model.train_qvic_freeze_encoder, model.verbose = False, False
    model.htm_pred_loss, model._easa_assignments = None, None
    model.htm_predictor = htm.HTMPredictor(32, 8)
    ctrl = htm.HTMMemoryController(16, recent_window_W=1, event_representation="tokens", group_tokens=True)
    upstream = torch.nn.Linear(6, 32)

    def encode(images, image_sizes, questions, **kwargs):
        ctrl.reset(len(images), reset_thresholds=True)
        for t in range(3):
            frame = upstream(torch.stack([image[t] for image in images])).unsqueeze(1)
            ctrl.write_batch(frame, model.htm_predictor, True, retain_memory_grad=model.training)
        model.memory = ctrl.materialize_context_memory(frame)
        model.memory_event_assignments = ctrl.materialize_event_assignments(frame)

    model.embed_video_streaming = encode
    model.prepare_contexts_for_streaming = lambda: model.memory

    def clear(*args, **kwargs):
        model.memory_event_assignments = None
        ctrl.states = []

    model.clear_memory = clear
    images = [torch.randn(3, 4, 6)]
    ids = torch.tensor([[1, -200, 2, 3, 4]])
    labels = torch.tensor([[-100, -100, -100, 3, 4]])
    model.train()
    model.requires_grad_(False)
    output = model(input_ids=ids, input_ids_q=ids, images=images, modalities=["video"], labels=labels, use_cache=False)
    output.loss.backward()
    assert upstream.weight.grad.isfinite().all() and upstream.weight.grad.norm() > 0
    assignments = model._easa_assignments
    assert not assignments.requires_grad
    # Text and row newline slots are neutral, all visual tokens are normalized.
    assert (assignments[:, 0] == 0).all() and (assignments[:, -3:] == 0).all()
    visual = assignments.sum(-1) > 0
    assert torch.allclose(assignments.sum(-1)[visual], torch.ones(int(visual.sum())))
    model.eval()
    # Disabled EASA and lambda=0 traverse actual packing with identical logits.
    cfg.use_easa = False
    disabled = model(input_ids=ids, input_ids_q=ids, images=images, modalities=["video"], use_cache=False).logits
    cfg.use_easa, cfg.easa_lambda = True, 0.0
    zero = model(input_ids=ids, input_ids_q=ids, images=images, modalities=["video"], use_cache=False).logits
    assert torch.equal(disabled, zero)
    cfg.easa_lambda = .7
    # Exercise actual Llava.generate, generation metadata propagation and beams.
    for beams in (1, 2):
        answer = model.generate(ids, images=images, modalities=["video"], input_ids_q=ids,
                                do_sample=False, num_beams=beams, max_new_tokens=3)
        assert answer.shape[0] == 1 and answer.shape[1] > 0
        assert ctrl.states == []  # metadata was captured before inference clears memory
    # JSON config round-trip restores all knobs without adding checkpoint weights.
    restored = Qwen2Config.from_dict(__import__("json").loads(cfg.to_json_string()))
    assert restored.use_easa and restored.easa_lambda == cfg.easa_lambda
    assert restored.easa_temperature == cfg.easa_temperature and restored.easa_epsilon == cfg.easa_epsilon
    assert restored.easa_owner_bias == cfg.easa_owner_bias


def main():
    torch.manual_seed(7)
    easa, htm = load_modules()
    qwen, llava = load_decoder(easa)
    test_qwen(qwen)
    test_qvic_packing(llava, htm)
    print("PASS real tiny Qwen2 eager/SDPA, lambda=0 parity, cached/full parity and checkpointed QA gradients")
    print("PASS actual QViC packing, HTM QA gradients, inference cleanup, greedy/beam generation and config round-trip")


if __name__ == "__main__":
    main()
