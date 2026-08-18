from __future__ import annotations

import logging
import os
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import transformers
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments
from transformers.integrations import is_deepspeed_zero3_enabled

from qvic.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM
from qvic.model.language_model.modeling_qwen2 import Qwen2ForCausalLM
from qvic.train.data import DataCollatorForQViC, DataConfig, QViCSupervisedDataset
from qvic.train.qvic_trainer import PeakMemoryCallback, QViCTrainer, save_qvic_checkpoint

logger = logging.getLogger(__name__)

DEFAULT_LORA_TARGETS = (
    r"encoder\.model\.layers\.\d+\."
    r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
)


@dataclass
class ModelArguments:
    model_base: str = field(
        default="lmms-lab/LLaVA-Video-7B-Qwen2",
        metadata={"help": "Base LMM. Both the decoder and the visual compressor start from it."},
    )

    # --- QViC architecture (mirrors Table 1 / Table 2 of the supplementary) ---
    context_embed_tokens: int = field(default=16, metadata={"help": "Context tokens C per frame."})
    context_memory_length: int = field(default=256, metadata={"help": "Context memory capacity L."})
    max_frame_num_encoder: int = field(default=64, metadata={"help": "Compressor input budget K_v = K + K_r."})
    context_condition_frame_num: int = field(
        default=32, metadata={"help": "Recalled frames K_r at *inference*; unused while training (K_r = 0)."})
    question_guided_selective_attention: bool = field(default=True)
    guiding_context2vision: bool = field(default=True)
    ctx_attn_mask_type: str = field(
        default="framewise",
        metadata={"help": "framewise | ctx_causal | vanilla | single_frame | framewise_wo_block_c2t"},
    )
    fill_context_memory: bool = field(
        default=True, metadata={"help": "Randomise the effective memory length in [K, L] during training."})

    # --- attention kernels ---
    attn_implementation: str = field(default="sdpa", metadata={"help": "Decoder attention: sdpa | eager | flash_attention_2"})
    encoder_attn_implementation: str = field(
        default="sdpa",
        metadata={"help": "Compressor attention. QMSA injects an additive bias, so this must be sdpa or eager."},
    )

    # --- LoRA (Table 1) ---
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default="none")
    lora_target_modules: str = field(
        default=DEFAULT_LORA_TARGETS,
        metadata={"help": "Regex (re.fullmatch) over module names. Defaults to the compressor only."},
    )

    # --- trainable-module switches ---
    tune_context_embed: bool = field(default=True)
    tune_mm_projector: bool = field(default=False, metadata={"help": "Paper keeps the visual encoder frozen."})
    context_embed_init_std: float = field(
        default=-1.0,
        metadata={"help": "std for the context seed embedding; <0 means 1/sqrt(hidden_size)."},
    )

    quiet_qvic: bool = field(
        default=True,
        metadata={"help": "Silence the per-step rank0_print chatter inside the released QViC code."},
    )


@dataclass
class DataArguments:
    dataset_root: str = field(
        default="", metadata={"help": "Path to the LLaVA-Video-83K release directory."})
    annotation_path: Optional[str] = field(default=None)
    num_frames: int = field(default=64, metadata={"help": "Clip frames K."})
    conv_template: str = field(default="qwen_1_5")
    qa_mode: str = field(
        default="single",
        metadata={"help": "single -> one QA turn per record (83k samples/epoch); all -> every turn (280k)."},
    )
    qa_pick: str = field(default="random", metadata={"help": "random | first (only for qa_mode=single)"})
    qa_seed: int = field(default=42)
    media_backend: str = field(default="shards", metadata={"help": "shards | loose"})
    video_folder: Optional[str] = field(default=None)
    max_records: int = field(default=-1, metadata={"help": "Debug: truncate the annotation list."})
    skip_list_path: Optional[str] = field(
        default=None,
        metadata={"help": "JSON list of media keys to exclude. Defaults to <output_dir>/bad_media.json."},
    )
    inflight_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where workers drop a breadcrumb naming the media being decoded. "
                          "Defaults to <output_dir>/inflight."},
    )


@dataclass
class QViCTrainingArguments(TrainingArguments):
    model_max_length: int = field(
        default=32768,
        metadata={"help": "Token budget after the context embeddings replace <image>."},
    )
    cache_dir: Optional[str] = field(default=None)


def rank0(msg: str, *args) -> None:
    if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) == 0:
        logger.info(msg, *args)


def install_zero3_param_guard(model) -> None:
    import deepspeed
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    # `LlavaQwenForCausalLM.forward` calls the method on *itself*, so the patch
    # has to land on the unwrapped model, not on the PeftModel around it.
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    inner = base.get_model()
    bare_params = [p for p in (getattr(inner, "image_newline", None),
                               getattr(inner, "faster_token", None)) if p is not None]
    if not bare_params:
        return

    original = base.prepare_inputs_labels_for_qvic

    def guarded(*args, **kwargs):
        pending = [p for p in bare_params
                   if getattr(p, "ds_status", None) == ZeroParamStatus.NOT_AVAILABLE]
        if not pending:
            return original(*args, **kwargs)
        with deepspeed.zero.GatheredParameters(pending, modifier_rank=None):
            return original(*args, **kwargs)

    base.prepare_inputs_labels_for_qvic = guarded
    rank0("ZeRO-3: guarding %d bare parameter(s) around prepare_inputs_labels_for_qvic "
          "(gathered only while partitioned)", len(bare_params))


def install_qmsa_padding_fix(model) -> None:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    original = base.make_attention_mask_qmsa

    def patched(inputs_embeds_, token_ranges):
        mask = original(inputs_embeds_, token_ranges)
        for b, ranges in enumerate(token_ranges):
            p0, p1 = ranges["range_query_padding"]
            if p1 > p0:
                c0, c1 = ranges["range_context"]
                mask[b, :, c0:c1, p0:p1] = 0
        return mask

    base.make_attention_mask_qmsa = patched
    rank0("QMSA: masking context-to-padding attention (matters only for batch > 1)")


def silence_qvic_logging() -> None:
    """The released code rank0_print()s 3-4 lines per training step."""
    import qvic.model.qvic_meta_mixin as mixin
    import qvic.model.llava_arch as arch

    def _noop(*_args, **_kwargs):
        return None

    mixin.rank0_print = _noop
    arch.rank0_print = _noop


def build_model(model_args: ModelArguments, training_args: QViCTrainingArguments):
    compute_dtype = (torch.bfloat16 if training_args.bf16 else
                     (torch.float16 if training_args.fp16 else torch.float32))
    zero3 = is_deepspeed_zero3_enabled()

    config = LlavaQwenConfig.from_pretrained(model_args.model_base, cache_dir=training_args.cache_dir)
    # Same vocab fix the inference builder applies (LLaVA-NeXT issue #329).
    config.vocab_size = 152064
    config.tokenizer_model_max_length = training_args.model_max_length
    config.use_cache = False
    # QViC fields -- these get written to config.json and read back by
    # `initialize_context_embed` / `embed_video_streaming` at load time.
    config.context_embed_tokens = model_args.context_embed_tokens
    config.context_memory_length = model_args.context_memory_length
    config.question_guided_selective_attention = model_args.question_guided_selective_attention
    config.guiding_context2vision = model_args.guiding_context2vision
    config.ctx_attn_mask_type = model_args.ctx_attn_mask_type

    from_pretrained_kwargs = dict(
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=compute_dtype,
        attn_implementation=model_args.attn_implementation,
    )
    if not zero3:
        from_pretrained_kwargs["low_cpu_mem_usage"] = True

    rank0("loading decoder + visual encoder from %s", model_args.model_base)
    model = LlavaQwenForCausalLM.from_pretrained(model_args.model_base, **from_pretrained_kwargs)

    rank0("loading visual compressor (encoder LLM) from %s", model_args.model_base)
    encoder_kwargs = dict(
        cache_dir=training_args.cache_dir,
        torch_dtype=compute_dtype,
        attn_implementation=model_args.encoder_attn_implementation,
    )
    if not zero3:
        encoder_kwargs["low_cpu_mem_usage"] = True
    model.encoder = Qwen2ForCausalLM.from_pretrained(model_args.model_base, **encoder_kwargs)

    if (model_args.question_guided_selective_attention
            and model_args.encoder_attn_implementation not in ("sdpa", "eager")):
        raise ValueError(
            "QMSA needs an attention kernel that accepts an additive bias: "
            f"use --encoder_attn_implementation sdpa (got {model_args.encoder_attn_implementation})."
        )

    # Context seed embedding (paper eq.2). `initialize_context_embed` also
    # stamps the QMSA switches onto the config.
    model.initialize_context_embed(model_args)
    std = (model_args.context_embed_init_std if model_args.context_embed_init_std > 0
           else 1.0 / (config.hidden_size ** 0.5))
    with torch.no_grad():
        model.context_embed.weight.normal_(mean=0.0, std=std)
    model.context_embed.to(dtype=compute_dtype)
    rank0("context seed embedding: %s (init std %.5f)", tuple(model.context_embed.weight.shape), std)

    # Runtime knobs used by `embed_video_streaming`.
    model.set_context_memory_length(model_args.context_memory_length)
    model.max_frame_num_encoder = model_args.max_frame_num_encoder
    model.context_condition_frame_num = model_args.context_condition_frame_num
    # Training is one-way (K_r = 0): relevance-based feedback is inference-only
    # and the released code asserts `not self.training` inside that branch.
    model.compress_with_relevance = False
    model.fill_context_memory = model_args.fill_context_memory
    model.train_qvic_freeze_encoder = False
    model.verbose = False

    return model, compute_dtype


def apply_lora(model, model_args: ModelArguments):
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias=model_args.lora_bias,
        target_modules=model_args.lora_target_modules,
        task_type="CAUSAL_LM",
    )
    n_matched = sum(1 for n, _ in model.named_modules()
                    if re.fullmatch(model_args.lora_target_modules, n))
    if n_matched == 0:
        raise ValueError(
            f"LoRA target regex matched no module: {model_args.lora_target_modules!r}. "
            "Check the pattern against `model.named_modules()`."
        )
    rank0("LoRA r=%d alpha=%d dropout=%.2f -> %d target modules",
          model_args.lora_r, model_args.lora_alpha, model_args.lora_dropout, n_matched)

    model = get_peft_model(model, lora_config)

    # get_peft_model froze everything that is not `lora_*`; put the context seed
    # embedding back on the trainable list.
    if model_args.tune_context_embed:
        for name, param in model.named_parameters():
            if "context_embed" in name:
                param.requires_grad = True
    if model_args.tune_mm_projector:
        for name, param in model.named_parameters():
            if "mm_projector" in name:
                param.requires_grad = True
    return model


def summarise_trainables(model) -> None:
    # Under ZeRO-3 the frozen weights are already partitioned by `zero.Init` at
    # this point, so `p.numel()` reports 0 for them and the ratio would read
    # "100% trainable". `ds_numel` carries the real size.
    def _numel(p):
        return getattr(p, "ds_numel", None) or p.numel()

    total = sum(_numel(p) for p in model.parameters())
    trainable = [(n, _numel(p)) for n, p in model.named_parameters() if p.requires_grad]
    n_train = sum(x[1] for x in trainable)
    rank0("trainable: %d / %d params (%.4f%%)", n_train, total, 100.0 * n_train / max(total, 1))
    groups = {}
    for name, numel in trainable:
        key = "lora" if "lora_" in name else ("context_embed" if "context_embed" in name else "other")
        groups[key] = groups.get(key, 0) + numel
    for key, numel in sorted(groups.items()):
        rank0("  %-14s %d", key, numel)


def train() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, QViCTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if training_args.local_rank in (-1, 0) else logging.WARNING,
        stream=sys.stdout,
    )
    transformers.utils.logging.set_verbosity_warning()

    if not data_args.dataset_root and not data_args.annotation_path:
        raise ValueError("--dataset_root (or --annotation_path) is required")

    training_args.remove_unused_columns = False
    if training_args.gradient_checkpointing and training_args.gradient_checkpointing_kwargs is None:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    transformers.set_seed(training_args.seed)
    _ = training_args.device

    model, compute_dtype = build_model(model_args, training_args)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_base, cache_dir=training_args.cache_dir, use_fast=False,
        model_max_length=training_args.model_max_length, padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 151643  # Qwen2 <|endoftext|>, same value the evaluator uses
    model.pad_token_id = tokenizer.pad_token_id
    model.config.tokenizer_padding_side = "right"

    image_processor = model.get_vision_tower().image_processor

    # Freeze the world, then re-open exactly the two modules the paper trains.
    model.requires_grad_(False)
    model = apply_lora(model, model_args)
    summarise_trainables(model)

    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        # Reaches both LLMs: `_set_gradient_checkpointing` walks `self.modules()`,
        # and `model.encoder` is a registered submodule.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs)
        # Reentrant checkpointing silently produces no gradient if none of a
        # segment's tensor inputs require grad. Here they do (the trainable
        # context embedding is concatenated into both LLMs' inputs_embeds), but
        # the hook on the shared `embed_tokens` makes that independent of how the
        # QViC input assembly happens to be wired.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if is_deepspeed_zero3_enabled():
        install_zero3_param_guard(model)
    if model_args.question_guided_selective_attention:
        install_qmsa_padding_fix(model)
    if model_args.quiet_qvic:
        silence_qvic_logging()

    skip_list_path = data_args.skip_list_path or os.path.join(
        training_args.output_dir, "bad_media.json")
    inflight_dir = data_args.inflight_dir or os.path.join(
        training_args.output_dir, "inflight")
    os.makedirs(inflight_dir, exist_ok=True)

    data_cfg = DataConfig(
        dataset_root=data_args.dataset_root,
        annotation_path=data_args.annotation_path,
        num_frames=data_args.num_frames,
        conv_template=data_args.conv_template,
        qa_mode=data_args.qa_mode,
        qa_pick=data_args.qa_pick,
        qa_seed=data_args.qa_seed,
        media_backend=data_args.media_backend,
        video_folder=data_args.video_folder,
        max_records=data_args.max_records,
        model_max_length=training_args.model_max_length,
        skip_list_path=skip_list_path,
        inflight_dir=inflight_dir,
    )
    train_dataset = QViCSupervisedDataset(data_cfg, tokenizer, image_processor)
    collator = DataCollatorForQViC(
        tokenizer=tokenizer, pad_token_id=tokenizer.pad_token_id, frame_dtype=compute_dtype)

    trainer = QViCTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        lora_bias=model_args.lora_bias,
        callbacks=[PeakMemoryCallback()],
    )

    existing = sorted(pathlib.Path(training_args.output_dir).glob("checkpoint-*"),
                      key=lambda p: int(p.name.split("-")[-1]))
    resume = bool(existing)
    if resume:
        rank0("resuming from %s (%d checkpoint(s) present)", existing[-1], len(existing))
    else:
        rank0("no checkpoint in %s -- starting from scratch", training_args.output_dir)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_state()

    if torch.cuda.is_available():
        rank0("peak CUDA memory over the run: %.1f GiB allocated / %.1f GiB reserved",
              torch.cuda.max_memory_allocated() / 2 ** 30,
              torch.cuda.max_memory_reserved() / 2 ** 30)

    model.config.use_cache = True
    save_qvic_checkpoint(
        model, training_args.output_dir, model_args.lora_bias,
        is_main_process=training_args.should_save,
    )
    rank0("done. checkpoint written to %s", training_args.output_dir)


if __name__ == "__main__":
    train()
