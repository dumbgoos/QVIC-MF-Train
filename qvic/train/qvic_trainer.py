from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import torch
from transformers import Trainer, TrainerCallback

logger = logging.getLogger(__name__)


class PeakMemoryCallback(TrainerCallback):

    def __init__(self, report_at=(1, 2, 5, 20, 100)):
        self.report_at = set(report_at)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.report_at and torch.cuda.is_available():
            logger.info(
                "step %d: peak CUDA memory %.1f GiB allocated / %.1f GiB reserved",
                state.global_step,
                torch.cuda.max_memory_allocated() / 2 ** 30,
                torch.cuda.max_memory_reserved() / 2 ** 30,
            )



def maybe_zero_3(param, ignore_status: bool = False, name: Optional[str] = None):
    """Materialise one parameter, gathering it first if ZeRO-3 has sharded it."""
    if hasattr(param, "ds_id"):
        from deepspeed import zero
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE and not ignore_status:
            logger.warning("%s: param.ds_status != ACCESSIBLE", name)
        with zero.GatheredParameters([param]):
            return param.data.detach().cpu().clone()
    return param.detach().cpu().clone()


def get_peft_state_maybe_zero_3(named_params, bias: str = "none") -> Dict[str, torch.Tensor]:
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return, maybe_lora_bias, lora_bias_names = {}, {}, set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                lora_bias_names.add(k.split("lora_")[0] + "bias")
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias.items():
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError(f"unsupported lora bias mode: {bias}")
    return {k: maybe_zero_3(v, ignore_status=True, name=k) for k, v in to_return.items()}


def get_non_lora_trainables_maybe_zero_3(named_params) -> Dict[str, torch.Tensor]:
    to_return = {k: t for k, t in named_params if "lora_" not in k and t.requires_grad}
    return {k: maybe_zero_3(v, ignore_status=True, name=k) for k, v in to_return.items()}



def save_qvic_checkpoint(model, output_dir: str, lora_bias: str = "none",
                         is_main_process: bool = True) -> None:

    named = list(model.named_parameters())
    lora_state = get_peft_state_maybe_zero_3(named, lora_bias)
    non_lora_state = get_non_lora_trainables_maybe_zero_3(named)

    if not is_main_process:
        return

    os.makedirs(output_dir, exist_ok=True)

    model.config.save_pretrained(output_dir)

    model.save_pretrained(
        output_dir,
        state_dict=lora_state,
        safe_serialization=False,
        save_embedding_layers=False,
    )

    torch.save(non_lora_state, os.path.join(output_dir, "non_lora_trainables.bin"))
    logger.info(
        "saved QViC checkpoint to %s (%d LoRA tensors, %d non-LoRA trainables: %s)",
        output_dir, len(lora_state), len(non_lora_state), list(non_lora_state.keys()),
    )


class QViCTrainer(Trainer):

    def __init__(self, *args, lora_bias: str = "none", **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_bias = lora_bias

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False) -> None:
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        save_qvic_checkpoint(
            self.model, output_dir, self.lora_bias,
            is_main_process=self.args.should_save,
        )
        if self.args.should_save:
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
