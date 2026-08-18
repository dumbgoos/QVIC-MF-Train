# Copyright (c) 2026 Fujitsu Limited and the QViC-MF authors.
#
# This file is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).
# https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# This module contains the core QViC (Question-guided Video-in-Context) method
# implementation, isolated from the LLaVA-NeXT base class so that the upstream
# Apache-2.0 code in `qvic/model/llava_arch.py` stays close to its original form.
#
# Paper references (eq.X) below correspond to the QViC paper.

"""QViC mixin for `LlavaMetaForCausalLM`.

Provides:
  - Question-guided context encoding (paper eq.1, eq.3, eq.4)
  - Multi-frame Question-guided Mask Selective Attention (QMSA) (paper eq.6-10)
  - Streaming context memory bank with relevance-based compression
  - Debug visualization for attention maps and memory frames

This mixin must come BEFORE `ABC` in the MRO of `LlavaMetaForCausalLM` so that
its `__init__` cooperates with downstream `nn.Module`-based model `__init__`s.
"""

from typing import List, Optional, Tuple, Dict
from abc import abstractmethod

import os
import math
import time
import random

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Lock

import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter

from qvic.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from qvic.utils import rank0_print, rank_print


class QViCMetaMixin:
    """Mixin providing QViC's question-guided context encoding and memory bank.

    Designed to be mixed into `LlavaMetaForCausalLM` (LLaVA-NeXT) so that the
    QViC-specific extensions live in a separate, clearly-licensed module while
    sharing the same instance state via Python's cooperative multiple
    inheritance.
    """

    def __init__(self, config):
        super().__init__(config)
        self._init_qvic_state()

    def _init_qvic_state(self):
        """Initialize all QViC-specific instance attributes.

        Called from ``__init__`` so that subclasses do not need to repeat this
        boilerplate. Mirrors the legacy in-line initialization that used to live
        in ``LlavaMetaForCausalLM.__init__``.
        """
        self.context_embed = None

        # support video streaming mode
        self.context_memory = None  # set to torch.multiprocessing.Manager.list() when launching
        self.compression_size = None
        self.reflection_frequency = None
        self.relevance_memory = None
        self.frame_indices_memory = None

        self.top_k_head = 5
        self.relevance_layer_range = [16, 20]
        self.max_frame_num_encoder = 64  # 64
        self.context_condition_frame_num = 32  # < self.max_frame_num_encoder
        rank0_print(f"top_k_head for relevance: {self.top_k_head}")
        rank0_print(f"relevance_layer_range: {self.relevance_layer_range}")

        self.context_memory_length = None
        self.memory_for_each_batch = None
        self.context_memory_lock = Lock()
        self.has_memory_constructed = False
        self.compress_with_relevance = True
        self.fill_context_memory = False  # fill context memory by interpolating context embeddings (nearest neighbor)

        self.question_guided_selective_attention = False
        self.guiding_context2vision = True  # Only effective when self.question_guided_selective_attention is True
        self.ctx_attn_mask_type = "framewise"  # "framewise", "causal"
        self.train_qvic_freeze_encoder = False

        self.debug_drawing = False  # visualize attention weights / embeddings
        if self.debug_drawing:
            rank0_print(f"[DEBUG] debug_drawing: {self.debug_drawing}")
        self.debug_drawing_memory = False
        self.debug_input_image_info = None  # [DEBUG] path to save input images

        self.pad_token_id = 151643  # Qwen2; updated in initialize_vision_tokenizer()
        self.verbose = False

    @abstractmethod
    def get_encoder(self):
        pass

    # ------------------------------------------------------------------
    # Configuration / setters
    # ------------------------------------------------------------------

    def initialize_context_embed(self, model_args):
        self.get_model().config.context_embed_tokens = getattr(model_args, "context_embed_tokens", 64)
        self.context_embed_tokens = self.get_model().config.context_embed_tokens
        rank0_print("context_embed_tokens: ", self.context_embed_tokens)
        self.context_embed = nn.Embedding(
            self.context_embed_tokens, self.get_model().config.hidden_size, padding_idx=None
        )

        self.get_model().config.question_guided_selective_attention = getattr(model_args, "question_guided_selective_attention", False)
        self.question_guided_selective_attention = self.get_model().config.question_guided_selective_attention
        rank0_print("question_guided_selective_attention: ", self.question_guided_selective_attention)

        self.get_model().config.guiding_context2vision = getattr(model_args, "guiding_context2vision", True)
        self.guiding_context2vision = self.get_model().config.guiding_context2vision
        rank0_print("guiding_context2vision: ", self.guiding_context2vision)

        self.get_model().config.ctx_attn_mask_type = getattr(model_args, "ctx_attn_mask_type", "framewise")
        self.ctx_attn_mask_type = self.get_model().config.ctx_attn_mask_type
        rank0_print("ctx_attn_mask_type: ", self.ctx_attn_mask_type)

    def set_context_memory_length(self, context_memory_length):
        self.context_memory_length = context_memory_length
        rank0_print(f"context_memory_length: {self.context_memory_length} (updated)")

    def set_norm_loss_weight(self, norm_loss_weight):
        self.norm_loss_weight = norm_loss_weight
        rank0_print(f"norm-loss weight: {self.norm_loss_weight} (updated)")

    def set_question_guided_selective_attention(self, question_guided_selective_attention):
        self.question_guided_selective_attention = question_guided_selective_attention
        rank0_print(f"question_guided_selective_attention: {self.question_guided_selective_attention} (updated)")

    def set_guiding_context2vision(self, guiding_context2vision):
        self.guiding_context2vision = guiding_context2vision
        rank0_print(f"guiding_context2vision: {self.guiding_context2vision} (updated)")

    def set_debug_drawing(self, debug_drawing):
        self.debug_drawing = debug_drawing
        rank0_print(f"[DEBUG] debug_drawing: {self.debug_drawing} (updated)")

    def set_debug_drawing_memory(self, debug_drawing_memory):
        self.debug_drawing_memory = debug_drawing_memory
        rank0_print(f"[DEBUG] debug_drawing_memory: {self.debug_drawing_memory} (updated)")

    def set_debug_input_image_info(self, debug_input_image_info):
        self.debug_input_image_info = debug_input_image_info
        rank0_print(f"[DEBUG] debug_input_image_info: {self.debug_input_image_info} (updated)")

    def set_fill_context_memory(self, fill_context_memory):
        self.fill_context_memory = fill_context_memory
        rank0_print(f"fill_context_memory: {self.fill_context_memory} (updated)")

    def set_compress_with_relevance(self, compress_with_relevance):
        self.compress_with_relevance = compress_with_relevance
        rank0_print(f"compress_with_relevance: {self.compress_with_relevance} (updated)")

    def set_verbose(self, verbose):
        self.verbose = verbose
        rank0_print(f"verbose: {self.verbose} (updated)")

    def set_ctx_attn_mask_type(self, ctx_attn_mask_type):
        self.ctx_attn_mask_type = ctx_attn_mask_type
        rank0_print(f"ctx_attn_mask_type: {self.ctx_attn_mask_type} (updated)")

    def set_context_condition_frame_num(self, context_condition_frame_num):
        self.context_condition_frame_num = context_condition_frame_num
        rank0_print(f"context_condition_frame_num: {self.context_condition_frame_num} (updated)")

    # ------------------------------------------------------------------
    # Core QMSA / context encoding
    # ------------------------------------------------------------------

    def make_attention_mask_qmsa(self, inputs_embeds_, token_ranges):
        B, N, D = inputs_embeds_.shape

        if self.verbose or self.training:
            rank0_print(f"[INFO] ctx_attn_mask_type: {self.ctx_attn_mask_type}")

        # [MultiFrame-QMSA]
        # paper: eq.(6)
        attention_mask = torch.ones([B, 1, N, N], dtype=torch.bool, device=inputs_embeds_.device)
        attention_mask = torch.tril(attention_mask)  # causal mask
        for b in range(B):
            # paper: eq.(7)
            range_query_system = token_ranges[b]["range_query_system"]  # [2]
            range_image = token_ranges[b]["range_image"]  # [2]
            range_query_user = token_ranges[b]["range_query_user"]  # [2]
            range_context = token_ranges[b]["range_context"]  # [2]
            num_frames = token_ranges[b]["num_frames"]
            num_contexts = token_ranges[b]["num_contexts"]
            i0, i1 = range_image
            c0, c1 = range_context
            q0, q1 = range_query_user
            s0, s1 = range_query_system

            # paper: eq.(8)
            attention_mask[b, :, c0:c1, s0:s1] = 0  # mask context to system query
            attention_mask[b, :, c0:c1, q0:q1] = 0  # mask context to user query

            attention_mask[b, :, c0:c1, i0:i1] = 0  # mask context to all image initially

            Ni = (i1 - i0) // num_frames  # num image tokens per frame
            Nc = (c1 - c0) // num_contexts  # num context tokens per frame
            assert (i1 - i0) % num_frames == 0
            assert (c1 - c0) % num_contexts == 0
            assert num_frames == num_contexts

            if self.ctx_attn_mask_type == "framewise":  # default
                # paper: eq.(6)
                for f in range(num_frames):
                    i0_tar, i1_tar = (i0 + f * Ni, i0 + (f + 1) * Ni)
                    c0_tar, c1_tar = (c0 + f * Nc, c0 + (f + 1) * Nc)
                    attention_mask[b, :, c0_tar:c1_tar, i0_tar:i1_tar] = 1  # open context to corresponding image
                    attention_mask[b, :, c0_tar:c1_tar, c0:c0_tar] = 0  # mask context to previous context
            elif self.ctx_attn_mask_type == "ctx_causal":
                for f in range(num_frames):
                    i0_tar, i1_tar = (i0 + f * Ni, i0 + (f + 1) * Ni)
                    c0_tar, c1_tar = (c0 + f * Nc, c0 + (f + 1) * Nc)
                    attention_mask[b, :, c0_tar:c1_tar, i0:i1_tar] = 1  # open context to all previous image (causal)
            elif self.ctx_attn_mask_type == "vanilla" or self.ctx_attn_mask_type == "single_frame":
                attention_mask[b, :, c0:c1, s0:s1] = 1  # open context to system query
                attention_mask[b, :, c0:c1, q0:q1] = 1  # open context to user query
                attention_mask[b, :, c0:c1, i0:i1] = 1  # open context to all image
            elif self.ctx_attn_mask_type == "framewise_wo_block_c2t":  # ablation (no ctx2txt block)
                attention_mask[b, :, c0:c1, s0:s1] = 1  # open context to system query
                attention_mask[b, :, c0:c1, q0:q1] = 1  # open context to user query
                for f in range(num_frames):
                    i0_tar, i1_tar = (i0 + f * Ni, i0 + (f + 1) * Ni)
                    c0_tar, c1_tar = (c0 + f * Nc, c0 + (f + 1) * Nc)
                    attention_mask[b, :, c0_tar:c1_tar, i0_tar:i1_tar] = 1  # open context to corresponding image
                    attention_mask[b, :, c0_tar:c1_tar, c0:c0_tar] = 0  # mask context to previous context
            else:
                raise ValueError(f"Unexpected ctx_attn_mask_type: {self.ctx_attn_mask_type}")

        return attention_mask

    def forward_encoder(self, inputs_embeds, attention_mask, token_ranges=None, output_attentions=False):
        encoder_outputs = self.get_encoder().model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            token_ranges=token_ranges,
        )
        return encoder_outputs

    def get_context_embeds_from_encoder_outputs(self, encoder_outputs, token_ranges=None):
        if token_ranges is None:  # [QMSA]
            context_embeds = encoder_outputs.hidden_states[-1][:, -self.get_model().config.context_embed_tokens:, :]  # [B, ctx, D]

        else:
            # [MultiFrame-QMSA]
            B = len(token_ranges)
            context_embeds_list = []
            for batch in range(B):
                range_context = token_ranges[batch]["range_context"]
                num_contexts = token_ranges[batch]["num_contexts"]
                contexts = encoder_outputs.hidden_states[-1][batch, range_context[0]:range_context[1], :]  # [T*ctx, D]
                if num_contexts > 1:
                    contexts = contexts.view(num_contexts, -1, contexts.shape[-1])  # [T, ctx, D]
                else:
                    contexts = contexts.unsqueeze(0)  # [1, ctx, D]
                context_embeds_list.append(contexts)
            context_embeds = torch.stack(context_embeds_list, dim=0)  # [B, T, ctx, D]

        return context_embeds

    # ------------------------------------------------------------------
    # Main pipelines (training + inference)
    # ------------------------------------------------------------------

    def prepare_inputs_labels_for_qvic(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,  # [B, Nf, Np, C, H, W]
        image_sizes: Optional[List[List[int]]] = None,
        modalities: Optional[List[str]] = ["image"],
        input_ids_q: Optional[torch.LongTensor] = None,  # questions for context encoder
        disable_tqdm: bool = True,
    ):
        # all_video is True when modalities has only video
        # [TODO] mixed modalities
        if type(modalities) is not list:
            modalities = [modalities]
        all_video = all([x == "video" for x in modalities])
        all_image = all([x == "image" for x in modalities])
        assert all_video or all_image, "Mixed modalities are not supported yet. (QViC)"

        with torch.inference_mode((not self.training) or self.train_qvic_freeze_encoder):

            if all_video:
                # video modalities
                # images: B x [T, C, H, W]   # 2 x [64, 3, 384, 384]
                if not self.has_memory_constructed:
                    delete_cache = (not self.training) or self.train_qvic_freeze_encoder
                    self.embed_video_streaming(images, image_sizes, input_ids_q, disable_tqdm=disable_tqdm, delete_tmp_tensor=delete_cache)
                    context_embeds = self.prepare_contexts_for_streaming()  # B x [L, ctx, D]
                    if delete_cache:
                        self.clear_memory()
                else:
                    # assume that the context memory is already constructed in another thread
                    context_embeds = self.prepare_contexts_for_streaming()  # B x [L, ctx, D]
            else:
                # image modalities (debug)
                if type(images) is not list and images.dim() == 6:  # Video format [B, Nf=1, Np, C, H, W]
                    assert images.shape[1] == 1
                    images = images.squeeze(1)  # [B, Np, C, H, W]
                else:
                    images = [x.squeeze(1) if x.ndim == 5 else x for x in images]  # Single image, [B, Nf=1, Np, C, H, W] -> [B, Np, C, H, W]
                (_, _, attention_mask_encoder, _, inputs_embeds_, _, _) = self.prepare_inputs_labels_for_multimodal(
                    input_ids_q, None, None, None, None, images, modalities, image_sizes)
                assert(inputs_embeds_ is not None)

                if self.debug_drawing and (not self.training):
                    # --- [DEBUG]
                    encoder_outputs = self.forward_encoder(inputs_embeds_, attention_mask_encoder, output_attentions=True)
                    self.debug_analyze_attention(input_ids_q, encoder_outputs)
                    # ---
                else:
                    encoder_outputs = self.forward_encoder(inputs_embeds_, attention_mask_encoder)

                context_embeds = self.get_context_embeds_from_encoder_outputs(encoder_outputs)
                context_embeds = context_embeds.unsqueeze(1)  # B x [L=1, ctx, D]

        # insert image_newline token to context_embeds
        if context_embeds is not None:
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            new_context_embeds = []

            for batch_idx, context in enumerate(context_embeds):  # context: torch.Size([L, ctx, D])

                if self.verbose and batch_idx == 0:
                    rank0_print(f"[INFO] add image_newline to context. mm_newline_position: ({mm_newline_position}), mm_patch_merge_type: ({mm_patch_merge_type}), context: {context.shape}, all_video: {all_video}")

                if mm_newline_position == "grid":
                    # Grid-wise
                    context = self.add_token_per_grid(context)
                    new_context_embeds.append(context)
                elif mm_newline_position == "frame":
                    # Frame-wise
                    context = self.add_token_per_frame(context)
                    new_context_embeds.append(context.flatten(0, 1))
                elif mm_newline_position == "one_token":
                    # one-token
                    context = context.flatten(0, 1)
                    if 'unpad' in mm_patch_merge_type:
                        context = torch.cat((context, self.model.image_newline[None].to(context.device)), dim=0)
                    new_context_embeds.append(context)
                elif mm_newline_position == "no_token":
                    new_context_embeds.append(context.flatten(0, 1))
                else:
                    raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")

            context_embeds = new_context_embeds  # B x [L*ctx', D]

        (_, position_ids__, attention_mask__, past_key_values__, inputs_embeds__, labels__) = self.prepare_inputs_for_decoder(
            input_ids, position_ids, attention_mask, past_key_values, labels, context_embeds)

        return None, position_ids__, attention_mask__, past_key_values__, inputs_embeds__, labels__

    def prepare_inputs_for_decoder(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        context_embed,  # B x [L*ctx', D]
    ):
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = context_embed[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = context_embed[cur_image_idx]
                    except IndexError:
                        rank_print(f"[WARNING] IndexError: cur_image_idx={cur_image_idx}, context_embed.shape={len(context_embed)}, num_images={num_images}, cur_input_ids={cur_input_ids}")
                        cur_image_features = context_embed[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            # assert cur_image_idx == batch_idx + 1

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)

        new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
        new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    # ------------------------------------------------------------------
    # Streaming context memory bank
    # ------------------------------------------------------------------

    def prepare_contexts_for_streaming(self):
        # Have some tries to avoid deadlock
        attempt_times = 0
        context_memory = None
        while attempt_times < 300:
            try:
                with self.context_memory_lock:
                    # context_memory = self.memory_for_each_batch # B x [L, N, D] (list) or [B, L, N, D] (tensor)
                    context_memory = self.memory_for_each_batch  # [B, L, N, D] (tensor)
                    # if type(context_memory) is list:
                    #    context_memory = [memory.to(self.dtype).to(self.device) for memory in context_memory] # B x [L, N, D]
                    break
            except Exception as e:
                time.sleep(0.1)
                attempt_times += 1
                print(f'Attempt:{attempt_times} Failed to get context memory, Error: {e}')

        return context_memory

    def get_image_token_indices(self, image_sizes, input_ids_q, image_token_based_indices=False):
        image_token_index = torch.where(input_ids_q == -200)[1][0].item()  # batch=0 # n_query_system
        n_query_system_embeds = image_token_index
        n_image_embeds = self.get_vision_tower().config.num_hidden_layers * self.get_vision_tower().config.num_hidden_layers  # img_dim 576, 729
        range_image = [n_query_system_embeds, n_query_system_embeds + n_image_embeds]

        im_width = image_sizes[0][0]  # [px]
        im_height = image_sizes[0][1]  # [px]
        width = height = int(math.sqrt(n_image_embeds))  # tokens
        # [NOTE] images are padded to square, so width and height are the same.
        patch_size = max(image_sizes[0]) // width  # [px/token]
        padding = int((max(image_sizes[0]) - min(image_sizes[0])) / 2 / patch_size)  # [token]
        image_token_mask = np.ones((width, height), dtype=bool)  # [token, token]
        if im_width > im_height:
            # landscape
            image_token_mask[:padding, :] = False
            image_token_mask[-padding:, :] = False
        else:
            # portrait
            image_token_mask[:, :padding] = False
            image_token_mask[:, -padding:] = False
        image_token_mask = image_token_mask.flatten()  # [token]
        if image_token_based_indices:
            image_token_indices = np.where(image_token_mask)[0]  # [token]
        else:
            image_token_indices = range_image[0] + np.where(image_token_mask)[0]  # [token]

        return image_token_indices

    def merge_target_frame_on_memory(self, from_index, to_index):
        self.context_memory[:, to_index, :] = (
            self.context_memory[:, from_index, :] * self.compression_size[:, from_index, None] +
            self.context_memory[:, to_index, :] * self.compression_size[:, to_index, None]
        ) / (self.compression_size[:, from_index, None] + self.compression_size[:, to_index, None])
        if self.compress_with_relevance:
            self.relevance_memory[:, to_index] = (
                self.relevance_memory[:, from_index] * self.compression_size[:, from_index] +
                self.relevance_memory[:, to_index] * self.compression_size[:, to_index]
            ) / (self.compression_size[:, from_index] + self.compression_size[:, to_index])
            self.frame_indices_memory[:, to_index] = (
                self.frame_indices_memory[:, from_index] * self.compression_size[:, from_index] +
                self.frame_indices_memory[:, to_index] * self.compression_size[:, to_index]
            ) / (self.compression_size[:, from_index] + self.compression_size[:, to_index])
            self.reflection_frequency[:, to_index] = (
                self.reflection_frequency[:, from_index] * self.compression_size[:, from_index] +
                self.reflection_frequency[:, to_index] * self.compression_size[:, to_index]
            ) / (self.compression_size[:, from_index] + self.compression_size[:, to_index])
            self.compression_size[:, to_index] += self.compression_size[:, from_index]

        self.remove_target_frame_on_memory(from_index)

    def remove_target_frame_on_memory(self, index_remove):
        assert index_remove < self.context_memory.shape[1], f"index_remove {index_remove} is out of range for context_memory with shape {self.context_memory.shape}"
        if index_remove == 0:
            # remove the first frame
            self.context_memory = self.context_memory[:, 1:]
            self.compression_size = self.compression_size[:, 1:]
            if self.compress_with_relevance:
                self.relevance_memory = self.relevance_memory[:, 1:]
                self.frame_indices_memory = self.frame_indices_memory[:, 1:]
                self.reflection_frequency = self.reflection_frequency[:, 1:]
        elif index_remove == self.context_memory.shape[1] - 1:
            # remove the last frame
            self.context_memory = self.context_memory[:, :-1]
            self.compression_size = self.compression_size[:, :-1]
            if self.compress_with_relevance:
                self.relevance_memory = self.relevance_memory[:, :-1]
                self.frame_indices_memory = self.frame_indices_memory[:, :-1]
                self.reflection_frequency = self.reflection_frequency[:, :-1]
        else:
            self.context_memory = torch.cat([self.context_memory[:, :index_remove], self.context_memory[:, index_remove + 1:]], dim=1)
            self.compression_size = torch.cat([self.compression_size[:, :index_remove], self.compression_size[:, index_remove + 1:]], dim=1)
            if self.compress_with_relevance:
                self.relevance_memory = torch.cat([self.relevance_memory[:, :index_remove], self.relevance_memory[:, index_remove + 1:]], dim=1)
                self.frame_indices_memory = torch.cat([self.frame_indices_memory[:, :index_remove], self.frame_indices_memory[:, index_remove + 1:]], dim=1)
                self.reflection_frequency = torch.cat([self.reflection_frequency[:, :index_remove], self.reflection_frequency[:, index_remove + 1:]], dim=1)

    def embed_video_streaming(  # Asynchronous encoding with a SemLock, only for videos, batch size > 1
            self,
            images,  # B x [T, C, H, W]
            image_sizes,  # list, # [(1280, 720)] # B or 1
            input_ids_q,  # [B, N]
            reset_memory=True,  # treat the context memory independently for each batch
            to_cpu_memory=False,
            disable_tqdm=True,
            delete_tmp_tensor=True,
    ):
        if self.context_memory_length is None:
            self.context_memory_length = getattr(self.config, "context_memory_length", 64)

        if self.compress_with_relevance:
            # inference
            assert not self.training

            # --- [TODO] support batch size > 1
            B = len(images)
            assert B == 1, f"compress_with_relevance is only supported for batch size 1, but got {B}"
            # ---
            if self.verbose:
                rank0_print(f"[INFO] relevance from attention weights (query2image) in the last layer")
            relevance_memory_long = []  # DEBUG

        with self.context_memory_lock:
            if reset_memory:
                self.clear_memory_for_one_batch(delete_tensor=delete_tmp_tensor)

            if type(images) is list or images.ndim == 5:  # 6:
                T = images[0].size(0)  # num frames
                assert self.max_frame_num_encoder > self.context_condition_frame_num, f"max_frame_num_encoder {self.max_frame_num_encoder} must be greater than context_condition_frame_num {self.context_condition_frame_num}"

                if self.compress_with_relevance and not self.training:
                    num_frame_clip = self.max_frame_num_encoder - self.context_condition_frame_num  # 64-32
                else:
                    num_frame_clip = self.max_frame_num_encoder  # 64

                if T % num_frame_clip == 0:
                    num_clips = T // num_frame_clip
                else:
                    num_clips = (T // num_frame_clip) + 1

                for i_clip in tqdm(range(num_clips), disable=disable_tqdm):
                    start_idx = i_clip * num_frame_clip
                    end_idx = min((i_clip + 1) * num_frame_clip, T)
                    clip_indices = np.arange(start_idx, end_idx)

                    if self.context_memory is None or (num_frame_clip == self.max_frame_num_encoder):
                        images_encode = [images[b][clip_indices] for b in range(len(images))]  # clip
                        recall_memory_indices = None
                        if self.verbose and not self.training:
                            rank0_print(f"[INFO] Using all frames in the clip as context, start_idx: {start_idx}, end_idx: {end_idx}, num_frames: {end_idx - start_idx}")
                    else:
                        # -- Memory-feedback retrieval
                        # paper: eq.(1)
                        assert not self.training
                        assert self.relevance_memory.shape[1] == self.frame_indices_memory.shape[1]
                        top_indices = torch.argsort(self.relevance_memory[0], descending=True).tolist()
                        num_recall_frames = min(self.context_condition_frame_num, len(top_indices))
                        recall_memory_indices = sorted(top_indices[:num_recall_frames])  # keep causal order
                        recall_frame_indices = [int(self.frame_indices_memory[0, idx].item()) for idx in recall_memory_indices]
                        if self.verbose:
                            rank0_print(f"[INFO] Selected frame indices based on relevance: {recall_frame_indices} (original frame indices), {recall_memory_indices} (memory indices)")
                        images_recalled = torch.cat([images[0][idx:idx + 1] for idx in recall_frame_indices], dim=0)  # [T_selected, C, H, W]
                        images_clip = images[0][clip_indices]  # [T_clip, C, H, W]
                        images_encode = [torch.cat([images_recalled, images_clip], dim=0)]  # [B, T_selected + T_clip, C, H, W]
                        # --

                    modalities = ["video"] * len(images_encode)  # [B]

                    if self.ctx_attn_mask_type == "single_frame":
                        assert len(images_encode) == 1, f"single_frame is only supported for batch size 1, but got {len(images_encode)}"
                        # [B=1, T, C, H, W] -> [T_clip, B=1, C, H, W]
                        images_encode = images_encode[0].unsqueeze(1)  # [T_clip, B=1, C, H, W]
                        modalities = ["video"] * images_encode.shape[0]  # [T_clip]
                        input_ids_q_ = input_ids_q.repeat(images_encode.shape[0], 1)  # [T_clip, N]
                    else:
                        input_ids_q_ = input_ids_q  # [B, N]

                    (
                        _,  # input_ids_,
                        _,  # position_ids_,
                        attention_mask_encoder,
                        _,  # past_key_values_,
                        inputs_embeds_,
                        _,  # labels_
                        token_ranges,
                    ) = self.prepare_inputs_labels_for_multimodal(
                        input_ids_q_,  # [B, N]
                        None,  # position_ids,
                        None,  # attention_mask,
                        None,  # past_key_values,
                        None,  # labels,
                        images_encode,  # [B, T, C, H, W]
                        modalities,  # [B]
                        None,  # image_sizes
                    )

                    # paper: eq.(3)
                    encoder_outputs = self.forward_encoder(
                        inputs_embeds_,
                        attention_mask_encoder,
                        token_ranges if self.question_guided_selective_attention or self.ctx_attn_mask_type == "single_frame" else None,
                        output_attentions=self.compress_with_relevance,
                    )
                    context_embeds = self.get_context_embeds_from_encoder_outputs(encoder_outputs, token_ranges=token_ranges)  # [B, T, ctx, D]

                    if self.ctx_attn_mask_type == "single_frame":
                        # [T_clip, B=1, ctx, D] -> [B=1, T_clip, ctx, D]
                        context_embeds = context_embeds.transpose(0, 1)  # [B=1, T_clip, ctx, D]

                    if self.debug_drawing and not self.training:
                        image_indices = list(clip_indices)
                        if recall_memory_indices is not None:
                            recall_frame_indices = [int(self.frame_indices_memory[0, idx].item()) for idx in recall_memory_indices]
                            image_indices = sorted(list(set(image_indices) | set(recall_frame_indices)))
                        self.debug_analyze_attention(token_ranges, encoder_outputs, image_indices)

                    # Compute relevances
                    if self.compress_with_relevance:
                        output_attentions = [x.detach() for x in encoder_outputs.attentions]  # Layer x [batch=T, Head, Nq, Nk]
                        output_attentions = torch.stack(output_attentions, dim=0).detach()  # [Layer, batch=T, Head, Nq, Nk]
                        relevances = []
                        _, B, H, _, _ = output_attentions.shape
                        L1, L2 = self.relevance_layer_range
                        # multi-layer, top-k heads

                        if self.ctx_attn_mask_type == "single_frame":
                            for j in range(B):  # =T
                                i0, i1 = token_ranges[j]['range_image']
                                num_frames = token_ranges[j]['num_frames']
                                num_image_token = (i1 - i0) // num_frames
                                relevance_layers = []  # [layer]
                                for l in range(L1, L2):
                                    relevance_heads = []  # [head]
                                    for k in range(H):  # head
                                        relevance_heads.append(output_attentions[l, j, k, :, :num_image_token].mean().detach().cpu().float())  # q2v
                                    relevance_layers.append(torch.tensor(relevance_heads).sort(descending=True)[0][:self.top_k_head].mean())  # [1]
                                relevance = torch.stack(relevance_layers).mean().cpu().float()  # [1]
                                relevances.append(relevance)  # T
                            relevances = [torch.stack(relevances, dim=0)]  # B=1 x [T]

                        else:
                            for j in range(B):  # batch
                                # Calculate relevance scores
                                # paper: eq.(4)
                                _, _, Nq, Nk = output_attentions[-1].shape
                                num_frames = token_ranges[j]['num_frames']
                                num_image_token = Nk // num_frames
                                assert Nk % num_frames == 0, f"num_frames {num_frames} does not divide Nk {Nk}"

                                relevance_frames = []  # [T]
                                for i_frame in range(num_frames):  # T
                                    relevance_layers = []  # [layer]
                                    for l in range(L1, L2):
                                        relevance_heads = []  # [head]
                                        for k in range(H):  # head
                                            i0 = i_frame * num_image_token
                                            i1 = (i_frame + 1) * num_image_token
                                            relevance_heads.append(output_attentions[l, j, k, :-self.context_embed_tokens, i0:i1].mean().detach().cpu().float())  # q2v
                                        relevance_layers.append(torch.tensor(relevance_heads).sort(descending=True)[0][:self.top_k_head].mean())  # [1]
                                    relevance = torch.stack(relevance_layers).mean().cpu().float()  # [1]
                                    relevance_frames.append(relevance)  # T
                                relevance_batch = torch.stack(relevance_frames, dim=0)  # [T]
                                relevances.append(relevance_batch)  # B x [T]

                        relevances_outputs = torch.stack(relevances, dim=0)  # [B, T]

                    # Extract the part corresponding to the clip (excluding the recalled frames)
                    if self.compress_with_relevance and (num_frame_clip < self.max_frame_num_encoder) and recall_memory_indices is not None:
                        assert not self.training  # inference
                        num_recall_frames = len(recall_memory_indices)
                        relevance_to_update = relevances_outputs[:, :num_recall_frames]  # [B, T_selected]
                        relevance_clip = relevances_outputs[:, num_recall_frames:]  # [B, T_clip]

                        # Update the relevance memory
                        for idx, memory_idx in enumerate(recall_memory_indices):
                            self.relevance_memory[0, memory_idx] = relevance_to_update[0, idx]
                            # self.context_memory[0,mem_idx] = context_embeds[0,idx]
                            self.reflection_frequency[0, memory_idx] = self.reflection_frequency[0, memory_idx] + 1  # debug
                            frame_idx = int(self.frame_indices_memory[0, memory_idx].item())  # debug
                            relevance_memory_long[0, frame_idx] = relevance_to_update[0, idx]  # debug

                        relevances_outputs = relevance_clip  # [B, T_clip]
                        context_embeds = context_embeds[:, num_recall_frames:]  # [B, T_clip, ctx, D]

                    # Write to shared memory, need an I/O lock
                    # Context memory bank generation
                    for i_frame in range(context_embeds.shape[1]):
                        B, _, _, _ = context_embeds.shape
                        context_embed = context_embeds[:, i_frame:i_frame + 1]  # [B, 1, ctx, D]

                        if self.compress_with_relevance:
                            relevance = relevances_outputs[:, i_frame:i_frame + 1].to(self.device)  # [B, 1]
                            frame_idx = torch.tensor([start_idx + i_frame], dtype=torch.float32, device=self.device).unsqueeze(0).expand(B, -1)  # [B, 1]

                        size_constant = torch.ones(B, 1).to(context_embed.device)  # [B, 1]
                        freq_constant = torch.zeros(B, 1).to(context_embed.device)  # [B, 1]

                        if self.context_memory is None:
                            self.context_memory = context_embed  # [B, 1, ctx, D]
                            self.compression_size = size_constant
                            if self.compress_with_relevance:
                                self.relevance_memory = relevance
                                self.frame_indices_memory = frame_idx
                                self.reflection_frequency = freq_constant
                                relevance_memory_long = relevance  # debug
                        else:
                            self.context_memory = torch.cat([self.context_memory, context_embed], dim=1)  # [B, t+1, N, D]
                            self.compression_size = torch.cat([self.compression_size, size_constant], dim=1)  # [B, t+1]
                            if self.compress_with_relevance:
                                self.relevance_memory = torch.cat([self.relevance_memory, relevance], dim=1)  # [B, t+1]
                                self.frame_indices_memory = torch.cat([self.frame_indices_memory, frame_idx], dim=1)  # [B, t+1]
                                self.reflection_frequency = torch.cat([self.reflection_frequency, freq_constant], dim=1)  # [B, t+1]
                                relevance_memory_long = torch.cat([relevance_memory_long, relevance], dim=1)  # debug # [B, t+1]

                        if self.compress_with_relevance:
                            B, L, N, D = self.context_memory.shape
                            if L > 1:
                                # Calculate the similarity of the last adjacent frames, and remove the new frame if it exceeds the threshold
                                if L > self.context_memory_length:
                                    # Remove the frame with the lowest relevance
                                    index_min_relevance = torch.argmin(self.relevance_memory[0]).item()
                                    self.remove_target_frame_on_memory(index_min_relevance)

                        elif self.context_memory.size(1) > self.context_memory_length:
                            # Calculate the similarity of the adjacent frames in the context memory and average the adjacent frames with the highest similarity
                            B, L, N, D = self.context_memory.shape
                            similarity_matrix = F.cosine_similarity(
                                self.context_memory[:, :-1].view(B, L - 1, N * D).to(torch.float32),
                                self.context_memory[:, 1:].view(B, L - 1, N * D).to(torch.float32), dim=-1, eps=1e-8)
                            similarity_max, index_max = torch.max(similarity_matrix[0], dim=0)
                            self.merge_target_frame_on_memory(index_max, index_max + 1)

                if self.context_memory.size(1) < self.context_memory_length and self.fill_context_memory:

                    target_length = random.randint(self.context_memory.size(1), self.context_memory_length)
                    rank0_print(f"[INFO] randomly set target_length to {target_length}")

                    # nearest neighbor interpolation (keep the original context memory)
                    B, L, N, D = self.context_memory.shape
                    # [B, L', N, D] -> [B*N, D, L'] -> [B*N, D, L] -> [B, L, N, D]
                    self.context_memory = F.interpolate(
                        self.context_memory.permute(0, 2, 3, 1).contiguous().view(B * N, D, L),
                        size=target_length, mode='nearest'
                    ).view(B, N, D, target_length).permute(0, 3, 1, 2)
                    self.compression_size = F.interpolate(
                        self.compression_size.unsqueeze(1),
                        size=target_length, mode='nearest'
                    ).squeeze(1)
                    # self.relevance_memory = F.interpolate(
                    #    self.relevance_memory.unsqueeze(0).unsqueeze(2),
                    #    size=target_length, mode='nearest'
                    # ).squeeze(0).squeeze(1)
                    # self.frame_indices_memory = F.interpolate(
                    #    self.frame_indices_memory.unsqueeze(0).unsqueeze(2),
                    #    size=target_length, mode='nearest'
                    # ).squeeze(0).squeeze(1)
                    rank0_print(f"[INFO] interpolate context memory to {target_length} frames (from {L} frames)")

                if self.debug_drawing_memory and not self.training and self.compress_with_relevance:
                    rank0_print(f"[DEBUG] draw relevance plot (memory frames: {self.context_memory.size(1)})")

                    save_dir = f"./logs/context_memory/images/"
                    os.makedirs(save_dir, exist_ok=True)

                    batch_idx = 0
                    video_path, max_frames_num, fps = self.debug_input_image_info

                    video_name = video_path.split('/')[-1].split('.')[0]

                    import matplotlib.pyplot as plt
                    relevance_all_np = relevance_memory_long[batch_idx].detach().cpu().numpy()  # [T_total]
                    frame_indices_np = self.frame_indices_memory[batch_idx].detach().cpu().numpy().astype(int)  # [L]
                    relevance_np = self.relevance_memory[batch_idx].detach().cpu().numpy()
                    plt.figure(figsize=(8, 6))
                    plt.plot(relevance_all_np, "-o", label="Mean", linewidth=1)
                    plt.plot(frame_indices_np, relevance_np, "o", label="Memory Frames", color='red')
                    plt.xlabel('Frame')
                    plt.ylabel('Relevance')
                    plt.xlim(0, relevance_all_np.size)
                    plt.ylim(relevance_np.min() - 0.1 * (relevance_np.max() - relevance_np.min()), relevance_np.max() + 0.1 * (relevance_np.max() - relevance_np.min()))
                    plt.title(f"Relevance (Mean Attention Weights) (query2image)")
                    plt.grid()
                    savepath = os.path.join(save_dir, f'relevance_plot_{video_name}.png')
                    plt.savefig(savepath)
                    plt.close()
                    print(f"[DEBUG] Saved {savepath}")

                    relevance_all_np = relevance_memory_long[batch_idx].detach().cpu().numpy()  # [T_total]
                    frame_indices_np = self.frame_indices_memory[batch_idx].detach().cpu().numpy().astype(int)
                    frame_time = np.arange(0, len(relevance_all_np) / fps, 1 / fps)
                    relevance_np = self.relevance_memory[batch_idx].detach().cpu().numpy()
                    plt.figure(figsize=(10, 2))
                    plt.plot(frame_time, relevance_all_np, "-", label="Relevance scores", linewidth=0.5)
                    plt.plot(frame_time[frame_indices_np], relevance_np, "o", label="Context memory entries", color='blue', markersize=1)
                    plt.xlim(0, frame_time[-1])
                    # plt.ylim(relevance_np.min() - 0.1 * (relevance_np.max() - relevance_np.min()), 0.00012) # debug
                    plt.ylim(relevance_np.min() - 0.1 * (relevance_np.max() - relevance_np.min()), relevance_np.max() + 0.1 * (relevance_np.max() - relevance_np.min()))
                    plt.gca().yaxis.get_major_formatter().set_scientific(True)
                    plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 1))
                    plt.legend(loc='upper right', fontsize=8, ncol=2)
                    savepath = os.path.join(save_dir, f'relevance_plot_{video_name}_paper_vtight.png')
                    plt.savefig(savepath)
                    plt.savefig(savepath.replace(".png", ".eps"), format='eps')
                    plt.close()
                    print(f"[DEBUG] Saved {savepath} / {savepath.replace('.png', '.eps')}")

                    video_path, max_frames_num, fps = self.debug_input_image_info
                    from decord import VideoReader, cpu

                    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
                    total_frame_num = len(vr)
                    fps = round(vr.get_avg_fps() / fps)
                    memory_idx = [i for i in range(0, len(vr), fps)]
                    if max_frames_num > 0:  # (len(frame_idx) > max_frames_num) and max_frames_num > 0:
                        sample_fps = max_frames_num
                        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
                        memory_idx = uniform_sampled_frames.tolist()
                    frames_sampled = vr.get_batch(memory_idx).asnumpy()

                    frames_in_memory = frames_sampled[frame_indices_np]  # [L, H, W, C]
                    frames_in_memory = frames_in_memory[:, :, :, ::-1]  # BGR to RGB
                    n_frames = frames_in_memory.shape[0]
                    frame_width = frames_in_memory.shape[2]
                    frame_height = frames_in_memory.shape[1]
                    cols = rows = int(np.ceil(np.sqrt(n_frames)))
                    padding_frames = np.zeros((cols * rows - n_frames, frame_height, frame_width, 3), dtype=np.uint8)
                    frames_in_memory_tmp = np.concatenate([frames_in_memory, padding_frames], axis=0)
                    frame_reshaped = frames_in_memory_tmp.reshape(cols, rows, frame_height, frame_width, 3)
                    video_concat = np.concatenate([np.concatenate(frame_reshaped[i], axis=1) for i in range(cols)], axis=0)
                    video_concat = video_concat.astype(np.uint8)

                    # resize
                    width_max = 5000  # [px]
                    height_max = 4000  # [px]
                    if video_concat.shape[1] > width_max or video_concat.shape[0] > height_max:
                        scale = min(width_max / video_concat.shape[1], height_max / video_concat.shape[0])
                        video_concat = cv2.resize(video_concat, (int(video_concat.shape[1] * scale), int(video_concat.shape[0] * scale)), interpolation=cv2.INTER_LINEAR)

                    savepath = os.path.join(save_dir, f'video_ctx_memory_{video_name}.jpg')
                    cv2.imwrite(savepath, video_concat)
                    print(f"[DEBUG] Saved video context memory image: {savepath}")

                    # breakpoint() #debug

                if to_cpu_memory:  # multi-processing inference
                    self.memory_for_each_batch = self.context_memory.cpu()  # [B, L, N, D]
                else:
                    self.memory_for_each_batch = self.context_memory  # [B, L, N, D]
            else:
                raise NotImplementedError('Should input video frames, not a single image')

        return []

    def clear_memory_for_one_batch(self, delete_tensor=True):
        if self.context_memory is not None:
            if delete_tensor:
                del self.context_memory
            self.context_memory = None
        if self.compression_size is not None:
            if delete_tensor:
                del self.compression_size
            self.compression_size = None
        if self.relevance_memory is not None:
            if delete_tensor:
                del self.relevance_memory
            self.relevance_memory = None
        if self.frame_indices_memory is not None:
            if delete_tensor:
                del self.frame_indices_memory
            self.frame_indices_memory = None

    def clear_memory(self, delete_tensor=True):
        if self.memory_for_each_batch is not None:
            if delete_tensor:
                del self.memory_for_each_batch
            self.memory_for_each_batch = None
        self.clear_memory_for_one_batch()

    # ------------------------------------------------------------------
    # Helper used by `prepare_inputs_labels_for_multimodal` (in llava_arch.py)
    # ------------------------------------------------------------------

    def _qvic_append_context_tokens(
        self,
        new_input_embeds,
        attention_mask,
        _input_ids,
        image_features,
        num_frames,
    ):
        """Append QViC context tokens to ``new_input_embeds``, compute token_ranges,
        and (optionally) build the QMSA attention mask.

        This encapsulates the QViC-specific block that used to live inline at the
        tail of :meth:`LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal`
        (paper eq.(2) for the context concat and eq.(6)-(8) for the QMSA mask).

        Returns:
            new_input_embeds: ``[B, N + T*C, D]``
            attention_mask: 4D QMSA mask if ``self.question_guided_selective_attention``
                is True, otherwise the input ``attention_mask`` is returned unchanged.
            token_ranges: list of dicts (one per batch element) describing token
                spans for system / image / user-query / context.
        """
        # paper: eq.(2)
        # Concatenate context embedding to text tokens
        context_tokens = torch.arange(0, self.context_embed_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)  # [C]
        self.context_embed = self.context_embed.to(self.device)
        context_embedding = self.context_embed(context_tokens)  # [1, C, D]
        # Repeat the context for the number of images
        B = _input_ids.shape[0]
        context_embedding = context_embedding.repeat(B, num_frames, 1)  # [B, T*C, D]

        new_input_embeds = torch.cat([new_input_embeds, context_embedding], dim=1)

        # token ranges
        B, _, _ = new_input_embeds.shape  # [1, 18331, 3584]
        token_ranges = []
        for batch in range(B):
            n_image_embed_tokens = image_features[batch].shape[0]  # 729, 13440
            n_context_embed_tokens = context_embedding[batch].shape[0]  # 64, 4096
            image_token_index = torch.where(_input_ids == -200)[1][batch].item()  # 14
            # n_query_user_embeds = N - n_image_embed_tokens - n_context_embed_tokens - image_token_index # 781
            n_query_user_embeds = _input_ids.shape[1] - image_token_index - 1  # 781

            # token indices
            range_query_system = [0, image_token_index]  # [0, 14]
            range_image = [range_query_system[1], range_query_system[1] + n_image_embed_tokens]  # [14, 13454]
            range_query_user = [range_image[1], range_image[1] + n_query_user_embeds]  # [13454, 14235]
            range_context = [range_query_user[1], range_query_user[1] + n_context_embed_tokens]  # [14235, 18331]

            # count padding tokens in input_ids_q, extract actual number of query user embeds
            pad_token_count = torch.sum(_input_ids[batch] == self.pad_token_id).item()
            range_query_user[1] -= pad_token_count  # adjust range_query_user[1] to exclude padding tokens
            range_query_padding = [range_query_user[1], range_query_user[1] + pad_token_count]

            token_ranges.append({
                "range_query_system": range_query_system,
                "range_image": range_image,
                "range_query_user": range_query_user,
                "range_query_padding": range_query_padding,  # unused
                "range_context": range_context,
                "num_frames": num_frames,
                "num_contexts": num_frames,
                "guiding_context2vision": self.guiding_context2vision,
            })

        if self.training and self.verbose:
            rank0_print(f"[INFO] question-guided attention token ranges: {token_ranges}")

        if self.question_guided_selective_attention:
            attention_mask = self.make_attention_mask_qmsa(new_input_embeds, token_ranges)

        return new_input_embeds, attention_mask, token_ranges

    # ------------------------------------------------------------------
    # Debug visualization
    # ------------------------------------------------------------------

    def debug_analyze_attention(self, token_ranges, encoder_outputs, image_indices, suffix="avg",
                                n_layers=-1, save_dir=f"./logs/attention_weight/images/",
                                draw_matrix=False):
        '''
        Visualize attention weights of each layer
        encoder_outputs.attentions: [N_layer, N_batch, N_head, N_tokens, N_tokens]
        '''
        os.makedirs(save_dir, exist_ok=True)

        # attention
        output_attentions = [x.detach() for x in encoder_outputs.attentions]  # [Layer, batch, Head, N, N]
        attentions = torch.stack(output_attentions, dim=0).detach().cpu().float().numpy()  # [Layer, batch, Head, N, N]
        # compressor_attentions = np.mean(compressor_attentions, axis=1) # average over heads [Layer, N, N]

        L, B, H, N, N = attentions.shape
        assert B == 1, "Currently only support batch size = 1 for attention visualization"

        range_query_system = token_ranges[0]['range_query_system']
        range_image = token_ranges[0]['range_image']
        range_query_user = token_ranges[0]['range_query_user']
        range_context = token_ranges[0]['range_context']
        num_frames = token_ranges[0]['num_frames']
        num_contexts = token_ranges[0]['num_contexts']

        n_image_tokens = int((range_image[1] - range_image[0]) / num_frames)
        n_context_tokens = int((range_context[1] - range_context[0]) / num_contexts)
        assert n_image_tokens * num_frames == (range_image[1] - range_image[0])
        assert n_context_tokens * num_contexts == (range_context[1] - range_context[0])

        mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")
        if mm_newline_position == "grid":
            # delete mm_newline tokens from attention matrix
            D = 1 + 4 * n_image_tokens
            r = math.isqrt(D)
            assert r * r == D, "Image tokens should form a square grid plus one mm_newline token per row and column"
            width = height = (r - 1) // 2
            image_mask = np.ones((height, width + 1), dtype=bool)
            image_mask[:, -1:] = False  # mark mm_newline tokens
            image_mask = image_mask.flatten()
            assert n_image_tokens == (width + 1) * height, "Image tokens should form a square grid plus one mm_newline token per row"
        else:
            width = height = int(math.sqrt(n_image_tokens))
            image_mask = np.ones((height, width), dtype=bool)
            image_mask = image_mask.flatten()
            assert n_image_tokens == width * height, "Image tokens should form a square grid"

        # Draw context2image attention weight heatmap
        rows = 6  # math.ceil(math.sqrt(n_layers))
        cols = 6  # math.ceil(n_layers / cols)
        L1, L2 = self.relevance_layer_range

        if self.debug_input_image_info is not None:
            from PIL import Image
            import cv2
            from decord import VideoReader, cpu

            video_path, max_frames_num, fps = self.debug_input_image_info

            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            total_frame_num = len(vr)
            fps = round(vr.get_avg_fps() / fps)
            frame_idx = [i for i in range(0, len(vr), fps)]
            if max_frames_num > 0:  # (len(frame_idx) > max_frames_num) and max_frames_num > 0:
                sample_fps = max_frames_num
                uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
                frame_idx = uniform_sampled_frames.tolist()
            frames_sampled_all = vr.get_batch(frame_idx).asnumpy()
            frames_sampled = frames_sampled_all[image_indices]

            breakpoint()

            if True:
                attention_maps = []
                for idx, frame in enumerate(frames_sampled):
                    i0 = idx * n_image_tokens
                    i1 = i0 + n_image_tokens
                    # -- c2v
                    # attention_map = attentions[L1:L2, 0, :, -self.context_embed_tokens:, i0:i1] # [n_layers, n_heads, n_context_embeds, n_image_tokens]
                    # attention_map = attentions[:, 0, :, -self.context_embed_tokens:, i0:i1] # [n_layers, n_heads, n_context_embeds, n_image_tokens]
                    # attention_map = attentions[L1:L2, 0, :, -self.context_embed_tokens:, i0:i1] # [n_layers, n_heads, n_context_embeds, n_image_tokens]
                    attention_map = attentions[L1:, 0, :, -self.context_embed_tokens:, i0:i1]  # [n_layers, n_heads, n_context_embeds, n_image_tokens] # good
                    # attention_map = attentions[-2:-1, 0, :, -self.context_embed_tokens:, i0:i1] # [n_layers, n_heads, n_context_embeds, n_image_tokens] #
                    # -- q2v
                    # attention_map = attentions[L1:L2, 0, :, :-self.context_embed_tokens, i0:i1] # [n_layers, n_heads, n_context_embeds, n_image_tokens]
                    # attention_map = attentions[:, 0, :, :-self.context_embed_tokens, i0:i1] # [n_layers, n_heads, n_context_embeds, n_image_tokens]
                    # --

                    attention_maps_layer = []
                    for layer in range(attention_map.shape[0]):
                        attn_sum_list = []
                        for head in range(attention_map.shape[1]):
                            attn_sum_list.append(attention_map[layer, head].mean())
                        top_k_indices = np.argsort(np.array(attn_sum_list))[-self.top_k_head:]  # [top_k]
                        # top_k_indices = np.argsort(np.array(attn_sum_list)) # [all_heads]
                        attention_maps_layer.append(attention_map[layer, top_k_indices.flatten()])  # [n_heads, n_context_tokens*n_context_tokens, n_image_tokens]
                    attention_map = np.stack(attention_maps_layer, axis=0)  # [n_layers, top_k, n_context_tokens, n_image_tokens]
                    attention_map = attention_map[..., image_mask].mean(axis=(0, 1))
                    attention_maps.append(attention_map)
                attention_maps = np.stack(attention_maps, axis=0)  # [num_frames, n_context_embeds, n_image_tokens]
                aggregated_attention = attention_maps.mean(axis=1)  # [num_frames, n_image_tokens]

                # normalize_framewise = True # c2v
                normalize_framewise = False  # q2v

                if not normalize_framewise:
                    attn_percentile_95 = np.percentile(aggregated_attention, 95)
                    attn_percentile_5 = np.percentile(aggregated_attention, 5)
                    clip_range = [attn_percentile_5, attn_percentile_95]  # q2v
                    # clip_range = [1e-5, 1e-4] #c2v
                    aggregated_attention = np.log(np.clip(aggregated_attention, clip_range[0], clip_range[1]))
                    aggregated_attention = (aggregated_attention - aggregated_attention.min()) / (aggregated_attention.max() - aggregated_attention.min()) * 255

                heatmap_images = []
                sigmoid_images = []
                for idx, frame in enumerate(frames_sampled):
                    image = Image.fromarray(frame).convert("RGB")
                    image_size_original = image.size
                    aggregated_attention_overlay = aggregated_attention[idx].reshape(width, height)
                    aggregated_attention_overlay = gaussian_filter(aggregated_attention_overlay, sigma=1.0)
                    aggregated_attention_overlay = cv2.resize(aggregated_attention_overlay, image_size_original, interpolation=cv2.INTER_CUBIC)

                    if normalize_framewise:
                        attn_percentile_95 = np.percentile(aggregated_attention_overlay, 95)
                        attn_percentile_5 = np.percentile(aggregated_attention_overlay, 5)
                        clip_range = [attn_percentile_5, attn_percentile_95]
                        aggregated_attention_overlay = np.log(np.clip(aggregated_attention_overlay, clip_range[0], clip_range[1]))
                        aggregated_attention_overlay = (aggregated_attention_overlay - aggregated_attention_overlay.min()) / (aggregated_attention_overlay.max() - aggregated_attention_overlay.min()) * 255

                    aggregated_attention_overlay = aggregated_attention_overlay.astype(np.uint8)
                    heatmap_overlay = cv2.applyColorMap(aggregated_attention_overlay, cv2.COLORMAP_JET)
                    heatmap_overlay = cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB)
                    image_np = np.array(image)
                    alpha_ratio = 0.4
                    overlayed_image = cv2.addWeighted(image_np, alpha_ratio, heatmap_overlay, 1 - alpha_ratio, 0)
                    overlayed_image_pil = Image.fromarray(overlayed_image)
                    overlayed_image_pil.save(os.path.join(save_dir, f'overlay_context_to_image_mean_{suffix}_{idx}.png'))
                    heatmap_images.append(overlayed_image)

                    strength = 10
                    sigmoid_weight = 1 / (1 + np.exp(-strength * (aggregated_attention_overlay / 255.0 - 0.5)))
                    image_np_sigmoid = (image_np * sigmoid_weight[..., None]).astype(np.uint8)
                    image_np_sigmoid_pil = Image.fromarray(image_np_sigmoid)
                    image_np_sigmoid_pil.save(os.path.join(save_dir, f'overlay_context_to_image_sigmoid_{suffix}_{idx}.png'))
                    sigmoid_images.append(image_np_sigmoid)

                # concatenate heatmap_images and sigmoid_images and save
                if True:
                    heatmap_images = np.array(heatmap_images)
                    sigmoid_images = np.array(sigmoid_images)
                    L, H, W, C = heatmap_images.shape
                    cols = rows = int(np.ceil(np.sqrt(L)))
                    concat_heatmap_image = np.ones((rows * H, cols * W, C), dtype=np.uint8) * 255
                    concat_sigmoid_image = np.ones((rows * H, cols * W, C), dtype=np.uint8) * 255
                    for i in range(L):
                        r = i // cols
                        c = i % cols
                        concat_heatmap_image[r * H:(r + 1) * H, c * W:(c + 1) * W, :] = heatmap_images[i]
                        concat_sigmoid_image[r * H:(r + 1) * H, c * W:(c + 1) * W, :] = sigmoid_images[i]
                    # resize
                    width_max = 2000  # [px]
                    height_max = 1500  # [px]
                    scale = min(width_max / concat_heatmap_image.shape[1], height_max / concat_heatmap_image.shape[0], 1.0)
                    new_size = (int(concat_heatmap_image.shape[1] * scale), int(concat_heatmap_image.shape[0] * scale))
                    concat_heatmap_image_pil = Image.fromarray(concat_heatmap_image).resize(new_size, Image.Resampling.LANCZOS)
                    concat_sigmoid_image_pil = Image.fromarray(concat_sigmoid_image).resize(new_size, Image.Resampling.LANCZOS)
                    concat_heatmap_image_pil.save(os.path.join(save_dir, f'concat_overlay_context_to_image_mean_{suffix}.png'))
                    print("Saved:", os.path.join(save_dir, f'concat_overlay_context_to_image_mean_{suffix}.png'))
                    concat_sigmoid_image_pil.save(os.path.join(save_dir, f'concat_overlay_context_to_image_sigmoid_{suffix}.png'))
                    print("Saved:", os.path.join(save_dir, f'concat_overlay_context_to_image_sigmoid_{suffix}.png'))

        return
