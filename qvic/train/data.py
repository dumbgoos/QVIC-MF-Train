from __future__ import annotations

import contextlib
import copy
import io
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from qvic.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from qvic.conversation import conv_templates
from qvic.mm_utils import tokenizer_image_token

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Media access
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def suppressed_c_stderr(enabled: bool = True):
    """Silence libavcodec's per-frame chatter during a decode.

    LLaVA-Video's YouTube clips make the bundled h264 decoder emit
    "mmco: unref short failure" and friends -- hundreds of lines per long video,
    ~50k lines over an epoch. decord links its own ffmpeg and these go through
    `av_log` straight to fd 2, so no Python logging filter can reach them.

    Redirect the file descriptor for the duration of the decode only, so this
    module's own warnings (a failed sample, say) still reach the job log.
    """
    if not enabled:
        yield
        return
    sys.stderr.flush()
    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(devnull_fd)
        os.close(saved_fd)


def harvest_inflight(inflight_dir: str, skip_list_path: str) -> List[str]:
    """Fold breadcrumbs left by a killed dataloader worker into the skip list.

    decord segfaults on some of the corpus. That kills the worker *process*, so
    `__getitem__`'s try/except never runs and the whole job dies -- and on the
    next attempt the same sample is waiting. Each worker therefore records the
    media key it is about to decode; a breadcrumb still present at start-up
    belonged to a decode that was in flight when the process died.

    Run this once per launch, before the ranks start, so it is single-threaded
    and race-free. Returns the newly blacklisted keys.

    Every in-flight key is blacklisted, not just the guilty one (the survivors
    cannot be told apart after the fact), so a crash costs a handful of the
    83 000 records -- an acceptable price for a run that then completes.
    """
    if not os.path.isdir(inflight_dir):
        return []
    found: List[str] = []
    for name in sorted(os.listdir(inflight_dir)):
        path = os.path.join(inflight_dir, name)
        try:
            with open(path) as fh:
                key = json.load(fh)["video"]
            found.append(key)
        except Exception:  # noqa: BLE001 -- a torn write is itself uninformative
            pass
        try:
            os.remove(path)
        except OSError:
            pass
    if not found:
        return []

    existing: List[str] = []
    if os.path.exists(skip_list_path):
        with open(skip_list_path) as fh:
            existing = json.load(fh)
    merged = sorted(set(existing) | set(found))
    os.makedirs(os.path.dirname(skip_list_path) or ".", exist_ok=True)
    tmp = skip_list_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(merged, fh, indent=1)
    os.replace(tmp, skip_list_path)
    new = sorted(set(found) - set(existing))
    logger.warning("blacklisted %d media key(s) after a worker died: %s", len(new), new)
    return new


def uniform_frame_indices(n_total: int, k: int) -> np.ndarray:
    """The frame indices LLaVA-Video uses for `force_sample=True`.

    `np.linspace(0, n-1, k)` repeats frames when the clip is shorter than `k`,
    which is what we want for the llava_hound scenes (median 21 frames).
    """
    if n_total < 1:
        raise ValueError(f"empty media (n_total={n_total})")
    return np.linspace(0, n_total - 1, k, dtype=int)


class MediaLoader:
    """Resolve an annotation `video` field to `[K, H, W, 3]` uint8 frames.

    Two backends:
      * ``shards`` -- the packed LLaVA-Video-83K release (tar + byte index).
      * ``loose``  -- a plain directory tree (``<video_folder>/<video field>``),
                      useful if the shards were extracted or for a smoke test.
    """

    def __init__(self, backend: str = "shards", dataset_root: Optional[str] = None,
                 video_folder: Optional[str] = None, quiet_decoder: bool = True):
        self.backend = backend
        self.dataset_root = dataset_root
        self.video_folder = video_folder
        self.quiet_decoder = quiet_decoder
        self._reader = None
        if backend == "shards":
            if not dataset_root:
                raise ValueError("media_backend='shards' requires dataset_root")
            from qvic.train.shard_reader import ShardReader
            # Built once in the parent process; DataLoader workers inherit the
            # index through fork and re-open their own file handles (ShardReader
            # keys its fd cache on os.getpid()).
            self._reader = ShardReader(dataset_root)
        elif backend != "loose":
            raise ValueError(f"unknown media_backend: {backend}")

    # -- shards -------------------------------------------------------------

    def _load_shards(self, key: str, k: int) -> np.ndarray:
        kind = self._reader.kind_of(key)
        if kind == "scene":
            members = self._reader.scene_members(key)
            idx = uniform_frame_indices(len(members), k)
            from PIL import Image
            frames = [
                np.asarray(Image.open(io.BytesIO(self._reader.read(members[i]))).convert("RGB"))
                for i in idx
            ]
            return np.stack(frames, axis=0)
        vr = self._reader.video_reader(key, num_threads=1)
        idx = uniform_frame_indices(len(vr), k)
        return vr.get_batch(idx.tolist()).asnumpy()

    # -- loose --------------------------------------------------------------

    def _load_loose(self, key: str, k: int) -> np.ndarray:
        path = os.path.join(self.video_folder, key) if self.video_folder else key
        if os.path.isdir(path):
            names = sorted(os.listdir(path))
            idx = uniform_frame_indices(len(names), k)
            from PIL import Image
            frames = [np.asarray(Image.open(os.path.join(path, names[i])).convert("RGB")) for i in idx]
            return np.stack(frames, axis=0)
        from decord import VideoReader, cpu
        vr = VideoReader(path, ctx=cpu(0), num_threads=1)
        idx = uniform_frame_indices(len(vr), k)
        return vr.get_batch(idx.tolist()).asnumpy()

    def __call__(self, key: str, k: int) -> np.ndarray:
        with suppressed_c_stderr(self.quiet_decoder):
            if self.backend == "shards":
                return self._load_shards(key, k)
            return self._load_loose(key, k)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt_and_target(
    question: str,
    answer: str,
    tokenizer,
    conv_template: str = "qwen_1_5",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (input_ids, labels, input_ids_q).

    The prompt half is tokenized exactly the way `lmms_eval/models/qvic_mf.py`
    and `playground/demo/inference_qvic.py` do it, so train and test see the
    same token stream. The answer is tokenized separately and concatenated,
    which avoids any BPE-merge drift at the prompt/answer boundary.
    """
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    prompt_q = conv.get_prompt()

    q_ids: List[int] = tokenizer_image_token(prompt_q, tokenizer, IMAGE_TOKEN_INDEX)
    a_ids: List[int] = tokenizer(answer + conv.sep + "\n", add_special_tokens=False).input_ids

    input_ids = torch.tensor(q_ids + a_ids, dtype=torch.long)
    labels = torch.tensor([IGNORE_INDEX] * len(q_ids) + a_ids, dtype=torch.long)
    input_ids_q = torch.tensor(q_ids, dtype=torch.long)
    return input_ids, labels, input_ids_q


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    dataset_root: str
    annotation_path: Optional[str] = None
    num_frames: int = 64
    conv_template: str = "qwen_1_5"
    qa_mode: str = "single"          # "single": one QA turn per record | "all": every turn
    qa_pick: str = "random"          # "random" | "first"   (only used when qa_mode == "single")
    qa_seed: int = 42
    media_backend: str = "shards"
    video_folder: Optional[str] = None
    max_records: int = -1
    model_max_length: int = 32768
    max_retries: int = 20
    quiet_decoder: bool = True
    skip_list_path: Optional[str] = None
    inflight_dir: Optional[str] = None


class QViCSupervisedDataset(Dataset):
    """LLaVA-Video-83K -> single-question QViC training samples."""

    def __init__(self, cfg: DataConfig, tokenizer, image_processor):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        ann = cfg.annotation_path or os.path.join(
            cfg.dataset_root, "annotations", "all_sampled.json")
        with open(ann, "r") as fh:
            records = json.load(fh)
        if cfg.max_records and cfg.max_records > 0:
            records = records[: cfg.max_records]

        # Media that previously killed a dataloader worker (see harvest_inflight).
        skipped = set()
        if cfg.skip_list_path and os.path.exists(cfg.skip_list_path):
            with open(cfg.skip_list_path) as fh:
                skipped = set(json.load(fh))
        if skipped:
            before = len(records)
            records = [r for r in records if r["video"] not in skipped]
            logger.info("skip list: dropped %d record(s) over %d blacklisted media key(s)",
                        before - len(records), len(skipped))
        self.records = records

        # (record_index, qa_index) pairs. A "QA index" is the index of the human
        # turn inside record["conversations"] (turns alternate human/gpt).
        self.index: List[Tuple[int, int]] = []
        rng = random.Random(cfg.qa_seed)
        n_pairs_total = 0
        for ri, rec in enumerate(records):
            conv = rec["conversations"]
            turns = [ti for ti in range(0, len(conv) - 1, 2)
                     if conv[ti]["from"] == "human" and conv[ti + 1]["from"] == "gpt"]
            if not turns:
                continue
            n_pairs_total += len(turns)
            if cfg.qa_mode == "all":
                self.index.extend((ri, t) for t in turns)
            elif cfg.qa_pick == "first":
                self.index.append((ri, turns[0]))
            else:
                self.index.append((ri, rng.choice(turns)))

        logger.info(
            "LLaVA-Video-83K: %d records, %d QA pairs -> %d training samples "
            "(qa_mode=%s, qa_pick=%s, seed=%d)",
            len(records), n_pairs_total, len(self.index),
            cfg.qa_mode, cfg.qa_pick, cfg.qa_seed,
        )

        self.media = MediaLoader(
            backend=cfg.media_backend,
            dataset_root=cfg.dataset_root,
            video_folder=cfg.video_folder,
            quiet_decoder=cfg.quiet_decoder,
        )

    def __len__(self) -> int:
        return len(self.index)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _clean_question(text: str) -> str:
        """Strip the <image> placeholder; we re-insert exactly one ourselves."""
        return text.replace(DEFAULT_IMAGE_TOKEN, "").strip()

    def _build(self, pos: int) -> Dict[str, Any]:
        ri, ti = self.index[pos]
        rec = self.records[ri]
        conv = rec["conversations"]
        question = self._clean_question(conv[ti]["value"])
        answer = conv[ti + 1]["value"].strip()

        # Breadcrumb: a segfault in decord takes the worker process down, so this
        # file is the only record of which media it was.
        crumb = None
        if self.cfg.inflight_dir:
            crumb = os.path.join(self.cfg.inflight_dir, f"{os.getpid()}.json")
            try:
                with open(crumb, "w") as fh:
                    json.dump({"video": rec["video"], "id": rec.get("id", "")}, fh)
            except OSError:
                crumb = None

        frames = self.media(rec["video"], self.cfg.num_frames)          # [K, H, W, 3] uint8

        if crumb:
            try:
                os.remove(crumb)
            except OSError:
                pass
        pixel_values = self.image_processor.preprocess(
            frames, return_tensors="pt")["pixel_values"]                 # [K, 3, 384, 384] fp32

        input_ids, labels, input_ids_q = build_prompt_and_target(
            question, answer, self.tokenizer, self.cfg.conv_template)

        m = self.cfg.model_max_length
        if m and len(input_ids) > m:
            input_ids, labels = input_ids[:m], labels[:m]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "input_ids_q": input_ids_q,
            "image": pixel_values,
            "modality": "video",
            "record_id": rec.get("id", ""),
        }

    def __getitem__(self, pos: int) -> Dict[str, Any]:
        # Broken media in a 429 GiB corpus should cost one sample, not the run.
        for attempt in range(self.cfg.max_retries):
            try:
                return self._build(pos)
            except Exception as exc:  # noqa: BLE001
                ri, _ = self.index[pos]
                logger.warning(
                    "sample %d (record %s, video %s) failed: %s -- substituting another sample",
                    pos, self.records[ri].get("id", "?"), self.records[ri].get("video", "?"), exc,
                )
                pos = (pos + 1 + attempt) % len(self.index)
        raise RuntimeError(f"{self.cfg.max_retries} consecutive samples failed to load")


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorForQViC:
    """Right-pad the text streams; keep the frame tensors as a list.

    `images` must stay a Python list of `[K, C, H, W]` tensors: that is the
    shape `QViCMetaMixin.embed_video_streaming` expects (`images[b][clip_idx]`).
    """

    tokenizer: Any
    pad_token_id: int
    frame_dtype: torch.dtype = torch.bfloat16

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [x["input_ids"] for x in instances]
        labels = [x["labels"] for x in instances]
        input_ids_q = [x["input_ids_q"] for x in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
        # `_qvic_append_context_tokens` recovers the true question length by
        # counting pad ids, so the padding value here must be pad_token_id.
        input_ids_q = torch.nn.utils.rnn.pad_sequence(
            input_ids_q, batch_first=True, padding_value=self.pad_token_id)

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            # .long(), matching `lmms_eval/models/qvic_mf.py`
            "attention_mask": input_ids.ne(self.pad_token_id).long(),
            "input_ids_q": input_ids_q,
            "images": [x["image"].to(self.frame_dtype) for x in instances],
            "image_sizes": None,
            "modalities": [x["modality"] for x in instances],
        }
        return batch
