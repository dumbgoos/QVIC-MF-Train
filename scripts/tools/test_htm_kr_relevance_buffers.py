#!/usr/bin/env python3
# #@HTM
"""CPU unit test: HTM + K_r relevance tracking helpers (no VLM weights).

Avoids importing ``qvic_meta_mixin`` (scipy/cv2/VLM deps). Copies the prune
helper body so it stays reviewable next to the mixin change.
"""

from __future__ import annotations

import torch


class _Buffers:
    def __init__(self, L: int = 4):
        self.relevance_memory = None
        self.frame_indices_memory = None
        self.reflection_frequency = None
        self.context_memory_length = L

    def _remove_relevance_tracking_entry(self, index_remove):
        # Keep in sync with qvic/model/qvic_meta_mixin.py::_remove_relevance_tracking_entry
        assert self.relevance_memory is not None and self.frame_indices_memory is not None
        assert index_remove < self.relevance_memory.shape[1]
        if index_remove == 0:
            self.relevance_memory = self.relevance_memory[:, 1:]
            self.frame_indices_memory = self.frame_indices_memory[:, 1:]
            if self.reflection_frequency is not None:
                self.reflection_frequency = self.reflection_frequency[:, 1:]
        elif index_remove == self.relevance_memory.shape[1] - 1:
            self.relevance_memory = self.relevance_memory[:, :-1]
            self.frame_indices_memory = self.frame_indices_memory[:, :-1]
            if self.reflection_frequency is not None:
                self.reflection_frequency = self.reflection_frequency[:, :-1]
        else:
            self.relevance_memory = torch.cat(
                [self.relevance_memory[:, :index_remove], self.relevance_memory[:, index_remove + 1:]], dim=1
            )
            self.frame_indices_memory = torch.cat(
                [self.frame_indices_memory[:, :index_remove], self.frame_indices_memory[:, index_remove + 1:]], dim=1
            )
            if self.reflection_frequency is not None:
                self.reflection_frequency = torch.cat(
                    [self.reflection_frequency[:, :index_remove], self.reflection_frequency[:, index_remove + 1:]], dim=1
                )


def _append_like_htm(obj: _Buffers, start_idx: int, relevances: torch.Tensor) -> None:
    """Mirror the HTM K_r append/prune block in embed_video_streaming."""
    B = relevances.shape[0]
    for i_frame in range(relevances.shape[1]):
        relevance = relevances[:, i_frame : i_frame + 1]
        frame_idx = (
            torch.tensor([start_idx + i_frame], dtype=torch.float32)
            .unsqueeze(0)
            .expand(B, -1)
        )
        freq_constant = torch.zeros(B, 1)
        if obj.relevance_memory is None:
            obj.relevance_memory = relevance
            obj.frame_indices_memory = frame_idx
            obj.reflection_frequency = freq_constant
        else:
            obj.relevance_memory = torch.cat([obj.relevance_memory, relevance], dim=1)
            obj.frame_indices_memory = torch.cat([obj.frame_indices_memory, frame_idx], dim=1)
            obj.reflection_frequency = torch.cat([obj.reflection_frequency, freq_constant], dim=1)
        if obj.relevance_memory.shape[1] > obj.context_memory_length:
            index_min = torch.argmin(obj.relevance_memory[0]).item()
            obj._remove_relevance_tracking_entry(index_min)


def main() -> None:
    obj = _Buffers(L=4)
    # Clip 0: three frames → buffers initialized (None → crash on clip 1 before fix).
    _append_like_htm(obj, start_idx=0, relevances=torch.tensor([[0.1, 0.5, 0.3]]))
    assert obj.relevance_memory is not None
    assert obj.relevance_memory.shape == (1, 3)
    assert obj.frame_indices_memory.tolist() == [[0.0, 1.0, 2.0]]

    # Clip 1: two more frames; L=4 → prune lowest (0.1 at frame 0) after capacity breach.
    _append_like_htm(obj, start_idx=3, relevances=torch.tensor([[0.9, 0.2]]))
    assert obj.relevance_memory.shape[1] == obj.context_memory_length == 4
    assert obj.relevance_memory.shape[1] == obj.frame_indices_memory.shape[1]
    scores = set(round(x, 5) for x in obj.relevance_memory[0].tolist())
    assert scores == {0.5, 0.3, 0.9, 0.2}
    assert 0.0 not in obj.frame_indices_memory[0].tolist()

    top = torch.argsort(obj.relevance_memory[0], descending=True).tolist()
    assert top[0] == obj.relevance_memory[0].argmax().item()

    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "qvic/model/qvic_meta_mixin.py").read_text()
    assert "def _remove_relevance_tracking_entry" in src
    assert "HTM write path must maintain K_r tracking buffers" in src
    assert "K_r read path is orthogonal to HTM write" in src

    print("ok", {"L": obj.context_memory_length, "scores": obj.relevance_memory[0].tolist()})


if __name__ == "__main__":
    main()
