# #@HTM
"""HTM-v1 Stage A helpers: GRU predictor, dual surprise, event merge, token budget.

Borrowed patterns (see internal/htm-prior-art.md):
  - GRU + linear next-state (hyssong/memorymodel EM-GRU; CES VCP)
  - Prediction-error event gating (saakur/EventSegmentation; GateL0RD)
  - Count-weighted running-mean merge (MF compression_size mean)
  - Fixed-capacity eviction (EM-GRU n_max_mem / MF prune)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# #@HTM
def htm_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine distance + MSE over the last dim. Returns per-batch scores [...]."""
    a_f = a.float()
    b_f = b.float()
    cos = 1.0 - F.cosine_similarity(a_f, b_f, dim=-1, eps=1e-8)
    mse = (a_f - b_f).pow(2).mean(dim=-1)
    return cos + mse


# #@HTM
class HTMPredictor(nn.Module):
    """Small GRU next-state predictor on pooled context vectors ``\\bar c_t``."""

    def __init__(self, dim: int, hidden_size: Optional[int] = None):
        super().__init__()
        hidden = int(hidden_size) if hidden_size is not None else dim
        self.dim = dim
        self.hidden_size = hidden
        self.gru = nn.GRU(input_size=dim, hidden_size=hidden, num_layers=1, batch_first=True)
        self.proj = nn.Linear(hidden, dim)

    def forward_step(
        self, c_bar: torch.Tensor, h: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            c_bar: ``[B, D]`` pooled context
            h: ``[1, B, H]`` or None
        Returns:
            c_hat_next: ``[B, D]`` prediction for the next clip
            h_new: ``[1, B, H]``
        """
        out, h_new = self.gru(c_bar.unsqueeze(1), h)
        c_hat_next = self.proj(out.squeeze(1))
        return c_hat_next, h_new


# #@HTM
class QuantileThresholds:
    """Adaptive τ_s / τ_d from a rolling score buffer (~70–80th percentile)."""

    def __init__(self, quantile: float = 0.9, maxlen: int = 4096, warmup: int = 32):  # #@HTM locked default
        self.quantile = float(quantile)
        self.warmup = int(warmup)
        self._s: Deque[float] = deque(maxlen=maxlen)
        self._d: Deque[float] = deque(maxlen=maxlen)
        self.tau_s = 1.0
        self.tau_d = 1.0

    def update(self, s: Optional[float] = None, d: Optional[float] = None) -> None:
        if s is not None and s == s:  # not NaN
            self._s.append(float(s))
        if d is not None and d == d:
            self._d.append(float(d))
        if len(self._s) >= self.warmup:
            t = torch.tensor(list(self._s), dtype=torch.float32)
            self.tau_s = float(torch.quantile(t, self.quantile).item())
        if len(self._d) >= self.warmup:
            t = torch.tensor(list(self._d), dtype=torch.float32)
            self.tau_d = float(torch.quantile(t, self.quantile).item())

    def ready(self) -> bool:
        return len(self._s) >= self.warmup and len(self._d) >= self.warmup


# #@HTM
@dataclass
class HTMEvent:
    z: torch.Tensor  # [D]
    t_s: int
    t_e: int
    S: float
    n: int
    tokens: Optional[torch.Tensor] = None  # [C, D], optional token-preserving ablation


# #@HTM
@dataclass
class HTMSampleState:
    """Per-sample online write state (one video stream)."""

    h: Optional[torch.Tensor] = None  # [1, 1, H] when present
    c_hat: Optional[torch.Tensor] = None  # [D] prediction for current clip
    open_event: Optional[HTMEvent] = None
    events: List[HTMEvent] = field(default_factory=list)
    recent: Deque[torch.Tensor] = field(default_factory=deque)  # each [C, D]
    recent_t: Deque[int] = field(default_factory=deque)
    t: int = 0
    age_counter: int = 0


# #@HTM
class HTMMemoryController:
    """Online HTM write/consolidation producing a unified token-budget memory."""

    def __init__(
        self,
        token_budget_L: int,
        recent_window_W: int = 4,
        alpha: float = 1.0,
        beta: float = 0.0,
        gamma: float = 0.01,
        quantile: float = 0.9,  # #@HTM locked default
        event_representation: str = "pooled",
        memory_layout: str = "legacy",
        group_tokens: bool = False,
    ):
        if event_representation not in ("pooled", "tokens"):
            raise ValueError("event_representation must be pooled or tokens")
        self.event_representation = event_representation
        if memory_layout not in ("legacy", "chronological", "frames"):
            raise ValueError("Unknown HTM memory layout")
        if memory_layout == "frames" and event_representation != "tokens":
            raise ValueError("Frame baseline requires token representation")
        self.memory_layout = memory_layout
        if group_tokens and event_representation != "tokens":
            raise ValueError("Grouped decoder memory requires token events")
        self.group_tokens = bool(group_tokens)
        self.L = int(token_budget_L)
        self.W = int(recent_window_W)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.thresholds = QuantileThresholds(quantile=quantile)
        self.states: List[HTMSampleState] = []
        # Eval telemetry (survives clear_memory wiping ``states`` if snapshotted first).
        self._eval_n_steps = 0
        self._eval_n_merges = 0
        self._eval_sum_S = 0.0
        self._eval_n_S = 0
        self._eval_sum_D = 0.0
        self._eval_n_D = 0

    def reset(self, batch_size: int, reset_thresholds: bool = False) -> None:
        if reset_thresholds:
            # Evaluation questions are independent streams. Preserve the
            # configured quantile/window, but discard scores from prior videos.
            self.thresholds = QuantileThresholds(
                quantile=self.thresholds.quantile,
                maxlen=self.thresholds._s.maxlen,
                warmup=self.thresholds.warmup,
            )
        self.states = [HTMSampleState(recent=deque(maxlen=self.W), recent_t=deque(maxlen=self.W))
                       for _ in range(batch_size)]
        self._eval_n_steps = 0
        self._eval_n_merges = 0
        self._eval_sum_S = 0.0
        self._eval_n_S = 0
        self._eval_sum_D = 0.0
        self._eval_n_D = 0

    def ensure_batch(self, batch_size: int) -> None:
        if len(self.states) != batch_size:
            self.reset(batch_size)

    @torch.no_grad()
    def _priority(self, surprise: float, relevance: float, age: float) -> float:
        return self.alpha * surprise + self.beta * relevance - self.gamma * age

    def step_sample(
        self,
        state: HTMSampleState,
        c_tokens: torch.Tensor,  # [C, D]
        predictor: HTMPredictor,
        collect_threshold_stats: bool,
        step_stats: Optional[List[Dict]] = None,  # #@HTM
        retain_memory_grad: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Advance one sample by one clip. Returns (c_bar detached, pred_loss or None)."""
        c_bar = c_tokens.mean(dim=0)  # [D]
        c_bar_det = c_bar.detach()
        # Stage B QA supervision must reach the compressor through readable
        # memory. Predictor targets and discrete routing remain stop-gradient.
        c_memory = c_bar if retain_memory_grad else c_bar_det
        event_tokens = None
        if self.event_representation == "tokens":
            event_tokens = c_tokens if retain_memory_grad else c_tokens.detach()
        pred_loss: Optional[torch.Tensor] = None
        S_t = 0.0
        has_pred = state.c_hat is not None

        if state.c_hat is not None:
            # Surprise vs prediction; L_pred uses stop-grad target (design §4.3).
            s_tensor = htm_distance(state.c_hat, c_bar_det)
            S_t = float(s_tensor.detach().item())
            pred_loss = s_tensor

        D_t = float("inf")
        if state.open_event is not None:
            d_tensor = htm_distance(c_bar_det, state.open_event.z.detach())
            D_t = float(d_tensor.detach().item())

        if collect_threshold_stats and state.c_hat is not None and state.open_event is not None:
            self.thresholds.update(s=S_t, d=D_t)

        merge = False
        if self.memory_layout != "frames" and state.open_event is not None and state.c_hat is not None:
            if self.thresholds.ready():
                merge = (S_t < self.thresholds.tau_s) and (D_t < self.thresholds.tau_d)
            else:
                # Warmup: merge when both scores are below running means (lenient).
                merge = (S_t < self.thresholds.tau_s) and (D_t < self.thresholds.tau_d)

        action = "open"
        if state.open_event is None:
            state.open_event = HTMEvent(
                z=c_memory.clone(), t_s=state.t, t_e=state.t, S=S_t, n=1,
                tokens=event_tokens.clone() if event_tokens is not None else None,
            )
            action = "open"
        elif merge:
            ev = state.open_event
            n = ev.n
            ev.z = (n * ev.z + c_memory) / (n + 1)
            if event_tokens is not None:
                ev.tokens = (n * ev.tokens + event_tokens) / (n + 1)
            ev.n = n + 1
            ev.t_e = state.t
            ev.S = max(ev.S, S_t)  # max aggregate (design §11)
            action = "merge"
        else:
            state.events.append(state.open_event)
            state.open_event = HTMEvent(
                z=c_memory.clone(), t_s=state.t, t_e=state.t, S=S_t, n=1,
                tokens=event_tokens.clone() if event_tokens is not None else None,
            )
            action = "new_event"

        # Cheap running counters for QA eval (HTM_EVAL_LOG_TAU); Stage A stats use step_stats.
        self._eval_n_steps += 1
        if action == "merge":
            self._eval_n_merges += 1
        if has_pred:
            self._eval_sum_S += float(S_t)
            self._eval_n_S += 1
        if D_t != float("inf"):
            self._eval_sum_D += float(D_t)
            self._eval_n_D += 1

        # Recent window keeps raw C tokens.
        state.recent.append(c_tokens if retain_memory_grad else c_tokens.detach())
        state.recent_t.append(state.t)

        # Predictor update (grads flow into GRU / proj). Detach h for truncated BPTT-1.
        c_hat_next, h_new = predictor.forward_step(c_bar_det.unsqueeze(0), state.h)
        state.h = h_new.detach()
        state.c_hat = c_hat_next.squeeze(0)
        t_idx = state.t
        state.t += 1
        state.age_counter += 1

        # #@HTM — optional per-clip telemetry for Stage A stats tooling
        if step_stats is not None:
            n_events = len(state.events) + (1 if state.open_event is not None else 0)
            step_stats.append({
                "t": t_idx,
                "S_t": S_t if has_pred else None,
                "D_t": None if D_t == float("inf") else D_t,
                "merged": action == "merge",
                "action": action,
                "n_events": n_events,
                "tau_s": self.thresholds.tau_s,
                "tau_d": self.thresholds.tau_d,
            })
        return c_bar_det, pred_loss

    def snapshot_eval_write_stats(self, batch_idx: int = 0) -> Dict:
        """merge_rate / |M| / τ after a full encode (call before clear_memory)."""
        n_steps = int(self._eval_n_steps)
        n_merges = int(self._eval_n_merges)
        decidable = max(n_steps - 1, 1)
        comp: Dict = {}
        if self.states and 0 <= batch_idx < len(self.states):
            comp = self.memory_composition(self.states[batch_idx])
        return {
            "n_steps": n_steps,
            "n_merges": n_merges,
            "merge_rate": n_merges / decidable,
            "mean_S": (self._eval_sum_S / self._eval_n_S) if self._eval_n_S else None,
            "mean_D": (self._eval_sum_D / self._eval_n_D) if self._eval_n_D else None,
            "tau_s": float(self.thresholds.tau_s),
            "tau_d": float(self.thresholds.tau_d),
            "thresholds_ready": bool(self.thresholds.ready()),
            "final_M": comp.get("final_M"),
            "n_events": comp.get("n_events"),
            "token_budget_L": comp.get("token_budget_L", self.L),
        }

    # #@HTM
    def memory_composition(self, state: HTMSampleState) -> Dict:
        """Count raw recent and event tokens before and after budget eviction."""
        recent_raw_tokens = int(sum(int(c.shape[0]) for c in state.recent))
        n_events = len(state.events) + (1 if state.open_event is not None else 0)
        all_events = state.events + ([state.open_event] if state.open_event is not None else [])
        event_tokens = sum(self._event_memory(ev).shape[0] for ev in all_events)
        mem = self.build_memory_tokens(state)
        final_M = int(mem.shape[0]) if mem.numel() else 0
        return {
            "recent_raw_tokens": recent_raw_tokens,
            "event_tokens": event_tokens,
            "n_events": n_events,
            "n_recent_clips": len(state.recent),
            "final_M": final_M,
            "token_budget_L": self.L,
            "final_M_over_L": (final_M / self.L) if self.L > 0 else None,
        }

    def _event_memory(self, event: HTMEvent) -> torch.Tensor:
        if self.event_representation == "tokens":
            if event.tokens is None:
                raise RuntimeError("Token representation must be configured before writing a stream")
            return event.tokens
        return event.z.unsqueeze(0)

    def _select_memory_units(self, state: HTMSampleState) -> List[Dict]:
        """Select units once with the same eviction/order policy for values and metadata."""
        units: List[Dict] = []
        # Recent raw clips (each costs C).
        for i, clip in enumerate(state.recent):
            if self.memory_layout == "frames":
                break  # Each original frame already has one event; avoid duplication.
            units.append({
                "kind": "recent",
                "tokens": clip,  # [C, D]
                "cost": int(clip.shape[0]),
                "S": float(state.open_event.S) if state.open_event is not None else 0.0,
                "R": 0.0,
                "age": float(state.age_counter - i),
                "time": (state.recent_t[i], state.recent_t[i], 1),
            })
        # Closed events + open event, charging their actual token count.
        all_events = list(state.events)
        if state.open_event is not None:
            all_events = all_events + [state.open_event]
        for j, ev in enumerate(all_events):
            event_memory = self._event_memory(ev)
            units.append({
                "kind": "event",
                "tokens": event_memory,
                "cost": int(event_memory.shape[0]),
                "S": float(ev.S),
                "R": 0.0,
                "age": float(len(all_events) - j),
                "time": (ev.t_s, ev.t_e, 0),
            })

        total = sum(u["cost"] for u in units)
        if total > self.L:
            # Evict lowest priority until within budget. Never drop the newest recent if possible.
            while total > self.L and len(units) > 1:
                scored = []
                for idx, u in enumerate(units):
                    # Protect the most recent raw clip (last recent unit).
                    if u["kind"] == "recent" and idx == 0 and any(x["kind"] == "recent" for x in units):
                        # Prefer evicting oldest recent (front) via low priority from age.
                        pass
                    p = self._priority(u["S"], u["R"], u["age"])
                    scored.append((p, idx))
                scored.sort(key=lambda x: x[0])
                drop_idx = scored[0][1]
                total -= units[drop_idx]["cost"]
                units.pop(drop_idx)

        if not units:
            return []
        if self.memory_layout in ("chronological", "frames"):
            # Sort only after eviction so chronological ablation preserves the
            # same selected units, including overlap with recent raw frames.
            units.sort(key=lambda u: u["time"])
        return units

    def build_memory_tokens(self, state: HTMSampleState) -> torch.Tensor:
        """Flat readable memory ``[T, D]``, with event cost 1 or C, ``T ≤ L``."""
        units = self._select_memory_units(state)
        if not units:
            return torch.zeros(0, 0)
        tokens = torch.cat([u["tokens"] for u in units], dim=0)  # [T, D]
        if tokens.shape[0] > self.L:
            tokens = tokens[: self.L]
        return tokens

    def write_batch(
        self,
        context_frames: torch.Tensor,  # [B, 1, C, D]
        predictor: HTMPredictor,
        collect_threshold_stats: bool,
        retain_memory_grad: bool = False,
    ) -> Optional[torch.Tensor]:
        """Write one frame/clip for a batch. Returns mean pred loss or None."""
        # #@HTM — late-attached GRU is not covered by HF device_map="auto" and
        # stays on CPU after eval load; match compressor feature device/dtype
        # (same idea as context_embed.to(self.device) in the QViC mixin).
        p0 = next(predictor.parameters())
        if p0.device != context_frames.device or p0.dtype != context_frames.dtype:
            predictor.to(device=context_frames.device, dtype=context_frames.dtype)
        B = context_frames.shape[0]
        self.ensure_batch(B)
        losses: List[torch.Tensor] = []
        for b in range(B):
            c_tokens = context_frames[b, 0]  # [C, D]
            _, loss_b = self.step_sample(
                self.states[b], c_tokens, predictor, collect_threshold_stats,
                retain_memory_grad=retain_memory_grad,
            )
            if loss_b is not None:
                losses.append(loss_b)
        if not losses:
            return None
        return torch.stack(losses).mean()

    def materialize_context_memory(self, reference: torch.Tensor) -> torch.Tensor:
        """Build ``[B, T, 1, D]`` padded to the max length in-batch (T ≤ L)."""
        B = len(self.states)
        per: List[torch.Tensor] = []
        for st in self.states:
            tok = self.build_memory_tokens(st)  # [T, D]
            if tok.numel() == 0:
                D = reference.shape[-1]
                tok = reference.new_zeros(1, D)
            per.append(tok)
        T = max(x.shape[0] for x in per)
        T = min(T, self.L)
        D = per[0].shape[-1]
        out = reference.new_zeros(B, T, 1, D)
        for b, tok in enumerate(per):
            t = min(tok.shape[0], T)
            out[b, :t, 0] = tok[:t].to(dtype=reference.dtype, device=reference.device)
        if self.memory_layout == "frames" or self.group_tokens:
            # Preserve native frame groups for decoder newline/grid insertion.
            C = reference.shape[-2]
            if T % C:
                raise ValueError("Frame baseline needs a budget large enough for whole C-token frames")
            out = out.reshape(B, T // C, C, D)
        return out

    @torch.no_grad()
    def materialize_event_assignments(self, reference: torch.Tensor, temperature: float = 0.1, owner_bias: float = 2.0) -> torch.Tensor:
        """Membership aligned exactly with materialized memory [B, groups, C, E].

        Use every discovered event (including the open event) as a prototype.
        Each retained unit gets one membership vector repeated over its tokens;
        this preserves the temporal observation granularity even for token events.
        """
        from qvic.model.easa import soft_event_assignments

        lengths, per = [], []
        for state in self.states:
            units = self._select_memory_units(state)
            events = state.events + ([state.open_event] if state.open_event is not None else [])
            if not units or not events:
                raise RuntimeError("EASA requires a populated HTM stream")
            prototypes = torch.stack([ev.z.detach() for ev in events])
            observations = torch.stack([u["tokens"].detach().mean(dim=0) for u in units])
            owners = []
            for unit in units:
                start, end, _ = unit["time"]
                owner = next((i for i, ev in enumerate(events) if ev.t_s <= start and end <= ev.t_e), None)
                if owner is None:
                    raise RuntimeError("Retained HTM memory unit has no event owner")
                owners.append(owner)
            P = soft_event_assignments(observations, prototypes, temperature,
                                       owners=torch.tensor(owners, device=observations.device), owner_bias=owner_bias)
            counts = torch.tensor([u["cost"] for u in units], device=P.device)
            P = P.repeat_interleave(counts, dim=0)[:self.L]
            per.append(P)
            lengths.append(P.shape[0])
        T, E = max(lengths), max(p.shape[1] for p in per)
        out = torch.zeros(len(per), T, E, device=reference.device, dtype=torch.float32)
        for b, P in enumerate(per):
            out[b, :P.shape[0], :P.shape[1]] = P.to(reference.device)
        C = reference.shape[-2] if self.memory_layout == "frames" or self.group_tokens else 1
        return out.reshape(len(per), T // C, C, E)
