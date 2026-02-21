from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GatingConfig:
    motion_threshold: float = 0.2
    embedding_drift_threshold: float = 0.15
    audio_spike_threshold: float = 0.25
    stable_window_limit: int = 2


@dataclass(slots=True)
class GateDecision:
    should_call_vlm: bool
    reason: str
    stable_windows: int


class GatingEngine:
    def __init__(self, config: GatingConfig | None = None) -> None:
        self.config = config or GatingConfig()
        self._stable_windows = 0

    def decide(self, *, motion: float, embedding_drift: float, audio_spike: float) -> GateDecision:
        signal_changed = (
            motion >= self.config.motion_threshold
            or embedding_drift >= self.config.embedding_drift_threshold
            or audio_spike >= self.config.audio_spike_threshold
        )
        if signal_changed:
            self._stable_windows = 0
            return GateDecision(should_call_vlm=True, reason="change_detected", stable_windows=0)

        self._stable_windows += 1
        if self._stable_windows <= self.config.stable_window_limit:
            return GateDecision(
                should_call_vlm=True,
                reason="periodic_refresh",
                stable_windows=self._stable_windows,
            )
        return GateDecision(
            should_call_vlm=False,
            reason="stable_extend_previous",
            stable_windows=self._stable_windows,
        )
