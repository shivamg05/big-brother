from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .schema import Episode, Phase, SubtaskEvent


@dataclass(slots=True)
class EpisodeBoundaryConfig:
    travel_split_seconds: float = 45.0
    idle_split_seconds: float = 60.0
    label_refresh_seconds: float = 180.0


@dataclass(slots=True)
class EpisodeUpdate:
    closed: list[Episode] = field(default_factory=list)
    open_episode: Episode | None = None


class EpisodeBuilder:
    def __init__(self, config: EpisodeBoundaryConfig | None = None) -> None:
        self.config = config or EpisodeBoundaryConfig()
        self._current: Episode | None = None
        self._current_event_confidences: list[float] = []
        self._idle_acc = 0.0
        self._travel_acc = 0.0
        self._last_tool = "unknown"
        self._last_phase = "unknown"

    def update(self, event: SubtaskEvent) -> EpisodeUpdate:
        closed: list[Episode] = []
        duration = max(0.0, event.t_end - event.t_start)

        if event.phase == Phase.IDLE:
            self._idle_acc += duration
        elif event.phase == Phase.TRAVEL:
            self._travel_acc += duration
        else:
            self._idle_acc = 0.0
            self._travel_acc = 0.0

        should_split = (
            self._current is None
            or self._idle_acc > self.config.idle_split_seconds
            or self._travel_acc > self.config.travel_split_seconds
            or event.location_hint.value == "new_area"
            or (
                self._last_tool not in {"unknown", event.tool.value}
                and self._last_phase not in {"unknown", event.phase.value}
                and self._last_phase != event.phase.value
            )
        )

        if should_split and self._current is not None:
            self._current.status = "closed"
            closed.append(self._finalize(self._current))
            self._current = None
            self._current_event_confidences = []

        if self._current is None:
            self._current = Episode(
                t_start=event.t_start,
                t_end=event.t_end,
                event_ids=[event.event_id],
                dominant_phase=event.phase.value,
                tools_used=[event.tool.value],
                zone_id=event.location_hint.value,
                worker_id=event.worker_id,
            )
            self._current_event_confidences = [event.confidence]
        else:
            self._current.event_ids.append(event.event_id)
            self._current.t_end = event.t_end
            if event.tool.value not in self._current.tools_used:
                self._current.tools_used.append(event.tool.value)
            self._current_event_confidences.append(event.confidence)

        self._last_tool = event.tool.value
        self._last_phase = event.phase.value
        self._current.dominant_phase = self._estimate_dominant_phase(self._current, event)
        self._current.confidence = self._estimate_episode_confidence(self._current_event_confidences, closed=False)
        return EpisodeUpdate(closed=closed, open_episode=self._current)

    def close_open_episode(self) -> Episode | None:
        if self._current is None:
            return None
        self._current.status = "closed"
        closed = self._finalize(self._current)
        self._current = None
        self._current_event_confidences = []
        return closed

    def _estimate_dominant_phase(self, episode: Episode, event: SubtaskEvent) -> str:
        if len(episode.event_ids) <= 1:
            return event.phase.value
        phases = [episode.dominant_phase, event.phase.value]
        return Counter(phases).most_common(1)[0][0]

    def _finalize(self, episode: Episode) -> Episode:
        episode.confidence = self._estimate_episode_confidence(self._current_event_confidences, closed=True)
        if episode.label == "unknown" and episode.tools_used:
            if "drill" in episode.tools_used:
                episode.label = "drilling sequence"
            elif "nail_gun" in episode.tools_used:
                episode.label = "framing sequence"
        return episode

    @staticmethod
    def _estimate_episode_confidence(event_confidences: list[float], *, closed: bool) -> float:
        if not event_confidences:
            return 0.0
        avg = sum(event_confidences) / len(event_confidences)
        count = len(event_confidences)
        maturity = min(1.0, count / 4.0)
        if closed:
            confidence = avg + min(0.15, 0.02 * count)
        else:
            confidence = avg * (0.5 + 0.5 * maturity)
        return max(0.05, min(0.98, confidence))
