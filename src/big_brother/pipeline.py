from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .episode import EpisodeBoundaryConfig, EpisodeBuilder
from .extractor import HeuristicExtractor, SubtaskExtractor
from .gating import GatingConfig, GatingEngine
from .labeler import EpisodeLabeler, HeuristicEpisodeLabeler
from .query import QueryAPI
from .schema import Episode, SubtaskEvent
from .state import RollingState
from .storage import MemoryStore


@dataclass(slots=True)
class PipelineConfig:
    worker_id: str = "worker-1"
    video_id: str = "video-1"


@dataclass(slots=True)
class WindowInput:
    window_id: str
    t_start: float
    t_end: float
    sampled_frames: list[dict[str, Any]]
    motion: float
    embedding_drift: float
    audio_spike: float


@dataclass(slots=True)
class ProcessResult:
    window_id: str
    t_start: float
    t_end: float
    decision_reason: str
    event: SubtaskEvent | None = None
    extended_event_id: str | None = None
    open_episode: Episode | None = None
    closed_episodes: list[Episode] = field(default_factory=list)
    labeled_episodes: list[Episode] = field(default_factory=list)


class WorkerMemoryPipeline:
    def __init__(
        self,
        *,
        config: PipelineConfig | None = None,
        store: MemoryStore | None = None,
        extractor: SubtaskExtractor | None = None,
        episode_labeler: EpisodeLabeler | None = None,
        gating_config: GatingConfig | None = None,
        episode_config: EpisodeBoundaryConfig | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.store = store or MemoryStore()
        self.extractor = extractor or HeuristicExtractor()
        self.episode_labeler = episode_labeler or HeuristicEpisodeLabeler()
        self.gating = GatingEngine(gating_config)
        self.state = RollingState()
        self.episode_builder = EpisodeBuilder(episode_config)
        self.query = QueryAPI(self.store)
        self._last_event_id: str | None = None
        self._event_cache: dict[str, SubtaskEvent] = {}

    def process_window_detailed(self, window: WindowInput) -> ProcessResult:
        decision = self.gating.decide(
            motion=window.motion,
            embedding_drift=window.embedding_drift,
            audio_spike=window.audio_spike,
        )
        if not decision.should_call_vlm and self._last_event_id:
            self.store.extend_event_end(self._last_event_id, window.t_end)
            return ProcessResult(
                window_id=window.window_id,
                t_start=window.t_start,
                t_end=window.t_end,
                decision_reason=decision.reason,
                extended_event_id=self._last_event_id,
            )

        event = self.extractor.extract(
            sampled_frames=window.sampled_frames,
            state=self.state.as_prompt_context(),
            t_start=window.t_start,
            t_end=window.t_end,
            worker_id=self.config.worker_id,
            video_id=self.config.video_id,
            source_window_id=window.window_id,
        )
        self.store.append_event(event)
        self._event_cache[event.event_id] = event
        if len(self._event_cache) > 1000:
            oldest = next(iter(self._event_cache))
            self._event_cache.pop(oldest, None)
        self.state.update(event)
        episode_update = self.episode_builder.update(event)
        if episode_update.open_episode:
            self.store.upsert_episode(episode_update.open_episode)
        labeled: list[Episode] = []
        for closed in episode_update.closed:
            closed = self._label_episode(closed)
            self.store.upsert_episode(closed)
            labeled.append(closed)
        self._last_event_id = event.event_id
        return ProcessResult(
            window_id=window.window_id,
            t_start=window.t_start,
            t_end=window.t_end,
            decision_reason=decision.reason,
            event=event,
            open_episode=episode_update.open_episode,
            closed_episodes=episode_update.closed,
            labeled_episodes=labeled,
        )

    def process_window(self, window: WindowInput) -> SubtaskEvent | None:
        return self.process_window_detailed(window).event

    def finalize_current_episode(self) -> Episode | None:
        maybe_closed = self.episode_builder.close_open_episode()
        if maybe_closed:
            maybe_closed = self._label_episode(maybe_closed)
            self.store.upsert_episode(maybe_closed)
        return maybe_closed

    def _label_episode(self, episode: Episode) -> Episode:
        events: list[SubtaskEvent] = []
        for event_id in episode.event_ids:
            event = self._event_cache.get(event_id)
            if event is not None:
                events.append(event)
        return self.episode_labeler.label_episode(episode, events)

    def close(self) -> None:
        self.finalize_current_episode()
        self.store.close()
