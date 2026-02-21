from dataclasses import dataclass

from big_brother.labeler import EpisodeLabeler
from big_brother.pipeline import WindowInput, WorkerMemoryPipeline
from big_brother.schema import Episode, SubtaskEvent


@dataclass
class _FakeExtractor:
    calls: int = 0

    def extract(
        self,
        *,
        sampled_frames,
        state,
        t_start,
        t_end,
        worker_id,
        video_id,
        source_window_id,
    ) -> SubtaskEvent:
        self.calls += 1
        location_hint = "new_area" if self.calls == 2 else "same_area"
        return SubtaskEvent.from_model_output(
            {
                "phase": "execute",
                "action": "nail",
                "tool": "nail_gun",
                "materials": ["wood"],
                "people_nearby": "0",
                "speaking": "none",
                "location_hint": location_hint,
                "confidence": 0.9,
                "evidence": "test",
            },
            t_start=t_start,
            t_end=t_end,
            worker_id=worker_id,
            video_id=video_id,
            source_window_id=source_window_id,
        )


class _FakeLabeler(EpisodeLabeler):
    def label_episode(self, episode: Episode, events: list[SubtaskEvent]) -> Episode:
        episode.label = "framing_wall"
        episode.reasoning = f"{len(events)} event sequence with nail_gun usage"
        episode.label_source = "vlm"
        episode.label_version = "test-v1"
        return episode


def test_pipeline_applies_labeler_to_closed_episode() -> None:
    pipeline = WorkerMemoryPipeline(extractor=_FakeExtractor(), episode_labeler=_FakeLabeler())
    w1 = WindowInput(
        window_id="w1",
        t_start=0.0,
        t_end=15.0,
        sampled_frames=[{}],
        motion=0.5,
        embedding_drift=0.5,
        audio_spike=0.0,
    )
    w2 = WindowInput(
        window_id="w2",
        t_start=15.0,
        t_end=30.0,
        sampled_frames=[{}],
        motion=0.5,
        embedding_drift=0.5,
        audio_spike=0.0,
    )
    r1 = pipeline.process_window_detailed(w1)
    r2 = pipeline.process_window_detailed(w2)
    assert r1.event is not None
    assert len(r2.labeled_episodes) == 1
    assert r2.labeled_episodes[0].label == "framing_wall"
    assert r2.labeled_episodes[0].label_source == "vlm"
    pipeline.close()

