from big_brother.episode import EpisodeBoundaryConfig, EpisodeBuilder
from big_brother.schema import SubtaskEvent


def _event(t_start: float, t_end: float, phase: str, location_hint: str = "same_area") -> SubtaskEvent:
    return SubtaskEvent.from_model_output(
        {
            "phase": phase,
            "action": "walk" if phase == "travel" else "idle",
            "tool": "none",
            "materials": ["wood"],
            "people_nearby": "0",
            "speaking": "none",
            "location_hint": location_hint,
            "confidence": 0.8,
            "evidence": "test",
        },
        t_start=t_start,
        t_end=t_end,
        worker_id="w1",
        video_id="v1",
        source_window_id=f"w-{t_start}",
    )


def test_episode_splits_on_idle_threshold() -> None:
    builder = EpisodeBuilder(EpisodeBoundaryConfig(idle_split_seconds=30.0, travel_split_seconds=30.0))
    u1 = builder.update(_event(0, 20, "setup"))
    assert u1.open_episode is not None
    assert u1.open_episode.confidence > 0.0
    u2 = builder.update(_event(20, 55, "idle"))
    assert len(u2.closed) == 1
    assert u2.closed[0].status == "closed"
    assert u2.closed[0].confidence > 0.3
