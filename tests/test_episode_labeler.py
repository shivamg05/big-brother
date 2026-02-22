from big_brother.labeler import HeuristicEpisodeLabeler
from big_brother.schema import Episode, SubtaskEvent


def _event(*, action: str, tool: str, materials: list[str], t_start: float, t_end: float) -> SubtaskEvent:
    return SubtaskEvent.from_model_output(
        {
            "phase": "execute",
            "action": action,
            "tool": tool,
            "materials": materials,
            "people_nearby": "0",
            "speaking": "none",
            "location_hint": "same_area",
            "confidence": 0.9,
            "evidence": "test",
        },
        t_start=t_start,
        t_end=t_end,
        worker_id="w",
        video_id="v",
        source_window_id="w1",
    )


def _episode() -> Episode:
    return Episode(
        t_start=0.0,
        t_end=45.0,
        event_ids=[],
        dominant_phase="execute",
        tools_used=[],
        zone_id="same_area",
    )


def test_heuristic_labels_concrete_prep_transport() -> None:
    labeler = HeuristicEpisodeLabeler()
    events = [
        _event(action="transport", tool="shovel", materials=["concrete_mix", "sand"], t_start=0.0, t_end=15.0),
        _event(action="shovel", tool="shovel", materials=["gravel", "cement"], t_start=15.0, t_end=30.0),
        _event(action="align", tool="none", materials=["concrete_block", "rebar"], t_start=30.0, t_end=45.0),
    ]
    ep = labeler.label_episode(_episode(), events)
    assert ep.label == "concrete_prep_transport"
    assert "Concrete" in ep.reasoning


def test_heuristic_labels_masonry_blockwork_from_trowel() -> None:
    labeler = HeuristicEpisodeLabeler()
    events = [
        _event(action="trowel", tool="trowel", materials=["mortar", "concrete_block"], t_start=0.0, t_end=15.0),
        _event(action="align", tool="trowel", materials=["grout", "concrete_block"], t_start=15.0, t_end=30.0),
    ]
    ep = labeler.label_episode(_episode(), events)
    assert ep.label == "masonry_blockwork"
    assert "masonry" in ep.reasoning.lower()
