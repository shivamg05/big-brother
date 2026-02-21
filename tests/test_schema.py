from big_brother.schema import Action, LocationHint, Phase, SubtaskEvent, Tool


def test_schema_coerces_unknown_enum_values() -> None:
    payload = {
        "phase": "invalid",
        "action": "nail",
        "tool": "not-real",
        "materials": [],
        "people_nearby": "4",
        "speaking": "loud",
        "location_hint": "moon",
        "confidence": 1.5,
        "evidence": "raw",
    }
    event = SubtaskEvent.from_model_output(
        payload,
        t_start=0.0,
        t_end=10.0,
        worker_id="w1",
        video_id="v1",
        source_window_id="win-1",
    )
    assert event.phase == Phase.UNKNOWN
    assert event.action == Action.NAIL
    assert event.tool == Tool.UNKNOWN
    assert event.location_hint == LocationHint.UNKNOWN
    assert event.confidence == 1.0
    assert event.materials == ["unknown"]

