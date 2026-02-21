from big_brother.gating import GatingConfig
from big_brother.pipeline import WindowInput, WorkerMemoryPipeline


def test_pipeline_extends_previous_event_when_stable() -> None:
    pipeline = WorkerMemoryPipeline(gating_config=GatingConfig(stable_window_limit=0))
    w1 = WindowInput(
        window_id="w1",
        t_start=0,
        t_end=15,
        sampled_frames=[{"motion": 0.5, "tool_signal": 0.8}],
        motion=0.5,
        embedding_drift=0.5,
        audio_spike=0.5,
    )
    w2 = WindowInput(
        window_id="w2",
        t_start=15,
        t_end=30,
        sampled_frames=[{"motion": 0.01, "tool_signal": 0.01}],
        motion=0.01,
        embedding_drift=0.01,
        audio_spike=0.01,
    )
    e1 = pipeline.process_window(w1)
    e2 = pipeline.process_window(w2)
    assert e1 is not None
    assert e2 is None
    rows = pipeline.query.get_events(0, 60)
    assert len(rows) == 1
    assert rows[0]["t_end"] == 30.0
    pipeline.close()

