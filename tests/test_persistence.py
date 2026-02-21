from big_brother.pipeline import WindowInput, WorkerMemoryPipeline
from big_brother.storage import MemoryStore


def test_pipeline_persists_to_sqlite_file(tmp_path) -> None:
    db_path = tmp_path / "memory.db"
    store = MemoryStore(db_path=db_path)
    pipeline = WorkerMemoryPipeline(store=store)

    windows = [
        WindowInput(
            window_id="w1",
            t_start=0.0,
            t_end=15.0,
            sampled_frames=[{"motion": 0.5, "tool_signal": 0.8}],
            motion=0.5,
            embedding_drift=0.5,
            audio_spike=0.0,
        ),
        WindowInput(
            window_id="w2",
            t_start=15.0,
            t_end=30.0,
            sampled_frames=[{"motion": 0.6, "tool_signal": 0.2}],
            motion=0.6,
            embedding_drift=0.4,
            audio_spike=0.0,
        ),
    ]
    for w in windows:
        pipeline.process_window_detailed(w)
    pipeline.close()

    reopened = MemoryStore(db_path=db_path)
    events = reopened.get_events(start_ts=0.0, end_ts=60.0)
    episodes = reopened.get_episodes(start_ts=0.0, end_ts=60.0)
    reopened.close()

    assert len(events) >= 1
    assert len(episodes) >= 1
