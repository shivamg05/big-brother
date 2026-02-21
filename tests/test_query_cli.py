from big_brother.query import QueryAPI
from big_brother.query_cli import run_query
from big_brother.schema import SubtaskEvent
from big_brother.storage import MemoryStore


def _event(t_start: float, t_end: float, phase: str, action: str, tool: str) -> SubtaskEvent:
    return SubtaskEvent.from_model_output(
        {
            "phase": phase,
            "action": action,
            "tool": tool,
            "materials": ["wood"],
            "people_nearby": "0",
            "speaking": "none",
            "location_hint": "same_area",
            "confidence": 0.8,
            "evidence": "test",
        },
        t_start=t_start,
        t_end=t_end,
        worker_id="worker-1",
        video_id="v1",
        source_window_id=f"w-{t_start}",
    )


def test_run_query_events_and_metrics() -> None:
    store = MemoryStore()
    store.append_event(_event(0, 10, "execute", "drill", "drill"))
    store.append_event(_event(10, 20, "idle", "idle", "none"))
    api = QueryAPI(store)

    events = run_query(api, query_type="events", start_ts=0, end_ts=100, worker_id="worker-1", limit=10)
    tool_usage = run_query(
        api, query_type="tool-usage", start_ts=0, end_ts=100, worker_id="worker-1", tool="drill"
    )
    idle_ratio = run_query(api, query_type="idle-ratio", start_ts=0, end_ts=100, worker_id="worker-1")
    store.close()

    assert isinstance(events, list)
    assert len(events) == 2
    assert tool_usage["tool_usage_seconds"] == 10.0
    assert idle_ratio["idle_ratio"] == 0.5
