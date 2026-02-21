from big_brother.query import QueryAPI
from big_brother.schema import SubtaskEvent
from big_brother.storage import MemoryStore


def _event(t_start: float, t_end: float, phase: str, tool: str) -> SubtaskEvent:
    return SubtaskEvent.from_model_output(
        {
            "phase": phase,
            "action": "idle" if phase == "idle" else "drill",
            "tool": tool,
            "materials": ["wood"],
            "people_nearby": "0",
            "speaking": "none",
            "location_hint": "same_area",
            "confidence": 0.9,
            "evidence": "test",
        },
        t_start=t_start,
        t_end=t_end,
        worker_id="w1",
        video_id="v1",
        source_window_id=f"w-{t_start}",
    )


def test_query_metrics() -> None:
    store = MemoryStore()
    api = QueryAPI(store)
    store.append_event(_event(0, 10, "execute", "drill"))
    store.append_event(_event(10, 20, "idle", "none"))
    store.append_event(_event(20, 35, "search", "none"))

    drill = api.get_tool_usage("drill", 0, 40, worker_id="w1")
    idle = api.get_idle_ratio(0, 40, worker_id="w1")
    search = api.get_search_time(0, 40, worker_id="w1")
    assert drill["tool_usage_seconds"] == 10.0
    assert abs(idle["idle_ratio"] - (10 / 35)) < 1e-9
    assert search["search_seconds"] == 15.0

