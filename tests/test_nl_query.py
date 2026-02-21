import json

from big_brother.nl_query import GeminiNLQueryEngine
from big_brother.query import QueryAPI
from big_brother.schema import SubtaskEvent
from big_brother.storage import MemoryStore


class _Resp:
    def __init__(self, text: str) -> None:
        self.text = text


class _Models:
    def __init__(self) -> None:
        self.calls = 0

    def generate_content(self, *, model: str, contents, config):
        self.calls += 1
        if self.calls == 1:
            return _Resp(
                json.dumps(
                    {
                        "query_type": "tool-usage",
                        "start_ts": 0,
                        "end_ts": 100,
                        "worker_id": "worker-1",
                        "tool": "nail_gun",
                        "label": None,
                        "limit": 50,
                    }
                )
            )
        return _Resp("Nail gun was used for 10.0 seconds in the selected interval.")


class _Client:
    def __init__(self) -> None:
        self.models = _Models()


def test_nl_query_engine_round_trip() -> None:
    store = MemoryStore()
    store.append_event(
        SubtaskEvent.from_model_output(
            {
                "phase": "execute",
                "action": "nail",
                "tool": "nail_gun",
                "materials": ["wood"],
                "people_nearby": "0",
                "speaking": "none",
                "location_hint": "same_area",
                "confidence": 0.9,
                "evidence": "test",
            },
            t_start=0.0,
            t_end=10.0,
            worker_id="worker-1",
            video_id="v1",
            source_window_id="w1",
        )
    )
    api = QueryAPI(store)
    engine = GeminiNLQueryEngine(client=_Client())
    out = engine.ask(api=api, question="How long was nail gun used?")
    store.close()

    assert out["structured_query"]["query_type"] == "tool-usage"
    assert out["result"]["tool_usage_seconds"] == 10.0
    assert "Nail gun" in out["answer"]

