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
        return _Resp(
            "<reasoning>Matched tool usage in one event spanning 10s.</reasoning>"
            "<ans>Nail gun was used for 10.0 seconds in the selected interval.</ans>"
        )


class _Client:
    def __init__(self) -> None:
        self.models = _Models()


class _ParserOnlyModels:
    def __init__(self) -> None:
        self.calls = 0

    def generate_content(self, *, model: str, contents, config):
        self.calls += 1
        return _Resp(
            json.dumps(
                {
                    "query_type": "tool-usage",
                    "start_ts": 0,
                    "end_ts": 100,
                    "worker_id": "worker-1",
                    "tool": "nail gun",
                    "label": None,
                    "limit": 50,
                }
            )
        )


class _ParserOnlyClient:
    def __init__(self) -> None:
        self.models = _ParserOnlyModels()


class _ParserNullToolModels:
    def generate_content(self, *, model: str, contents, config):
        return _Resp(
            json.dumps(
                {
                    "query_type": "tool-usage",
                    "start_ts": 0,
                    "end_ts": 100,
                    "worker_id": "worker-1",
                    "tool": None,
                    "label": None,
                    "limit": 50,
                }
            )
        )


class _ParserNullToolClient:
    def __init__(self) -> None:
        self.models = _ParserNullToolModels()


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
    assert "Matched tool usage" in out["reasoning_trace"]


def test_nl_query_normalizes_nail_gun_alias() -> None:
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

    assert out["structured_query"]["tool"] == "nail_gun"
    assert out["result"]["tool_usage_seconds"] == 10.0


def test_nl_query_infers_where_and_purpose_from_events() -> None:
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
            t_start=10.0,
            t_end=30.0,
            worker_id="worker-1",
            video_id="v1",
            source_window_id="w1",
        )
    )
    api = QueryAPI(store)
    engine = GeminiNLQueryEngine(client=_ParserOnlyClient(), use_deterministic_answers=True)
    out = engine.ask(api=api, question="where is he using the nail gun and for what purpose")
    store.close()

    assert out["structured_query"]["query_type"] == "events"
    assert out["structured_query"]["tool"] == "nail_gun"
    assert "same_area" in out["answer"]
    assert "fastening/framing" in out["answer"]


def test_nl_query_walking_duration_answer() -> None:
    store = MemoryStore()
    store.append_event(
        SubtaskEvent.from_model_output(
            {
                "phase": "travel",
                "action": "walk",
                "tool": "none",
                "materials": ["unknown"],
                "people_nearby": "0",
                "speaking": "none",
                "location_hint": "same_area",
                "confidence": 0.8,
                "evidence": "walk",
            },
            t_start=0.0,
            t_end=20.0,
            worker_id="worker-1",
            video_id="v1",
            source_window_id="w1",
        )
    )
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
                "evidence": "work",
            },
            t_start=20.0,
            t_end=100.0,
            worker_id="worker-1",
            video_id="v1",
            source_window_id="w2",
        )
    )
    api = QueryAPI(store)
    engine = GeminiNLQueryEngine(client=_ParserOnlyClient(), use_deterministic_answers=True)
    out = engine.ask(api=api, question="how long is the worker walking for?")
    store.close()

    assert "20.0s" in out["answer"]
    assert "20.0%" in out["answer"]


def test_nl_query_tool_alias_with_punctuation() -> None:
    store = MemoryStore()
    store.append_event(
        SubtaskEvent.from_model_output(
            {
                "phase": "setup",
                "action": "measure",
                "tool": "tape_measure",
                "materials": ["wood"],
                "people_nearby": "0",
                "speaking": "none",
                "location_hint": "same_area",
                "confidence": 0.9,
                "evidence": "measure",
            },
            t_start=0.0,
            t_end=10.0,
            worker_id="worker-1",
            video_id="v1",
            source_window_id="w1",
        )
    )
    api = QueryAPI(store)
    engine = GeminiNLQueryEngine(client=_ParserNullToolClient(), use_deterministic_answers=True)
    out = engine.ask(api=api, question="How long did the worker use a tape measure, and why?")
    store.close()

    assert out["structured_query"]["tool"] == "tape_measure"


def test_nl_query_tools_inventory_question() -> None:
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
    store.append_event(
        SubtaskEvent.from_model_output(
            {
                "phase": "setup",
                "action": "measure",
                "tool": "tape_measure",
                "materials": ["wood"],
                "people_nearby": "0",
                "speaking": "none",
                "location_hint": "same_area",
                "confidence": 0.9,
                "evidence": "test",
            },
            t_start=10.0,
            t_end=20.0,
            worker_id="worker-1",
            video_id="v1",
            source_window_id="w2",
        )
    )
    api = QueryAPI(store)
    engine = GeminiNLQueryEngine(client=_ParserNullToolClient(), use_deterministic_answers=True)
    out = engine.ask(api=api, question="what are the tools that are being used")
    store.close()

    assert out["structured_query"]["query_type"] == "events"
    assert "nail_gun" in out["answer"]
    assert "tape_measure" in out["answer"]


def test_nl_query_marking_purpose_is_action_grounded() -> None:
    store = MemoryStore()
    store.append_event(
        SubtaskEvent.from_model_output(
            {
                "phase": "prepare",
                "action": "mark",
                "tool": "speed_square",
                "materials": ["wood"],
                "people_nearby": "0",
                "speaking": "none",
                "location_hint": "same_area",
                "confidence": 0.9,
                "evidence": "mark line",
            },
            t_start=0.0,
            t_end=10.0,
            worker_id="worker-1",
            video_id="v1",
            source_window_id="w1",
        )
    )
    store.append_event(
        SubtaskEvent.from_model_output(
            {
                "phase": "prepare",
                "action": "measure",
                "tool": "tape_measure",
                "materials": ["wood"],
                "people_nearby": "0",
                "speaking": "none",
                "location_hint": "same_area",
                "confidence": 0.9,
                "evidence": "measure after mark",
            },
            t_start=10.0,
            t_end=20.0,
            worker_id="worker-1",
            video_id="v1",
            source_window_id="w2",
        )
    )
    store.append_event(
        SubtaskEvent.from_model_output(
            {
                "phase": "execute",
                "action": "cut",
                "tool": "circular_saw",
                "materials": ["wood"],
                "people_nearby": "0",
                "speaking": "none",
                "location_hint": "same_area",
                "confidence": 0.9,
                "evidence": "cut after measure",
            },
            t_start=20.0,
            t_end=30.0,
            worker_id="worker-1",
            video_id="v1",
            source_window_id="w3",
        )
    )
    api = QueryAPI(store)
    engine = GeminiNLQueryEngine(client=_ParserNullToolClient(), use_deterministic_answers=True)
    out = engine.ask(api=api, question="what was the worker marking and for what goal/purpose")
    store.close()

    assert out["structured_query"]["action"] == "mark"
    assert "action `mark` appears in 1 events (~10.0s)" in out["answer"]
    assert "layout/measurement" in out["answer"] or "cutting material" in out["answer"]


def test_nl_query_overrides_generic_worker_id_with_default() -> None:
    store = MemoryStore()
    store.append_event(
        SubtaskEvent.from_model_output(
            {
                "phase": "prepare",
                "action": "mark",
                "tool": "speed_square",
                "materials": ["wood"],
                "people_nearby": "0",
                "speaking": "none",
                "location_hint": "same_area",
                "confidence": 0.9,
                "evidence": "mark line",
            },
            t_start=0.0,
            t_end=10.0,
            worker_id="juan",
            video_id="juan",
            source_window_id="w1",
        )
    )
    api = QueryAPI(store)
    engine = GeminiNLQueryEngine(client=_ParserNullToolClient(), use_deterministic_answers=True)
    out = engine.ask(api=api, question="how long has the worker spent marking stuff", default_worker_id="juan")
    store.close()

    assert out["structured_query"]["worker_id"] == "juan"
