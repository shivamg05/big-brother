import json

import pytest

from big_brother.extractor import GeminiExtractor


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeModels:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_content(self, *, model: str, contents: str, config: dict[str, object]) -> _FakeResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        payload = {
            "phase": "execute",
            "action": "drill",
            "tool": "drill",
            "materials": ["wood"],
            "people_nearby": "0",
            "speaking": "none",
            "location_hint": "same_area",
            "confidence": 0.88,
            "evidence": "detected active drilling posture",
        }
        return _FakeResponse(json.dumps(payload))


class _FakeClient:
    def __init__(self) -> None:
        self.models = _FakeModels()


class _RateLimitError(Exception):
    def __init__(self, message: str = "429 RESOURCE_EXHAUSTED. Please retry in 0.1s.") -> None:
        super().__init__(message)
        self.status_code = 429


class _FlakyModels:
    def __init__(self) -> None:
        self.calls = 0

    def generate_content(self, *, model: str, contents: str, config: dict[str, object]) -> _FakeResponse:
        self.calls += 1
        if self.calls == 1:
            raise _RateLimitError()
        payload = {
            "phase": "setup",
            "action": "measure",
            "tool": "tape_measure",
            "materials": ["wood"],
            "people_nearby": "0",
            "speaking": "none",
            "location_hint": "same_area",
            "confidence": 0.75,
            "evidence": "measuring setup",
        }
        return _FakeResponse(json.dumps(payload))


class _FlakyClient:
    def __init__(self) -> None:
        self.models = _FlakyModels()


class _NoTextThenJsonModels:
    def __init__(self) -> None:
        self.calls = 0

    def generate_content(self, *, model: str, contents: str, config: dict[str, object]):
        self.calls += 1
        if self.calls == 1:
            class _NoTextResponse:
                text = None

            return _NoTextResponse()
        payload = {
            "phase": "execute",
            "action": "nail",
            "tool": "nail_gun",
            "materials": ["wood"],
            "people_nearby": "0",
            "speaking": "none",
            "location_hint": "same_area",
            "confidence": 0.9,
            "evidence": "success on retry",
        }
        return _FakeResponse(json.dumps(payload))


class _NoTextThenJsonClient:
    def __init__(self) -> None:
        self.models = _NoTextThenJsonModels()


class _AlwaysNoTextModels:
    def generate_content(self, *, model: str, contents: str, config: dict[str, object]):
        class _NoTextResponse:
            text = None

        return _NoTextResponse()


class _AlwaysNoTextClient:
    def __init__(self) -> None:
        self.models = _AlwaysNoTextModels()


def test_gemini_extractor_parses_json_response() -> None:
    client = _FakeClient()
    extractor = GeminiExtractor(client=client)
    event = extractor.extract(
        sampled_frames=[{"motion": 0.8, "tool_signal": 0.9}],
        state={"last_phase": "setup"},
        t_start=0.0,
        t_end=15.0,
        worker_id="w1",
        video_id="v1",
        source_window_id="win-1",
    )
    assert event.phase.value == "execute"
    assert event.tool.value == "drill"
    assert len(client.models.calls) == 1
    assert client.models.calls[0]["model"] == "gemini-2.5-flash"
    assert client.models.calls[0]["config"]["response_mime_type"] == "application/json"
    assert "response_json_schema" in client.models.calls[0]["config"]


def test_gemini_extractor_requires_api_key_without_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr("big_brother.extractor.load_dotenv", lambda: None)
    with pytest.raises(ValueError):
        GeminiExtractor()


def test_gemini_extractor_retries_on_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr("big_brother.extractor.time.sleep", lambda x: sleep_calls.append(float(x)))
    extractor = GeminiExtractor(client=_FlakyClient(), requests_per_minute=0, max_retries=2)
    event = extractor.extract(
        sampled_frames=[{"motion": 0.2}],
        state={},
        t_start=0.0,
        t_end=15.0,
        worker_id="w1",
        video_id="v1",
        source_window_id="win-2",
    )
    assert event.phase.value == "setup"
    assert len(sleep_calls) >= 1


def test_gemini_prompt_is_pov_anchored() -> None:
    extractor = GeminiExtractor(client=_FakeClient())
    prompt = extractor._build_prompt(
        sampled_frames=[{"timestamp_s": 1.0, "motion": 0.2, "embedding_drift": 0.3, "tool_signal": 0.4}],
        state={"last_phase": "travel"},
        t_start=0.0,
        t_end=15.0,
    )
    assert "camera wearer (POV worker)" in prompt
    assert "ignore that unless the camera wearer is clearly the one performing the action" in prompt


def test_gemini_extractor_downgrades_bystander_tool_activity() -> None:
    class _BystanderModels:
        def generate_content(self, *, model: str, contents: str, config: dict[str, object]) -> _FakeResponse:
            payload = {
                "phase": "execute",
                "action": "cut",
                "tool": "saw",
                "materials": ["wood"],
                "people_nearby": "1",
                "speaking": "none",
                "location_hint": "same_area",
                "confidence": 0.81,
                "evidence": "another worker is cutting wood on a sawhorse",
            }
            return _FakeResponse(json.dumps(payload))

    class _BystanderClient:
        def __init__(self) -> None:
            self.models = _BystanderModels()

    extractor = GeminiExtractor(client=_BystanderClient())
    event = extractor.extract(
        sampled_frames=[{"motion": 0.8, "tool_signal": 0.9}],
        state={"last_phase": "travel"},
        t_start=0.0,
        t_end=15.0,
        worker_id="w1",
        video_id="v1",
        source_window_id="win-3",
    )
    assert event.phase.value == "travel"
    assert event.action.value == "walk"
    assert event.tool.value == "none"
    assert event.confidence == 0.5


def test_gemini_extractor_retries_when_text_missing() -> None:
    extractor = GeminiExtractor(client=_NoTextThenJsonClient(), requests_per_minute=0, parse_retries=2)
    event = extractor.extract(
        sampled_frames=[{"motion": 0.8}],
        state={},
        t_start=0.0,
        t_end=15.0,
        worker_id="w1",
        video_id="v1",
        source_window_id="w-retry",
    )
    assert event.tool.value == "nail_gun"
    assert event.evidence == "success on retry"


def test_gemini_extractor_falls_back_when_text_missing_persistently() -> None:
    extractor = GeminiExtractor(client=_AlwaysNoTextClient(), requests_per_minute=0, parse_retries=1)
    event = extractor.extract(
        sampled_frames=[{"motion": 0.8}],
        state={"last_phase": "travel", "last_location_hint": "same_area"},
        t_start=0.0,
        t_end=15.0,
        worker_id="w1",
        video_id="v1",
        source_window_id="w-fallback",
    )
    assert event.action.value == "other"
    assert event.tool.value == "unknown"
    assert event.confidence == 0.2
    assert "fallback_due_to_parse_failure" in event.evidence
