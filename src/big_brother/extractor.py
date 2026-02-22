from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

from dotenv import load_dotenv

from .schema import Action, LocationHint, PeopleNearby, Phase, Speaking, SubtaskEvent, Tool


class SubtaskExtractor(Protocol):
    def extract(
        self,
        *,
        sampled_frames: list[dict[str, Any]],
        state: dict[str, object],
        t_start: float,
        t_end: float,
        worker_id: str,
        video_id: str,
        source_window_id: str,
    ) -> SubtaskEvent: ...


@dataclass(slots=True)
class GeminiExtractor:
    """Gemini-backed extractor for structured subtask events."""

    model: str = "gemini-2.5-flash"
    api_key: str | None = None
    client: Any | None = None
    requests_per_minute: int = 4
    max_retries: int = 8
    parse_retries: int = 2
    _last_request_ts: float = 0.0
    _bystander_markers: tuple[str, ...] = (
        "another worker",
        "other worker",
        "someone else",
        "another person",
        "other person",
        "bystander",
        "coworker",
        "co-worker",
    )
    _pov_markers: tuple[str, ...] = (
        "first-person",
        "first person",
        "camera wearer",
        "wearer",
        "pov",
        "camera motion",
        "camera turns",
        "hands in frame",
        "hands visible",
    )

    def __post_init__(self) -> None:
        if self.client is not None:
            return
        load_dotenv()
        api_key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing Gemini API key. Set it in .env or as GEMINI_API_KEY/GOOGLE_API_KEY.")
        try:
            import google.genai as genai
        except ImportError as exc:  # pragma: no cover - depends on environment packages
            raise RuntimeError("google-genai is not installed. Add dependency: google-genai") from exc
        self.client = genai.Client(api_key=api_key)

    def extract(
        self,
        *,
        sampled_frames: list[dict[str, Any]],
        state: dict[str, object],
        t_start: float,
        t_end: float,
        worker_id: str,
        video_id: str,
        source_window_id: str,
    ) -> SubtaskEvent:
        if self.client is None:
            raise RuntimeError("Gemini client is not initialized")

        prompt = self._build_prompt(sampled_frames=sampled_frames, state=state, t_start=t_start, t_end=t_end)
        contents: list[Any] = [prompt]
        from google.genai import types

        for frame in sampled_frames:
            jpeg_bytes = frame.get("jpeg_bytes")
            if isinstance(jpeg_bytes, bytes) and jpeg_bytes:
                contents.append(
                    types.Part.from_bytes(
                        data=jpeg_bytes,
                        mime_type=str(frame.get("mime_type", "image/jpeg")),
                    )
                )
        payload = self._extract_payload_with_retries(contents=contents, state=state)
        payload = self._enforce_pov_consistency(payload, state=state)
        return SubtaskEvent.from_model_output(
            payload,
            t_start=t_start,
            t_end=t_end,
            worker_id=worker_id,
            video_id=video_id,
            source_window_id=source_window_id,
        )

    def _extract_payload_with_retries(self, *, contents: list[Any], state: dict[str, object]) -> dict[str, Any]:
        attempts = max(1, self.parse_retries + 1)
        last_error: Exception | None = None
        for _ in range(attempts):
            response = self._generate_content_with_backoff(contents)
            try:
                return self._parse_response_json(response)
            except RuntimeError as exc:
                last_error = exc
                continue
        return self._fallback_payload(state=state, error=str(last_error) if last_error else "parse_failed")

    def _build_prompt(
        self,
        *,
        sampled_frames: list[dict[str, Any]],
        state: dict[str, object],
        t_start: float,
        t_end: float,
    ) -> str:
        safe_frames = []
        for frame in sampled_frames:
            safe_frames.append(
                {
                    "timestamp_s": frame.get("timestamp_s"),
                    "motion": frame.get("motion"),
                    "embedding_drift": frame.get("embedding_drift"),
                    "tool_signal": frame.get("tool_signal"),
                }
            )
        frame_summary = json.dumps(safe_frames, separators=(",", ":"))
        state_summary = json.dumps(state, separators=(",", ":"))
        phase_values = "|".join(self._enum_values(Phase))
        action_values = "|".join(self._enum_values(Action))
        tool_values = "|".join(self._enum_values(Tool))
        people_values = "|".join(self._enum_values(PeopleNearby))
        speaking_values = "|".join(self._enum_values(Speaking))
        location_values = "|".join(self._enum_values(LocationHint))
        return f"""
You are extracting one atomic worker event from a short video window.
Return strictly one JSON object with only these keys:
phase, action, tool, materials, people_nearby, speaking, location_hint, confidence, evidence

Allowed enums:
- phase: {phase_values}
- action: {action_values}
- tool: {tool_values}
- people_nearby: {people_values}
- speaking: {speaking_values}
- location_hint: {location_values}

Rules:
- The target actor is the camera wearer (POV worker), not other people in frame.
- If another worker is visible using a tool, ignore that unless the camera wearer is clearly the one performing the action.
- Only assign execute/tool use when there is direct egocentric evidence (hands/body/camera-coupled motion) that the POV worker is doing it.
- If evidence is ambiguous or primarily about other people, prefer conservative outputs:
  phase in travel|search|unknown, action in walk|other|unknown, tool in none|unknown.
- Use unknown when uncertain.
- confidence must be 0..1.
- evidence must be short and concrete, and describe the POV worker cues.
- materials must be an array of distinct, short strings.

Window:
- t_start: {t_start}
- t_end: {t_end}
- sampled_frames_signals: {frame_summary}
- Images: attached as additional multimodal parts (ordered start->mid->end).
- rolling_state: {state_summary}
""".strip()

    @staticmethod
    def _enum_values(enum_type: Any) -> list[str]:
        return [e.value for e in enum_type]

    def _enforce_pov_consistency(self, payload: dict[str, Any], *, state: dict[str, object]) -> dict[str, Any]:
        evidence = str(payload.get("evidence", "")).lower()
        if not evidence:
            return payload

        mentions_bystander = any(marker in evidence for marker in self._bystander_markers)
        mentions_pov = any(marker in evidence for marker in self._pov_markers)
        if not mentions_bystander or mentions_pov:
            return payload

        payload["tool"] = "none"
        phase = str(payload.get("phase", "unknown"))
        if phase == "execute":
            last_phase = str(state.get("last_phase", "unknown"))
            payload["phase"] = last_phase if last_phase in {"travel", "search", "setup", "unknown"} else "travel"

        if str(payload.get("action", "unknown")) in {"drill", "cut", "nail", "fasten"}:
            payload["action"] = "walk" if payload.get("phase") == "travel" else "other"

        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        payload["confidence"] = min(confidence, 0.5)
        payload["evidence"] = "bystander activity observed; POV worker action unclear"
        return payload

    @staticmethod
    def _parse_response_json(response: Any) -> dict[str, Any]:
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, dict):
            return parsed
        text = getattr(response, "text", None)
        if not text:
            text = GeminiExtractor._extract_text_from_candidates(response)
        if not text:
            raise RuntimeError("Gemini response did not contain text JSON")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Gemini returned non-JSON output: {text}") from exc

    @staticmethod
    def _extract_text_from_candidates(response: Any) -> str | None:
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return None
        try:
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                parts = getattr(content, "parts", None) if content is not None else None
                if not parts:
                    continue
                for part in parts:
                    txt = getattr(part, "text", None)
                    if txt:
                        return str(txt)
        except Exception:
            return None
        return None

    @staticmethod
    def _fallback_payload(*, state: dict[str, object], error: str) -> dict[str, Any]:
        last_phase = str(state.get("last_phase", "unknown"))
        safe_phase = last_phase if last_phase in {"travel", "search", "setup", "inspect", "unknown"} else "unknown"
        return {
            "phase": safe_phase,
            "action": "other",
            "tool": "unknown",
            "materials": ["unknown"],
            "people_nearby": "unknown",
            "speaking": "unknown",
            "location_hint": str(state.get("last_location_hint", "unknown")),
            "confidence": 0.2,
            "evidence": f"fallback_due_to_parse_failure:{error}",
        }

    def _generate_content_with_backoff(self, contents: list[Any]) -> Any:
        if self.client is None:
            raise RuntimeError("Gemini client is not initialized")
        attempt = 0
        while True:
            self._respect_rate_limit()
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config={
                        "response_mime_type": "application/json",
                        "response_json_schema": self._response_json_schema(),
                        "temperature": 0.1,
                    },
                )
                self._last_request_ts = time.monotonic()
                return response
            except Exception as exc:
                self._last_request_ts = time.monotonic()
                if not self._is_rate_limit_error(exc):
                    raise
                attempt += 1
                if attempt > self.max_retries:
                    raise RuntimeError(
                        "Gemini rate-limit retries exhausted. Reduce request rate or increase window size."
                    ) from exc
                wait_s = self._retry_delay_from_error(exc) or self._min_interval_seconds()
                time.sleep(wait_s)

    def _respect_rate_limit(self) -> None:
        min_interval = self._min_interval_seconds()
        if min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    def _min_interval_seconds(self) -> float:
        if self.requests_per_minute <= 0:
            return 0.0
        return 60.0 / float(self.requests_per_minute)

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            return True
        text = str(exc).lower()
        return "429" in text or "resource_exhausted" in text or "quota exceeded" in text

    @staticmethod
    def _retry_delay_from_error(exc: Exception) -> float | None:
        text = str(exc)
        match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", text, flags=re.IGNORECASE)
        if match:
            return max(0.1, float(match.group(1)))
        match = re.search(r"'retryDelay': '([0-9]+)s'", text)
        if match:
            return max(0.1, float(match.group(1)))
        return None

    @staticmethod
    def _response_json_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "phase": {
                    "type": "string",
                    "enum": GeminiExtractor._enum_values(Phase),
                },
                "action": {
                    "type": "string",
                    "enum": GeminiExtractor._enum_values(Action),
                },
                "tool": {
                    "type": "string",
                    "enum": GeminiExtractor._enum_values(Tool),
                },
                "materials": {"type": "array", "items": {"type": "string"}},
                "people_nearby": {"type": "string", "enum": GeminiExtractor._enum_values(PeopleNearby)},
                "speaking": {"type": "string", "enum": GeminiExtractor._enum_values(Speaking)},
                "location_hint": {"type": "string", "enum": GeminiExtractor._enum_values(LocationHint)},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "evidence": {"type": "string"},
            },
            "required": [
                "phase",
                "action",
                "tool",
                "materials",
                "people_nearby",
                "speaking",
                "location_hint",
                "confidence",
                "evidence",
            ],
            "additionalProperties": False,
        }


@dataclass(slots=True)
class HeuristicExtractor:
    """A placeholder extractor for MVP wiring before a production VLM adapter."""

    def extract(
        self,
        *,
        sampled_frames: list[dict[str, Any]],
        state: dict[str, object],
        t_start: float,
        t_end: float,
        worker_id: str,
        video_id: str,
        source_window_id: str,
    ) -> SubtaskEvent:
        motion = max((f.get("motion", 0.0) for f in sampled_frames), default=0.0)
        tool_score = max((f.get("tool_signal", 0.0) for f in sampled_frames), default=0.0)
        if motion < 0.1:
            phase = "idle"
            action = "idle"
            tool = "none"
        elif tool_score > 0.7:
            phase = "execute"
            action = "drill"
            tool = "drill"
        else:
            phase = "setup"
            action = "measure"
            tool = "tape_measure"

        payload = {
            "phase": phase,
            "action": action,
            "tool": tool,
            "materials": ["wood"],
            "people_nearby": "0",
            "speaking": "none",
            "location_hint": state.get("last_location_hint", "same_area"),
            "confidence": 0.8,
            "evidence": "heuristic extractor signal",
        }
        return SubtaskEvent.from_model_output(
            payload,
            t_start=t_start,
            t_end=t_end,
            worker_id=worker_id,
            video_id=video_id,
            source_window_id=source_window_id,
        )
