from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from dotenv import load_dotenv

from .schema import Episode, SubtaskEvent

ALLOWED_EPISODE_LABELS: tuple[str, ...] = (
    "framing_wall",
    "floor_sheathing",
    "cut_and_assemble",
    "measurement_layout",
    "material_transport",
    "inspection",
    "idle_transition",
    "concrete_prep_transport",
    "block_rebar_alignment",
    "masonry_blockwork",
    "tile_setting",
    "unknown",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EpisodeLabeler(Protocol):
    def label_episode(self, episode: Episode, events: list[SubtaskEvent]) -> Episode: ...


@dataclass(slots=True)
class NoopEpisodeLabeler:
    def label_episode(self, episode: Episode, events: list[SubtaskEvent]) -> Episode:
        return episode


@dataclass(slots=True)
class HeuristicEpisodeLabeler:
    label_version: str = "heuristic-v1"

    def label_episode(self, episode: Episode, events: list[SubtaskEvent]) -> Episode:
        if not events:
            episode.label = "unknown"
            episode.reasoning = "No events available for episode labeling."
            episode.label_source = "heuristic"
            episode.label_version = self.label_version
            episode.label_updated_at = _now_iso()
            return episode

        actions = [e.action.value for e in events]
        tools = [e.tool.value for e in events]
        materials = [m.strip().lower() for e in events for m in e.materials]
        has_concrete = any(
            kw in m for m in materials for kw in ("concrete", "cement", "gravel", "sand", "mortar", "mix")
        )
        has_block = any(("block" in m) or ("cmu" in m) for m in materials)
        has_rebar = any("rebar" in m for m in materials)
        has_tile = any("tile" in m for m in materials)

        if tools.count("nail_gun") >= 2 and ("measure" in actions or "align" in actions):
            episode.label = "framing_wall"
            episode.reasoning = "Repeated nailing with measuring/alignment pattern."
        elif tools.count("saw") >= 2 and ("nail_gun" in tools):
            episode.label = "cut_and_assemble"
            episode.reasoning = "Alternating saw cuts and nail-gun assembly."
        elif actions.count("measure") >= 2:
            episode.label = "measurement_layout"
            episode.reasoning = "Mostly repeated measurement/setup actions."
        elif has_concrete and any(a in actions for a in ("transport", "carry", "move", "walk", "shovel", "mix", "pour")):
            episode.label = "concrete_prep_transport"
            episode.reasoning = "Concrete/cement materials with transport/mix/shovel actions."
        elif "trowel" in tools and (has_block or any(x in materials for x in ("mortar", "grout"))):
            episode.label = "masonry_blockwork"
            episode.reasoning = "Trowel work with block/mortar/grout indicates masonry blockwork."
        elif "trowel" in tools and has_tile:
            episode.label = "tile_setting"
            episode.reasoning = "Trowel work with tile/grout materials indicates tile setting."
        elif (has_block or has_rebar) and any(a in actions for a in ("align", "level", "plumb", "square", "fasten")):
            episode.label = "block_rebar_alignment"
            episode.reasoning = "Block/rebar materials with alignment/placement actions."
        elif actions.count("walk") >= max(2, len(actions) // 2):
            episode.label = "material_transport"
            episode.reasoning = "Episode dominated by travel/carry behavior."
        else:
            episode.label = "unknown"
            episode.reasoning = "No strong high-level pattern detected."
        episode.label_source = "heuristic"
        episode.label_version = self.label_version
        episode.label_updated_at = _now_iso()
        return episode


@dataclass(slots=True)
class GeminiEpisodeLabeler:
    model: str = "gemini-2.5-flash"
    api_key: str | None = None
    client: Any | None = None
    requests_per_minute: int = 4
    max_retries: int = 6
    label_version: str = "gemini-v1"
    _last_request_ts: float = 0.0

    def __post_init__(self) -> None:
        if self.client is not None:
            return
        load_dotenv()
        api_key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing Gemini API key for episode labeler.")
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("google-genai is not installed. Add dependency: google-genai") from exc
        self.client = genai.Client(api_key=api_key)

    def label_episode(self, episode: Episode, events: list[SubtaskEvent]) -> Episode:
        payload = self._events_payload(events)
        prompt = (
            "Infer ONE high-level construction task label from this episode event sequence. "
            "Return JSON with keys: label, confidence, reasoning. "
            f"Allowed labels: {', '.join(ALLOWED_EPISODE_LABELS)}. "
            f"Episode events: {json.dumps(payload, separators=(',', ':'))}"
        )
        response = self._generate_with_backoff(prompt)
        text = getattr(response, "text", None)
        if not text:
            return HeuristicEpisodeLabeler().label_episode(episode, events)
        try:
            out = json.loads(text)
        except json.JSONDecodeError:
            return HeuristicEpisodeLabeler().label_episode(episode, events)
        episode.label = str(out.get("label", "unknown"))
        episode.confidence = max(0.0, min(1.0, float(out.get("confidence", episode.confidence))))
        episode.reasoning = str(out.get("reasoning", ""))
        episode.label_source = "vlm"
        episode.label_version = self.label_version
        episode.label_updated_at = _now_iso()
        return episode

    @staticmethod
    def _events_payload(events: list[SubtaskEvent]) -> list[dict[str, Any]]:
        trimmed = events[-40:]
        return [
            {
                "t_start": e.t_start,
                "t_end": e.t_end,
                "phase": e.phase.value,
                "action": e.action.value,
                "tool": e.tool.value,
                "materials": e.materials,
                "location_hint": e.location_hint.value,
                "confidence": e.confidence,
            }
            for e in trimmed
        ]

    def _generate_with_backoff(self, prompt: str) -> Any:
        if self.client is None:
            raise RuntimeError("Gemini client is not initialized")
        attempt = 0
        while True:
            self._respect_rate_limit()
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_json_schema": {
                            "type": "object",
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "enum": list(ALLOWED_EPISODE_LABELS),
                                },
                                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "reasoning": {"type": "string"},
                            },
                            "required": ["label", "confidence", "reasoning"],
                            "additionalProperties": False,
                        },
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
                    raise
                time.sleep(self._retry_delay(exc) or self._min_interval_seconds())

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
        status = getattr(exc, "status_code", None)
        text = str(exc).lower()
        return status == 429 or "resource_exhausted" in text or "quota exceeded" in text

    @staticmethod
    def _retry_delay(exc: Exception) -> float | None:
        text = str(exc)
        match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", text, flags=re.IGNORECASE)
        if match:
            return max(0.1, float(match.group(1)))
        return None
