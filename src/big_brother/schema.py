from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import math
from typing import Any
from uuid import uuid4


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def truncate_confidence(value: float, places: int = 4) -> float:
    scale = 10**places
    return math.trunc(value * scale) / scale


class Phase(StrEnum):
    TRAVEL = "travel"
    SEARCH = "search"
    SETUP = "setup"
    EXECUTE = "execute"
    INSPECT = "inspect"
    COMMUNICATE = "communicate"
    IDLE = "idle"
    CLEANUP = "cleanup"
    UNKNOWN = "unknown"


class Action(StrEnum):
    CARRY = "carry"
    MEASURE = "measure"
    CUT = "cut"
    DRILL = "drill"
    NAIL = "nail"
    ALIGN = "align"
    LIFT = "lift"
    FASTEN = "fasten"
    INSPECT = "inspect"
    WALK = "walk"
    IDLE = "idle"
    OTHER = "other"
    UNKNOWN = "unknown"


class Tool(StrEnum):
    NAIL_GUN = "nail_gun"
    DRILL = "drill"
    SAW = "saw"
    HAMMER = "hammer"
    TAPE_MEASURE = "tape_measure"
    NONE = "none"
    UNKNOWN = "unknown"


class PeopleNearby(StrEnum):
    ZERO = "0"
    ONE = "1"
    TWO_PLUS = "2+"
    UNKNOWN = "unknown"


class Speaking(StrEnum):
    NONE = "none"
    BRIEF = "brief"
    SUSTAINED = "sustained"
    UNKNOWN = "unknown"


class LocationHint(StrEnum):
    SAME_AREA = "same_area"
    NEW_AREA = "new_area"
    UNKNOWN = "unknown"


def _coerce_enum(value: str, enum_type: type[StrEnum], default: StrEnum) -> StrEnum:
    try:
        return enum_type(value)
    except ValueError:
        return default


@dataclass(slots=True)
class SubtaskEvent:
    t_start: float
    t_end: float
    phase: Phase
    action: Action
    tool: Tool
    materials: list[str]
    people_nearby: PeopleNearby
    speaking: Speaking
    location_hint: LocationHint
    confidence: float
    evidence: str
    event_id: str = field(default_factory=lambda: str(uuid4()))
    worker_id: str = "worker-1"
    video_id: str = "video-1"
    source_window_id: str = "window-unknown"

    def __post_init__(self) -> None:
        if self.t_end < self.t_start:
            raise ValueError("t_end must be >= t_start")
        self.confidence = truncate_confidence(max(0.0, min(1.0, self.confidence)))
        self.materials = self.materials or ["unknown"]

    @classmethod
    def from_model_output(
        cls,
        payload: dict[str, Any],
        *,
        t_start: float,
        t_end: float,
        worker_id: str,
        video_id: str,
        source_window_id: str,
    ) -> "SubtaskEvent":
        return cls(
            t_start=t_start,
            t_end=t_end,
            phase=_coerce_enum(payload.get("phase", "unknown"), Phase, Phase.UNKNOWN),  # type: ignore[arg-type]
            action=_coerce_enum(payload.get("action", "unknown"), Action, Action.UNKNOWN),  # type: ignore[arg-type]
            tool=_coerce_enum(payload.get("tool", "unknown"), Tool, Tool.UNKNOWN),  # type: ignore[arg-type]
            materials=[str(x) for x in payload.get("materials", ["unknown"])],
            people_nearby=_coerce_enum(  # type: ignore[arg-type]
                payload.get("people_nearby", "unknown"), PeopleNearby, PeopleNearby.UNKNOWN
            ),
            speaking=_coerce_enum(payload.get("speaking", "unknown"), Speaking, Speaking.UNKNOWN),  # type: ignore[arg-type]
            location_hint=_coerce_enum(  # type: ignore[arg-type]
                payload.get("location_hint", "unknown"), LocationHint, LocationHint.UNKNOWN
            ),
            confidence=float(payload.get("confidence", 0.0)),
            evidence=str(payload.get("evidence", "")),
            worker_id=worker_id,
            video_id=video_id,
            source_window_id=source_window_id,
        )


@dataclass(slots=True)
class Episode:
    t_start: float
    t_end: float
    event_ids: list[str]
    dominant_phase: str
    tools_used: list[str]
    zone_id: str
    label: str = "unknown"
    confidence: float = 0.0
    status: str = "open"
    episode_id: str = field(default_factory=lambda: str(uuid4()))
    worker_id: str = "worker-1"
    created_at: str = field(default_factory=utcnow_iso)

    def __post_init__(self) -> None:
        self.confidence = truncate_confidence(max(0.0, min(1.0, self.confidence)))
