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
    HANDOFF = "handoff"
    TEST = "test"
    PREPARE = "prepare"
    VERIFY = "verify"
    IDLE = "idle"
    BREAK = "break"
    CLEANUP = "cleanup"
    UNKNOWN = "unknown"


class Action(StrEnum):
    CARRY = "carry"
    MOVE = "move"
    TRANSPORT = "transport"
    MEASURE = "measure"
    MARK = "mark"
    LAYOUT = "layout"
    CUT = "cut"
    RIP_CUT = "rip_cut"
    CROSS_CUT = "cross_cut"
    MITER_CUT = "miter_cut"
    BEVEL_CUT = "bevel_cut"
    DRILL = "drill"
    PRE_DRILL = "pre_drill"
    BORE = "bore"
    SCREW = "screw"
    UNSCREW = "unscrew"
    NAIL = "nail"
    STAPLE = "staple"
    ALIGN = "align"
    LEVEL = "level"
    PLUMB = "plumb"
    SQUARE = "square"
    CLAMP = "clamp"
    LIFT = "lift"
    LOWER = "lower"
    PUSH = "push"
    PULL = "pull"
    FASTEN = "fasten"
    BOLT = "bolt"
    WELD = "weld"
    GLUE = "glue"
    SEAL = "seal"
    CAULK = "caulk"
    SAND = "sand"
    GRIND = "grind"
    CHISEL = "chisel"
    PLANE = "plane"
    ROUTE = "route"
    MIX = "mix"
    POUR = "pour"
    SPRAY = "spray"
    PAINT = "paint"
    PRIME = "prime"
    BRUSH = "brush"
    ROLL = "roll"
    TROWEL = "trowel"
    SHOVEL = "shovel"
    DIG = "dig"
    TEST = "test"
    SCAN = "scan"
    READ_PLAN = "read_plan"
    INSPECT = "inspect"
    COMMUNICATE = "communicate"
    SIGNAL = "signal"
    HANDOFF = "handoff"
    WALK = "walk"
    CLIMB = "climb"
    KNEEL = "kneel"
    CROUCH = "crouch"
    WAIT = "wait"
    CLEAN = "clean"
    SWEEP = "sweep"
    ORGANIZE = "organize"
    COLLECT_DEBRIS = "collect_debris"
    IDLE = "idle"
    OTHER = "other"
    UNKNOWN = "unknown"


class Tool(StrEnum):
    NAIL_GUN = "nail_gun"
    DRILL = "drill"
    IMPACT_DRIVER = "impact_driver"
    SCREWDRIVER = "screwdriver"
    SAW = "saw"
    CIRCULAR_SAW = "circular_saw"
    MITER_SAW = "miter_saw"
    JIGSAW = "jigsaw"
    RECIPROCATING_SAW = "reciprocating_saw"
    TABLE_SAW = "table_saw"
    TRACK_SAW = "track_saw"
    OSCILLATING_TOOL = "oscillating_tool"
    HAMMER = "hammer"
    MALLET = "mallet"
    WRENCH = "wrench"
    SOCKET_WRENCH = "socket_wrench"
    RATCHET = "ratchet"
    PLIERS = "pliers"
    CLAMP = "clamp"
    VISE_GRIP = "vise_grip"
    CHISEL = "chisel"
    HAND_PLANE = "hand_plane"
    ROUTER = "router"
    ANGLE_GRINDER = "angle_grinder"
    SANDER = "sander"
    LEVEL = "level"
    LASER_LEVEL = "laser_level"
    SQUARE = "square"
    CHALK_LINE = "chalk_line"
    TAPE_MEASURE = "tape_measure"
    RULER = "ruler"
    SPEED_SQUARE = "speed_square"
    STUD_FINDER = "stud_finder"
    UTILITY_KNIFE = "utility_knife"
    BOX_CUTTER = "box_cutter"
    STAPLE_GUN = "staple_gun"
    CAULK_GUN = "caulk_gun"
    PAINT_BRUSH = "paint_brush"
    PAINT_ROLLER = "paint_roller"
    SPRAY_GUN = "spray_gun"
    TROWEL = "trowel"
    GROUT_FLOAT = "grout_float"
    SHOVEL = "shovel"
    WHEELBARROW = "wheelbarrow"
    CLEANING_BRUSH = "cleaning_brush"
    LADDER = "ladder"
    SCAFFOLD = "scaffold"
    MIXER = "mixer"
    WELDER = "welder"
    TORCH = "torch"
    MULTIMETER = "multimeter"
    WIRE_STRIPPER = "wire_stripper"
    CRIMPER = "crimper"
    PIPE_CUTTER = "pipe_cutter"
    HEAT_GUN = "heat_gun"
    AIR_COMPRESSOR = "air_compressor"
    GENERATOR = "generator"
    VACUUM = "vacuum"
    BROOM = "broom"
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
    reasoning: str = ""
    label_source: str = "heuristic"
    label_version: str = "v1"
    label_updated_at: str = field(default_factory=utcnow_iso)
    status: str = "open"
    episode_id: str = field(default_factory=lambda: str(uuid4()))
    worker_id: str = "worker-1"
    created_at: str = field(default_factory=utcnow_iso)

    def __post_init__(self) -> None:
        self.confidence = truncate_confidence(max(0.0, min(1.0, self.confidence)))
