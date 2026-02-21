from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .schema import LocationHint, SubtaskEvent


@dataclass(slots=True)
class RollingState:
    last_phase: str = "unknown"
    current_tool_active: str = "unknown"
    last_location_hint: str = "unknown"
    recent_actions: deque[str] = field(default_factory=lambda: deque(maxlen=5))

    def update(self, event: SubtaskEvent) -> None:
        self.last_phase = event.phase.value
        self.current_tool_active = event.tool.value
        self.last_location_hint = event.location_hint.value
        self.recent_actions.append(event.action.value)

    def as_prompt_context(self) -> dict[str, object]:
        return {
            "last_phase": self.last_phase,
            "current_tool_active": self.current_tool_active,
            "last_location_hint": self.last_location_hint,
            "recent_actions": list(self.recent_actions),
        }

    def location_changed(self, new_hint: LocationHint) -> bool:
        return self.last_location_hint != "unknown" and new_hint.value != self.last_location_hint

