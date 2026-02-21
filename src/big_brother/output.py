from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .schema import Episode, SubtaskEvent


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def event_to_dict(event: SubtaskEvent) -> dict[str, Any]:
    return _json_safe(asdict(event))


def episode_to_dict(episode: Episode) -> dict[str, Any]:
    return _json_safe(asdict(episode))


@dataclass(slots=True)
class JsonStreamWriter:
    base_dir: Path
    video_stem: str
    stream_stdout: bool = True
    run_dir: Path | None = None
    events_path: Path | None = None
    episodes_path: Path | None = None
    windows_path: Path | None = None
    summary_path: Path | None = None

    def __post_init__(self) -> None:
        self.run_dir = self.base_dir / self.video_stem
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.episodes_path = self.run_dir / "episodes.jsonl"
        self.windows_path = self.run_dir / "windows.jsonl"
        self.summary_path = self.run_dir / "summary.json"

    def write_window(self, payload: dict[str, Any]) -> None:
        assert self.windows_path is not None
        self._append_jsonl(self.windows_path, payload)
        if self.stream_stdout:
            print(f"[window] {json.dumps(payload, separators=(',', ':'))}", flush=True)

    def write_event(self, event: SubtaskEvent) -> None:
        assert self.events_path is not None
        payload = event_to_dict(event)
        self._append_jsonl(self.events_path, payload)
        if self.stream_stdout:
            print(f"[event] {json.dumps(payload, separators=(',', ':'))}", flush=True)

    def write_episode(self, episode: Episode, *, kind: str) -> None:
        assert self.episodes_path is not None
        payload = {"kind": kind, **episode_to_dict(episode)}
        self._append_jsonl(self.episodes_path, payload)
        if self.stream_stdout:
            print(f"[episode] {json.dumps(payload, separators=(',', ':'))}", flush=True)

    def write_summary(self, payload: dict[str, Any]) -> None:
        assert self.summary_path is not None
        self.summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if self.stream_stdout:
            print(f"[summary_file] {self.summary_path}", flush=True)

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")
            f.flush()
