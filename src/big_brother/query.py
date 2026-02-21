from __future__ import annotations

from dataclasses import dataclass

from .storage import MemoryStore


@dataclass(slots=True)
class QueryAPI:
    store: MemoryStore

    def get_events(self, start_ts: float, end_ts: float, worker_id: str = "worker-1") -> list[dict[str, object]]:
        return [dict(row) for row in self.store.get_events(start_ts=start_ts, end_ts=end_ts, worker_id=worker_id)]

    def get_tool_usage(
        self, tool: str, start_ts: float, end_ts: float, worker_id: str = "worker-1"
    ) -> dict[str, float]:
        seconds = self.store.get_tool_usage_seconds(
            tool=tool, start_ts=start_ts, end_ts=end_ts, worker_id=worker_id
        )
        return {"tool_usage_seconds": seconds}

    def get_idle_ratio(self, start_ts: float, end_ts: float, worker_id: str = "worker-1") -> dict[str, float]:
        return {"idle_ratio": self.store.get_idle_ratio(start_ts=start_ts, end_ts=end_ts, worker_id=worker_id)}

    def get_episodes(
        self, start_ts: float, end_ts: float, worker_id: str = "worker-1", label: str | None = None
    ) -> list[dict[str, object]]:
        return [
            dict(row)
            for row in self.store.get_episodes(start_ts=start_ts, end_ts=end_ts, worker_id=worker_id, label=label)
        ]

    def get_search_time(self, start_ts: float, end_ts: float, worker_id: str = "worker-1") -> dict[str, float]:
        return {"search_seconds": self.store.get_search_time(start_ts=start_ts, end_ts=end_ts, worker_id=worker_id)}

