from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from .schema import Episode, SubtaskEvent


class MemoryStore:
    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                worker_id TEXT NOT NULL,
                video_id TEXT NOT NULL,
                source_window_id TEXT NOT NULL,
                t_start REAL NOT NULL,
                t_end REAL NOT NULL,
                phase TEXT NOT NULL,
                action TEXT NOT NULL,
                tool TEXT NOT NULL,
                materials TEXT NOT NULL,
                people_nearby TEXT NOT NULL,
                speaking TEXT NOT NULL,
                location_hint TEXT NOT NULL,
                confidence REAL NOT NULL,
                evidence TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_worker_t_start ON events(worker_id, t_start);
            CREATE INDEX IF NOT EXISTS idx_events_tool_t_start ON events(tool, t_start);
            CREATE INDEX IF NOT EXISTS idx_events_phase_t_start ON events(phase, t_start);
            CREATE INDEX IF NOT EXISTS idx_events_video_window ON events(video_id, source_window_id);

            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                worker_id TEXT NOT NULL,
                t_start REAL NOT NULL,
                t_end REAL NOT NULL,
                event_ids TEXT NOT NULL,
                dominant_phase TEXT NOT NULL,
                tools_used TEXT NOT NULL,
                zone_id TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_worker_t_start ON episodes(worker_id, t_start);
            CREATE INDEX IF NOT EXISTS idx_episodes_label_t_start ON episodes(label, t_start);
            CREATE INDEX IF NOT EXISTS idx_episodes_status_t_end ON episodes(status, t_end);
            """
        )
        self.conn.commit()

    def append_event(self, event: SubtaskEvent) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO events (
                event_id, worker_id, video_id, source_window_id, t_start, t_end, phase, action, tool,
                materials, people_nearby, speaking, location_hint, confidence, evidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.worker_id,
                event.video_id,
                event.source_window_id,
                event.t_start,
                event.t_end,
                event.phase.value,
                event.action.value,
                event.tool.value,
                json.dumps(event.materials),
                event.people_nearby.value,
                event.speaking.value,
                event.location_hint.value,
                event.confidence,
                event.evidence,
            ),
        )
        self.conn.commit()

    def upsert_episode(self, episode: Episode) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO episodes (
                episode_id, worker_id, t_start, t_end, event_ids, dominant_phase, tools_used,
                zone_id, label, confidence, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode.episode_id,
                episode.worker_id,
                episode.t_start,
                episode.t_end,
                json.dumps(episode.event_ids),
                episode.dominant_phase,
                json.dumps(episode.tools_used),
                episode.zone_id,
                episode.label,
                episode.confidence,
                episode.status,
                episode.created_at,
            ),
        )
        self.conn.commit()

    def extend_event_end(self, event_id: str, new_t_end: float) -> None:
        self.conn.execute("UPDATE events SET t_end = ? WHERE event_id = ?", (new_t_end, event_id))
        self.conn.commit()

    def get_events(self, *, start_ts: float, end_ts: float, worker_id: str = "worker-1") -> list[sqlite3.Row]:
        rows = self.conn.execute(
            """
            SELECT * FROM events
            WHERE worker_id = ? AND t_end >= ? AND t_start <= ?
            ORDER BY t_start ASC
            """,
            (worker_id, start_ts, end_ts),
        ).fetchall()
        return rows

    def get_tool_usage_seconds(
        self, *, tool: str, start_ts: float, end_ts: float, worker_id: str = "worker-1"
    ) -> float:
        row = self.conn.execute(
            """
            SELECT COALESCE(SUM(t_end - t_start), 0.0) AS seconds
            FROM events
            WHERE worker_id = ? AND tool = ? AND t_end >= ? AND t_start <= ?
            """,
            (worker_id, tool, start_ts, end_ts),
        ).fetchone()
        return float(row["seconds"])

    def get_idle_ratio(self, *, start_ts: float, end_ts: float, worker_id: str = "worker-1") -> float:
        row = self.conn.execute(
            """
            SELECT
                COALESCE(SUM(CASE WHEN phase = 'idle' THEN t_end - t_start ELSE 0 END), 0.0) AS idle_seconds,
                COALESCE(SUM(t_end - t_start), 0.0) AS total_seconds
            FROM events
            WHERE worker_id = ? AND t_end >= ? AND t_start <= ?
            """,
            (worker_id, start_ts, end_ts),
        ).fetchone()
        total = float(row["total_seconds"])
        if total == 0:
            return 0.0
        return float(row["idle_seconds"]) / total

    def get_search_time(self, *, start_ts: float, end_ts: float, worker_id: str = "worker-1") -> float:
        row = self.conn.execute(
            """
            SELECT COALESCE(SUM(t_end - t_start), 0.0) AS seconds
            FROM events
            WHERE worker_id = ? AND phase = 'search' AND t_end >= ? AND t_start <= ?
            """,
            (worker_id, start_ts, end_ts),
        ).fetchone()
        return float(row["seconds"])

    def get_episodes(
        self, *, start_ts: float, end_ts: float, worker_id: str = "worker-1", label: str | None = None
    ) -> list[sqlite3.Row]:
        if label:
            return self.conn.execute(
                """
                SELECT * FROM episodes
                WHERE worker_id = ? AND label = ? AND t_end >= ? AND t_start <= ?
                ORDER BY t_start ASC
                """,
                (worker_id, label, start_ts, end_ts),
            ).fetchall()
        return self.conn.execute(
            """
            SELECT * FROM episodes
            WHERE worker_id = ? AND t_end >= ? AND t_start <= ?
            ORDER BY t_start ASC
            """,
            (worker_id, start_ts, end_ts),
        ).fetchall()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

