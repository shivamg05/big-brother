from __future__ import annotations

import argparse
import json
from typing import Any

from .query import QueryAPI
from .storage import MemoryStore


def run_query(
    api: QueryAPI,
    *,
    query_type: str,
    start_ts: float,
    end_ts: float,
    worker_id: str,
    tool: str | None = None,
    label: str | None = None,
    limit: int = 200,
) -> dict[str, Any] | list[dict[str, Any]]:
    if query_type == "events":
        rows = api.get_events(start_ts=start_ts, end_ts=end_ts, worker_id=worker_id)
        return rows[-limit:]
    if query_type == "episodes":
        rows = api.get_episodes(start_ts=start_ts, end_ts=end_ts, worker_id=worker_id, label=label)
        return rows[-limit:]
    if query_type == "tool-usage":
        if not tool:
            raise ValueError("--tool is required for tool-usage")
        return api.get_tool_usage(tool=tool, start_ts=start_ts, end_ts=end_ts, worker_id=worker_id)
    if query_type == "idle-ratio":
        return api.get_idle_ratio(start_ts=start_ts, end_ts=end_ts, worker_id=worker_id)
    if query_type == "search-time":
        return api.get_search_time(start_ts=start_ts, end_ts=end_ts, worker_id=worker_id)
    raise ValueError(f"Unsupported query type: {query_type}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Query persisted worker memory SQLite DB.")
    parser.add_argument("--db-path", required=True, help="Path to SQLite memory DB.")
    parser.add_argument(
        "--query",
        required=True,
        choices=["events", "episodes", "tool-usage", "idle-ratio", "search-time"],
        help="Query type.",
    )
    parser.add_argument("--start-ts", type=float, default=0.0, help="Start timestamp (seconds).")
    parser.add_argument("--end-ts", type=float, default=1e12, help="End timestamp (seconds).")
    parser.add_argument("--worker-id", default="worker-1", help="Worker id filter.")
    parser.add_argument("--tool", default=None, help="Tool filter for tool-usage.")
    parser.add_argument("--label", default=None, help="Episode label filter for episodes query.")
    parser.add_argument("--limit", type=int, default=200, help="Limit for events/episodes results.")
    args = parser.parse_args()

    store = MemoryStore(db_path=args.db_path)
    try:
        api = QueryAPI(store)
        result = run_query(
            api,
            query_type=args.query,
            start_ts=args.start_ts,
            end_ts=args.end_ts,
            worker_id=args.worker_id,
            tool=args.tool,
            label=args.label,
            limit=args.limit,
        )
        print(json.dumps(result, indent=2))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

