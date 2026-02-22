from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import shutil


TARGET_KINDS = {"closed", "labeled", "final_closed"}
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


@dataclass(slots=True)
class RunMergeResult:
    run: str
    merged_groups: int
    deleted_episodes: int
    metadata_rows_removed: int
    metadata_rows_updated: int


def _json_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(v) for v in raw]
    if not isinstance(raw, str):
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(v) for v in parsed]


def _merge_unique(left: list[str], right: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in left + right:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _touching(prev_end: float, next_start: float, tol: float) -> bool:
    return abs(float(prev_end) - float(next_start)) <= tol


def _combine_reasoning(values: list[str]) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return " | ".join(out)


def _fetch_closed_episode_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            episode_id, worker_id, t_start, t_end, event_ids, dominant_phase, tools_used,
            zone_id, label, confidence, reasoning, label_source, label_version, label_updated_at,
            status, created_at
        FROM episodes
        WHERE status = 'closed'
        ORDER BY worker_id ASC, t_start ASC, t_end ASC, created_at ASC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def _merge_closed_rows(
    rows: list[dict[str, Any]], *, touch_tolerance: float
) -> tuple[dict[str, dict[str, Any]], set[str], int]:
    updates: dict[str, dict[str, Any]] = {}
    deleted_ids: set[str] = set()
    merged_groups = 0

    current: dict[str, Any] | None = None
    current_members: list[str] = []

    def finalize_group() -> None:
        nonlocal merged_groups, current, current_members
        if current is None:
            return
        if len(current_members) > 1:
            merged_groups += 1
            keep_id = str(current_members[0])
            updates[keep_id] = current.copy()
            for to_delete in current_members[1:]:
                deleted_ids.add(str(to_delete))
        current = None
        current_members = []

    for row in rows:
        if current is None:
            current = row.copy()
            current_members = [str(row["episode_id"])]
            current["event_ids"] = _json_list(row.get("event_ids"))
            current["tools_used"] = _json_list(row.get("tools_used"))
            current["confidence"] = float(row.get("confidence", 0.0))
            current["reasoning"] = str(row.get("reasoning", "") or "")
            continue

        same_label = str(current.get("label", "")) == str(row.get("label", ""))
        same_worker = str(current.get("worker_id", "")) == str(row.get("worker_id", ""))
        contiguous = _touching(float(current.get("t_end", 0.0)), float(row.get("t_start", 0.0)), touch_tolerance)
        if same_label and same_worker and contiguous:
            current["t_end"] = float(row.get("t_end", current["t_end"]))
            current["event_ids"] = _merge_unique(current["event_ids"], _json_list(row.get("event_ids")))
            current["tools_used"] = _merge_unique(current["tools_used"], _json_list(row.get("tools_used")))
            current["confidence"] = max(float(current["confidence"]), float(row.get("confidence", 0.0)))
            current["reasoning"] = _combine_reasoning([str(current.get("reasoning", "")), str(row.get("reasoning", ""))])
            current["label_updated_at"] = max(
                str(current.get("label_updated_at", "")),
                str(row.get("label_updated_at", "")),
            )
            current_members.append(str(row["episode_id"]))
            continue

        finalize_group()
        current = row.copy()
        current_members = [str(row["episode_id"])]
        current["event_ids"] = _json_list(row.get("event_ids"))
        current["tools_used"] = _json_list(row.get("tools_used"))
        current["confidence"] = float(row.get("confidence", 0.0))
        current["reasoning"] = str(row.get("reasoning", "") or "")

    finalize_group()
    return updates, deleted_ids, merged_groups


def _apply_db_updates(conn: sqlite3.Connection, updates: dict[str, dict[str, Any]], deleted_ids: set[str]) -> None:
    for keep_id, merged in updates.items():
        conn.execute(
            """
            UPDATE episodes
            SET
                t_start = ?, t_end = ?, event_ids = ?, dominant_phase = ?, tools_used = ?,
                zone_id = ?, label = ?, confidence = ?, reasoning = ?, label_source = ?,
                label_version = ?, label_updated_at = ?, status = ?, worker_id = ?, created_at = ?
            WHERE episode_id = ?
            """,
            (
                float(merged["t_start"]),
                float(merged["t_end"]),
                json.dumps(merged["event_ids"]),
                str(merged["dominant_phase"]),
                json.dumps(merged["tools_used"]),
                str(merged["zone_id"]),
                str(merged["label"]),
                float(merged["confidence"]),
                str(merged["reasoning"]),
                str(merged["label_source"]),
                str(merged["label_version"]),
                str(merged["label_updated_at"]),
                str(merged["status"]),
                str(merged["worker_id"]),
                str(merged["created_at"]),
                keep_id,
            ),
        )

    if deleted_ids:
        placeholders = ",".join("?" for _ in deleted_ids)
        conn.execute(f"DELETE FROM episodes WHERE episode_id IN ({placeholders})", tuple(sorted(deleted_ids)))


def _events_fingerprint(conn: sqlite3.Connection) -> tuple[int, float, float]:
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS n,
            COALESCE(SUM(t_start), 0.0) AS sum_start,
            COALESCE(SUM(t_end), 0.0) AS sum_end
        FROM events
        """
    ).fetchone()
    return (int(row[0]), float(row[1]), float(row[2]))


def _update_episodes_jsonl(
    path: Path, *, updates: dict[str, dict[str, Any]], deleted_ids: set[str]
) -> tuple[int, int]:
    if not path.exists():
        return (0, 0)

    rows_out: list[dict[str, Any]] = []
    removed = 0
    updated = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        episode_id = str(row.get("episode_id", ""))
        kind = str(row.get("kind", ""))
        if episode_id in deleted_ids:
            removed += 1
            continue
        if episode_id in updates and kind in TARGET_KINDS:
            merged = updates[episode_id]
            row["t_start"] = float(merged["t_start"])
            row["t_end"] = float(merged["t_end"])
            row["event_ids"] = list(merged["event_ids"])
            row["dominant_phase"] = str(merged["dominant_phase"])
            row["tools_used"] = list(merged["tools_used"])
            row["zone_id"] = str(merged["zone_id"])
            row["label"] = str(merged["label"])
            row["confidence"] = float(merged["confidence"])
            row["reasoning"] = str(merged["reasoning"])
            row["label_source"] = str(merged["label_source"])
            row["label_version"] = str(merged["label_version"])
            row["label_updated_at"] = str(merged["label_updated_at"])
            row["status"] = str(merged["status"])
            row["worker_id"] = str(merged["worker_id"])
            row["created_at"] = str(merged["created_at"])
            updated += 1
        rows_out.append(row)

    content = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows_out)
    path.write_text(content, encoding="utf-8")
    return (removed, updated)


def merge_run(run_dir: Path, *, touch_tolerance: float = 1e-6, dry_run: bool = False) -> RunMergeResult:
    db_path = run_dir / "memory.db"
    if not db_path.exists():
        return RunMergeResult(run=run_dir.name, merged_groups=0, deleted_episodes=0, metadata_rows_removed=0, metadata_rows_updated=0)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = _fetch_closed_episode_rows(conn)
        updates, deleted_ids, merged_groups = _merge_closed_rows(rows, touch_tolerance=touch_tolerance)
        removed = 0
        updated = 0
        if not dry_run and (updates or deleted_ids):
            # Backups for safe rollback outside SQL transaction scope.
            db_backup = run_dir / "memory.db.pre_merge.bak"
            if not db_backup.exists():
                shutil.copy2(db_path, db_backup)
            episodes_path = run_dir / "episodes.jsonl"
            episodes_backup = run_dir / "episodes.pre_merge.bak.jsonl"
            if episodes_path.exists() and not episodes_backup.exists():
                shutil.copy2(episodes_path, episodes_backup)

            before_events = _events_fingerprint(conn)
            conn.execute("BEGIN IMMEDIATE")
            _apply_db_updates(conn, updates, deleted_ids)
            after_events = _events_fingerprint(conn)
            if before_events != after_events:
                conn.rollback()
                raise RuntimeError(
                    f"Safety check failed for {run_dir.name}: events table changed "
                    f"(before={before_events}, after={after_events})."
                )
            conn.commit()
            removed, updated = _update_episodes_jsonl(run_dir / "episodes.jsonl", updates=updates, deleted_ids=deleted_ids)
    finally:
        conn.close()

    return RunMergeResult(
        run=run_dir.name,
        merged_groups=merged_groups,
        deleted_episodes=len(deleted_ids),
        metadata_rows_removed=removed,
        metadata_rows_updated=updated,
    )


def _resolve_runs(outputs_dir: Path, runs: list[str] | None) -> list[Path]:
    if runs:
        return [outputs_dir / run for run in runs]
    if not outputs_dir.exists():
        return []
    return sorted([p for p in outputs_dir.iterdir() if p.is_dir()])


def _has_matching_video(videos_dir: Path, run_name: str) -> bool:
    for ext in VIDEO_EXTENSIONS:
        if (videos_dir / f"{run_name}{ext}").exists():
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge back-to-back closed episodes with identical labels and touching boundaries "
            "in both outputs/<run>/memory.db and outputs/<run>/episodes.jsonl."
        )
    )
    parser.add_argument("--outputs-dir", default="outputs", help="Directory containing run subfolders.")
    parser.add_argument("--videos-dir", default="videos", help="Directory containing source videos.")
    parser.add_argument("--run", action="append", dest="runs", help="Run name to process (repeatable).")
    parser.add_argument(
        "--touch-tolerance",
        type=float,
        default=1e-6,
        help="Max absolute gap (seconds) to consider episode boundaries touching.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report merges without modifying files.")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    videos_dir = Path(args.videos_dir)
    run_dirs = _resolve_runs(outputs_dir, args.runs)
    if not run_dirs:
        print(f"No runs found in {outputs_dir}")
        return 1

    total_groups = 0
    total_deleted = 0
    for run_dir in run_dirs:
        if not _has_matching_video(videos_dir, run_dir.name):
            print(f"{run_dir.name}: skipped (no matching video in {videos_dir})")
            continue
        result = merge_run(run_dir, touch_tolerance=float(args.touch_tolerance), dry_run=bool(args.dry_run))
        total_groups += result.merged_groups
        total_deleted += result.deleted_episodes
        print(
            f"{result.run}: merged_groups={result.merged_groups}, "
            f"deleted_episodes={result.deleted_episodes}, "
            f"metadata_removed={result.metadata_rows_removed}, metadata_updated={result.metadata_rows_updated}"
        )

    mode = "DRY-RUN" if args.dry_run else "APPLIED"
    print(f"[{mode}] total_merged_groups={total_groups}, total_deleted_episodes={total_deleted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
