import json
import sqlite3
from pathlib import Path

from big_brother.merge_episodes import merge_run
from big_brother.schema import Episode
from big_brother.storage import MemoryStore


def _episode(
    *,
    episode_id: str,
    worker_id: str,
    t_start: float,
    t_end: float,
    label: str,
    event_ids: list[str],
    reasoning: str,
) -> Episode:
    return Episode(
        episode_id=episode_id,
        worker_id=worker_id,
        t_start=t_start,
        t_end=t_end,
        event_ids=event_ids,
        dominant_phase="setup",
        tools_used=["tape_measure"],
        zone_id="z1",
        label=label,
        confidence=0.8,
        reasoning=reasoning,
        label_source="heuristic",
        label_version="heuristic-v1",
        status="closed",
    )


def test_merge_run_merges_back_to_back_same_label_in_db_and_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "r1"
    run_dir.mkdir(parents=True)
    db_path = run_dir / "memory.db"
    store = MemoryStore(db_path=db_path)
    try:
        store.upsert_episode(
            _episode(
                episode_id="ep-a",
                worker_id="w1",
                t_start=0.0,
                t_end=10.0,
                label="framing_wall",
                event_ids=["e1"],
                reasoning="phase 1",
            )
        )
        store.upsert_episode(
            _episode(
                episode_id="ep-b",
                worker_id="w1",
                t_start=10.0,
                t_end=20.0,
                label="framing_wall",
                event_ids=["e2"],
                reasoning="phase 2",
            )
        )
        store.upsert_episode(
            _episode(
                episode_id="ep-c",
                worker_id="w1",
                t_start=20.0,
                t_end=30.0,
                label="tile_setting",
                event_ids=["e3"],
                reasoning="different label",
            )
        )
    finally:
        store.close()

    episodes_jsonl = run_dir / "episodes.jsonl"
    episodes_jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "kind": "labeled",
                        "episode_id": "ep-a",
                        "worker_id": "w1",
                        "t_start": 0.0,
                        "t_end": 10.0,
                        "event_ids": ["e1"],
                        "dominant_phase": "setup",
                        "tools_used": ["tape_measure"],
                        "zone_id": "z1",
                        "label": "framing_wall",
                        "confidence": 0.8,
                        "reasoning": "phase 1",
                        "label_source": "heuristic",
                        "label_version": "heuristic-v1",
                        "label_updated_at": "",
                        "status": "closed",
                        "created_at": "",
                    }
                ),
                json.dumps(
                    {
                        "kind": "labeled",
                        "episode_id": "ep-b",
                        "worker_id": "w1",
                        "t_start": 10.0,
                        "t_end": 20.0,
                        "event_ids": ["e2"],
                        "dominant_phase": "setup",
                        "tools_used": ["tape_measure"],
                        "zone_id": "z1",
                        "label": "framing_wall",
                        "confidence": 0.8,
                        "reasoning": "phase 2",
                        "label_source": "heuristic",
                        "label_version": "heuristic-v1",
                        "label_updated_at": "",
                        "status": "closed",
                        "created_at": "",
                    }
                ),
                json.dumps(
                    {
                        "kind": "labeled",
                        "episode_id": "ep-c",
                        "worker_id": "w1",
                        "t_start": 20.0,
                        "t_end": 30.0,
                        "event_ids": ["e3"],
                        "dominant_phase": "setup",
                        "tools_used": ["tape_measure"],
                        "zone_id": "z1",
                        "label": "tile_setting",
                        "confidence": 0.8,
                        "reasoning": "different label",
                        "label_source": "heuristic",
                        "label_version": "heuristic-v1",
                        "label_updated_at": "",
                        "status": "closed",
                        "created_at": "",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = merge_run(run_dir)
    assert result.merged_groups == 1
    assert result.deleted_episodes == 1

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT episode_id, t_start, t_end, event_ids, label, reasoning FROM episodes ORDER BY t_start ASC"
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 2
    first = rows[0]
    assert first[0] == "ep-a"
    assert first[1] == 0.0
    assert first[2] == 20.0
    assert json.loads(first[3]) == ["e1", "e2"]
    assert first[4] == "framing_wall"
    assert "phase 1" in first[5]
    assert "phase 2" in first[5]

    metadata_rows = [json.loads(line) for line in episodes_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["episode_id"] for row in metadata_rows]
    assert "ep-b" not in ids
    keeper = next(row for row in metadata_rows if row["episode_id"] == "ep-a")
    assert keeper["t_end"] == 20.0
    assert keeper["event_ids"] == ["e1", "e2"]
