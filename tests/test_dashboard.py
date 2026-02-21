from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from big_brother.dashboard import create_app
from big_brother.schema import SubtaskEvent
from big_brother.storage import MemoryStore


def _write_test_video(path: Path, *, width: int = 64, height: int = 48, fps: int = 10, seconds: int = 1) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("failed to open VideoWriter for test video")
    total_frames = fps * seconds
    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(frame, str(i), (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()


def test_dashboard_snapshot_and_frame(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    videos_dir = tmp_path / "videos"
    run_dir = outputs_dir / "t1"
    run_dir.mkdir(parents=True)
    videos_dir.mkdir(parents=True)

    (run_dir / "events.jsonl").write_text(
        '\n'.join(
            [
                '{"event_id":"e1","t_start":0.0,"t_end":1.0,"phase":"setup","action":"measure","tool":"none","confidence":0.8,"evidence":"x"}',
                '{"event_id":"e2","t_start":1.0,"t_end":2.0,"phase":"execute","action":"nail","tool":"nail_gun","confidence":0.9,"evidence":"y"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "episodes.jsonl").write_text(
        '{"kind":"labeled","episode_id":"ep1","t_start":0.0,"t_end":2.0,"event_ids":["e1","e2"],"label":"framing_wall","confidence":0.88,"label_source":"vlm","reasoning":"pattern"}\n',
        encoding="utf-8",
    )
    (run_dir / "windows.jsonl").write_text('{"window_id":"w1"}\n', encoding="utf-8")
    _write_test_video(videos_dir / "t1.avi")
    db = MemoryStore(db_path=run_dir / "memory.db")
    db.append_event(
        SubtaskEvent.from_model_output(
            {
                "phase": "execute",
                "action": "nail",
                "tool": "nail_gun",
                "materials": ["wood"],
                "people_nearby": "0",
                "speaking": "none",
                "location_hint": "same_area",
                "confidence": 0.9,
                "evidence": "db-test",
            },
            t_start=0.0,
            t_end=2.0,
            worker_id="worker-1",
            video_id="t1",
            source_window_id="w-db",
        )
    )
    db.close()

    class _FakeNLEngine:
        def ask(self, *, api, question, default_worker_id):
            return {
                "question": question,
                "structured_query": {"query_type": "events", "start_ts": 0, "end_ts": 10},
                "result": api.get_events(start_ts=0, end_ts=10, worker_id=default_worker_id),
                "answer": "Found matching events.",
            }

    app = create_app(outputs_dir=outputs_dir, videos_dir=videos_dir, nl_engine=_FakeNLEngine())
    client = TestClient(app)

    runs = client.get("/api/runs")
    assert runs.status_code == 200
    assert "t1" in runs.json()["runs"]

    snap = client.get("/api/snapshot", params={"run": "t1", "limit": 10})
    assert snap.status_code == 200
    body = snap.json()
    assert body["event_count"] == 2
    assert len(body["labeled_episodes"]) == 1
    assert "frame_url" in body["events"][0]

    frame = client.get("/api/frame", params={"run": "t1", "t": 0.2})
    assert frame.status_code == 200
    assert frame.headers["content-type"] == "image/jpeg"
    assert len(frame.content) > 100

    q_events = client.get("/api/query", params={"run": "t1", "type": "events", "start_ts": 0, "end_ts": 10})
    assert q_events.status_code == 200
    q_events_body = q_events.json()
    assert q_events_body["query_type"] == "events"
    assert len(q_events_body["result"]) >= 1

    q_metric = client.get(
        "/api/query",
        params={"run": "t1", "type": "tool-usage", "tool": "nail_gun", "start_ts": 0, "end_ts": 10},
    )
    assert q_metric.status_code == 200
    q_metric_body = q_metric.json()
    assert q_metric_body["result"]["tool_usage_seconds"] >= 2.0

    ask = client.get("/api/ask", params={"run": "t1", "q": "what happened in the first 10 seconds?"})
    assert ask.status_code == 200
    ask_body = ask.json()
    assert ask_body["answer"] == "Found matching events."
    assert ask_body["structured_query"]["query_type"] == "events"
