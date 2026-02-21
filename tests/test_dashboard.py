from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from big_brother.dashboard import create_app


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

    app = create_app(outputs_dir=outputs_dir, videos_dir=videos_dir)
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

