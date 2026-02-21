from pathlib import Path
import json

import cv2
import numpy as np

from big_brother.pipeline import WorkerMemoryPipeline
from big_brother.runner import analyze_video
from big_brother.video import VideoIngestConfig, iter_video_windows


def _write_test_video(path: Path, *, width: int = 64, height: int = 48, fps: int = 10, seconds: int = 2) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("failed to open VideoWriter for test video")
    total_frames = fps * seconds
    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        x = (i * 3) % width
        cv2.rectangle(frame, (x, 10), (min(width - 1, x + 12), 30), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def test_iter_video_windows_splits_and_samples(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.avi"
    _write_test_video(video_path, seconds=2, fps=10)

    windows = list(
        iter_video_windows(
            video_path,
            config=VideoIngestConfig(window_size_s=1.0, overlap_s=0.0, frames_per_window=3),
        )
    )

    assert len(windows) == 2
    assert all(len(w.sampled_frames) == 3 for w in windows)
    first_frame = windows[0].sampled_frames[0]
    assert isinstance(first_frame["jpeg_bytes"], bytes)
    assert first_frame["mime_type"] == "image/jpeg"


def test_analyze_video_runs_end_to_end(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.avi"
    _write_test_video(video_path, seconds=2, fps=10)
    pipeline = WorkerMemoryPipeline()
    summary = analyze_video(
        video_path,
        pipeline,
        ingest=VideoIngestConfig(window_size_s=1.0, overlap_s=0.0, frames_per_window=3),
    )
    events = pipeline.query.get_events(0.0, 10.0)
    pipeline.close()

    assert summary.windows_processed == 2
    assert summary.events_created >= 1
    assert len(events) >= 1


def test_analyze_video_writes_json_artifacts_and_streams(tmp_path: Path, capsys) -> None:
    video_path = tmp_path / "sample.avi"
    output_dir = tmp_path / "outputs"
    _write_test_video(video_path, seconds=2, fps=10)
    pipeline = WorkerMemoryPipeline()

    summary = analyze_video(
        video_path,
        pipeline,
        ingest=VideoIngestConfig(window_size_s=1.0, overlap_s=0.0, frames_per_window=3),
        output_dir=output_dir,
        stream=True,
    )
    pipeline.close()
    out = capsys.readouterr().out

    run_dir = output_dir / video_path.stem
    events_path = run_dir / "events.jsonl"
    episodes_path = run_dir / "episodes.jsonl"
    windows_path = run_dir / "windows.jsonl"
    summary_path = run_dir / "summary.json"

    assert summary.windows_processed == 2
    assert events_path.exists()
    assert episodes_path.exists()
    assert windows_path.exists()
    assert summary_path.exists()
    assert "[event]" in out
    assert "[window]" in out
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["windows_processed"] == 2
