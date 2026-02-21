from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .pipeline import WindowInput


@dataclass(slots=True)
class VideoIngestConfig:
    window_size_s: float = 15.0
    overlap_s: float = 0.0
    frames_per_window: int = 3


def iter_video_windows(video_path: str | Path, config: VideoIngestConfig | None = None) -> Iterator[WindowInput]:
    cfg = config or VideoIngestConfig()
    if cfg.window_size_s <= 0:
        raise ValueError("window_size_s must be > 0")
    if cfg.overlap_s < 0 or cfg.overlap_s >= cfg.window_size_s:
        raise ValueError("overlap_s must be >= 0 and < window_size_s")
    if cfg.frames_per_window < 2:
        raise ValueError("frames_per_window must be >= 2")

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            raise ValueError("Video FPS is invalid")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = frame_count / fps
        step_s = cfg.window_size_s - cfg.overlap_s

        window_idx = 0
        t_start = 0.0
        while t_start < duration_s:
            t_end = min(duration_s, t_start + cfg.window_size_s)
            sampled_frames = _sample_window_frames(cap, t_start=t_start, t_end=t_end, count=cfg.frames_per_window, fps=fps)
            motion, drift = _window_change_features(sampled_frames)
            yield WindowInput(
                window_id=f"{video_path.stem}-w{window_idx}",
                t_start=t_start,
                t_end=t_end,
                sampled_frames=sampled_frames,
                motion=motion,
                embedding_drift=drift,
                audio_spike=0.0,
            )
            window_idx += 1
            t_start += step_s
    finally:
        cap.release()


def _sample_window_frames(
    cap: cv2.VideoCapture,
    *,
    t_start: float,
    t_end: float,
    count: int,
    fps: float,
) -> list[dict[str, object]]:
    if t_end <= t_start:
        return []
    span = t_end - t_start
    max_ts = max(t_start, t_end - (1.0 / max(fps, 1.0)))
    positions = [t_start + ((max_ts - t_start) * i / (count - 1)) for i in range(count)]
    out: list[dict[str, object]] = []
    prev_gray: np.ndarray | None = None

    for ts in positions:
        frame_idx = int(max(0, round(ts * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = 0.0
        if prev_gray is not None:
            motion = float(np.mean(cv2.absdiff(gray, prev_gray)) / 255.0)
        prev_gray = gray

        hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
        hist = hist / (float(np.sum(hist)) + 1e-9)
        ok_jpg, enc = cv2.imencode(".jpg", frame)
        if not ok_jpg:
            continue
        out.append(
            {
                "timestamp_s": ts,
                "motion": motion,
                "embedding_drift": float(np.std(hist)),
                "tool_signal": 0.0,
                "jpeg_bytes": enc.tobytes(),
                "mime_type": "image/jpeg",
            }
        )

    return out


def _window_change_features(sampled_frames: list[dict[str, object]]) -> tuple[float, float]:
    if not sampled_frames:
        return 0.0, 0.0
    motions = [float(f.get("motion", 0.0)) for f in sampled_frames]
    drifts = [float(f.get("embedding_drift", 0.0)) for f in sampled_frames]
    return max(motions), max(drifts)
