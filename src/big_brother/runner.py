from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .output import JsonStreamWriter
from .pipeline import WorkerMemoryPipeline
from .video import VideoIngestConfig, iter_video_windows


@dataclass(slots=True)
class AnalysisSummary:
    video_path: str
    windows_processed: int
    events_created: int
    events_extended: int


def analyze_video(
    video_path: str | Path,
    pipeline: WorkerMemoryPipeline,
    *,
    ingest: VideoIngestConfig | None = None,
    output_dir: str | Path | None = None,
    stream: bool = True,
) -> AnalysisSummary:
    video_path = Path(video_path)
    windows = 0
    created = 0
    extended = 0
    writer: JsonStreamWriter | None = None
    if output_dir is not None:
        writer = JsonStreamWriter(Path(output_dir), video_path.stem, stream_stdout=stream)

    for window in iter_video_windows(video_path, config=ingest):
        windows += 1
        result = pipeline.process_window_detailed(window)
        if result.event is None:
            extended += 1
        else:
            created += 1
        if writer is not None:
            writer.write_window(
                {
                    "window_id": result.window_id,
                    "t_start": result.t_start,
                    "t_end": result.t_end,
                    "decision_reason": result.decision_reason,
                    "extended_event_id": result.extended_event_id,
                    "event_id": result.event.event_id if result.event else None,
                    "closed_episode_count": len(result.closed_episodes),
                }
            )
            if result.event is not None:
                writer.write_event(result.event)
            if result.open_episode is not None:
                writer.write_episode(result.open_episode, kind="open")
            for closed in result.closed_episodes:
                writer.write_episode(closed, kind="closed")

    final_episode = pipeline.finalize_current_episode()
    if writer is not None and final_episode is not None:
        writer.write_episode(final_episode, kind="final_closed")

    summary = AnalysisSummary(
        video_path=str(video_path),
        windows_processed=windows,
        events_created=created,
        events_extended=extended,
    )
    if writer is not None:
        writer.write_summary(asdict(summary))
        if stream:
            print(f"[output_dir] {writer.run_dir}", flush=True)
    return summary
