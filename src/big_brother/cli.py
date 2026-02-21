from __future__ import annotations

import argparse
from pathlib import Path

from .extractor import GeminiExtractor, HeuristicExtractor
from .labeler import GeminiEpisodeLabeler, HeuristicEpisodeLabeler, NoopEpisodeLabeler
from .pipeline import WorkerMemoryPipeline
from .runner import analyze_video
from .storage import MemoryStore
from .video import VideoIngestConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze videos into queryable worker memory.")
    parser.add_argument("--videos-dir", default="videos", help="Directory containing input videos.")
    parser.add_argument("--video", default=None, help="Single video filename in videos-dir to process.")
    parser.add_argument("--window-size", type=float, default=15.0, help="Window size in seconds.")
    parser.add_argument("--overlap", type=float, default=0.0, help="Overlap between windows in seconds.")
    parser.add_argument("--frames-per-window", type=int, default=3, help="Sampled frames per window.")
    parser.add_argument(
        "--requests-per-minute",
        type=int,
        default=4,
        help="Gemini request cap to stay under quota (free tier is typically 5 RPM).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Max retries for Gemini 429/RESOURCE_EXHAUSTED errors.",
    )
    parser.add_argument(
        "--extractor",
        choices=["gemini", "heuristic"],
        default="gemini",
        help="Extractor backend.",
    )
    parser.add_argument(
        "--episode-labeler",
        choices=["gemini", "heuristic", "off"],
        default="gemini",
        help="Larger-action labeling backend for closed episodes.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where streaming JSON memory artifacts are written.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable live stdout JSON streaming.",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Persistent SQLite DB path. If omitted, uses outputs/<video_stem>/memory.db.",
    )
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    ingest_cfg = VideoIngestConfig(
        window_size_s=args.window_size,
        overlap_s=args.overlap,
        frames_per_window=args.frames_per_window,
    )

    if args.extractor == "gemini":
        extractor = GeminiExtractor(
            model="gemini-2.5-flash",
            requests_per_minute=args.requests_per_minute,
            max_retries=args.max_retries,
        )
    else:
        extractor = HeuristicExtractor()

    if args.episode_labeler == "gemini":
        episode_labeler = GeminiEpisodeLabeler(
            model="gemini-2.5-flash",
            requests_per_minute=args.requests_per_minute,
            max_retries=args.max_retries,
        )
    elif args.episode_labeler == "heuristic":
        episode_labeler = HeuristicEpisodeLabeler()
    else:
        episode_labeler = NoopEpisodeLabeler()

    if args.video:
        paths = [videos_dir / args.video]
    else:
        paths = sorted(
            p for p in videos_dir.glob("*") if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
        )

    if not paths:
        print(f"No videos found in {videos_dir}")
        return 1
    if args.db_path and len(paths) > 1:
        print("--db-path can only be used with a single --video run.")
        return 1

    for path in paths:
        if args.db_path:
            db_path = Path(args.db_path)
        else:
            db_path = Path(args.output_dir) / path.stem / "memory.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        store = MemoryStore(db_path=db_path)
        pipeline = WorkerMemoryPipeline(extractor=extractor, episode_labeler=episode_labeler, store=store)
        try:
            pipeline.config.video_id = path.stem
            summary = analyze_video(
                path,
                pipeline,
                ingest=ingest_cfg,
                output_dir=args.output_dir,
                stream=not args.no_stream,
            )
            print(
                f"{path.name}: windows={summary.windows_processed}, "
                f"events_created={summary.events_created}, events_extended={summary.events_extended}"
            )
            print(f"{path.name}: db={db_path}")
        finally:
            pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
