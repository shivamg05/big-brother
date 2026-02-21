from __future__ import annotations

import argparse
from pathlib import Path

from .extractor import GeminiExtractor, HeuristicExtractor
from .pipeline import WorkerMemoryPipeline
from .runner import analyze_video
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
        "--output-dir",
        default="outputs",
        help="Directory where streaming JSON memory artifacts are written.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable live stdout JSON streaming.",
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

    pipeline = WorkerMemoryPipeline(extractor=extractor)
    try:
        if args.video:
            paths = [videos_dir / args.video]
        else:
            paths = sorted(
                p for p in videos_dir.glob("*") if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
            )

        if not paths:
            print(f"No videos found in {videos_dir}")
            return 1

        for path in paths:
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
    finally:
        pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
