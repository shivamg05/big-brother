# Video Pipeline

Put raw input videos in `videos/`.

## What is implemented

- Reads a local video file.
- Splits into windows (`window_size_s`, optional overlap).
- Samples representative frames per window.
- Computes cheap change signals (`motion`, `embedding_drift`) for gating.
- Sends windows into `WorkerMemoryPipeline` for event/episode analysis.

## Example

```python
from big_brother import GeminiExtractor, WorkerMemoryPipeline, VideoIngestConfig, analyze_video

pipeline = WorkerMemoryPipeline(extractor=GeminiExtractor(model="gemini-2.5-flash"))
summary = analyze_video(
    "videos/example.mp4",
    pipeline,
    ingest=VideoIngestConfig(window_size_s=15.0, overlap_s=2.0, frames_per_window=3),
)
pipeline.close()
print(summary)
```

## CLI

After `uv sync`, you can process all videos in `videos/`:

```bash
uv run big-brother --videos-dir videos --extractor gemini --window-size 15 --overlap 2 --output-dir outputs
```

Or a single file:

```bash
uv run big-brother --videos-dir videos --video t1.mp4 --extractor gemini --output-dir outputs --requests-per-minute 60 --episode-labeler gemini

# Free-tier friendly rate limit (Gemini often limited to 5 RPM)
uv run big-brother --videos-dir videos --video sample.mp4 --extractor gemini --requests-per-minute 4 --max-retries 8

# Enable larger-action inference on closed episodes
uv run big-brother --videos-dir videos --video sample.mp4 --extractor gemini --episode-labeler gemini
```

## Streaming + Saved Artifacts

During runs, JSON lines are streamed to stdout (`[window]`, `[event]`, `[episode]`).

Files are written incrementally to:

- `outputs/<video_stem>/windows.jsonl`
- `outputs/<video_stem>/events.jsonl`
- `outputs/<video_stem>/episodes.jsonl`
- `outputs/<video_stem>/summary.json`

Use `--no-stream` to disable live stdout streaming while still saving files.

`episodes.jsonl` includes `kind:"labeled"` records when sequence-level episode labeling runs.

You can monitor runs in real-time with the dashboard in `docs/dashboard.md`.
