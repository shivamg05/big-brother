# Video Pipeline

## What it does

For each video run, the pipeline:

- windows video frames over time
- extracts event-level activity (`events.jsonl`)
- builds/labels episodes (`episodes.jsonl`)
- persists queryable state in SQLite (`memory.db`)

## CLI

Process all videos in `videos/`:

```bash
uv run big-brother --videos-dir videos --extractor gemini --output-dir outputs
```

Process a single file:

```bash
uv run big-brother --videos-dir videos --video t1.mp4 --extractor gemini --episode-labeler gemini --output-dir outputs
```

Heuristic-only local run:

```bash
uv run big-brother --videos-dir videos --video t1.mp4 --extractor heuristic --episode-labeler heuristic --output-dir outputs
```

## Output artifacts

Written to `outputs/<video_stem>/`:

- `windows.jsonl`
- `events.jsonl`
- `episodes.jsonl`
- `summary.json`
- `memory.db`

`episodes.jsonl` contains records with `kind` values such as `open`, `closed`, `labeled`, and `final_closed`.

## Streaming

During run, stdout emits structured lines (`[window]`, `[event]`, `[episode]`).

Use `--no-stream` to disable live stdout streaming.

## Important run/video mapping

The dashboard resolves frame previews from `videos/<run>.<ext>`.

If output run is `outputs/t1`, frame previews require `videos/t1.mp4` (or another supported extension).

If video stem and run name do not match, `/api/frame` requests return `404`.
