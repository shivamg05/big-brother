# Big Brother Overview

Big Brother converts long-form POV video into a persisted, queryable activity memory.

## End-to-end flow

1. Video ingestion and windowing
2. Event extraction per window
3. Episode construction + labeling
4. Persist artifacts and SQLite memory
5. Query via CLI/API/dashboard

## Core artifacts per run

`outputs/<run>/`:

- `windows.jsonl` (window processing records)
- `events.jsonl` (event stream)
- `episodes.jsonl` (open/closed/labeled/final_closed episode records)
- `summary.json` (run summary)
- `memory.db` (SQLite for querying)

## Main CLI entrypoints

- `big-brother` (analyze video)
- `big-brother-dashboard` (web dashboard)
- `big-brother-query` (query persisted DB)
- `big-brother-merge-episodes` (merge touching same-label episodes safely)

## Query model

`big-brother-query` supports:

- `events`
- `episodes`
- `tool-usage`
- `idle-ratio`
- `search-time`

Dashboard NL chat converts user questions into structured query calls and returns natural-language answers.

## Dashboard model

The dashboard reads run folders from `outputs/` and frame previews from `videos/`.

Important: frame preview requires stem match between run and video.

- run: `outputs/t1`
- required video: `videos/t1.mp4` (or another supported extension)

If the video is missing, `/api/frame` returns `404` and thumbnails are unavailable.

## Episode merge utility

`big-brother-merge-episodes` merges adjacent closed episodes with:

- same `worker_id`
- same `label`
- touching boundaries (`prev.t_end ~= next.t_start`)

Safety guarantees:

- skips runs with missing source video
- backups created before writes
- events-table fingerprint check to ensure event data is unchanged

## Typical workflow

1. Put videos in `videos/`
2. Run analysis with `big-brother`
3. Open dashboard with `big-brother-dashboard`
4. Optionally run `big-brother-merge-episodes`
5. Query with dashboard NL or `big-brother-query`
