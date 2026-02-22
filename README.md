# big-brother

`big-brother` turns long-form worker POV video into structured memory you can query.

This repo is not trying to build a cinematic demo. It is trying to build a reliable, inspectable memory pipeline:

1. Break video into time windows.
2. Extract atomic events from sampled frames.
3. Group events into longer episodes.
4. Persist everything to JSONL + SQLite.
5. Query it deterministically (CLI/API), with optional NL-to-SQL on top.

## What We Are Building

Given a video like `videos/t1.mp4`, the system produces a run folder like `outputs/t1/` containing:

- `windows.jsonl`: window-level processing decisions (including gate decisions)
- `events.jsonl`: short-duration structured events
- `episodes.jsonl`: open/closed/labeled episode records over time
- `summary.json`: run-level summary
- `memory.db`: SQLite store used by query APIs and dashboard

The dashboard reads these artifacts directly, so you can inspect behavior as it runs.

## Approach (And Why)

Most failure modes in video intelligence come from trying to do too much in one shot. This project intentionally uses a layered approach:

- Short-window extraction for local accuracy
- Episode construction from event sequences (not single frames)
- Gating to skip expensive model calls when signals are stable
- Deterministic storage/query layer as the source of truth

This gives you lower cost, better debuggability, and easier regression testing than end-to-end black-box prompting.

## System Flow

`video -> windows -> sampled frames -> subtask events -> episode builder -> episode labeler -> storage/query`

Main components:

- `src/big_brother/video.py`: windowing + frame sampling + cheap motion signals
- `src/big_brother/gating.py`: decide whether to call extractor or extend prior event
- `src/big_brother/extractor.py`: event extraction backend (`gemini` or heuristic)
- `src/big_brother/episode.py`: episode boundary logic
- `src/big_brother/labeler.py`: episode labeling backend (`gemini`, heuristic, or off)
- `src/big_brother/storage.py`: SQLite persistence
- `src/big_brother/query.py` + `query_cli.py`: deterministic query surface
- `src/big_brother/dashboard.py`: FastAPI + UI for live inspection and querying

## Quick Start

Prereqs:

- Python `>=3.13` (required by `pyproject.toml`)
- `uv`

Install deps:

```bash
uv sync
```

Add `.env` (for Gemini mode):

```bash
GEMINI_API_KEY=your_key
```

Place a video in `videos/`, then run:

```bash
uv run big-brother \
  --videos-dir videos \
  --video t1.mp4 \
  --extractor gemini \
  --episode-labeler gemini \
  --output-dir outputs
```

No-key local run:

```bash
uv run big-brother \
  --videos-dir videos \
  --video t1.mp4 \
  --extractor heuristic \
  --episode-labeler heuristic \
  --output-dir outputs
```

Start dashboard:

```bash
uv run big-brother-dashboard --outputs-dir outputs --videos-dir videos --port 8008
```

Open `http://127.0.0.1:8008`.

## Querying

CLI query examples:

```bash
uv run big-brother-query \
  --db-path outputs/t1/memory.db \
  --query events \
  --start-ts 0 \
  --end-ts 400
```

```bash
uv run big-brother-query \
  --db-path outputs/t1/memory.db \
  --query tool-usage \
  --tool nail_gun \
  --start-ts 0 \
  --end-ts 400
```

Supported query types:

- `events`
- `episodes`
- `tool-usage`
- `idle-ratio`
- `search-time`

## Design Principles

- Prefer explicit schemas and enums over free-form text.
- Treat persistence as a first-class product surface.
- Keep model calls narrow and explainable.
- Make every stage inspectable with artifacts you can diff.
- Allow heuristic fallback for offline/dev workflows.

## What This Is Not

- Not a fully general activity-recognition benchmark framework.
- Not a one-shot prompt wrapper with hidden state.
- Not optimized for visual storytelling; optimized for memory quality and queryability.

## Repo Docs

- `docs/local_setup.md`: first-time setup and troubleshooting
- `docs/video_pipeline.md`: pipeline details and outputs
- `docs/dashboard.md`: dashboard and live query behavior
- `docs/persistence_query.md`: persistence/query examples
- `docs/overview.md`: higher-level system spec
