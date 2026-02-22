# big-brother

`big-brother` turns long-form worker POV video into structured, queryable operational memory.

This is a working system prototype, not a polished commercial product. The goal is to show that construction video can become reliable, inspectable memory that supports analysis and decision-making instead of sitting as unsearchable footage.

## Why This Matters

Teams record large amounts of field video, but most of it is hard to use after capture. You can watch clips manually, but you cannot quickly answer practical questions like:

- What tools were used most over the last 30 minutes?
- How much time was spent searching vs executing?
- What larger task episodes occurred, and when?

Our thesis: if we convert raw video into a structured timeline of events and episodes, we unlock much better retrieval and downstream automation.

## What We Built

Given an input like `videos/t1.mp4`, the system generates `outputs/t1/` with:

- `windows.jsonl`: per-window processing and gate decisions
- `events.jsonl`: atomic structured activity events
- `episodes.jsonl`: higher-level grouped episodes (open/closed/labeled)
- `summary.json`: run-level processing stats
- `memory.db`: SQLite memory used by query APIs and dashboard

The dashboard reads these artifacts directly, so the whole pipeline is inspectable while running.

## Technical Approach

We deliberately use a layered architecture instead of one large end-to-end prompt:

1. Window video (`10-20s` typical).
2. Sample representative frames per window.
3. Compute cheap change signals (motion/drift) for gating.
4. Extract one structured event per relevant window (Gemini or heuristic fallback).
5. Build longer episodes from event sequences.
6. Label closed episodes from sequence context.
7. Persist to SQLite + JSONL for deterministic querying and auditability.

This gives us lower cost, clearer failure modes, and better debuggability than black-box video summarization.

## Core Features (Current)

- Multi-video CLI processing with configurable window size, overlap, and frame count.
- Gating engine to avoid unnecessary model calls in stable segments.
- Structured extraction schema with strict enums (`phase`, `action`, `tool`, etc.).
- Rolling state passed between windows for temporal continuity.
- Episode boundary detection with heuristic split logic (idle/travel/location/tool shifts).
- Episode labeling via Gemini or heuristic labeler.
- Deterministic query layer over SQLite (`events`, `episodes`, `tool-usage`, `idle-ratio`, `search-time`).
- FastAPI dashboard for:
  - run snapshot inspection
  - frame retrieval at timestamps
  - structured queries
  - NL ask endpoint powered by SQL agent
  - worker/session upload + auto-run kickoff
- JSONL streaming output for live traceability.
- Heuristic extractor/labeler path for no-key or local fallback mode.

## Architecture At A Glance

`video -> windowing -> gating -> event extraction -> episode builder -> episode labeler -> storage -> query/dashboard`

Main modules:

- `src/big_brother/video.py` - ingest, windowing, frame sampling, change signals
- `src/big_brother/gating.py` - model-call decisioning
- `src/big_brother/extractor.py` - Gemini + heuristic event extraction
- `src/big_brother/episode.py` - episode construction
- `src/big_brother/labeler.py` - episode labeling backends
- `src/big_brother/storage.py` - SQLite persistence (`events`, `episodes`)
- `src/big_brother/query.py` / `query_cli.py` - deterministic queries
- `src/big_brother/dashboard.py` - UI + API surface
- `src/big_brother/sql_agent.py` - read-only LLM-generated SQL answering

For a deeper module-level breakdown, see `architecture.md`.

## Quick Start

Prereqs:

- Python `>=3.13`
- `uv`

Install deps:

```bash
uv sync
```

Set API key for Gemini-backed paths:

```bash
GEMINI_API_KEY=your_key
```

Run one video:

```bash
uv run big-brother \
  --videos-dir videos \
  --video t1.mp4 \
  --extractor gemini \
  --episode-labeler gemini \
  --output-dir outputs
```

No-key local fallback:

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

## Query Examples

Events in a time range:

```bash
uv run big-brother-query \
  --db-path outputs/t1/memory.db \
  --query events \
  --start-ts 0 \
  --end-ts 400
```

Tool usage duration:

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

## What We Learned Building It

- Structured intermediate memory is much easier to trust and debug than free-form summaries.
- Gating is high leverage for cost control without losing timeline continuity.
- Persisting both JSONL and SQLite gives good developer ergonomics: human-inspectable logs + machine-queryable state.
- Episode-level reasoning is stronger when derived from event sequences instead of direct single-frame guesses.

## Next Iteration

A simple, high-impact next step is connecting memory across runs/videos so an LLM can analyze behavior over longer horizons (cross-day/cross-worker retrieval and orchestration), not just one run at a time.

## Scope Note

This repository demonstrates a robust technical approach and working prototype. It is not yet packaged as a production SaaS product.

## Docs

- `architecture.md` - current technical architecture
- `docs/local_setup.md` - setup and troubleshooting
- `docs/video_pipeline.md` - video pipeline behavior and outputs
- `docs/dashboard.md` - dashboard behavior
- `docs/persistence_query.md` - persistence/query details
- `docs/overview.md` - high-level original spec
