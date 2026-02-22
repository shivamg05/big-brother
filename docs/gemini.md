# Gemini Integration

Gemini is used for:

- event extraction (`GeminiExtractor`)
- episode labeling (`GeminiEpisodeLabeler`)
- natural-language dashboard Q&A (`GeminiNLQueryEngine`)

## Setup

1. Install dependencies:

```bash
uv sync
```

2. Set API key (`.env` or shell env):

- `GEMINI_API_KEY=...` (preferred)
- `GOOGLE_API_KEY=...` (also supported)

## Usage examples

Pipeline run:

```bash
uv run big-brother --videos-dir videos --video t1.mp4 --extractor gemini --episode-labeler gemini --output-dir outputs
```

Dashboard NL Q&A:

```bash
uv run big-brother-dashboard --outputs-dir outputs --videos-dir videos --port 8008
```

## Quota / failure behavior

- If credits/quota are exhausted, Gemini-backed flows can fail.
- For local fallback, use heuristic mode:

```bash
uv run big-brother --videos-dir videos --video t1.mp4 --extractor heuristic --episode-labeler heuristic --output-dir outputs
```

## Dependency note

`google-genai` must be installed in the environment running the CLI/dashboard.
