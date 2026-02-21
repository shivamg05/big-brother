# Local Setup (First Time)

Use this guide when running the project locally for the first time after cloning the repo.

## 1. Prerequisites

- macOS or Linux shell environment
- `git`
- `uv` installed
- Python `3.13+` (project requires `>=3.13`)

Install `uv` (if needed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify tools:

```bash
uv --version
python3 --version
```

## 2. Clone and install dependencies

```bash
git clone <YOUR_REPO_URL>
cd big-brother
uv sync
```

`uv sync` creates/updates the local virtual environment and installs all dependencies from `pyproject.toml`/`uv.lock`.

## 3. Configure environment variables

Create a `.env` file in repo root:

```bash
cat > .env << 'EOF'
GEMINI_API_KEY=your_api_key_here
EOF
```

Alternative variable name also supported:

- `GOOGLE_API_KEY`

If you do not want to use Gemini yet, skip this and run with `--extractor heuristic --episode-labeler heuristic`.

## 4. Add input videos

Put videos into `videos/` (supported extensions: `.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`).

Example:

```bash
cp /path/to/video.mp4 videos/t1.mp4
```

## 5. Run analysis

Gemini-backed run:

```bash
uv run big-brother \
  --videos-dir videos \
  --video t1.mp4 \
  --extractor gemini \
  --episode-labeler gemini \
  --output-dir outputs \
  --requests-per-minute 4 \
  --max-retries 8
```

No-API-key local run (heuristic):

```bash
uv run big-brother \
  --videos-dir videos \
  --video t1.mp4 \
  --extractor heuristic \
  --episode-labeler heuristic \
  --output-dir outputs
```

Artifacts are written to:

- `outputs/<video_stem>/windows.jsonl`
- `outputs/<video_stem>/events.jsonl`
- `outputs/<video_stem>/episodes.jsonl`
- `outputs/<video_stem>/summary.json`
- `outputs/<video_stem>/memory.db`

## 6. Start the dashboard

```bash
uv run big-brother-dashboard --outputs-dir outputs --videos-dir videos --port 8008
```

Open:

- `http://127.0.0.1:8008`

## 7. Query persisted memory from CLI

Example event query:

```bash
uv run big-brother-query \
  --db-path outputs/t1/memory.db \
  --query events \
  --start-ts 0 \
  --end-ts 400
```

Example tool-usage query:

```bash
uv run big-brother-query \
  --db-path outputs/t1/memory.db \
  --query tool-usage \
  --tool nail_gun \
  --start-ts 0 \
  --end-ts 400
```

## 8. Optional: run tests

```bash
uv run pytest -q
```

## Troubleshooting

- `Missing Gemini API key`:
  - add `GEMINI_API_KEY` to `.env` or run heuristic mode.
- `google-genai is not installed`:
  - run `uv sync` again.
- `No videos found in videos`:
  - confirm video files are in `videos/` and pass `--video <filename>`.
- Dashboard shows no runs:
  - confirm analysis has completed and `outputs/<run>/events.jsonl` exists.
- Frame previews missing:
  - ensure video name matches run folder stem (for `outputs/t1`, video should be `videos/t1.<ext>`).
