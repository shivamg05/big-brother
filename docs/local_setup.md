# Local Setup

## 1. Prerequisites

- macOS or Linux shell
- Python `>=3.13`
- `uv`

Install `uv` (if needed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If shell says `uv: command not found`, add it to PATH:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

## 2. Install

```bash
git clone <YOUR_REPO_URL>
cd big-brother
uv sync
```

## 3. Configure API key (Gemini mode)

Create `.env` in repo root:

```bash
cat > .env << 'EOF_ENV'
GEMINI_API_KEY=your_api_key_here
EOF_ENV
```

Also supported: `GOOGLE_API_KEY`.

## 4. Add videos

Put files in `videos/` (`.mp4/.mov/.avi/.mkv/.m4v`).

```bash
cp /path/to/video.mp4 videos/t1.mp4
```

## 5. Analyze video

Gemini mode:

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

Heuristic mode (no API key):

```bash
uv run big-brother \
  --videos-dir videos \
  --video t1.mp4 \
  --extractor heuristic \
  --episode-labeler heuristic \
  --output-dir outputs
```

Artifacts are written to `outputs/<video_stem>/`.

## 6. Run dashboard

```bash
uv run big-brother-dashboard --outputs-dir outputs --videos-dir videos --port 8008
```

Open `http://127.0.0.1:8008`.

## 7. Query persisted memory

```bash
uv run big-brother-query --db-path outputs/t1/memory.db --query events --start-ts 0 --end-ts 400
```

## 8. Optional post-process: merge touching same-label episodes

```bash
uv run big-brother-merge-episodes --outputs-dir outputs --videos-dir videos
```

Dry run:

```bash
uv run big-brother-merge-episodes --outputs-dir outputs --videos-dir videos --dry-run
```

Notes:

- The merge tool skips runs with no matching source video in `videos/`.
- It creates backups before applying:
  - `memory.db.pre_merge.bak`
  - `episodes.pre_merge.bak.jsonl`

## 9. Tests

```bash
uv run pytest -q
```

## Troubleshooting

- `uv: command not found`:
  - add `$HOME/.local/bin` to PATH.
- `Missing Gemini API key`:
  - set `GEMINI_API_KEY` or `GOOGLE_API_KEY`.
- `google-genai is not installed`:
  - run `uv sync`.
- `/api/frame ... 404` in dashboard:
  - missing `videos/<run>.<ext>` for the selected run.
- Dashboard shows no runs:
  - confirm `outputs/<run>/events.jsonl` exists.
