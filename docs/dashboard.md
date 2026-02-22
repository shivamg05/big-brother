# Dashboard

Run a local UI for browsing runs, querying memory, and inspecting episodes.

## Run

```bash
uv run big-brother-dashboard --outputs-dir outputs --videos-dir videos --port 8008
```

Open: `http://127.0.0.1:8008`

If `uv` is not on PATH, run with full path:

```bash
$HOME/.local/bin/uv run big-brother-dashboard --outputs-dir outputs --videos-dir videos --port 8008
```

## What the Dashboard Shows

- Worker/session overview cards.
- Event Log tab:
  - event cards with phase/action/tool chips
  - frame preview per event timestamp
  - filters for phase/action/tool
  - sort toggle (newest/oldest)
- Episodes tab:
  - horizontal episode timeline with alternating nodes
  - clickable episode details pane
  - zoom in/out controls with center-preserving zoom
- Chat panel:
  - natural-language Q&A over persisted memory
  - markdown-rendered assistant answers
  - typing animation + thinking indicator

## Data Requirements

Dashboard data is read from:

- `outputs/<run>/events.jsonl`
- `outputs/<run>/episodes.jsonl`
- `outputs/<run>/windows.jsonl`
- `outputs/<run>/summary.json`
- `outputs/<run>/memory.db`

Frame previews require a matching video file:

- `videos/<run>.mp4` (or `.mov`, `.avi`, `.mkv`, `.m4v`)

If the video is missing, `/api/frame` returns `404` and event thumbnails cannot render.

## Notes

- Run list is based on folders under `outputs/`.
- Snapshot polling is periodic, so newly written artifacts appear automatically.
- NL answers use Gemini-backed query flow and require API key/dependencies.
