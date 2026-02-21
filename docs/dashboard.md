# Live Dashboard

Spin up a real-time dashboard that tails output artifacts and shows:

- live event log
- associated video frames per event timestamp
- labeled episode feed
- phase/action/tool distributions
- live query panel over persisted memory DB

## Run

```bash
uv run big-brother-dashboard --outputs-dir outputs --videos-dir videos --port 8008
```

Open:

`http://127.0.0.1:8008`

## Notes

- The dashboard polls every 2 seconds.
- It expects per-run folders like `outputs/<run>/events.jsonl`.
- Frame previews are resolved from `videos/<run>.(mp4|mov|avi|mkv|m4v)`.
- Live query reads `outputs/<run>/memory.db`.

## Dashboard Querying

The "Live Query" panel supports:

- `events`
- `episodes`
- `tool-usage`
- `idle-ratio`
- `search-time`

Filter input format:

- `tool=nail_gun` (for tool-usage)
- `label=framing_wall` (for episodes)
