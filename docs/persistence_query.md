# Persistence and Querying

This project now persists in-video memory to SQLite and provides a query CLI.

## Analyze with persistence

By default, analysis writes a DB per video:

- `outputs/<video_stem>/memory.db`

Example:

```bash
uv run big-brother --videos-dir videos --video t1.mp4 --extractor gemini --output-dir outputs
```

You can also provide an explicit DB path for a single video run:

```bash
uv run big-brother --videos-dir videos --video t1.mp4 --db-path outputs/t1/custom.db
```

## Query persisted memory

### Events

```bash
uv run big-brother-query --db-path outputs/t1/memory.db --query events --start-ts 0 --end-ts 400
```

### Episodes

```bash
uv run big-brother-query --db-path outputs/t1/memory.db --query episodes --start-ts 0 --end-ts 400
```

### Tool usage

```bash
uv run big-brother-query --db-path outputs/t1/memory.db --query tool-usage --tool nail_gun --start-ts 0 --end-ts 400
```

### Idle ratio

```bash
uv run big-brother-query --db-path outputs/t1/memory.db --query idle-ratio --start-ts 0 --end-ts 400
```

### Search time

```bash
uv run big-brother-query --db-path outputs/t1/memory.db --query search-time --start-ts 0 --end-ts 400
```

