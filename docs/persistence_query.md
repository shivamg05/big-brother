# Persistence and Querying

## Persistence model

Each run stores SQLite memory at:

- `outputs/<run>/memory.db`

This DB powers query CLI + dashboard API endpoints.

## Analyze with persistence

```bash
uv run big-brother --videos-dir videos --video t1.mp4 --extractor gemini --output-dir outputs
```

Optional explicit DB path (single video only):

```bash
uv run big-brother --videos-dir videos --video t1.mp4 --db-path outputs/t1/custom.db
```

## Query CLI

Events:

```bash
uv run big-brother-query --db-path outputs/t1/memory.db --query events --start-ts 0 --end-ts 400
```

Episodes:

```bash
uv run big-brother-query --db-path outputs/t1/memory.db --query episodes --start-ts 0 --end-ts 400
```

Tool usage:

```bash
uv run big-brother-query --db-path outputs/t1/memory.db --query tool-usage --tool nail_gun --start-ts 0 --end-ts 400
```

Idle ratio:

```bash
uv run big-brother-query --db-path outputs/t1/memory.db --query idle-ratio --start-ts 0 --end-ts 400
```

Search time:

```bash
uv run big-brother-query --db-path outputs/t1/memory.db --query search-time --start-ts 0 --end-ts 400
```

## Episode merge utility

Merge touching same-label closed episodes in DB + `episodes.jsonl`:

```bash
uv run big-brother-merge-episodes --outputs-dir outputs --videos-dir videos
```

Dry run:

```bash
uv run big-brother-merge-episodes --outputs-dir outputs --videos-dir videos --dry-run
```

Safety behavior:

- skips runs with no matching video in `videos/`
- creates backups before write (`memory.db.pre_merge.bak`, `episodes.pre_merge.bak.jsonl`)
- verifies `events` table fingerprint remains unchanged
