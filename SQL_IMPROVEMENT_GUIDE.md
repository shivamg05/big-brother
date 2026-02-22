# SQL Agent Performance Issues & Solutions

## Current Problems

### 1. **Data Sparsity**
- Only 27 events for a 404-second (6.7 min) video
- That's 1 event per 15 seconds - way too sparse!
- Most tools are "unknown" or "none"

### 2. **SQL Query Quality**
From the logs, the SQL agent:
- ✅ Generates correct SQL syntax
- ✅ Has comprehensive logging (added earlier)
- ❌ Limited by poor underlying data
- ❌ Generates simplistic queries for complex questions

### 3. **Data Distribution**
```
nail_gun: 12 events
none: 8 events
unknown: 2 events
tape_measure: 2 events
circular_saw: 2 events
speed_square: 1 event
```

## Solutions

### Immediate Fix: Better SQL Prompting

Update the SQL agent prompt to handle sparse data better:

```python
# In sql_agent.py, update the prompt:
prompt = (
    "You are an expert SQLite analyst for a construction activity memory database.\n"
    "IMPORTANT: The data may be sparse. When looking for continuous periods:\n"
    "- Check for sequences of events with the same tool\n"
    "- Consider gaps between events as potential continuation\n"
    "- Use window functions (LAG/LEAD) to detect patterns\n"
    "- For 'longest continuous', look at both single events AND sequences\n"
    "\n"
    "For questions about continuous tool use, try this pattern:\n"
    "WITH tool_sequences AS (\n"
    "  SELECT *, \n"
    "    LAG(tool) OVER (ORDER BY t_start) as prev_tool,\n"
    "    LAG(t_end) OVER (ORDER BY t_start) as prev_end\n"
    "  FROM Events WHERE worker_id = 'juan'\n"
    "),\n"
    "grouped AS (\n"
    "  SELECT *,\n"
    "    CASE WHEN tool != prev_tool OR t_start - prev_end > 30 THEN 1 ELSE 0 END as new_group,\n"
    "    SUM(CASE WHEN tool != prev_tool OR t_start - prev_end > 30 THEN 1 ELSE 0 END) \n"
    "      OVER (ORDER BY t_start) as group_id\n"
    "  FROM tool_sequences\n"
    ")\n"
    "SELECT tool, MIN(t_start) as start, MAX(t_end) as end, MAX(t_end) - MIN(t_start) as duration\n"
    "FROM grouped GROUP BY group_id, tool\n"
)
```

### Long-term Fix: Re-process Video

The real issue is data extraction. Re-run with:

```bash
# More dense sampling (every 5 seconds instead of 15)
python3.11 -m src.big_brother.cli \
    --video juan.mp4 \
    --window-size 5.0 \
    --frames-per-window 3 \
    --overlap 2.5 \
    --output-dir outputs_dense
```

### SQL Query Improvements

For better retrieval, add these query templates:

```sql
-- Find actual continuous tool usage (accounting for gaps)
WITH tool_runs AS (
    SELECT
        tool,
        t_start,
        t_end,
        LAG(tool) OVER (ORDER BY t_start) as prev_tool,
        LAG(t_end) OVER (ORDER BY t_start) as prev_end,
        t_start - LAG(t_end) OVER (ORDER BY t_start) as gap
    FROM Events
    WHERE worker_id = 'juan' AND tool != 'none'
),
continuous_periods AS (
    SELECT
        tool,
        t_start,
        t_end,
        CASE
            WHEN prev_tool = tool AND gap < 30 THEN 0  -- Same tool within 30s
            ELSE 1
        END as new_period
    FROM tool_runs
),
period_groups AS (
    SELECT
        tool,
        t_start,
        t_end,
        SUM(new_period) OVER (ORDER BY t_start) as period_id
    FROM continuous_periods
)
SELECT
    tool,
    MIN(t_start) as period_start,
    MAX(t_end) as period_end,
    MAX(t_end) - MIN(t_start) as total_duration,
    COUNT(*) as event_count
FROM period_groups
GROUP BY tool, period_id
ORDER BY total_duration DESC;
```

### Dashboard Improvements

The "Show reasoning trace" checkbox likely doesn't affect SQL generation. To make it useful:

1. Add query explanation to the response
2. Show multiple candidate queries
3. Display confidence scores

### Why Answers Are Bad

Looking at your example: "longest continuous tool use = unknown for 15s"

This is technically correct but useless. The issue:
- Sparse data (27 events for 404 seconds)
- Poor tool recognition ("unknown", "none")
- No continuous period detection (just max single event duration)

## Action Items

1. **Immediate**: Update SQL prompt to handle sparse data better
2. **Quick Win**: Add continuous period detection logic
3. **Best Fix**: Re-process video with denser sampling (5s windows)
4. **UI**: Show data quality metrics in dashboard