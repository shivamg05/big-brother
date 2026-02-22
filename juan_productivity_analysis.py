#!/usr/bin/env python3
"""
Juan's Productivity Deep Dive Analysis
Analyzes worker productivity from Big Brother event data
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import statistics

def analyze_juan_productivity(db_path: str = "outputs/juan/memory.db"):
    """Comprehensive productivity analysis for Juan"""

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    print("\n" + "="*80)
    print("🏗️  JUAN'S PRODUCTIVITY DEEP DIVE ANALYSIS")
    print("="*80)

    # 1. Overview Statistics
    print("\n📊 OVERALL STATISTICS")
    print("-"*40)

    cursor = conn.execute("""
        SELECT
            COUNT(DISTINCT event_id) as total_events,
            MIN(t_start) as earliest_time,
            MAX(t_end) as latest_time,
            COUNT(DISTINCT tool) as unique_tools,
            COUNT(DISTINCT action) as unique_actions,
            COUNT(DISTINCT phase) as unique_phases
        FROM Events
        WHERE worker_id = 'juan'
    """)

    stats = dict(cursor.fetchone())
    total_duration = stats['latest_time'] - stats['earliest_time']

    print(f"📹 Video Duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"📝 Total Events: {stats['total_events']}")
    print(f"🔧 Unique Tools: {stats['unique_tools']}")
    print(f"⚡ Unique Actions: {stats['unique_actions']}")
    print(f"📋 Work Phases: {stats['unique_phases']}")

    # 2. Remove Overlapping Events - Take latest run only
    print("\n🔍 HANDLING OVERLAPPING EVENTS")
    print("-"*40)

    # Get events grouped by approximate time windows to identify duplicates
    cursor = conn.execute("""
        WITH deduplicated AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY CAST(t_start/10 as INTEGER)
                       ORDER BY event_id DESC
                   ) as rn
            FROM Events
            WHERE worker_id = 'juan'
        )
        SELECT COUNT(*) as original_count,
               SUM(CASE WHEN rn = 1 THEN 1 ELSE 0 END) as deduplicated_count
        FROM deduplicated
    """)

    dedup_stats = dict(cursor.fetchone())
    print(f"Original Events: {dedup_stats['original_count']}")
    print(f"After Deduplication: {dedup_stats['deduplicated_count']}")
    print(f"Removed Duplicates: {dedup_stats['original_count'] - dedup_stats['deduplicated_count']}")

    # 3. Time Distribution Analysis (using deduplicated data)
    print("\n⏱️  TIME DISTRIBUTION")
    print("-"*40)

    cursor = conn.execute("""
        WITH deduplicated AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY CAST(t_start/10 as INTEGER)
                       ORDER BY event_id DESC
                   ) as rn
            FROM Events
            WHERE worker_id = 'juan'
        ),
        phase_times AS (
            SELECT
                phase,
                SUM(t_end - t_start) as total_time
            FROM deduplicated
            WHERE rn = 1
            GROUP BY phase
        )
        SELECT
            phase,
            total_time,
            total_time * 100.0 / (SELECT SUM(total_time) FROM phase_times) as percentage
        FROM phase_times
        ORDER BY total_time DESC
    """)

    phase_breakdown = cursor.fetchall()
    for row in phase_breakdown:
        phase = row['phase'] or 'unknown'
        time = row['total_time']
        pct = row['percentage']
        print(f"{phase.upper():12} {time:6.1f}s ({pct:5.1f}%) {'🔴' if phase == 'idle' else '🟢'}")

    # 4. Tool Usage Analysis
    print("\n🔧 TOOL USAGE ANALYSIS")
    print("-"*40)

    cursor = conn.execute("""
        WITH deduplicated AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY CAST(t_start/10 as INTEGER)
                       ORDER BY event_id DESC
                   ) as rn
            FROM Events
            WHERE worker_id = 'juan'
        ),
        tool_stats AS (
            SELECT
                tool,
                COUNT(*) as usage_count,
                SUM(t_end - t_start) as total_time
            FROM deduplicated
            WHERE rn = 1 AND tool IS NOT NULL AND tool != 'none'
            GROUP BY tool
        )
        SELECT
            tool,
            usage_count,
            total_time,
            total_time * 100.0 / (SELECT SUM(total_time) FROM tool_stats) as percentage
        FROM tool_stats
        ORDER BY total_time DESC
    """)

    tool_stats = cursor.fetchall()
    for row in tool_stats:
        tool = row['tool']
        count = row['usage_count']
        time = row['total_time']
        pct = row['percentage']
        print(f"{tool:15} Used {count:3}x for {time:6.1f}s ({pct:5.1f}%)")

    # 5. Tool Switching Analysis
    print("\n🔄 TOOL SWITCHING EFFICIENCY")
    print("-"*40)

    cursor = conn.execute("""
        WITH deduplicated AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY CAST(t_start/10 as INTEGER)
                       ORDER BY event_id DESC
                   ) as rn
            FROM Events
            WHERE worker_id = 'juan'
        ),
        tool_sequences AS (
            SELECT
                tool,
                t_start,
                t_end,
                LAG(tool) OVER (ORDER BY t_start) as prev_tool,
                LAG(t_end) OVER (ORDER BY t_start) as prev_end
            FROM deduplicated
            WHERE rn = 1 AND tool IS NOT NULL
            ORDER BY t_start
        ),
        switches AS (
            SELECT
                COUNT(CASE WHEN tool != prev_tool THEN 1 END) as switch_count,
                AVG(CASE WHEN tool != prev_tool THEN t_start - prev_end END) as avg_switch_time
            FROM tool_sequences
            WHERE prev_tool IS NOT NULL
        )
        SELECT * FROM switches
    """)

    switch_stats = dict(cursor.fetchone())
    print(f"Tool Switches: {switch_stats['switch_count'] or 0}")
    print(f"Avg Switch Time: {switch_stats['avg_switch_time'] or 0:.1f}s")

    # 6. Idle Time Analysis
    print("\n😴 IDLE TIME ANALYSIS")
    print("-"*40)

    cursor = conn.execute("""
        WITH deduplicated AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY CAST(t_start/10 as INTEGER)
                       ORDER BY event_id DESC
                   ) as rn
            FROM Events
            WHERE worker_id = 'juan'
        )
        SELECT
            t_start,
            t_end,
            (t_end - t_start) as duration,
            evidence
        FROM deduplicated
        WHERE rn = 1 AND phase = 'idle'
        ORDER BY duration DESC
        LIMIT 5
    """)

    idle_periods = cursor.fetchall()
    if idle_periods:
        print("Top 5 Longest Idle Periods:")
        for i, row in enumerate(idle_periods, 1):
            start = row['t_start']
            end = row['t_end']
            duration = row['duration']
            evidence = row['evidence'] or 'No details'
            print(f"{i}. {start:.0f}-{end:.0f}s ({duration:.1f}s): {evidence[:50]}")
    else:
        print("No idle periods detected")

    # 7. Productivity Patterns
    print("\n📈 PRODUCTIVITY PATTERNS")
    print("-"*40)

    cursor = conn.execute("""
        WITH deduplicated AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY CAST(t_start/10 as INTEGER)
                       ORDER BY event_id DESC
                   ) as rn
            FROM Events
            WHERE worker_id = 'juan'
        ),
        time_segments AS (
            SELECT
                CAST(t_start / 60 as INTEGER) as minute,
                phase,
                COUNT(*) as event_count,
                SUM(CASE WHEN phase = 'execute' THEN 1 ELSE 0 END) as productive_events,
                SUM(CASE WHEN phase = 'idle' THEN 1 ELSE 0 END) as idle_events
            FROM deduplicated
            WHERE rn = 1
            GROUP BY minute
        )
        SELECT
            minute,
            event_count,
            productive_events * 100.0 / event_count as productivity_rate
        FROM time_segments
        ORDER BY minute
    """)

    productivity_by_minute = cursor.fetchall()
    if productivity_by_minute:
        print("Productivity by Minute:")
        for row in productivity_by_minute:
            minute = row['minute']
            rate = row['productivity_rate'] or 0
            bar = '█' * int(rate / 10)
            print(f"Min {minute}: {bar:10} {rate:.0f}%")

    # 8. Workflow Patterns
    print("\n🔁 COMMON WORKFLOW PATTERNS")
    print("-"*40)

    cursor = conn.execute("""
        WITH deduplicated AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY CAST(t_start/10 as INTEGER)
                       ORDER BY event_id DESC
                   ) as rn
            FROM Events
            WHERE worker_id = 'juan'
        ),
        sequences AS (
            SELECT
                action,
                LEAD(action, 1) OVER (ORDER BY t_start) as next_action,
                LEAD(action, 2) OVER (ORDER BY t_start) as next_next_action
            FROM deduplicated
            WHERE rn = 1 AND action IS NOT NULL
        )
        SELECT
            action || ' → ' || next_action || ' → ' || next_next_action as pattern,
            COUNT(*) as frequency
        FROM sequences
        WHERE next_action IS NOT NULL AND next_next_action IS NOT NULL
        GROUP BY pattern
        HAVING COUNT(*) > 1
        ORDER BY frequency DESC
        LIMIT 5
    """)

    patterns = cursor.fetchall()
    if patterns:
        print("Most Common 3-Step Patterns:")
        for row in patterns:
            pattern = row['pattern']
            freq = row['frequency']
            print(f"  {pattern} (occurred {freq}x)")

    # 9. Improvement Opportunities
    print("\n💡 IMPROVEMENT OPPORTUNITIES")
    print("-"*40)

    # Calculate metrics for recommendations
    cursor = conn.execute("""
        WITH deduplicated AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY CAST(t_start/10 as INTEGER)
                       ORDER BY event_id DESC
                   ) as rn
            FROM Events
            WHERE worker_id = 'juan'
        )
        SELECT
            SUM(CASE WHEN phase = 'idle' THEN t_end - t_start ELSE 0 END) as total_idle,
            SUM(CASE WHEN phase = 'travel' THEN t_end - t_start ELSE 0 END) as total_travel,
            COUNT(DISTINCT CASE WHEN tool != LAG(tool) OVER (ORDER BY t_start)
                           THEN event_id END) as tool_switches,
            MAX(t_end) - MIN(t_start) as total_time
        FROM deduplicated
        WHERE rn = 1
    """)

    metrics = dict(cursor.fetchone())

    improvements = []

    # Idle time reduction
    if metrics['total_idle'] and metrics['total_time']:
        idle_pct = (metrics['total_idle'] / metrics['total_time']) * 100
        if idle_pct > 10:
            potential_savings = metrics['total_idle'] * 0.5  # Could reduce by 50%
            improvements.append(f"🔴 IDLE TIME: {idle_pct:.1f}% of time is idle")
            improvements.append(f"   → Could save {potential_savings:.0f}s by better task planning")

    # Travel time reduction
    if metrics['total_travel'] and metrics['total_time']:
        travel_pct = (metrics['total_travel'] / metrics['total_time']) * 100
        if travel_pct > 15:
            improvements.append(f"🔴 TRAVEL TIME: {travel_pct:.1f}% spent moving")
            improvements.append(f"   → Reorganize workspace to reduce movement")

    # Tool switching
    if metrics['tool_switches'] and metrics['total_time']:
        switches_per_min = (metrics['tool_switches'] / (metrics['total_time'] / 60))
        if switches_per_min > 2:
            improvements.append(f"🔴 TOOL SWITCHING: {switches_per_min:.1f} switches/min")
            improvements.append(f"   → Batch similar tasks to reduce tool changes")

    if improvements:
        for imp in improvements:
            print(imp)
    else:
        print("✅ Workflow appears optimized!")

    # 10. Summary Recommendations
    print("\n📋 SUMMARY & RECOMMENDATIONS")
    print("-"*40)

    print("KEY FINDINGS:")
    print(f"1. Worker active for {total_duration/60:.1f} minutes")
    print(f"2. {stats['total_events']} distinct work events captured")

    if metrics['total_idle'] and metrics['total_time']:
        idle_pct = (metrics['total_idle'] / metrics['total_time']) * 100
        print(f"3. Idle time accounts for {idle_pct:.1f}% of work")

    print("\nTOP RECOMMENDATIONS:")
    print("1. ⚡ Reduce idle time through better task sequencing")
    print("2. 🔧 Organize tools by frequency of use")
    print("3. 📦 Pre-stage materials to reduce preparation time")
    print("4. 🎯 Batch similar tasks to minimize tool switches")

    conn.close()

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)

if __name__ == "__main__":
    analyze_juan_productivity()