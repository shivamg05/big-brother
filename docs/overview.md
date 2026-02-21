# Spatial-Temporal Worker Memory System (Spec v1)

## Overview

This system converts long-form egocentric (POV) construction video into a **structured, queryable memory** of a worker’s activity over time.

It focuses on:
- **Subtask extraction** (short time windows)
- **Episode construction** (multi-minute tasks)
- **Temporal + spatial grounding**
- **Efficient, stateful inference using VLMs**

The system avoids heavy reconstruction and instead builds a **hierarchical understanding of behavior over time**.

---

## Core Concepts

### 1. Subtasks (Atomic Events)
Short-duration actions (10–20s windows)

Examples:
- carrying wood
- using nail gun
- measuring
- walking
- idle

Derived directly from frames.

---

### 2. Episodes (Encasing Tasks)
Multi-minute, higher-level tasks composed of subtasks

Examples:
- building door frame
- installing drywall
- wiring setup

Derived from **patterns of subtasks over time**, not single frames.

---

### 3. State (Memory)
A lightweight rolling context passed between iterations.

---

## System Architecture

Video → Windowing → Subtask Extraction → Event Stream → Episode Builder → Episode Labeling → Query Layer

---

## Pipeline

### Step 1: Video Chunking

- Input video split into windows:
  - Duration: **10–20 seconds**
  - Overlap: optional (0–5s)

```python
windows = split_video(video, window_size=15s)


⸻

Step 2: Frame Sampling

Per window:
	•	Sample 2–4 frames
	•	start
	•	mid
	•	end
	•	Optional:
	•	detect motion peaks
	•	include 1 frame with highest change

frames = sample_frames(window, strategy="start_mid_end")


⸻

Step 3: Subtask Extraction (VLM Call)

Input
	•	sampled frames
	•	small rolling state (text only)

Output (STRICT SCHEMA)

{
  "t_start": "...",
  "t_end": "...",
  "phase": "travel | search | setup | execute | inspect | communicate | idle | cleanup",
  "action": "carry | measure | cut | drill | nail | align | lift | fasten | inspect | walk | idle | other",
  "tool": "nail_gun | drill | saw | hammer | tape_measure | none | unknown",
  "materials": ["wood", "metal", "wire", "unknown"],
  "people_nearby": "0 | 1 | 2+",
  "speaking": "none | brief | sustained",
  "location_hint": "same_area | new_area | unknown",
  "confidence": 0.0-1.0,
  "evidence": "short explanation"
}

Notes
	•	Use enums wherever possible
	•	Allow "unknown" to reduce hallucination
	•	Keep response minimal and consistent

⸻

Step 4: Event Stream

Append each subtask to a global list:

events.append(event)

This becomes the primary memory representation.

⸻

Episode Construction

Goal

Group subtasks into multi-minute episodes

⸻

Episode Data Structure

{
  "episode_id": "...",
  "t_start": "...",
  "t_end": "...",
  "events": [...],
  "dominant_phase": "...",
  "tools_used": [...],
  "zone_id": "...",
  "label": "unknown | building frame | installing drywall | ...",
  "confidence": 0.0-1.0
}


⸻

Episode Rules (v1: heuristic)

Maintain a current_episode

Continue episode if:
	•	Same zone (or location_hint = same_area)
	•	Phase is mostly:
	•	setup / execute / inspect
	•	No long idle or travel

Start new episode if:
	•	travel > 30–60s
	•	idle > 45–90s
	•	location change
	•	major tool/context switch

⸻

Pseudocode

def update_episodes(events, current_episode):
    if boundary_detected(events[-1]):
        close(current_episode)
        current_episode = new_episode()
    
    current_episode.events.append(events[-1])


⸻

Episode Labeling (Encasing Task)

Key Idea

Do NOT infer from frames. Infer from event sequences.

⸻

Trigger Conditions
	•	Episode closed
	•	OR every N minutes (e.g. 2–5 min)

⸻

Input to VLM
	•	Last 20–40 structured events

[
  {"phase": "carry", "tool": "none", ...},
  {"phase": "measure", "tool": "tape_measure", ...},
  {"phase": "execute", "tool": "nail_gun", ...}
]


⸻

Output

{
  "label": "building door frame",
  "confidence": 0.82,
  "reasoning": "sequence of measuring, aligning, and nailing wood structures"
}


⸻

Important
	•	No images used here
	•	Cheap and high signal

⸻

State Management

Rolling State (passed to subtask VLM)

Keep minimal:

{
  "last_phase": "...",
  "current_tool_active": "...",
  "last_location_hint": "...",
  "recent_actions": ["carry", "measure"]
}


⸻

Principles
	•	Never pass full history
	•	Only pass compressed context
	•	Let system memory handle persistence

⸻

Reducing VLM Cost

1. Gating (VERY IMPORTANT)

Only call VLM when needed.

Use cheap signals:
	•	optical flow change
	•	frame embedding drift
	•	audio spike (tool/speech)

If no change:

extend_previous_event()


⸻

2. Delta Outputs

Prompt VLM to output:
	•	only changes
	•	start/stop events

⸻

3. Keyframe Compression

Instead of many frames:
	•	2–4 representative frames

⸻

4. Periodic Summaries

Every 10–15 minutes:
	•	summarize events (no images)
	•	update global context

⸻

Optional Spatial Layer (SLAM-lite)

Goal

Track worker as a moving dot

⸻

Pipeline
	1.	Run SLAM → get camera poses
	2.	Cluster poses → zones
	3.	Attach events to poses

⸻

Output
	•	trajectory line
	•	zone heatmap
	•	event markers

⸻

Query Layer

Enables:
	•	“When was drill used?”
	•	“How long was worker idle?”
	•	“What tasks were completed?”

⸻

Backed by:
	•	event stream
	•	episode summaries

⸻

Metrics (Derived)

Productivity
	•	active vs idle ratio
	•	time per episode

Tool Usage
	•	total usage time
	•	fragmentation

Search Time
	•	search phase duration

Movement
	•	distance traveled per work unit

⸻

Implementation Plan (Step 1)

MUST HAVE
	•	video chunking
	•	frame sampling
	•	subtask VLM call
	•	event schema + storage
	•	basic episode grouping

NICE TO HAVE
	•	episode labeling
	•	gating logic
	•	simple trajectory (even fake)

⸻

Key Design Principles
	1.	Hierarchy over flat labeling
	•	events → episodes
	2.	Temporal reasoning over frame reasoning
	•	never infer big tasks from single frames
	3.	State outside the model
	•	use structured memory, not context stuffing
	4.	Cheap first, smart later
	•	rules + light VLM > heavy VLM everywhere

⸻

Summary

This system transforms raw POV video into:
	•	structured event stream
	•	grouped task episodes
	•	queryable memory

The core innovation is:

building high-level understanding from consistent low-level signals over time

Not from single-frame intelligence.

