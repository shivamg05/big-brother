from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
import uvicorn


VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".m4v"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _distribution(events: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for event in events:
        value = str(event.get(key, "unknown"))
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items(), key=lambda x: x[1], reverse=True))


def _find_video_file(videos_dir: Path, run: str) -> Path | None:
    for ext in VIDEO_EXTENSIONS:
        candidate = videos_dir / f"{run}{ext}"
        if candidate.exists():
            return candidate
    return None


def _extract_frame_jpeg(video_path: Path, t_seconds: float) -> bytes:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=404, detail=f"Cannot open video {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            raise HTTPException(status_code=400, detail="Invalid video stream")
        duration = frame_count / fps
        clamped_t = max(0.0, min(float(t_seconds), max(0.0, duration - (1.0 / fps))))
        frame_idx = int(round(clamped_t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            raise HTTPException(status_code=404, detail="Failed to decode frame")
        ok, enc = cv2.imencode(".jpg", frame)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode frame")
        return enc.tobytes()
    finally:
        cap.release()


def create_app(*, outputs_dir: Path, videos_dir: Path) -> FastAPI:
    app = FastAPI(title="Big Brother Dashboard")

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return _dashboard_html()

    @app.get("/api/runs")
    def runs() -> JSONResponse:
        if not outputs_dir.exists():
            return JSONResponse({"runs": []})
        run_dirs = sorted([p.name for p in outputs_dir.iterdir() if p.is_dir()])
        return JSONResponse({"runs": run_dirs})

    @app.get("/api/snapshot")
    def snapshot(
        run: str = Query(...),
        limit: int = Query(40, ge=1, le=300),
    ) -> JSONResponse:
        run_dir = outputs_dir / run
        events = _read_jsonl(run_dir / "events.jsonl")
        episodes = _read_jsonl(run_dir / "episodes.jsonl")
        windows = _read_jsonl(run_dir / "windows.jsonl")
        summary_path = run_dir / "summary.json"
        summary = {}
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        recent_events = events[-limit:]
        for event in recent_events:
            midpoint = (float(event.get("t_start", 0.0)) + float(event.get("t_end", 0.0))) / 2.0
            event["frame_url"] = f"/api/frame?run={run}&t={midpoint:.3f}"

        latest_episode_by_id: dict[str, dict[str, Any]] = {}
        for ep in episodes:
            episode_id = str(ep.get("episode_id", ""))
            if episode_id:
                latest_episode_by_id[episode_id] = ep

        labeled = [ep for ep in episodes if str(ep.get("kind", "")) in {"labeled", "final_closed"}]

        payload = {
            "run": run,
            "summary": summary,
            "event_count": len(events),
            "episode_count": len(episodes),
            "window_count": len(windows),
            "events": recent_events,
            "episodes_latest": list(latest_episode_by_id.values())[-limit:],
            "labeled_episodes": labeled[-limit:],
            "distributions": {
                "phase": _distribution(events, "phase"),
                "action": _distribution(events, "action"),
                "tool": _distribution(events, "tool"),
            },
        }
        return JSONResponse(payload)

    @app.get("/api/frame")
    def frame(run: str = Query(...), t: float = Query(...)) -> Response:
        video_path = _find_video_file(videos_dir, run)
        if video_path is None:
            raise HTTPException(status_code=404, detail=f"No video found for run '{run}' in {videos_dir}")
        image_bytes = _extract_frame_jpeg(video_path, t_seconds=t)
        return Response(content=image_bytes, media_type="image/jpeg")

    return app


def _dashboard_html() -> str:
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Worker Memory Live Dashboard</title>
  <style>
    :root {
      --bg-a: #f8f4e8;
      --bg-b: #d6eadf;
      --panel: #ffffffee;
      --ink: #152022;
      --muted: #4f6266;
      --accent: #116466;
      --bar: #2c7a7b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: radial-gradient(circle at 20% 20%, var(--bg-b), var(--bg-a));
      font-family: "Avenir Next", "Trebuchet MS", sans-serif;
    }
    .shell { max-width: 1260px; margin: 0 auto; padding: 16px; }
    .top {
      display: grid; grid-template-columns: 260px 1fr; gap: 12px;
      margin-bottom: 12px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid #d8e0d6;
      border-radius: 12px;
      padding: 12px;
      box-shadow: 0 4px 12px #00000010;
    }
    h1 { margin: 0 0 8px; font-size: 20px; letter-spacing: 0.2px; }
    .meta { color: var(--muted); font-size: 13px; }
    .main {
      display: grid; grid-template-columns: 1.2fr 1fr; gap: 12px;
      margin-bottom: 12px;
    }
    .log {
      max-height: 62vh; overflow-y: auto; display: grid; gap: 8px;
    }
    .event-card {
      border: 1px solid #d6e2dd;
      border-radius: 10px;
      overflow: hidden;
      background: #fff;
    }
    .event-head {
      display: flex; justify-content: space-between; align-items: center;
      background: #f2f7f5; padding: 8px 10px; font-size: 13px;
    }
    .event-body {
      display: grid; grid-template-columns: 170px 1fr; gap: 10px; padding: 8px;
    }
    .event-body img {
      width: 170px; height: 100px; object-fit: cover; border-radius: 6px; border: 1px solid #d6e2dd;
    }
    .kv { font-size: 13px; color: var(--ink); line-height: 1.4; }
    .small { font-size: 12px; color: var(--muted); }
    .dist-grid {
      display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px;
    }
    .bar { display: grid; grid-template-columns: 1fr 70px; gap: 8px; margin: 5px 0; align-items: center; }
    .bar-track { height: 9px; background: #e3ece8; border-radius: 999px; overflow: hidden; }
    .bar-fill { height: 100%; background: var(--bar); }
    select {
      width: 100%; padding: 8px; border-radius: 8px; border: 1px solid #c9d8d0; background: #fff;
    }
    .pill {
      display: inline-block; background: #e9f2f0; color: var(--accent); padding: 2px 8px; border-radius: 999px;
      font-size: 12px; margin-right: 4px;
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="top">
      <section class="panel">
        <h1>Live Runs</h1>
        <div class="small" style="margin-bottom:6px;">Auto-refresh every 2s</div>
        <select id="runSelect"></select>
      </section>
      <section class="panel">
        <h1 id="title">Worker Memory</h1>
        <div id="summary" class="meta">Waiting for data...</div>
      </section>
    </div>

    <div class="main">
      <section class="panel">
        <h1>Event Log + Frames</h1>
        <div id="events" class="log"></div>
      </section>
      <section class="panel">
        <h1>Labeled Episodes</h1>
        <div id="episodes" class="log"></div>
      </section>
    </div>

    <section class="panel">
      <h1>State Distributions</h1>
      <div class="dist-grid">
        <div><div class="small">Phase</div><div id="distPhase"></div></div>
        <div><div class="small">Action</div><div id="distAction"></div></div>
        <div><div class="small">Tool</div><div id="distTool"></div></div>
      </div>
    </section>
  </div>

  <script>
    const runSelect = document.getElementById("runSelect");
    let currentRun = null;

    async function fetchRuns() {
      const res = await fetch("/api/runs");
      const data = await res.json();
      const runs = data.runs || [];
      if (runs.length === 0) {
        runSelect.innerHTML = "<option>No runs found</option>";
        return;
      }
      const existing = new Set(Array.from(runSelect.options).map(o => o.value));
      for (const run of runs) {
        if (!existing.has(run)) {
          const opt = document.createElement("option");
          opt.value = run;
          opt.textContent = run;
          runSelect.appendChild(opt);
        }
      }
      if (!currentRun || !runs.includes(currentRun)) {
        currentRun = runs[runs.length - 1];
        runSelect.value = currentRun;
      }
    }

    function renderDistribution(containerId, dist) {
      const root = document.getElementById(containerId);
      const entries = Object.entries(dist || {});
      const total = entries.reduce((a, [,v]) => a + v, 0) || 1;
      root.innerHTML = "";
      for (const [name, count] of entries.slice(0, 8)) {
        const pct = Math.round((count / total) * 100);
        const row = document.createElement("div");
        row.className = "bar";
        row.innerHTML = `
          <div>
            <div class="small">${name}</div>
            <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
          </div>
          <div class="small">${count} (${pct}%)</div>
        `;
        root.appendChild(row);
      }
    }

    function renderEvents(events) {
      const root = document.getElementById("events");
      root.innerHTML = "";
      for (const e of events.slice().reverse()) {
        const card = document.createElement("div");
        card.className = "event-card";
        card.innerHTML = `
          <div class="event-head">
            <div>
              <span class="pill">${e.phase}</span>
              <span class="pill">${e.action}</span>
              <span class="pill">${e.tool}</span>
            </div>
            <div class="small">${e.t_start.toFixed(1)}s -> ${e.t_end.toFixed(1)}s</div>
          </div>
          <div class="event-body">
            <img src="${e.frame_url}" loading="lazy" />
            <div class="kv">
              <div><b>Event</b>: ${e.event_id}</div>
              <div><b>Confidence</b>: ${Number(e.confidence).toFixed(2)}</div>
              <div><b>Evidence</b>: ${e.evidence || ""}</div>
            </div>
          </div>
        `;
        root.appendChild(card);
      }
    }

    function renderEpisodes(episodes) {
      const root = document.getElementById("episodes");
      root.innerHTML = "";
      const items = episodes.slice().reverse();
      for (const ep of items) {
        const card = document.createElement("div");
        card.className = "event-card";
        card.innerHTML = `
          <div class="event-head">
            <div><span class="pill">${ep.kind || "episode"}</span> <b>${ep.label || "unknown"}</b></div>
            <div class="small">${ep.t_start.toFixed(1)}s -> ${ep.t_end.toFixed(1)}s</div>
          </div>
          <div class="event-body" style="grid-template-columns:1fr;">
            <div class="kv">
              <div><b>Episode</b>: ${ep.episode_id}</div>
              <div><b>Events</b>: ${(ep.event_ids || []).length}</div>
              <div><b>Confidence</b>: ${Number(ep.confidence || 0).toFixed(2)} (${ep.label_source || "n/a"})</div>
              <div><b>Reasoning</b>: ${ep.reasoning || ""}</div>
            </div>
          </div>
        `;
        root.appendChild(card);
      }
    }

    async function refreshSnapshot() {
      if (!currentRun) return;
      const res = await fetch(`/api/snapshot?run=${encodeURIComponent(currentRun)}&limit=80`);
      if (!res.ok) return;
      const data = await res.json();
      document.getElementById("title").textContent = `Worker Memory: ${data.run}`;
      document.getElementById("summary").textContent =
        `events=${data.event_count} | episodes=${data.episode_count} | windows=${data.window_count}`;
      renderEvents(data.events || []);
      renderEpisodes(data.labeled_episodes || []);
      renderDistribution("distPhase", data.distributions?.phase || {});
      renderDistribution("distAction", data.distributions?.action || {});
      renderDistribution("distTool", data.distributions?.tool || {});
    }

    runSelect.addEventListener("change", () => {
      currentRun = runSelect.value;
      refreshSnapshot();
    });

    async function tick() {
      await fetchRuns();
      await refreshSnapshot();
    }
    tick();
    setInterval(tick, 2000);
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-time dashboard for Big Brother outputs.")
    parser.add_argument("--outputs-dir", default="outputs", help="Directory containing per-run output artifacts.")
    parser.add_argument("--videos-dir", default="videos", help="Directory containing source videos.")
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard bind host.")
    parser.add_argument("--port", type=int, default=8008, help="Dashboard bind port.")
    args = parser.parse_args()

    app = create_app(outputs_dir=Path(args.outputs_dir), videos_dir=Path(args.videos_dir))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

