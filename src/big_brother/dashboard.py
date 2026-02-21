from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
import uvicorn

from .nl_query import GeminiNLQueryEngine
from .query import QueryAPI
from .query_cli import run_query
from .storage import MemoryStore


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


def create_app(*, outputs_dir: Path, videos_dir: Path, nl_engine: GeminiNLQueryEngine | None = None) -> FastAPI:
    app = FastAPI(title="Big Brother Dashboard")
    engine = nl_engine

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

    @app.get("/api/query")
    def query(
        run: str = Query(...),
        query_type: str = Query(..., alias="type"),
        start_ts: float = Query(0.0),
        end_ts: float = Query(1e12),
        worker_id: str = Query("worker-1"),
        tool: str | None = Query(None),
        label: str | None = Query(None),
        limit: int = Query(200, ge=1, le=1000),
    ) -> JSONResponse:
        db_path = outputs_dir / run / "memory.db"
        if not db_path.exists():
            raise HTTPException(status_code=404, detail=f"No memory DB found for run '{run}' at {db_path}")
        store = MemoryStore(db_path=db_path)
        try:
            api = QueryAPI(store)
            result = run_query(
                api,
                query_type=query_type,
                start_ts=start_ts,
                end_ts=end_ts,
                worker_id=worker_id,
                tool=tool,
                label=label,
                limit=limit,
            )
            return JSONResponse({"run": run, "query_type": query_type, "result": result})
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            store.close()

    @app.get("/api/ask")
    def ask(
        run: str = Query(...),
        q: str = Query(..., min_length=2),
        worker_id: str = Query("worker-1"),
    ) -> JSONResponse:
        nonlocal engine
        db_path = outputs_dir / run / "memory.db"
        if not db_path.exists():
            raise HTTPException(status_code=404, detail=f"No memory DB found for run '{run}' at {db_path}")
        if engine is None:
            try:
                engine = GeminiNLQueryEngine()
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"NL engine unavailable: {exc}") from exc
        store = MemoryStore(db_path=db_path)
        try:
            api = QueryAPI(store)
            try:
                out = engine.ask(api=api, question=q, default_worker_id=worker_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"NL query failed: {exc}") from exc
            return JSONResponse(out)
        finally:
            store.close()

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
    .query-grid {
      display: grid; grid-template-columns: 160px 120px 120px 140px 1fr auto; gap: 8px; margin-top: 8px;
      align-items: center;
    }
    input, button {
      width: 100%; padding: 8px; border-radius: 8px; border: 1px solid #c9d8d0; background: #fff;
      color: var(--ink);
    }
    button {
      background: #1f7a7a; color: #fff; border: none; cursor: pointer;
      font-weight: 600;
    }
    pre {
      margin: 8px 0 0; padding: 10px; border-radius: 8px; border: 1px solid #d6e2dd;
      background: #f7fbfa; max-height: 220px; overflow: auto; font-size: 12px;
    }
    .ask-row { display: grid; grid-template-columns: 1fr auto; gap: 8px; margin-top: 10px; }
    .toggle-row { display: flex; align-items: center; gap: 8px; margin-top: 8px; }
    .toggle-row input { width: auto; }
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

    <section class="panel">
      <h1>Live Query</h1>
      <div class="query-grid">
        <select id="queryType">
          <option value="events">events</option>
          <option value="episodes">episodes</option>
          <option value="tool-usage">tool-usage</option>
          <option value="idle-ratio">idle-ratio</option>
          <option value="search-time">search-time</option>
        </select>
        <input id="startTs" type="number" step="0.1" value="0" placeholder="start ts"/>
        <input id="endTs" type="number" step="0.1" value="1000000000000" placeholder="end ts"/>
        <input id="limit" type="number" step="1" value="50" placeholder="limit"/>
        <input id="filterInput" type="text" placeholder="tool=nail_gun or label=framing_wall"/>
        <button id="runQueryBtn">Run Query</button>
      </div>
      <pre id="queryOutput">Run a query to inspect persisted memory.db results.</pre>
      <div class="ask-row">
        <input id="askInput" type="text" placeholder="Ask naturally: 'How much time was nail_gun used between 120s and 300s?'" />
        <button id="askBtn">Ask</button>
      </div>
      <div class="toggle-row">
        <input id="showReasoning" type="checkbox" />
        <label for="showReasoning" class="small">Show reasoning trace</label>
      </div>
      <div id="askStatus" class="small" style="margin-top:6px;"></div>
      <pre id="askOutput">Natural-language answers will appear here.</pre>
      <pre id="reasoningOutput" style="display:none;">Reasoning trace will appear here.</pre>
    </section>
  </div>

  <script>
    const runSelect = document.getElementById("runSelect");
    let currentRun = null;
    const queryType = document.getElementById("queryType");
    const startTs = document.getElementById("startTs");
    const endTs = document.getElementById("endTs");
    const limit = document.getElementById("limit");
    const filterInput = document.getElementById("filterInput");
    const runQueryBtn = document.getElementById("runQueryBtn");
    const queryOutput = document.getElementById("queryOutput");
    const askInput = document.getElementById("askInput");
    const askBtn = document.getElementById("askBtn");
    const askOutput = document.getElementById("askOutput");
    const askStatus = document.getElementById("askStatus");
    const showReasoning = document.getElementById("showReasoning");
    const reasoningOutput = document.getElementById("reasoningOutput");

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

    async function runLiveQuery() {
      if (!currentRun) return;
      const params = new URLSearchParams();
      params.set("run", currentRun);
      params.set("type", queryType.value);
      params.set("start_ts", startTs.value || "0");
      params.set("end_ts", endTs.value || "1000000000000");
      params.set("limit", limit.value || "50");

      const raw = (filterInput.value || "").trim();
      if (raw.includes("=")) {
        const [k, ...rest] = raw.split("=");
        const v = rest.join("=").trim();
        if (k.trim() === "tool" && v) params.set("tool", v);
        if (k.trim() === "label" && v) params.set("label", v);
      }

      const res = await fetch(`/api/query?${params.toString()}`);
      const data = await res.json();
      queryOutput.textContent = JSON.stringify(data, null, 2);
    }

    async function askNaturalLanguage() {
      if (!currentRun) return;
      const q = (askInput.value || "").trim();
      if (!q) return;
      const params = new URLSearchParams();
      params.set("run", currentRun);
      params.set("q", q);
      askBtn.disabled = true;
      askBtn.textContent = "Asking...";
      askStatus.textContent = "Running natural-language query...";
      try {
        const res = await fetch(`/api/ask?${params.toString()}`);
        const raw = await res.text();
        let data = null;
        try {
          data = JSON.parse(raw);
        } catch (_e) {
          data = { detail: raw };
        }
        if (!res.ok) {
          askOutput.textContent = `Error: ${data.detail || "Request failed"}`;
          askStatus.textContent = "Query failed.";
          reasoningOutput.style.display = "none";
          reasoningOutput.textContent = "";
          return;
        }
        askOutput.textContent = data.answer || "No answer generated.";
        if (showReasoning.checked && data.reasoning_trace) {
          reasoningOutput.style.display = "block";
          reasoningOutput.textContent = data.reasoning_trace;
        } else {
          reasoningOutput.style.display = "none";
          reasoningOutput.textContent = "";
        }
        askStatus.textContent = "Query completed.";
      } catch (err) {
        askOutput.textContent = `Error: ${err}`;
        askStatus.textContent = "Query failed.";
        reasoningOutput.style.display = "none";
        reasoningOutput.textContent = "";
      } finally {
        askBtn.disabled = false;
        askBtn.textContent = "Ask";
      }
    }

    runSelect.addEventListener("change", () => {
      currentRun = runSelect.value;
      refreshSnapshot();
    });
    runQueryBtn.addEventListener("click", runLiveQuery);
    askBtn.addEventListener("click", askNaturalLanguage);
    showReasoning.addEventListener("change", () => {
      if (!showReasoning.checked) {
        reasoningOutput.style.display = "none";
      }
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
