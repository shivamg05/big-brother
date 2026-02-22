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
        worker_id: str | None = Query(None),
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
            resolved_worker_id = worker_id or run
            result = run_query(
                api,
                query_type=query_type,
                start_ts=start_ts,
                end_ts=end_ts,
                worker_id=resolved_worker_id,
                tool=tool,
                label=label,
                limit=limit,
            )
            # Backward compatibility for older runs that used worker-1.
            if (query_type in {"events", "episodes"} and isinstance(result, list) and not result) or (
                isinstance(result, dict)
                and query_type in {"tool-usage", "idle-ratio", "search-time"}
                and all(float(v) == 0.0 for v in result.values())
            ):
                if resolved_worker_id != "worker-1":
                    result = run_query(
                        api,
                        query_type=query_type,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        worker_id="worker-1",
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
        worker_id: str | None = Query(None),
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
            resolved_worker_id = worker_id or run
            try:
                out = engine.ask(api=api, question=q, default_worker_id=resolved_worker_id)
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
  <title>Worker Memory Dashboard</title>
  <style>
    :root {
      --bg: #f8f5ef;
      --surface: #fdfbf7;
      --surface-2: #f4f0e8;
      --text: #22201d;
      --muted: #6f6961;
      --border: #e5dfd5;
      --accent: #486b5f;
      --accent-soft: #e9f0ed;
      --shadow: 0 6px 20px rgba(25, 20, 12, 0.06);
      --radius: 14px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Inter", "Avenir Next", "Segoe UI", sans-serif;
      line-height: 1.45;
    }
    .container {
      max-width: 1240px;
      margin: 0 auto;
      padding: 28px 22px 40px;
      min-width: 0;
    }
    .title {
      font-family: "Iowan Old Style", "Baskerville", "Times New Roman", serif;
      font-size: clamp(32px, 4vw, 46px);
      line-height: 1.15;
      margin: 0;
      letter-spacing: -0.02em;
      font-weight: 500;
    }
    .subtitle {
      margin-top: 10px;
      color: var(--muted);
      font-size: 14px;
      letter-spacing: 0.02em;
    }
    .grid-top {
      margin-top: 18px;
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
    }
    .grid-main {
      margin-top: 14px;
      display: grid;
      grid-template-columns: 1.3fr 1fr;
      gap: 14px;
    }
    .stack {
      margin-top: 14px;
      display: grid;
      gap: 14px;
    }
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 20px;
      min-width: 0;
    }
    .card h2 {
      margin: 0;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
      font-weight: 600;
    }
    .meta {
      margin-top: 8px;
      color: var(--muted);
      font-size: 14px;
    }
    .select, .input, .btn, .btn-outline {
      width: 100%;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: #fff;
      padding: 10px 14px;
      color: var(--text);
      font-size: 14px;
      transition: all 140ms ease;
      outline: none;
    }
    .select:focus, .input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(72, 107, 95, 0.15);
    }
    .btn {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
      cursor: pointer;
      font-weight: 600;
    }
    .btn:disabled {
      opacity: 0.65;
      cursor: not-allowed;
    }
    .btn-outline {
      background: #fff;
      border-color: #ccc4b8;
      cursor: pointer;
    }
    .btn:hover, .btn-outline:hover, .card:hover {
      transform: translateY(-1px);
    }
    .log {
      margin-top: 12px;
      max-height: 62vh;
      overflow: auto;
      display: grid;
      gap: 10px;
    }
    .item {
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      transition: border-color 140ms ease, box-shadow 140ms ease;
    }
    .item:hover { border-color: #d5cebf; }
    .item-head {
      background: var(--surface-2);
      padding: 9px 11px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      border-bottom: 1px solid var(--border);
    }
    .item-body {
      padding: 11px;
      display: grid;
      grid-template-columns: 170px 1fr;
      gap: 10px;
    }
    .item-body img {
      width: 170px;
      height: 102px;
      border-radius: 8px;
      border: 1px solid var(--border);
      object-fit: cover;
      background: #f3efe8;
    }
    .chips { display: flex; gap: 6px; flex-wrap: wrap; }
    .chip {
      display: inline-flex;
      align-items: center;
      padding: 3px 9px;
      border-radius: 999px;
      border: 1px solid #d8d2c8;
      background: #f7f3ea;
      font-size: 12px;
      color: #564f45;
    }
    .kv { font-size: 13px; color: #3e3a34; }
    .small { font-size: 12px; color: var(--muted); }
    .status-row {
      margin-top: 10px;
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      color: var(--muted);
    }
    .status-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
      background: #c1b8a7;
    }
    .status-dot.live { background: #7d8f4f; }
    .status-dot.done { background: #486b5f; }
    .dist {
      margin-top: 10px;
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 12px;
    }
    .bar-row { margin-bottom: 7px; }
    .bar-line {
      height: 7px;
      border-radius: 999px;
      background: #ece6dc;
      overflow: hidden;
      margin-top: 3px;
    }
    .bar-fill {
      height: 100%;
      background: var(--accent);
    }
    .ask-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
    }
    .chat-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 14px;
      transition: opacity 180ms ease, transform 180ms ease;
      opacity: 0.98;
    }
    .chat-meta {
      margin-top: 8px;
      font-size: 12px;
      color: var(--muted);
    }
    .chat-thread {
      margin-top: 10px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: #fcfaf6;
      padding: 10px;
      height: 280px;
      overflow-y: auto;
      overflow-x: hidden;
      display: grid;
      gap: 8px;
    }
    .chat-msg {
      border: 1px solid var(--border);
      background: #fff;
      border-radius: 10px;
      padding: 8px 10px;
      min-width: 0;
    }
    .chat-msg.user {
      background: var(--accent-soft);
      border-color: #cfddd7;
    }
    .chat-role {
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 4px;
      font-weight: 600;
    }
    .chat-text {
      font-size: 14px;
      color: #2e2a25;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .chat-empty {
      font-size: 13px;
      color: var(--muted);
      align-self: center;
      justify-self: center;
      text-align: center;
      padding: 12px;
    }
    .chat-input-wrap .input::placeholder {
      color: rgba(255,255,255,0.82);
    }
    pre {
      margin: 10px 0 0;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: #fcfaf6;
      padding: 12px;
      overflow-x: auto;
      overflow-y: auto;
      max-height: 220px;
      min-width: 0;
      width: 100%;
      color: #4e473f;
      font-size: 12px;
      line-height: 1.5;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .skeleton {
      height: 100px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: linear-gradient(90deg, #f6f2ea 25%, #f0ebdf 37%, #f6f2ea 63%);
      background-size: 400% 100%;
      animation: shimmer 1.3s ease infinite;
    }
    @keyframes shimmer {
      0% { background-position: 100% 0; }
      100% { background-position: 0 0; }
    }
    @media (max-width: 1024px) {
      .grid-top, .grid-main { grid-template-columns: 1fr; }
      .item-body { grid-template-columns: 1fr; }
      .item-body img { width: 100%; height: 180px; }
      .dist { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div id="root"></div>
  <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <script type="text/babel">
    const { useEffect, useMemo, useState } = React;

    function fmtTime(seconds) {
      const total = Math.max(0, Math.floor(Number(seconds) || 0));
      const h = Math.floor(total / 3600);
      const m = Math.floor((total % 3600) / 60);
      const s = total % 60;
      const mm = String(m).padStart(2, "0");
      const ss = String(s).padStart(2, "0");
      return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
    }

    function fmtLabel(label) {
      const text = String(label || "unknown").replaceAll("_", " ").trim();
      if (!text) return "Unknown";
      return text.split(/\\s+/).map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
    }

    function DistributionCard({ title, dist }) {
      const entries = Object.entries(dist || {});
      const total = entries.reduce((a, [,v]) => a + v, 0) || 1;
      return (
        <div>
          <div className="small" style={{ textTransform: "uppercase", letterSpacing: "0.08em" }}>{title}</div>
          <div style={{ marginTop: 8 }}>
            {entries.slice(0, 8).map(([name, count]) => {
              const pct = Math.round((count / total) * 100);
              return (
                <div className="bar-row" key={name}>
                  <div className="small">{name} · {count} ({pct}%)</div>
                  <div className="bar-line"><div className="bar-fill" style={{ width: `${pct}%` }} /></div>
                </div>
              );
            })}
          </div>
        </div>
      );
    }

    function EventCard({ e }) {
      return (
        <div className="item">
          <div className="item-head">
            <div className="chips">
              <span className="chip">Phase: {e.phase}</span>
              <span className="chip">Action: {e.action}</span>
              <span className="chip">Tool: {e.tool}</span>
            </div>
            <div className="small">{fmtTime(e.t_start)} -> {fmtTime(e.t_end)}</div>
          </div>
          <div className="item-body">
            <img src={e.frame_url} loading="lazy" />
            <div className="kv">
              <div>{e.evidence || ""}</div>
            </div>
          </div>
        </div>
      );
    }

    function EpisodeCard({ ep }) {
      return (
        <div className="item">
          <div className="item-head">
            <div style={{ fontWeight: 700, fontSize: 15 }}>{fmtLabel(ep.label)}</div>
            <div className="small">{fmtTime(ep.t_start)} -> {fmtTime(ep.t_end)}</div>
          </div>
          <div style={{ padding: 11 }} className="kv">
            <div>{ep.reasoning || "No reasoning available."}</div>
          </div>
        </div>
      );
    }

    function App() {
      const [runs, setRuns] = useState([]);
      const [run, setRun] = useState("");
      const [snapshot, setSnapshot] = useState(null);
      const [loadingSnapshot, setLoadingSnapshot] = useState(true);

      const [askInput, setAskInput] = useState("");
      const [askStatus, setAskStatus] = useState("");
      const [asking, setAsking] = useState(false);
      const [chatHistory, setChatHistory] = useState([]);
      const [showReasoning, setShowReasoning] = useState(false);
      const [messageId, setMessageId] = useState(1);

      useEffect(() => {
        let alive = true;
        async function tick() {
          try {
            const runsRes = await fetch("/api/runs");
            const runsData = await runsRes.json();
            if (!alive) return;
            const nextRuns = runsData.runs || [];
            setRuns((prev) => {
              if (prev.length === nextRuns.length && prev.every((v, i) => v === nextRuns[i])) return prev;
              return nextRuns;
            });
            if (!run && nextRuns.length > 0) setRun(nextRuns[nextRuns.length - 1]);
          } catch (_e) {}
        }
        tick();
        const id = setInterval(tick, 6000);
        return () => { alive = false; clearInterval(id); };
      }, [run]);

      useEffect(() => {
        const node = document.getElementById("chat-thread");
        if (node) node.scrollTop = node.scrollHeight;
      }, [chatHistory, asking]);

      useEffect(() => {
        setChatHistory([]);
        setAskStatus("");
      }, [run]);

      useEffect(() => {
        let alive = true;
        async function loadSnapshot(showLoading) {
          if (!run) return;
          if (showLoading) setLoadingSnapshot(true);
          try {
            const res = await fetch(`/api/snapshot?run=${encodeURIComponent(run)}&limit=80`);
            if (!res.ok) return;
            const data = await res.json();
            if (!alive) return;
            setSnapshot((prev) => {
              if (!prev) return data;
              const same =
                prev.event_count === data.event_count &&
                prev.episode_count === data.episode_count &&
                prev.window_count === data.window_count &&
                prev.run === data.run;
              return same ? prev : data;
            });
          } finally {
            if (alive && showLoading) setLoadingSnapshot(false);
          }
        }
        loadSnapshot(true);
        const id = setInterval(() => loadSnapshot(false), 5000);
        return () => { alive = false; clearInterval(id); };
      }, [run]);

      async function runAsk() {
        if (!run) return;
        const q = (askInput || "").trim();
        if (!q) return;
        const userMessageId = messageId;
        setMessageId((v) => v + 1);
        setChatHistory((prev) => [...prev, { id: userMessageId, role: "user", text: q }]);
        setAskInput("");
        setAsking(true);
        setAskStatus("Running natural-language query...");
        try {
          const params = new URLSearchParams();
          params.set("run", run);
          params.set("q", q);
          const res = await fetch(`/api/ask?${params.toString()}`);
          const raw = await res.text();
          let data = {};
          try { data = JSON.parse(raw); } catch { data = { detail: raw }; }
          if (!res.ok) {
            const errId = userMessageId + 1000000;
            setChatHistory((prev) => [
              ...prev,
              { id: errId, role: "assistant", text: `Error: ${data.detail || "Request failed"}`, reasoning: "" },
            ]);
            setAskStatus("Query failed.");
            return;
          }
          const assistantId = userMessageId + 2000000;
          setChatHistory((prev) => [
            ...prev,
            {
              id: assistantId,
              role: "assistant",
              text: data.answer || "No answer generated.",
              reasoning: data.reasoning_trace || "",
            },
          ]);
          setAskStatus("Query completed.");
        } catch (e) {
          const errId = userMessageId + 3000000;
          setChatHistory((prev) => [
            ...prev,
            { id: errId, role: "assistant", text: `Error: ${e}`, reasoning: "" },
          ]);
          setAskStatus("Query failed.");
        } finally {
          setAsking(false);
        }
      }

      const events = useMemo(() => (snapshot?.events || []).slice().reverse(), [snapshot]);
      const episodes = useMemo(() => (snapshot?.labeled_episodes || []).slice().reverse(), [snapshot]);
      const isFinished = Boolean(snapshot?.summary && Object.keys(snapshot.summary).length > 0);
      const statusText = !run
        ? "No worker selected"
        : (isFinished ? "Work finished" : "Currently working");
      const statusClass = !run ? "" : (isFinished ? "done" : "live");

      return (
        <div className="container">
          <h1 className="title">Worker Memory</h1>
          <div className="subtitle">Real-time event memory, episode labels, and query interface.</div>

          <div className="grid-top">
            <section className="card">
              <h2>Select Worker</h2>
              <div style={{ marginTop: 12 }}>
                <select className="select" value={run} onChange={(e) => setRun(e.target.value)}>
                  {runs.length === 0 ? <option>No runs found</option> : runs.map((r) => <option key={r} value={r}>{r}</option>)}
                </select>
              </div>
              <div className="status-row">
                <span className={`status-dot ${statusClass}`}></span>
                <span>{statusText}</span>
              </div>
            </section>

            <section className="chat-card">
              <div className="chat-input-wrap">
                <div className="ask-row" style={{ marginTop: 0, gap: 6, gridTemplateColumns: "1fr 44px" }}>
                  <input
                    className="input"
                    value={askInput}
                    onChange={(e) => setAskInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter" && !asking) runAsk(); }}
                    placeholder="How long did the worker use a tape measure and why?"
                    style={{
                      background: "var(--accent)",
                      borderColor: "var(--accent)",
                      color: "#fff",
                      boxShadow: "none",
                    }}
                  />
                  <button
                    className="btn"
                    disabled={asking}
                    onClick={runAsk}
                    style={{
                      padding: "10px 0",
                      borderRadius: "999px",
                      background: "var(--accent)",
                      borderColor: "var(--accent)",
                      color: "#fff",
                      fontSize: "18px",
                      lineHeight: 1,
                      fontWeight: 700,
                    }}
                  >
                    {asking ? "…" : "→"}
                  </button>
                </div>
                <div style={{ marginTop: 10, display: "flex", alignItems: "center", gap: 8 }}>
                  <input id="showReasoning" type="checkbox" checked={showReasoning} onChange={(e) => setShowReasoning(e.target.checked)} />
                  <label htmlFor="showReasoning" className="small">Show reasoning trace</label>
                </div>
                <div className="chat-meta">{askStatus}</div>
              </div>
              <div id="chat-thread" className="chat-thread">
                {chatHistory.length === 0 ? <div className="chat-empty">Ask a question to start the conversation.</div> : null}
                {chatHistory.map((m) => (
                  <div key={m.id} className={`chat-msg ${m.role === "user" ? "user" : ""}`}>
                    <div className="chat-role">{m.role === "user" ? "You" : "Assistant"}</div>
                    <div className="chat-text">{m.text}</div>
                    {m.role !== "user" && showReasoning && m.reasoning ? (
                      <pre style={{ marginTop: 8 }}>
{`Reasoning
${m.reasoning}`}
                      </pre>
                    ) : null}
                  </div>
                ))}
                {asking ? (
                  <div className="chat-msg">
                    <div className="chat-role">Assistant</div>
                    <div className="chat-text">Thinking...</div>
                  </div>
                ) : null}
              </div>
            </section>
          </div>

          <div className="grid-main">
            <section className="card">
              <h2>Event log</h2>
              <div className="log">
                {loadingSnapshot && <><div className="skeleton" /><div className="skeleton" /></>}
                {!loadingSnapshot && events.map((e) => <EventCard key={e.event_id} e={e} />)}
              </div>
            </section>
            <section className="card">
              <h2>Episodes</h2>
              <div className="log">
                {loadingSnapshot && <><div className="skeleton" /><div className="skeleton" /></>}
                {!loadingSnapshot && episodes.map((ep) => <EpisodeCard key={`${ep.episode_id}-${ep.kind || "k"}`} ep={ep} />)}
              </div>
            </section>
          </div>

          <div className="stack">
            <section className="card">
              <h2>State Distributions</h2>
              <div className="dist">
                <DistributionCard title="Phase" dist={snapshot?.distributions?.phase || {}} />
                <DistributionCard title="Action" dist={snapshot?.distributions?.action || {}} />
                <DistributionCard title="Tool" dist={snapshot?.distributions?.tool || {}} />
              </div>
            </section>
          </div>

        </div>
      );
    }

    const root = ReactDOM.createRoot(document.getElementById("root"));
    root.render(<App />);
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
