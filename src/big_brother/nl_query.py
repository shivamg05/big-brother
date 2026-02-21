from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from .query import QueryAPI
from .query_cli import run_query
from .schema import Tool

TOOL_ALIASES = {
    "nail gun": "nail_gun",
    "nailgun": "nail_gun",
    "drill": "drill",
    "saw": "saw",
    "hammer": "hammer",
    "tape measure": "tape_measure",
    "measuring tape": "tape_measure",
}

PURPOSE_BY_ACTION = {
    "nail": "fastening/framing",
    "fasten": "fastening/framing",
    "measure": "layout/measurement",
    "cut": "cutting material",
    "drill": "drilling/anchoring",
    "align": "alignment/setup",
    "inspect": "inspection/verification",
    "walk": "movement/transport",
    "carry": "material transport",
}


@dataclass(slots=True)
class GeminiNLQueryEngine:
    model: str = "gemini-2.5-flash"
    api_key: str | None = None
    client: Any | None = None
    requests_per_minute: int = 20
    max_retries: int = 4
    _last_request_ts: float = 0.0

    def __post_init__(self) -> None:
        if self.client is not None:
            return
        load_dotenv()
        api_key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing Gemini API key for NL querying.")
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("google-genai is not installed. Add dependency: google-genai") from exc
        self.client = genai.Client(api_key=api_key)

    def ask(
        self,
        *,
        api: QueryAPI,
        question: str,
        default_worker_id: str = "worker-1",
    ) -> dict[str, Any]:
        structured = self._nl_to_structured(question=question, default_worker_id=default_worker_id)
        structured = self._postprocess_structured(
            question=question,
            structured=structured,
            default_worker_id=default_worker_id,
        )
        result = run_query(
            api,
            query_type=str(structured["query_type"]),
            start_ts=float(structured["start_ts"]),
            end_ts=float(structured["end_ts"]),
            worker_id=str(structured.get("worker_id", default_worker_id)),
            tool=structured.get("tool"),
            label=structured.get("label"),
            limit=int(structured.get("limit", 200)),
        )
        context = self._build_context(
            api=api,
            structured=structured,
            primary_result=result,
            question=question,
        )
        deterministic = self._deterministic_answer(question=question, structured=structured, context=context)
        answer = deterministic or self._results_to_nl(question=question, structured=structured, result=result, context=context)
        return {
            "question": question,
            "structured_query": structured,
            "result": result,
            "context": context,
            "answer": answer,
        }

    def _nl_to_structured(self, *, question: str, default_worker_id: str) -> dict[str, Any]:
        prompt = (
            "Convert the question to one query object. "
            "Valid query_type: events|episodes|tool-usage|idle-ratio|search-time. "
            "Return JSON keys: query_type,start_ts,end_ts,worker_id,tool,label,limit. "
            "Use null for tool/label when not applicable. "
            f"default worker_id is {default_worker_id}. "
            f"Question: {question}"
        )
        response = self._generate_with_backoff(
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": {
                    "type": "object",
                    "properties": {
                        "query_type": {
                            "type": "string",
                            "enum": ["events", "episodes", "tool-usage", "idle-ratio", "search-time"],
                        },
                        "start_ts": {"type": "number"},
                        "end_ts": {"type": "number"},
                        "worker_id": {"type": "string"},
                        "tool": {"type": ["string", "null"]},
                        "label": {"type": ["string", "null"]},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
                    },
                    "required": ["query_type", "start_ts", "end_ts", "worker_id", "tool", "label", "limit"],
                    "additionalProperties": False,
                },
                "temperature": 0.0,
            },
        )
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("NL query parser returned empty response.")
        out = json.loads(text)
        if out["end_ts"] < out["start_ts"]:
            out["start_ts"], out["end_ts"] = out["end_ts"], out["start_ts"]
        return out

    def _results_to_nl(
        self,
        *,
        question: str,
        structured: dict[str, Any],
        result: dict[str, Any] | list[dict[str, Any]],
        context: dict[str, Any],
    ) -> str:
        result_preview = result
        if isinstance(result, list) and len(result) > 25:
            result_preview = result[-25:]
        prompt = (
            "Answer the user question using the query result. "
            "Be concise and factual. Mention uncertainty if result is sparse. "
            f"Question: {question}\n"
            f"Structured Query: {json.dumps(structured, separators=(',', ':'))}\n"
            f"Query Result: {json.dumps(result_preview, separators=(',', ':'))}\n"
            f"Derived Context: {json.dumps(context, separators=(',', ':'))}"
        )
        response = self._generate_with_backoff(
            contents=prompt,
            config={
                "temperature": 0.2,
            },
        )
        text = getattr(response, "text", None)
        if not text:
            return "No answer generated."
        return text.strip()

    def _postprocess_structured(
        self,
        *,
        question: str,
        structured: dict[str, Any],
        default_worker_id: str,
    ) -> dict[str, Any]:
        out = dict(structured)
        out["worker_id"] = str(out.get("worker_id") or default_worker_id)
        tool = out.get("tool")
        normalized_tool = self._normalize_tool(tool)
        if normalized_tool is None:
            normalized_tool = self._infer_tool_from_question(question)
        out["tool"] = normalized_tool

        q = question.lower()
        asks_where = "where" in q or "location" in q
        asks_purpose = "purpose" in q or "for what" in q or "what for" in q
        asks_walking_duration = ("walk" in q or "walking" in q or "travel" in q) and (
            "how long" in q or "duration" in q or "time" in q
        )
        asks_tool_inventory = (
            "what are the tools" in q
            or "which tools" in q
            or "tools being used" in q
            or "tools are being used" in q
        )
        if (asks_where or asks_purpose) and out.get("tool"):
            out["query_type"] = "events"
            out["limit"] = max(int(out.get("limit", 200)), 300)
        if asks_walking_duration:
            out["query_type"] = "events"
            out["tool"] = None
            out["limit"] = max(int(out.get("limit", 200)), 500)
        if asks_tool_inventory:
            out["query_type"] = "events"
            out["tool"] = None
            out["limit"] = max(int(out.get("limit", 200)), 500)
        if out.get("query_type") == "tool-usage" and not out.get("tool"):
            inferred = self._infer_tool_from_question(question)
            if inferred:
                out["tool"] = inferred
            else:
                out["query_type"] = "events"
                out["limit"] = max(int(out.get("limit", 200)), 500)
        return out

    @staticmethod
    def _normalize_tool(tool: Any) -> str | None:
        if not isinstance(tool, str):
            return None
        text = tool.strip().lower()
        text = re.sub(r"[^a-z0-9_ -]", "", text)
        text = text.replace("-", " ")
        if not text:
            return None
        if text in TOOL_ALIASES:
            return TOOL_ALIASES[text]
        text_u = text.replace(" ", "_")
        known = {t.value for t in Tool}
        if text_u in known:
            return text_u
        return None

    @staticmethod
    def _infer_tool_from_question(question: str) -> str | None:
        q = re.sub(r"[^a-z0-9_ -]", " ", question.lower())
        for alias, canonical in TOOL_ALIASES.items():
            if alias in q:
                return canonical
        return None

    def _build_context(
        self,
        *,
        api: QueryAPI,
        structured: dict[str, Any],
        primary_result: dict[str, Any] | list[dict[str, Any]],
        question: str,
    ) -> dict[str, Any]:
        start_ts = float(structured["start_ts"])
        end_ts = float(structured["end_ts"])
        worker_id = str(structured["worker_id"])
        tool = structured.get("tool")

        context: dict[str, Any] = {}
        if isinstance(primary_result, dict) and "tool_usage_seconds" in primary_result:
            context["tool_usage_seconds"] = float(primary_result["tool_usage_seconds"])

        events = api.get_events(start_ts=start_ts, end_ts=end_ts, worker_id=worker_id)
        if tool:
            events = [e for e in events if str(e.get("tool")) == str(tool)]
        context["matching_event_count"] = len(events)
        if not events:
            return context

        locations: dict[str, int] = {}
        actions: dict[str, int] = {}
        tools: dict[str, int] = {}
        tool_seconds: dict[str, float] = {}
        materials: dict[str, int] = {}
        total_s = 0.0
        for e in events:
            loc = str(e.get("location_hint", "unknown"))
            act = str(e.get("action", "unknown"))
            tool_name = str(e.get("tool", "unknown"))
            locations[loc] = locations.get(loc, 0) + 1
            actions[act] = actions.get(act, 0) + 1
            tools[tool_name] = tools.get(tool_name, 0) + 1
            mats_raw = e.get("materials", "[]")
            try:
                mats = json.loads(mats_raw) if isinstance(mats_raw, str) else []
            except json.JSONDecodeError:
                mats = []
            for m in mats:
                m_s = str(m)
                materials[m_s] = materials.get(m_s, 0) + 1
            dur = max(0.0, float(e.get("t_end", 0.0)) - float(e.get("t_start", 0.0)))
            total_s += dur
            tool_seconds[tool_name] = tool_seconds.get(tool_name, 0.0) + dur
        q = question.lower()
        if "walk" in q or "walking" in q or "travel" in q:
            walk_s = 0.0
            total_all_s = 0.0
            for e in api.get_events(start_ts=start_ts, end_ts=end_ts, worker_id=worker_id):
                dur = max(0.0, float(e.get("t_end", 0.0)) - float(e.get("t_start", 0.0)))
                total_all_s += dur
                action = str(e.get("action", ""))
                phase = str(e.get("phase", ""))
                if action == "walk" or phase == "travel":
                    walk_s += dur
            context["walk_seconds"] = walk_s
            context["total_seconds"] = total_all_s
            context["walk_ratio"] = (walk_s / total_all_s) if total_all_s > 0 else 0.0
        context["usage_seconds_from_events"] = total_s
        context["top_tools"] = sorted(tools.items(), key=lambda x: x[1], reverse=True)[:5]
        context["tool_seconds"] = dict(sorted(tool_seconds.items(), key=lambda x: x[1], reverse=True))
        context["top_locations"] = sorted(locations.items(), key=lambda x: x[1], reverse=True)[:3]
        context["top_actions"] = sorted(actions.items(), key=lambda x: x[1], reverse=True)[:3]
        context["top_materials"] = sorted(materials.items(), key=lambda x: x[1], reverse=True)[:3]
        return context

    def _deterministic_answer(self, *, question: str, structured: dict[str, Any], context: dict[str, Any]) -> str | None:
        q = question.lower()
        asks_where = "where" in q or "location" in q
        asks_purpose = "purpose" in q or "for what" in q or "what for" in q
        asks_walk_duration = ("walk" in q or "walking" in q or "travel" in q) and (
            "how long" in q or "duration" in q or "time" in q
        )
        asks_tool_inventory = (
            "what are the tools" in q
            or "which tools" in q
            or "tools being used" in q
            or "tools are being used" in q
        )
        if asks_tool_inventory:
            tool_seconds = context.get("tool_seconds", {})
            if not isinstance(tool_seconds, dict) or not tool_seconds:
                return "No tool usage events were found in the selected time range."
            parts = []
            for tool_name, seconds in list(tool_seconds.items())[:5]:
                parts.append(f"{tool_name}: {float(seconds):.1f}s")
            return "Tools observed in this range: " + "; ".join(parts) + "."
        if asks_walk_duration:
            walk_s = float(context.get("walk_seconds", 0.0))
            total_s = float(context.get("total_seconds", 0.0))
            ratio = float(context.get("walk_ratio", 0.0)) * 100.0
            if total_s <= 0:
                return "Walking duration could not be computed because no events were found in the selected range."
            return f"The worker is walking/traveling for about {walk_s:.1f}s ({ratio:.1f}% of {total_s:.1f}s observed)."
        if not (asks_where or asks_purpose):
            return None
        tool = structured.get("tool")
        count = int(context.get("matching_event_count", 0))
        if count == 0:
            if tool:
                return f"No instances of `{tool}` were found in the selected time range."
            return "No matching events were found in the selected time range."
        seconds = float(context.get("usage_seconds_from_events", context.get("tool_usage_seconds", 0.0)))
        top_locations = context.get("top_locations", [])
        top_actions = context.get("top_actions", [])
        top_materials = context.get("top_materials", [])

        loc_text = " and ".join([str(x[0]) for x in top_locations[:2]]) if top_locations else "unknown areas"
        purposes = []
        for action, _ in top_actions:
            purposes.append(PURPOSE_BY_ACTION.get(str(action), str(action)))
        purpose_text = ", ".join(dict.fromkeys(purposes)) if purposes else "unclear purpose"
        material_text = ", ".join([str(x[0]) for x in top_materials[:3]]) if top_materials else "unspecified materials"
        tool_text = str(tool) if tool else "the tool"
        return (
            f"`{tool_text}` appears in {count} events (~{seconds:.1f}s). "
            f"It is mostly used in {loc_text}, primarily for {purpose_text}, "
            f"often around {material_text}."
        )

    def _generate_with_backoff(self, *, contents: Any, config: dict[str, Any]) -> Any:
        if self.client is None:
            raise RuntimeError("Gemini client is not initialized")
        attempt = 0
        while True:
            self._respect_rate_limit()
            try:
                response = self.client.models.generate_content(model=self.model, contents=contents, config=config)
                self._last_request_ts = time.monotonic()
                return response
            except Exception as exc:
                self._last_request_ts = time.monotonic()
                if not self._is_rate_limit_error(exc):
                    raise
                attempt += 1
                if attempt > self.max_retries:
                    raise
                time.sleep(self._retry_delay(exc) or self._min_interval_seconds())

    def _respect_rate_limit(self) -> None:
        min_interval = self._min_interval_seconds()
        if min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    def _min_interval_seconds(self) -> float:
        if self.requests_per_minute <= 0:
            return 0.0
        return 60.0 / float(self.requests_per_minute)

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        text = str(exc).lower()
        return status == 429 or "resource_exhausted" in text or "quota exceeded" in text

    @staticmethod
    def _retry_delay(exc: Exception) -> float | None:
        text = str(exc)
        m = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", text, flags=re.IGNORECASE)
        if m:
            return max(0.1, float(m.group(1)))
        return None
