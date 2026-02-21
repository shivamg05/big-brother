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
        answer = self._results_to_nl(question=question, structured=structured, result=result)
        return {
            "question": question,
            "structured_query": structured,
            "result": result,
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

    def _results_to_nl(self, *, question: str, structured: dict[str, Any], result: dict[str, Any] | list[dict[str, Any]]) -> str:
        result_preview = result
        if isinstance(result, list) and len(result) > 25:
            result_preview = result[-25:]
        prompt = (
            "Answer the user question using the query result. "
            "Be concise and factual. Mention uncertainty if result is sparse. "
            f"Question: {question}\n"
            f"Structured Query: {json.dumps(structured, separators=(',', ':'))}\n"
            f"Query Result: {json.dumps(result_preview, separators=(',', ':'))}"
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

