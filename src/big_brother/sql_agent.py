from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from dotenv import load_dotenv

from .schema import Action, Phase, Tool


TOOL_ALIASES = {
    "nail gun": "nail_gun",
    "nailgun": "nail_gun",
    "tape measure": "tape_measure",
    "measuring tape": "tape_measure",
    "impact driver": "impact_driver",
    "circular saw": "circular_saw",
    "miter saw": "miter_saw",
}


@dataclass(slots=True)
class SQLAgent:
    """LLM-backed agent that generates read-only SQL and answers questions from memory.db."""

    db_path: str
    model: str = "gemini-2.5-flash"
    api_key: str | None = None
    client: Any | None = None
    max_sql_attempts: int = 3

    def __post_init__(self) -> None:
        if self.client is not None:
            return
        load_dotenv()
        api_key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing Gemini API key for SQL querying.")
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("google-genai is not installed. Add dependency: google-genai") from exc
        self.client = genai.Client(api_key=api_key)

    def ask(self, *, question: str, default_worker_id: str | None = None) -> dict[str, Any]:
        # Create logs directory if it doesn't exist
        from pathlib import Path
        log_dir = Path(self.db_path).parent / "sql_logs"
        log_dir.mkdir(exist_ok=True)

        # Create log file for this query
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = log_dir / f"sql_{timestamp}.txt"

        # Start logging
        log_content = []
        log_content.append("="*80)
        log_content.append(f"🎯 SQL QUERY LOG - {datetime.now().isoformat()}")
        log_content.append("="*80)
        log_content.append(f"📝 Question: {question}")
        log_content.append("-"*80)

        schema = self._schema_context()
        worker_clause = (
            f"Use worker_id = '{default_worker_id}' when question does not explicitly request another worker or all workers."
            if default_worker_id
            else ""
        )
        prompt = (
            "You are an expert SQLite analyst for a construction activity memory database.\n"
            "Return only JSON with keys: sql, explanation.\n"
            "Rules:\n"
            "- Generate a single read-only SQLite query.\n"
            "- Prefer SELECT or WITH queries only.\n"
            "- Convert MM:SS timestamps to seconds when needed (2:30 => 150).\n"
            "- Include ORDER BY t_start for event lists.\n"
            "- Add LIMIT when returning raw rows (<= 200).\n"
            f"- {worker_clause}\n\n"
            f"{schema}\n\n"
            f"Question: {question}"
        )
        sql_query = ""
        results: list[dict[str, Any]] = []
        last_sql_error: Exception | None = None

        for attempt in range(1, self.max_sql_attempts + 1):
            log_content.append(f"\n🔧 Attempt {attempt}/{self.max_sql_attempts}")
            try:
                log_content.append("  Calling LLM to generate SQL...")
                query_plan = self._generate_json(prompt)
                log_content.append(f"  LLM Response: {json.dumps(query_plan, indent=2)}")
            except Exception as exc:
                log_content.append(f"  ❌ LLM Error: {str(exc)}")
                last_sql_error = exc
                if attempt == self.max_sql_attempts:
                    break
                prompt = self._repair_prompt(
                    schema=schema,
                    question=question,
                    previous_sql=sql_query,
                    previous_error=str(exc),
                    default_worker_id=default_worker_id,
                )
                continue
            sql_query = str(query_plan.get("sql", "")).strip()
            log_content.append(f"\n🤖 Generated SQL Query:")
            log_content.append("-"*40)
            log_content.append(sql_query)
            log_content.append("-"*40)

            if not sql_query:
                log_content.append("  ❌ SQL query is empty!")
                last_sql_error = RuntimeError("SQL agent returned an empty SQL query")
                prompt = self._repair_prompt(
                    schema=schema,
                    question=question,
                    previous_sql=sql_query,
                    previous_error=str(last_sql_error),
                    default_worker_id=default_worker_id,
                )
                continue
            try:
                log_content.append("\n📊 Executing SQL...")
                results = self._execute_readonly_sql(sql_query)
                log_content.append(f"  ✅ Success! Found {len(results)} results")
                if results and len(results) > 0:
                    log_content.append(f"  First result: {json.dumps(results[0], default=str, indent=2)}")
                last_sql_error = None
                break
            except Exception as exc:
                log_content.append(f"  ❌ SQL Execution Error: {str(exc)}")
                last_sql_error = exc
                if attempt == self.max_sql_attempts:
                    break
                prompt = self._repair_prompt(
                    schema=schema,
                    question=question,
                    previous_sql=sql_query,
                    previous_error=str(exc),
                    default_worker_id=default_worker_id,
                )
        if last_sql_error is not None:
            log_content.append(f"\n❌ FAILED after {self.max_sql_attempts} attempts: {last_sql_error}")
            # Save log file even on failure
            with open(log_file, 'w') as f:
                f.write('\n'.join(log_content))
            print(f"📁 SQL log saved to: {log_file}")
            raise RuntimeError(f"SQL agent failed after {self.max_sql_attempts} attempts: {last_sql_error}") from last_sql_error

        answer_prompt = (
            "Answer the question from SQL output. Keep it concise and factual.\n"
            "If no rows were returned, say that directly.\n"
            "Format times as MM:SS when possible.\n"
            f"Question: {question}\n"
            f"SQL: {sql_query}\n"
            f"Rows: {json.dumps(results[:50], separators=(',', ':'), default=str)}"
        )

        log_content.append("\n🗣️ Generating natural language answer...")
        answer = self._generate_text(answer_prompt)
        log_content.append(f"💬 Answer: {answer}")

        # Log full results
        log_content.append("\n" + "="*80)
        log_content.append("📋 FULL RESULTS (up to 50)")
        log_content.append("="*80)
        log_content.append(json.dumps(results[:50], default=str, indent=2))

        # Save the complete log file
        with open(log_file, 'w') as f:
            f.write('\n'.join(log_content))

        print(f"📁 SQL log saved to: {log_file}")

        return {
            "question": question,
            "sql_query": sql_query,
            "result_count": len(results),
            "results": results[:10],
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "engine": "sql_agent",
            "sql_attempts": self.max_sql_attempts if last_sql_error is not None else attempt,
        }

    def _repair_prompt(
        self,
        *,
        schema: str,
        question: str,
        previous_sql: str,
        previous_error: str,
        default_worker_id: str | None,
    ) -> str:
        worker_clause = (
            f"Use worker_id = '{default_worker_id}' when question does not explicitly request another worker or all workers."
            if default_worker_id
            else ""
        )
        return (
            "Fix the SQL query using the execution error. Return only JSON with keys: sql, explanation.\n"
            "Rules:\n"
            "- Output one read-only SQLite query.\n"
            "- Use SELECT or WITH.\n"
            "- Keep column/table names exactly as schema defines.\n"
            f"- {worker_clause}\n\n"
            f"{schema}\n\n"
            f"Question: {question}\n"
            f"Previous SQL: {previous_sql}\n"
            f"Execution error: {previous_error}"
        )

    def _schema_context(self) -> str:
        conn = sqlite3.connect(self.db_path)
        try:
            tables = ["Events", "Episodes"]
            schema_lines: list[str] = ["Database schema:"]
            for table in tables:
                columns = conn.execute(f"PRAGMA table_info({table})").fetchall()
                if not columns:
                    continue
                schema_lines.append(f"- {table}:")
                for col in columns:
                    schema_lines.append(f"  - {col[1]} {col[2]}")

            bounds = conn.execute(
                """
                SELECT
                    MIN(t_start) AS min_t_start,
                    MAX(t_end) AS max_t_end,
                    COUNT(*) AS event_count
                FROM Events
                """
            ).fetchone()
            if bounds and bounds[2]:
                schema_lines.append(
                    "Observed event time range:"
                    f" start={float(bounds[0]):.3f}s, end={float(bounds[1]):.3f}s, rows={int(bounds[2])}"
                )

            workers = conn.execute(
                """
                SELECT worker_id, COUNT(*) AS c
                FROM Events
                GROUP BY worker_id
                ORDER BY c DESC
                LIMIT 10
                """
            ).fetchall()
            if workers:
                schema_lines.append("Workers in Events (top by rows):")
                for worker_id, count in workers:
                    schema_lines.append(f"- {worker_id}: {int(count)}")

            schema_lines.append("Canonical enums:")
            schema_lines.append(f"- phase: {', '.join(v.value for v in Phase)}")
            schema_lines.append(f"- action: {', '.join(v.value for v in Action)}")
            schema_lines.append(f"- tool: {', '.join(v.value for v in Tool)}")
            schema_lines.append(
                "- tool_aliases: "
                + ", ".join([f"{k}=>{v}" for k, v in sorted(TOOL_ALIASES.items(), key=lambda x: x[0])])
            )

            schema_lines.append("SQL patterns:")
            schema_lines.append(
                "- recent_events: SELECT t_start,t_end,phase,action,tool,worker_id FROM Events "
                "WHERE worker_id='X' ORDER BY t_start DESC LIMIT 50"
            )
            schema_lines.append(
                "- tool_usage_seconds: SELECT tool, SUM(MAX(t_end - t_start, 0.0)) AS seconds "
                "FROM Events WHERE worker_id='X' GROUP BY tool ORDER BY seconds DESC LIMIT 20"
            )
            schema_lines.append(
                "- labeled_episodes: SELECT t_start,t_end,label,confidence FROM Episodes "
                "WHERE worker_id='X' ORDER BY t_start DESC LIMIT 50"
            )

            episodes = conn.execute(
                """
                SELECT t_start, t_end, label
                FROM Episodes
                ORDER BY t_start DESC
                LIMIT 10
                """
            ).fetchall()
            if episodes:
                schema_lines.append("Recent episodes:")
                for t_start, t_end, label in episodes:
                    schema_lines.append(f"- {int(float(t_start))}-{int(float(t_end))}: {label}")
            return "\n".join(schema_lines)
        finally:
            conn.close()

    def _execute_readonly_sql(self, query: str) -> list[dict[str, Any]]:
        query = query.strip().rstrip(";")
        if not query:
            raise ValueError("Generated SQL query is empty")

        normalized = re.sub(r"\s+", " ", query).strip().lower()
        if not (normalized.startswith("select") or normalized.startswith("with")):
            raise ValueError("Only SELECT/WITH read-only SQL statements are allowed")

        forbidden = {
            "insert",
            "update",
            "delete",
            "drop",
            "alter",
            "create",
            "attach",
            "detach",
            "pragma",
            "vacuum",
            "reindex",
            "analyze",
            "replace",
        }
        tokens = set(re.findall(r"[a-z_]+", normalized))
        if tokens & forbidden:
            raise ValueError("Generated SQL includes forbidden write/ddl keywords")

        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(query).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def _generate_json(self, prompt: str) -> dict[str, Any]:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("SQL planner returned empty response")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"SQL planner returned invalid JSON: {text}") from exc

    def _generate_text(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"temperature": 0.2},
        )
        text = getattr(response, "text", None)
        if not text:
            return "No answer generated."
        return str(text).strip()
