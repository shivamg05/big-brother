"""
Enhanced SQL Agent that generates multiple queries and fallbacks for better answers.
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


class BetterSQLAgent:
    """SQL agent that generates multiple queries for comprehensive answers."""

    def __init__(self, *, db_path: str, api_key: str | None = None, max_sql_attempts: int = 3):
        self.db_path = db_path
        self.api_key = api_key
        self.max_sql_attempts = max_sql_attempts
        self._init_client()

    def _init_client(self) -> None:
        load_dotenv()
        api_key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing Gemini API key for SQL querying.")
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("google-genai is not installed") from exc
        self.client = genai.Client(api_key=api_key)

    def ask(self, *, question: str, default_worker_id: str | None = None) -> dict[str, Any]:
        """Generate multiple SQL queries for comprehensive answers."""

        # Create logs directory
        log_dir = Path(self.db_path).parent / "sql_logs"
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = log_dir / f"sql_{timestamp}.txt"

        log_content = []
        log_content.append("=" * 80)
        log_content.append(f"🎯 BETTER SQL AGENT - {datetime.now().isoformat()}")
        log_content.append("=" * 80)
        log_content.append(f"📝 Question: {question}")

        # Generate MULTIPLE SQL queries
        multi_query_prompt = f"""
You are an expert SQLite analyst. Generate MULTIPLE SQL queries to answer this question comprehensively.

Database schema:
CREATE TABLE Events (
    event_id TEXT PRIMARY KEY,
    worker_id TEXT,
    t_start REAL,
    t_end REAL,
    tool TEXT,
    action TEXT,
    phase TEXT,
    confidence REAL,
    evidence TEXT
);

Question: {question}

For questions about "longest continuous tool use", generate THREE queries:
1. Simple max duration of single events
2. Continuous periods where same tool is used with gaps < 30s
3. Total accumulated time per tool

For questions about "what happened at X", generate TWO queries:
1. Exact moment query
2. Context query (events 10s before/after)

For questions about "top/most used", generate TWO queries:
1. By count of events
2. By total duration

Return JSON with structure:
{{
    "queries": [
        {{"sql": "...", "purpose": "..."}},
        {{"sql": "...", "purpose": "..."}}
    ],
    "reasoning": "..."
}}

Default worker: {default_worker_id or 'juan'}
"""

        log_content.append("\n🔮 Generating multiple SQL queries...")

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=multi_query_prompt,
                config={"response_mime_type": "application/json"}
            )
            query_plan = json.loads(response.text)
            log_content.append(f"Generated {len(query_plan.get('queries', []))} queries")

        except Exception as e:
            log_content.append(f"❌ Failed to generate queries: {e}")
            # Fallback to single query
            query_plan = {
                "queries": [{
                    "sql": f"SELECT * FROM Events WHERE worker_id = '{default_worker_id or 'juan'}' LIMIT 10",
                    "purpose": "fallback"
                }]
            }

        # Execute ALL queries
        all_results = {}
        for i, query_obj in enumerate(query_plan.get("queries", []), 1):
            sql = query_obj.get("sql", "").strip()
            purpose = query_obj.get("purpose", "unknown")

            log_content.append(f"\n📊 Query {i}: {purpose}")
            log_content.append(f"SQL: {sql}")

            try:
                results = self._execute_readonly_sql(sql)
                all_results[purpose] = results
                log_content.append(f"✅ Found {len(results)} results")

                if results and len(results) > 0:
                    log_content.append(f"Sample: {json.dumps(results[0], default=str)[:200]}")

            except Exception as e:
                log_content.append(f"❌ Failed: {e}")
                all_results[purpose] = []

        # Synthesize answer from ALL query results
        synthesis_prompt = f"""
Based on these query results, provide a comprehensive answer to: {question}

Results:
{json.dumps(all_results, default=str, indent=2)[:5000]}

Rules:
- If looking for continuous periods, check all result sets
- Prefer results with longer durations over single events
- Format times as MM:SS
- Be specific with numbers
- If no good data, say "insufficient data"
"""

        log_content.append("\n🧠 Synthesizing answer from all queries...")

        try:
            answer_response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=synthesis_prompt
            )
            answer = answer_response.text.strip()
        except:
            answer = "Failed to synthesize answer"

        log_content.append(f"💬 Final Answer: {answer}")

        # Save log
        with open(log_file, 'w') as f:
            f.write('\n'.join(log_content))

        print(f"📁 SQL log saved to: {log_file}")

        return {
            "sql_queries": query_plan.get("queries", []),
            "results": all_results,
            "answer": answer,
            "used_smart_agent": True
        }

    def _execute_readonly_sql(self, query: str) -> list[dict[str, Any]]:
        """Execute read-only SQL query."""
        # Safety check
        normalized = query.strip().lower()
        if not (normalized.startswith("select") or normalized.startswith("with")):
            raise ValueError("Only SELECT/WITH queries allowed")

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(query)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()