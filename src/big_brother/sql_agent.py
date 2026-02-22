"""
SQL Query Agent: Single LLM agent that generates and executes SQL queries.
Clean implementation with no hardcoded triggers or multiple agents.
"""

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

try:
    import google.genai as genai
except ImportError:
    import google.generativeai as genai


@dataclass
class SQLAgent:
    """Single agent that uses LLM to generate SQL queries for answering questions."""

    db_path: str
    api_key: Optional[str] = None

    def __init__(self, db_path: str, api_key: Optional[str] = None):
        self.db_path = db_path
        load_dotenv()
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing API key for SQL Agent")

        # Configure Gemini
        try:
            self.client = genai.Client(api_key=self.api_key)
            self.model_name = "gemini-2.5-flash"
        except:
            genai.configure(api_key=self.api_key)
            self.client = genai
            self.model_name = "gemini-2.5-flash"

    def get_schema_context(self) -> str:
        """Get database schema and recent episodes for context."""
        conn = sqlite3.connect(self.db_path)

        # Get schema
        schema = """
        Database Schema:

        Events table:
        - event_id TEXT PRIMARY KEY
        - worker_id TEXT
        - t_start REAL (seconds from video start)
        - t_end REAL (seconds from video start)
        - phase TEXT (prepare|execute|travel|idle)
        - action TEXT (nail|cut|carry|align|measure|etc)
        - tool TEXT (nail_gun|hammer|saw|none|etc)
        - materials TEXT (JSON array)
        - confidence REAL
        - evidence TEXT (description)

        Episodes table:
        - episode_id TEXT PRIMARY KEY
        - worker_id TEXT
        - t_start REAL
        - t_end REAL
        - label TEXT (task description)
        - reasoning TEXT
        """

        # Get some sample episodes for semantic context
        cursor = conn.execute("""
            SELECT t_start, t_end, label
            FROM Episodes
            ORDER BY t_start
            LIMIT 10
        """)
        episodes = cursor.fetchall()
        conn.close()

        if episodes:
            schema += "\n\nRecent Episodes (for context):\n"
            for start, end, label in episodes:
                mins_start = int(start // 60)
                secs_start = int(start % 60)
                mins_end = int(end // 60)
                secs_end = int(end % 60)
                schema += f"- {mins_start}:{secs_start:02d} to {mins_end}:{secs_end:02d}: {label}\n"

        return schema

    def execute_sql(self, query: str) -> List[Dict]:
        """Execute SQL query and return results."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.execute(query)
            results = [dict(row) for row in cursor.fetchall()]
            return results
        except Exception as e:
            return [{"error": str(e), "query": query}]
        finally:
            conn.close()

    def ask(self, question: str) -> Dict[str, Any]:
        """Answer a question by generating and executing SQL queries."""

        print("\n" + "="*80)
        print("🎯 BIG BROTHER SQL AGENT - NEW QUERY")
        print("="*80)
        print(f"📝 User Question: {question}")
        print("-"*80)

        # Get schema context
        schema = self.get_schema_context()

        # Generate SQL query using LLM
        prompt = f"""
        {schema}

        User Question: {question}

        Generate a SQL query to answer this question.

        Important notes:
        - Times are stored as seconds from video start (e.g., 150.0 = 2:30)
        - For time-based questions like "What happened at 2:30?", convert to seconds (2*60 + 30 = 150)
        - Use appropriate WHERE clauses for time ranges
        - Join tables when needed for comprehensive answers
        - Order results by t_start for chronological order

        Return a JSON object with:
        {{
            "sql": "the SQL query",
            "explanation": "brief explanation of what the query does"
        }}
        """

        print("🔧 Calling Gemini 2.5 Flash to generate SQL...")

        # Get SQL from LLM
        try:
            if hasattr(self.client, 'models'):
                # New API
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={"response_mime_type": "application/json"}
                )
            else:
                # Old API
                model = genai.GenerativeModel(self.model_name)
                response = model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"}
                )

            query_plan = json.loads(response.text)
            sql_query = query_plan.get("sql", "")

            print(f"\n🤖 Generated SQL Query:")
            print("-"*40)
            print(sql_query)
            print("-"*40)

            print(f"\n📊 Executing SQL against database...")

            # Execute the SQL
            results = self.execute_sql(sql_query)

            print(f"✅ Found {len(results)} results")
            if results and len(results) > 0:
                print(f"📋 First result: {json.dumps(results[0], default=str, indent=2)}")

            # Generate natural language answer
            answer_prompt = f"""
            Question: {question}

            SQL Query: {sql_query}

            Query Results: {json.dumps(results[:50], default=str)}

            Provide a clear, concise answer to the original question based on these results.
            Include specific timestamps and details from the data.
            Format timestamps as MM:SS (e.g., 2:30 for 150 seconds).
            """

            print(f"\n🗣️ Generating natural language answer...")

            if hasattr(self.client, 'models'):
                answer_response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=answer_prompt
                )
            else:
                model = genai.GenerativeModel(self.model_name)
                answer_response = model.generate_content(answer_prompt)

            print(f"\n💬 Answer: {answer_response.text[:200]}..." if len(answer_response.text) > 200 else f"\n💬 Answer: {answer_response.text}")
            print("="*80 + "\n")

            return {
                "question": question,
                "sql_query": sql_query,
                "result_count": len(results),
                "results": results[:10],  # Limit results in response
                "answer": answer_response.text,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "question": question,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }