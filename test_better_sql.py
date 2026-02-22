#!/usr/bin/env python3
"""Test the better SQL agent with multiple queries."""

import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyB-nJKX7JSJItUXgZjWGxqjcGVTJ0l6JKo"

from src.big_brother.better_sql_agent import BetterSQLAgent

def test_queries():
    agent = BetterSQLAgent(db_path="outputs/juan/memory.db")

    questions = [
        "What was the longest continuous tool use?",
        "What are the top 3 most used tools?",
        "What happened at 2:30?"
    ]

    for q in questions:
        print("\n" + "="*80)
        print(f"❓ Question: {q}")
        print("="*80)

        try:
            result = agent.ask(question=q, default_worker_id="juan")

            print(f"\n📊 Generated {len(result.get('sql_queries', []))} queries")

            for query in result.get('sql_queries', []):
                print(f"  - {query.get('purpose', 'unknown')}")

            print(f"\n✅ Answer: {result['answer']}")

        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_queries()