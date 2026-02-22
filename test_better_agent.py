#!/usr/bin/env python3
"""Test the BetterSmartAgent with tool calling"""

import sys
import json
import os
sys.path.insert(0, 'src')

from big_brother.better_smart_agent import BetterSmartAgent

# Test the agent
db_path = "outputs/juan/memory.db"
os.environ["GOOGLE_API_KEY"] = "AIzaSyCsm7gAGh4tb2kVOCWV1vmiDQQV-pxP-Jc"

print("🚀 Testing BetterSmartAgent with Tool Calling")
print("=" * 60)

# Test queries
test_queries = [
    "When was the nail gun first used?",
    "What percentage of time was spent idle?",
    "What repetitive patterns can you identify?",
    "What were the productivity bottlenecks?"
]

agent = BetterSmartAgent(db_path=db_path)

for question in test_queries:
    print(f"\n❓ Question: {question}")
    print("-" * 40)

    try:
        result = agent.ask(question, show_reasoning=True)

        print(f"✅ Success!")

        if result.get("tool_calls"):
            print(f"\n📞 Tool Calls Made:")
            for call in result["tool_calls"]:
                print(f"  • {call['function']}({call.get('args', {})})")

        if result.get("reasoning"):
            print(f"\n🧠 Reasoning Steps:")
            for step in result["reasoning"]:
                print(f"  • {step}")

        if result.get("tool_results"):
            print(f"\n🔧 Tool Results Summary:")
            for tool, data in result["tool_results"].items():
                if isinstance(data, list):
                    print(f"  • {tool}: {len(data)} results")
                elif isinstance(data, dict):
                    print(f"  • {tool}: {data}")

        print(f"\n📝 Answer:")
        print(f"  {result.get('answer', 'No answer')[:300]}...")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("✅ Testing Complete!")