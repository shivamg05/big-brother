#!/usr/bin/env python3
"""
Comprehensive benchmark queries to test BigBrother's capabilities
and identify where current systems fail.
"""

BENCHMARK_QUERIES = {

    # ========== TEMPORAL PRECISION QUERIES ==========
    "temporal_precision": [
        "When was the nail gun first used?",
        "What happened at exactly 2:30?",
        "How many times did the worker switch tools between 100-150 seconds?",
        "What was the longest continuous period of using the same tool?",
        "At what times did the worker take breaks?",
        "When did the worker transition from preparation to execution phase?",
        "What was happening 10 seconds before the first nail gun use?",
        "How long did the worker spend idle in the first 5 minutes?",
    ],

    # ========== COMPLEX PATTERN RECOGNITION ==========
    "pattern_recognition": [
        "What repetitive workflow patterns can you identify?",
        "Which tools are used together most frequently?",
        "What is the typical sequence of actions when using the nail gun?",
        "Are there any inefficient movement patterns?",
        "What actions always precede tool changes?",
        "Identify all instances where the worker had to redo work",
        "What patterns indicate the worker is about to take a break?",
        "Which tasks take longer than expected and why?",
    ],

    # ========== PRODUCTIVITY ANALYSIS ==========
    "productivity_analysis": [
        "What were the main productivity bottlenecks?",
        "Compare the first 100 seconds to the last 100 seconds - which was more productive?",
        "What percentage of time was spent idle vs actively working?",
        "Calculate the ratio of preparation time to execution time",
        "Which hour of the day showed highest productivity?",
        "How much time was wasted due to poor tool organization?",
        "What's the average time between productive actions?",
        "Identify the top 3 time-wasting activities",
    ],

    # ========== COMPARATIVE ANALYSIS ==========
    "comparative_analysis": [
        "Compare morning efficiency vs afternoon efficiency",
        "Which tool had the highest utilization rate?",
        "Compare the worker's speed at the beginning vs end of shift",
        "Which phase (prepare/execute/travel) consumed most time?",
        "Compare nail gun usage frequency vs hammer usage",
        "Which tasks improved in speed over time?",
        "Compare solo work efficiency vs collaborative work",
        "Which 5-minute window was most productive and why?",
    ],

    # ========== SAFETY & COMPLIANCE ==========
    "safety_compliance": [
        "Were there any unsafe tool handling instances?",
        "How many times did the worker work without proper PPE?",
        "Identify moments when safety protocols were violated",
        "When did the worker work in hazardous positions?",
        "Were tools left in unsafe locations?",
        "How many near-miss incidents occurred?",
        "Did the worker follow proper lifting techniques?",
        "Were there any moments of dangerous tool combinations?",
    ],

    # ========== TOOL & MATERIAL TRACKING ==========
    "tool_material_tracking": [
        "How many different tools were used in total?",
        "What materials were handled most frequently?",
        "Which tool was abandoned most often?",
        "Track the movement of lumber pieces throughout the video",
        "How many times were tools misplaced?",
        "Which materials were wasted or discarded?",
        "What's the average time a tool remains in use once picked up?",
        "Identify all tool malfunctions or issues",
    ],

    # ========== COLLABORATION & COMMUNICATION ==========
    "collaboration": [
        "How many other workers were present and when?",
        "When did collaborative work occur?",
        "How many times did workers communicate?",
        "Which tasks required multiple people?",
        "When did the worker receive or give instructions?",
        "Identify moments of workflow conflicts between workers",
        "How did collaboration affect productivity?",
        "When did the worker work alone vs with others?",
    ],

    # ========== COMPLEX REASONING QUERIES ==========
    "complex_reasoning": [
        "Why did the worker switch from nail gun to hammer at minute 3?",
        "What caused the 30-second delay at timestamp 250?",
        "Predict what tool the worker will need next based on current pattern",
        "What would happen if the worker had a drill instead of just a hammer?",
        "Explain the worker's strategy for organizing the workspace",
        "Why was productivity lower in the middle section of the video?",
        "What external factors affected the worker's efficiency?",
        "Infer the worker's skill level based on their actions",
    ],

    # ========== WORKFLOW OPTIMIZATION ==========
    "workflow_optimization": [
        "How could the worker save 20% of time with better organization?",
        "What's the optimal tool arrangement to minimize travel time?",
        "Which repetitive tasks could be batched for efficiency?",
        "Suggest a better sequence of actions for the observed tasks",
        "How much time could be saved with a second nail gun?",
        "Identify the critical path in the workflow",
        "What pre-positioning of materials would improve flow?",
        "Which breaks were necessary vs unnecessary?",
    ],

    # ========== ANOMALY DETECTION ==========
    "anomaly_detection": [
        "Identify any unusual or unexpected events",
        "When did the worker deviate from normal patterns?",
        "Were there any tool usage anomalies?",
        "Find instances where expected actions didn't occur",
        "Detect any suspicious gaps in activity",
        "When did the workflow break down?",
        "Identify outlier events in terms of duration",
        "What unexpected obstacles did the worker encounter?",
    ],

    # ========== QUANTITATIVE METRICS ==========
    "quantitative_metrics": [
        "Calculate total distance traveled by the worker",
        "What's the average duration of each nail gun use?",
        "How many nails were used in total?",
        "Calculate the work completion rate per hour",
        "What's the tool switching frequency per minute?",
        "Measure the average idle time between tasks",
        "How many discrete tasks were completed?",
        "Calculate the efficiency score for each 50-second window",
    ],

    # ========== SEQUENTIAL DEPENDENCIES ==========
    "sequential_dependencies": [
        "What tasks must be completed before using the nail gun?",
        "Identify all prerequisite actions for each major task",
        "Which actions always follow lumber positioning?",
        "What's the critical sequence that can't be reordered?",
        "Find all causal relationships between events",
        "Which tasks could be done in parallel?",
        "What's the dependency graph of the observed workflow?",
        "Identify blocking tasks that prevented progress",
    ],
}

# ========== QUERIES THAT WILL FAIL WITH STANDARD NLP ==========
FAILURE_CASES = [
    # Requires precise temporal grounding
    "What happened in the 3 seconds before each tool change?",
    "Find all events that lasted exactly 5-7 seconds",

    # Requires complex state tracking
    "How many times did the worker return to a previously abandoned task?",
    "Track the complete lifecycle of each piece of lumber",

    # Requires causal reasoning
    "Why did productivity drop after the first break?",
    "What caused the worker to switch strategies mid-task?",

    # Requires predictive modeling
    "Based on current pace, when will the task be completed?",
    "Predict the next 3 tools the worker will use",

    # Requires multi-hop reasoning
    "Find all instances where poor planning led to rework within 30 seconds",
    "Which tool changes were unnecessary based on upcoming tasks?",

    # Requires understanding implicit information
    "When was the worker frustrated or stressed?",
    "Identify moments of uncertainty or confusion",

    # Requires comparative analysis across time
    "How did the worker's technique improve throughout the day?",
    "Which mistakes were repeated vs learned from?",
]

def test_query(query: str, api_endpoint: str = "http://localhost:8008/api/ask"):
    """Test a single query against the BigBrother API."""
    import requests
    import json

    params = {
        "run": "juan",
        "q": query
    }

    try:
        response = requests.get(api_endpoint, params=params, timeout=30)
        data = response.json()

        return {
            "query": query,
            "success": response.status_code == 200,
            "used_smart_agent": data.get("used_smart_agent", False),
            "answer": data.get("answer", "No answer"),
            "confidence": len(data.get("answer", "")) > 50,  # Simple heuristic
            "response_time": response.elapsed.total_seconds()
        }
    except Exception as e:
        return {
            "query": query,
            "success": False,
            "error": str(e)
        }

def run_benchmark(category: str = None):
    """Run benchmark queries and generate report."""
    import time
    from datetime import datetime

    results = {}

    categories = [category] if category else BENCHMARK_QUERIES.keys()

    for cat in categories:
        print(f"\n{'='*60}")
        print(f"Testing category: {cat.upper()}")
        print(f"{'='*60}")

        results[cat] = []
        queries = BENCHMARK_QUERIES.get(cat, [])

        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] {query}")
            result = test_query(query)
            results[cat].append(result)

            if result["success"]:
                print(f"✓ Success (Smart Agent: {result['used_smart_agent']}, Time: {result['response_time']:.2f}s)")
                print(f"  Answer preview: {result['answer'][:100]}...")
            else:
                print(f"✗ Failed: {result.get('error', 'Unknown error')}")

            time.sleep(1)  # Rate limiting

    # Generate report
    generate_report(results)

    return results

def generate_report(results):
    """Generate a comprehensive benchmark report."""

    print("\n" + "="*60)
    print("BENCHMARK REPORT")
    print("="*60)

    total_queries = 0
    successful_queries = 0
    smart_agent_queries = 0
    total_time = 0

    for category, category_results in results.items():
        cat_success = sum(1 for r in category_results if r.get("success", False))
        cat_smart = sum(1 for r in category_results if r.get("used_smart_agent", False))
        cat_time = sum(r.get("response_time", 0) for r in category_results)

        print(f"\n{category.upper()}:")
        print(f"  Success Rate: {cat_success}/{len(category_results)} ({cat_success/len(category_results)*100:.1f}%)")
        print(f"  Smart Agent Used: {cat_smart}/{len(category_results)} ({cat_smart/len(category_results)*100:.1f}%)")
        print(f"  Avg Response Time: {cat_time/len(category_results):.2f}s")

        total_queries += len(category_results)
        successful_queries += cat_success
        smart_agent_queries += cat_smart
        total_time += cat_time

    print("\n" + "-"*60)
    print("OVERALL METRICS:")
    print(f"  Total Queries: {total_queries}")
    print(f"  Success Rate: {successful_queries}/{total_queries} ({successful_queries/total_queries*100:.1f}%)")
    print(f"  Smart Agent Usage: {smart_agent_queries}/{total_queries} ({smart_agent_queries/total_queries*100:.1f}%)")
    print(f"  Total Time: {total_time:.1f}s")
    print(f"  Avg Time per Query: {total_time/total_queries:.2f}s")

    # Save results to file
    import json
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to benchmark_results.json")

if __name__ == "__main__":
    # Run full benchmark
    # results = run_benchmark()

    # Or test specific category
    # results = run_benchmark("temporal_precision")

    # Or test failure cases
    print("\nTesting queries that will fail with standard NLP:")
    for query in FAILURE_CASES[:3]:
        print(f"\n• {query}")
        result = test_query(query)
        if result["success"]:
            print(f"  → {result['answer'][:150]}...")
        else:
            print(f"  → Failed: {result.get('error', 'No answer')}")