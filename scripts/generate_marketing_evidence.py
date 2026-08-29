import asyncio
import time
import pytest
from pytest_mockllm.core import TokenUsage, estimate_cost
from pytest_mockllm.providers.openai import OpenAIMock

async def benchmark_mock():
    mock = OpenAIMock()
    # Mocking a batch of 50 calls
    for i in range(50):
        mock.add_response(
            f"Response {i}",
            token_usage=TokenUsage(prompt_tokens=200, completion_tokens=500)
        )
    
    start_time = time.perf_counter()
    
    # Simulate application usage
    for i in range(50):
        # We simulate the overhead of the mock
        response = mock._get_next_response()
        # In a real app, this takes < 1ms
        
    end_time = time.perf_counter()
    mock_duration = end_time - start_time
    
    # Assumptions for "Real API"
    avg_real_api_latency = 1.5 # seconds
    real_duration = 50 * avg_real_api_latency
    
    # Calculate Savings
    time_saved_hours = (real_duration - mock_duration) / 3600
    
    prompt_tokens = 50 * 200
    completion_tokens = 50 * 500
    cost_saved = estimate_cost("gpt-4o", prompt_tokens, completion_tokens)
    
    report = f"""# Performance and ROI Benchmark

| Metric | Value |
| :--- | :--- |
| **Batch Size** | 50 LLM Calls |
| **Mock Execution Time** | {mock_duration*1000:.2f} ms |
| **Est. Real API Time** | {real_duration:.2f} s |
| **Speedup Factor** | ~{real_duration/mock_duration:,.0f}x |
| **Total Cost Saved** | ${cost_saved:.4f} |
| **Total Time Saved** | {real_duration - mock_duration:.2f} s |

## Projected Savings (1,000 builds)
- **Est. Money Saved**: ${cost_saved * 1000:,.2f}
- **Est. Time Saved**: {time_saved_hours * 1000:.1f} hours

*Generated on: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}*
"""
    
    import os
    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/BENCHMARK.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print(report)

if __name__ == "__main__":
    asyncio.run(benchmark_mock())
