"""
LLM Latency Benchmarking Script
Measures and reports key latency metrics for large language models (LLMs) during inference.
1. Measures time-to-first-token (TTFT)
2. Measures time-per-token during generation
3. Calculates tokens per second throughput
4. Separates prefill vs decode phases
5. Reports p50, p90, p99 latencies

These metrics matter for applications like chatbots, code generation, and content creation
 where responsiveness is critical and user experience depends on low latency and high throughput.
It helps identify bottlenecks and optimize model deployment for real-time use cases.

usage:
    python llm_latency_benchmark.py

Author: Hamna
Target: NVIDIA LLM Optimization
"""

import time
import numpy as np
from collections import defaultdict


class LLMLatencyBenchmark:
    def __init__(self):
        self.metrics = defaultdict(list)

    def simulate_llm_inference(self, prompt_tokens, output_tokens):
        """
        Simulate LLM inference with realistic timing

        Prefill (process prompt): ~0.5ms per token
        Decode (generate): ~10ms per token
        """
        # Prefill phase - process all prompt tokens at once
        prefill_time = prompt_tokens * 0.0005  # 0.5ms per token
        time.sleep(prefill_time)

        # First token latency = prefill time
        first_token_latency = prefill_time

        # Decode phase - generate one token at a time
        token_latencies = []
        for _ in range(output_tokens):
            decode_time = 0.010 + np.random.normal(0, 0.001)  # 10ms +/- 1ms
            time.sleep(max(0.001, decode_time))
            token_latencies.append(decode_time)

        return first_token_latency, token_latencies

    def run_benchmark(self, num_runs=100, prompt_tokens=512, output_tokens=128):
        """Run multiple inference iterations and collect metrics"""
        print(f"Running {num_runs} iterations...")
        print(f"Prompt tokens: {prompt_tokens}, Output tokens: {output_tokens}\n")

        for i in range(num_runs):
            start = time.time()

            # Run inference
            first_token_lat, token_lats = self.simulate_llm_inference(prompt_tokens, output_tokens)

            end = time.time()
            total_time = end - start

            # Record metrics
            self.metrics["first_token_latency"].append(first_token_lat * 1000)  # ms
            self.metrics["decode_latencies"].extend([t * 1000 for t in token_lats])  # ms
            self.metrics["total_latency"].append(total_time * 1000)  # ms
            self.metrics["tokens_per_second"].append(output_tokens / total_time)

            if (i + 1) % 20 == 0:
                print(f"Completed {i + 1}/{num_runs} runs")

    def calculate_percentiles(self, data, metric_name):
        """Calculate p50, p90, p99 percentiles"""
        p50 = np.percentile(data, 50)
        p90 = np.percentile(data, 90)
        p99 = np.percentile(data, 99)

        print(f"\n{metric_name}:")
        print(f"  P50 (median): {p50:.2f} ms")
        print(f"  P90:          {p90:.2f} ms")
        print(f"  P99:          {p99:.2f} ms")
        print(f"  Min:          {min(data):.2f} ms")
        print(f"  Max:          {max(data):.2f} ms")
        print(f"  Mean:         {np.mean(data):.2f} ms")
        print(f"  Std Dev:      {np.std(data):.2f} ms")

        return p50, p90, p99

    def report_metrics(self):
        """Generate comprehensive latency report"""
        print("\n" + "=" * 60)
        print("LATENCY BENCHMARK RESULTS")
        print("=" * 60)

        # Time to First Token (TTFT)
        print("\n--- Time to First Token (TTFT) ---")
        print("(Prefill phase - user perceived start)")
        self.calculate_percentiles(self.metrics["first_token_latency"], "First Token Latency")

        # Per-token decode latency
        print("\n--- Per-Token Decode Latency ---")
        print("(Generation phase - ongoing speed)")
        self.calculate_percentiles(self.metrics["decode_latencies"], "Decode Token Latency")

        # Total end-to-end latency
        print("\n--- Total End-to-End Latency ---")
        self.calculate_percentiles(self.metrics["total_latency"], "Total Latency")

        # Throughput
        print("\n--- Throughput ---")
        tps = self.metrics["tokens_per_second"]
        print("Tokens per Second:")
        print(f"  P50:  {np.percentile(tps, 50):.2f} tokens/s")
        print(f"  P90:  {np.percentile(tps, 90):.2f} tokens/s")
        print(f"  Mean: {np.mean(tps):.2f} tokens/s")

        # Analysis
        self.analyze_results()

    def analyze_results(self):
        """Provide interpretation of results"""
        print("\n" + "=" * 60)
        print("ANALYSIS")
        print("=" * 60)

        first_token_p99 = np.percentile(self.metrics["first_token_latency"], 99)
        decode_p99 = np.percentile(self.metrics["decode_latencies"], 99)

        print("\nKey Insights:")

        # TTFT analysis
        if first_token_p99 < 100:
            print("✅ First token latency is excellent (<100ms p99)")
            print("   Users will perceive instant response")
        elif first_token_p99 < 500:
            print("⚠️  First token latency is acceptable (100-500ms p99)")
            print("   Noticeable delay but tolerable for most use cases")
        else:
            print("❌ First token latency is high (>500ms p99)")
            print("   Users will notice significant delay - optimize prefill")

        # Decode analysis
        if decode_p99 < 50:
            print("\n✅ Decode latency is excellent (<50ms p99)")
            print("   Smooth, fast text generation")
        elif decode_p99 < 100:
            print("\n⚠️  Decode latency is acceptable (50-100ms p99)")
        else:
            print("\n❌ Decode latency is high (>100ms p99)")
            print("   Users may notice stuttering in generation")

        # Use case recommendations
        print("\n--- Use Case Fit ---")
        print("Chatbot (interactive):  TTFT critical, decode important")
        print("Code generation:        Both TTFT and decode important")
        print("Batch summarization:    Total throughput critical, latency less so")
        print("Real-time translation:  Decode latency critical")


def main():
    benchmark = LLMLatencyBenchmark()

    # Run benchmark
    benchmark.run_benchmark(num_runs=100, prompt_tokens=512, output_tokens=128)

    # Report results
    benchmark.report_metrics()

    print("\n" + "=" * 60)
    print("WHY THESE METRICS MATTER")
    print("=" * 60)
    print("""
First Token Latency (TTFT):
  - User-perceived responsiveness
  - Critical for chat/interactive use cases
  - Dominated by prefill phase (processing prompt)
  - Optimization: reduce prompt size, use continuous batching

Per-Token Decode Latency:
  - Generation smoothness
  - Affects perceived "typing speed"
  - Each token depends on previous tokens (sequential)
  - Optimization: better GPU utilization, quantization

Throughput (tokens/second):
  - System capacity
  - Important for batch processing
  - Can increase with larger batches (but increases latency)
  - Optimization: larger batch sizes, better batching strategies

P99 vs Mean:
  - Mean hides outliers
  - P99 shows worst-case user experience
  - In production, you design for P99, not mean
  - Tail latency = what causes user complaints
    """)


if __name__ == "__main__":
    main()
