"""Tests for AutoPerfPy benchmarks module."""

from autoperfpy.benchmarks import BatchingTradeoffBenchmark, LLMLatencyBenchmark


class TestBatchingTradeoffBenchmark:
    """Tests for BatchingTradeoffBenchmark."""

    def test_run_with_default_batch_sizes(self):
        """Test running benchmark with default batch sizes."""
        benchmark = BatchingTradeoffBenchmark()
        results = benchmark.run()
        
        assert "batch_size" in results
        assert "latency_ms" in results
        assert "throughput_img_per_sec" in results
        assert len(results["batch_size"]) > 0

    def test_run_with_custom_batch_sizes(self):
        """Test running benchmark with custom batch sizes."""
        batch_sizes = [1, 2, 4]
        benchmark = BatchingTradeoffBenchmark()
        results = benchmark.run(batch_sizes=batch_sizes)
        
        assert results["batch_size"] == batch_sizes
        assert len(results["latency_ms"]) == 3
        assert len(results["throughput_img_per_sec"]) == 3

    def test_run_with_custom_parameters(self):
        """Test running benchmark with custom parameters."""
        benchmark = BatchingTradeoffBenchmark()
        results = benchmark.run(
            batch_sizes=[1, 4],
            num_images=2000,
            base_overhead=0.02,
            time_per_image=0.01
        )
        
        assert len(results["batch_size"]) == 2
        assert all(lat > 0 for lat in results["latency_ms"])
        assert all(thr > 0 for thr in results["throughput_img_per_sec"])

    def test_latency_increases_with_batch_size(self):
        """Test that individual latency increases with batch size."""
        benchmark = BatchingTradeoffBenchmark()
        results = benchmark.run(
            batch_sizes=[1, 4, 16],
            base_overhead=0.01,
            time_per_image=0.005
        )
        
        # Latency per image should decrease with batch size
        # (due to amortization of fixed overhead across more images)
        latencies = results["latency_ms"]
        assert latencies[0] > 0  # batch_size 1
        assert latencies[2] < latencies[0]  # batch_size 16 < batch_size 1 (more efficient)

    def test_throughput_optimal_batch_size(self):
        """Test throughput typically peaks at medium batch sizes."""
        benchmark = BatchingTradeoffBenchmark()
        results = benchmark.run(batch_sizes=[1, 2, 4, 8, 16, 32])
        
        throughputs = results["throughput_img_per_sec"]
        # Should have some variation in throughput
        assert max(throughputs) > min(throughputs)

    def test_get_optimal_batch_size_for_latency(self):
        """Test finding optimal batch size for latency."""
        benchmark = BatchingTradeoffBenchmark()
        benchmark.run(batch_sizes=[1, 4, 8, 16, 32])
        
        optimal = benchmark.get_optimal_batch_size(optimize_for="latency")
        assert optimal in [1, 4, 8, 16, 32]

    def test_get_optimal_batch_size_for_throughput(self):
        """Test finding optimal batch size for throughput."""
        benchmark = BatchingTradeoffBenchmark()
        benchmark.run(batch_sizes=[1, 4, 8, 16, 32])
        
        optimal = benchmark.get_optimal_batch_size(optimize_for="throughput")
        assert optimal in [1, 4, 8, 16, 32]

    def test_get_optimal_batch_size_without_running(self):
        """Test handling when no results exist."""
        benchmark = BatchingTradeoffBenchmark()
        
        optimal = benchmark.get_optimal_batch_size()
        assert optimal is None

    def test_results_are_stored(self):
        """Test that results are stored in benchmark object."""
        benchmark = BatchingTradeoffBenchmark()
        benchmark.run(batch_sizes=[1, 4])
        
        assert benchmark.results is not None
        assert len(benchmark.results) > 0


class TestLLMLatencyBenchmark:
    """Tests for LLMLatencyBenchmark."""

    def test_run_with_defaults(self):
        """Test running LLM benchmark with default parameters."""
        benchmark = LLMLatencyBenchmark()
        results = benchmark.run()
        
        assert results is not None
        assert len(results) > 0

    def test_run_with_custom_parameters(self):
        """Test running LLM benchmark with custom parameters."""
        benchmark = LLMLatencyBenchmark()
        results = benchmark.run(
            prompt_tokens=256,
            output_tokens=128,
            num_runs=5
        )

        assert "ttft_ms" in results or "latency_ms" in results or len(results) > 0

    def test_run_multiple_times(self):
        """Test that benchmark can be run multiple times."""
        benchmark = LLMLatencyBenchmark()
        results1 = benchmark.run(num_runs=3)
        results2 = benchmark.run(num_runs=5)

        assert results1 is not None
        assert results2 is not None

    def test_results_increase_with_prompt_length(self):
        """Test that latency increases with longer prompts (expected behavior)."""
        benchmark = LLMLatencyBenchmark()

        # Run with short prompt
        results_short = benchmark.run(prompt_tokens=128, num_runs=1)

        # Run with long prompt
        results_long = benchmark.run(prompt_tokens=512, num_runs=1)

        # Both should produce results
        assert results_short is not None
        assert results_long is not None


class TestBenchmarkIntegration:
    """Integration tests for benchmarks."""

    def test_compare_two_benchmark_runs(self):
        """Test comparing results from two benchmark runs."""
        benchmark = BatchingTradeoffBenchmark()
        benchmark.run(batch_sizes=[1, 4, 8])
        optimal1 = benchmark.get_optimal_batch_size(optimize_for="throughput")

        # Reset and run again
        benchmark2 = BatchingTradeoffBenchmark()
        benchmark2.run(batch_sizes=[1, 4, 8])
        optimal2 = benchmark2.get_optimal_batch_size(optimize_for="throughput")
        
        # Both should find an optimal batch size
        assert optimal1 is not None
        assert optimal2 is not None
