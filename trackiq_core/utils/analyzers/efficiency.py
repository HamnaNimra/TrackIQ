"""Efficiency analyzer for performance data."""

from typing import Any

import numpy as np

from trackiq_core.schemas import AnalysisResult
from trackiq_core.utils.analysis_utils import DataLoader
from trackiq_core.utils.base import BaseAnalyzer
from trackiq_core.utils.efficiency import BatchEfficiencyAnalyzer, EfficiencyCalculator


class EfficiencyAnalyzer(BaseAnalyzer):
    """Analyze efficiency metrics from benchmark data.

    Calculates:
    - Performance per Watt (Perf/W)
    - Energy per inference (Joules, kWh)
    - Cost per inference (USD)
    - Power efficiency percentage
    - Batch size efficiency optimization
    """

    def __init__(
        self,
        config=None,
        electricity_rate: float = None,
        gpu_tdp: float = None,
    ):
        """Initialize analyzer.

        Args:
            config: Optional configuration object
            electricity_rate: Electricity rate in USD/kWh
            gpu_tdp: GPU TDP in Watts for efficiency calculations
        """
        super().__init__("EfficiencyAnalyzer")
        self.config = config
        self.calculator = EfficiencyCalculator(
            electricity_rate=electricity_rate,
            gpu_tdp=gpu_tdp,
        )
        self.batch_analyzer = BatchEfficiencyAnalyzer(self.calculator)

    def analyze(
        self,
        csv_filepath: str,
        throughput_col: str = "throughput",
        latency_col: str = "latency_ms",
        power_col: str = "power_w",
        group_by: str = "workload",
    ) -> AnalysisResult:
        """Analyze efficiency from benchmark CSV file.

        Args:
            csv_filepath: Path to benchmark CSV
            throughput_col: Column name for throughput (or None to calculate from latency)
            latency_col: Column name for latency in milliseconds
            power_col: Column name for power in Watts
            group_by: Column to group analysis by

        Returns:
            AnalysisResult with efficiency metrics
        """
        df = DataLoader.load_csv(csv_filepath)

        # Validate required columns
        required_cols = [latency_col, power_col]
        if throughput_col and throughput_col in df.columns:
            required_cols.append(throughput_col)

        DataLoader.validate_columns(df, required_cols)

        metrics = {}
        for group_name, group_df in df.groupby(group_by):
            latencies = group_df[latency_col].tolist()
            power_samples = group_df[power_col].tolist()
            avg_latency = np.mean(latencies)

            # Calculate throughput from latency if not provided
            if throughput_col and throughput_col in group_df.columns:
                throughput = group_df[throughput_col].mean()
            else:
                throughput = 1000.0 / avg_latency if avg_latency > 0 else 0.0

            efficiency_metrics = self.calculator.calculate_efficiency_metrics(
                throughput=throughput,
                latency_ms=avg_latency,
                power_samples=power_samples,
                throughput_unit="ops/sec",
                include_cost=True,
            )

            metrics[str(group_name)] = efficiency_metrics.to_dict()
            metrics[str(group_name)]["num_samples"] = len(latencies)

        result = AnalysisResult(
            name="Efficiency Analysis",
            metrics=metrics,
            raw_data=df,
        )
        self.add_result(result)
        return result

    def analyze_batch_efficiency(
        self,
        csv_filepath: str,
        batch_col: str = "batch_size",
        latency_col: str = "latency_ms",
        power_col: str = "power_w",
        throughput_col: str = None,
    ) -> AnalysisResult:
        """Analyze efficiency across different batch sizes.

        Args:
            csv_filepath: Path to benchmark CSV
            batch_col: Column name for batch size
            latency_col: Column name for latency
            power_col: Column name for power
            throughput_col: Column name for throughput (calculated if None)

        Returns:
            AnalysisResult with batch efficiency analysis
        """
        df = DataLoader.load_csv(csv_filepath)
        DataLoader.validate_columns(df, [batch_col, latency_col, power_col])

        batch_results = []
        for batch_size, batch_df in df.groupby(batch_col):
            latencies = batch_df[latency_col].tolist()
            power_samples = batch_df[power_col].tolist()

            if throughput_col and throughput_col in batch_df.columns:
                throughput = batch_df[throughput_col].mean()
            else:
                avg_latency = np.mean(latencies)
                throughput = 1000.0 / avg_latency if avg_latency > 0 else 0.0

            batch_results.append(
                {
                    "batch_size": int(batch_size),
                    "throughput": throughput,
                    "latency_ms": np.mean(latencies),
                    "power_samples": power_samples,
                }
            )

        analysis = self.batch_analyzer.analyze_batch_efficiency(batch_results)
        pareto_optimal = self.batch_analyzer.find_pareto_optimal(batch_results)

        metrics = {
            "batch_analysis": analysis,
            "pareto_optimal_batches": pareto_optimal,
            "recommendation": self._generate_batch_recommendation(analysis),
        }

        result = AnalysisResult(
            name="Batch Efficiency Analysis",
            metrics=metrics,
            raw_data=df,
        )
        self.add_result(result)
        return result

    def _generate_batch_recommendation(
        self,
        analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate batch size recommendation based on efficiency analysis.

        Args:
            analysis: Batch efficiency analysis results

        Returns:
            Recommendation dictionary
        """
        opt_efficiency = analysis.get("optimal_for_efficiency")
        opt_throughput = analysis.get("optimal_for_throughput")
        opt_energy = analysis.get("optimal_for_energy")

        recommendation = {
            "best_overall": opt_efficiency,
            "reasoning": [],
        }

        if opt_efficiency == opt_throughput == opt_energy:
            recommendation["reasoning"].append(
                f"Batch size {opt_efficiency} is optimal for efficiency, throughput, AND energy."
            )
            recommendation["confidence"] = "high"
        elif opt_efficiency == opt_throughput:
            recommendation["reasoning"].append(f"Batch size {opt_efficiency} balances efficiency and throughput.")
            recommendation["confidence"] = "high"
        elif opt_efficiency == opt_energy:
            recommendation["reasoning"].append(
                f"Batch size {opt_efficiency} optimizes efficiency and energy consumption."
            )
            recommendation["confidence"] = "medium"
        else:
            recommendation["reasoning"].append(
                f"Trade-off exists: Batch {opt_efficiency} for efficiency, "
                f"{opt_throughput} for throughput, {opt_energy} for energy."
            )
            recommendation["confidence"] = "low"
            recommendation["alternatives"] = {
                "for_throughput": opt_throughput,
                "for_energy": opt_energy,
            }

        return recommendation

    def compare_configurations(
        self,
        baseline_metrics: dict[str, float],
        current_metrics: dict[str, float],
    ) -> dict[str, Any]:
        """Compare efficiency between two configurations.

        Args:
            baseline_metrics: Dict with throughput, latency_ms, power_w
            current_metrics: Dict with throughput, latency_ms, power_w

        Returns:
            Comparison results
        """
        baseline_eff = self.calculator.calculate_efficiency_metrics(
            throughput=baseline_metrics["throughput"],
            latency_ms=baseline_metrics["latency_ms"],
            power_samples=[baseline_metrics["power_w"]],
        )

        current_eff = self.calculator.calculate_efficiency_metrics(
            throughput=current_metrics["throughput"],
            latency_ms=current_metrics["latency_ms"],
            power_samples=[current_metrics["power_w"]],
        )

        return self.calculator.compare_efficiency(baseline_eff, current_eff)

    def summarize(self) -> dict[str, Any]:
        """Summarize all efficiency analysis results.

        Returns:
            Summary dictionary
        """
        if not self.results:
            return {}

        summary = {
            "total_analyses": len(self.results),
            "best_perf_per_watt": 0.0,
            "lowest_energy_per_inference": float("inf"),
            "configurations_analyzed": [],
        }

        for result in self.results:
            for config_name, metrics in result.metrics.items():
                if isinstance(metrics, dict):
                    summary["configurations_analyzed"].append(config_name)

                    perf_per_watt = metrics.get("perf_per_watt", 0)
                    if perf_per_watt > summary["best_perf_per_watt"]:
                        summary["best_perf_per_watt"] = perf_per_watt
                        summary["best_config_for_efficiency"] = config_name

                    energy = metrics.get("energy_per_inference_joules", float("inf"))
                    if energy < summary["lowest_energy_per_inference"]:
                        summary["lowest_energy_per_inference"] = energy
                        summary["best_config_for_energy"] = config_name

        return summary


__all__ = ["EfficiencyAnalyzer"]
