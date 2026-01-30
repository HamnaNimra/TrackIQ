"""Efficiency metrics for performance analysis.

This module provides metrics for analyzing performance efficiency including:
- Performance per Watt (Perf/W)
- Energy per inference (Joules)
- Cost estimation for cloud deployments
- Efficiency comparisons across configurations
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass
class EfficiencyMetrics:
    """Container for efficiency metric results."""

    # Core efficiency metrics
    perf_per_watt: float  # Throughput / Power (ops/W or images/W)
    energy_per_inference_joules: float  # Energy consumed per inference
    energy_per_inference_kwh: float  # Energy in kWh (for cost calculations)

    # Throughput metrics used
    throughput: float  # ops/sec or images/sec
    throughput_unit: str  # "images/sec", "tokens/sec", "requests/sec"

    # Power metrics used
    avg_power_watts: float
    peak_power_watts: float
    power_efficiency_percent: float  # Actual vs TDP utilization

    # Optional cost metrics
    cost_per_inference_usd: Optional[float] = None
    cost_per_1k_inferences_usd: Optional[float] = None
    electricity_rate_per_kwh: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "perf_per_watt": self.perf_per_watt,
            "energy_per_inference_joules": self.energy_per_inference_joules,
            "energy_per_inference_kwh": self.energy_per_inference_kwh,
            "throughput": self.throughput,
            "throughput_unit": self.throughput_unit,
            "avg_power_watts": self.avg_power_watts,
            "peak_power_watts": self.peak_power_watts,
            "power_efficiency_percent": self.power_efficiency_percent,
            "cost_per_inference_usd": self.cost_per_inference_usd,
            "cost_per_1k_inferences_usd": self.cost_per_1k_inferences_usd,
            "electricity_rate_per_kwh": self.electricity_rate_per_kwh,
        }


class EfficiencyCalculator:
    """Calculate efficiency metrics from performance and power data."""

    # Default electricity rates (USD per kWh) for different regions
    ELECTRICITY_RATES = {
        "us_average": 0.12,
        "us_california": 0.22,
        "us_texas": 0.11,
        "eu_average": 0.25,
        "eu_germany": 0.35,
        "asia_average": 0.10,
        "cloud_estimate": 0.15,  # Typical cloud provider rate
    }

    # Common GPU TDP values (Watts) for power efficiency calculations
    GPU_TDP = {
        "nvidia_a100_40gb": 400,
        "nvidia_a100_80gb": 400,
        "nvidia_h100_sxm": 700,
        "nvidia_h100_pcie": 350,
        "nvidia_l40s": 350,
        "nvidia_rtx_4090": 450,
        "nvidia_rtx_3090": 350,
        "nvidia_t4": 70,
        "nvidia_v100": 300,
        "default": 300,
    }

    def __init__(
        self,
        electricity_rate: float = None,
        gpu_tdp: float = None,
        region: str = "cloud_estimate",
        gpu_model: str = "default",
    ):
        """Initialize efficiency calculator.

        Args:
            electricity_rate: Custom electricity rate in USD/kWh
            gpu_tdp: Custom GPU TDP in Watts
            region: Region for default electricity rate
            gpu_model: GPU model for default TDP
        """
        self.electricity_rate = electricity_rate or self.ELECTRICITY_RATES.get(
            region, self.ELECTRICITY_RATES["cloud_estimate"]
        )
        self.gpu_tdp = gpu_tdp or self.GPU_TDP.get(
            gpu_model, self.GPU_TDP["default"]
        )

    def calculate_perf_per_watt(
        self,
        throughput: float,
        power_watts: float,
    ) -> float:
        """Calculate performance per watt.

        Args:
            throughput: Operations per second (images/sec, tokens/sec, etc.)
            power_watts: Average power consumption in Watts

        Returns:
            Performance per Watt (ops/W)
        """
        if power_watts <= 0:
            return 0.0
        return throughput / power_watts

    def calculate_energy_per_inference(
        self,
        latency_ms: float,
        power_watts: float,
    ) -> Dict[str, float]:
        """Calculate energy consumed per inference.

        Args:
            latency_ms: Inference latency in milliseconds
            power_watts: Average power during inference in Watts

        Returns:
            Dictionary with energy in Joules and kWh
        """
        # Energy (J) = Power (W) × Time (s)
        latency_seconds = latency_ms / 1000.0
        energy_joules = power_watts * latency_seconds

        # Convert to kWh: 1 kWh = 3,600,000 J
        energy_kwh = energy_joules / 3_600_000

        return {
            "joules": energy_joules,
            "kwh": energy_kwh,
        }

    def calculate_cost_per_inference(
        self,
        energy_kwh: float,
        electricity_rate: float = None,
    ) -> Dict[str, float]:
        """Calculate cost per inference based on energy consumption.

        Args:
            energy_kwh: Energy per inference in kWh
            electricity_rate: Electricity rate in USD/kWh (uses default if None)

        Returns:
            Dictionary with cost per inference and per 1000 inferences
        """
        rate = electricity_rate or self.electricity_rate
        cost_per_inference = energy_kwh * rate
        cost_per_1k = cost_per_inference * 1000

        return {
            "per_inference_usd": cost_per_inference,
            "per_1k_inferences_usd": cost_per_1k,
            "electricity_rate": rate,
        }

    def calculate_power_efficiency(
        self,
        actual_power: float,
        tdp: float = None,
    ) -> float:
        """Calculate power efficiency as percentage of TDP.

        Args:
            actual_power: Actual power consumption in Watts
            tdp: Thermal Design Power (uses default if None)

        Returns:
            Power efficiency percentage (0-100+)
        """
        gpu_tdp = tdp or self.gpu_tdp
        if gpu_tdp <= 0:
            return 0.0
        return (actual_power / gpu_tdp) * 100

    def calculate_efficiency_metrics(
        self,
        throughput: float,
        latency_ms: float,
        power_samples: List[float],
        throughput_unit: str = "images/sec",
        include_cost: bool = True,
    ) -> EfficiencyMetrics:
        """Calculate comprehensive efficiency metrics.

        Args:
            throughput: Operations per second
            latency_ms: Average latency in milliseconds
            power_samples: List of power readings in Watts
            throughput_unit: Unit for throughput metric
            include_cost: Whether to include cost calculations

        Returns:
            EfficiencyMetrics dataclass with all metrics
        """
        if not power_samples:
            raise ValueError("Power samples required for efficiency calculation")

        avg_power = np.mean(power_samples)
        peak_power = np.max(power_samples)

        # Core efficiency calculations
        perf_per_watt = self.calculate_perf_per_watt(throughput, avg_power)
        energy = self.calculate_energy_per_inference(latency_ms, avg_power)
        power_efficiency = self.calculate_power_efficiency(avg_power)

        # Cost calculations (optional)
        cost_per_inference = None
        cost_per_1k = None
        electricity_rate = None

        if include_cost:
            cost = self.calculate_cost_per_inference(energy["kwh"])
            cost_per_inference = cost["per_inference_usd"]
            cost_per_1k = cost["per_1k_inferences_usd"]
            electricity_rate = cost["electricity_rate"]

        return EfficiencyMetrics(
            perf_per_watt=perf_per_watt,
            energy_per_inference_joules=energy["joules"],
            energy_per_inference_kwh=energy["kwh"],
            throughput=throughput,
            throughput_unit=throughput_unit,
            avg_power_watts=avg_power,
            peak_power_watts=peak_power,
            power_efficiency_percent=power_efficiency,
            cost_per_inference_usd=cost_per_inference,
            cost_per_1k_inferences_usd=cost_per_1k,
            electricity_rate_per_kwh=electricity_rate,
        )

    def compare_efficiency(
        self,
        baseline: EfficiencyMetrics,
        current: EfficiencyMetrics,
    ) -> Dict[str, Any]:
        """Compare efficiency between two configurations.

        Args:
            baseline: Baseline efficiency metrics
            current: Current efficiency metrics

        Returns:
            Dictionary with comparison results
        """
        def safe_percent_change(base: float, curr: float) -> float:
            if base == 0:
                return float('inf') if curr > 0 else 0.0
            return ((curr - base) / base) * 100

        return {
            "perf_per_watt_change_percent": safe_percent_change(
                baseline.perf_per_watt, current.perf_per_watt
            ),
            "energy_per_inference_change_percent": safe_percent_change(
                baseline.energy_per_inference_joules,
                current.energy_per_inference_joules,
            ),
            "power_change_percent": safe_percent_change(
                baseline.avg_power_watts, current.avg_power_watts
            ),
            "throughput_change_percent": safe_percent_change(
                baseline.throughput, current.throughput
            ),
            "is_more_efficient": current.perf_per_watt > baseline.perf_per_watt,
            "efficiency_improvement_percent": safe_percent_change(
                baseline.perf_per_watt, current.perf_per_watt
            ),
            "baseline": baseline.to_dict(),
            "current": current.to_dict(),
        }


class BatchEfficiencyAnalyzer:
    """Analyze efficiency across different batch sizes."""

    def __init__(self, calculator: EfficiencyCalculator = None):
        """Initialize analyzer.

        Args:
            calculator: EfficiencyCalculator instance (creates default if None)
        """
        self.calculator = calculator or EfficiencyCalculator()

    def analyze_batch_efficiency(
        self,
        batch_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyze efficiency across batch sizes.

        Args:
            batch_results: List of dicts with keys:
                - batch_size: int
                - throughput: float (ops/sec)
                - latency_ms: float
                - power_samples: List[float] (Watts)

        Returns:
            Dictionary with efficiency analysis per batch size
        """
        results = {
            "batch_efficiencies": [],
            "optimal_for_efficiency": None,
            "optimal_for_throughput": None,
            "optimal_for_energy": None,
        }

        best_efficiency = 0
        best_throughput = 0
        lowest_energy = float('inf')

        for batch_data in batch_results:
            batch_size = batch_data["batch_size"]
            throughput = batch_data["throughput"]
            latency_ms = batch_data["latency_ms"]
            power_samples = batch_data["power_samples"]

            metrics = self.calculator.calculate_efficiency_metrics(
                throughput=throughput,
                latency_ms=latency_ms,
                power_samples=power_samples,
                throughput_unit="images/sec",
            )

            batch_result = {
                "batch_size": batch_size,
                "metrics": metrics.to_dict(),
            }
            results["batch_efficiencies"].append(batch_result)

            # Track optimal configurations
            if metrics.perf_per_watt > best_efficiency:
                best_efficiency = metrics.perf_per_watt
                results["optimal_for_efficiency"] = batch_size

            if metrics.throughput > best_throughput:
                best_throughput = metrics.throughput
                results["optimal_for_throughput"] = batch_size

            if metrics.energy_per_inference_joules < lowest_energy:
                lowest_energy = metrics.energy_per_inference_joules
                results["optimal_for_energy"] = batch_size

        return results

    def find_pareto_optimal(
        self,
        batch_results: List[Dict[str, Any]],
    ) -> List[int]:
        """Find Pareto-optimal batch sizes (throughput vs energy trade-off).

        Args:
            batch_results: List of batch result dicts

        Returns:
            List of Pareto-optimal batch sizes
        """
        points = []
        for batch_data in batch_results:
            batch_size = batch_data["batch_size"]
            throughput = batch_data["throughput"]
            latency_ms = batch_data["latency_ms"]
            power_samples = batch_data["power_samples"]

            avg_power = np.mean(power_samples)
            energy_j = avg_power * (latency_ms / 1000.0)

            points.append({
                "batch_size": batch_size,
                "throughput": throughput,
                "energy": energy_j,
            })

        # Find Pareto frontier (maximize throughput, minimize energy)
        pareto_optimal = []
        for p in points:
            is_dominated = False
            for q in points:
                # q dominates p if q has higher throughput AND lower energy
                if q["throughput"] > p["throughput"] and q["energy"] < p["energy"]:
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_optimal.append(p["batch_size"])

        return sorted(pareto_optimal)


__all__ = [
    "EfficiencyMetrics",
    "EfficiencyCalculator",
    "BatchEfficiencyAnalyzer",
]
