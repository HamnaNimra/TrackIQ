"""Core utilities for TrackIQ (DataLoader, LatencyStats, PerformanceComparator)."""

from typing import Any

import numpy as np
import pandas as pd


class DataLoader:
    """Load and validate data for analysis."""

    @staticmethod
    def load_csv(filepath: str) -> pd.DataFrame:
        """Load CSV file safely."""
        try:
            return pd.read_csv(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
        except pd.errors.ParserError:
            raise ValueError(f"Invalid CSV format: {filepath}")

    @staticmethod
    def validate_columns(df: pd.DataFrame, required_cols: list[str]) -> bool:
        """Validate that required columns exist."""
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True


class LatencyStats:
    """Calculate latency statistics."""

    @staticmethod
    def calculate_percentiles(latencies: list[float], percentiles: list[int] = [50, 95, 99]) -> dict:
        if not latencies:
            return {}
        result = {}
        for p in percentiles:
            result[f"p{p}"] = np.percentile(latencies, p)
        result["mean"] = np.mean(latencies)
        result["std"] = np.std(latencies)
        result["min"] = np.min(latencies)
        result["max"] = np.max(latencies)
        return result

    @staticmethod
    def calculate_extended_stats(
        latencies: list[float],
        percentiles: list[int] = [50, 95, 99],
        power_samples: list[float] | None = None,
    ) -> dict[str, Any]:
        if not latencies:
            return {}
        result = LatencyStats.calculate_percentiles(latencies, percentiles)
        result["jitter"] = LatencyStats.calculate_jitter(latencies)
        result["cv"] = LatencyStats.calculate_coefficient_of_variation(latencies)
        result["iqr"] = LatencyStats.calculate_iqr(latencies)
        result["outlier_count"] = LatencyStats.count_outliers(latencies)
        if power_samples and len(power_samples) > 0:
            avg_power = np.mean(power_samples)
            avg_latency = result["mean"]
            throughput = 1000.0 / avg_latency if avg_latency > 0 else 0.0
            result["throughput_per_sec"] = throughput
            result["perf_per_watt"] = throughput / avg_power if avg_power > 0 else 0.0
            result["energy_per_inference_j"] = avg_power * (avg_latency / 1000.0)
            result["power_mean_w"] = avg_power
            result["power_max_w"] = np.max(power_samples)
            result["power_min_w"] = np.min(power_samples)
        return result

    @staticmethod
    def calculate_jitter(latencies: list[float]) -> float:
        if len(latencies) < 2:
            return 0.0
        differences = np.abs(np.diff(latencies))
        return float(np.mean(differences))

    @staticmethod
    def calculate_coefficient_of_variation(latencies: list[float]) -> float:
        if not latencies:
            return 0.0
        mean = np.mean(latencies)
        if mean == 0:
            return 0.0
        return float(np.std(latencies) / mean)

    @staticmethod
    def calculate_iqr(latencies: list[float]) -> float:
        if not latencies:
            return 0.0
        q1 = np.percentile(latencies, 25)
        q3 = np.percentile(latencies, 75)
        return float(q3 - q1)

    @staticmethod
    def count_outliers(latencies: list[float], sigma: float = 3.0) -> int:
        if len(latencies) < 2:
            return 0
        mean = np.mean(latencies)
        std = np.std(latencies)
        if std == 0:
            return 0
        lower_bound = mean - (sigma * std)
        upper_bound = mean + (sigma * std)
        outliers = [x for x in latencies if x < lower_bound or x > upper_bound]
        return len(outliers)

    @staticmethod
    def calculate_for_groups(df: pd.DataFrame, latency_col: str, group_by: str) -> dict:
        result = {}
        for group_name, group_df in df.groupby(group_by):
            latencies = group_df[latency_col].tolist()
            result[group_name] = LatencyStats.calculate_percentiles(latencies)
        return result

    @staticmethod
    def calculate_for_groups_extended(
        df: pd.DataFrame,
        latency_col: str,
        group_by: str,
        power_col: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        result = {}
        for group_name, group_df in df.groupby(group_by):
            latencies = group_df[latency_col].tolist()
            power_samples = None
            if power_col and power_col in group_df.columns:
                power_samples = group_df[power_col].tolist()
            result[group_name] = LatencyStats.calculate_extended_stats(latencies, power_samples=power_samples)
        return result


class PerformanceComparator:
    """Compare performance across different configurations."""

    @staticmethod
    def compare_latency_throughput(
        batch_sizes: list[int], latencies: list[float], images_per_second: list[float]
    ) -> dict:
        return {
            "batch_size": batch_sizes,
            "latency_ms": latencies,
            "throughput_img_sec": images_per_second,
            "optimal_for_latency": batch_sizes[np.argmin(latencies)],
            "optimal_for_throughput": batch_sizes[np.argmax(images_per_second)],
        }
