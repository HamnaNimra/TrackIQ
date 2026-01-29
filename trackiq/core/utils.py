"""Core utilities for TrackIQ."""

import pandas as pd
from typing import List, Dict, Any, Optional
import numpy as np


class DataLoader:
    """Load and validate data for analysis."""

    @staticmethod
    def load_csv(filepath: str) -> pd.DataFrame:
        """Load CSV file safely.

        Args:
            filepath: Path to CSV file

        Returns:
            DataFrame with data
        """
        try:
            return pd.read_csv(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
        except pd.errors.ParserError:
            raise ValueError(f"Invalid CSV format: {filepath}")

    @staticmethod
    def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> bool:
        """Validate that required columns exist.

        Args:
            df: DataFrame to validate
            required_cols: Required column names

        Returns:
            True if valid, raises error otherwise
        """
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True


class LatencyStats:
    """Calculate latency statistics."""

    @staticmethod
    def calculate_percentiles(latencies: List[float], percentiles: List[int] = [50, 95, 99]) -> dict:
        """Calculate percentile latencies.

        Args:
            latencies: List of latency values
            percentiles: Percentiles to calculate

        Returns:
            Dictionary with percentile values
        """
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
        latencies: List[float],
        percentiles: List[int] = [50, 95, 99],
        power_samples: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Calculate extended latency statistics including efficiency metrics.

        Args:
            latencies: List of latency values in milliseconds
            percentiles: Percentiles to calculate
            power_samples: Optional list of power readings in Watts

        Returns:
            Dictionary with extended statistics including:
            - Standard percentile stats
            - Variability metrics (jitter, CV, IQR)
            - Efficiency metrics (if power_samples provided)
        """
        if not latencies:
            return {}

        # Basic percentile stats
        result = LatencyStats.calculate_percentiles(latencies, percentiles)

        # Variability metrics
        result["jitter"] = LatencyStats.calculate_jitter(latencies)
        result["cv"] = LatencyStats.calculate_coefficient_of_variation(latencies)
        result["iqr"] = LatencyStats.calculate_iqr(latencies)
        result["outlier_count"] = LatencyStats.count_outliers(latencies)

        # Efficiency metrics (if power data available)
        if power_samples and len(power_samples) > 0:
            avg_power = np.mean(power_samples)
            avg_latency = result["mean"]

            # Throughput (inferences per second)
            throughput = 1000.0 / avg_latency if avg_latency > 0 else 0.0

            # Perf/Watt
            result["throughput_per_sec"] = throughput
            result["perf_per_watt"] = throughput / avg_power if avg_power > 0 else 0.0

            # Energy per inference (Joules = Watts × seconds)
            result["energy_per_inference_j"] = avg_power * (avg_latency / 1000.0)

            # Power stats
            result["power_mean_w"] = avg_power
            result["power_max_w"] = np.max(power_samples)
            result["power_min_w"] = np.min(power_samples)

        return result

    @staticmethod
    def calculate_jitter(latencies: List[float]) -> float:
        """Calculate jitter (variation between consecutive measurements).

        Jitter is the average absolute difference between consecutive latencies.

        Args:
            latencies: List of latency values

        Returns:
            Jitter value (same unit as latencies)
        """
        if len(latencies) < 2:
            return 0.0

        differences = np.abs(np.diff(latencies))
        return float(np.mean(differences))

    @staticmethod
    def calculate_coefficient_of_variation(latencies: List[float]) -> float:
        """Calculate coefficient of variation (CV = std/mean).

        CV is a standardized measure of dispersion. Lower is more consistent.
        - CV < 0.1: Very consistent
        - CV 0.1-0.3: Moderately consistent
        - CV > 0.3: High variability

        Args:
            latencies: List of latency values

        Returns:
            Coefficient of variation (dimensionless ratio)
        """
        if not latencies:
            return 0.0

        mean = np.mean(latencies)
        if mean == 0:
            return 0.0

        return float(np.std(latencies) / mean)

    @staticmethod
    def calculate_iqr(latencies: List[float]) -> float:
        """Calculate interquartile range (IQR = Q3 - Q1).

        IQR is a robust measure of spread, less sensitive to outliers than std.

        Args:
            latencies: List of latency values

        Returns:
            Interquartile range
        """
        if not latencies:
            return 0.0

        q1 = np.percentile(latencies, 25)
        q3 = np.percentile(latencies, 75)
        return float(q3 - q1)

    @staticmethod
    def count_outliers(latencies: List[float], sigma: float = 3.0) -> int:
        """Count outliers beyond N standard deviations.

        Args:
            latencies: List of latency values
            sigma: Number of standard deviations for outlier threshold

        Returns:
            Number of outliers
        """
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
        """Calculate statistics per group.

        Args:
            df: DataFrame with data
            latency_col: Column name for latencies
            group_by: Column to group by

        Returns:
            Dictionary with stats per group
        """
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
        power_col: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate extended statistics per group.

        Args:
            df: DataFrame with data
            latency_col: Column name for latencies
            group_by: Column to group by
            power_col: Optional column name for power readings

        Returns:
            Dictionary with extended stats per group
        """
        result = {}
        for group_name, group_df in df.groupby(group_by):
            latencies = group_df[latency_col].tolist()
            power_samples = None
            if power_col and power_col in group_df.columns:
                power_samples = group_df[power_col].tolist()
            result[group_name] = LatencyStats.calculate_extended_stats(
                latencies, power_samples=power_samples
            )
        return result


class PerformanceComparator:
    """Compare performance across different configurations."""

    @staticmethod
    def compare_latency_throughput(
        batch_sizes: List[int], latencies: List[float], images_per_second: List[float]
    ) -> dict:
        """Compare latency vs throughput trade-offs.

        Args:
            batch_sizes: List of batch sizes tested
            latencies: Corresponding latencies
            images_per_second: Corresponding throughput

        Returns:
            Dictionary with comparison results
        """
        comparison = {
            "batch_size": batch_sizes,
            "latency_ms": latencies,
            "throughput_img_sec": images_per_second,
            "optimal_for_latency": batch_sizes[np.argmin(latencies)],
            "optimal_for_throughput": batch_sizes[np.argmax(images_per_second)],
        }
        return comparison
