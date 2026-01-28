"""Core utilities for AutoPerfPy."""

import pandas as pd
from typing import List
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
