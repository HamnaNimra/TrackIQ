"""Variability and consistency metrics analyzers."""

from typing import Dict, Any, Optional, List
from trackiq_core.utils.base import BaseAnalyzer
from trackiq_core.schemas import AnalysisResult
from trackiq_core.utils.analysis_utils import DataLoader, LatencyStats


class VariabilityAnalyzer(BaseAnalyzer):
    """Analyze variability and consistency metrics from performance data.

    This analyzer calculates metrics that measure how consistent or variable
    performance measurements are:

    - Jitter: Average absolute difference between consecutive latencies.
      Lower values indicate more stable, predictable performance.

    - Coefficient of Variation (CV): Standard deviation divided by mean.
      A normalized measure of dispersion:
        - CV < 0.1: Very consistent
        - CV 0.1-0.3: Moderately consistent
        - CV > 0.3: High variability

    - Interquartile Range (IQR): Q3 - Q1, the spread of the middle 50%.
      More robust to outliers than standard deviation.

    - Outlier Count: Number of measurements exceeding N standard deviations
      from the mean (default: 3 sigma).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize analyzer.

        Args:
            config: Optional configuration with:
                - sigma: Standard deviations for outlier detection (default: 3.0)
                - percentiles: Percentiles to calculate (default: [50, 95, 99])
        """
        super().__init__("VariabilityAnalyzer")
        self.config = config or {}
        self.sigma = self.config.get("sigma", 3.0)
        self.percentiles = self.config.get("percentiles", [50, 95, 99])

    def analyze(
        self,
        csv_filepath: str,
        latency_col: str = "latency_ms",
        group_by: Optional[str] = None,
    ) -> AnalysisResult:
        """Analyze variability metrics from a CSV file.

        Args:
            csv_filepath: Path to CSV file with latency data
            latency_col: Column name containing latency values
            group_by: Optional column to group by (e.g., 'workload', 'batch_size')

        Returns:
            AnalysisResult with variability metrics per group
        """
        df = DataLoader.load_csv(csv_filepath)
        DataLoader.validate_columns(df, [latency_col])

        metrics = {}

        if group_by and group_by in df.columns:
            for group_name, group_df in df.groupby(group_by):
                latencies = group_df[latency_col].tolist()
                metrics[str(group_name)] = self._calculate_variability_metrics(latencies)
        else:
            latencies = df[latency_col].tolist()
            metrics["all"] = self._calculate_variability_metrics(latencies)

        result = AnalysisResult(
            name="Variability Analysis",
            metrics=metrics,
            raw_data=df,
        )
        self.add_result(result)
        return result

    def analyze_from_list(
        self,
        latencies: List[float],
        name: str = "analysis",
    ) -> AnalysisResult:
        """Analyze variability metrics from a list of latencies.

        Args:
            latencies: List of latency values
            name: Name identifier for this analysis

        Returns:
            AnalysisResult with variability metrics
        """
        metrics = {name: self._calculate_variability_metrics(latencies)}

        result = AnalysisResult(
            name="Variability Analysis",
            metrics=metrics,
            raw_data=latencies,
        )
        self.add_result(result)
        return result

    def _calculate_variability_metrics(self, latencies: List[float]) -> Dict[str, Any]:
        """Calculate all variability metrics for a list of latencies.

        Args:
            latencies: List of latency values

        Returns:
            Dictionary with variability metrics
        """
        if not latencies:
            return {
                "jitter": 0.0,
                "cv": 0.0,
                "iqr": 0.0,
                "outlier_count": 0,
                "num_samples": 0,
                "consistency_rating": "insufficient_data",
            }

        # Basic stats
        basic_stats = LatencyStats.calculate_percentiles(latencies, self.percentiles)

        # Variability metrics
        jitter = LatencyStats.calculate_jitter(latencies)
        cv = LatencyStats.calculate_coefficient_of_variation(latencies)
        iqr = LatencyStats.calculate_iqr(latencies)
        outlier_count = LatencyStats.count_outliers(latencies, sigma=self.sigma)

        # Determine consistency rating based on CV
        if cv < 0.1:
            consistency_rating = "very_consistent"
        elif cv < 0.2:
            consistency_rating = "consistent"
        elif cv < 0.3:
            consistency_rating = "moderately_consistent"
        else:
            consistency_rating = "high_variability"

        # Calculate outlier percentage
        outlier_percentage = (outlier_count / len(latencies) * 100) if latencies else 0.0

        return {
            **basic_stats,
            "jitter": jitter,
            "cv": cv,
            "iqr": iqr,
            "outlier_count": outlier_count,
            "outlier_percentage": outlier_percentage,
            "sigma_threshold": self.sigma,
            "num_samples": len(latencies),
            "consistency_rating": consistency_rating,
        }

    def summarize(self) -> Dict[str, Any]:
        """Summarize all variability analyses.

        Returns:
            Summary with aggregate statistics and recommendations
        """
        if not self.results:
            return {"total_analyses": 0}

        summary = {
            "total_analyses": len(self.results),
            "most_consistent": None,
            "most_variable": None,
            "highest_jitter": None,
            "most_outliers": None,
            "groups_analyzed": [],
        }

        best_cv = float("inf")
        worst_cv = 0.0
        highest_jitter = 0.0
        most_outlier_count = 0

        for result in self.results:
            for group_name, metrics in result.metrics.items():
                summary["groups_analyzed"].append(group_name)

                cv = metrics.get("cv", 0.0)
                jitter = metrics.get("jitter", 0.0)
                outlier_count = metrics.get("outlier_count", 0)

                if cv < best_cv and cv > 0:
                    best_cv = cv
                    summary["most_consistent"] = {
                        "group": group_name,
                        "cv": cv,
                        "consistency_rating": metrics.get("consistency_rating"),
                    }

                if cv > worst_cv:
                    worst_cv = cv
                    summary["most_variable"] = {
                        "group": group_name,
                        "cv": cv,
                        "consistency_rating": metrics.get("consistency_rating"),
                    }

                if jitter > highest_jitter:
                    highest_jitter = jitter
                    summary["highest_jitter"] = {
                        "group": group_name,
                        "jitter": jitter,
                    }

                if outlier_count > most_outlier_count:
                    most_outlier_count = outlier_count
                    summary["most_outliers"] = {
                        "group": group_name,
                        "outlier_count": outlier_count,
                        "outlier_percentage": metrics.get("outlier_percentage", 0.0),
                    }

        return summary

    def compare_variability(
        self,
        baseline_metrics: Dict[str, Any],
        current_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare variability metrics between two configurations.

        Args:
            baseline_metrics: Metrics from baseline configuration
            current_metrics: Metrics from current configuration

        Returns:
            Comparison results with improvement indicators
        """
        baseline_cv = baseline_metrics.get("cv", 0.0)
        current_cv = current_metrics.get("cv", 0.0)

        baseline_jitter = baseline_metrics.get("jitter", 0.0)
        current_jitter = current_metrics.get("jitter", 0.0)

        baseline_outliers = baseline_metrics.get("outlier_count", 0)
        current_outliers = current_metrics.get("outlier_count", 0)

        # Calculate percentage changes (negative = improvement for variability metrics)
        cv_change = ((current_cv - baseline_cv) / baseline_cv * 100) if baseline_cv > 0 else 0.0
        jitter_change = (
            (current_jitter - baseline_jitter) / baseline_jitter * 100
        ) if baseline_jitter > 0 else 0.0

        return {
            "cv_change_percent": cv_change,
            "jitter_change_percent": jitter_change,
            "outlier_count_change": current_outliers - baseline_outliers,
            "is_more_consistent": current_cv < baseline_cv,
            "is_less_jittery": current_jitter < baseline_jitter,
            "has_fewer_outliers": current_outliers < baseline_outliers,
            "overall_improvement": (
                current_cv < baseline_cv
                and current_jitter < baseline_jitter
            ),
            "baseline": {
                "cv": baseline_cv,
                "jitter": baseline_jitter,
                "outlier_count": baseline_outliers,
            },
            "current": {
                "cv": current_cv,
                "jitter": current_jitter,
                "outlier_count": current_outliers,
            },
        }


__all__ = ["VariabilityAnalyzer"]
