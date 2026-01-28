"""Core abstractions and base classes for AutoPerfPy."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AnalysisResult:
    """Base class for analysis results."""

    name: str
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)
    raw_data: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
        }


class BaseAnalyzer(ABC):
    """Base class for all analyzers."""

    def __init__(self, name: str):
        """Initialize analyzer.

        Args:
            name: Name of the analyzer
        """
        self.name = name
        self.results: List[AnalysisResult] = []

    @abstractmethod
    def analyze(self, data: Any) -> AnalysisResult:
        """Perform analysis on data.

        Args:
            data: Input data to analyze

        Returns:
            AnalysisResult with metrics
        """
        pass

    def add_result(self, result: AnalysisResult) -> None:
        """Store analysis result.

        Args:
            result: AnalysisResult to store
        """
        self.results.append(result)

    def get_results(self) -> List[AnalysisResult]:
        """Get all stored results.

        Returns:
            List of AnalysisResult objects
        """
        return self.results


class BaseBenchmark(ABC):
    """Base class for all benchmarks."""

    def __init__(self, name: str):
        """Initialize benchmark.

        Args:
            name: Name of the benchmark
        """
        self.name = name
        self.results: Dict[str, Any] = {}

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """Run the benchmark.

        Args:
            **kwargs: Benchmark-specific parameters

        Returns:
            Dictionary with benchmark results
        """
        pass

    def get_results(self) -> Dict[str, Any]:
        """Get benchmark results.

        Returns:
            Dictionary with results
        """
        return self.results


class BaseMonitor(ABC):
    """Base class for all monitors."""

    def __init__(self, name: str):
        """Initialize monitor.

        Args:
            name: Name of the monitor
        """
        self.name = name
        self.metrics: List[Dict[str, Any]] = []

    @abstractmethod
    def start(self) -> None:
        """Start monitoring."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop monitoring."""
        pass

    @abstractmethod
    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get collected metrics.

        Returns:
            List of metric snapshots
        """
        pass
