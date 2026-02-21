"""Core abstractions and base classes for TrackIQ."""

from abc import ABC, abstractmethod
from typing import Any

from trackiq_core.schemas import AnalysisResult


class BaseAnalyzer(ABC):
    """Base class for all analyzers."""

    def __init__(self, name: str):
        self.name = name
        self.results: list[AnalysisResult] = []

    @abstractmethod
    def analyze(self, data: Any) -> AnalysisResult:
        """Perform analysis on data."""
        pass

    def add_result(self, result: AnalysisResult) -> None:
        """Store analysis result."""
        self.results.append(result)

    def get_results(self) -> list[AnalysisResult]:
        """Get all stored results."""
        return self.results


class BaseBenchmark(ABC):
    """Base class for all benchmarks."""

    def __init__(self, name: str):
        self.name = name
        self.results: dict[str, Any] = {}

    @abstractmethod
    def run(self, **kwargs) -> dict[str, Any]:
        """Run the benchmark."""
        pass

    def get_results(self) -> dict[str, Any]:
        """Get benchmark results."""
        return self.results


class BaseMonitor(ABC):
    """Base class for all monitors."""

    def __init__(self, name: str):
        self.name = name
        self.metrics: list[dict[str, Any]] = []

    @abstractmethod
    def start(self) -> None:
        """Start monitoring."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop monitoring."""
        pass

    @abstractmethod
    def get_metrics(self) -> list[dict[str, Any]]:
        """Get collected metrics."""
        pass
