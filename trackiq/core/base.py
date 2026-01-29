"""Core abstractions and base classes for TrackIQ."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from trackiq.results import AnalysisResult


class BaseAnalyzer(ABC):
    """Base class for all analyzers."""

    def __init__(self, name: str):
        self.name = name
        self.results: List[AnalysisResult] = []

    @abstractmethod
    def analyze(self, data: Any) -> AnalysisResult:
        """Perform analysis on data."""
        pass

    def add_result(self, result: AnalysisResult) -> None:
        """Store analysis result."""
        self.results.append(result)

    def get_results(self) -> List[AnalysisResult]:
        """Get all stored results."""
        return self.results


class BaseBenchmark(ABC):
    """Base class for all benchmarks."""

    def __init__(self, name: str):
        self.name = name
        self.results: Dict[str, Any] = {}

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """Run the benchmark."""
        pass

    def get_results(self) -> Dict[str, Any]:
        """Get benchmark results."""
        return self.results


class BaseMonitor(ABC):
    """Base class for all monitors."""

    def __init__(self, name: str):
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
        """Get collected metrics."""
        pass
