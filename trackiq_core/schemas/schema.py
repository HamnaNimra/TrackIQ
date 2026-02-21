"""Result schema definitions for TrackIQ."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class AnalysisResult:
    """Base schema for analysis results."""

    name: str
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: dict[str, Any] = field(default_factory=dict)
    raw_data: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
        }
