"""Result schema definitions for TrackIQ."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass
class AnalysisResult:
    """Base schema for analysis results."""

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
