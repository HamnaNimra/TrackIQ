"""Run-history loading and trend chart component for TrackIQ dashboards."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trackiq_core.schema import TrackiqResult
from trackiq_core.serializer import load_trackiq_result
from trackiq_core.ui.theme import DARK_THEME, TrackiqTheme


class RunHistoryLoader:
    """Load TrackiqResult history files from a directory."""

    def __init__(self, history_dir: str, pattern: str = "*.json") -> None:
        self.history_dir = history_dir
        self.pattern = pattern

    def load(self) -> list[TrackiqResult]:
        """Load and timestamp-sort valid TrackiqResult files."""
        directory = Path(self.history_dir)
        if not directory.exists() or not directory.is_dir():
            return []

        results: list[TrackiqResult] = []
        for json_path in sorted(directory.glob(self.pattern)):
            try:
                results.append(load_trackiq_result(json_path))
            except Exception:
                continue
        results.sort(key=lambda item: _timestamp_sort_key(item.timestamp))
        return results


class TrendChart:
    """Serialize and render metric-over-time trends from run history."""

    DEFAULT_METRICS = (
        "throughput_samples_per_sec",
        "latency_p99_ms",
        "performance_per_watt",
    )

    METRIC_LABELS = {
        "throughput_samples_per_sec": "Throughput (samples/s)",
        "latency_p99_ms": "P99 Latency (ms)",
        "performance_per_watt": "Performance per Watt",
    }

    def __init__(
        self,
        results: Sequence[TrackiqResult],
        metric_names: Iterable[str] | None = None,
        theme: TrackiqTheme = DARK_THEME,
    ) -> None:
        self.results = sorted(list(results), key=lambda item: _timestamp_sort_key(item.timestamp))
        self.metric_names = list(metric_names) if metric_names is not None else list(self.DEFAULT_METRICS)
        self.theme = theme

    def _metric_points(self, metric_name: str) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        for result in self.results:
            value = getattr(result.metrics, metric_name, None)
            if value is None:
                continue
            points.append(
                {
                    "timestamp": result.timestamp.isoformat(),
                    "value": float(value),
                    "tool_name": result.tool_name,
                    "workload_name": result.workload.name,
                }
            )
        return points

    def to_dict(self) -> dict[str, Any]:
        """Return trend data suitable for tests or API export."""
        trends = {metric: self._metric_points(metric) for metric in self.metric_names}
        return {
            "run_count": len(self.results),
            "metric_names": list(self.metric_names),
            "trends": trends,
        }

    def render(self) -> None:
        """Render trend charts for selected metrics in Streamlit."""
        import streamlit as st

        payload = self.to_dict()
        st.markdown(
            f"<div style='font-weight:700;color:{self.theme.text_color};'>Run Trends</div>",
            unsafe_allow_html=True,
        )
        if payload["run_count"] == 0:
            st.info("No runs available for trend analysis.")
            return

        for metric_name in payload["metric_names"]:
            points = payload["trends"].get(metric_name, [])
            label = self.METRIC_LABELS.get(metric_name, metric_name.replace("_", " ").title())
            st.markdown(
                f"<div style='margin-top:8px;font-weight:600;color:{self.theme.text_color};'>{label}</div>",
                unsafe_allow_html=True,
            )
            if not points:
                st.caption("No data for this metric in selected runs.")
                continue

            rows = [{"timestamp": item["timestamp"], label: item["value"]} for item in points]
            st.line_chart(rows, x="timestamp", y=label)


def _timestamp_sort_key(value: datetime) -> float:
    """Normalize naive/aware timestamps to UTC epoch for stable sorting."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc).timestamp()
    return value.astimezone(timezone.utc).timestamp()
