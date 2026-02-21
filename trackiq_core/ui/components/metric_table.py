"""Metric table component for TrackIQ dashboards."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Literal, Optional, Union

from trackiq_core.schema import TrackiqResult
from trackiq_core.ui.theme import DARK_THEME, TrackiqTheme


LOWER_IS_BETTER = {"power_consumption_watts", "energy_per_step_joules"}


class MetricTable:
    """Render and serialize metric tabular views for one or two results."""

    def __init__(
        self,
        result: Union[TrackiqResult, List[TrackiqResult]],
        mode: Literal["single", "comparison"] = "single",
        theme: TrackiqTheme = DARK_THEME,
    ) -> None:
        self.result = result
        self.mode = mode
        self.theme = theme

    def _format_value(self, value: Optional[float]) -> Any:
        return "N/A" if value is None else value

    def _single_payload(self) -> Dict[str, Any]:
        result = self.result[0] if isinstance(self.result, list) else self.result
        metrics = asdict(result.metrics)
        return {
            "mode": "single",
            "metrics": {name: self._format_value(value) for name, value in metrics.items()},
        }

    def _compare_metric(
        self, name: str, value_a: Optional[float], value_b: Optional[float]
    ) -> Dict[str, Any]:
        if value_a is None or value_b is None:
            return {
                "metric": name,
                "result_a": self._format_value(value_a),
                "result_b": self._format_value(value_b),
                "delta_percent": "N/A",
                "winner": "not_comparable",
            }

        delta_percent = (
            ((float(value_b) - float(value_a)) / float(value_a)) * 100.0
            if value_a != 0
            else (0.0 if value_b == 0 else float("inf"))
        )
        lower_is_better = name in LOWER_IS_BETTER
        if value_a == value_b:
            winner = "tie"
        elif lower_is_better:
            winner = "A" if value_a < value_b else "B"
        else:
            winner = "A" if value_a > value_b else "B"

        return {
            "metric": name,
            "result_a": float(value_a),
            "result_b": float(value_b),
            "delta_percent": delta_percent,
            "winner": winner,
        }

    def _comparison_payload(self) -> Dict[str, Any]:
        if not isinstance(self.result, list) or len(self.result) != 2:
            raise ValueError("Comparison mode requires exactly two TrackiqResult objects.")

        metrics_a = asdict(self.result[0].metrics)
        metrics_b = asdict(self.result[1].metrics)
        all_names = sorted(set(metrics_a.keys()) | set(metrics_b.keys()))
        rows = [
            self._compare_metric(name, metrics_a.get(name), metrics_b.get(name))
            for name in all_names
        ]
        return {"mode": "comparison", "metrics": rows}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize table payload without any Streamlit dependency."""
        if self.mode == "single":
            return self._single_payload()
        return self._comparison_payload()

    def render(self) -> None:
        """Render metric table in Streamlit."""
        import streamlit as st

        payload = self.to_dict()
        st.markdown(
            f"<div style='font-weight:700;color:{self.theme.text_color};'>Metrics</div>",
            unsafe_allow_html=True,
        )
        if payload["mode"] == "single":
            st.markdown(
                f"<div style='background:{self.theme.surface_color};height:8px;border-radius:{self.theme.border_radius};margin:4px 0 8px 0;'></div>",
                unsafe_allow_html=True,
            )
            rows = [
                {"Metric": k, "Value": v}
                for k, v in payload["metrics"].items()
            ]
            st.table(rows)
            return

        st.markdown(
            (
                f"<div style='margin:4px 0 8px 0;'>"
                f"<span style='color:{self.theme.pass_color};font-weight:700;'>Result B better</span> | "
                f"<span style='color:{self.theme.fail_color};font-weight:700;'>Result A better</span> | "
                f"<span style='color:{self.theme.warning_color};font-weight:700;'>tie</span>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        rows = []
        for row in payload["metrics"]:
            winner = row["winner"]
            if winner == "A":
                winner_text = "Result A"
            elif winner == "B":
                winner_text = "Result B"
            elif winner == "tie":
                winner_text = "tie"
            else:
                winner_text = "N/A"
            rows.append(
                {
                    "Metric": row["metric"],
                    "Result A": row["result_a"],
                    "Result B": row["result_b"],
                    "Delta %": row["delta_percent"],
                    "Winner": winner_text,
                }
            )
        st.table(rows)
