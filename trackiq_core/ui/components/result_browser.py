"""Result browser component for loading TrackiqResult JSON files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from trackiq_core.serializer import load_trackiq_result
from trackiq_core.ui.theme import DARK_THEME, TrackiqTheme


class ResultBrowser:
    """Browse and load TrackiqResult files from configured directories."""

    def __init__(
        self,
        search_paths: Optional[List[str]] = None,
        allowed_tools: Optional[List[str]] = None,
        theme: TrackiqTheme = DARK_THEME,
    ) -> None:
        self.search_paths = search_paths or [
            "./output",
            "./minicluster_results",
            "./autoperfpy_results",
            "./trackiq_compare_results",
        ]
        self.allowed_tools = (
            {str(tool).strip().lower() for tool in allowed_tools if str(tool).strip()}
            if allowed_tools
            else None
        )
        self.theme = theme

    def _scan_results(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for search_path in self.search_paths:
            path = Path(search_path)
            if not path.exists() or not path.is_dir():
                continue
            for json_path in path.glob("*.json"):
                try:
                    result = load_trackiq_result(json_path)
                except Exception:
                    continue
                if self.allowed_tools is not None:
                    tool_name = str(result.tool_name).strip().lower()
                    if tool_name not in self.allowed_tools:
                        continue
                stat = json_path.stat()
                rows.append(
                    {
                        "path": str(json_path),
                        "tool_name": result.tool_name,
                        "workload_name": result.workload.name,
                        "timestamp": result.timestamp,
                        "regression_status": result.regression.status,
                        "file_size_bytes": int(stat.st_size),
                    }
                )
        rows.sort(key=lambda row: row["timestamp"], reverse=True)
        return rows

    def to_dict(self) -> List[Dict[str, Any]]:
        """Return metadata for discovered result files."""
        rows = self._scan_results()
        normalized: List[Dict[str, Any]] = []
        for row in rows:
            normalized.append(
                {
                    "path": row["path"],
                    "tool_name": row["tool_name"],
                    "workload_name": row["workload_name"],
                    "timestamp": row["timestamp"].isoformat(),
                    "regression_status": row["regression_status"],
                    "file_size_bytes": row["file_size_bytes"],
                }
            )
        return normalized

    def render(self) -> None:
        """Render interactive result browser with load actions."""
        import streamlit as st

        col_refresh, _ = st.columns([1, 4])
        with col_refresh:
            st.button("Refresh", key="trackiq_results_refresh")

        rows = self._scan_results()
        if not rows:
            st.info("No valid TrackiqResult JSON files found.")
        for idx, row in enumerate(rows):
            status_color = (
                self.theme.pass_color
                if row["regression_status"] == "pass"
                else self.theme.fail_color
            )
            col_info, col_load = st.columns([4, 1])
            with col_info:
                st.markdown(
                    f"<div style='background:{self.theme.surface_color};padding:8px;border-radius:{self.theme.border_radius};'>"
                    f"<div><b>{row['tool_name']}</b> - {row['workload_name']}</div>"
                    f"<div>{row['timestamp'].isoformat()} | {row['file_size_bytes']} bytes</div>"
                    f"<div style='color:{status_color};font-weight:700;'>{row['regression_status'].upper()}</div>"
                    f"<div style='font-size:12px;'>{row['path']}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            with col_load:
                if st.button("Load", key=f"trackiq_load_{idx}"):
                    loaded = load_trackiq_result(row["path"])
                    tool_name = str(loaded.tool_name).strip().lower()
                    if self.allowed_tools is not None and tool_name not in self.allowed_tools:
                        st.error(
                            f"Loaded result tool '{loaded.tool_name}' is not allowed in this dashboard."
                        )
                        continue
                    st.session_state["loaded_result"] = loaded
                    st.session_state["loaded_result_path"] = row["path"]
                    if hasattr(st, "rerun"):
                        st.rerun()
                    else:  # pragma: no cover - compatibility fallback
                        st.experimental_rerun()

        st.markdown("---")
        manual_path = st.text_input("Manual file path", key="trackiq_manual_result_path")
        if st.button("Load Manual Path", key="trackiq_manual_load"):
            try:
                loaded = load_trackiq_result(manual_path)
                tool_name = str(loaded.tool_name).strip().lower()
                if self.allowed_tools is not None and tool_name not in self.allowed_tools:
                    allowed = ", ".join(sorted(self.allowed_tools))
                    st.error(
                        f"Result tool '{loaded.tool_name}' is not allowed here. "
                        f"Allowed tools: {allowed}"
                    )
                    return
                st.session_state["loaded_result"] = loaded
                st.session_state["loaded_result_path"] = manual_path
                if hasattr(st, "rerun"):
                    st.rerun()
                else:  # pragma: no cover - compatibility fallback
                    st.experimental_rerun()
            except Exception as exc:
                st.error(f"Failed to load result file: {exc}")
