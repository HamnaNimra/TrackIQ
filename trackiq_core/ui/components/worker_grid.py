"""Worker status grid component."""

from typing import Any, Dict, List

from trackiq_core.ui.theme import DARK_THEME, TrackiqTheme


class WorkerGrid:
    """Render distributed worker health cards."""

    def __init__(
        self,
        workers: List[Dict[str, Any]],
        theme: TrackiqTheme = DARK_THEME,
    ) -> None:
        self.workers = workers
        self.theme = theme

    def to_dict(self) -> Dict[str, Any]:
        """Return raw worker entries."""
        return {"workers": list(self.workers)}

    def render(self) -> None:
        """Render worker cards in a responsive grid."""
        import streamlit as st

        st.subheader("Worker Status")
        if not self.workers:
            st.info("No worker data available.")
            return

        columns = st.columns(min(4, max(1, len(self.workers))))
        for idx, worker in enumerate(self.workers):
            status = str(worker.get("status", "unknown")).lower()
            if status == "healthy":
                color = self.theme.pass_color
            elif status == "failed":
                color = self.theme.fail_color
            else:
                color = self.theme.warning_color

            with columns[idx % len(columns)]:
                st.markdown(
                    f"""
                    <div class="trackiq-card">
                        <div><b>Worker:</b> {worker.get("worker_id", "N/A")}</div>
                        <div><b>Throughput:</b> {worker.get("throughput", "N/A")}</div>
                        <div><b>AllReduce ms:</b> {worker.get("allreduce_time_ms", "N/A")}</div>
                        <div style="color:{color};"><b>Status:</b> {worker.get("status", "N/A")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

