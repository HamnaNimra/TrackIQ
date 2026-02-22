"""MiniCluster dashboard built on TrackIQ shared UI components."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from minicluster.monitor.health_reader import HealthReader
from minicluster.reporting import MiniClusterHtmlReporter
from trackiq_core.schema import TrackiqResult
from trackiq_core.ui import (
    DARK_THEME,
    LIGHT_THEME,
    LossChart,
    MetricTable,
    PowerGauge,
    RegressionBadge,
    TrackiqDashboard,
    TrackiqTheme,
    WorkerGrid,
)


class MiniClusterDashboard(TrackiqDashboard):
    """Dashboard for MiniCluster distributed training result files."""

    def __init__(
        self,
        result: TrackiqResult,
        theme: TrackiqTheme = LIGHT_THEME,
        title: str = "MiniCluster Dashboard",
    ) -> None:
        super().__init__(result=result, theme=theme, title=title)

    def expected_tool_names(self) -> list[str]:
        """MiniCluster dashboard should only load MiniCluster results."""
        return ["minicluster"]

    def _build_html_report(self, result: TrackiqResult) -> str:
        """Generate the same HTML artifact as `minicluster report html`."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "minicluster_dashboard_report.html"
            MiniClusterHtmlReporter().generate(
                output_path=str(report_path),
                results=[result],
                title="MiniCluster Performance Report",
            )
            return report_path.read_text(encoding="utf-8")

    def _tool_payload(self) -> dict[str, Any]:
        result = self._primary_result()
        return result.tool_payload if isinstance(result.tool_payload, dict) else {}

    def _payload_from_checkpoint(
        self,
        checkpoint: Any,
        base_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        workers = [
            {
                "worker_id": worker.worker_id,
                "throughput": worker.throughput_samples_per_sec,
                "allreduce_time_ms": worker.allreduce_time_ms,
                "status": worker.status,
                "loss": worker.loss,
            }
            for worker in checkpoint.workers
        ]
        snapshots = []
        for worker in checkpoint.workers:
            snapshots.append(
                {
                    "step": worker.step,
                    "loss": worker.loss,
                    "worker_id": worker.worker_id,
                }
            )
        payload = dict(base_payload) if isinstance(base_payload, dict) else {}
        payload["workers"] = workers
        payload["steps"] = snapshots
        payload.setdefault("health_checkpoint_path", self._tool_payload().get("health_checkpoint_path"))
        payload.setdefault("faults_detected", self._tool_payload().get("faults_detected"))
        return payload

    def _render_dynamic_sections(self, payload: dict[str, Any]) -> None:
        """Render worker grid and loss chart from the provided payload."""
        import streamlit as st

        result = self._primary_result()
        workers = payload.get("workers", [])
        steps_data: list[dict[str, Any]] = payload.get("steps", []) if isinstance(payload.get("steps"), list) else []
        steps = [int(item.get("step", idx)) for idx, item in enumerate(steps_data)]
        losses = [float(item.get("loss", 0.0)) for item in steps_data]

        left, right = st.columns(2)
        with left:
            MetricTable(result=result, mode="single", theme=self.theme).render()
        with right:
            if isinstance(workers, list) and workers:
                WorkerGrid(workers=workers, theme=self.theme).render()
            else:
                st.markdown(
                    "<div class='trackiq-card'>Worker data not available in tool payload.</div>",
                    unsafe_allow_html=True,
                )

        if steps_data:
            LossChart(
                steps=steps,
                loss_values=losses,
                baseline_values=None,
                theme=self.theme,
            ).render()
        else:
            st.info("Loss curve unavailable in tool payload.")

    def _render_config_section(self, payload: dict[str, Any]) -> None:
        """Render run configuration and execution context."""
        import streamlit as st

        config = payload.get("config")
        if not isinstance(config, dict):
            st.info("Run configuration not available in tool payload.")
            return

        st.markdown("### Run Configuration")
        left, right = st.columns(2)
        with left:
            st.markdown("**Training**")
            st.markdown(f"- Steps: `{config.get('num_steps', 'N/A')}`")
            st.markdown(f"- Batch Size: `{config.get('batch_size', 'N/A')}`")
            st.markdown(f"- Learning Rate: `{config.get('learning_rate', 'N/A')}`")
            st.markdown(f"- Layers: `{config.get('num_layers', 'N/A')}`")
            st.markdown(f"- Hidden Size: `{config.get('hidden_size', 'N/A')}`")
        with right:
            st.markdown("**Runtime**")
            st.markdown(f"- Workers: `{config.get('num_processes', config.get('num_workers', 'N/A'))}`")
            st.markdown(
                f"- Collective Backend: `{config.get('collective_backend', payload.get('collective_backend', 'N/A'))}`"
            )
            st.markdown(f"- Workload: `{config.get('workload', payload.get('workload_type', 'N/A'))}`")
            st.markdown(f"- Baseline Throughput: `{config.get('baseline_throughput', 'N/A')}`")
            st.markdown(f"- Seed: `{config.get('seed', 'N/A')}`")
            st.markdown(f"- TDP (W): `{config.get('tdp_watts', 'N/A')}`")
            st.markdown(f"- Loss Tolerance: `{config.get('loss_tolerance', 'N/A')}`")
            st.markdown(f"- Regression Threshold (%): `{config.get('regression_threshold', 'N/A')}`")

    def _render_cluster_health_summary(self, payload: dict[str, Any]) -> None:
        """Render aggregate all-reduce and scaling metrics for this run."""
        import streamlit as st

        result = self._primary_result()
        avg_thr = payload.get("average_throughput_samples_per_sec", result.metrics.throughput_samples_per_sec)
        p99_allreduce = payload.get("p99_allreduce_ms")
        p95_allreduce = payload.get("p95_allreduce_ms")
        allreduce_stdev = payload.get("allreduce_stdev_ms")
        scaling_efficiency = payload.get("scaling_efficiency_pct", result.metrics.scaling_efficiency_pct)

        st.markdown("### Cluster Health Summary")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric(
                "Avg Throughput (samples/s)", f"{float(avg_thr):.2f}" if isinstance(avg_thr, (int, float)) else "N/A"
            )
        with col_b:
            st.metric(
                "P99 All-Reduce (ms)",
                f"{float(p99_allreduce):.3f}" if isinstance(p99_allreduce, (int, float)) else "N/A",
            )
        with col_c:
            st.metric(
                "P95 All-Reduce (ms)",
                f"{float(p95_allreduce):.3f}" if isinstance(p95_allreduce, (int, float)) else "N/A",
            )
        with col_d:
            st.metric(
                "Scaling Efficiency (%)",
                f"{float(scaling_efficiency):.2f}" if isinstance(scaling_efficiency, (int, float)) else "N/A",
            )
        st.caption(
            "All-Reduce variability (stdev ms): "
            + (f"{float(allreduce_stdev):.3f}" if isinstance(allreduce_stdev, (int, float)) else "N/A")
        )

    def _render_training_graphs(self, payload: dict[str, Any]) -> None:
        """Render graph-heavy training timelines from step payload."""
        import streamlit as st

        steps_data = payload.get("steps")
        if not isinstance(steps_data, list) or not steps_data:
            st.info("No per-step data available for training graphs.")
            return

        rows: list[dict[str, float]] = []
        for idx, item in enumerate(steps_data):
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "step": float(item.get("step", idx)),
                    "loss": float(item.get("loss", 0.0)),
                    "throughput": float(item.get("throughput_samples_per_sec", 0.0)),
                    "allreduce_ms": float(item.get("allreduce_time_ms", 0.0)),
                    "compute_ms": float(item.get("compute_time_ms", 0.0)),
                }
            )
        if not rows:
            st.info("Step data is malformed; unable to render graphs.")
            return

        st.markdown("### Training Graphs")
        try:
            import pandas as pd
            import plotly.express as px
            import plotly.graph_objects as go
        except Exception:
            # Fallback when plotly/pandas are unavailable
            st.line_chart(
                {
                    "loss": [row["loss"] for row in rows],
                    "throughput": [row["throughput"] for row in rows],
                }
            )
            return

        df = pd.DataFrame(rows)
        plotly_template = "plotly_dark" if self.theme.name == DARK_THEME.name else "plotly_white"
        col_a, col_b = st.columns(2)
        with col_a:
            fig_loss = px.line(
                df,
                x="step",
                y="loss",
                title="Loss by Step",
                labels={"step": "Step", "loss": "Loss"},
            )
            fig_loss.update_layout(template=plotly_template)
            st.plotly_chart(fig_loss, use_container_width=True, key="minicluster_training_loss_by_step_chart")
        with col_b:
            fig_thr = px.line(
                df,
                x="step",
                y="throughput",
                title="Throughput by Step",
                labels={"step": "Step", "throughput": "Samples/sec"},
            )
            fig_thr.update_layout(template=plotly_template)
            st.plotly_chart(fig_thr, use_container_width=True, key="minicluster_training_throughput_by_step_chart")

        fig_timing = go.Figure()
        fig_timing.add_trace(
            go.Bar(
                x=df["step"],
                y=df["compute_ms"],
                name="Compute (ms)",
                marker_color="#2563eb",
            )
        )
        fig_timing.add_trace(
            go.Bar(
                x=df["step"],
                y=df["allreduce_ms"],
                name="Allreduce (ms)",
                marker_color="#dc2626",
            )
        )
        fig_timing.update_layout(
            title="Per-Step Timing Breakdown",
            xaxis_title="Step",
            yaxis_title="Time (ms)",
            barmode="stack",
            template=plotly_template,
        )
        st.plotly_chart(fig_timing, use_container_width=True, key="minicluster_training_timing_breakdown_chart")

        allreduce_values = [row["allreduce_ms"] for row in rows if row["allreduce_ms"] > 0]
        if allreduce_values:
            fig_hist = px.histogram(
                x=allreduce_values,
                nbins=30,
                title="All-Reduce Latency Distribution (ms)",
                labels={"x": "allreduce_time_ms"},
            )
            fig_hist.update_layout(template=plotly_template)
            p99 = payload.get("p99_allreduce_ms")
            if isinstance(p99, (int, float)):
                fig_hist.add_vline(
                    x=float(p99),
                    line_dash="dash",
                    line_color="#dc2626",
                    annotation_text="P99",
                    annotation_position="top right",
                )
            st.plotly_chart(fig_hist, use_container_width=True, key="minicluster_training_allreduce_histogram_chart")

    def render_sidebar(self) -> None:
        """Render shared sidebar plus MiniCluster auto-refresh controls."""
        import streamlit as st

        super().render_sidebar()
        with st.sidebar:
            st.markdown("---")
            st.subheader("Live Refresh")
            current = st.session_state.get("minicluster_auto_refresh", False)
            auto_refresh = st.toggle(
                "Auto-refresh (2s)",
                value=bool(current),
                key="minicluster_auto_refresh",
            )
            st.session_state["trackiq_live_indicator"] = bool(auto_refresh)
            if st.button("Refresh Now", key="minicluster_refresh_now"):
                st.session_state["minicluster_force_refresh"] = True
                if hasattr(st, "rerun"):
                    st.rerun()
                else:  # pragma: no cover
                    st.experimental_rerun()

    def build_components(self) -> dict[str, object]:
        """Build component instances for testable, reusable rendering."""
        result = self._primary_result()
        payload = self._tool_payload()
        steps_data: list[dict[str, Any]] = payload.get("steps", []) if isinstance(payload.get("steps"), list) else []
        steps = [int(item.get("step", idx)) for idx, item in enumerate(steps_data)]
        losses = [float(item.get("loss", 0.0)) for item in steps_data]
        workers = payload.get("workers", [])

        return {
            "regression_badge": RegressionBadge(result.regression, theme=self.theme),
            "metric_table": MetricTable(result=result, mode="single", theme=self.theme),
            "worker_grid": WorkerGrid(workers=workers, theme=self.theme),
            "loss_chart": LossChart(
                steps=steps,
                loss_values=losses,
                baseline_values=None,
                theme=self.theme,
            ),
            "power_gauge": PowerGauge(
                metrics=result.metrics,
                tool_payload=result.tool_payload,
                theme=self.theme,
            ),
        }

    def render_body(self) -> None:
        """Render MiniCluster-specific dashboard content."""
        import streamlit as st

        components = self.build_components()
        payload = self._tool_payload()
        tab_overview, tab_health, tab_training, tab_config, tab_power_kv, tab_faults, tab_downloads = st.tabs(
            [
                "Overview",
                "Cluster Health",
                "Training Graphs",
                "Run Configuration",
                "Power & KV Cache",
                "Faults",
                "Downloads",
            ]
        )
        with tab_overview:
            components["regression_badge"].render()
            dynamic_container = st.empty()
            refresh_status_container = st.empty()
        checkpoint_path = payload.get("health_checkpoint_path")
        force_refresh = bool(st.session_state.pop("minicluster_force_refresh", False))
        auto_refresh = bool(st.session_state.get("minicluster_auto_refresh", False))
        local_payload = payload
        if (auto_refresh or force_refresh) and checkpoint_path:
            reader = HealthReader(str(checkpoint_path), timeout_seconds=2.0)
            checkpoint = reader.read()
            if checkpoint is not None:
                local_payload = self._payload_from_checkpoint(checkpoint, base_payload=local_payload)

        with dynamic_container.container():
            self._render_dynamic_sections(local_payload)

        if auto_refresh and checkpoint_path:
            reader = HealthReader(str(checkpoint_path), timeout_seconds=2.0)
            checkpoint = reader.read()
            if checkpoint is not None:
                st.session_state["minicluster_refresh_failures"] = 0
                local_payload = self._payload_from_checkpoint(checkpoint, base_payload=local_payload)
                with dynamic_container.container():
                    self._render_dynamic_sections(local_payload)
                if checkpoint.is_complete:
                    with refresh_status_container.container():
                        st.success("Live run complete. Auto-refresh stopped.")
                    st.session_state["minicluster_auto_refresh"] = False
                    st.session_state["trackiq_live_indicator"] = False
                else:
                    with refresh_status_container.container():
                        st.caption("Auto-refresh active. Updating every 2 seconds.")
                    time.sleep(2)
                    if hasattr(st, "rerun"):
                        st.rerun()
                    else:  # pragma: no cover
                        st.experimental_rerun()
            else:
                failures = int(st.session_state.get("minicluster_refresh_failures", 0)) + 1
                st.session_state["minicluster_refresh_failures"] = failures
                if failures >= 3:
                    with refresh_status_container.container():
                        st.warning(
                            "Auto-refresh stopped: live checkpoint is unavailable. "
                            "Use Refresh Now or disable auto-refresh."
                        )
                    st.session_state["minicluster_auto_refresh"] = False
                    st.session_state["trackiq_live_indicator"] = False
                else:
                    with refresh_status_container.container():
                        st.caption("Waiting for live checkpoint...")
                    time.sleep(2)
                    if hasattr(st, "rerun"):
                        st.rerun()
                    else:  # pragma: no cover
                        st.experimental_rerun()
        else:
            st.session_state["minicluster_refresh_failures"] = 0

        with tab_health:
            self._render_cluster_health_summary(local_payload)

        with tab_training:
            self._render_training_graphs(local_payload)

        with tab_config:
            self._render_config_section(local_payload)

        with tab_power_kv:
            components["power_gauge"].render()
            self.render_kv_cache_section()

        with tab_faults:
            faults = local_payload.get("faults_detected")
            if faults is None:
                st.info("No fault detection data available.")
            else:
                st.json(faults)

        with tab_downloads:
            self.render_download_section()
