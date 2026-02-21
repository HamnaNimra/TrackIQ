"""Report command handlers for AutoPerfPy CLI."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from typing import Any

import numpy as np

from autoperfpy.reporting import HTMLReportGenerator, PDFReportGenerator, PerformanceVisualizer
from trackiq_core.reporting import PDF_BACKEND_AUTO, PdfBackendError
from trackiq_core.utils.errors import DependencyError, HardwareNotFoundError


def run_report_html(
    args: Any,
    config: Any,
    *,
    run_default_benchmark: Callable[[str | None, int], tuple[dict, str | None, str | None]],
    normalize_report_input_data: Callable[[object], dict[str, Any]],
    output_path: Callable[[Any, str], str],
    save_trackiq_wrapped_json: Callable[[str, Any, str, str], None],
    write_result_to_csv: Callable[[dict[str, Any], str], bool],
) -> Any:
    """Generate HTML report."""
    del config
    import pandas as pd

    json_path_to_cleanup = None
    if not getattr(args, "csv", None) and not getattr(args, "json", None):
        print("No --csv/--json provided; running a quick benchmark to generate data...")
        try:
            _, _, json_path = run_default_benchmark(
                device_id=getattr(args, "device", None),
                duration_seconds=getattr(args, "duration", 10),
            )
            if json_path:
                args.json = json_path
                json_path_to_cleanup = json_path
        except (HardwareNotFoundError, DependencyError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return None

    try:
        report = HTMLReportGenerator(
            title=args.title,
            author=args.author,
            theme=args.theme,
        )

        viz = PerformanceVisualizer()

        data_source = "Sample Data"
        if getattr(args, "csv", None):
            data_source = args.csv
        elif getattr(args, "json", None):
            data_source = args.json
        report.add_metadata("Data Source", data_source)

        if getattr(args, "json", None):
            with open(args.json, encoding="utf-8") as handle:
                data = normalize_report_input_data(json.load(handle))

            if "comparisons" in data and "summary" in data and "config" in data:
                report.add_metadata("Validation Type", "Distributed Training")
                config_data = data["config"]
                report.add_metadata("Training Steps", str(config_data.get("num_steps", 0)))
                report.add_metadata("Processes", str(config_data.get("num_processes", 1)))
                report.add_metadata("Loss Tolerance", str(config_data.get("loss_tolerance", 0.01)))

                summary = data["summary"]
                report.add_summary_item("Total Steps", summary["total_steps"], "", "neutral")
                report.add_summary_item(
                    "Passed Steps", summary["passed_steps"], "", "good" if summary["passed_steps"] > 0 else "neutral"
                )
                report.add_summary_item(
                    "Failed Steps",
                    summary["failed_steps"],
                    "",
                    "critical" if summary["failed_steps"] > 0 else "neutral",
                )
                report.add_summary_item(
                    "Pass Rate", f"{summary['pass_rate']:.1%}", "", "good" if summary["overall_pass"] else "critical"
                )
                status = "good" if summary["overall_pass"] else "critical"
                report.add_summary_item(
                    "Overall Status",
                    "PASS" if summary["overall_pass"] else "FAIL",
                    "",
                    status,
                )

                report.add_section(
                    "Step-by-Step Loss Comparison", "Comparison of single-process vs multi-process losses"
                )
                comparisons = data["comparisons"]
                table_data = []
                for comparison in comparisons:
                    step = comparison["step"]
                    single_loss = comparison["single_process_loss"]
                    multi_loss = comparison["multi_process_loss"]
                    delta = comparison["absolute_delta"]
                    rel_delta = comparison["relative_delta"]
                    passed = comparison["passed"]
                    status_icon = "PASS" if passed else "FAIL"
                    table_data.append(
                        [
                            str(step),
                            f"{single_loss:.6f}",
                            f"{multi_loss:.6f}",
                            f"{delta:.6f}",
                            f"{rel_delta:.4f}",
                            status_icon,
                        ]
                    )

                headers = ["Step", "Single Loss", "Multi Loss", "Abs Delta", "Rel Delta", "Status"]
                report.add_table("Loss Comparison", headers, table_data, "Step-by-Step Loss Comparison")

                if len(comparisons) > 1:
                    import matplotlib.pyplot as plt

                    steps = [comparison["step"] for comparison in comparisons]
                    single_losses = [comparison["single_process_loss"] for comparison in comparisons]
                    multi_losses = [comparison["multi_process_loss"] for comparison in comparisons]

                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(steps, single_losses, label="Single Process", marker="o")
                    ax.plot(steps, multi_losses, label="Multi Process", marker="s")
                    ax.set_xlabel("Training Step")
                    ax.set_ylabel("Loss")
                    ax.set_title("Loss Comparison: Single vs Multi Process")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    report.add_figure(fig, "Loss Comparison Chart", "Step-by-Step Loss Comparison")

            else:
                report.add_metadata("Collector", data.get("collector_name", "-"))
                if data.get("profile"):
                    report.add_metadata("Profile", data["profile"])
                summary = data.get("summary", {})
                sample_count = data.get("sample_count") or summary.get("sample_count") or len(data.get("samples", []))
                report.add_summary_item("Samples", sample_count, "", "neutral")
                latency = summary.get("latency", {})
                if latency:
                    report.add_summary_item("P99 Latency", f"{latency.get('p99_ms', 0):.2f}", "ms", "neutral")
                    report.add_summary_item("P50 Latency", f"{latency.get('p50_ms', 0):.2f}", "ms", "neutral")
                    report.add_summary_item("Mean Latency", f"{latency.get('mean_ms', 0):.2f}", "ms", "neutral")
                throughput = summary.get("throughput", {})
                if throughput:
                    report.add_summary_item(
                        "Mean Throughput",
                        f"{throughput.get('mean_fps', 0):.1f}",
                        "FPS",
                        "neutral",
                    )
                power = summary.get("power", {})
                if power and power.get("mean_w") is not None:
                    report.add_summary_item("Mean Power", f"{power.get('mean_w', 0):.1f}", "W", "neutral")
                if data.get("validation"):
                    validation = data["validation"]
                    status = "good" if validation.get("overall_pass") else "critical"
                    report.add_summary_item(
                        "Run Status",
                        "PASS" if validation.get("overall_pass") else "FAIL",
                        "",
                        status,
                    )
                samples = data.get("samples", [])
                if samples:
                    from autoperfpy.reports.charts import add_charts_to_html_report, samples_to_dataframe

                    report_df = samples_to_dataframe(samples)
                    if "latency_ms" in report_df.columns and "throughput_fps" not in report_df.columns:
                        report_df["throughput_fps"] = 1000.0 / report_df["latency_ms"].replace(0, np.nan)
                    add_charts_to_html_report(report, report_df, summary)
                    if not report.html_figures:
                        latencies = []
                        for sample in samples:
                            metrics = sample.get("metrics", sample) if isinstance(sample, dict) else {}
                            if isinstance(metrics, dict) and "latency_ms" in metrics:
                                latencies.append(metrics["latency_ms"])
                        if latencies:
                            report.add_section("Latency Analysis", "From benchmark samples")
                            by_run = {
                                "Run": {
                                    "P50": float(np.percentile(latencies, 50)),
                                    "P95": float(np.percentile(latencies, 95)),
                                    "P99": float(np.percentile(latencies, 99)),
                                }
                            }
                            fig = viz.plot_latency_percentiles(by_run)
                            report.add_figure(fig, "Latency Percentiles", "Latency Analysis")
                base = os.path.splitext(os.path.basename(args.output))[0]
                json_out = getattr(args, "export_json", None) or (base + "_data.json")
                csv_out = getattr(args, "export_csv", None) or (base + "_data.csv")
                json_out = output_path(args, json_out)
                csv_out = output_path(args, csv_out)
                save_trackiq_wrapped_json(
                    json_out,
                    data,
                    workload_name="html_report_data",
                    workload_type="inference",
                )
                print(f"[OK] JSON exported to: {json_out}")
                if write_result_to_csv(data, csv_out):
                    print(f"[OK] CSV exported to: {csv_out}")
                report_output = output_path(args, args.output)
                report_output = report.generate_html(report_output)
                print(f"\n[OK] HTML report generated: {report_output}")
                return {"output_path": report_output}

        if getattr(args, "csv", None):
            df = pd.read_csv(args.csv)

        if "latency_ms" in df.columns:
            report.add_summary_item("Samples", len(df), "", "neutral")
            report.add_summary_item("Mean Latency", f"{df['latency_ms'].mean():.2f}", "ms", "neutral")
            report.add_summary_item("P99 Latency", f"{df['latency_ms'].quantile(0.99):.2f}", "ms", "neutral")

            cv = df["latency_ms"].std() / df["latency_ms"].mean() * 100
            status = "good" if cv < 10 else "warning" if cv < 20 else "critical"
            report.add_summary_item("CV", f"{cv:.1f}", "%", status)

            if "workload" in df.columns and "latency_ms" in df.columns:
                report.add_section("Latency Analysis", "Percentile latency comparisons across workloads")

                latencies_by_workload = {}
                for workload in df["workload"].unique():
                    workload_df = df[df["workload"] == workload]["latency_ms"]
                    latencies_by_workload[workload] = {
                        "P50": workload_df.quantile(0.5),
                        "P95": workload_df.quantile(0.95),
                        "P99": workload_df.quantile(0.99),
                    }
                fig = viz.plot_latency_percentiles(latencies_by_workload)
                report.add_figure(fig, "Latency Percentiles by Workload", "Latency Analysis")

                data_dict = {
                    workload: df[df["workload"] == workload]["latency_ms"].tolist()
                    for workload in df["workload"].unique()
                }
                fig = viz.plot_distribution(data_dict, "Latency Distribution Comparison")
                report.add_figure(fig, "Latency Distribution", "Latency Analysis")

            if "batch_size" in df.columns and "latency_ms" in df.columns:
                report.add_section("Batch Analysis", "Performance vs batch size")

                batch_df = df.groupby("batch_size").agg({"latency_ms": "mean"}).reset_index()

                if "throughput" not in df.columns:
                    batch_df["throughput"] = batch_df["batch_size"] * 1000 / batch_df["latency_ms"]
                else:
                    batch_df["throughput"] = df.groupby("batch_size")["throughput"].mean().values

                fig = viz.plot_latency_throughput_tradeoff(
                    batch_df["batch_size"].tolist(),
                    batch_df["latency_ms"].tolist(),
                    batch_df["throughput"].tolist(),
                )
                report.add_figure(fig, "Latency vs Throughput Trade-off", "Batch Analysis")

            if "power_w" in df.columns:
                report.add_section("Power Analysis", "Power consumption and efficiency metrics")

                if "workload" in df.columns:
                    workloads = df["workload"].unique().tolist()
                    power_values = [df[df["workload"] == workload]["power_w"].mean() for workload in workloads]
                    if "latency_ms" in df.columns:
                        perf_values = [
                            1000 / df[df["workload"] == workload]["latency_ms"].mean() for workload in workloads
                        ]
                        fig = viz.plot_power_vs_performance(workloads, power_values, perf_values)
                        report.add_figure(fig, "Power vs Performance", "Power Analysis")

            if len(df) > 0:
                sample_df = df.head(20)
                headers = sample_df.columns.tolist()
                rows = sample_df.values.tolist()
                rows = [[f"{value:.2f}" if isinstance(value, float) else value for value in row] for row in rows]
                report.add_table("Sample Data (First 20 rows)", headers, rows, "Data Overview")
                report.add_section("Data Overview", "Raw data samples")

        else:
            report.add_summary_item("Status", "No data file provided", "", "warning")
            report.add_summary_item(
                "Note",
                "Provide --csv or --json for dynamic graphs, or run without options to auto-run a benchmark.",
                "",
                "neutral",
            )
            report.add_section("No Data", "Run a benchmark and pass --csv/--json to generate visualizations.")

        report_output = report.generate_html(output_path(args, args.output))
        print(f"\n[OK] HTML report generated: {report_output}")
        return {"output_path": report_output}
    finally:
        if json_path_to_cleanup and os.path.exists(json_path_to_cleanup):
            try:
                os.unlink(json_path_to_cleanup)
            except OSError:
                pass


def run_report_pdf(
    args: Any,
    config: Any,
    *,
    run_default_benchmark: Callable[[str | None, int], tuple[dict, str | None, str | None]],
    normalize_report_input_data: Callable[[object], dict[str, Any]],
    output_path: Callable[[Any, str], str],
    save_trackiq_wrapped_json: Callable[[str, Any, str, str], None],
    write_result_to_csv: Callable[[dict[str, Any], str], bool],
) -> Any:
    """Generate PDF report (same content as HTML, converted to PDF)."""
    del config
    import pandas as pd

    json_path_to_cleanup = None
    if not getattr(args, "csv", None) and not getattr(args, "json", None):
        print("No --csv/--json provided; running a quick benchmark to generate data...")
        try:
            _, _, json_path = run_default_benchmark(
                device_id=getattr(args, "device", None),
                duration_seconds=getattr(args, "duration", 10),
            )
            if json_path:
                args.json = json_path
                json_path_to_cleanup = json_path
        except (HardwareNotFoundError, DependencyError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return None

    try:
        report = PDFReportGenerator(
            title=args.title,
            author=args.author,
            pdf_backend=getattr(args, "pdf_backend", PDF_BACKEND_AUTO),
        )

        data_source = getattr(args, "csv", None) or getattr(args, "json", None) or "Sample Data"
        report.add_metadata("Data Source", data_source)

        if getattr(args, "json", None):
            with open(args.json, encoding="utf-8") as handle:
                data = normalize_report_input_data(json.load(handle))
            samples = data.get("samples", [])
            summary = data.get("summary", {})
            report.add_metadata("Collector", data.get("collector_name", "-"))
            report.add_metadata("Total Samples", str(len(samples)))

            latency = summary.get("latency", {})
            if latency:
                report.add_summary_item("P99 Latency", f"{latency.get('p99_ms', 0):.2f}", "ms", "neutral")
                report.add_summary_item("Mean Latency", f"{latency.get('mean_ms', 0):.2f}", "ms", "neutral")
            throughput = summary.get("throughput", {})
            if throughput:
                report.add_summary_item("Mean Throughput", f"{throughput.get('mean_fps', 0):.1f}", "FPS", "neutral")
            power = summary.get("power", {})
            if power and power.get("mean_w") is not None:
                report.add_summary_item("Mean Power", f"{power.get('mean_w', 0):.1f}", "W", "neutral")

            if samples:
                report.add_charts_from_data(samples, summary)

            base = os.path.splitext(os.path.basename(args.output))[0]
            json_out = getattr(args, "export_json", None) or (base + "_data.json")
            csv_out = getattr(args, "export_csv", None) or (base + "_data.csv")
            json_out = output_path(args, json_out)
            csv_out = output_path(args, csv_out)
            save_trackiq_wrapped_json(
                json_out,
                data,
                workload_name="pdf_report_data",
                workload_type="inference",
            )
            print(f"[OK] JSON exported to: {json_out}")
            if write_result_to_csv(data, csv_out):
                print(f"[OK] CSV exported to: {csv_out}")

            report_output = report.generate_pdf(
                output_path(args, args.output),
                backend=getattr(args, "pdf_backend", PDF_BACKEND_AUTO),
            )
            if report.last_render_outcome and report.last_render_outcome.used_fallback:
                print(
                    "[WARN] Primary PDF backend unavailable; used matplotlib fallback.",
                    file=sys.stderr,
                )
            print(f"\n[OK] PDF report generated: {report_output}")
            return {"output_path": report_output}

        if getattr(args, "csv", None):
            from autoperfpy.reports.charts import compute_summary_from_dataframe, ensure_throughput_column

            df = pd.read_csv(args.csv)
            report.add_metadata("Total Samples", str(len(df)))

            if "throughput" in df.columns and "throughput_fps" not in df.columns:
                df["throughput_fps"] = df["throughput"]
            ensure_throughput_column(df)
            summary = compute_summary_from_dataframe(df)

            if summary.get("latency"):
                report.add_summary_item("P99 Latency", f"{summary['latency']['p99_ms']:.2f}", "ms", "neutral")

            samples = [{"timestamp": row.get("timestamp", 0), "metrics": row.to_dict()} for _, row in df.iterrows()]
            if samples:
                report.add_charts_from_data(samples, summary)
        else:
            report.add_metadata(
                "Status",
                "No data file provided. Use --csv/--json or run without options to auto-run a benchmark.",
            )

        report_output = report.generate_pdf(
            output_path(args, args.output),
            backend=getattr(args, "pdf_backend", PDF_BACKEND_AUTO),
        )
        if report.last_render_outcome and report.last_render_outcome.used_fallback:
            print(
                "[WARN] Primary PDF backend unavailable; used matplotlib fallback.",
                file=sys.stderr,
            )
        print(f"\n[OK] PDF report generated: {report_output}")
        return {"output_path": report_output}
    except PdfBackendError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return None
    finally:
        if json_path_to_cleanup and os.path.exists(json_path_to_cleanup):
            try:
                os.unlink(json_path_to_cleanup)
            except OSError:
                pass
