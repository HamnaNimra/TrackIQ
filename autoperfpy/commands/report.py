"""Report command handlers for AutoPerfPy CLI."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from typing import Any

from trackiq_core.reporting import PDF_BACKEND_AUTO, PdfBackendError
from trackiq_core.utils.errors import DependencyError, HardwareNotFoundError


def _has_samples(payload: object) -> bool:
    if isinstance(payload, dict):
        samples = payload.get("samples")
        if isinstance(samples, list) and len(samples) > 0:
            return True
        tool_payload = payload.get("tool_payload")
        if isinstance(tool_payload, dict):
            nested_samples = tool_payload.get("samples")
            return isinstance(nested_samples, list) and len(nested_samples) > 0
    return False


def _is_plotly_report_candidate(payload: object) -> bool:
    """Return True when payload should use inference Plotly report path."""
    if not isinstance(payload, dict):
        return False
    if _has_samples(payload):
        return False
    if "mean_ttft_ms" in payload or "throughput_tokens_per_sec" in payload:
        return True
    metrics = payload.get("metrics")
    if isinstance(metrics, dict) and (
        "ttft_ms" in metrics
        or "tokens_per_sec" in metrics
        or "latency_p50_ms" in metrics
        or "latency_p95_ms" in metrics
        or "latency_p99_ms" in metrics
    ):
        return True
    if payload.get("tool_name") == "autoperfpy":
        return True
    return False


def _write_multi_run_csv(runs: list[dict[str, Any]], path: str) -> bool:
    """Write consolidated CSV rows for multiple runs."""
    rows: list[tuple[Any, ...]] = []
    for idx, run in enumerate(runs):
        if not isinstance(run, dict):
            continue
        run_label = run.get("run_label") or run.get("collector_name") or f"run_{idx + 1}"
        batch_size = (
            run.get("inference_config", {}).get("batch_size", 1) if isinstance(run.get("inference_config"), dict) else 1
        )
        samples = run.get("samples", [])
        if not isinstance(samples, list):
            continue
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            ts = sample.get("timestamp", 0)
            metrics = sample.get("metrics", sample) if isinstance(sample, dict) else {}
            if not isinstance(metrics, dict):
                continue
            latency = metrics.get("latency_ms", 0)
            power = metrics.get("power_w", 0)
            throughput = (1000.0 / latency) if isinstance(latency, (int, float)) and latency else 0
            rows.append((ts, run_label, "default", batch_size, latency, power, throughput))
    if not rows:
        return False
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("timestamp,run_label,workload,batch_size,latency_ms,power_w,throughput\n")
        for row in rows:
            handle.write(",".join(str(value) for value in row) + "\n")
    return True


def _export_report_data(
    *,
    args: Any,
    export_data: Any,
    output_path: Callable[[Any, str], str],
    save_trackiq_wrapped_json: Callable[[str, Any, str, str], None],
    write_result_to_csv: Callable[[dict[str, Any], str], bool],
) -> None:
    """Export report data payload as wrapped JSON and best-effort CSV."""
    base = os.path.splitext(os.path.basename(args.output))[0]
    json_out = getattr(args, "export_json", None) or (base + "_data.json")
    csv_out = getattr(args, "export_csv", None) or (base + "_data.csv")
    json_out = output_path(args, json_out)
    csv_out = output_path(args, csv_out)
    save_trackiq_wrapped_json(
        json_out,
        export_data,
        workload_name="html_report_data",
        workload_type="inference",
    )
    print(f"[OK] JSON exported to: {json_out}")
    csv_written = False
    if isinstance(export_data, list):
        csv_written = _write_multi_run_csv(export_data, csv_out)
    elif isinstance(export_data, dict):
        csv_written = write_result_to_csv(export_data, csv_out)
    if csv_written:
        print(f"[OK] CSV exported to: {csv_out}")


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

    from autoperfpy.reporting import HTMLReportGenerator
    from autoperfpy.reports.plotly_report import generate_plotly_report
    from autoperfpy.reports.report_builder import (
        populate_multi_run_html_report,
        populate_standard_html_report,
    )

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
            print(f"[ERROR] {exc}", file=sys.stderr)
            return None

    try:
        report = HTMLReportGenerator(title=args.title, author=args.author, theme=args.theme)
        data_source = getattr(args, "csv", None) or getattr(args, "json", None) or "Sample Data"

        has_report_data = False
        export_data: Any | None = None

        if getattr(args, "json", None):
            with open(args.json, encoding="utf-8") as handle:
                raw_data = json.load(handle)

            if _is_plotly_report_candidate(raw_data):
                report_output = generate_plotly_report(raw_data, output_path(args, args.output))
                _export_report_data(
                    args=args,
                    export_data=raw_data,
                    output_path=output_path,
                    save_trackiq_wrapped_json=save_trackiq_wrapped_json,
                    write_result_to_csv=write_result_to_csv,
                )
                print(f"\n[OK] HTML report generated: {report_output}")
                return {"output_path": report_output}

            if not has_report_data and isinstance(raw_data, list):
                runs = [normalize_report_input_data(item) for item in raw_data]
                runs = [run for run in runs if isinstance(run, dict) and run]
                if runs:
                    has_report_data = True
                    populate_multi_run_html_report(
                        report,
                        runs,
                        data_source=data_source,
                        include_run_details=True,
                        chart_engine="plotly",
                    )
                    export_data = runs
            elif not has_report_data:
                data = normalize_report_input_data(raw_data)

                if "comparisons" in data and "summary" in data and "config" in data:
                    has_report_data = True
                    report.add_metadata("Data Source", data_source)
                    report.add_metadata("Validation Type", "Distributed Training")
                    config_data = data["config"]
                    report.add_metadata("Training Steps", str(config_data.get("num_steps", 0)))
                    report.add_metadata("Processes", str(config_data.get("num_processes", 1)))
                    report.add_metadata("Loss Tolerance", str(config_data.get("loss_tolerance", 0.01)))

                    summary = data["summary"]
                    report.add_summary_item("Total Steps", summary["total_steps"], "", "neutral")
                    report.add_summary_item(
                        "Passed Steps",
                        summary["passed_steps"],
                        "",
                        "good" if summary["passed_steps"] > 0 else "neutral",
                    )
                    report.add_summary_item(
                        "Failed Steps",
                        summary["failed_steps"],
                        "",
                        "critical" if summary["failed_steps"] > 0 else "neutral",
                    )
                    report.add_summary_item(
                        "Pass Rate",
                        f"{summary['pass_rate']:.1%}",
                        "",
                        "good" if summary["overall_pass"] else "critical",
                    )
                    status = "good" if summary["overall_pass"] else "critical"
                    report.add_summary_item("Overall Status", "PASS" if summary["overall_pass"] else "FAIL", "", status)

                    report.add_section(
                        "Step-by-Step Loss Comparison",
                        "Comparison of single-process vs multi-process losses",
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
                    has_report_data = True
                    populate_standard_html_report(
                        report,
                        data,
                        data_source=data_source,
                        chart_engine="plotly",
                    )
                    export_data = data
        elif getattr(args, "csv", None):
            has_report_data = True
            df = pd.read_csv(args.csv)
            csv_data = {
                "collector_name": os.path.basename(args.csv),
                "samples": df.to_dict("records"),
                "summary": {},
            }
            populate_standard_html_report(
                report,
                csv_data,
                data_source=data_source,
                df=df,
                chart_engine="plotly",
            )

        if export_data is not None:
            _export_report_data(
                args=args,
                export_data=export_data,
                output_path=output_path,
                save_trackiq_wrapped_json=save_trackiq_wrapped_json,
                write_result_to_csv=write_result_to_csv,
            )

        if not has_report_data:
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

    from autoperfpy.reporting import PDFReportGenerator

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
            print(f"[ERROR] {exc}", file=sys.stderr)
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
        print(f"[ERROR] {exc}", file=sys.stderr)
        return None
    finally:
        if json_path_to_cleanup and os.path.exists(json_path_to_cleanup):
            try:
                os.unlink(json_path_to_cleanup)
            except OSError:
                pass
