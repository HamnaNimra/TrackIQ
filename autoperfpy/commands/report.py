"""Report command handlers for AutoPerfPy CLI."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from autoperfpy.reports.export_csv import write_multi_run_csv as _write_multi_run_samples_csv
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
    return _write_multi_run_samples_csv(runs, path)


def _bootstrap_report_input(
    args: Any,
    *,
    run_default_benchmark: Callable[[str | None, int], tuple[dict, str | None, str | None]],
) -> str | None:
    """Ensure --json/--csv exists, optionally generating a quick benchmark JSON."""
    if getattr(args, "csv", None) or getattr(args, "json", None):
        return None
    print("No --csv/--json provided; running a quick benchmark to generate data...")
    _, _, json_path = run_default_benchmark(
        getattr(args, "device", None),
        getattr(args, "duration", 10),
    )
    if json_path:
        args.json = json_path
        return json_path
    return None


def _cleanup_generated_json(path: str | None) -> None:
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


def _append_report_context(
    export_data: Any,
    *,
    report_kind: str,
    data_source: str,
) -> Any:
    """Attach export context to payload without mutating caller-owned data."""
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    context = {
        "report_context": {
            "format": report_kind,
            "data_source": data_source,
            "generated_at_utc": generated_at_utc,
            "purpose": (
                "Human-readable performance artifact. Use HTML/PDF for reviews, "
                "JSON/CSV for automation and longitudinal analysis."
            ),
            "consumer_guidance": [
                "Check p99 latency and throughput together before making go/no-go decisions.",
                "Use CSV/JSON exports for regression automation and longitudinal trend analysis.",
            ],
        }
    }

    if isinstance(export_data, dict):
        payload = deepcopy(export_data)
        payload.update(context)
        return payload
    if isinstance(export_data, list):
        payload_list: list[Any] = []
        for item in export_data:
            if isinstance(item, dict):
                enriched = deepcopy(item)
                enriched.update(context)
                payload_list.append(enriched)
            else:
                payload_list.append(item)
        return payload_list
    return export_data


def _export_report_data(
    *,
    args: Any,
    export_data: Any,
    output_path: Callable[[Any, str], str],
    save_trackiq_wrapped_json: Callable[[str, Any, str, str], None],
    write_result_to_csv: Callable[[dict[str, Any], str], bool],
    report_kind: str,
    data_source: str,
) -> None:
    """Export report data payload as wrapped JSON and best-effort CSV."""
    base = os.path.splitext(os.path.basename(args.output))[0]
    json_out = getattr(args, "export_json", None) or (base + "_data.json")
    csv_out = getattr(args, "export_csv", None) or (base + "_data.csv")
    json_out = output_path(args, json_out)
    csv_out = output_path(args, csv_out)
    enriched_export_data = _append_report_context(export_data, report_kind=report_kind, data_source=data_source)
    save_trackiq_wrapped_json(
        json_out,
        enriched_export_data,
        workload_name=f"{report_kind}_report_data",
        workload_type="inference",
    )
    print(f"[OK] JSON exported to: {json_out}")
    csv_written = False
    if isinstance(enriched_export_data, list):
        csv_written = _write_multi_run_csv(enriched_export_data, csv_out)
    elif isinstance(enriched_export_data, dict):
        csv_written = write_result_to_csv(enriched_export_data, csv_out)
    if csv_written:
        print(f"[OK] CSV exported to: {csv_out}")


def _is_distributed_validation_payload(data: dict[str, Any]) -> bool:
    """Return True if payload matches distributed validation schema."""
    return "comparisons" in data and "summary" in data and "config" in data


def _add_distributed_validation_html(report: Any, data: dict[str, Any], data_source: str) -> None:
    """Render distributed validation payload into HTML report sections."""
    report.add_metadata("Data Source", data_source)
    report.add_metadata("Validation Type", "Distributed Training")
    config_data = data["config"]
    report.add_metadata("Training Steps", str(config_data.get("num_steps", 0)))
    report.add_metadata("Processes", str(config_data.get("num_processes", 1)))
    report.add_metadata("Loss Tolerance", str(config_data.get("loss_tolerance", 0.01)))

    summary = data["summary"]
    total_steps = summary.get("total_steps", 0)
    passed_steps = summary.get("passed_steps", 0)
    failed_steps = summary.get("failed_steps", 0)
    pass_rate = summary.get("pass_rate", 0.0)
    overall_pass = bool(summary.get("overall_pass", False))
    report.add_summary_item("Total Steps", total_steps, "", "neutral")
    report.add_summary_item("Passed Steps", passed_steps, "", "good" if passed_steps > 0 else "neutral")
    report.add_summary_item("Failed Steps", failed_steps, "", "critical" if failed_steps > 0 else "neutral")
    report.add_summary_item("Pass Rate", f"{pass_rate:.1%}", "", "good" if overall_pass else "critical")
    report.add_summary_item(
        "Overall Status", "PASS" if overall_pass else "FAIL", "", "good" if overall_pass else "critical"
    )

    report.add_section(
        "Step-by-Step Loss Comparison",
        "Comparison of single-process vs multi-process losses.",
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
        table_data.append(
            [
                str(step),
                f"{single_loss:.6f}",
                f"{multi_loss:.6f}",
                f"{delta:.6f}",
                f"{rel_delta:.4f}",
                "PASS" if passed else "FAIL",
            ]
        )
    report.add_table(
        "Loss Comparison",
        ["Step", "Single Loss", "Multi Loss", "Abs Delta", "Rel Delta", "Status"],
        table_data,
        "Step-by-Step Loss Comparison",
    )

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


def _add_multi_run_pdf_summary(report: Any, runs: list[dict[str, Any]]) -> None:
    """Add concise multi-run comparison tables/summary to PDF reports."""
    report.add_metadata("Report Type", "Multi-Run Comparison")
    report.add_metadata("Run Count", str(len(runs)))

    rows: list[list[str]] = []
    best_throughput = ("N/A", float("-inf"))
    best_p99 = ("N/A", float("inf"))
    for idx, run in enumerate(runs):
        label = str(run.get("run_label") or run.get("collector_name") or f"run_{idx + 1}")
        platform = run.get("platform_metadata")
        inf_cfg = run.get("inference_config")
        summary = run.get("summary")
        platform = platform if isinstance(platform, dict) else {}
        inf_cfg = inf_cfg if isinstance(inf_cfg, dict) else {}
        summary = summary if isinstance(summary, dict) else {}
        latency = summary.get("latency") if isinstance(summary.get("latency"), dict) else {}
        throughput = summary.get("throughput") if isinstance(summary.get("throughput"), dict) else {}
        power = summary.get("power") if isinstance(summary.get("power"), dict) else {}

        p99 = latency.get("p99_ms")
        mean_fps = throughput.get("mean_fps")
        mean_power = power.get("mean_w")

        if isinstance(mean_fps, (int, float)) and float(mean_fps) > best_throughput[1]:
            best_throughput = (label, float(mean_fps))
        if isinstance(p99, (int, float)) and float(p99) < best_p99[1]:
            best_p99 = (label, float(p99))

        rows.append(
            [
                label,
                str(platform.get("device_name") or inf_cfg.get("accelerator") or "N/A"),
                str(inf_cfg.get("precision") or "N/A"),
                str(inf_cfg.get("batch_size") or "N/A"),
                str(summary.get("sample_count") or 0),
                f"{float(p99):.2f}" if isinstance(p99, (int, float)) else "N/A",
                f"{float(mean_fps):.2f}" if isinstance(mean_fps, (int, float)) else "N/A",
                f"{float(mean_power):.2f}" if isinstance(mean_power, (int, float)) else "N/A",
            ]
        )

    report.add_summary_item("Best Throughput Run", best_throughput[0], "", "good")
    report.add_summary_item("Lowest P99 Run", best_p99[0], "", "good")
    report.add_table(
        "Run Overview",
        ["Run", "Device", "Precision", "Batch", "Samples", "P99 (ms)", "Throughput (FPS)", "Power (W)"],
        rows,
        section="Run Comparison",
    )


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

    try:
        json_path_to_cleanup = _bootstrap_report_input(args, run_default_benchmark=run_default_benchmark)
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
                    report_kind="html",
                    data_source=data_source,
                )
                print(f"\n[OK] HTML report generated: {report_output}")
                return {"output_path": report_output}

            if isinstance(raw_data, list):
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
            else:
                data = normalize_report_input_data(raw_data)

                if _is_distributed_validation_payload(data):
                    has_report_data = True
                    _add_distributed_validation_html(report, data, data_source)
                    export_data = data
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
                report_kind="html",
                data_source=data_source,
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
        _cleanup_generated_json(json_path_to_cleanup)


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

    try:
        json_path_to_cleanup = _bootstrap_report_input(args, run_default_benchmark=run_default_benchmark)
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
                raw_data = json.load(handle)

            export_data: Any | None = None
            if isinstance(raw_data, list):
                runs = [normalize_report_input_data(item) for item in raw_data]
                runs = [run for run in runs if isinstance(run, dict) and run]
                if runs:
                    export_data = runs
                    _add_multi_run_pdf_summary(report, runs)
            else:
                data = normalize_report_input_data(raw_data)
                export_data = data
                if _is_distributed_validation_payload(data):
                    summary = data.get("summary", {})
                    report.add_metadata("Validation Type", "Distributed Training")
                    report.add_metadata("Training Steps", str(data.get("config", {}).get("num_steps", 0)))
                    report.add_metadata("Processes", str(data.get("config", {}).get("num_processes", 1)))
                    report.add_summary_item("Pass Rate", f"{summary.get('pass_rate', 0.0):.1%}", "", "neutral")
                    report.add_summary_item(
                        "Overall Status",
                        "PASS" if summary.get("overall_pass", False) else "FAIL",
                        "",
                        "good" if summary.get("overall_pass", False) else "critical",
                    )
                else:
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
                        report.add_summary_item(
                            "Mean Throughput",
                            f"{throughput.get('mean_fps', 0):.1f}",
                            "FPS",
                            "neutral",
                        )
                    power = summary.get("power", {})
                    if power and power.get("mean_w") is not None:
                        report.add_summary_item("Mean Power", f"{power.get('mean_w', 0):.1f}", "W", "neutral")

                    if samples:
                        report.add_charts_from_data(samples, summary)

            if export_data is not None:
                _export_report_data(
                    args=args,
                    export_data=export_data,
                    output_path=output_path,
                    save_trackiq_wrapped_json=save_trackiq_wrapped_json,
                    write_result_to_csv=write_result_to_csv,
                    report_kind="pdf",
                    data_source=data_source,
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
        _cleanup_generated_json(json_path_to_cleanup)
