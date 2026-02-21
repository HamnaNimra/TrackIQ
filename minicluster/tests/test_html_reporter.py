"""Tests for MiniCluster HTML reporter and CLI HTML report command."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from minicluster.cli import cmd_report_html
from minicluster.reporting import MiniClusterHtmlReporter
from trackiq_core.serializer import load_trackiq_result, save_trackiq_result

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "tool_outputs" / "minicluster_real_output.json"


def _variant_result():
    result = load_trackiq_result(FIXTURE)
    variant = deepcopy(result)
    payload = variant.tool_payload if isinstance(variant.tool_payload, dict) else {}
    config = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
    config["num_processes"] = 2
    config["batch_size"] = 32
    config["learning_rate"] = 0.005
    payload["config"] = config
    payload["num_workers"] = 2

    if isinstance(payload.get("steps"), list):
        for item in payload["steps"]:
            if not isinstance(item, dict):
                continue
            item["loss"] = float(item.get("loss", 0.0)) * 0.85
            item["throughput_samples_per_sec"] = float(item.get("throughput_samples_per_sec", 0.0)) * 1.35
    payload["final_loss"] = float(payload.get("final_loss", 0.0)) * 0.85
    payload["total_time_sec"] = float(payload.get("total_time_sec", 0.0)) * 0.75
    variant.tool_payload = payload
    variant.workload.batch_size = 32
    variant.metrics.throughput_samples_per_sec *= 1.35
    if variant.metrics.power_consumption_watts is not None:
        variant.metrics.power_consumption_watts *= 1.10
    return variant


def test_minicluster_html_reporter_single_run_contains_training_graphs(tmp_path) -> None:
    """Single result HTML should include training graph sections."""
    result = load_trackiq_result(FIXTURE)
    out = tmp_path / "single_report.html"
    MiniClusterHtmlReporter().generate(str(out), [result], title="MiniCluster Single Report")
    html = out.read_text(encoding="utf-8")

    assert "MiniCluster HTML Report" in html
    assert "Training Graphs" in html
    assert "Loss by Step" in html
    assert "Throughput by Step" in html
    assert "Per-Step Timing (Compute + AllReduce)" in html


def test_minicluster_html_reporter_multi_run_contains_consolidated_graphs(tmp_path) -> None:
    """Multiple results should render consolidated graph and pie sections."""
    base = load_trackiq_result(FIXTURE)
    variant = _variant_result()
    out = tmp_path / "consolidated_report.html"
    MiniClusterHtmlReporter().generate(str(out), [base, variant], title="MiniCluster Consolidated Report")
    html = out.read_text(encoding="utf-8")

    assert "Consolidated Graphs" in html
    assert "Configuration Comparison" in html
    assert "Loss Curves Overlay" in html
    assert "Key-Metric Winner Share" in html
    assert "conic-gradient(" in html
    assert "w2-b32-s10-lr0.005" in html


def test_cmd_report_html_writes_consolidated_report(tmp_path) -> None:
    """CLI HTML report command should support multiple result inputs."""
    base = load_trackiq_result(FIXTURE)
    variant = _variant_result()
    result_a = tmp_path / "result_a.json"
    result_b = tmp_path / "result_b.json"
    save_trackiq_result(base, result_a)
    save_trackiq_result(variant, result_b)

    output = tmp_path / "report.html"
    args = argparse.Namespace(
        result=[str(result_a), str(result_b)],
        output=str(output),
        title="CLI Consolidated MiniCluster Report",
    )
    cmd_report_html(args)

    assert output.exists()
    html = output.read_text(encoding="utf-8")
    assert "Consolidated Graphs" in html
    assert "CLI Consolidated MiniCluster Report" in html
