"""CLI contract tests for minicluster argument handling."""

from __future__ import annotations

import sys

import pytest

import minicluster.cli as minicluster_cli


def test_report_without_subcommand_exits_nonzero(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """`minicluster report` without report subcommand should fail with exit code 1."""
    monkeypatch.setattr(sys, "argv", ["minicluster", "report"])
    with pytest.raises(SystemExit) as exc:
        minicluster_cli.main()
    assert exc.value.code == 1
    output = capsys.readouterr().out.lower()
    assert "usage: minicluster report" in output


def test_run_parser_accepts_backend_workload_and_baseline() -> None:
    """`minicluster run` parser should accept new cluster-health options."""
    parser = minicluster_cli.setup_main_parser()
    args = parser.parse_args(
        [
            "run",
            "--workers",
            "2",
            "--backend",
            "gloo",
            "--workload",
            "mlp",
            "--baseline-throughput",
            "42.0",
        ]
    )
    assert args.command == "run"
    assert args.backend == "gloo"
    assert args.workload == "mlp"
    assert args.baseline_throughput == pytest.approx(42.0)


def test_bench_collective_parser_accepts_expected_args() -> None:
    """`minicluster bench-collective` parser should accept benchmark arguments."""
    parser = minicluster_cli.setup_main_parser()
    args = parser.parse_args(
        [
            "bench-collective",
            "--workers",
            "4",
            "--size-mb",
            "64",
            "--iterations",
            "10",
            "--backend",
            "gloo",
            "--output",
            "bench.json",
        ]
    )
    assert args.command == "bench-collective"
    assert args.workers == 4
    assert args.size_mb == pytest.approx(64.0)
    assert args.iterations == 10
    assert args.backend == "gloo"
    assert args.output == "bench.json"


def test_report_heatmap_parser_accepts_expected_args() -> None:
    """`minicluster report heatmap` parser should accept expected arguments."""
    parser = minicluster_cli.setup_main_parser()
    args = parser.parse_args(
        [
            "report",
            "heatmap",
            "--results-dir",
            "worker_results",
            "--metric",
            "p99_allreduce_ms",
            "--output",
            "heatmap.html",
        ]
    )
    assert args.command == "report"
    assert args.report_cmd == "heatmap"
    assert args.results_dir == "worker_results"
    assert args.metric == "p99_allreduce_ms"
    assert args.output == "heatmap.html"


def test_report_fault_timeline_parser_accepts_expected_args() -> None:
    """`minicluster report fault-timeline` parser should accept expected arguments."""
    parser = minicluster_cli.setup_main_parser()
    args = parser.parse_args(
        [
            "report",
            "fault-timeline",
            "--json",
            "fault_report.json",
            "--output",
            "fault_timeline.html",
        ]
    )
    assert args.command == "report"
    assert args.report_cmd == "fault-timeline"
    assert args.json == "fault_report.json"
    assert args.output == "fault_timeline.html"
