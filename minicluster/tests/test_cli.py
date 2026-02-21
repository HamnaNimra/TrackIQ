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
