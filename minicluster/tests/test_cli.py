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
