"""CLI version flag tests across TrackIQ tools."""

import pytest

import autoperfpy.cli as autoperf_cli
import minicluster.cli as minicluster_cli
import trackiq_compare.cli as compare_cli


def test_autoperfpy_cli_version_flag_prints_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """`autoperfpy --version` should print version and exit with code 0."""
    parser = autoperf_cli.setup_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out.strip().lower()
    assert "autoperfpy" in out


def test_minicluster_cli_version_flag_prints_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """`minicluster --version` should print version and exit with code 0."""
    parser = minicluster_cli.setup_main_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out.strip().lower()
    assert "minicluster" in out


def test_trackiq_compare_cli_version_flag_prints_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """`trackiq-compare --version` should print version and exit with code 0."""
    parser = compare_cli.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out.strip().lower()
    assert "trackiq-compare" in out
