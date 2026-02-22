"""CLI help text tests for profile discoverability."""

import pytest

import autoperfpy.cli as autoperf_cli


def test_autoperfpy_run_help_lists_available_profiles(capsys: pytest.CaptureFixture[str]) -> None:
    """`autoperfpy run --help` should show concrete profile names in --profile help text."""
    parser = autoperf_cli.setup_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["run", "--help"])
    assert exc.value.code == 0

    out = capsys.readouterr().out
    assert "Profile to use (" in out
    assert "automotive_safety" in out
    assert "ci_smoke" in out
