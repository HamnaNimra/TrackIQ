"""Convenience launcher for the unified TrackIQ Streamlit dashboard.

This wrapper avoids Streamlit argument-separator confusion by always invoking:
    python -m streamlit run dashboard.py -- <dashboard args>
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch TrackIQ unified dashboard via Streamlit."
    )
    parser.add_argument(
        "--tool",
        required=True,
        choices=["autoperfpy", "minicluster", "compare"],
        help="Tool dashboard to launch",
    )
    parser.add_argument("--result", help="Single TrackiqResult JSON path")
    parser.add_argument("--result-a", help="Compare mode: result A path")
    parser.add_argument("--result-b", help="Compare mode: result B path")
    parser.add_argument("--label-a", help="Compare mode: display label A")
    parser.add_argument("--label-b", help="Compare mode: display label B")
    return parser


def _build_streamlit_command(args: argparse.Namespace) -> List[str]:
    dashboard_path = Path(__file__).resolve().parent / "dashboard.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path),
        "--",
        "--tool",
        args.tool,
    ]
    if args.result:
        cmd.extend(["--result", args.result])
    if args.result_a:
        cmd.extend(["--result-a", args.result_a])
    if args.result_b:
        cmd.extend(["--result-b", args.result_b])
    if args.label_a:
        cmd.extend(["--label-a", args.label_a])
    if args.label_b:
        cmd.extend(["--label-b", args.label_b])
    return cmd


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    command = _build_streamlit_command(args)
    return int(subprocess.call(command))


if __name__ == "__main__":
    raise SystemExit(main())
