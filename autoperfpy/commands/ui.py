"""UI command handler for AutoPerfPy CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any


def run_ui(args: Any, *, cli_file: str) -> int:
    """Launch Streamlit dashboard."""
    ui_module = Path(cli_file).parent / "ui" / "streamlit_app.py"

    if not ui_module.exists():
        print(f"[ERROR] Streamlit app not found at {ui_module}", file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ui_module),
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
    ]

    if args.no_browser:
        cmd.extend(["--server.headless", "true"])

    if args.data:
        cmd.extend(["--", "--data", args.data])

    print("Launching AutoPerfPy Dashboard...")
    print(f"URL: http://{args.host}:{args.port}")
    if args.data:
        print(f"Data file: {args.data}")
    print("\nPress Ctrl+C to stop the server\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped")
    except subprocess.CalledProcessError as exc:
        print(f"Error launching Streamlit: {exc}", file=sys.stderr)
        print("\nMake sure Streamlit is installed: pip install streamlit plotly pandas")
        return 1
    except FileNotFoundError:
        print(
            "[ERROR] Streamlit not found. Install with: pip install streamlit",
            file=sys.stderr,
        )
        return 1

    return 0

