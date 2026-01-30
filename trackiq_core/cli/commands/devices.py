"""Devices command for trackiq_core CLI."""

import sys
from trackiq_core.hardware import get_all_devices


def run_devices_list(args) -> int:
    """List all detected devices."""
    devices = get_all_devices()
    if not devices:
        print("No devices detected.", file=sys.stderr)
        print(
            "Run on CPU with: --device cpu_0",
            file=sys.stderr,
        )
        return 1
    print("Detected devices:")
    print("=" * 72)
    for d in devices:
        name = (d.device_name or d.device_id).strip()
        parts = [d.device_id, d.device_type, name]
        if d.gpu_model:
            parts.append(f"gpu={d.gpu_model}")
        if d.soc:
            parts.append(f"soc={d.soc}")
        if d.cpu_model:
            parts.append(f"cpu={d.cpu_model}")
        print("  " + "  ".join(parts))
    print("=" * 72)
    print(f"Total: {len(devices)} device(s)")
    return 0
