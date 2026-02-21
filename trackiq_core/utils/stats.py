"""Shared statistics utilities (percentile, stats from numeric lists)."""


def percentile(values: list[float], p: float) -> float:
    """Calculate percentile of a list of values (linear interpolation).

    Args:
        values: List of numeric values
        p: Percentile to calculate (0-100)

    Returns:
        Percentile value, or 0.0 if values is empty
    """
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_values) else f
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


def stats_from_values(values: list[float]) -> dict[str, float | None]:
    """Calculate mean, min, max for a list of values (handles None/empty).

    Args:
        values: List of numeric values (None entries are skipped)

    Returns:
        Dict with keys mean, min, max; or empty dict if no valid values
    """
    valid = [v for v in values if v is not None]
    if not valid:
        return {}
    n = len(valid)
    return {
        "mean": sum(valid) / n,
        "min": min(valid),
        "max": max(valid),
    }
