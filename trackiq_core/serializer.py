"""Serialization helpers for canonical TrackIQ results."""

import json
from pathlib import Path
from typing import Union

from trackiq_core.schema import TrackiqResult
from trackiq_core.validator import validate_trackiq_result, validate_trackiq_result_obj


PathLike = Union[str, Path]


def save_trackiq_result(result: TrackiqResult, path: PathLike) -> None:
    """Save a TrackiqResult to JSON."""
    validate_trackiq_result_obj(result)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2)


def load_trackiq_result(path: PathLike) -> TrackiqResult:
    """Load a TrackiqResult from JSON."""
    in_path = Path(path)
    with in_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_trackiq_result(payload)
    return TrackiqResult.from_dict(payload)
