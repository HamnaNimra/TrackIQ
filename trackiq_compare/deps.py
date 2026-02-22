"""Centralized TrackIQ-core dependencies for trackiq_compare.

This module loads only required ``trackiq_core`` source files directly so we
can reuse canonical implementations without importing ``trackiq_core`` package
initializers that may pull optional heavy dependencies.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_ROOT = Path(__file__).resolve().parents[1]


def _package_dir_for(pkg_name: str) -> Path | None:
    """Return local package directory for ``pkg_name`` when available."""
    candidate = _ROOT.joinpath(*pkg_name.split("."))
    return candidate if candidate.is_dir() else None


def _ensure_parent_packages(module_name: str) -> None:
    """Create placeholder parent packages in ``sys.modules`` as needed."""
    parts = module_name.split(".")
    for idx in range(1, len(parts)):
        pkg_name = ".".join(parts[:idx])
        package_dir = _package_dir_for(pkg_name)
        if pkg_name in sys.modules:
            # Keep existing modules, but ensure namespace path includes local source dir.
            pkg = sys.modules[pkg_name]
            if package_dir is not None and hasattr(pkg, "__path__"):
                pkg_path = list(getattr(pkg, "__path__", []))
                package_dir_str = str(package_dir)
                if package_dir_str not in pkg_path:
                    pkg_path.append(package_dir_str)
                    pkg.__path__ = pkg_path  # type: ignore[attr-defined]
            continue
        pkg = ModuleType(pkg_name)
        pkg.__path__ = [str(package_dir)] if package_dir is not None else []  # type: ignore[attr-defined]
        sys.modules[pkg_name] = pkg


def _load_core_module(module_name: str, file_path: Path) -> ModuleType:
    """Load a module from ``file_path`` under its canonical dotted name."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    _ensure_parent_packages(module_name)
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_SCHEMA = _load_core_module("trackiq_core.schema", _ROOT / "trackiq_core" / "schema.py")
_VALIDATOR = _load_core_module("trackiq_core.validator", _ROOT / "trackiq_core" / "validator.py")
_SERIALIZER = _load_core_module("trackiq_core.serializer", _ROOT / "trackiq_core" / "serializer.py")
_CONFIG_IO = _load_core_module(
    "trackiq_core.configs.config_io",
    _ROOT / "trackiq_core" / "configs" / "config_io.py",
)
_REGRESSION = _load_core_module(
    "trackiq_core.utils.compare.regression",
    _ROOT / "trackiq_core" / "utils" / "compare" / "regression.py",
)
_PDF = _load_core_module(
    "trackiq_core.reporting.pdf",
    _ROOT / "trackiq_core" / "reporting" / "pdf.py",
)

TrackiqResult = _SCHEMA.TrackiqResult
PlatformInfo = _SCHEMA.PlatformInfo
WorkloadInfo = _SCHEMA.WorkloadInfo
Metrics = _SCHEMA.Metrics
RegressionInfo = _SCHEMA.RegressionInfo

RegressionDetector = _REGRESSION.RegressionDetector
RegressionThreshold = _REGRESSION.RegressionThreshold
render_pdf_from_html_file = _PDF.render_pdf_from_html_file
PDF_BACKENDS = _PDF.PDF_BACKENDS
PdfBackendError = _PDF.PdfBackendError
save_trackiq_result = _SERIALIZER.save_trackiq_result
load_trackiq_result = _SERIALIZER.load_trackiq_result
validate_trackiq_result = _VALIDATOR.validate_trackiq_result
ensure_parent_dir = _CONFIG_IO.ensure_parent_dir


__all__ = [
    "TrackiqResult",
    "PlatformInfo",
    "WorkloadInfo",
    "Metrics",
    "RegressionInfo",
    "load_trackiq_result",
    "save_trackiq_result",
    "validate_trackiq_result",
    "RegressionDetector",
    "RegressionThreshold",
    "render_pdf_from_html_file",
    "PDF_BACKENDS",
    "PdfBackendError",
    "ensure_parent_dir",
]
