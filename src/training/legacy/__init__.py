"""Legacy training modules (deprecated).

This module intentionally avoids importing heavy legacy trainers at import
time to prevent deprecation warnings and import-side effects during test
collection. Accessing the legacy symbol will import it on demand and emit a
DeprecationWarning.
"""

from typing import Any
import importlib
import warnings

__all__ = ["LayerwiseTrainer"]


def __getattr__(name: str) -> Any:
    """Lazily import legacy symbols on attribute access.

    This defers importing the actual legacy trainer until it's referenced,
    avoiding import-time deprecation warnings during test collection.
    """
    if name == "LayerwiseTrainer":
        warnings.warn(
            "LayerwiseTrainer is deprecated. Use UnifiedTrainer from src.training.core instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        mod = importlib.import_module(".layerwise_trainer", __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
