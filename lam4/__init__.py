"""lam4: clean scratch rebuild of Mena-LAM.

Rules: every file stays under 400 lines (tests/lam4/test_module_size.py); the old `lam` package is
imported only for data readers and units, never copied.
"""
from __future__ import annotations

__all__ = ["Lam4Calculator", "MenaCalculator"]


def __getattr__(name: str):
    if name in ("MenaCalculator", "Lam4Calculator"):
        from lam4.calc import Lam4Calculator, MenaCalculator
        return MenaCalculator if name == "MenaCalculator" else Lam4Calculator
    raise AttributeError(f"module 'lam4' has no attribute {name!r}")
