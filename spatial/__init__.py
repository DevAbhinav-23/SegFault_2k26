"""Spatial DSL for NPUs, lowering to MLIR-AIR.

A `@sp.kernel` plain-Python loop nest plus an ignorable `sp.schedule`: delete the schedule and
the program still runs in CPython and *is* the specification. See design/00-README.md §6.

Scaffold only — no module here is implemented yet. The surface of design/06-interfaces.md §7.1
(`kernel`, `schedule`, `Schedule`, `ax`) is re-exported from this package in the next phase.
"""

from spatial.model import CONTRACT_VERSION

__all__ = ["CONTRACT_VERSION"]
