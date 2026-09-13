"""Spatial DSL for NPUs, lowering to MLIR-AIR.

A `@sp.kernel` plain-Python loop nest plus an ignorable `sp.schedule`: delete the schedule and
the program still runs in CPython and *is* the specification. See design/00-README.md §6.

As of 2026-09-13: `model` (M0), `m4_mapping`, `m4_selfcheck`, `m5_emit` and the off-device half
of `m6_tools` are built; `m1_frontend`, `m2_schedule` and `m3_legality` are still stubs, so the
surface of design/06-interfaces.md §7.1 (`kernel`, `schedule`, `Schedule`, `ax`) is **not** yet
re-exported from this package. See design/PROGRESS-B.md §P7.
"""

from spatial.model import CONTRACT_VERSION

__all__ = ["CONTRACT_VERSION"]
