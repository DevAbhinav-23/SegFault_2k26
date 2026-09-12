"""M5 — AIR emitter. Owner: Person B. LLD: design/03-lld-M5-emitter.md.

Entry point per design/06-interfaces.md §7.2. `air` is imported lazily, inside the function, so
the surface, the checker and the mapper stay usable on a machine with no toolchain (FR-S20).
"""

from __future__ import annotations

_NOT_BUILT = "m5_emit: not built yet; see design/03-lld-M5-emitter.md"


def emit(plan: MappingPlan, target: Target) -> EmitResult:  # noqa: F821
    """Translate the plan mechanically into air.api. Raises EmissionError."""
    raise NotImplementedError(_NOT_BUILT)
