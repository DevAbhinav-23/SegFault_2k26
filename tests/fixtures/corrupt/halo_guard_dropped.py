"""The synthetic halo with the north ghost get's guard dropped — the **per-branch** rule.

`halo.plan(PI=2, north_guard=False)`. `ToNorth`'s get is indexed `tx` and guarded `tx < PI-1`:
the guard is what keeps the index inside `[0, PI-1)`. Without it PE 1 gets at `ToNorth[1]`, an
index of a `size=(1,)` bundle that no put can ever reach, so the key shows `0 puts, 1 get` and
`Diagnostic.details` names the coordinate `[1]` it came from.
"""

from __future__ import annotations

from spatial.model import MappingPlan

from tests.fixtures.corrupt import halo


def plan() -> MappingPlan:
    """The 2-PE halo with an unguarded north ghost get (`BALANCE`, per-branch, `0 != 1`)."""
    return halo.plan(PI=2, north_guard=False)


__all__ = ["plan"]
