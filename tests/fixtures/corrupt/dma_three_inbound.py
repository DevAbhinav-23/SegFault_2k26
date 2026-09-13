"""A 3-PE halo whose interior PE needs three circuit-switched inbound channels — `DMA-CHANNELS`.

`halo.plan(PI=3, staging=("UIn",))`: every PE stages its strip from L3 and the interior PE 1 also
gets both ghost rows, so its inbound set is `{UIn, ToNorth, ToSouth}` — three against an AIE2 core
tile's two S2MM channels. `UIn` is the only L3-attached inbound channel, so the per-column shim
pressure is 1 and nothing is upgraded to a packet flow: all three are circuit-switched
(`03-lld-M4-mapping.md` §3.8, `spatial.m4_selfcheck.may_packet`).

This is finding N-1, and it is measured: at `PI = 4` the same shape makes `aircc` fail with
`'aie.connect' op … TileID(1, 2) targets same dst` about twenty seconds in, naming a physical tile
the user has never heard of (REVIEW-round1 P-R2, regenerated at P3). `plan(PI)` takes the extent
so the `PI = 4` row of `03-lld-M4-mapping.md` §7 can use the same fixture.
"""

from __future__ import annotations

from spatial.model import MappingPlan

from tests.fixtures.corrupt import halo


def plan(PI: int = 3) -> MappingPlan:
    """The halo at `PI` PEs with L3 staging (`DMA-CHANNELS`, inbound, budget 2)."""
    return halo.plan(PI=PI, staging=("UIn",))


__all__ = ["plan"]
