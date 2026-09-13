"""A 2-PE halo with three L3-staged inbound channels — over budget, but a **warning**.

`halo.plan(PI=2, staging=("UIn", "QIn", "RIn"))`. Each PE names four inbound channels, which is
over the two-S2MM budget, but three of them are L3→L1: the per-column shim pressure is 3, above
`shim-dma-channels-per-col`, so `air-dma-to-channel` upgrades all three to
`channel_type = "npu_dma_packet"` and they multiplex onto one shim DMA channel. Only the halo get
is circuit-switched, so `self_check` **does not raise** and `m4_selfcheck.warnings` records the
uncertainty by name. This is W3's shape in miniature and the reason §3.8 splits error from
warning: a checker that rejected this would reject a program `aircc` accepts (P-R4).

**Substitution, recorded per the brief.** The brief asks for "three inbound of which two are
packet-capable". Packet capability is not per channel: the upstream rule upgrades **every**
L3-attached channel of a direction at once, or none (`AIRDmaToChannel.cpp:1727-1740`), so a
two-of-three split is unreachable. The nearest constructible shape with the same meaning — over
budget by named channels, within budget by circuit-switched ones — is three packet-capable plus
one circuit.
"""

from __future__ import annotations

from spatial.model import MappingPlan

from tests.fixtures.corrupt import halo


def plan() -> MappingPlan:
    """The 2-PE halo with `UIn`, `QIn`, `RIn` staged from L3 (a warning, no raise)."""
    return halo.plan(PI=2, staging=("UIn", "QIn", "RIn"))


__all__ = ["plan"]
