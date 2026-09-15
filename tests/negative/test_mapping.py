"""Level N — the mapping corpus (M4). Owner: Person A (the corpus); M4 is Person B's.

The five `mapping` codes of `06-interfaces.md` §6.3 are reached the way `04-test-plan.md` §5
asks for: through **M4's own entry points**, `m4_selfcheck.self_check(plan)` and
`m4_mapping.plan(mapping)`, on the hand-corrupted plan literals B built in
`tests/fixtures/corrupt/`. Nothing here re-tests M4 — `tests/unit/test_m4_selfcheck.py` does
that, case by case, and asserts the `details` tables. What this module owes is one raiser per
code so that `test_D3_catalogue_complete` can be honest (FR-D3) and `test_D1_schema` can read
their diagnostics (FR-D1).
"""

from __future__ import annotations

import pytest

from spatial import m4_selfcheck
from spatial.model import MappingError
from tests.fixtures.corrupt import (dma_three_inbound, dropped_get, halo_reversed,
                                    iv_bundle_index, tensor_order)
from tests.negative import _corpus


def raises_balance():
    """W1 with the `C2L3` drain loop shortened: `1 put, 0 gets` on two PE columns (FR-M9)."""
    m4_selfcheck.self_check(dropped_get.plan())


def raises_channel_cycle():
    """The halo emitted `[GET, GET, PUT, PUT]`: a four-site cycle inside one timestep
    (FR-M10). The plan is still *balanced*, which is why P2b exists."""
    m4_selfcheck.self_check(halo_reversed.plan())


def raises_bundle_index_is_iv():
    """A bundle index that is an `scf.for` induction variable, which `ChannelPutOp::verify`
    rejects (`AIRDialect.cpp:3586-3593`)."""
    m4_selfcheck.self_check(iv_bundle_index.plan())


def raises_protocol_unsupported():
    """The written tensor declared before the read-only one: `air.api`'s `_check_interface`
    would raise a bare `RuntimeError` at trace time, so the self-check names it first."""
    m4_selfcheck.self_check(tensor_order.plan())


def raises_dma_channels():
    """Three circuit-switched inbound channels on one core against an AIE2 tile's two S2MM
    channels — the shape `aircc` is measured to fail on, twenty seconds in (R-18)."""
    m4_selfcheck.self_check(dma_three_inbound.plan())


def _module():
    import sys
    return sys.modules[__name__]


@pytest.mark.fr("FR-D3")
def test_mapping_codes():
    """Every corpus function raises the `MappingError` code its name spells."""
    _corpus.assert_codes(_module(), MappingError)
