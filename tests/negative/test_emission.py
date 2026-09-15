"""Level N — the emission corpus (M5). Owner: Person A (the corpus); M5 is Person B's.

Both `emission` codes of `06-interfaces.md` §6.3, through M5's one public entry point,
`m5_emit.emit(plan, target)`, on B's W1 plan fixture. `tests/unit/test_m5_emit.py` asserts what
these messages carry; what this module owes is a raiser per code (FR-D3) whose diagnostic
`test_D1_schema` can read (FR-D1).

Neither case needs a device or a toolchain binary: `EMIT-AIR-API` is refused before `air` is
imported at all, and `EMIT-VERIFY` patches `air.api`'s own `LaunchContext.build` — the wheel is
importable with no NPU present (VF §D.9).
"""

from __future__ import annotations

from unittest import mock

import pytest

from spatial import m5_emit
from spatial.model import EmissionError
from tests.fixtures.plans import w1_plan
from tests.negative import _corpus

_VERIFY_FAILURE = ("air.api emitted invalid IR -- this is a bug in the DSL, not in the "
                   "kernel:\n'affine.apply' op using value defined outside the region")
"""The marker text `LaunchContext._verify` raises on (`python/air/api/_compile.py:174-180`).
That marker, and only that marker, is `EMIT-VERIFY`; every other exception out of `air.api` is
`EMIT-AIR-API` with the original text kept verbatim."""


def raises_emit_air_api():
    """`"auto"` never reaches the emitter: resolving it shells out to `xrt-smi`, and decision
    D-13 keeps that in M6 (`_trace.py:164-183`)."""
    m5_emit.emit(w1_plan.plan("npu1"), "auto")


def raises_emit_verify():
    """A `module.operation.verify()` failure is classified, not leaked as a `RuntimeError`.

    No plan we can build reaches this path — M5 checks its own preconditions first and every
    W1-shaped module verifies — so `air.api`'s own failure is raised in its place. What is
    under test is M5's classification of the marker, which is the only signal the wheel gives.
    """
    from air.api._compile import LaunchContext

    def broken(self, target=None):
        raise RuntimeError(_VERIFY_FAILURE)

    with mock.patch.object(LaunchContext, "build", broken):
        m5_emit.emit(w1_plan.plan("npu1"), "npu1")


def _module():
    import sys
    return sys.modules[__name__]


@pytest.mark.fr("FR-D3")
def test_emission_codes():
    """Every corpus function raises the `EmissionError` code its name spells."""
    _corpus.assert_codes(_module(), EmissionError)
