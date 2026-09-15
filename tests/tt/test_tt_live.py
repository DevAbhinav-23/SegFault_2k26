"""The **live surface** on ttsim: `kernels/w3_sw.py` → Tensix, checked against CPython.

Not in the default run. Build the environment and run it with::

    source scripts/tt_env.sh
    .venv-tt/bin/python -m pytest -m requires_ttsim tests/tt

What this adds to `test_tt_w3.py`, which executes the same workload: both ends are the user's.
That module starts from the hand-written `LegalMapping` literal of `tests/fixtures/mappings/`
and compares against `test_semantics.smith_waterman`, a textbook DP written for the purpose.
This one starts at `kernels.w3_sw.schedule("npu1")` — `@sp.kernel`, clauses, `.plan()`, the
whole surface a user touches — and compares against **`kernels.w3_sw.sw` run in CPython**, the
kernel exactly as it is written. Delete every `s.*` clause and that function is the
specification (FR-S18); this asserts that four Tensix cores reproduce it element for element.

`tests/integration/test_kernels_live.py::test_live_plan_emits_the_same_tt_program` is the
device-free half of the same claim — that the live plan and the fixture literal emit the *same*
`TTProgram` — and it runs in the default suite. Together they are what puts the second backend
on the live path rather than beside it, and `demo/run_demo.py`'s 4:05 beat runs this same pair
of lines on stage.

One simulator run, ≈2 s (`design/PROGRESS-TT.md` §Status). The negative controls for W3 are
`test_tt_w3.py`'s and are not repeated.
"""

from __future__ import annotations

import numpy as np
import pytest

from kernels import w3_sw
from spatial import m5tt_emit as m5tt
from tests.integration.test_semantics import w3_inputs

pytestmark = pytest.mark.requires_ttsim


@pytest.fixture(scope="module")
def runner():
    """`spatial.m6tt_run`, or a skip naming exactly what is missing (FR-TT12)."""
    from spatial import m6tt_run
    try:
        m6tt_run._check_environment()
    except m6tt_run.TTRunError as exc:
        pytest.skip(str(exc))
    try:
        import ttnn                                              # noqa: F401
    except ImportError as exc:                                   # pragma: no cover - env gate
        pytest.skip(f"ttnn is not importable ({exc}); run this under .venv-tt")
    return m6tt_run


@pytest.mark.fr("FR-TT6")
def test_the_live_W3_schedule_is_exact_on_ttsim(runner):
    """`kernels.w3_sw.schedule("npu1").plan()` → TT-Metalium → ttsim == the kernel in CPython.

    Exactly, not to a tolerance: the scores are `i32` and every intermediate is an integer.
    The oracle is called on a **fresh** zeroed matrix, because `sw` writes through its third
    argument and the one uploaded to the device must stay the device's.
    """
    program = m5tt.emit(w3_sw.schedule("npu1").plan())
    given = w3_inputs()
    executed = runner.run(program, given)

    expected = np.zeros_like(given["S"])
    w3_sw.sw(given["q"], given["r"], expected)

    assert executed["S"].dtype == np.int32
    assert np.array_equal(executed["S"], expected)
    assert expected[1:, 1:].any(), "an all-zero score matrix would pass vacuously"
