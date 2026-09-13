"""W1 GEMM on Tenstorrent's functional simulator, end to end from the `MappingPlan`.

Not in the default run. Build the environment and run it with::

    source scripts/tt_env.sh
    .venv-tt/bin/python -m pytest -m requires_ttsim tests/tt

Every test here costs a simulator execution — roughly a minute of wall time for W1's 4 cores —
so the module runs the device exactly **twice**: once for the emitted program and once for the
negative control. `design/PROGRESS-TT.md` §4 records the measured numbers.

What the three assertions establish, in order:

1. `C == A @ B` **exactly**. The inputs are `tests/integration/test_semantics.py`'s: integer
   valued `f32` in `[-8, 8)` with `K = 64`, so every partial sum is below `2**24` and `==` is
   the right comparison — a tolerance would hide a real difference.
2. A one-character mutation of the emitted kernel changes the answer. Without it, assertion 1
   would also pass against a simulator that never ran our kernel at all and left `C` holding
   something that happened to be right; it is the same argument
   `test_sem_compute_nodes_is_the_plan_not_the_kernel` makes about the interpreter.
3. The device agrees with `tests/helpers/plan_interp.py`, element for element. That is the claim
   that matters for the project: **the plan's semantics and the simulator's are the same**, on a
   backend the plan was not written for.

The target string (`"npu1"`) is an AIR target — an AIE generation for `air.api`'s `build()`. It
is irrelevant here and is passed only because `w1_legal.legal` takes one;
`test_target_does_not_reach_the_tt_program` proves the TT program does not depend on it.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pytest

from spatial import m4_mapping as m4
from spatial import m5tt_emit as m5tt
from tests.fixtures.mappings import w1_legal
from tests.helpers import plan_interp
from tests.integration.test_semantics import inputs

pytestmark = pytest.mark.requires_ttsim

_COMPUTE = "] + (a["
"""The one `+` in W1's emitted compute node, `acc[…] = (acc[…] + (a[…] * b[…]))`."""


@pytest.fixture(scope="module")
def runner():
    """`spatial.m6tt_run`, or a skip naming exactly what is missing."""
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


@pytest.fixture(scope="module")
def program():
    return m5tt.emit(m4.plan(w1_legal.legal("npu1")))


@pytest.fixture(scope="module")
def executed(runner, program):
    """The one clean simulator run this module pays for."""
    start = time.time()
    result = runner.run(program, inputs())
    print(f"\nW1 on ttsim: {time.time() - start:.1f}s wall for "
          f"{program.grid} cores", flush=True)
    return result


def test_W1_is_exact_on_ttsim(executed):
    """The whole point: `MappingPlan` -> TT-Metalium -> ttsim reproduces `A @ B` exactly."""
    given = inputs()
    assert executed["C"].dtype == np.float32
    assert np.array_equal(executed["C"], given["A"] @ given["B"])


def test_W1_leaves_its_inputs_alone(executed):
    """A kernel that wrote outside `C` would not be the plan's."""
    given = inputs()
    assert np.array_equal(executed["A"], given["A"])
    assert np.array_equal(executed["B"], given["B"])


def test_W1_matches_the_plan_interpreter(executed):
    """Plan semantics == simulator semantics, element for element."""
    interpreted = plan_interp.run(m4.plan(w1_legal.legal("npu1")), inputs())
    assert np.array_equal(executed["C"], interpreted["C"])


def test_mutating_the_kernel_changes_the_answer(runner, program, executed):
    """The negative control: turn the compute node's `+` into `-` and the simulator must return
    something else. This is what proves the device executed *our* kernel."""
    assert program.source.count(_COMPUTE) == 1, "the compute node is no longer unique"
    mutated = dataclasses.replace(
        program, source=program.source.replace(_COMPUTE, "] - (a["))
    assert mutated.source != program.source

    given = inputs()
    result = runner.run(mutated, given)
    assert not np.array_equal(result["C"], given["A"] @ given["B"])
    assert not np.array_equal(result["C"], executed["C"])
