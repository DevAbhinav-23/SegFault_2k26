"""W2 Jacobi on Tenstorrent's functional simulator: the bidirectional halo, and f32 exactness.

Not in the default run. Build the environment and run it with::

    source scripts/tt_env.sh
    .venv-tt/bin/python -m pytest -m requires_ttsim tests/tt

What this module establishes that `test_tt_w3.py` and `test_tt_flip.py` could not:

1. **Two links per interior PE, in opposite directions.** W3 and the cascade are chains: every
   core consumes from one side and produces to the other, and the dependency graph is a path.
   W2's two PEs each *put* before they *get*, on two links that run the other way — the first
   plan whose link graph has a cycle in it, and the first that can deadlock.
2. **Occurrence pairing, ruling R-TT-B.** The same link carries two put occurrences and two get
   occurrences per `t` trip, and the halo row lands in `cur` on one and in `next` on the other.
   The k-th put is consumed by the k-th get, so the destination of a remote write is read off
   the get it is paired with — not off "the channel's landing site", which does not exist here.
3. **Q-TT4 — f32 exactness under soft-float.** W1, W3 and the cascade are integer-valued or
   integer; W2 multiplies by `0.2` sixteen times per element and is the only workload that can
   answer whether a Tensix data-movement core's scalar `float` reproduces numpy bit for bit.

Three simulator runs, each costing wall time, plus one `slow` run that is *expected to hang*:

* `T = 4` — the even case, two `STEP`s per `t` trip;
* `T = 5` — the odd-`T` peel, a fifth `STEP` after the loop;
* the negative control, one perturbed stencil constant;
* `slow`: the same program with one credit per link, which deadlocks (§R-TT-B's credit rule).
"""

from __future__ import annotations

import dataclasses
import pathlib
import subprocess
import sys
import time

import numpy as np
import pytest

from spatial import m4_mapping as m4
from spatial import m5tt_emit as m5tt
from tests.fixtures.mappings import w2_legal
from tests.helpers import plan_interp
from tests.integration.test_semantics import H, WIDTH, jacobi, w2_inputs

pytestmark = pytest.mark.requires_ttsim

_CONSTANT = "((float)(0.2))"
"""W2's one f32 literal, as `design/08-tt-backend.md` §3.7 emits it: **narrowed to `float`
before it is used**. That narrowing is load-bearing and was measured — with the bare `0.2` the
multiply happens in `double` and the answer is no longer bit-exact (§T3 of
`design/PROGRESS-TT.md`: 358 of 896 interior elements off by one ulp)."""

_CREDIT = (("(ToNorth_put_n < 1 ? 0 : ToNorth_put_n - 1)", "ToNorth_put_n"),
           ("(ToSouth_put_n < 1 ? 0 : ToSouth_put_n - 1)", "ToSouth_put_n"))
"""The two credit expressions of R-TT-B, each beside the bare counter that replaces it. The bare
counter is the depth-1 FIFO of T2, which the `slow` test below shows deadlocks."""

_DEADLOCK = """\
import sys
sys.path.insert(0, {root!r})
from dataclasses import replace
from spatial import m4_mapping as m4, m5tt_emit as m5tt, m6tt_run as m6tt
from tests.fixtures.mappings import w2_legal
from tests.integration.test_semantics import w2_inputs

program = m5tt.emit(m4.plan(w2_legal.legal("npu1", T=4)))
source = program.source
for credit, counter in {pairs!r}:
    assert credit in source
    source = source.replace(credit, counter)
m6tt.run(replace(program, source=source), w2_inputs(4))
print("COMPLETED")
"""


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


def _program(T: int) -> m5tt.TTProgram:
    return m5tt.emit(m4.plan(w2_legal.legal("npu1", T=T)))


@pytest.fixture(scope="module")
def executed(runner):
    """The two clean simulator runs this module pays for, one per parity of `T`."""
    results = {}
    for T in (4, 5):
        program = _program(T)
        start = time.time()
        results[T] = runner.run(program, w2_inputs(T))
        print(f"\nW2 T={T} on ttsim: {time.time() - start:.1f}s wall for {program.grid} cores, "
              f"{len(program.semaphores)} semaphores", flush=True)
    return results


# -- the program, before any device ------------------------------------------

@pytest.mark.parametrize("T", (4, 5))
def test_W2_allocates_four_semaphores_over_two_links(T):
    """Two channels of `size=(1,)`, one link each, `full` on the consumer and `empty` on the
    producer: `design/08-tt-backend.md` §4's W2 row, which says 4."""
    program = _program(T)
    assert [(sem.id, sem.name) for sem in program.semaphores] == [
        (0, "ToNorth.full[0]"), (1, "ToNorth.empty[0]"),
        (2, "ToSouth.full[0]"), (3, "ToSouth.empty[0]")]
    assert all(sem.initial_value == 0 for sem in program.semaphores)
    assert len(program.semaphores) <= m5tt.TT_SEM_LIMIT


# -- the device --------------------------------------------------------------

@pytest.mark.parametrize("T", (4, 5))
def test_W2_is_exact_on_ttsim(executed, T):
    """**Q-TT4.** The whole tensor, element for element, against the two-loop numpy Jacobi —
    `np.array_equal`, not a tolerance.

    `tests/integration/test_semantics.py` allows `1e-5` for the AIR side because the plan and
    the kernel text associate the five-term sum the same way only if nothing re-associates it.
    Nothing does: the interpreter already matched at max abs error **0.0**, so exact is the
    right bar here, and soft-float scalar `float` on a Tensix data-movement core meets it.
    """
    expected = jacobi(w2_inputs(T)["U"], T)
    out = executed[T]["U"]
    assert out.dtype == np.float32
    assert np.array_equal(out, expected), (
        f"max abs error {np.abs(out - expected).max()}")
    assert np.abs(out[1:, 1:H + 1, 1:WIDTH - 1]).max() > 0, "an all-zero result would pass"


@pytest.mark.parametrize("T", (4, 5))
def test_W2_carries_the_dirichlet_boundary_into_every_plane(executed, T):
    """B-P25, from the device side. Rows `0` / `H+1` are never drained and columns `0` /
    `W-1` travel with the strip, so every written plane carries plane 0's read-only boundary —
    which is what makes the plan and the kernel text agree at all (`test_semantics.w2_inputs`).
    """
    out, plane0 = executed[T]["U"], w2_inputs(T)["U"][0]
    assert np.array_equal(out[0], plane0), "plane 0 is read-only and is never written back"
    assert np.array_equal(out[1:, 0], np.broadcast_to(plane0[0], (T, WIDTH)))
    assert np.array_equal(out[1:, H + 1], np.broadcast_to(plane0[H + 1], (T, WIDTH)))
    for column in (0, WIDTH - 1):
        assert np.array_equal(out[1:, 1:H + 1, column],
                              np.broadcast_to(plane0[1:H + 1, column], (T, H)))


@pytest.mark.parametrize("T", (4, 5))
def test_W2_matches_the_plan_interpreter(executed, T):
    """Plan semantics == simulator semantics, element for element. One plan, two unrelated
    executions, the same numbers — the backend-neutrality claim, now on the workload whose
    arithmetic is not integer-valued."""
    interpreted = plan_interp.run(m4.plan(w2_legal.legal("npu1", T=T)), w2_inputs(T))
    assert np.array_equal(executed[T]["U"], interpreted["U"])


def test_mutating_the_stencil_constant_changes_the_answer(runner, executed):
    """The negative control: `0.2` becomes `0.25` and the simulator must return something else.

    Without it the exactness test would also pass against a simulator that never ran our kernel
    and left `U` holding the planes it was uploaded with — which, under B-P25, already carry a
    plausible-looking boundary.
    """
    program = _program(4)
    assert program.source.count(_CONSTANT) == 2, "the two STEP bodies no longer spell it twice"
    mutated = dataclasses.replace(
        program, source=program.source.replace(_CONSTANT, "((float)(0.25))"))
    assert mutated.source != program.source

    result = runner.run(mutated, w2_inputs(4))
    assert not np.array_equal(result["U"], jacobi(w2_inputs(4)["U"], 4))
    assert not np.array_equal(result["U"], executed[4]["U"])


@pytest.mark.slow
def test_one_credit_per_link_deadlocks(runner):
    """**The measurement behind R-TT-B's credit rule.** T2 fixed a depth-1 FIFO: the producer's
    put `n` waits for `empty >= n`, one slot outstanding. W3 and the cascade are chains and that
    is correct for them. W2 is not a chain — each PE puts on its outbound link before it gets on
    its inbound one — so with one credit PE0 blocks on a slot only PE1's *later* get can free and
    PE1 blocks on one only PE0's later get can free.

    R-TT-B's credit is derived instead from the pairing: consecutive payloads on a link land in
    `cur` and then in `next`, which do not alias, so two may be in flight and the pair makes
    progress. This test runs the one-credit form in its own process, because a deadlocked
    simulator does not return.
    """
    root = str(pathlib.Path(__file__).resolve().parents[2])
    script = _DEADLOCK.format(root=root, pairs=list(_CREDIT))
    start = time.time()
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run([sys.executable, "-c", script], timeout=60,
                       capture_output=True, text=True, check=False)
    print(f"\nR-TT-B credit rule: the one-credit form did not complete in "
          f"{time.time() - start:.0f}s (the clean run takes ~3s)", flush=True)
