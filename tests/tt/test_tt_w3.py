"""W3 Smith-Waterman on Tenstorrent's functional simulator: the first core↔core protocol.

Not in the default run. Build the environment and run it with::

    source scripts/tt_env.sh
    .venv-tt/bin/python -m pytest -m requires_ttsim tests/tt

What this module establishes that `test_tt_w1.py` could not: W1 has **no** core↔core channel, so
its whole program is DRAM traffic on four independent cores. W3's `West` is a three-link chain,
and every value that crosses it does so through the depth-1 FIFO of
`design/08-tt-backend.md` §3.5 — a direct remote L1 write, a barrier, and two counting
semaphores. The wavefront is emergent: nothing in the emitted kernel expresses the skew, each PE
blocks on its own `noc_semaphore_wait_min`, and the diagonal order follows.

Four simulator runs, each costing wall time, so every assertion below is written against one of
them:

1. the clean program — exact against the textbook DP, inputs untouched, equal to the plan
   interpreter element for element;
2. the negative control — one mutated `Select` arm, which must produce a different answer;
3. Q-TT3 — `noc_semaphore_inc` moved *before* `noc_async_write_barrier()`, to see whether the
   simulator notices the reversed order;
4. A-TT1 — a probe kernel reporting `get_write_ptr` and `get_semaphore` from every core, which
   is the assumption that makes a remote write addressable at all.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pytest

from spatial import m4_mapping as m4
from spatial import m5tt_emit as m5tt
from tests.fixtures.mappings import w3_legal
from tests.helpers import plan_interp
from tests.integration.test_semantics import smith_waterman, w3_inputs

pytestmark = pytest.mark.requires_ttsim

_MATCH_ARM = "? ((int32_t)(2)) : ((int32_t)(-1))"
"""W3's one `Select`: `q[i-1] == r[j-1] ? MATCH : MISMATCH`. The negative control turns every
MATCH into a MISMATCH, which changes the score of any pair of sequences that align at all. It
appears **eight** times, not once: `MaxMin` folds to a nested ternary, so each of the two row
bodies spells its second operand four times (`design/08-tt-backend.md` §3.7)."""

_BARRIER = "noc_async_write_barrier();"
_INC = ("noc_semaphore_inc(get_noc_addr(West_put_x, West_put_y, "
        "get_semaphore((uint32_t)((tx * 2)))), 1);")
"""§3.5 point 4, as emitted: the payload is made visible **before** the flag that advertises it.
Q-TT3 swaps exactly these two lines at the `West` put sites."""

_PROBE = """\
// A-TT1 probe. Reports this core's circular-buffer and semaphore addresses.
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {{
    constexpr auto dbg_args = TensorAccessorArgs<TA_dbg>();
    const auto dbg_ta = TensorAccessor(dbg_args, get_arg_val<uint32_t>(1));
    const int32_t tx = (int32_t)get_arg_val<uint32_t>(2);
    const uint32_t rep_l1 = get_write_ptr(0);
    volatile tt_l1_ptr uint32_t* rep = (volatile tt_l1_ptr uint32_t*)rep_l1;
{body}
    noc_async_write(rep_l1, dbg_ta.get_noc_addr((uint32_t)tx, 0), {row});
    noc_async_write_barrier();
}}
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


@pytest.fixture(scope="module")
def program():
    return m5tt.emit(m4.plan(w3_legal.legal("npu1")))


@pytest.fixture(scope="module")
def executed(runner, program):
    """The one clean simulator run this module pays for."""
    start = time.time()
    result = runner.run(program, w3_inputs())
    print(f"\nW3 on ttsim: {time.time() - start:.1f}s wall for {program.grid} cores, "
          f"{len(program.semaphores)} semaphores", flush=True)
    return result


# -- the program, before any device ------------------------------------------

def test_W3_allocates_two_semaphores_per_link(program):
    """`West` has `size=(3,)` — three links over a four-PE chain — and each carries a `full` on
    its consumer and an `empty` on its producer. `WestIn` and `EastOut` have an L3 end and are
    DRAM transfers, so they carry none (`design/08-tt-backend.md` §3.6)."""
    assert [(sem.id, sem.name) for sem in program.semaphores] == [
        (0, "West.full[0]"), (1, "West.empty[0]"),
        (2, "West.full[1]"), (3, "West.empty[1]"),
        (4, "West.full[2]"), (5, "West.empty[2]")]
    assert all(sem.initial_value == 0 for sem in program.semaphores)
    assert len(program.semaphores) <= m5tt.TT_SEM_LIMIT


# -- the device --------------------------------------------------------------

def test_W3_is_exact_on_ttsim(executed):
    """The whole point: `MappingPlan` → TT-Metalium → ttsim reproduces the textbook DP exactly.

    Exactly, not to a tolerance: the scores are `i32` and every intermediate is an integer.
    """
    given = w3_inputs()
    assert executed["S"].dtype == np.int32
    assert np.array_equal(executed["S"], smith_waterman(given["q"], given["r"]))
    assert executed["S"][1:, 1:].any(), "an all-zero score matrix would pass vacuously"


def test_W3_leaves_the_boundary_and_its_inputs_alone(executed):
    """Row 0 and column 0 are the DP's boundary and nothing writes them — which is what makes
    `WestIn`'s constant source legitimate. `q` and `r` are read-only."""
    given = w3_inputs()
    assert not executed["S"][0].any() and not executed["S"][:, 0].any()
    assert np.array_equal(executed["q"], given["q"])
    assert np.array_equal(executed["r"], given["r"])


def test_W3_matches_the_plan_interpreter(executed):
    """Plan semantics == simulator semantics, element for element. One plan, two unrelated
    executions, the same numbers — the backend-neutrality claim."""
    interpreted = plan_interp.run(m4.plan(w3_legal.legal("npu1")), w3_inputs())
    assert np.array_equal(executed["S"], interpreted["S"])


def test_mutating_the_match_score_changes_the_answer(runner, program, executed):
    """The negative control: turn every MATCH into a MISMATCH and the simulator must return
    something else. Without it, the exactness test would also pass against a simulator that never
    ran our kernel and left `S` holding the zeros it was uploaded with."""
    assert program.source.count(_MATCH_ARM) == 8, "the Select fold is no longer eight-fold"
    mutated = dataclasses.replace(
        program, source=program.source.replace(_MATCH_ARM, "? ((int32_t)(-1)) : ((int32_t)(-1))"))
    assert mutated.source != program.source

    given = w3_inputs()
    result = runner.run(mutated, given)
    assert not np.array_equal(result["S"], smith_waterman(given["q"], given["r"]))
    assert not np.array_equal(result["S"], executed["S"])


def test_Q_TT3_reversing_the_barrier_and_the_increment(runner, program, executed, capfd):
    """**Q-TT3**, measured rather than argued: move `noc_semaphore_inc` *before*
    `noc_async_write_barrier()` at every `West` put — advertising the payload before making it
    visible — and record what the simulator does.

    The test asserts only that the run completes and reports what happened; the spec's order is
    what the emitter emits either way, and this run is a measurement of whether ttsim can tell.
    Recorded in `design/PROGRESS-TT.md` §T2.
    """
    ordered = f"{_BARRIER}\n            {_INC}"
    reversed_ = f"{_INC}\n            {_BARRIER}"
    assert program.source.count(ordered) == 2, "the put sites are no longer the pair of rows"
    mutated = dataclasses.replace(program,
                                  source=program.source.replace(ordered, reversed_))

    given = w3_inputs()
    result = runner.run(mutated, given)
    printed = capfd.readouterr()
    same = np.array_equal(result["S"], executed["S"])
    undefined = "UndefinedBehavior" in printed.out + printed.err
    print(f"\nQ-TT3: increment before barrier -> answer unchanged: {same}; "
          f"ttsim reported UndefinedBehavior: {undefined}", flush=True)
    assert result["S"].shape == executed["S"].shape


def test_A_TT1_every_core_sees_the_same_addresses(runner, program):
    """**A-TT1 / Q-TT1**, measured on *this* program's circular-buffer table: a buffer's L1
    address and a semaphore id's L1 address are the same number on every core.

    This is what makes a remote write addressable — the producer computes the consumer's
    `edge_in` address as its own — so it is checked rather than assumed. The probe replaces the
    kernel and the L3 tensors and keeps W3's circular buffers and semaphores, reporting
    `get_write_ptr` and `get_semaphore` into a debug tensor, one row per core.
    """
    values = list(range(len(program.cbs))) + [sem.id for sem in program.semaphores]
    row = 16
    assert len(values) <= row
    body = "\n".join(
        [f"    rep[{cb.index}] = get_write_ptr({cb.index});" for cb in program.cbs]
        + [f"    rep[{len(program.cbs) + slot}] = (uint32_t)get_semaphore({sem.id});"
           for slot, sem in enumerate(program.semaphores)]
        + [f"    rep[{slot}] = 0u;" for slot in range(len(values), row)])
    dtype = program.cbs[0].dtype
    tensors = (
        m5tt.TTTensor(name="pad", shape=(1, row), dtype=dtype,
                      page_bytes=row * dtype.sizeof, cta_define="TA_pad"),
        m5tt.TTTensor(name="dbg", shape=(program.grid[0], row), dtype=dtype,
                      page_bytes=row * dtype.sizeof, cta_define="TA_dbg"))
    probe = dataclasses.replace(
        program, io_tensors=tensors,
        source=_PROBE.format(body=body, row=row * dtype.sizeof),
        runtime_args=tuple((core, (("addr", "pad"), ("addr", "dbg"), ("const", core[0])))
                           for core, _ in program.runtime_args))
    out = runner.run(probe, {"pad": np.zeros((1, row), dtype=np.int32),
                             "dbg": np.zeros((program.grid[0], row), dtype=np.int32)})["dbg"]
    print(f"\nA-TT1: CB and semaphore addresses per core:\n{out[:, :len(values)]}", flush=True)
    for core in range(1, program.grid[0]):
        assert np.array_equal(out[core, :len(values)], out[0, :len(values)]), (
            f"core {core} sees different addresses from core 0; A-TT1 does not hold and the "
            f"consumer's buffer address would have to become a runtime-arg block")
    addresses = out[0, :len(program.cbs)].astype(np.int64)
    assert all(address % 32 == 0 for address in addresses), (
        f"a circular-buffer base is not 32 B aligned ({addresses}); R-TT-A′'s L1 residues "
        f"assume it is")
