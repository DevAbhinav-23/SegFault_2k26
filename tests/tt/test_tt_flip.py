"""W1-flip, the cascade chain, on ttsim — the second core↔core workload, and not a planned one.

Gate **T4** in `design/08-tt-backend.md` §7. It is here because the T2 emitter turned out to
cover it: nothing in `spatial/m5tt_emit.py` mentions a workload, so the same depth-1 FIFO that
carries W3's four-byte score column carries the flip's **32 × 64 partial tile** — 8 192 B as
thirty-two 256-byte remote writes inside one generated loop nest, which W3's single-element edge
never exercises. `ChannelPlan.chain_direction` is read and never derived (D-14); on TT it decides
nothing, because the plan's own guards and index expressions already name which end of each link
is the producer.

Two simulator runs, roughly a minute each::

    source scripts/tt_env.sh
    .venv-tt/bin/python -m pytest -m requires_ttsim tests/tt/test_tt_flip.py
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pytest

from spatial import m4_mapping as m4
from spatial import m5tt_emit as m5tt
from tests.fixtures.mappings import w1flip_legal
from tests.helpers import plan_interp
from tests.integration.test_semantics import inputs

pytestmark = pytest.mark.requires_ttsim

_ACCUMULATE = "acc[((i1 * 64) + j)] = (acc[((i1 * 64) + j)] + recv[((i1 * 64) + j)]);"
"""The middle-PE block's one line: `acc += recv`, the thing the chain exists to do. The negative
control drops the received partial, which is FR-TT8's 'a skipped accumulate'."""


@pytest.fixture(scope="module")
def runner():
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
    return m5tt.emit(m4.plan(w1flip_legal.legal("npu1")))


@pytest.fixture(scope="module")
def executed(runner, program):
    start = time.time()
    result = runner.run(program, inputs())
    print(f"\nW1-flip on ttsim: {time.time() - start:.1f}s wall for {program.grid} cores, "
          f"{len(program.semaphores)} semaphores", flush=True)
    return result


def test_flip_carries_a_whole_tile_over_the_link(program):
    """The cascade moves `acc` into the next PE's `recv` a row at a time — thirty-two 256-byte
    writes under one barrier and one `full` increment, which is the multi-row shape of §3.5 that
    W3's four-byte edge cannot reach."""
    assert [sem.name for sem in program.semaphores] == [
        "CascadeK.full[0]", "CascadeK.empty[0]", "CascadeK.full[1]", "CascadeK.empty[1]",
        "CascadeK.full[2]", "CascadeK.empty[2]"]
    assert ("noc_async_write(acc_l1 + (uint32_t)((_r3_0 * 64) * 4), "
            "get_noc_addr(CascadeK_put_x, CascadeK_put_y, "
            "recv_l1 + (uint32_t)((_r3_0 * 64) * 4)), 256);") in program.source
    assert "for (int32_t _r3_0 = 0; _r3_0 < 32; _r3_0 += 1) {" in program.source


def test_flip_is_exact_on_ttsim(executed):
    """Same kernel and same data as W1, a different dataflow: each PE owns a `k`-slice, sums its
    own partial tile and hands it up the chain, and only `tx == PK-1` reaches L3. The values are
    integer-valued `f32` in `[-8, 8)` with `K = 64`, so `==` is the right comparison."""
    given = inputs()
    assert executed["C"].dtype == np.float32
    assert np.array_equal(executed["C"], given["A"] @ given["B"])


def test_flip_matches_the_plan_interpreter(executed):
    interpreted = plan_interp.run(m4.plan(w1flip_legal.legal("npu1")), inputs())
    assert np.array_equal(executed["C"], interpreted["C"])


def test_dropping_the_cascade_accumulate_changes_the_answer(runner, program, executed):
    """The negative control: throw away what arrived over the link. The chain then reports only
    the tail PE's own `k`-slice, so the product is wrong — which is what proves the three links
    carried anything at all."""
    assert program.source.count(_ACCUMULATE) == 1
    mutated = dataclasses.replace(program, source=program.source.replace(
        _ACCUMULATE, "acc[((i1 * 64) + j)] = acc[((i1 * 64) + j)];"))
    assert mutated.source != program.source

    given = inputs()
    result = runner.run(mutated, given)
    assert not np.array_equal(result["C"], given["A"] @ given["B"])
    assert not np.array_equal(result["C"], executed["C"])
