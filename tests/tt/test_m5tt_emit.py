"""M5-TT unit tests: the `TTProgram` W1 emits, and the emitter's own discipline.

These run in the project's own `.venv` and join the default suite: `spatial.m5tt_emit` imports
nothing but the standard library and `spatial.model`, which is the point of keeping `ttnn`
confined to `spatial.m6tt_run`. The simulator run is `tests/tt/test_tt_w1.py`, marked
`requires_ttsim`.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import re
import subprocess
import sys
from pathlib import Path

import pytest

from spatial import m4_mapping as m4
from spatial import m5tt_emit as m5tt
from spatial.model import ChannelSite
from tests.fixtures.mappings import w1_legal, w1flip_legal, w2_legal, w3_legal

SOURCE = inspect.getsource(m5tt)
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def w1():
    """W1's `TTProgram`. The target string is an **AIR** target and means nothing to a Tensix
    core, so `emit` takes none; `test_target_does_not_reach_the_tt_program` holds it to that."""
    return m5tt.emit(m4.plan(w1_legal.legal("npu1")))


# -- the program W1 produces -------------------------------------------------

def test_W1_grid_is_the_logical_herd(w1):
    """`HerdPlan.grid` is `(2, 2)`, and the core range is its 4 cores — not the AIE physical
    herd `(1, 2)` with `repeats`, which is strip-mining and has no meaning on a Tensix grid."""
    assert w1.grid == (2, 2)
    assert w1.core_range == ((0, 0), (1, 1))
    assert [core for core, _ in w1.runtime_args] == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_W1_runtime_args_are_addresses_then_coordinates(w1):
    """The ABI: one base address per L3 tensor in `plan.tensors` order, then the herd
    coordinates in `HerdPlan.coords` order. Only the emitter states it."""
    assert dict(w1.runtime_args)[(1, 0)] == (
        ("addr", "A"), ("addr", "B"), ("addr", "C"), ("const", 1), ("const", 0))


def test_W1_cbs_are_one_per_L1_buffer(w1):
    """`acc` 4096 B, `a` 2048 B, `b` 2048 B, one page each — `ping_pong_candidate` is ignored,
    so `a` and `b` are **not** doubled."""
    assert [(cb.index, cb.name, cb.bytes, cb.page_bytes) for cb in w1.cbs] == [
        (0, "acc", 4096, 4096), (1, "a", 2048, 2048), (2, "b", 2048, 2048)]


def test_W1_io_tensors_are_paged_by_the_row(w1):
    """One page is one row: `shape[-1] * dtype.sizeof`. `m6tt_run` checks the device agrees."""
    assert [(t.name, t.shape, t.page_bytes, t.cta_define) for t in w1.io_tensors] == [
        ("A", (64, 64), 256, "TA_A"), ("B", (64, 64), 256, "TA_B"),
        ("C", (64, 64), 256, "TA_C")]


def test_W1_kernel_has_the_k0_loop_with_four_trips(w1):
    """W1's reduction is tiled `TK = 16` over `K = 64`: four trips, from the plan's `LoopPlan`."""
    assert "for (int32_t k0 = 0; k0 < 64; k0 += 16) {" in w1.source


def test_W1_kernel_reads_the_A_and_B_slabs_row_by_row(w1):
    """The herd side does the DRAM transfer itself.

    `A2L1` is a broadcast channel of `size=(2, 1)` over `broadcast_shape=(2, 2)`, so the core at
    `(tx, ty)` is fed by source index `tx % 2`: its 32 rows start at row `(tx % 2) * 32` and its
    16 columns at `k0`, one 64 B run per row into `a`. `B2L1` is `size=(1, 2)`: 16 rows from
    `k0`, 32 columns from `(ty % 2) * 32`, 128 B a run into `b`.
    """
    assert ("noc_async_read(A_ta.get_noc_addr((uint32_t)((((tx % 2) * 32) + _r1_0)), "
            "(uint32_t)(k0 * 4)), a_l1 + (uint32_t)((_r1_0 * 16) * 4), 64);") in w1.source
    assert "for (int32_t _r1_0 = 0; _r1_0 < 32; _r1_0 += 1) {" in w1.source
    assert ("noc_async_read(B_ta.get_noc_addr((uint32_t)((k0 + _r2_0)), "
            "(uint32_t)(((ty % 2) * 32) * 4)), b_l1 + (uint32_t)((_r2_0 * 32) * 4), 128);"
            ) in w1.source
    assert "for (int32_t _r2_0 = 0; _r2_0 < 16; _r2_0 += 1) {" in w1.source
    assert w1.source.count("noc_async_read_barrier();") == 2


def test_W1_kernel_zeroes_the_accumulator_then_accumulates(w1):
    """Both nests come from the plan: the zeroing `LoopPlan` of `StoreNode(Const 0)`, and the
    `i1`/`j1`/`k1` nest carrying W1's one desugared `accumulate`."""
    assert "acc[((i1 * 32) + j1)] = ((float)(0.0));" in w1.source
    assert "for (int32_t k1 = 0; k1 < 16; k1 += 1) {" in w1.source
    assert ("acc[((i1 * 32) + j1)] = (acc[((i1 * 32) + j1)] + "
            "(a[((i1 * 16) + k1)] * b[((k1 * 32) + j1)]));") in w1.source


def test_W1_kernel_writes_the_C_tile_back(w1):
    """`C2L3` is unbroadcast, so the segment get's `(i_drain, j_drain)` inverts to `(tx, ty)`
    and the tile lands at `C[tx * 32 : +32, ty * 32 : +32]`, row by row."""
    assert ("noc_async_write(acc_l1 + (uint32_t)((_r3_0 * 32) * 4), "
            "C_ta.get_noc_addr((uint32_t)(((tx * 32) + _r3_0)), (uint32_t)((ty * 32) * 4)), "
            "128);") in w1.source
    assert w1.source.count("noc_async_write_barrier();") == 1


def test_W1_kernel_declares_one_l1_pointer_per_cb(w1):
    for cb in w1.cbs:
        assert f"const uint32_t {cb.name}_l1 = get_write_ptr({cb.index});" in w1.source
        assert (f"volatile tt_l1_ptr float* {cb.name} = "
                f"(volatile tt_l1_ptr float*){cb.name}_l1;") in w1.source


def test_W1_kernel_has_balanced_braces(w1):
    assert w1.source.count("{") == w1.source.count("}")
    assert w1.source.rstrip().endswith("}")


# -- the program W3 produces (T2) --------------------------------------------

@pytest.fixture(scope="module")
def w1flip():
    """W1-flip's `TTProgram`: a 1-D grid and an ascending cascade, gate T4."""
    return m5tt.emit(m4.plan(w1flip_legal.legal("npu1")))


@pytest.fixture(scope="module")
def w3():
    """W3's `TTProgram`: a 1-D chain of four PEs with the first core↔core channel."""
    return m5tt.emit(m4.plan(w3_legal.legal("npu1")))


def test_W3_grid_is_the_four_pe_chain(w3):
    assert w3.grid == (4,)
    assert w3.core_range == ((0, 0), (3, 0))
    assert [core for core, _ in w3.runtime_args] == [(0, 0), (1, 0), (2, 0), (3, 0)]


def test_W3_has_two_semaphores_per_link_and_none_for_the_L3_edges(w3):
    """`West` is `size=(3,)`, three links over the chain, each with a `full` on its consumer and
    an `empty` on its producer. `WestIn` and `EastOut` have an L3 end, so they are DRAM
    transfers and carry no semaphore at all (`design/08-tt-backend.md` §3.6)."""
    assert [(sem.id, sem.name, sem.initial_value) for sem in w3.semaphores] == [
        (0, "West.full[0]", 0), (1, "West.empty[0]", 0),
        (2, "West.full[1]", 0), (3, "West.empty[1]", 0),
        (4, "West.full[2]", 0), (5, "West.empty[2]", 0)]


def test_W3_runtime_args_carry_the_peer_of_each_channel_end(w3):
    """Block B (`design/08-tt-backend.md` §3.4): per core↔core channel, the peer this core puts
    to and the peer it gets from, as logical cores the runner converts. The head has no `West`
    get and the tail no `West` put, so those slots hold the core's own coordinates and the guard
    keeps them unused."""
    args = dict(w3.runtime_args)
    assert args[(0, 0)] == (("addr", "q"), ("addr", "r"), ("addr", "S"), ("const", 0),
                            ("noc_x", (1, 0)), ("noc_y", (1, 0)),      # puts to core 1
                            ("noc_x", (0, 0)), ("noc_y", (0, 0)))      # no get: itself
    assert args[(2, 0)] == (("addr", "q"), ("addr", "r"), ("addr", "S"), ("const", 2),
                            ("noc_x", (3, 0)), ("noc_y", (3, 0)),
                            ("noc_x", (1, 0)), ("noc_y", (1, 0)))
    assert args[(3, 0)][-4:] == (("noc_x", (3, 0)), ("noc_y", (3, 0)),  # no put: itself
                                 ("noc_x", (2, 0)), ("noc_y", (2, 0)))


def test_W3_cbs_are_rounded_to_the_circular_buffer_alignment(w3):
    """`prev`/`cur` are 36 B in the plan and `edge_in`/`edge_out` 4 B. Rounding each up to 32 B
    keeps every CB base 32 B aligned, which is what R-TT-A′'s L1 residues rest on; the flat index
    arithmetic never sees the padding."""
    assert [(cb.index, cb.name, cb.bytes) for cb in w3.cbs] == [
        (0, "qb", 128), (1, "rb", 32), (2, "prev", 64), (3, "cur", 64),
        (4, "edge_in", 32), (5, "edge_out", 32)]


def test_W3_put_site_is_the_depth_one_fifo(w3):
    """§3.5, the whole protocol in one site: wait for a free slot, write straight into the
    consumer's `edge_in`, make the payload visible, then advertise it."""
    assert ("noc_semaphore_wait_min((volatile tt_l1_ptr uint32_t*)"
            "get_semaphore((uint32_t)(((tx * 2) + 1))), West_put_n);\n"
            "            noc_async_write(edge_out_l1, "
            "get_noc_addr(West_put_x, West_put_y, edge_in_l1), 4);\n"
            "            noc_async_write_barrier();\n"
            "            noc_semaphore_inc(get_noc_addr(West_put_x, West_put_y, "
            "get_semaphore((uint32_t)((tx * 2)))), 1);\n"
            "            West_put_n += 1;") in w3.source


def test_W3_get_site_releases_the_previous_slot_before_it_waits(w3):
    """§3.5 point 5 in its operational form: the `empty` increment sits at the top of the *next*
    get on the link, never straight after the wait (which would let the producer overwrite a
    region still being read) and never at the end of the enclosing body (which deadlocks, because
    W3's `i` loop body holds two rows)."""
    assert ("if (West_get_n > 0) {\n"
            "                noc_semaphore_inc(get_noc_addr(West_get_x, West_get_y, "
            "get_semaphore((uint32_t)((((tx + -1) * 2) + 1)))), 1);\n"
            "            }\n"
            "            noc_semaphore_wait_min((volatile tt_l1_ptr uint32_t*)"
            "get_semaphore((uint32_t)(((tx + -1) * 2))), West_get_n + 1);\n"
            "            West_get_n += 1;") in w3.source


def test_W3_binds_the_twinless_segment_loops_to_occurrence_counters(w3):
    """§3.3 rule 5. W3's `i_source` and `i_drain` run `[1, 33)` step 1 at segment scope while the
    herd runs `i` over `[1, 33)` **step 2** with two row bodies inside, so there is no twin to
    match. The channel is a FIFO, so the site's n-th transfer is the loop's n-th trip."""
    for counter in ("WestIn_get_n", "EastOut_put_n", "SOut_put_n", "West_put_n", "West_get_n"):
        assert f"int32_t {counter} = 0;" in w3.source
    assert w3.source.count("SOut_put_n += 1;") == 2          # one per row body
    assert ("noc_async_write(cur_l1 + (uint32_t)(1 * 4), "
            "S_ta.get_noc_addr((uint32_t)((1 + SOut_put_n)), "
            "(uint32_t)(((tx * 8) + 1) * 4)), 32);") in w3.source
    assert ("noc_async_read(S_ta.get_noc_addr((uint32_t)((1 + WestIn_get_n)), "
            "(uint32_t)(0 * 4)), edge_in_l1, 4);") in w3.source


def test_W3_computes_the_recurrence_as_nested_ternaries(w3):
    """§3.7: `MaxMin` left-folds to nested ternaries and `Select` is an expression, never control
    flow, so the association is the interpreter's."""
    assert w3.source.count("? ((int32_t)(2)) : ((int32_t)(-1))") == 8
    assert "cur[0] = edge_in[0];" in w3.source and "prev[0] = edge_in[0];" in w3.source
    assert "edge_out[0] = cur[8];" in w3.source and "edge_out[0] = prev[8];" in w3.source


def test_W3_kernel_has_balanced_braces(w3):
    assert w3.source.count("{") == w3.source.count("}")


# -- R-TT-A': the DRAM layout policy -----------------------------------------

def test_leading_pad_is_the_smallest_that_makes_every_transfer_congruent():
    """The solver, on its own. `(modulus, delta)` is what one transfer asks: `delta` is its DRAM
    byte offset minus its L1 byte offset with no pad, and the pad has to close the gap."""
    assert m5tt._leading_pad(4, [(16, 0)]) == 0
    assert m5tt._leading_pad(4, [(16, -4)]) == 1        # L1 starts 4 B in; DRAM must follow
    assert m5tt._leading_pad(4, [(32, -8)]) == 2
    assert m5tt._leading_pad(4, [(32, 0), (16, 0)]) == 0
    assert m5tt._leading_pad(4, [(32, 0), (16, -4)]) is None    # 0 mod 8 and 1 mod 4 at once


@pytest.mark.parametrize("fixture,pads,pages", (
    (w1_legal, {"A": 0, "B": 0, "C": 0}, {"A": 256, "B": 256, "C": 256}),
    (w2_legal, {"U": 0}, {"U": 64}),
    (w3_legal, {"q": 0, "r": 0, "S": 0}, {"q": 128, "r": 128, "S": 160}),
))
def test_the_measured_layout_is_no_leading_pad_and_a_padded_row(fixture, pads, pages):
    """R-TT-A′'s consequence, and it is not the one the ruling predicted: with the NoC's rule a
    **relative** congruence, `p = 0` satisfies every transfer of every workload, including W3's
    `S`. The only padding left is the row itself — 33 `i32` = 132 B → 160 B — which is what keeps
    a page index from moving a residue. `p = 7` would put `S[i, 0]` at byte 28 against an L1
    offset of 0 and is refused.
    """
    program = m5tt.emit(m4.plan(fixture.legal("npu1")))
    assert {t.name: t.pad_elems for t in program.io_tensors} == pads
    assert {t.name: t.page_bytes for t in program.io_tensors} == pages
    assert all(t.page_bytes % 32 == 0 for t in program.io_tensors)


def test_a_plan_no_leading_pad_can_satisfy_raises_TT_ALIGNMENT():
    """The negative: shift W3's `SOut` drain one column left at its **segment** end only, so the
    32-byte slab starts at DRAM byte `32·tx` while its source is still `cur + 4`. That transfer
    now wants `p ≡ 1 (mod 4)` and `WestIn` still wants `p ≡ 0 (mod 8)`; no pad does both."""
    plan = m4.plan(w3_legal.legal("npu1"))
    bundle = plan.segment_body[5]                       # for pj_bundle: for i_drain: SOut.get
    drain = bundle.body[0]
    site = drain.body[0]
    assert site.channel == "SOut" and site.scope == "segment"
    row, column = site.region.offsets
    moved = dataclasses.replace(
        site, region=dataclasses.replace(
            site.region, offsets=(row, dataclasses.replace(column, const=0))))
    broken = dataclasses.replace(
        plan,
        segment_body=plan.segment_body[:5] + (
            dataclasses.replace(bundle, body=(dataclasses.replace(drain, body=(moved,)),)),),
        # the same site object is reachable from plan.channels, and the plan's own validator
        # refuses two sites sharing an id
        channels=tuple(
            dataclasses.replace(channel, sites=tuple(
                moved if other.id == site.id else other for other in channel.sites))
            if channel.name == site.channel else channel
            for channel in plan.channels))

    with pytest.raises(m5tt.TTAlignmentError, match=r"TT-ALIGNMENT.*'S'.*no pad"):
        m5tt.emit(broken)


def test_TT_ALIGNMENT_is_not_in_the_frozen_error_catalogue():
    """`design/08-tt-backend.md` §6.1: the five TT codes are *proposed*, not adopted, so the TT
    emitter raises its own exception type and nothing reaches `06-interfaces.md`'s 43."""
    assert issubclass(m5tt.TTAlignmentError, m5tt.TTEmitError)
    from spatial.model import EmissionError
    assert not issubclass(m5tt.TTEmitError, EmissionError)


# -- TT-P3, the resource model -----------------------------------------------

def test_semaphore_and_L1_budgets_are_measured_constants():
    """`design/08-tt-backend.md` §4. The semaphore limit is measured (the host refuses id 16 with
    'Semaphore id 16 exceeds max value 15'); the L1 figure is an estimate and says so."""
    assert m5tt.TT_SEM_LIMIT == 16
    assert m5tt.TT_L1_USABLE == 1_499_136 - 32_768


@pytest.mark.parametrize("fixture,cb_bytes,sems", (
    (w1_legal, 8192, 0),
    (w2_legal, 1280, 4),
    (w3_legal, 352, 6),
))
def test_TT_P3_budget_rows(fixture, cb_bytes, sems):
    """One row of §4's table each, against the constants rather than a comment."""
    program = m5tt.emit(m4.plan(fixture.legal("npu1")))
    assert sum(cb.bytes for cb in program.cbs) == cb_bytes
    assert len(program.semaphores) == sems
    assert len(program.semaphores) <= m5tt.TT_SEM_LIMIT
    assert cb_bytes + 16 * sems <= m5tt.TT_L1_USABLE


def test_a_plan_over_the_semaphore_limit_is_refused():
    """Nine links would need 18 ids against the measured 16, and the refusal must name the
    number and where it came from."""
    plan = m4.plan(w3_legal.legal("npu1"))
    west = next(c for c in plan.channels if c.name == "West")
    wide = dataclasses.replace(plan, channels=tuple(
        dataclasses.replace(c, size=(9,)) if c is west else c for c in plan.channels))
    with pytest.raises(m5tt.TTNotImplemented, match=r"18 semaphores.*16 \(ids 0\.\.15"):
        m5tt.emit(wide)


# -- R-TT-B: occurrence pairing, and the credit it forces ---------------------

@pytest.fixture(scope="module")
def w2():
    """W2's `TTProgram` at `T = 4`. The simulator run is `tests/tt/test_tt_w2.py`."""
    return m5tt.emit(m4.plan(w2_legal.legal("npu1", T=4)))


def test_W2_has_two_links_and_four_semaphores(w2):
    """`design/08-tt-backend.md` §4's W2 row: `ToNorth[0]` and `ToSouth[0]`, one `full` on each
    consumer and one `empty` on each producer. `UIn`/`UOut` have an L3 end and carry none."""
    assert [(sem.id, sem.name) for sem in w2.semaphores] == [
        (0, "ToNorth.full[0]"), (1, "ToNorth.empty[0]"),
        (2, "ToSouth.full[0]"), (3, "ToSouth.empty[0]")]
    assert w2.grid == (2,) and [cb.name for cb in w2.cbs] == ["cur", "next"]


def test_W2_put_occurrences_land_in_alternating_buffers(w2):
    """**Ruling R-TT-B.** The blocker T2 hit was that `ToNorth` has gets landing in *both* `cur`
    and `next`, and a remote write has one destination. The pairing answers it: the producer's
    k-th put occurrence is consumed by the consumer's k-th get occurrence, so the first put of a
    `t` trip writes row 9 of the consumer's `cur` and the second writes row 9 of its `next`.

    Row 9 is `HS + 1` and 16 `f32` is 64 B, so each is one `noc_async_write` with no loop.
    """
    for channel, row in (("ToNorth", 9), ("ToSouth", 0)):
        writes = [line.strip() for line in w2.source.splitlines()
                  if f"get_noc_addr({channel}_put_x" in line and "noc_async_write(" in line]
        assert len(writes) == 2, f"{channel} should have two put occurrences per t trip"
        assert writes[0] != writes[1], f"{channel}'s two puts must not share a destination"
        for occurrence, buffer in enumerate(("cur", "next")):
            assert (f"get_noc_addr({channel}_put_x, {channel}_put_y, "
                    f"{buffer}_l1 + (uint32_t)(({row} * 16) * 4)), 64);") in writes[occurrence]


def test_W2_gives_each_link_two_credits_and_W3_the_flip_one(w2):
    """R-TT-B's credit: how many payloads may be in flight is the smallest gap between two
    landing regions that alias. W2's alternate between `cur` and `next`, which do not, so the
    gap is 2 — and 1 **deadlocks**, because each PE puts before it gets on links that run the
    other way (measured: `tests/tt/test_tt_w2.py::test_one_credit_per_link_deadlocks`). W3 and
    the cascade land every payload in the same buffer, so they keep T2's depth-1 FIFO and their
    emitted kernels are byte-identical to T2's.
    """
    for counter in ("ToNorth_put_n", "ToSouth_put_n"):
        assert f"({counter} < 1 ? 0 : {counter} - 1));" in w2.source
        assert w2.source.count(f"), {counter});") == 0, "a bare counter is one credit"
    for fixture, counter in ((w3_legal, "West_put_n"), (w1flip_legal, "CascadeK_put_n")):
        source = m5tt.emit(m4.plan(fixture.legal("npu1"))).source
        assert f"), {counter});" in source and f"({counter} < " not in source


def test_W2_seeds_next_and_drains_eight_rows_per_step(w2):
    """The rest of W2's shape, which is ordinary emitter machinery rather than R-TT-B: the seed
    copy is a plain `StoreNode` nest (`design/PROGRESS-B.md` §P5), the `UIn` stage reads all ten
    rows of the strip including both ghosts, and each `STEP` drains eight interior rows into
    plane `t + t_off + 1` through the twinless-loop occurrence counter of §3.3 rule 5."""
    assert "next[((i1 * 16) + j)] = cur[((i1 * 16) + j)];" in w2.source
    assert "for (int32_t _r1_0 = 0; _r1_0 < 10; _r1_0 += 1) {" in w2.source
    assert w2.source.count("UOut_put_n += 1;") == 2          # one per STEP
    for buffer, loop in (("next", "_r4_0"), ("cur", "_r7_0")):
        assert (f"noc_async_write({buffer}_l1 + (uint32_t)(((1 + {loop}) * 16) * 4), "
                f"U_ta.get_noc_addr((uint32_t)((((UOut_put_n + 1) * 18) + "
                f"(((tx * 8) + 1) + {loop}))), (uint32_t)(0 * 4)), 64);") in w2.source


@pytest.mark.parametrize("fixture", (w1_legal, w2_legal, w3_legal, w1flip_legal))
def test_f32_constants_are_narrowed_before_they_are_used(fixture):
    """**Q-TT4's other half, measured.** In C++ an unsuffixed `0.2` is a `double`, so
    `0.2 * <float>` would be evaluated in `double` and narrowed only on the store — which is not
    what `numpy.float32(0.2) * <float32>` computes. The emitter narrows every `f32` `Const` at
    the point of use (`((float)(0.2))`), and the cast is load-bearing: with the bare literal the
    device returns 358 of 896 interior elements off by one ulp (`design/PROGRESS-TT.md` §T3).

    `0.2f` measures identical — same answer, same cycle count — but the cast also covers a
    `Const.text` with no decimal point, where `5f` would not compile.
    """
    source = m5tt.emit(m4.plan(fixture.legal("npu1"))).source
    assert "double" not in source
    for line in source.splitlines():
        for token in re.findall(r"(?<![\w.])\d+\.\d*(?:[eE][-+]?\d+)?", line):
            assert f"((float)({token}))" in line, (
                f"the literal {token} is not narrowed to float where it is used: {line.strip()}")


def test_the_cascade_plan_emits_three_links(w1flip):
    """W1-flip's `CascadeK` is the same core↔core shape as `West` — three links, six
    semaphores — and it reached the device at T2 with no emitter change (gate T4)."""
    assert [sem.name for sem in w1flip.semaphores] == [
        "CascadeK.full[0]", "CascadeK.empty[0]", "CascadeK.full[1]", "CascadeK.empty[1]",
        "CascadeK.full[2]", "CascadeK.empty[2]"]


# -- what this emitter still does not do -------------------------------------

def test_an_unpairable_link_is_refused_and_says_why():
    """R-TT-B needs the two occurrence sequences to have the same length: a FIFO's n-th payload
    is consumed by its n-th get. Drop W2's second `ToNorth` get and the emitter must refuse
    rather than pair a put with a get that will not run."""
    plan = m4.plan(w2_legal.legal("npu1", T=4))
    loop = plan.herd_body[4]
    dropped = tuple(node for node in loop.body
                    if not (isinstance(node, ChannelSite) and node.id == "ToNorth.get.8@herd"))
    broken = dataclasses.replace(plan, herd_body=plan.herd_body[:4] + (
        dataclasses.replace(loop, body=dropped),))
    with pytest.raises(m5tt.TTEmitError, match=r"'ToNorth' link 0 has 2 put occurrence"):
        m5tt.emit(broken)


# -- D-14, and the module's own dependencies ---------------------------------

def _code_only(source: str) -> str:
    """`source` with its comments and every docstring removed — the *code*, which is what D-14
    is about. The module docstring has to be free to say which names the code may not use."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            del body[0]
    return ast.unparse(tree)


def test_emitter_makes_no_decisions():
    """D-14, as for M5: the TT emitter reads no field of the legality or schedule contracts and
    no kernel size. Everything it emits it read out of the `MappingPlan`."""
    code = _code_only(SOURCE)
    for forbidden in ("plan.mapping", "LegalMapping", "KernelModel", "ScheduleModel",
                      "physical_herd", "repeats", ".schedule", ".kernel", "if tx ==",
                      "npu1", "npu2"):
        assert forbidden not in code, f"M5-TT's code names {forbidden!r}"


def test_emitter_imports_nothing_but_stdlib_and_spatial():
    """It has to run in the project's own `.venv`, which cannot hold `ttnn`: the wheel pins
    `numpy<2` against the project's `numpy==2.5.3`."""
    allowed = {"spatial", "dataclasses", "math", "typing", "__future__"}
    for node in ast.walk(ast.parse(SOURCE)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert name.split(".")[0] in allowed, f"M5-TT imports {name!r}"


def test_runner_imports_ttnn_lazily():
    """`spatial.m6tt_run` is importable without `ttnn`, so the default suite can collect it."""
    source = (ROOT / "spatial" / "m6tt_run.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    top = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
           for alias in node.names if _at_module_level(tree, node)}
    assert "ttnn" not in top, "m6tt_run imports ttnn at module level"
    assert any(isinstance(node, ast.Import) and any(a.name == "ttnn" for a in node.names)
               for node in ast.walk(tree)), "m6tt_run never imports ttnn at all"


def _at_module_level(tree: ast.Module, target: ast.AST) -> bool:
    return any(node is target for node in tree.body)


@pytest.mark.parametrize("fixture", ("w1_legal", "w2_legal", "w3_legal", "w1flip_legal"))
def test_emission_is_deterministic_across_hash_seeds(fixture):
    """No set or dict iteration order reaches the text: two interpreters with different
    `PYTHONHASHSEED` must emit byte-identical C++ and an identical program. W3 and the flip are
    the ones that could go wrong — semaphore ids, peer coordinates and occurrence counters all
    come out of dictionaries keyed by channel and core."""
    script = ("import sys;"
              "from spatial import m4_mapping as m4, m5tt_emit as tt;"
              f"from tests.fixtures.mappings import {fixture} as fx;"
              "p = tt.emit(m4.plan(fx.legal('npu1')));"
              "sys.stdout.write(repr((p.source, p.cbs, p.io_tensors, p.runtime_args, "
              "p.semaphores)))")
    outputs = [subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True,
                              capture_output=True, text=True,
                              env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"}).stdout
               for seed in ("0", "12345")]
    assert outputs[0] == outputs[1]


def test_target_does_not_reach_the_tt_program():
    """`"npu1"` and `"npu2"` select an AIE generation for `air.api`'s `build()`. M4's plan is the
    same either way, so the TT program must be too — the claim `design/PROGRESS-TT.md` §4 makes
    about the plan being backend-neutral is exactly this."""
    assert (m5tt.emit(m4.plan(w1_legal.legal("npu1")))
            == m5tt.emit(m4.plan(w1_legal.legal("npu2"))))
    assert "target" not in inspect.signature(m5tt.emit).parameters
