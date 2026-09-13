"""Level O/G — the three semantic checks of `design/04-test-plan.md` §3.4, over the **plan**.

We cannot execute the emitted AIR as a functional oracle: `air.api` has no interpreter
(VF §D.10), `air-runner` is a timing model (VF §S7), and the CPU backend JITs a *lowered* module
and so tests the compiler's output rather than the user's source (PC §1.4 item 1). The semantic
link is therefore established structurally and by interpreting the plan
(`tests/helpers/plan_interp.py`, decision **D-9**), which is what these three tests are:

1. `test_sem_access_regions` — every `ChannelSite.region`, reconstructed into an index set, is
   the image of the corresponding `AccessMap` over that PE's iteration subdomain. This is the
   check that catches a wrong slice, the one class of bug an oracle alone cannot see.
2. `test_sem_compute_nodes` — the plan's `StoreNode`s, replayed over numpy with the plan's own
   channel protocol, reproduce the oracle.
3. `test_sem_coverage` — the union of the drained regions is exactly the kernel's write domain,
   with no overlap.

**Only a device run establishes end-to-end semantics**; that is the honest-limits slide, and it
is unchanged by these tests. They need no toolchain.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from spatial import m4_mapping as m4
from spatial.model import Expr
from tests.fixtures.mappings import w1_legal, w1flip_legal, w2_legal, w3_legal
from tests.helpers import plan_interp

TARGETS = ("npu1", "npu2")
M = N = K = 64
TM = TN = 32
TK = 16


def inputs(seed: int = 0) -> dict[str, np.ndarray]:
    """W1's fixture data: integer-valued `f32` in `[-8, 8)`, so `A @ B` is exact in `f32`."""
    rng = np.random.default_rng(seed)
    return {"A": rng.integers(-8, 8, (M, K)).astype(np.float32),
            "B": rng.integers(-8, 8, (K, N)).astype(np.float32),
            "C": np.zeros((M, N), dtype=np.float32)}


def _resolve(expr: Expr, env: dict[str, int], bindings: dict[str, int]) -> int:
    """One `Expr` over kernel axes and shape parameters."""
    return expr.const + sum(coeff * (env[name] if name in env else bindings[name])
                            for name, coeff in expr.coeffs)


def image(mapping, access, subdomain: dict[str, range]) -> set[tuple[int, ...]]:
    """`{ M_a·x + c_a : x in subdomain }` — the L3 elements one access touches.

    Only the axes with a non-zero column matter: an axis the access does not index contributes
    nothing to the image, which is exactly the statement `CLASSIFY` turns into multicast.
    """
    axes = [axis.name for axis in mapping.kernel.axes]
    bindings = dict(mapping.kernel.bindings)
    used = [position for position in range(len(axes))
            if any(row[position] for row in access.matrix)]
    out = set()
    for point in product(*(subdomain[axes[position]] for position in used)):
        env = dict(zip((axes[position] for position in used), point))
        out.add(tuple(_resolve(access.offsets[dim], env, bindings)
                      + sum(row[position] * env[axes[position]] for position in used)
                      for dim, row in enumerate(access.matrix)))
    return out


@pytest.mark.fr("FR-K1", "FR-M8")
@pytest.mark.parametrize("target", TARGETS)
def test_sem_access_regions(target):
    """Every L3 region is the `AccessMap` image of the iteration subdomain it claims to cover.

    W1's `A2L1` put at bundle index `pi`, K trip `kk`, covers rows `pi·32 …` and columns
    `kk …` — `03-lld-M4-mapping.md` §3.4's worked example, checked as an index set rather than
    as three tuples, so an off-by-one in offsets, sizes **or** strides shows up.
    """
    mapping = w1_legal.legal(target)
    plan = m4.plan(mapping)
    sites = {(c.name, s.kind): s for c in plan.channels for s in c.sites}
    reads = {access.operand: access for access in mapping.kernel.statements[0].reads}
    write = mapping.kernel.statements[0].target
    checked = 0
    for pi, kk in product(range(2), range(0, K, TK)):
        subdomain = {"i": range(pi * TM, (pi + 1) * TM), "j": range(N), "k": range(kk, kk + TK)}
        got = plan_interp.region_indices(sites[("A2L1", "put")].region,
                                         {"pi_bundle": pi, "k0": kk})
        assert got == image(mapping, reads["A"], subdomain), ("A2L1", pi, kk)
        subdomain = {"i": range(M), "j": range(pi * TN, (pi + 1) * TN), "k": range(kk, kk + TK)}
        got = plan_interp.region_indices(sites[("B2L1", "put")].region,
                                         {"pj_bundle": pi, "k0": kk})
        assert got == image(mapping, reads["B"], subdomain), ("B2L1", pi, kk)
        checked += 2
    for i, j in product(range(2), repeat=2):
        subdomain = {"i": range(i * TM, (i + 1) * TM), "j": range(j * TN, (j + 1) * TN),
                     "k": range(K)}
        got = plan_interp.region_indices(sites[("C2L3", "get")].region,
                                         {"i_drain": i, "j_drain": j})
        assert got == image(mapping, write, subdomain), ("C2L3", i, j)
        checked += 1
    assert checked == 20
    # the L1 end of a whole-buffer transfer is the empty region, which names no index set
    assert sites[("A2L1", "get")].region.offsets == ()


@pytest.mark.fr("FR-K1")
@pytest.mark.parametrize("target", TARGETS)
def test_sem_compute_nodes(target):
    """Interpreting W1's plan reproduces `A @ B` **exactly**, on both targets.

    `04-test-plan.md` §3.4 item 2 asks for a one-PE grid; this runs the real 2×2 fixture, which
    is strictly stronger — it exercises the fan-out, the four PEs' private accumulators and the
    drain, not only the arithmetic. The values are integer-valued `f32` in `[-8, 8)` and
    `K = 64`, so every partial sum is below `2**24` and `==` is the right comparison.
    """
    tensors = inputs()
    expected = tensors["A"] @ tensors["B"]
    out = plan_interp.run(m4.plan(w1_legal.legal(target)), tensors)
    assert np.array_equal(out["C"], expected)
    assert out["C"].dtype == np.float32


@pytest.mark.fr("FR-K1")
def test_sem_compute_nodes_is_the_plan_not_the_kernel():
    """A corrupted `StoreNode` changes the answer — the test above is not vacuous."""
    from dataclasses import replace

    from spatial.model import Const, Dtype

    plan = m4.plan(w1_legal.legal())
    loop = plan.herd_body[2].body[4].body[0].body[0]                 # the innermost k1 loop
    broken = replace(loop, body=(replace(loop.body[0],
                                         expr=Const(value=0.0, text="0.0",
                                                    dtype=Dtype.f32)),))
    nest = plan.herd_body[2].body[4]
    outer = replace(nest, body=(replace(nest.body[0], body=(broken,)),))
    corrupted = replace(plan, herd_body=plan.herd_body[:2] + (
        replace(plan.herd_body[2], body=plan.herd_body[2].body[:4] + (outer,)),)
        + plan.herd_body[3:])
    tensors = inputs()
    assert not np.array_equal(plan_interp.run(corrupted, tensors)["C"],
                              tensors["A"] @ tensors["B"])


@pytest.mark.fr("FR-K1", "FR-M8")
@pytest.mark.parametrize("target", TARGETS)
def test_sem_coverage(target):
    """The drained regions partition the kernel's write domain: no gap, no overlap.

    Two assertions, so a partial drain fails loudly rather than by accident (`04-test-plan.md`
    §3.4 item 3): the union of the drained index sets equals the domain the plan **claims** to
    drain, and that claim equals the kernel's full write domain.
    """
    mapping = w1_legal.legal(target)
    plan = m4.plan(mapping)
    get = next(s for c in plan.channels if c.name == "C2L3" for s in c.sites if s.kind == "get")
    drained: list[set[tuple[int, ...]]] = [
        plan_interp.region_indices(get.region, {"i_drain": i, "j_drain": j})
        for i, j in product(range(2), repeat=2)]
    union: set[tuple[int, ...]] = set().union(*drained)
    assert sum(len(part) for part in drained) == len(union) == M * N      # (a) no overlap
    write = mapping.kernel.statements[0].target
    domain = {axis.name: range(axis.extent) for axis in mapping.kernel.axes}
    assert union == image(mapping, write, domain)                         # (b) the whole domain


# --------------------------------------------------------------------------------------------
# W3 — the wavefront. Spec: design/04-test-plan.md §3.4, §4. Added by B at P4.
# --------------------------------------------------------------------------------------------

MQ = NR = 32
PJ, CW = 4, 8
MATCH, MISMATCH, GAP = 2, -1, 1


def w3_inputs(seed: int = 0) -> dict[str, np.ndarray]:
    """W3's fixture data: two `i32` sequences over a four-letter alphabet, and a zeroed `S`."""
    rng = np.random.default_rng(seed)
    return {"q": rng.integers(0, 4, MQ).astype(np.int32),
            "r": rng.integers(0, 4, NR).astype(np.int32),
            "S": np.zeros((MQ + 1, NR + 1), dtype=np.int32)}


def smith_waterman(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """The oracle of `04-test-plan.md` §4: a textbook two-loop DP, written here and nowhere else.

    It is deliberately **not** the kernel, not the plan and not derived from either: it is the
    definition of the recurrence FR-K4 names, in plain numpy, so "the plan computes
    Smith-Waterman" is a claim about two independent programs agreeing.
    """
    S = np.zeros((len(q) + 1, len(r) + 1), dtype=np.int32)
    for i in range(1, len(q) + 1):
        for j in range(1, len(r) + 1):
            S[i, j] = max(0, S[i - 1, j - 1] + (MATCH if q[i - 1] == r[j - 1] else MISMATCH),
                          S[i - 1, j] - GAP, S[i, j - 1] - GAP)
    return S


@pytest.mark.fr("FR-K4", "FR-M5")
@pytest.mark.parametrize("target", TARGETS)
def test_sem_compute_nodes_w3(target):
    """Interpreting W3's plan reproduces the textbook DP **exactly**, on both targets.

    Exactly, not to a tolerance: the scores are `i32` and every intermediate is an integer, so
    `tol = 0.0` is the right comparison (`04-test-plan.md` §4). The boundary row and column stay
    zero, which is what makes the source and the drain of ruling **R-W3-1** legitimate: `WestIn`
    puts column 0 and nothing ever writes it.
    """
    tensors = w3_inputs()
    expected = smith_waterman(tensors["q"], tensors["r"])
    out = plan_interp.run(m4.plan(w3_legal.legal(target)), tensors)
    assert np.array_equal(out["S"], expected)
    assert out["S"].dtype == np.int32
    assert not out["S"][0].any() and not out["S"][:, 0].any()
    assert out["S"][1:, 1:].any(), "an all-zero score matrix would pass vacuously"
    # the inputs are untouched: the plan reads `q` and `r` and writes only `S`
    fresh = w3_inputs()
    assert np.array_equal(out["q"], fresh["q"]) and np.array_equal(out["r"], fresh["r"])


@pytest.mark.fr("FR-K4")
def test_sem_compute_nodes_w3_is_the_plan_not_the_kernel():
    """A corrupted `StoreNode` changes the answer — the test above is not vacuous."""
    from dataclasses import replace

    from spatial.model import Const, Dtype

    plan = m4.plan(w3_legal.legal())
    row, compute = plan.herd_body[9], plan.herd_body[9].body[2]
    broken = replace(compute, body=(replace(compute.body[0],
                                            expr=Const(value=0, text="0", dtype=Dtype.i32)),))
    corrupted = replace(plan, herd_body=plan.herd_body[:9] + (
        replace(row, body=row.body[:2] + (broken,) + row.body[3:]),))
    tensors = w3_inputs()
    assert not np.array_equal(plan_interp.run(corrupted, tensors)["S"],
                              smith_waterman(*(w3_inputs()[k] for k in ("q", "r"))))


@pytest.mark.fr("FR-K1", "FR-M8")
@pytest.mark.parametrize("target", TARGETS)
def test_sem_access_regions_w3(target):
    """Every W3 L3 region is the `AccessMap` image of the subdomain it claims (§3.4 item 1).

    The four L3-attached channels: `QIn` stages the whole of `q` (it is indexed by `i` alone, so
    the image over any PE's subdomain is the whole vector), `RIn` stages PE `p`'s own slice of
    `r`, and `WestIn`/`EastOut` name single boundary cells of `S` that no PE writes through
    them.
    """
    mapping = w3_legal.legal(target)
    plan = m4.plan(mapping)
    sites = {(c.name, s.kind): s for c in plan.channels for s in c.sites}
    reads = {access.operand: access for access in mapping.kernel.statements[0].reads}
    for p in range(PJ):
        subdomain = {"i": range(1, MQ + 1), "j": range(1 + p * CW, 1 + (p + 1) * CW)}
        got = plan_interp.region_indices(sites[("QIn", "put")].region, {"pj_bundle": p})
        assert got == image(mapping, reads["q"], subdomain), ("QIn", p)
        got = plan_interp.region_indices(sites[("RIn", "put")].region, {"pj_bundle": p})
        assert got == image(mapping, reads["r"], subdomain), ("RIn", p)
    for i in range(1, MQ + 1):
        assert plan_interp.region_indices(sites[("WestIn", "put")].region,
                                          {"i_source": i}) == {(i, 0)}
        assert plan_interp.region_indices(sites[("EastOut", "get")].region,
                                          {"i_drain": i}) == {(i, NR)}
    # the L1 end of a whole-buffer transfer is the empty region; `SOut`'s is the owned band
    assert sites[("QIn", "get")].region.offsets == ()
    assert (sites[("SOut", "put")].region.sizes, sites[("SOut", "put")].region.strides) == (
        (CW,), (1,))


@pytest.mark.fr("FR-K4", "FR-M8")
@pytest.mark.parametrize("target", TARGETS)
def test_sem_coverage_w3(target):
    """`SOut` drains rows `1..MQ` × cols `1..NR` exactly: no gap, no overlap (§3.4 item 3).

    Draining only the last row of each PE — which is what the upstream-API probe of this shape
    does — would make (b) false by construction, and FR-K4 would have to be restated to compare
    one row against the oracle.
    """
    mapping = w3_legal.legal(target)
    plan = m4.plan(mapping)
    get = next(s for c in plan.channels if c.name == "SOut" for s in c.sites if s.kind == "get")
    drained = [plan_interp.region_indices(get.region, {"i_drain": i, "pj_bundle": p})
               for i, p in product(range(1, MQ + 1), range(PJ))]
    union: set[tuple[int, ...]] = set().union(*drained)
    assert sum(len(part) for part in drained) == len(union) == MQ * NR      # (a) no overlap
    write = mapping.kernel.statements[0].target
    domain = {"i": range(1, MQ + 1), "j": range(1, NR + 1)}
    assert union == image(mapping, write, domain)                           # (b) the whole domain
    # ...and the rows the drain covers are exactly the rows the interpreter writes
    tensors = w3_inputs()
    out = plan_interp.run(plan, tensors)
    assert set(map(tuple, np.argwhere(out["S"] != 0))) <= union


# --------------------------------------------------------------------------------------------
# W2 — the halo exchange. Spec: design/04-test-plan.md §3.4, §4, §8 item 5. Added by B at P5.
# --------------------------------------------------------------------------------------------

H = WIDTH = 16
HS, PI = 8, 2
TOL = 1e-5
"""`02-hld.md` §7.2: the update multiplies by `0.2`, so `f32` rounding is not associative and
W2's diff uses `tol = 1e-5` where W1, W1-flip and W3 are exact (`04-test-plan.md` §8 item 5)."""


def w2_inputs(T: int, seed: int = 0) -> dict[str, np.ndarray]:
    """W2's fixture data: plane 0 integer-valued `f32`, Dirichlet rows and columns included.

    **B-P25, the fixture invariant.** Planes `1..T` carry plane 0's boundary rows (`0`, `H+1`)
    and columns (`0`, `W-1`). The plan carries plane 0's boundary *forward* — the staged ghost
    rows at the domain edge and the seeded `next` are what every later plane reads there — while
    the kernel text reads plane `t`'s **own** rows `0`/`H+1` and columns `0`/`W-1` out of the
    input array. The two agree exactly when the input's boundary is the same in every plane,
    which is what "read-only Dirichlet boundary" (`02-hld.md` §7.2) means for a one-array
    kernel: a time-invariant boundary. Any other input makes the plan and the kernel text
    compute different things, so Person C's `make_fixture.py` must satisfy this too.
    """
    U = np.zeros((T + 1, H + 2, WIDTH), dtype=np.float32)
    U[0] = np.random.default_rng(seed).integers(-8, 8, (H + 2, WIDTH)).astype(np.float32)
    U[1:, 0], U[1:, H + 1] = U[0, 0], U[0, H + 1]                       # the Dirichlet rows
    U[1:, :, 0], U[1:, :, WIDTH - 1] = U[0, :, 0], U[0, :, WIDTH - 1]   # ...and columns
    return {"U": U}


def jacobi(U: np.ndarray, T: int) -> np.ndarray:
    """The oracle of `04-test-plan.md` §4: a two-loop numpy Jacobi, written here and nowhere else.

    It is deliberately not the kernel and not the plan. **The boundary is carried forward**: rows
    `0`/`H+1` and columns `0`/`W-1` are the read-only Dirichlet boundary (`02-hld.md` §7.2), so
    plane `t+1` starts as plane `t` and only the interior is recomputed. That is the same fact
    §6.3's coverage paragraph states from the other side — "the value written back is the one
    that was staged in" — and it is why the plan seeds **both** halves of the swap pair.
    """
    E = U.copy()
    for t in range(T):
        E[t + 1] = E[t]
        for i in range(1, H + 1):
            for j in range(1, WIDTH - 1):
                E[t + 1, i, j] = np.float32(0.2) * (E[t, i, j] + E[t, i - 1, j] + E[t, i + 1, j]
                                                    + E[t, i, j - 1] + E[t, i, j + 1])
    return E


@pytest.mark.fr("FR-K3", "FR-M4")
@pytest.mark.parametrize("T", [4, 5])
@pytest.mark.parametrize("target", TARGETS)
def test_sem_compute_nodes_w2(target, T):
    """Interpreting W2's plan reproduces the two-loop Jacobi within `1e-5`, both parities.

    The interpreter's round-robin scheduler is what makes put-before-get progress: at `t = 0`
    PE 0 puts its south boundary and then **blocks** on its north ghost get until PE 1 has put
    — which PE 1 does before blocking on its own. A `deadlock` here would mean the plan is
    wrong, never that the scheduler needs fixing.
    """
    tensors = w2_inputs(T)
    expected = jacobi(tensors["U"], T)
    out = plan_interp.run(m4.plan(w2_legal.legal(target, T=T)), tensors)["U"]
    assert out.dtype == np.float32
    error = np.abs(out[1:, 1:H + 1, 1:WIDTH - 1] - expected[1:, 1:H + 1, 1:WIDTH - 1]).max()
    assert error <= TOL, f"max abs error {error} over planes 1..{T}"
    assert np.abs(out[1:, 1:H + 1, 1:WIDTH - 1]).max() > 0, "an all-zero result would pass"
    # the read-only columns travel with the strip: every drained plane carries plane 0's
    plane0 = w2_inputs(T)["U"][0]
    for column in (0, WIDTH - 1):
        assert np.array_equal(out[1:, 1:H + 1, column],
                              np.broadcast_to(plane0[1:H + 1, column], (T, H)))
    # rows 0 and H+1 are never drained — the drain covers rows 1..H — so they stay as the input
    # left them, which under B-P25 is plane 0's Dirichlet boundary
    assert np.array_equal(out[1:, 0], np.broadcast_to(plane0[0], (T, WIDTH)))
    assert np.array_equal(out[1:, H + 1], np.broadcast_to(plane0[H + 1], (T, WIDTH)))
    assert np.array_equal(out[0], plane0), "plane 0 is read-only and is never written back"


@pytest.mark.fr("FR-K3")
def test_sem_compute_nodes_w2_is_the_plan_not_the_kernel():
    """A corrupted `StoreNode` changes the answer — the test above is not vacuous."""
    from dataclasses import replace

    from spatial.model import Const, Dtype

    plan = m4.plan(w2_legal.legal())
    loop = plan.herd_body[4]
    nest = loop.body[4]
    broken = replace(nest, body=(replace(nest.body[0], body=(
        replace(nest.body[0].body[0], expr=Const(value=0.0, text="0.0", dtype=Dtype.f32)),)),))
    corrupted = replace(plan, herd_body=plan.herd_body[:4] + (
        replace(loop, body=loop.body[:4] + (broken,) + loop.body[5:]),))
    out = plan_interp.run(corrupted, w2_inputs(4))["U"]
    expected = jacobi(w2_inputs(4)["U"], 4)
    assert not np.allclose(out[1:, 1:H + 1, 1:WIDTH - 1],
                           expected[1:, 1:H + 1, 1:WIDTH - 1], atol=TOL)


@pytest.mark.fr("FR-K3", "FR-M8")
@pytest.mark.parametrize("target", TARGETS)
def test_sem_access_regions_w2(target):
    """Every W2 L3 region is the `AccessMap` image of the subdomain it claims (§3.4 item 1).

    `UIn` stages PE `p`'s owned rows **plus both ghost rows** of plane 0 — the union of the read
    accesses' images over that PE's rows, which is what makes `t = 0`'s first update correct with
    no prologue put. `UOut` drains the write access's image over the same rows at plane `t+1`,
    widened to the whole staged width because the strip is put back whole (§6.3).
    """
    mapping = w2_legal.legal(target)
    plan = m4.plan(mapping)
    sites = {(c.name, s.kind): s for c in plan.channels for s in c.sites if s.scope == "segment"}
    reads = mapping.kernel.statements[0].reads
    write = mapping.kernel.statements[0].target
    for p in range(PI):
        rows = range(1 + p * HS, 1 + (p + 1) * HS)
        subdomain = {"t": range(1), "i": rows, "j": range(1, WIDTH - 1)}
        staged = plan_interp.region_indices(sites[("UIn", "put")].region, {"pi_bundle": p})
        assert staged == set().union(*(image(mapping, access, subdomain) for access in reads)) \
            | {(0, i, j) for i in (rows[0] - 1, rows[-1] + 1) for j in (0, WIDTH - 1)} \
            | {(0, i, j) for i in rows for j in (0, WIDTH - 1)}
        for t in range(4):
            drained = plan_interp.region_indices(sites[("UOut", "get")].region,
                                                 {"t_drain": t, "pi_bundle": p})
            written = image(mapping, write, {"t": range(t, t + 1), "i": rows,
                                             "j": range(1, WIDTH - 1)})
            assert written < drained, "the strip is put whole, so the columns are a superset"
            assert drained - written == {(t + 1, i, j) for i in rows for j in (0, WIDTH - 1)}
    assert sites[("UIn", "put")].region.sizes == (1, HS + 2, WIDTH)
    assert sites[("UOut", "get")].region.sizes == (1, HS, WIDTH)
    # the L1 end of a whole-strip transfer is the empty region
    assert next(s for c in plan.channels if c.name == "UIn" for s in c.sites
                if s.scope == "herd").region.offsets == ()


@pytest.mark.fr("FR-K3", "FR-M8")
@pytest.mark.parametrize("T", [4, 5])
@pytest.mark.parametrize("target", TARGETS)
def test_sem_coverage_w2(target, T):
    """The drained regions partition planes `1..T` × rows `1..H`: no gap, no overlap (§3.4 item 3).

    Two assertions, and the second is where W2 differs from W1 and W3: the plan's **claimed**
    drain domain is planes `1..T` × rows `1..H` × **all** `W` columns, and the kernel's write
    domain is the same box narrowed to columns `1..W-2`. The claim is a strict superset in the
    column direction only, because the strip is put back whole and the two boundary columns carry
    back the values that were staged in — `03-lld-M4-mapping.md` §6.3's coverage paragraph is
    that sentence, and `test_sem_compute_nodes_w2` checks the values themselves.
    """
    mapping = w2_legal.legal(target, T=T)
    plan = m4.plan(mapping)
    get = next(s for c in plan.channels if c.name == "UOut" for s in c.sites
               if s.scope == "segment")
    drained = [plan_interp.region_indices(get.region, {"t_drain": t, "pi_bundle": p})
               for t in range(T) for p in range(PI)]
    union: set[tuple[int, ...]] = set().union(*drained)
    claimed = {(t, i, j) for t in range(1, T + 1) for i in range(1, H + 1)
               for j in range(WIDTH)}
    assert sum(len(part) for part in drained) == len(union) == len(claimed)   # (a) no overlap
    assert union == claimed
    write = mapping.kernel.statements[0].target
    domain = {"t": range(T), "i": range(1, H + 1), "j": range(1, WIDTH - 1)}
    written = image(mapping, write, domain)
    assert written < union, (                                                # (b) the claim
        "the claimed drain domain is a strict superset of the kernel's write domain in the "
        "column direction only: the strip is put back whole, so columns 0 and W-1 of every "
        "drained plane carry back the staged read-only boundary (03-lld-M4-mapping.md §6.3)")
    assert union - written == {(t, i, j) for t in range(1, T + 1) for i in range(1, H + 1)
                               for j in (0, WIDTH - 1)}


# --------------------------------------------------------------------------------------------
# W1-flip — the cascade. Spec: design/04-test-plan.md §3.4, §8 item 5. Added by B at P6.
# --------------------------------------------------------------------------------------------

PK = 4


@pytest.mark.fr("FR-K2", "FR-M6")
@pytest.mark.parametrize("target", TARGETS)
def test_sem_compute_nodes_flip(target):
    """Interpreting the flip's plan reproduces `A @ B` **exactly**, on both targets.

    Same kernel, same data, a different dataflow: each PE owns a `k`-slice, accumulates its own
    partial `[32,64]` tile and hands it up the chain, and only `tx == PK-1` reaches L3. An
    off-by-one in the chain would show as a missing or doubled `k`-slice, which `==` catches
    because the values are integer-valued `f32` in `[-8, 8)` (`04-test-plan.md` §8 item 5).
    """
    tensors = inputs()
    expected = tensors["A"] @ tensors["B"]
    out = plan_interp.run(m4.plan(w1flip_legal.legal(target)), tensors)
    assert np.array_equal(out["C"], expected)
    assert out["C"].dtype == np.float32


@pytest.mark.fr("FR-K2", "FR-M6")
def test_sem_compute_nodes_flip_2d():
    """The 2-D descending variant computes the same product — the orientation is not the maths.

    It is a **test fixture**, not a demo artifact: `grid(1, 4)` with `place(px=ax.i0,
    py=ax.k0)`, `i0` of extent 1, no `double_buffer` and no temporal tile axis at all, so the
    whole `[64,64]` accumulator is resident and the chain descends in `ty`.
    """
    tensors = inputs()
    out = plan_interp.run(m4.plan(w1flip_legal.legal(grid2d=True)), tensors)
    assert np.array_equal(out["C"], tensors["A"] @ tensors["B"])


@pytest.mark.fr("FR-K2")
def test_sem_compute_nodes_flip_is_the_plan_not_the_kernel():
    """Dropping the cascade accumulate changes the answer — the test above is not vacuous.

    The corruption is the one that matters for FR-M6: keep every transfer, but throw the
    received partial tile away instead of adding it. The chain then delivers only the tail PE's
    own `k`-slice, and `C` is a quarter of the product.
    """
    from dataclasses import replace

    plan = m4.plan(w1flip_legal.legal())
    loop = plan.herd_body[-1]
    branch = loop.body[-1]
    get, accumulate, inner = branch.otherwise
    broken = replace(branch, otherwise=(get, inner))          # the accumulate nest is gone
    corrupted = replace(plan, herd_body=plan.herd_body[:-1] + (
        replace(loop, body=loop.body[:-1] + (broken,)),))
    tensors = inputs()
    out = plan_interp.run(corrupted, tensors)["C"]
    assert not np.array_equal(out, tensors["A"] @ tensors["B"])
    assert np.array_equal(out, tensors["A"][:, 3 * TK:] @ tensors["B"][3 * TK:, :])


@pytest.mark.fr("FR-K2", "FR-M8")
@pytest.mark.parametrize("target", TARGETS)
def test_sem_access_regions_flip(target):
    """Every L3 region of the flip is the `AccessMap` image of the subdomain it claims.

    `A2L1[pk]` at `i0` covers rows `i0 … i0+32` and the PE's own `k`-slice; `B2L1[pk]` covers
    that `k`-slice against **all** `N` columns, which is the whole of RULING 9 as an index set;
    `C2L3[0]` at `i0` covers the row-block the tail PE drained.
    """
    mapping = w1flip_legal.legal(target)
    plan = m4.plan(mapping)
    sites = {(c.name, s.kind): s for c in plan.channels for s in c.sites}
    reads = {access.operand: access for access in mapping.kernel.statements[0].reads}
    write = mapping.kernel.statements[0].target
    checked = 0
    for pk, i0 in product(range(PK), range(0, M, TM)):
        subdomain = {"i": range(i0, i0 + TM), "j": range(N),
                     "k": range(pk * TK, (pk + 1) * TK)}
        got = plan_interp.region_indices(sites[("A2L1", "put")].region,
                                         {"pk_bundle": pk, "i0": i0})
        assert got == image(mapping, reads["A"], subdomain), ("A2L1", pk, i0)
        got = plan_interp.region_indices(sites[("B2L1", "put")].region, {"pk_bundle": pk})
        assert got == image(mapping, reads["B"], subdomain), ("B2L1", pk)
        got = plan_interp.region_indices(sites[("C2L3", "get")].region, {"i0_drain": i0})
        assert got == image(mapping, write, subdomain), ("C2L3", i0)
        checked += 3
    assert checked == 24
    # the cascade's payload is the whole accumulator: an empty region names no index set
    assert all(s.region.offsets == () for s in sites_of(plan, "CascadeK"))


def sites_of(plan, channel):
    """Every site of one channel of a plan, in plan order."""
    return next(c for c in plan.channels if c.name == channel).sites


@pytest.mark.fr("FR-K2", "FR-M8")
@pytest.mark.parametrize("target", TARGETS)
def test_sem_coverage_flip(target):
    """The tail PE's two drained row-blocks partition `C`: no gap, no overlap."""
    mapping = w1flip_legal.legal(target)
    plan = m4.plan(mapping)
    get = next(s for s in sites_of(plan, "C2L3") if s.kind == "get")
    drained = [plan_interp.region_indices(get.region, {"i0_drain": i0})
               for i0 in range(0, M, TM)]
    union: set[tuple[int, ...]] = set().union(*drained)
    assert sum(len(part) for part in drained) == len(union) == M * N      # (a) no overlap
    write = mapping.kernel.statements[0].target
    domain = {axis.name: range(axis.extent) for axis in mapping.kernel.axes}
    assert union == image(mapping, write, domain)                         # (b) the whole domain
