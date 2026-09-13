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
from tests.fixtures.mappings import w1_legal
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
