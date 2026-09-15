"""Gate G2 end to end from the **live surface**: `kernels/*.py` → the committed goldens.

Every other golden test starts from a hand-written `LegalMapping` literal or from B's plan
fixtures. This one starts where a user starts — a `@sp.kernel` function and a `schedule()`
in `kernels/` — and asserts M1→M2→M3→M4→M5 reproduces the same bytes. It is the test
`design/PROGRESS-B.md`'s "load-bearing caveat" asks for: *no test here proves the legality
checker will produce them*, said of the four `LegalMapping` literals the whole suite rests on.

Three artefacts are compared per (variant, target):

* `*.air.mlir` — byte for byte;
* `*.summary.txt` — byte for byte;
* `*.plan.json` — after normalising the **two fields that record where the kernel text sits**,
  and nothing else. `KernelModel.source` is the captured text: the fixtures spell it as a
  `SOURCE` string that opens with a parameter line (`M = N = K = 64`) which
  `inspect.getsource` never sees, so the live capture can never equal it. `Statement.line` is
  a line number **into that text**, so it moves with it. Both are set aside here and every
  other field — params, axes, dependences, σ/π, the plan tree, the channels — is compared as
  canonical JSON.

`w1_gemm_bf16` has no golden (`03-lld-M8-kernels-demo.md` §4 gives `w1_large` levels I and S
only). It **is** emittable since B-P32 and the three tests at the foot of this module are what
that item rests on; `aircc` still refuses it, for its **grid** and not for its dtypes — and the
passing control at `grid(1, 2)` compiles the same `bf16`→`f32` cast on both generations
(B-P33, B-P34).

The TT test at the foot is the same claim as the golden ones for the **second** backend: the
`MappingPlan` is backend-neutral, so the plan the live surface builds must produce the very same
`TTProgram` as the hand-written `LegalMapping` literal does. It needs no device — `m5tt_emit`
imports nothing but the standard library — so it runs in the default suite.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import spatial as sp
from kernels import w1_gemm as w1
from kernels import w1_gemm_bf16 as w1_large
from kernels import w2_jacobi as w2
from kernels import w3_sw as w3
from spatial import m3_legality, m4_mapping as m4
from spatial import m5_emit, m5tt_emit, m6_tools as m6
from spatial.model import LegalityError, MappingError, ToolchainError, to_json
from tests.fixtures.mappings import w1_legal, w1flip_legal, w2_legal, w3_legal
from tests.helpers import plan_interp
from tests.helpers.golden import GOLDEN_DIR

TARGETS = ("npu1", "npu2")

VARIANTS = {
    "w1.base": w1.schedule_os,
    "w1.flip": w1.schedule_ws,
    "w2.base": w2.schedule,
    "w3.base": w3.schedule,
}
"""Golden stem → the `kernels/` entry point that must reproduce it."""


def require_pin() -> None:
    """Skip — never fail — when the installed toolchain is not the pinned one (§8 rule 2)."""
    try:
        m6.check_pin()
    except ToolchainError as exc:
        pytest.skip(f"{exc.diagnostic.reason}: {exc.diagnostic.details['mismatched']}")


def _without_source_positions(obj):
    """Drop `source` and blank every `line`, at any depth. See the module docstring."""
    if isinstance(obj, dict):
        return {k: (0 if k == "line" else _without_source_positions(v))
                for k, v in obj.items() if k != "source"}
    if isinstance(obj, list):
        return [_without_source_positions(x) for x in obj]
    return obj


@pytest.mark.parametrize("stem", sorted(VARIANTS))
@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.fr("FR-E7")
def test_live_kernel_emits_the_golden_text(stem, target):
    """`kernels.<w>.schedule(target).mlir()` is the committed `*.air.mlir`, byte for byte."""
    require_pin()
    actual = VARIANTS[stem](target).mlir()
    expected = (GOLDEN_DIR / f"{stem}.{target}.air.mlir").read_text(encoding="utf-8")
    assert actual == expected


@pytest.mark.parametrize("stem", sorted(VARIANTS))
@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.fr("FR-M11")
def test_live_kernel_renders_the_golden_summary(stem, target):
    """`.summary()` is the committed `*.summary.txt`, byte for byte."""
    require_pin()
    actual = VARIANTS[stem](target).summary() + "\n"
    expected = (GOLDEN_DIR / f"{stem}.{target}.summary.txt").read_text(encoding="utf-8")
    assert actual == expected


@pytest.mark.parametrize("stem", sorted(VARIANTS))
@pytest.mark.parametrize("target", TARGETS)
def test_live_kernel_plans_the_golden_plan(stem, target):
    """`.plan()` is the committed `*.plan.json`, modulo where the kernel text sits."""
    require_pin()
    actual = json.loads(to_json(VARIANTS[stem](target).plan()))
    expected = json.loads((GOLDEN_DIR / f"{stem}.{target}.plan.json").read_text(encoding="utf-8"))
    assert _without_source_positions(actual) == _without_source_positions(expected)


# --------------------------------------------------------------------------------------------
# The second backend: one plan, two emitters
# --------------------------------------------------------------------------------------------

TT_VARIANTS = {
    "w1.base": (w1.schedule_os, w1_legal.legal),
    "w1.flip": (w1.schedule_ws, w1flip_legal.legal),
    "w2.base": (w2.schedule, w2_legal.legal),
    "w3.base": (w3.schedule, w3_legal.legal),
}
"""Golden stem → (the live `kernels/` entry point, the `LegalMapping` literal it must match).

Both sides are read at `"npu1"`, which is the AIR target the fixtures are spelled at; the TT
emitter takes no target at all, because `"npu1"`/`"npu2"` names an AIE generation and means
nothing on a Tensix core (`spatial/m5tt_emit.py::emit`)."""


@pytest.mark.parametrize("stem", sorted(TT_VARIANTS))
@pytest.mark.fr("FR-TT1")
def test_live_plan_emits_the_same_tt_program(stem):
    """The live surface and the fixture literal produce the **same** `TTProgram`.

    `TTProgram` is a plain frozen dataclass, so `==` is structural and compares the kernel C++,
    the core range, every circular buffer, every semaphore, every io tensor and the per-core
    runtime-arg ABI at once. This is the AIE goldens' claim for the second backend, and it is
    what lets the demo's Tenstorrent beat stand on the same `schedule()` the pitch has been
    showing: the plan is backend-neutral, so neither emitter can be reading the surface.
    """
    live, fixture = TT_VARIANTS[stem]
    assert m5tt_emit.emit(live("npu1").plan()) == m5tt_emit.emit(m4.plan(fixture("npu1")))


# --------------------------------------------------------------------------------------------
# W1-large — bf16 in, f32 out (B-P32)
# --------------------------------------------------------------------------------------------

CONTROL_M = CONTROL_N = CONTROL_K = 256
CONTROL_TK = 32
"""The f32 control's `k` tile. `w1_gemm_bf16` tiles `k` by 64 at two bytes an element; 32 at
four bytes gives the control the **same 49 152 B** of L1 and the same 4×4 grid, so the only
thing that differs between the two `aircc` runs below is the element type."""


@sp.kernel
def gemm_f32_large(A: sp.f32[CONTROL_M, CONTROL_K], B: sp.f32[CONTROL_K, CONTROL_N],
                   C: sp.f32[CONTROL_M, CONTROL_N]):
    for i in range(CONTROL_M):
        for j in range(CONTROL_N):
            for k in range(CONTROL_K):
                C[i, j] += A[i, k] * B[k, j]


def _control(target):
    """`w1_gemm_bf16._schedule_os_clauses` clause for clause, in f32 — so no cast is emitted."""
    s = sp.schedule(gemm_f32_large, target=target)
    ax = s.axes()
    s.grid(w1_large.PI, w1_large.PJ)
    s.tile(ax.i, w1_large.TM)
    s.tile(ax.j, w1_large.TN)
    s.tile(ax.k, CONTROL_TK)
    s.reduce(ax.k, op="+")
    s.place(px=ax.i0, py=ax.j0)
    s.stationary("C")
    s.reside(A="L1", B="L1", C="L1")
    s.double_buffer("A", "B")
    s.pipeline(ax.k0)
    return s


def _aircc_verdict(text, target, tmp_path, stem):
    """`aircc --output-format=none` on `text`: `(None, ())` for exit 0, else code + diagnostics."""
    path = tmp_path / f"{stem}.{target}.mlir"
    path.write_text(text, encoding="utf-8")
    try:
        m6.artifact(str(path), target, "none", workdir=str(tmp_path))
    except ToolchainError as exc:
        # `aiecc` segfaults after printing some diagnostics (B-P36), so `aircc` exits with no
        # parsable `error:` line and the text is only in the captured tail.
        details = exc.diagnostic.details
        return exc.diagnostic.code, details.get("diagnostics") or details.get("stderr_tail", ())
    return None, ()


@pytest.mark.parametrize("target", TARGETS)
def test_live_bf16_gemm_widens_every_load_to_the_accumulator(target):
    """B-P32: the `bf16` GEMM emits one `arith.extf` per `bf16` operand.

    Two, not one: the ruling is that each **load** is cast to the destination's dtype before
    the arithmetic, so the product and the accumulate are both f32 — which is what "bf16 in,
    f32 out" means. A cast around the product would be one `extf` and a bf16 multiply.

    Read at `_fits`'s `grid(1, 2)`, not at W1-large's own `grid(4, 4)`: since **R-HERD-1** and
    **R-L1-3** (2026-09-15) the checker refuses the 4×4 shape before anything is emitted, which
    is the test below. The clauses, the tiles and the dtypes here are `w1_gemm_bf16`'s.
    """
    require_pin()
    text = _fits(target).mlir()
    assert text.count("arith.extf") == 2, "one widening per bf16 operand"
    assert "truncf" not in text, "nothing narrows back: the accumulator stays f32"
    assert "bf16" in text and "f32" in text


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.fr("FR-L9", "FR-L10")
def test_live_bf16_gemm_shape_is_refused_by_the_checker(target):
    """W1-large's 4×4 grid is refused **before codegen**, by a different code per target.

    The fixture's shape is fixed at 256³ / 4×4 / `TM = TN = TK = 64` by
    `03-lld-M8-kernels-demo.md` §4. At that shape the logical grid fits neither physical herd —
    `(1, 4)` on npu1, `(2, 4)` on npu2 — so M4 would emit a **repeat loop**, and both rulings
    of 2026-09-15 bite on it:

    * **npu1, `repeats == (4, 1)`** — `HERD-PHYSICAL` (R-HERD-1). The repeat loop is unrolled
      by 2, so a trip-4 loop leaves a trip-2 loop whose induction variable is still a channel
      bundle index; `air-to-aie` answered *"'air.channel.get' op failed to get MM2S tile for L3
      allocation"* (B-P33). Measured again this pass with the ruling monkeypatched off, below.
    * **npu2, `repeats == (2, 1)`** — `L1-CAPACITY` (R-L1-3). Every buffer is allocated twice:
      2 × (16 384 + 8 192 + 8 192) = **65 536** B against the 63 488 B budget, which is exactly
      the six `aie.buffer`s per tile `AIEAssignBuffers` refused (B-P34).

    Before the rulings both shapes passed the checker and died inside `aircc`; this is the
    whole of D3/D4 in one assertion.
    """
    with pytest.raises(LegalityError) as caught:
        w1_large.schedule_os(target).check()
    diagnostic = caught.value.diagnostic
    assert diagnostic.code == {"npu1": "HERD-PHYSICAL", "npu2": "L1-CAPACITY"}[target]
    if target == "npu1":
        assert diagnostic.details["repeats"] == (4, 1)
        assert diagnostic.details["unroll_factor"] == 2
    else:
        assert (diagnostic.details["total"], diagnostic.details["budget"]) == (65536, 63488)
        assert diagnostic.details["repeats"] == (2, 1)


@pytest.mark.slow
@pytest.mark.requires_aircc
@pytest.mark.parametrize("target", TARGETS)
def test_live_bf16_gemm_is_refused_by_aircc_for_its_shape(target, tmp_path, capsys,
                                                          monkeypatch):
    """The checker's refusal above is right: `aircc` refuses the same module, cast or no cast.

    R-HERD-1 and R-L1-3 are switched off for the length of this test — `_check_repeats` to a
    no-op and every L1 budget to a number nothing reaches — so the module the checker now
    refuses can still be built and handed to `aircc`. That is the evidence the rulings rest
    on, and it is re-measured here rather than quoted:

    * npu1 (`repeats (4, 1)`): `air-to-aie` fails the launch-side `C2L3` get.
    * npu2 (`repeats (2, 1)`): `AIEAssignBuffers` refuses six buffers in a 64 KB tile.

    The **f32 control** at the same grid, the same 49 152 B of pre-R-L1-3 L1 and zero casts
    draws the same verdict, which is what makes the refusal a statement about the shape and
    not about B-P32's cast. If a later toolchain compiles W1-large this fails on the control
    too, and both rulings are re-opened.
    """
    require_pin()
    from spatial import m4_selfcheck

    monkeypatch.setattr(m3_legality, "_check_repeats", lambda *args, **kw: None)
    monkeypatch.setattr(m3_legality, "_L1_USABLE", 1 << 24)
    monkeypatch.setattr(m4, "L1_USABLE", 1 << 24)
    monkeypatch.setattr(m4_selfcheck, "L1_USABLE", 1 << 24)
    # B-P36's rule refuses this shape too — five fill flows into a four-row column — but the
    # two verdicts below are both raised *before* the packet router runs, so it is the rulings
    # under test here and P3's master-select count is switched off with them.
    monkeypatch.setattr(m4_selfcheck, "msels", lambda plan: None)
    control = _control(target).mlir()
    assert control.count("arith.extf") == 0, "a cast-free control, or it proves nothing"
    bf16_code, bf16_said = _aircc_verdict(w1_large.schedule_os(target).mlir(),
                                          target, tmp_path, "bf16")
    control_code, control_said = _aircc_verdict(control, target, tmp_path, "f32")
    with capsys.disabled():
        print(f"\n  aircc --device {target} bf16: {bf16_code or 'exit 0'} {bf16_said[:1]}"
              f"\n  aircc --device {target} f32 control: {control_code or 'exit 0'} "
              f"{control_said[:1]}")
    assert bf16_code == control_code, (
        f"bf16 and the cast-free f32 control get different aircc verdicts on {target} "
        f"({bf16_code} vs {control_code}); the dtype decides something after all")
    assert bf16_code is not None, (
        f"aircc now compiles the shape the checker refuses on {target}; R-HERD-1 / R-L1-3 "
        f"must be re-measured")


# --------------------------------------------------------------------------------------------
# The passing control — the same tiles and the same L1, one grid that fits the herd
# --------------------------------------------------------------------------------------------

FITS_M, FITS_N, FITS_K = 64, 128, 256
"""The control's shape. `M // TM == 1` and `N // TN == 2` give `grid(1, 2)`, which is inside
both physical herds (`(1, 4)` on npu1, `(2, 4)` on npu2), so `repeats == (1, 1)` and M4 emits no
repeat loop. Everything else is `w1_gemm_bf16`'s: `TM = TN = TK = 64`, `bf16` in and `f32` out,
the same nine clauses, and the same **49 152 B** of L1."""


@sp.kernel
def gemm_bf16_fits(A: sp.bf16[FITS_M, FITS_K], B: sp.bf16[FITS_K, FITS_N],
                   C: sp.f32[FITS_M, FITS_N]):
    for i in range(FITS_M):
        for j in range(FITS_N):
            for k in range(FITS_K):
                C[i, j] += A[i, k] * B[k, j]


def _fits(target):
    """`w1_gemm_bf16._schedule_os_clauses` clause for clause, at `grid(1, 2)`."""
    s = sp.schedule(gemm_bf16_fits, target=target)
    ax = s.axes()
    s.grid(FITS_M // w1_large.TM, FITS_N // w1_large.TN)
    s.tile(ax.i, w1_large.TM)
    s.tile(ax.j, w1_large.TN)
    s.tile(ax.k, w1_large.TK)
    s.reduce(ax.k, op="+")
    s.place(px=ax.i0, py=ax.j0)
    s.stationary("C")
    s.reside(A="L1", B="L1", C="L1")
    s.double_buffer("A", "B")
    s.pipeline(ax.k0)
    return s


@pytest.mark.slow
@pytest.mark.requires_aircc
@pytest.mark.parametrize("target", TARGETS)
def test_live_bf16_gemm_compiles_when_the_grid_fits_the_herd(target, tmp_path, capsys):
    """The control for the test above: **`aircc` exit 0, zero `error:` lines, both targets.**

    One parameter moves — the grid, `(4, 4)` → `(1, 2)`. The tiles, the dtypes, the clauses and
    the 49 152 B of L1 are W1-large's, and the module still carries B-P32's two `arith.extf`, so
    this compiles the *same cast* that W1-large does. What it does not carry is a repeat loop:
    `grid(1, 2)` fits both physical herds, so `repeats == (1, 1)`, the `C2L3` bundle index is
    constant per core and each core allocates one accumulator, not two.

    Together with the refusal above this is the whole of B-P33/B-P34's claim: at this shape our
    `l1_bytes` (49 152) is identical on both sides of the line, so **`l1_bytes` is not what
    decides** — `repeats` is, and M3/M4 do not charge it. Measured per tile in the lowered IR:
    five buffers, 49 152 B, against six and 65 536 B at `repeats == (2, 1)`.
    """
    require_pin()
    schedule = _fits(target)
    # 49 152 B: the tiles W1-large carries, charged once each because this grid fits the herd.
    # W1-large's own 4×4 grid charged the same 49 152 until R-L1-3 made it 65 536 and refused
    # it; `check()` on it now raises, which is why the equality is spelled out here instead.
    assert schedule.check().l1_bytes == 49152
    assert schedule.check().repeats == (1, 1), "the control's grid must fit the herd"
    text = schedule.mlir()
    assert text.count("arith.extf") == 2, "the control compiles B-P32's cast, not a cast-free IR"
    code, said = _aircc_verdict(text, target, tmp_path, "fits")
    with capsys.disabled():
        print(f"\n  aircc --device {target} bf16 grid(1,2): {code or 'exit 0'} {said[:1]}")
    assert code is None, f"the control must compile on {target}; aircc said {code}: {said[:2]}"


# --------------------------------------------------------------------------------------------
# Degenerate grids — one PE, and a 1-D grid over a one-tile axis (defects D1 and D2)
# --------------------------------------------------------------------------------------------

SMALL = 64
"""The degenerate-grid GEMM's extent. 64³, so a `grid(1)` schedule tiles `i` and `j` by the
whole extent and still fits: `acc [64,64] f32` 16 384 + `a [64,16]` 4 096 ×2 + `b [16,64]`
4 096 ×2 = 32 768 B, inside the 63 488 B budget."""


@sp.kernel
def gemm_small(A: sp.f32[SMALL, SMALL], B: sp.f32[SMALL, SMALL], C: sp.f32[SMALL, SMALL]):
    for i in range(SMALL):
        for j in range(SMALL):
            for k in range(SMALL):
                C[i, j] += A[i, k] * B[k, j]


def _degenerate(grid, tiles, target):
    """W1's nine clauses at an arbitrary grid and tiling — the only two things that move."""
    s = sp.schedule(gemm_small, target=target)
    ax = s.axes()
    s.grid(*grid)
    s.tile(ax.i, tiles[0]); s.tile(ax.j, tiles[1]); s.tile(ax.k, tiles[2])
    s.reduce(ax.k, op="+")
    if len(grid) == 2:
        s.place(px=ax.i0, py=ax.j0)
    else:
        s.place(px=ax.i0)
    s.stationary("C")
    s.reside(A="L1", B="L1", C="L1")
    s.double_buffer("A", "B")
    s.pipeline(ax.k0)
    return s


DEGENERATE = {
    "grid(1,1)": ((1, 1), (64, 64, 16)),      # D1: one PE, no bundle loop anywhere
    "grid(1,)": ((1,), (64, 64, 16)),         # D2: 1-D, and `j` is one whole tile
    "grid(2,)": ((2,), (32, 64, 16)),         # D2: 1-D over two PEs, `j` still one tile
}
"""The three shapes defects D1 and D2 were measured on (`design/PROGRESS-B.md` B-P35).

`grid(1, 1)` tripped M4's own `order` bookkeeping: with every bundle extent 1 the drain builds
no loop, so its `get` sits in the segment body itself and its `order` is its index **there**,
not 0. `grid(1,)` and `grid(2,)` leave `j` tiled by its whole extent, which used to count as a
temporal axis that "moves" `B` — two staged operands re-fetched per different axes, and one
streaming loop to put them in."""


def _sites_in_order(body, out):
    """Every `ChannelSite.order` against its index in the body holding it (§5.2), recursively."""
    from spatial.model import ChannelSite, LoopPlan

    for index, node in enumerate(body):
        if isinstance(node, ChannelSite):
            out.append((node.id, node.order, index))
        if isinstance(node, LoopPlan):
            _sites_in_order(node.body, out)
    return out


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("shape", sorted(DEGENERATE), ids=sorted(DEGENERATE))
@pytest.mark.fr("FR-M7", "FR-M9")
def test_degenerate_grid_plans_and_self_checks(shape, target):
    """Each degenerate grid plans, and every site's `order` **is** its index in its body.

    `m4.plan` runs the whole self-check on what it built, so reaching a `MappingPlan` is
    already most of the claim; the order table is re-derived here from the finished plan so
    the test does not read M4's own `_check_orders` back to itself.
    """
    grid, tiles = DEGENERATE[shape]
    plan = _degenerate(grid, tiles, target).plan()
    assert plan.herd.grid == grid
    assert plan.mapping.repeats == tuple(1 for _ in grid), "these shapes emit no repeat loop"
    seen = _sites_in_order(plan.segment_body, []) + _sites_in_order(plan.herd_body, [])
    assert seen, "a plan with no channel site would make this vacuous"
    assert [(name, order) for name, order, _ in seen] == [(name, index)
                                                          for name, _, index in seen]
    assert plan.summary.l1_bytes <= 63488


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("shape", sorted(DEGENERATE), ids=sorted(DEGENERATE))
@pytest.mark.fr("FR-E7")
def test_degenerate_grid_emits(shape, target):
    """Each degenerate grid reaches `air.api` and comes back as a module."""
    require_pin()
    grid, tiles = DEGENERATE[shape]
    text = _degenerate(grid, tiles, target).mlir()
    assert "air.launch" in text and "air.herd" in text and "gemm_small" in text


@pytest.mark.parametrize("shape", sorted(DEGENERATE), ids=sorted(DEGENERATE))
@pytest.mark.fr("FR-K1")
def test_degenerate_grid_interprets_to_the_cpython_result(shape):
    """The plan, executed over numpy, is `A @ B` **exactly** (`tests/helpers/plan_interp.py`).

    Integer-valued `f32` in `[-8, 8)` with `K = 64` keeps every partial sum under `2**24`, so
    `np.array_equal` is the right comparison, and a one-PE plan has nowhere to hide a wrong
    slice: PE 0 owns the whole output.
    """
    grid, tiles = DEGENERATE[shape]
    rng = np.random.default_rng(0)
    tensors = {"A": rng.integers(-8, 8, (SMALL, SMALL)).astype(np.float32),
               "B": rng.integers(-8, 8, (SMALL, SMALL)).astype(np.float32),
               "C": np.zeros((SMALL, SMALL), dtype=np.float32)}
    expected = tensors["A"] @ tensors["B"]
    out = plan_interp.run(_degenerate(grid, tiles, "npu1").plan(), tensors)
    assert np.array_equal(out["C"], expected)


@pytest.mark.slow
@pytest.mark.requires_aircc
@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("shape", sorted(DEGENERATE), ids=sorted(DEGENERATE))
def test_degenerate_grid_compiles(shape, target, tmp_path, capsys):
    """`aircc --output-format=none` exits 0 on every degenerate grid, both generations."""
    require_pin()
    grid, tiles = DEGENERATE[shape]
    stem = shape.replace("(", "").replace(")", "").replace(",", "_")
    code, said = _aircc_verdict(_degenerate(grid, tiles, target).mlir(), target, tmp_path, stem)
    with capsys.disabled():
        print(f"\n  aircc --device {target} {shape}: {code or 'exit 0'} {said[:1]}")
    assert code is None, f"{shape} must compile on {target}; aircc said {code}: {said[:2]}"


# --------------------------------------------------------------------------------------------
# B-P37's controls — a herd taller than it is wide, and the same cores laid out the other way
# --------------------------------------------------------------------------------------------

TALL_M, TALL_N = 64, 96
"""The B-P36/B-P37 shape. `96` is divisible by 2, 3 and 4, so one kernel reaches `grid(2, 4)` —
npu2's whole 8-PE 2-D herd (`air.api`'s `PHYSICAL_HERD`, `_trace.py:88-91`) — and every control
beside it: the square `grid(2, 2)`, the one-column `grid(1, 4)` and `grid(1, 3)`. Every one of
them is `repeats (1, 1)` on npu2 and between 10 240 and 22 528 B of L1 on either generation,
far under the 63 488 B budget, which is what makes each refusal a statement about the herd's
**shape** alone."""


@sp.kernel
def gemm_tall(A: sp.f32[TALL_M, SMALL], B: sp.f32[SMALL, TALL_N], C: sp.f32[TALL_M, TALL_N]):
    for i in range(TALL_M):
        for j in range(TALL_N):
            for k in range(SMALL):
                C[i, j] += A[i, k] * B[k, j]


def _tall(grid, target):
    """W1's nine clauses on `gemm_tall`; the tiles follow the grid so `repeats == (1, 1)`."""
    s = sp.schedule(gemm_tall, target=target)
    ax = s.axes()
    s.grid(*grid)
    s.tile(ax.i, TALL_M // grid[0]); s.tile(ax.j, TALL_N // grid[1]); s.tile(ax.k, 16)
    s.reduce(ax.k, op="+")
    s.place(px=ax.i0, py=ax.j0)
    s.stationary("C")
    s.reside(A="L1", B="L1", C="L1")
    s.double_buffer("A", "B")
    s.pipeline(ax.k0)
    return s


@pytest.mark.fr("FR-L10")
def test_tall_herd_is_refused_for_the_broadcast_guard_on_npu2():
    """P3 refuses npu2's whole 8-PE herd **before codegen**, naming the guard bound (B-P37).

    `air-specialize-dma-broadcast` splits `A2L1` on the column axis and bounds the *row*
    coordinate of the guard it builds by the **column** count (`AIRMiscPasses.cpp:265-271`, the
    argument is `herd.getNumCols()` at `:347-349`), so a 2×4 herd admits rows 0..1 and loses the
    other two — and the first split channel then has a broadcast destination nothing allocates.
    """
    with pytest.raises(MappingError) as caught:
        _tall((2, 4), "npu2").plan()
    diagnostic = caught.value.diagnostic
    assert diagnostic.code == "DMA-CHANNELS"
    assert diagnostic.details["channel"] == "A2L1"
    assert diagnostic.details["herd_physical"] == (2, 4)
    assert (diagnostic.details["columns"], diagnostic.details["rows"]) == (2, 4)
    assert diagnostic.details["rows_unserved"] == 2
    assert diagnostic.details["specialize_dim"] == 0


@pytest.mark.fr("FR-L10")
def test_a_repeat_on_the_split_axis_takes_the_guard_out_of_range():
    """The B-P37 rule's exemption, which is why npu1 never reaches it.

    npu1's 2-D physical herd is one column, so `grid(2, 3)` folds onto `(1, 3)` with
    `repeats (2, 1)`; the bundle index is then an `affine.apply` of the herd id and the repeat
    variable rather than the bare id, upstream takes its `scf.if` arm (`AIRMiscPasses.cpp:
    380-422`) which bounds nothing, and `aircc` exits 0 — measured below.
    """
    plan = _tall((2, 3), "npu1").plan()
    assert (plan.summary.herd_physical, plan.summary.repeats) == ((1, 3), (2, 1))


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.fr("FR-L10")
def test_square_herd_is_taken_by_the_checker(target):
    """The control: the same kernel at `grid(2, 2)` plans on both generations.

    npu1 folds it onto one column, so **R-L1-3** doubles every buffer there — 2 × (6 144 +
    2 048 + 3 072) = 22 528 against npu2's undoubled 6 144 + 2 × 2 048 + 2 × 3 072 = 16 384.
    """
    plan = _tall((2, 2), target).plan()
    assert plan.summary.l1_bytes == {"npu1": 22528, "npu2": 16384}[target]
    assert plan.summary.herd_physical == {"npu1": (1, 2), "npu2": (2, 2)}[target]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.fr("FR-L10")
def test_four_row_column_is_refused_for_its_master_selects(target):
    """P3 refuses a four-row column on both generations, counting flows against 4 (B-P36).

    Five L3→L1 fill flows meet at the column's bottom tile — the `A2L1` multicast plus one
    `B2L1` per row — and the multicast's `{DMA, North}` port set only partially overlaps each
    `{DMA}`/`{North}`, so none of them share a master select. Four is the cap
    (`numMselsPerArbiter`).
    """
    with pytest.raises(MappingError) as caught:
        _tall((1, 4), target).plan()
    diagnostic = caught.value.diagnostic
    assert diagnostic.code == "DMA-CHANNELS"
    assert (diagnostic.details["flows"], diagnostic.details["budget"]) == (5, 4)
    assert diagnostic.details["herd_physical"] == (1, 4)
    assert diagnostic.details["multicast"] == ("A2L1",)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.fr("FR-L10")
def test_three_row_column_sits_on_the_master_select_cap(target):
    """The knife-edge control: four flows into a three-row column plan, and compile.

    This is the shape that makes the B-P36 rule a **count** rather than a guess — one flow
    fewer than the refusal above, exactly on the cap, accepted by the checker and by `aircc`.
    """
    plan = _tall((1, 3), target).plan()
    assert plan.summary.herd_physical == (1, 3)


@pytest.mark.slow
@pytest.mark.requires_aircc
def test_tall_herd_is_refused_by_aircc_and_the_square_one_is_not(tmp_path, capsys, monkeypatch):
    """The checker's refusal above is right: `aircc` refuses the same npu2 module (B-P37).

    The new P3 checks are switched off for the length of this test so the modules they now
    refuse can still be built and handed to `aircc`. The square control at the same clauses, the
    same two columns and a *larger* L1 draws exit 0, which is what makes the refusal a statement
    about the herd's aspect and not about its size. If a later toolchain compiles `grid(2, 4)`
    this fails and the rule is re-opened.
    """
    require_pin()
    from spatial import m4_selfcheck

    monkeypatch.setattr(m4_selfcheck, "broadcast_guard", lambda plan: None)
    monkeypatch.setattr(m4_selfcheck, "msels", lambda plan: None)
    tall_code, tall_said = _aircc_verdict(_tall((2, 4), "npu2").mlir(), "npu2", tmp_path, "tall")
    square_code, square_said = _aircc_verdict(_tall((2, 2), "npu2").mlir(), "npu2", tmp_path,
                                              "square")
    with capsys.disabled():
        print(f"\n  aircc --device npu2 grid(2,4): {tall_code or 'exit 0'} {tall_said[:1]}"
              f"\n  aircc --device npu2 grid(2,2): {square_code or 'exit 0'} {square_said[:1]}")
    assert square_code is None, (
        f"the square control must compile; aircc said {square_code}: {square_said[:2]}")
    assert tall_code is not None, (
        "aircc now compiles grid(2, 4) on npu2; the B-P37 rule must be re-measured")
    assert any("failed to get S2MM tile for L3 allocation" in line for line in tall_said), (
        f"grid(2, 4) fails for a different reason now: {tall_said[:2]}")


@pytest.mark.slow
@pytest.mark.requires_aircc
@pytest.mark.parametrize("target", TARGETS)
def test_the_master_select_cap_is_where_aircc_stops(target, tmp_path, capsys, monkeypatch):
    """B-P36's pair: five fill flows into one column are refused, four compile.

    `aiecc` segfaults *after* printing its diagnostic on the failing side, so the verdict is
    read from the diagnostic text rather than from the exit code alone. `test_W1_flip_2d_
    aircc_none` is the third leg of this measurement: **eight** flows into the same one-column
    four-row herd, none of them a multicast, and `aircc` exits 0.
    """
    require_pin()
    from spatial import m4_selfcheck

    monkeypatch.setattr(m4_selfcheck, "msels", lambda plan: None)
    over_code, over_said = _aircc_verdict(_tall((1, 4), target).mlir(), target, tmp_path, "five")
    at_code, at_said = _aircc_verdict(_tall((1, 3), target).mlir(), target, tmp_path, "four")
    said = next((line for line in over_said if "error:" in line), "")
    with capsys.disabled():
        print(f"\n  aircc --device {target} grid(1,4) 5 flows: {over_code or 'exit 0'} {said}"
              f"\n  aircc --device {target} grid(1,3) 4 flows: {at_code or 'exit 0'} {at_said[:1]}")
    assert at_code is None, (
        f"four flows must compile on {target}; aircc said {at_code}: {at_said[:2]}")
    assert over_code is not None, (
        f"aircc now compiles five fill flows into one column on {target}; B-P36 must be "
        f"re-measured")
    assert any("used up all its msels" in line for line in over_said), (
        f"five flows fail for a different reason now on {target}: {over_said[-2:]}")


# --------------------------------------------------------------------------------------------
# R-HERD-1's controls — the repeat factor the toolchain lowers, and the one it does not
# --------------------------------------------------------------------------------------------

REPEAT_M, REPEAT_N = 128, 64
"""The `repeats (4, 1)` control: `M // TM == 4` against npu1's one-column herd. W1's own tiles,
so its L1 is 8 192 B undoubled and 16 384 B doubled — far under the budget, which is what makes
the `aircc` refusal below a statement about the **repeat factor** alone."""


@sp.kernel
def gemm_repeat4(A: sp.f32[REPEAT_M, SMALL], B: sp.f32[SMALL, REPEAT_N],
                 C: sp.f32[REPEAT_M, REPEAT_N]):
    for i in range(REPEAT_M):
        for j in range(REPEAT_N):
            for k in range(SMALL):
                C[i, j] += A[i, k] * B[k, j]


def _repeat4(target="npu1"):
    """`grid(4, 2)` on npu1: physical `(1, 2)`, `repeats (4, 1)` — W1's clauses and tiles."""
    s = sp.schedule(gemm_repeat4, target=target)
    ax = s.axes()
    s.grid(REPEAT_M // w1.TM, REPEAT_N // w1.TN)
    s.tile(ax.i, w1.TM); s.tile(ax.j, w1.TN); s.tile(ax.k, w1.TK)
    s.reduce(ax.k, op="+")
    s.place(px=ax.i0, py=ax.j0)
    s.stationary("C")
    s.reside(A="L1", B="L1", C="L1")
    s.double_buffer("A", "B")
    s.pipeline(ax.k0)
    return s


@pytest.mark.fr("FR-L10")
def test_repeat_four_is_rejected_and_repeat_two_is_not():
    """R-HERD-1: the checker draws the line at the unroll factor, and names it.

    Both shapes carry W1's tiles, so their L1 fits even doubled — the repeat factor is the
    only thing that differs, which is the claim.
    """
    with pytest.raises(LegalityError) as caught:
        _repeat4("npu1").check()
    diagnostic = caught.value.diagnostic
    assert diagnostic.code == "HERD-PHYSICAL"
    assert (diagnostic.details["grid"], diagnostic.details["physical_herd"],
            diagnostic.details["repeats"]) == ((4, 2), (1, 2), (4, 1))
    assert diagnostic.details["unroll_factor"] == 2
    assert "npu1 1-D 8, 2-D (2, 8)" in diagnostic.fix
    assert "npu2 1-D 16, 2-D (4, 8)" in diagnostic.fix
    # ...and the same tiles at `repeats (2, 1)` — W1 base itself — are accepted.
    accepted = w1.schedule_os("npu1").check()
    assert (accepted.physical_herd, accepted.repeats) == ((1, 2), (2, 1))


@pytest.mark.slow
@pytest.mark.requires_aircc
def test_repeat_four_is_what_aircc_refuses(tmp_path, capsys, monkeypatch):
    """The control R-HERD-1 rests on: `aircc` refuses `repeats (4, 1)` and takes `(2, 1)`.

    The rejected module is built with `_check_repeats` monkeypatched to a no-op, so this
    measures the toolchain rather than quoting B-P33. The passing control is W1 base on npu1 —
    same tiles, same clauses, `repeats (2, 1)`.
    """
    require_pin()
    monkeypatch.setattr(m3_legality, "_check_repeats", lambda *args, **kw: None)
    bad_code, bad_said = _aircc_verdict(_repeat4("npu1").mlir(), "npu1", tmp_path, "repeat4")
    good_code, good_said = _aircc_verdict(w1.schedule_os("npu1").mlir(), "npu1", tmp_path,
                                          "repeat2")
    with capsys.disabled():
        print(f"\n  aircc --device npu1 repeats(4,1): {bad_code or 'exit 0'} {bad_said[:1]}"
              f"\n  aircc --device npu1 repeats(2,1): {good_code or 'exit 0'} {good_said[:1]}")
    assert bad_code is not None, "aircc now lowers repeats (4, 1); R-HERD-1 must be re-measured"
    assert any("MM2S tile for L3 allocation" in line for line in bad_said), bad_said[:3]
    assert good_code is None, f"repeats (2, 1) must compile; aircc said {good_code}"
