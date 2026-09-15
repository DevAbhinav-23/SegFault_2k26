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

import pytest

import spatial as sp
from kernels import w1_gemm as w1
from kernels import w1_gemm_bf16 as w1_large
from kernels import w2_jacobi as w2
from kernels import w3_sw as w3
from spatial import m4_mapping as m4
from spatial import m5_emit, m5tt_emit, m6_tools as m6
from spatial.model import ToolchainError, to_json
from tests.fixtures.mappings import w1_legal, w1flip_legal, w2_legal, w3_legal
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
        return exc.diagnostic.code, exc.diagnostic.details.get("diagnostics", ())
    return None, ()


@pytest.mark.parametrize("target", TARGETS)
def test_live_bf16_gemm_widens_every_load_to_the_accumulator(target):
    """B-P32: W1-large plans and emits, with one `arith.extf` per `bf16` operand.

    Two, not one: the ruling is that each **load** is cast to the destination's dtype before
    the arithmetic, so the product and the accumulate are both f32 — which is what "bf16 in,
    f32 out" means. A cast around the product would be one `extf` and a bf16 multiply.
    """
    require_pin()
    text = w1_large.schedule_os(target).mlir()
    assert text.count("arith.extf") == 2, "one widening per bf16 operand"
    assert "truncf" not in text, "nothing narrows back: the accumulator stays f32"
    assert "bf16" in text and "f32" in text


@pytest.mark.slow
@pytest.mark.requires_aircc
@pytest.mark.parametrize("target", TARGETS)
def test_live_bf16_gemm_is_refused_by_aircc_for_its_shape(target, tmp_path, capsys):
    """`aircc` refuses W1-large — and refuses the **f32 control** at the same grid identically.

    The fixture's shape is fixed at 256³ / 4×4 / `TM = TN = TK = 64` by
    `03-lld-M8-kernels-demo.md` §4 (VF §E.5's measured probe). At that shape the logical 4×4
    grid does not fit either physical herd — `(1, 4)` on npu1, `(2, 4)` on npu2 — so M4 emits a
    **repeat loop**, and both failures are downstream of that (measured 2026-09-15,
    `design/PROGRESS-B.md` B-P33 and B-P34):

    * **npu1, `repeats == (4, 1)`** — `air-to-aie` at
      `AIRToAIESchedulingUtils.cpp:3892-3894`: *"'air.channel.get' op failed to get MM2S tile
      for L3 allocation."* The herd-side `C2L3` put still indexes the bundle with a loop
      induction variable (`affine.apply ()[%arg10, %arg14] -> (s0 * 4 + s1)`), so the bundle
      position never resolves to a producer. Upstream names this cause in
      `test/Conversion/AIRToAIE/segment_id_remap_no_unroll.mlir:14-19`.
    * **npu2, `repeats == (2, 1)`** — the repeat loop gives each core **two** accumulators, so
      the tile asks 2 × 16 384 + 4 × 8 192 = 65 536 B and `AIEAssignBuffers` answers *"'aie.tile'
      op allocated buffers exceeded available memory"*, its `MemoryMap` note ending at `0x107FF`
      — the 2 048 B stack plus 65 536 B of buffers in a 65 536 B tile.

    Neither figure is what our checker charges: `l1_bytes` is 49 152 for this schedule **and**
    for the control below, which is exactly why the control is the test. The f32 control here
    makes it a statement about the shape rather than about B-P32's cast: same clauses, same
    grid, same 49 152 B of L1, **zero** casts, and the assertion is that the two verdicts agree.
    If a later toolchain compiles W1-large, this fails on the control too and the whole row is
    re-measured.
    """
    require_pin()
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
    assert schedule.check().l1_bytes == w1_large.schedule_os(target).check().l1_bytes == 49152
    assert schedule.check().repeats == (1, 1), "the control's grid must fit the herd"
    text = schedule.mlir()
    assert text.count("arith.extf") == 2, "the control compiles B-P32's cast, not a cast-free IR"
    code, said = _aircc_verdict(text, target, tmp_path, "fits")
    with capsys.disabled():
        print(f"\n  aircc --device {target} bf16 grid(1,2): {code or 'exit 0'} {said[:1]}")
    assert code is None, f"the control must compile on {target}; aircc said {code}: {said[:2]}"
