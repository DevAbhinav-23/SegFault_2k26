"""Level G — the golden pipeline. Spec: design/04-test-plan.md §3.1, design/06-interfaces.md §8.

All four variants: `w1.base`, `w1.flip`, `w2.base` (+ `w2.odd`) and `w3.base`.

A golden is valid **only for the pinned wheel** (FR-T6), so a pin mismatch **skips** every test
here with the reason rather than failing it (§8 rule 2).

Written by B at P1, extended at P2 with the M4 path, P4 with W3, P5 with W2 and P6 with the
flip; C owns `tests/helpers/golden.py`.
"""

from __future__ import annotations

import json

import pytest

from spatial import m4_mapping as m4, m5_emit, m6_tools as m6
from spatial.model import Expr, ToolchainError, to_json
from tests.fixtures.mappings import w1_legal, w1flip_legal, w2_legal, w3_legal
from tests.fixtures.plans import w1_plan
from tests.helpers.golden import assert_golden

TARGETS = ("npu1", "npu2")


def require_pin() -> None:
    """Skip — never fail — when the installed toolchain is not the pinned one (§8 rule 2)."""
    try:
        m6.check_pin()
    except ToolchainError as exc:
        pytest.skip(f"{exc.diagnostic.reason}: {exc.diagnostic.details['mismatched']}")


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.fr("FR-E7", "FR-T6")
def test_golden_w1(target):
    """The AIR text byte for byte, and the plan as canonical JSON, for both targets."""
    require_pin()
    plan = w1_plan.plan(target)
    result = m5_emit.emit(plan, target)
    assert_golden(f"w1.base.{target}.air.mlir", result.mlir, kind="text")
    # `to_json` already emits §8's canonical form; parsing it back hands `assert_golden` the
    # object it compares (and re-serialises identically), rather than a JSON string of a string.
    assert_golden(f"w1.base.{target}.plan.json", json.loads(to_json(plan)), kind="json")


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.fr("FR-M11")
def test_golden_w1_summary(target):
    """The rendered summary (FR-M11), **per target** — the B-P14 ruling.

    `06-interfaces.md` §8's target-less `<workload>.<variant>.summary.txt` could not hold both
    targets: W1's summary carries the herd's *physical* shape and repeats, which differ between
    npu1 and npu2. The architect ruled the path gains `<target>`; §8 carries the erratum.
    """
    require_pin()
    summary = w1_plan.plan(target).summary
    assert_golden(f"w1.base.{target}.summary.txt", "\n".join(summary.lines) + "\n", kind="text")


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.fr("FR-E7", "FR-M12")
def test_W1_legal_to_text(target):
    """Gate G2's B half: `LegalMapping` → M4 → M5 → the compiled module's own golden text.

    The literal plan is M4's stand-in no longer: the text M5 emits from the plan M4 *derives*
    is byte for byte the `w1.base.<target>.air.mlir` golden that the hand-written literal
    produced, which is the whole of `m4.plan` + `m5.emit` against one file.
    """
    require_pin()
    result = m5_emit.emit(m4.plan(w1_legal.legal(target)), target)
    assert_golden(f"w1.base.{target}.air.mlir", result.mlir, kind="text")


def test_golden_skips_on_pin_mismatch(monkeypatch):
    """A pin mismatch **skips** every golden test with a reason; it never fails one."""
    monkeypatch.setitem(m6.PIN, "mlir_air", "0.0.0+not-the-pinned-wheel")
    with pytest.raises(pytest.skip.Exception) as excinfo:
        require_pin()
    assert "mlir_air" in str(excinfo.value)
    assert "not-the-pinned-wheel" in str(excinfo.value)


# --------------------------------------------------------------------------------------------
# W3 — the wavefront, derived by M4 and emitted by M5. Added by B at P4.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.fr("FR-E7", "FR-M5", "FR-K4", "FR-T6")
def test_golden_w3(target):
    """Gate G3's B half: `LegalMapping` → M4 → M5 → the AIR text, byte for byte, per target.

    There is no hand-written W3 plan literal to cross-check against — W1's existed because M5
    was built before M4 — so the golden is the whole of the contract, and the `aircc` and
    interpreter tests beside it are what say the text means what it should.
    """
    require_pin()
    plan = m4.plan(w3_legal.legal(target))
    result = m5_emit.emit(plan, target)
    assert_golden(f"w3.base.{target}.air.mlir", result.mlir, kind="text")
    assert_golden(f"w3.base.{target}.plan.json", json.loads(to_json(plan)), kind="json")


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.fr("FR-M11")
def test_golden_w3_summary(target):
    """W3's summary, including the delivery block of **B-P22** and the §3.8 DMA warnings."""
    require_pin()
    summary = m4.plan(w3_legal.legal(target)).summary
    assert_golden(f"w3.base.{target}.summary.txt", "\n".join(summary.lines) + "\n", kind="text")
    for line in ("S: forward along px (declared)", "q: multicast along px (derived)",
                 "r: stationary (derived)", "L1: 240 of 65536 bytes"):
        assert line in summary.lines
    warnings = [line for line in summary.lines if line.startswith("warning: ")]
    assert len(warnings) == 4 and all("packet" in line for line in warnings)


# --------------------------------------------------------------------------------------------
# W2 — the halo exchange, derived by M4 and emitted by M5. Added by B at P5.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.fr("FR-E7", "FR-M4", "FR-K3", "FR-T6")
def test_golden_w2(target):
    """Gate G4's B half: `LegalMapping` → M4 → M5 → the AIR text, byte for byte, per target."""
    require_pin()
    plan = m4.plan(w2_legal.legal(target))
    result = m5_emit.emit(plan, target)
    assert_golden(f"w2.base.{target}.air.mlir", result.mlir, kind="text")
    assert_golden(f"w2.base.{target}.plan.json", json.loads(to_json(plan)), kind="json")


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.fr("FR-M4", "FR-L14", "FR-T6")
def test_golden_w2_odd(target):
    """`w2.odd` — the `T = 5` **plan** golden, and deliberately no module golden.

    `06-interfaces.md` §8's `<variant>` vocabulary gains `odd` beside `base` and `flip` (erratum,
    2026-09-13). Only `plan.json` is frozen: the module differs from `base` by the peeled tail
    alone, which `test_E_peel_is_plan_driven` asserts directly, so a second 300-line module
    golden would freeze the same fact twice and diff twice whenever the emitter changes.
    """
    require_pin()
    plan = m4.plan(w2_legal.legal(target, T=5))
    assert_golden(f"w2.odd.{target}.plan.json", json.loads(to_json(plan)), kind="json")
    assert (plan.herd_body[4].hi, plan.herd_body[4].step) == (Expr((), 4), Expr((), 2))
    assert len(plan.herd_body) == 11, "the peeled STEP is six nodes after the loop"


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.fr("FR-M11")
def test_golden_w2_summary(target):
    """W2's summary: the delivery block of **B-P22**, the residency of R-W2-3, no warning."""
    require_pin()
    summary = m4.plan(w2_legal.legal(target)).summary
    assert_golden(f"w2.base.{target}.summary.txt", "\n".join(summary.lines) + "\n", kind="text")
    for line in ("U: stationary (declared)",
                 "U: stationary (spatial), resident for the whole run",
                 "L1: 1280 of 65536 bytes",
                 "  ToNorth size=(1,)", "  ToSouth size=(1,)"):
        assert line in summary.lines
    assert not [line for line in summary.lines if line.startswith("warning: ")]


# --------------------------------------------------------------------------------------------
# W1-flip — the weight-stationary cascade, derived by M4 and emitted by M5. Added by B at P6.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.fr("FR-E7", "FR-M6", "FR-K2", "FR-T6")
def test_golden_w1_flip(target):
    """Gate G5's B half: the flip's `LegalMapping` → M4 → M5 → the AIR text, byte for byte.

    FR-K2's claim is that the flip is a **schedule** edit: `w1flip_legal` imports W1's kernel
    unchanged and changes four clauses plus the dropped `tile(ax.j, TN)`, and what comes out is
    a different dataflow — weights resident, `A` streaming, partial sums on an `npu_cascade`
    chain. The two goldens are the evidence that nothing in the kernel moved.
    """
    require_pin()
    assert w1flip_legal.kernel() == w1_legal.kernel(), "FR-K2: no source edit"
    plan = m4.plan(w1flip_legal.legal(target))
    result = m5_emit.emit(plan, target)
    assert_golden(f"w1.flip.{target}.air.mlir", result.mlir, kind="text")
    assert_golden(f"w1.flip.{target}.plan.json", json.loads(to_json(plan)), kind="json")


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.fr("FR-M11", "FR-K2")
def test_golden_w1_flip_summary(target):
    """The flip's summary, and the six lines `05-work-breakdown.md` §5 step 3 reads out loud."""
    require_pin()
    summary = m4.plan(w1flip_legal.legal(target)).summary
    assert_golden(f"w1.flip.{target}.summary.txt", "\n".join(summary.lines) + "\n", kind="text")
    for line in ("A: stationary (derived)", "B: stationary (declared)",
                 "C: cascade along px (derived)",
                 "A: stationary (spatial), re-fetched per i0",
                 "B: stationary (spatial), resident for the whole run",
                 "C: cascade along px, re-fetched per i0",
                 "reduction (tiled axes): R_time = span{e_k1}, R_space = span{e_k0}",
                 "L1: 24576 of 65536 bytes",
                 "  CascadeK size=(3,) type=npu_cascade"):
        assert line in summary.lines
    assert not [line for line in summary.lines if line.startswith("warning: ")]
    # W1's own summary is the other half of the demo: same kernel, the other dataflow
    base = w1_plan.plan(target).summary
    assert "C: stationary (declared)" in base.lines and "L1: 12288 of 65536 bytes" in base.lines
