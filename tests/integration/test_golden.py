"""Level G — the golden pipeline. Spec: design/04-test-plan.md §3.1, design/06-interfaces.md §8.

W1 only: the other three variants need the protocol builders of M4 §3.6, which land in
P4/P5/P6 (`design/PROGRESS-B.md`).

A golden is valid **only for the pinned wheel** (FR-T6), so a pin mismatch **skips** every test
here with the reason rather than failing it (§8 rule 2).

Written by B at P1, extended at P2 with the M4 path; C owns `tests/helpers/golden.py`.
"""

from __future__ import annotations

import json

import pytest

from spatial import m4_mapping as m4, m5_emit, m6_tools as m6
from spatial.model import ToolchainError, to_json
from tests.fixtures.mappings import w1_legal
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
