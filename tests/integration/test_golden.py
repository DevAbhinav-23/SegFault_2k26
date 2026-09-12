"""Level G — the golden pipeline. Spec: design/04-test-plan.md §3.1, design/06-interfaces.md §8.

W1 only: the other three variants need M4, or plan literals that do not exist yet
(`design/PROGRESS-B.md`, phase P1).

A golden is valid **only for the pinned wheel** (FR-T6), so a pin mismatch **skips** every test
here with the reason rather than failing it (§8 rule 2).

Written by B at P1; C owns `tests/helpers/golden.py` itself.
"""

from __future__ import annotations

import json

import pytest

from spatial import m5_emit, m6_tools as m6
from spatial.model import ToolchainError, to_json
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


@pytest.mark.fr("FR-M11")
def test_golden_w1_summary():
    """The rendered summary (FR-M11).

    One file, as `06-interfaces.md` §8 names it (`<workload>.<variant>.summary.txt`, no
    target) — so it is written for `npu1` only: W1's summary carries the herd's *physical*
    shape and repeats, which differ between the targets, and §8's path has nowhere to say
    which. `w1.base.npu2.plan.json` carries the npu2 rendering of the same lines, so nothing
    is unwitnessed. Recorded in `design/PROGRESS-B.md` as a §8 / §3.1 naming conflict.
    """
    require_pin()
    summary = w1_plan.plan("npu1").summary
    assert_golden("w1.base.summary.txt", "\n".join(summary.lines) + "\n", kind="text")


def test_golden_skips_on_pin_mismatch(monkeypatch):
    """A pin mismatch **skips** every golden test with a reason; it never fails one."""
    monkeypatch.setitem(m6.PIN, "mlir_air", "0.0.0+not-the-pinned-wheel")
    with pytest.raises(pytest.skip.Exception) as excinfo:
        require_pin()
    assert "mlir_air" in str(excinfo.value)
    assert "not-the-pinned-wheel" in str(excinfo.value)
