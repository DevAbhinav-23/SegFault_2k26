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
only) and is not emittable on this toolchain; see that module's docstring.
"""

from __future__ import annotations

import json

import pytest

from kernels import w1_gemm as w1
from kernels import w2_jacobi as w2
from kernels import w3_sw as w3
from spatial import m6_tools as m6
from spatial.model import ToolchainError, to_json
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
