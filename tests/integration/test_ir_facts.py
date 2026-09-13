"""Level I — `air-opt` inspection. Spec: design/04-test-plan.md §3.2.

Stub written by B at P0c to unblock; **Person C owns this file**.
"""

from __future__ import annotations

import re
from importlib.resources import files
from pathlib import Path

import pytest

from spatial import m6_tools as m6
from tests.helpers.golden import assert_golden
from tests.integration.test_golden import require_pin

MODULES = Path(str(files("tests"))) / "fixtures" / "modules"


@pytest.mark.requires_air_opt
@pytest.mark.fr("FR-T5")
def test_I_unpaired_channel(tmp_path):
    """The characterisation test FR-T5 exists for: a diagnostic on stderr **and exit 0**.

    This is the one test that runs a hand-written broken module rather than our emitter's
    output. It characterises the toolchain, not us: if upstream ever starts failing on this,
    the test breaks loudly and FR-T5 can be relaxed (`04-test-plan.md` §3.2).
    """
    if not m6.tool_available("air-opt"):
        pytest.skip("air-opt is not resolvable; install the toolchain (07-environment.md §2)")

    module = MODULES / "unpaired_channel.mlir"
    run = m6.invoke([m6.tool("air-opt"), module,
                     f"-pass-pipeline={m6.PIPELINES['pairs']}", "-o", tmp_path / "pairs.mlir"],
                    cwd=tmp_path)

    assert "found channel op not in pairs" in run.stderr
    assert run.returncode == 0, "the exit code is the half that lies (VF §B.6, §S11)"

    # ...and the wrapper fails it anyway, which is the whole of FR-T5.
    with pytest.raises(m6.ToolchainError) as excinfo:
        m6.verdict(run)
    assert excinfo.value.diagnostic.code == "TOOL-DIAGNOSTIC"
    assert excinfo.value.diagnostic.details["returncode"] == 0


# --------------------------------------------------------------------------------------------
# W1, emitted by M5. Spec: design/04-test-plan.md §3.2, design/03-lld-M5-emitter.md §3.8.
# Added by B at P1.
# --------------------------------------------------------------------------------------------


def _require_air_opt() -> None:
    if not m6.tool_available("air-opt"):
        pytest.skip("air-opt is not resolvable; install the toolchain (07-environment.md §2)")


def w1_module(tmp_path: Path) -> Path:
    """W1's emitted npu1 module, on disk, which is what `air-opt` reads."""
    from spatial import m5_emit
    from tests.fixtures.plans import w1_plan

    path = tmp_path / "w1.base.npu1.air.mlir"
    path.write_text(m5_emit.emit(w1_plan.plan("npu1"), "npu1").mlir, encoding="utf-8")
    return path


def w3_module(tmp_path: Path) -> Path:
    """W3's emitted npu1 module, on disk, which is what `air-opt` reads."""
    from spatial import m4_mapping as m4, m5_emit
    from tests.fixtures.mappings import w3_legal

    path = tmp_path / "w3.base.npu1.air.mlir"
    path.write_text(m5_emit.emit(m4.plan(w3_legal.legal("npu1")), "npu1").mlir, encoding="utf-8")
    return path


W1_FACTS = ("pingpong_unroll", "hoist_alloc_count", "cascade_channels",
            "broadcast_pattern_count", "pingpong_iter_args")
"""The fields `03-lld-M5-emitter.md` §3.8 owns for W1. `lock_init_histogram` is D6's."""


@pytest.mark.requires_air_opt
@pytest.mark.fr("FR-E3")
def test_I_pingpong_labels(tmp_path):
    """The K loop is labelled `unroll = 2 : i32` and both L1 tiles `hoist_alloc = true`.

    This is the post-pass half of FR-E3: `test_E3_alloc_is_direct_child` asserts the shape M5
    emitted, and this asserts that `air-label-scf-for-to-ping-pong` accepts it.
    """
    _require_air_opt()
    facts = m6.ir_facts(str(w1_module(tmp_path)), "npu1",
                        ["pingpong_unroll", "hoist_alloc_count"], workdir=tmp_path)
    assert facts["pingpong_unroll"] == 2
    assert facts["hoist_alloc_count"] == 2
    assert facts["_pipeline_pingpong"] == m6.PIPELINES["pingpong"].format(target="npu1")


@pytest.mark.requires_air_opt
@pytest.mark.fr("FR-E3")
def test_I_pingpong_transform(tmp_path):
    """After `air-ping-pong-transform` the K loop's step doubles and it carries 4 tokens."""
    _require_air_opt()
    module = w1_module(tmp_path)
    before = module.read_text(encoding="utf-8")
    # The K loop as M5 emitted it: 0..64 step 16, with the two L1 allocs first in its body.
    assert re.search(r"scf\.for %\w+ = %\w+ to %\w+ step %c16", before)

    pipeline = m6.PIPELINES["transform"].format(target="npu1")
    out = tmp_path / "transform.mlir"
    m6.verdict(m6.invoke([m6.tool("air-opt"), module, f"-pass-pipeline={pipeline}", "-o", out],
                         cwd=tmp_path))
    after = out.read_text(encoding="utf-8")

    doubled = [line for line in after.splitlines()
               if "scf.for" in line and "step %c32" in line and "iter_args" in line]
    assert len(doubled) == 1, f"expected one ping-ponged loop, got {doubled}"
    assert doubled[0].count("!air.async.token") == 4, doubled[0]


@pytest.mark.requires_air_opt
@pytest.mark.parametrize("workload", ["w1", "w3"])
@pytest.mark.fr("FR-E5")
def test_I_broadcast_count(tmp_path, workload):
    """R-04's tripwire: declaring `broadcast_shape` bypasses the detector (D-4, VF §S9).

    **Measured 0** on our emitted W1 (`03-lld-M5-emitter.md` §3.8, finding N-8) and, for the
    same reason, on W3 — `QIn` carries `broadcast_shape = [4]`, so the pass has nothing left to
    derive. If either number ever changes, record the new one: it means
    `air-broadcast-detection` now walks something we emit, and the reason belongs in the commit
    message, not in a patched expectation.
    """
    _require_air_opt()
    module = {"w1": w1_module, "w3": w3_module}[workload](tmp_path)
    facts = m6.ir_facts(str(module), "npu1", ["broadcast_pattern_count"], workdir=tmp_path)
    assert facts["broadcast_pattern_count"] == 0
    assert facts["_pipeline_broadcast"] == m6.PIPELINES["broadcast"].format(target="npu1")


@pytest.mark.requires_air_opt
@pytest.mark.fr("FR-E3", "FR-E5", "FR-T6")
def test_I_facts_golden(tmp_path):
    """Every §3.8 fact, with the pipeline string each came from, frozen as one small file."""
    _require_air_opt()
    require_pin()
    facts = m6.ir_facts(str(w1_module(tmp_path)), "npu1", W1_FACTS, workdir=tmp_path)
    assert_golden("w1.base.npu1.ir_facts.json", facts, kind="json")


# --------------------------------------------------------------------------------------------
# W3, emitted by M5. Spec: design/04-test-plan.md §3.2, design/03-lld-M5-emitter.md §3.8.
# Added by B at P4.
# --------------------------------------------------------------------------------------------


W3_FACTS = ("broadcast_pattern_count", "cascade_channels", "lock_init_histogram")
"""What §3.8 has to say about a wavefront: no derived broadcast, no cascade, and the lock inits
`air-to-aie` allocated — the fact `04-test-plan.md` §3.2 marks optional and P1 left undone."""


@pytest.mark.requires_air_opt
@pytest.mark.fr("FR-E5", "FR-T6")
def test_I_facts_golden_w3(tmp_path):
    """W3's §3.8 facts, with the pipeline each came from, frozen as one small file.

    `lock_init_histogram` is read from the `aie` pipeline, which now places herds before
    `air-to-aie` — without that the pass asks for column 4 of a 4-column device and fails
    (`m6_tools.PIPELINES`). Cross-checked once against `aircc`'s own
    `air_project/aie.w3.base.npu1.air.mlir`: identical.
    """
    _require_air_opt()
    require_pin()
    facts = m6.ir_facts(str(w3_module(tmp_path)), "npu1", W3_FACTS, workdir=tmp_path)
    assert facts["cascade_channels"] == 0, "a wavefront has no npu_cascade channel"
    assert sum(facts["lock_init_histogram"].values()) > 0
    assert_golden("w3.base.npu1.ir_facts.json", facts, kind="json")
