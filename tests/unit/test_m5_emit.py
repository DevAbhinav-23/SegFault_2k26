"""Level U — the AIR emitter. Spec: design/03-lld-M5-emitter.md §7, design/04-test-plan.md §2.

The W1 cases are driven by the hand-written W1 `MappingPlan` literal
(`tests/fixtures/plans/w1_plan.py`); the W1-flip, W2 and W3 cases are driven by the plans M4
**derives**, since no literal exists for any of them. Every row of §7 has a vehicle since P6.

Written by B at P1 together with `spatial/m5_emit.py`; extended at P4 with W3, P5 with W2 and
P6 with the W1-flip's cascade (rows 5c and 11, the two that had no plan to exercise them).
"""

from __future__ import annotations

import ast
import importlib
import re
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from spatial import m4_mapping as m4, m5_emit, m6_tools as m6
from spatial.model import BranchNode, EmissionError, LoopPlan, MappingPlan
from tests.fixtures.mappings import w1flip_legal, w2_legal, w3_legal
from tests.fixtures.plans import w1_plan
from tests.helpers import determinism
from tests.helpers.diagnostics import assert_diagnostic

SOURCE = Path(m5_emit.__file__).read_text(encoding="utf-8")
"""M5's own source, for the three lint tests (P-1, P-4, P-6, FR-E4)."""

_TEXTS: dict[str, str] = {}


def text(target: str = "npu1") -> str:
    """The emitted W1 module text for `target`, emitted once per process."""
    if target not in _TEXTS:
        _TEXTS[target] = m5_emit.emit(w1_plan.plan(target), target).mlir
    return _TEXTS[target]


_W3_TEXTS: dict[str, str] = {}


def w3_text(target: str = "npu1") -> str:
    """The emitted W3 module text for `target`, emitted once per process."""
    if target not in _W3_TEXTS:
        _W3_TEXTS[target] = m5_emit.emit(m4.plan(w3_legal.legal(target)), target).mlir
    return _W3_TEXTS[target]


_W2_TEXTS: dict[str, str] = {}


def w2_text(target: str = "npu1") -> str:
    """The emitted W2 module text for `target`, emitted once per process."""
    if target not in _W2_TEXTS:
        _W2_TEXTS[target] = m5_emit.emit(m4.plan(w2_legal.legal(target)), target).mlir
    return _W2_TEXTS[target]


_FLIP_TEXTS: dict[str, str] = {}


def flip_text(target: str = "npu1") -> str:
    """The emitted W1-flip module text for `target`, emitted once per process."""
    if target not in _FLIP_TEXTS:
        _FLIP_TEXTS[target] = m5_emit.emit(m4.plan(w1flip_legal.legal(target)), target).mlir
    return _FLIP_TEXTS[target]


def emit_w2_text(target: str = "npu1") -> str:
    """Module-level and picklable, so `determinism.in_fresh_process` can call it (FR-E10)."""
    return m5_emit.emit(m4.plan(w2_legal.legal(target)), target).mlir


def emit_w3_text(target: str = "npu1") -> str:
    """Module-level and picklable, so `determinism.in_fresh_process` can call it (FR-E10)."""
    return m5_emit.emit(m4.plan(w3_legal.legal(target)), target).mlir


def emit_w1_text(target: str = "npu1") -> str:
    """Module-level and picklable, so `determinism.in_fresh_process` can call it (FR-E10)."""
    return m5_emit.emit(w1_plan.plan(target), target).mlir


def air_in_sys_modules_after_importing_m5() -> bool:
    """True if merely importing M5 pulls in `air`. Run in a fresh process (FR-S20)."""
    importlib.import_module("spatial.m5_emit")
    return "air" in sys.modules


def _herd_body(lines: list[str]) -> list[str]:
    """The lines between `air.herd` and the end of its region."""
    start = next(i for i, line in enumerate(lines) if "air.herd" in line)
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = next(i for i in range(start + 1, len(lines))
               if lines[i].strip() == "}" and len(lines[i]) - len(lines[i].lstrip()) == indent)
    return lines[start + 1:end]


# --------------------------------------------------------------------------------------------
# FR-E1, FR-E2 — the hierarchy and the native body
# --------------------------------------------------------------------------------------------


@pytest.mark.fr("FR-E1")
def test_E1_hierarchy():
    """`air.launch` / `air.segment` / `air.herd`, in that nesting; `build()` ran verify()."""
    lines = text().splitlines()
    positions = {}
    for name in ("air.launch", "air.segment", "air.herd"):
        positions[name] = next(i for i, line in enumerate(lines) if name in line)
    assert positions["air.launch"] < positions["air.segment"] < positions["air.herd"]

    def indent(name):
        line = lines[positions[name]]
        return len(line) - len(line.lstrip())

    assert indent("air.launch") < indent("air.segment") < indent("air.herd")
    assert lines[0].startswith("#map") or lines[0] == "module {"
    # LaunchContext.build() runs module.operation.verify() (_compile.py:156-158); a failure
    # would have raised EMIT-VERIFY rather than returning this text.
    assert f"func.func @{w1_plan.plan().launch_name}(" in text()


@pytest.mark.fr("FR-E2")
def test_E2_native_body():
    """The herd body is native: nested `scf.for`, `memref` access, no call into an object."""
    body = "\n".join(_herd_body(text().splitlines()))
    assert body.count("scf.for") >= 3
    assert "memref.load" in body and "memref.store" in body
    assert "func.call" not in text()
    assert "link_with" not in text()
    assert "link_with" not in SOURCE and "extern" not in SOURCE


@pytest.mark.fr("FR-E2")
def test_E2_zero_is_compute_nodes():
    """The accumulator zeroing is `StoreNode`s, not `ops.fill` (M5 §3.6)."""
    assert re.search(r"%cst = arith\.constant 0\.0+e\+00 : f32", text())
    assert re.search(r"memref\.store %cst, %alloc\[", text())
    assert "ops.fill" not in SOURCE and "fill(" not in SOURCE


# --------------------------------------------------------------------------------------------
# FR-E3 — the ping-pong loop shape
# --------------------------------------------------------------------------------------------


@pytest.mark.fr("FR-E3")
def test_E3_alloc_is_direct_child():
    """Both L1 tiles are allocated directly inside the K `scf.for`, first in its body."""
    lines = text().splitlines()
    first = next(i for i, line in enumerate(lines)
                 if "memref.alloc()" in line and "memref<32x16xf32, 2 : i32>" in line)
    second = first + 1
    assert "memref.alloc()" in lines[second]
    assert "memref<16x32xf32, 2 : i32>" in lines[second]

    opener = lines[first - 1]
    assert "scf.for" in opener, f"the alloc is not the first op of a loop body: {opener!r}"
    alloc_indent = len(lines[first]) - len(lines[first].lstrip())
    for_indent = len(opener) - len(opener.lstrip())
    assert alloc_indent == for_indent + 2
    assert len(lines[second]) - len(lines[second].lstrip()) == alloc_indent
    # ...and it is the K loop: step 16 over 0..64.
    step = re.search(r"step %(\w+)", opener).group(1)
    assert re.search(rf"%{step} = arith\.constant 16 : index", "\n".join(lines[:first]))


@pytest.mark.fr("FR-E3")
def test_E3_no_hoist():
    """M5 emits an alloc where the plan puts it — above the K loop when the plan says depth 0.

    The plan variant is one M4 would never produce (`06-interfaces.md` §5.6 invariant 4 is what
    rejects it), so `ping_pong_candidate` goes with it: M0's `I41` refuses a candidate at
    `loop_depth == 0`, and M5's own precondition check refuses one outside a loop body.
    """
    base = w1_plan.plan("npu1")
    hoisted = replace(w1_plan.A_TILE, loop_depth=0, ping_pong_candidate=False)
    k_loop = base.herd_body[2]
    assert isinstance(k_loop, LoopPlan) and k_loop.axis == "k0"
    variant = replace(
        base,
        buffers=(w1_plan.ACC, hoisted, w1_plan.B_TILE),
        herd_body=(base.herd_body[0], hoisted, base.herd_body[1],
                   replace(k_loop, body=tuple(n for n in k_loop.body if n is not w1_plan.A_TILE)),
                   base.herd_body[3]))
    assert isinstance(variant, MappingPlan)

    lines = m5_emit.emit(variant, "npu1").mlir.splitlines()
    alloc = next(i for i, line in enumerate(lines)
                 if "memref.alloc()" in line and "memref<32x16xf32, 2 : i32>" in line)
    k_for = next(i for i, line in enumerate(lines)
                 if "scf.for" in line and i > alloc
                 and "memref<16x32xf32, 2 : i32>" in lines[i + 1])
    assert alloc < k_for, "M5 must follow the plan, not hoist or sink an alloc"


@pytest.mark.fr("FR-E4", "FR-S13")
def test_no_buffer_resources_arg():
    """`buffer_resources=` is never passed: `air.api` raises (`_channel.py:563`)."""
    assert "buffer_resources" not in SOURCE
    assert "pad_before" not in SOURCE and "pad_after" not in SOURCE
    assert "packet_ids" not in SOURCE


# --------------------------------------------------------------------------------------------
# FR-E5, FR-E6 — multicast and memory spaces
# --------------------------------------------------------------------------------------------


@pytest.mark.fr("FR-E5")
def test_E5_broadcast_emitted():
    """The measured spelling, `[2 : index, 2 : index]` and not `[2, 2]` (finding N-5)."""
    assert "  air.channel @A2L1 [2, 1] {broadcast_shape = [2 : index, 2 : index]}" in text()
    assert "  air.channel @B2L1 [1, 2] {broadcast_shape = [2 : index, 2 : index]}" in text()
    assert "  air.channel @C2L3 [2, 2]" in text()


@pytest.mark.fr("FR-E6")
def test_E6_memory_spaces():
    """Every L1 memref carries `2 : i32`; no L2 memref appears; no `<herd>.shared()`."""
    spaces = set(re.findall(r"memref<[^>]*, (\d+) : i32>", text()))
    assert spaces == {"2"}, f"expected only the L1 memory space, got {sorted(spaces)}"
    assert "1 : i32>" not in text()
    assert ".shared()" not in SOURCE and "per_core" not in SOURCE


# --------------------------------------------------------------------------------------------
# FR-E7 — the text re-parses
# --------------------------------------------------------------------------------------------


@pytest.mark.requires_air_opt
@pytest.mark.fr("FR-E7", "FR-T5")
def test_E7_text_roundtrip(tmp_path):
    """`air-opt <file> -o /dev/null`: exit 0 **and** no `error:` line (FR-T5)."""
    if not m6.tool_available("air-opt"):
        pytest.skip("air-opt is not resolvable; install the toolchain (07-environment.md §2)")
    path = tmp_path / "w1.base.npu1.air.mlir"
    path.write_text(text(), encoding="utf-8")
    m6.verdict(m6.invoke([m6.tool("air-opt"), path, "-o", "/dev/null"], cwd=tmp_path))


# --------------------------------------------------------------------------------------------
# FR-E9 — errors
# --------------------------------------------------------------------------------------------


def _oversized_plan() -> MappingPlan:
    """W1 with an accumulator of 96 KB: past `L1_BYTES = 65536` (`_trace.py:100`)."""
    base = w1_plan.plan("npu1")
    big = replace(w1_plan.ACC, shape=(128, 192), bytes=128 * 192 * 4)
    return replace(base, buffers=(big, w1_plan.A_TILE, w1_plan.B_TILE),
                   herd_body=(big,) + base.herd_body[1:])


@pytest.mark.fr("FR-E9")
def test_E9_air_api_message_preserved():
    """`air.api`'s own L1 explanation reaches `details` verbatim, and nothing else does."""
    with pytest.raises(EmissionError) as excinfo:
        m5_emit.emit(_oversized_plan(), "npu1")
    assert_diagnostic(excinfo, code="EMIT-AIR-API", clause="build()",
                      details_keys=("air_api_message", "node", "workload"))
    message = excinfo.value.diagnostic.details["air_api_message"]
    assert "air.alloc([128, 192]" in message
    assert "64 KB of L1" in message
    assert "BufferPlan" in excinfo.value.diagnostic.details["node"]


@pytest.mark.fr("FR-E9")
def test_E9_error_carries_no_traceback():
    """HLD §4.3: no raw diagnostic and no traceback in the top-level message."""
    with pytest.raises(EmissionError) as excinfo:
        m5_emit.emit(_oversized_plan(), "npu1")
    reason = excinfo.value.diagnostic.reason
    assert "Traceback" not in str(excinfo.value)
    assert "air.alloc([128, 192]" not in reason
    assert reason == "air.api rejected the BufferPlan of 'gemm'"


@pytest.mark.fr("FR-E9")
def test_E9_verify_surfaced(monkeypatch):
    """A `module.operation.verify()` failure is `EMIT-VERIFY`, not a bare MLIR exception.

    No plan we can build reaches this path — M5 checks its own preconditions first and every
    W1-shaped module verifies — so the `RuntimeError` `LaunchContext._verify` raises
    (`python/air/api/_compile.py:174-180`) is raised in its place. What is under test is M5's
    classification of that message, which is the only signal the wheel gives.
    """
    from air.api._compile import LaunchContext

    diagnostic = ("'affine.apply' op using value defined outside the region\n"
                  "note: see current operation")
    original = LaunchContext.build

    def broken(self, target=None):
        raise RuntimeError(f"air.api emitted invalid IR -- this is a bug in the DSL, not in "
                           f"the kernel:\n{diagnostic}")

    monkeypatch.setattr(LaunchContext, "build", broken)
    with pytest.raises(EmissionError) as excinfo:
        m5_emit.emit(w1_plan.plan("npu1"), "npu1")
    monkeypatch.setattr(LaunchContext, "build", original)

    assert_diagnostic(excinfo, code="EMIT-VERIFY", clause="build()",
                      details_keys=("mlir_diagnostic", "air_api_message", "node", "workload"))
    assert diagnostic in excinfo.value.diagnostic.details["mlir_diagnostic"]
    assert "affine.apply" not in excinfo.value.diagnostic.reason


# --------------------------------------------------------------------------------------------
# FR-E10 — determinism
# --------------------------------------------------------------------------------------------


@pytest.mark.fr("FR-E10")
def test_E10_byte_identical():
    """Twice in this process, once in a fresh one under another PYTHONHASHSEED."""
    first = emit_w1_text("npu1")
    second = emit_w1_text("npu1")
    third = determinism.in_fresh_process(emit_w1_text, "npu1")
    assert first == second
    assert first == third


@pytest.mark.fr("FR-S20")
def test_m5_lazy_air_import():
    """Importing M5 must not import `air` (FR-S20)."""
    assert not determinism.in_fresh_process(air_in_sys_modules_after_importing_m5)


# --------------------------------------------------------------------------------------------
# D-14 — the emitter decides nothing
# --------------------------------------------------------------------------------------------


@pytest.mark.fr("FR-E1")
def test_emitter_makes_no_decisions():
    """P-1: M5 reads no field of the legality or schedule contracts, and no kernel size."""
    for forbidden in ("plan.mapping", "LegalMapping", "KernelModel", "ScheduleModel",
                      "if tx ==", ".schedule", ".kernel", "physical_herd", "repeats"):
        assert forbidden not in SOURCE, f"M5's source names {forbidden!r}"


def _air_names(source: str) -> set[str]:
    """Every `air.…` / `self.air.…` attribute path the *code* names (docstrings excluded)."""
    names: set[str] = set()

    def dotted(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = dotted(node.value)
            return f"{base}.{node.attr}" if base else None
        return None

    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Attribute):
            continue
        path = dotted(node)
        if path is None:
            continue
        if path.startswith("self.air."):
            path = path[len("self."):]
        if path.startswith("air."):
            names.add(path)
    return names


ALLOWED_AIR_NAMES = {
    # design/03-lld-M5-emitter.md §3.2, one entry per row that names an `air.` construct
    "air.tensor",                 # row 1
    "air.launch",                 # row 2
    "air.segment",                # row 3
    "air.herd",                   # row 4
    "air.channel",                # rows 5a, 5b, 5c
    "air.alloc",                  # row 6
    "air.sequential",             # row 7
    "air.ops.branch",             # rows 11, 11b
    "air.ops.maximum",            # row 12, via §3.6
    "air.ops.minimum",            # row 12, via §3.6
    "air.ops.select",             # row 12, via §3.6
    # `Select`'s own predicate. §3.6 line 17 writes it as `EMIT_EXPR(l) <cmp> EMIT_EXPR(r)`,
    # which holds for `<`, `<=`, `>`, `>=` and **not** for `==`/`!=`: those are deliberately
    # left undefined on a buffer value (`_value.py:795-805`), so `x == y` is Python's identity
    # comparison and `ops.select` rejects the `bool` it produces by name (`ops.py:809-816`).
    # W3 is the first `Select` in the project; erratum recorded in design/PROGRESS-B.md, P4.
    "air.ops.equal",              # row 12, via §3.6 line 17
    "air.ops.not_equal",          # row 12, via §3.6 line 17
    # the element types rows 1 and 6 pass as their `dtype` argument; §3.2 names the argument
    # rather than the object, and `air.api` re-exports exactly these five of the ten it has.
    "air.f32", "air.f16", "air.bf16", "air.i32", "air.i8",
    # prefixes of the above, which the AST walk also yields
    "air.ops",
}


@pytest.mark.fr("FR-E1")
def test_emitter_construct_closure():
    """P-4: every `air.` construct M5 names appears in §3.2's table."""
    used = _air_names(SOURCE)
    assert used, "the walk found no air.* name at all, so it is not testing anything"
    assert used <= ALLOWED_AIR_NAMES, f"outside §3.2's table: {sorted(used - ALLOWED_AIR_NAMES)}"
    for required in ("air.tensor", "air.launch", "air.segment", "air.herd", "air.channel",
                     "air.alloc", "air.sequential", "air.ops.branch"):
        assert required in used, f"{required} is in §3.2's table and M5 never names it"


# --------------------------------------------------------------------------------------------
# Target pass-through and loop kinds
# --------------------------------------------------------------------------------------------


@pytest.mark.fr("FR-E1")
def test_E_target_passthrough(monkeypatch):
    """P-7, D-13: `"npu2"` reaches `build()` unchanged; `"auto"` is refused, never resolved."""
    from air.api._compile import LaunchContext

    seen = []
    original = LaunchContext.build

    def spy(self, target=None):
        seen.append(target)
        return original(self, target=target)

    monkeypatch.setattr(LaunchContext, "build", spy)
    result = m5_emit.emit(w1_plan.plan("npu2"), "npu2")
    assert seen == ["npu2"]
    assert result.target == "npu2"
    assert "in (%arg15=%c2, %arg16=%c2_10)" in result.mlir      # the npu2 herd is 2x2

    with pytest.raises(EmissionError) as excinfo:
        m5_emit.emit(w1_plan.plan("npu1"), "auto")
    assert_diagnostic(excinfo, code="EMIT-AIR-API", clause="build()",
                      mentions=("auto",), details_keys=("given", "accepted"))
    assert seen == ["npu2"], "auto must be refused before anything is traced"


def _sequential_emissions(nodes, trips: int, herd_body) -> int:
    """How many `scf.for` the plan itself asks for: one per *emission* of a sequential loop."""
    from spatial.model import BranchNode, HerdPlan

    count = 0
    for node in nodes:
        if isinstance(node, LoopPlan):
            if node.kind == "sequential":
                count += trips + _sequential_emissions(node.body, trips, herd_body)
            else:
                span = len(range(node.lo.const, node.hi.const, node.step.const))
                count += _sequential_emissions(node.body, trips * span, herd_body)
        elif isinstance(node, BranchNode):
            count += (_sequential_emissions(node.then, trips, herd_body)
                      + _sequential_emissions(node.otherwise, trips, herd_body))
        elif isinstance(node, HerdPlan):
            count += _sequential_emissions(herd_body, 1, herd_body)
    return count


@pytest.mark.parametrize("target", ["npu1", "npu2"])
@pytest.mark.fr("FR-S14")
def test_sequential_emits_scf_for(target):
    """One `scf.for` per `LoopPlan(kind="sequential")` emission, and none for `"unrolled"`.

    Plus **one** for the strip-mine loop `air.api` wraps the herd body in when `shape=` is
    smaller than the grid (finding N-7, `_trace.py:1675-1697`): on npu1 W1's 2x2 grid runs on a
    1x2 herd, on npu2 it does not. That loop is `air.api`'s, not the plan's, and M5 neither
    asks for it nor reconciles it (P-3).
    """
    plan = w1_plan.plan(target)
    strip = 1 if any(g // s > 1 for g, s in zip(plan.herd.grid, plan.herd.shape)) else 0
    expected = _sequential_emissions(plan.segment_body, 1, plan.herd_body) + strip
    assert text(target).count("scf.for") == expected
    # The unrolled loops are real: without them the expectation would be trivially satisfied.
    assert any(isinstance(n, LoopPlan) and n.kind == "unrolled" for n in plan.segment_body)


# --------------------------------------------------------------------------------------------
# The precondition checks of M5 §5: our defect, never the user's
# --------------------------------------------------------------------------------------------


@pytest.mark.fr("FR-E9")
def test_internal_consistency_strides():
    """A `Region` whose strides are not row-major is a bug report, not a clause edit (§3.7)."""
    base = w1_plan.plan("npu1")
    bad = replace(w1_plan.A_PUT,
                  region=replace(w1_plan.A_PUT.region, strides=(1, 64)))
    fill_a = replace(w1_plan.FILL_A,
                     body=(replace(w1_plan.FILL_A.body[0], body=(bad,)),))
    variant = replace(
        base,
        channels=(replace(base.channels[0], sites=(bad, w1_plan.A_GET)),) + base.channels[1:],
        segment_body=(fill_a,) + base.segment_body[1:])

    with pytest.raises(EmissionError) as excinfo:
        m5_emit.emit(variant, "npu1")
    assert_diagnostic(excinfo, code="EMIT-AIR-API", clause="build()",
                      mentions=("row-major",), details_keys=("internal_consistency", "strides"))
    assert "report it" in excinfo.value.diagnostic.fix


@pytest.mark.fr("FR-E3")
def test_internal_consistency_ping_pong_outside_a_loop():
    """`06-interfaces.md` §5.6 invariant 4: a candidate is a direct child of a `LoopPlan`."""
    base = w1_plan.plan("npu1")
    k_loop = base.herd_body[2]
    variant = replace(
        base,
        herd_body=(base.herd_body[0], w1_plan.A_TILE, base.herd_body[1],
                   replace(k_loop, body=tuple(n for n in k_loop.body if n is not w1_plan.A_TILE)),
                   base.herd_body[3]))
    with pytest.raises(EmissionError) as excinfo:
        m5_emit.emit(variant, "npu1")
    assert_diagnostic(excinfo, code="EMIT-AIR-API", clause="build()",
                      mentions=("ping-pong", "invariant 4"),
                      details_keys=("internal_consistency", "buffer"))


# --------------------------------------------------------------------------------------------
# W3 — the rows of M5 §7 the wavefront plan unblocks. Added by B at P4.
# --------------------------------------------------------------------------------------------


def _nodes(body):
    """Every node of a plan body, descending into loops and both arms of a branch."""
    out = []
    for node in body:
        out.append(node)
        if isinstance(node, LoopPlan):
            out += _nodes(node.body)
        elif isinstance(node, BranchNode):
            out += _nodes(node.then) + _nodes(node.otherwise)
    return out


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


@pytest.mark.fr("FR-E1")
@pytest.mark.parametrize("workload", ["w2", "w3"])
@pytest.mark.parametrize("target", ["npu1", "npu2"])
def test_E1_herd_arity(target, workload):
    """A 1-D herd body takes **one** positional coordinate; the tile space is 2-D (finding N-10).

    `_positional_arity` counts declared positional parameters (`_trace.py:725-731`), so a `*args`
    body would read as arity 0 and `air.api` would reject it — which is why M5 declares `body(c0)`
    and `body(c0, c1)` rather than one variadic function. What reaches the IR is always a 2-D
    tile space, with the second extent 1 on a rank-1 herd. §7's row names W2 (`grid(2)`); W3
    (`grid(4)`) is the same rule at another extent.
    """
    fixture, extent = {"w2": (w2_legal, "2"), "w3": (w3_legal, "4")}[workload]
    plan = m4.plan(fixture.legal(target))
    body = {"w2": w2_text, "w3": w3_text}[workload](target)
    assert plan.herd.coords == ("tx",)
    herd = next(line for line in body.splitlines() if "air.herd" in line)
    assert re.search(r"tile \(%\w+, %\w+\) in \(%\w+=%(\w+), %\w+=%(\w+)\)", herd), herd
    extents = re.search(r"in \(%\w+=%(\w+), %\w+=%(\w+)\)", herd).groups()
    constants = dict(re.findall(r"%(\w+) = arith\.constant (\d+) : index", body))
    assert [constants[name] for name in extents] == [extent, "1"]


@pytest.mark.fr("FR-M4", "FR-L14")
def test_E_peel_is_plan_driven():
    """`T = 5`'s text is the loop `step 2` to `4` plus one straight-line `STEP`, and `T` drains.

    M5 has no parity test of its own — line 20 of §6.3's `if T is odd` is the **plan** already
    holding or not holding the peeled nodes (§3.6.1 line 8), and the emitter walks whatever is
    in `plan.herd_body`. The lint below is that claim: M5's source names neither `%` nor `odd`.
    """
    even, odd = w2_text(), m5_emit.emit(m4.plan(w2_legal.legal("npu1", T=5)), "npu1").mlir
    loop = re.compile(r"scf\.for %\w+ = %(\w+) to %(\w+) step %(\w+)")
    constants = dict(re.findall(r"%(\w+) = arith\.constant (\d+) : index", odd))
    timestep = [m.groups() for m in loop.finditer(odd)
                if constants.get(m.group(3)) == "2"]
    assert len(timestep) == 1, "one timestep loop, unrolled by two"
    assert [constants[name] for name in timestep[0]] == ["0", "4", "2"]

    # `T` drains in total: 2 per trip inside the loop, plus the peeled one after it
    assert odd.count("air.channel.put  @UOut[") == 3          # two in the loop, one peeled
    assert even.count("air.channel.put  @UOut[") == 2
    assert odd.count("air.channel.get  @UOut[") == 2          # one per PE, in the drain loop
    drains = [m.groups() for m in loop.finditer(odd) if constants.get(m.group(3)) == "1"
              and constants.get(m.group(2)) == "5"]
    assert len(drains) == 2, "one plane drain per PE, air.sequential(0, T)"
    # the peeled STEP is four more guarded sites and one more update nest than the even text
    assert odd.count("scf.if") == even.count("scf.if") + 4 == 12
    assert odd.count("arith.mulf") == even.count("arith.mulf") + 1 == 3
    assert not re.search(r"\bT\s*%\s*2\b|\bodd\b|\bparity\b", SOURCE), "M5 has no parity test"


@pytest.mark.fr("FR-M5")
def test_M5_uses_branch():
    """FR-M5's second acceptance: the guards are `scf.if`, and M5's source has no `if tx ==`.

    `bool()` on a `Condition` is refused (`_cond.py:41-45`) because a herd body is traced once
    for every core at once, so a Python `if` on a coordinate cannot be written even by mistake;
    `air-to-aie` folds the branch away once the coordinate is a literal (`_cond.py:49-54`).
    """
    text = w3_text()
    assert "scf.if" in text
    assert text.count("scf.if") == 4                     # head and tail, once per unrolled row
    assert "if tx ==" not in SOURCE and "if coord" not in SOURCE
    # the guard is a comparison on the herd coordinate, not on a loop variable
    assert re.search(r"arith\.cmpi eq, %\w+, %c0\w* : index", text)
    assert re.search(r"arith\.cmpi eq, %\w+, %c3\w* : index", text)


@pytest.mark.fr("FR-E2", "FR-D9")
def test_E_select_emitted():
    """The substitution score is `arith.select`, a **value**, with no `scf.if` in the `j` loop.

    FR-D9: a conditional over buffer data is an expression. `ops.select` decides per element and
    evaluates both sides; `ops.branch` decides per core and runs one — the two halves of
    if-conversion, and picking the wrong one is what `_cond.py`'s table exists to prevent.
    """
    lines = w3_text().splitlines()
    starts = [i for i, line in enumerate(lines) if "arith.select" in line]
    assert len(starts) == 2, "one per unrolled row"
    for start in starts:
        opener = next(i for i in range(start, -1, -1)
                      if "scf.for" in lines[i] and _indent(lines[i]) < _indent(lines[start]))
        closer = next(i for i in range(start, len(lines))
                      if lines[i].strip() == "}" and _indent(lines[i]) == _indent(lines[opener]))
        body = lines[opener + 1:closer]
        assert "scf.if" not in "\n".join(body), "the score is a value, not control flow"
        assert any("arith.cmpi eq" in line and ": i32" in line for line in body)
        assert sum("arith.maxsi" in line for line in body) == 3     # the 4-ary max, folded


@pytest.mark.fr("FR-E1")
def test_E_branch_node():
    """One `scf.if` per `BranchNode`; `otherwise()` only when the arm is non-empty; and nesting
    nests."""
    plan = m4.plan(w3_legal.legal())
    branches = [node for node in _nodes(plan.herd_body) if isinstance(node, BranchNode)]
    assert len(branches) == 4 and all(branch.otherwise for branch in branches)
    assert w3_text().count("scf.if") == len(branches)

    row = plan.herd_body[9]
    head = row.body[0]
    trimmed = replace(plan, herd_body=plan.herd_body[:9] + (
        replace(row, body=(replace(head, otherwise=()),) + row.body[1:]),))
    text = m5_emit.emit(trimmed, "npu1").mlir
    assert text.count("scf.if") == 4                     # the region pair is still one scf.if
    assert (text.count("air.channel.get  @West[")
            == w3_text().count("air.channel.get  @West[") - 1), "the empty arm emitted nothing"

    nested = replace(plan, herd_body=plan.herd_body[:9] + (
        replace(row, body=(replace(head, then=(replace(head, otherwise=()),), otherwise=()),)
                + row.body[1:]),))
    lines = m5_emit.emit(nested, "npu1").mlir.splitlines()
    assert sum("scf.if" in line for line in lines) == 5
    outer = next(i for i, line in enumerate(lines) if "scf.if" in line)
    inner = next(i for i in range(outer + 1, len(lines)) if "scf.if" in lines[i])
    assert _indent(lines[inner]) > _indent(lines[outer])


@pytest.mark.fr("FR-E8", "FR-M6")
@pytest.mark.parametrize("target", ["npu1", "npu2"])
def test_E8_cascade_text(target):
    """Row 5c for real: `air.channel @CascadeK [3] {channel_type = "npu_cascade"}`, no broadcast.

    FR-E8's acceptance as written expects the attribute **three times**; one bundle of
    `size=(PK-1,)` prints it once, and the fact that matters — three physical links — is
    `ir_facts.cascade_channels == 3` after `air-to-aie` (`03-lld-B-open-questions.md` §4,
    ruling R-F-3). `test_I_cascade_channels` is the other half; this is the text half.
    """
    text = flip_text(target)
    assert 'air.channel @CascadeK [3] {channel_type = "npu_cascade"}' in text
    declaration = next(line for line in text.splitlines() if "@CascadeK" in line
                       and line.strip().startswith("air.channel"))
    assert "broadcast_shape" not in declaration, "_channel.py:170-177 forbids it on a cascade"
    assert text.count('channel_type = "npu_cascade"') == 1, "one bundle, not PK-1 channels"
    # the other three channels are plain, and the chain carries the accumulator whole
    for name, size in (("A2L1", "[4]"), ("B2L1", "[4]"), ("C2L3", "[1]")):
        assert f"air.channel @{name} {size}\n" in text
    assert text.count("air.channel.put  @CascadeK[") == 2      # head, and the middle PEs
    assert text.count("air.channel.get  @CascadeK[") == 1


@pytest.mark.fr("FR-E1")
def test_E_branch_node_flip():
    """The flip's two `BranchNode`s **nest**: `scf.if` inside the `else` of another `scf.if`.

    `_cond.py:57-60` has no `and`, so `tx != head and tx == tail` is written as nesting, and
    that is what M4 built (§3.6.3 lines 19-23). Every `otherwise` is non-empty here, so both
    branches emit their region pair.
    """
    plan = m4.plan(w1flip_legal.legal())
    branches = [node for node in _nodes(plan.herd_body) if isinstance(node, BranchNode)]
    assert len(branches) == 2 and all(branch.otherwise for branch in branches)
    lines = flip_text().splitlines()
    ifs = [i for i, line in enumerate(lines) if "scf.if" in line]
    assert len(ifs) == len(branches) == 2
    assert _indent(lines[ifs[1]]) > _indent(lines[ifs[0]]), "the tail guard nests in the else"
    # the guards compare the herd coordinate against 0 (head) and 3 (tail)
    assert re.search(r"arith\.cmpi eq, %\w+, %c0\w* : index", flip_text())
    assert re.search(r"arith\.cmpi eq, %\w+, %c3\w* : index", flip_text())
    # ...and the accumulate nest M4 synthesised sits inside the else arm, as two scf.for
    outer_else = next(i for i in range(ifs[0], len(lines)) if lines[i].strip().endswith("} else {")
                      and _indent(lines[i]) == _indent(lines[ifs[0]]))
    between = lines[outer_else:ifs[1]]
    assert sum("scf.for" in line for line in between) == 2
    assert sum("arith.addf" in line for line in between) == 1


@pytest.mark.fr("FR-E1", "FR-M7")
def test_E_tensor_order():
    """`air.tensor` in `plan.tensors` order — `q`, `r`, `S` — and the other order is rejected.

    `_check_interface` raises a bare `RuntimeError` (`_compile.py:226-240`) whose message lists
    the interface order it saw; M5 lets it through as `EMIT-AIR-API` with the text verbatim,
    and M4's invariant 6 is what makes it unreachable from a derived plan.
    """
    plan = m4.plan(w3_legal.legal())
    assert [t.name for t in plan.tensors] == ["q", "r", "S"]
    signature = next(line for line in w3_text().splitlines() if "func.func @sw(" in line)
    assert re.findall(r"memref<[^>]*>", signature) == ["memref<32xi32>", "memref<32xi32>",
                                                       "memref<33x33xi32>"]
    reordered = replace(plan, tensors=(plan.tensors[2], plan.tensors[0], plan.tensors[1]))
    with pytest.raises(EmissionError) as excinfo:
        m5_emit.emit(reordered, "npu1")
    assert_diagnostic(excinfo, code="EMIT-AIR-API", clause="build()",
                      details_keys=("air_api_message", "workload"))
    assert "output tensors must be declared after all input tensors" in (
        excinfo.value.diagnostic.details["air_api_message"])


@pytest.mark.fr("FR-E10")
def test_E10_byte_identical_w3():
    """W3 twice in this process, and once in a fresh one under another PYTHONHASHSEED."""
    assert w3_text("npu1") == emit_w3_text("npu1") == determinism.in_fresh_process(
        emit_w3_text, "npu1")
