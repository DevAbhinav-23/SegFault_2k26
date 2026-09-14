"""Level I — `air-opt` inspection. Spec: design/04-test-plan.md §3.2.

Stub written by B at P0c to unblock; **Person C owns this file**.
"""

from __future__ import annotations

import re
from importlib.resources import files
from pathlib import Path

import pytest

from spatial import m6_tools as m6
from spatial.model import ToolchainError
from tests.helpers.diagnostics import assert_diagnostic
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


W1_FACTS = ("pingpong_unroll", "hoist_alloc_count",
            "broadcast_pattern_count", "pingpong_iter_args")
"""The fields `03-lld-M5-emitter.md` §3.8 owns for W1. `lock_init_histogram` is D6's.

**`cascade_channels` is gone from this tuple as of P6** (ruling R-F-3): the fact is now read off
the `aie` pipeline, because it counts `aie.cascade_flow` ops, and W1's module does **not** lower
through `air-place-herds,air-to-aie` on its own — its `C2L3` bundle index goes through the
`repeats` strip-mine `affine_map`, which `air-to-aie` cannot fold to a constant, so the pass
replaces the puts with `air.wait_all` and then fails with *'air.channel.get' op failed to get
MM2S tile for L3 allocation*. `aircc` reaches `air-to-aie` with the dependency and
dma-to-channel passes already run; `PIPELINES["aie"]` does not. A GEMM with no cascade has
nothing to say about cascade flows, so the fact is asserted on W3 (0) and the flip (3) instead."""


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
@pytest.mark.parametrize("workload", ["w1", "w2", "w3", "flip"])
@pytest.mark.fr("FR-E5")
def test_I_broadcast_count(tmp_path, workload):
    """R-04's tripwire: declaring `broadcast_shape` bypasses the detector (D-4, VF §S9).

    **Measured 0** on our emitted W1 (`03-lld-M5-emitter.md` §3.8, finding N-8) and, for the
    same reason, on W3 — `QIn` carries `broadcast_shape = [4]`, so the pass has nothing left to
    derive. W2's 0 has a different reason and is worth having: **no** W2 channel is a broadcast
    at all, so the pass has nothing to find rather than nothing left to find. If any number ever
    changes, record the new one: it means `air-broadcast-detection` now walks something we emit,
    and the reason belongs in the commit message, not in a patched expectation.

    `flip` was added by B at P7 so this one test carries `04-test-plan.md` §8 item 6's "for all
    four variants" on its own; the flip's 0 was already frozen in its `ir_facts` golden.
    """
    _require_air_opt()
    module = {"w1": w1_module, "w2": w2_module,
              "w3": w3_module, "flip": flip_module}[workload](tmp_path)
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


# --------------------------------------------------------------------------------------------
# W2 — the halo exchange, and experiment **E1** run on our own module. Added by B at P5.
# --------------------------------------------------------------------------------------------


def w2_module(tmp_path: Path, target: str = "npu1") -> Path:
    """W2's emitted module, on disk, which is what `air-opt` reads."""
    from spatial import m4_mapping as m4, m5_emit
    from tests.fixtures.mappings import w2_legal

    path = tmp_path / f"w2.base.{target}.air.mlir"
    path.write_text(m5_emit.emit(m4.plan(w2_legal.legal(target)), target).mlir, encoding="utf-8")
    return path


W2_FACTS = ("broadcast_pattern_count", "pingpong_unroll", "lock_init_histogram")
"""What §3.8 has to say about a halo: no derived broadcast (nothing carries `broadcast_shape`),
no ping-pong (`DEPTH` is 0 for the `cur`/`next` pair, so there is no candidate loop — D-5's
second meaning, R-W2-4), and the lock inits `air-to-aie` allocated."""

_DEF = re.compile(r"^%([\w.]+) = ([\w.]+)(?: async)?\s*(?:\[([^\]]*)\])?")
_YIELD = re.compile(r"^scf\.yield %([\w.]+)")
_SITE = re.compile(r"^%[\w.]+ = air\.channel\.(put|get) async\s*(?:\[[^\]]*\])?\s*@(\w+)")


class _Flow:
    """The token-dependency graph of one `air-opt -air-dependency` module.

    MLIR prints SSA names **per region**, so `%57` inside one `scf.if` and `%57` inside the next
    are different values; a flat name table would fuse a put's token with a get's and make the E1
    check answer a different question. Names are therefore resolved lexically — innermost region
    first — and every definition gets a unique id.

    A token reaches a consumer either through that consumer's own dependency list or through an
    `scf.if` result: a guarded put yields its token out of the region, so whoever takes the
    result takes the put's. `scf.for` yields are deliberately **not** followed: that edge is the
    loop-carried one, and E1's question is about one iteration.
    """

    def __init__(self, text: str) -> None:
        self.dependents: dict[int, set[int]] = {}
        self.sites: list[tuple[str, str, int, list[int]]] = []
        self._scopes: list[tuple[dict[str, int], int | None]] = [({}, None)]
        self._next = 0
        for raw in text.splitlines():
            self._line(raw.strip())

    def _resolve(self, name: str) -> int:
        for names, _result in reversed(self._scopes):
            if name in names:
                return names[name]
        return -hash(name) % (1 << 30)             # a block argument: a key, never a target

    def _define(self, name: str) -> int:
        self._next += 1
        self._scopes[-1][0][name] = self._next
        return self._next

    def _line(self, line: str) -> None:
        if line.startswith("}") and len(self._scopes) > 1:
            carried = self._scopes.pop()[1]        # `} else {` re-opens the same scf.if
            if line.endswith("{"):
                self._scopes.append(({}, carried))
                return
        yielded = _YIELD.match(line)
        if yielded and self._scopes[-1][1] is not None:
            self.dependents.setdefault(self._resolve(yielded.group(1)),
                                       set()).add(self._scopes[-1][1])
        site = _SITE.match(line)
        defined = _DEF.match(line)
        dependencies = [self._resolve(name)
                        for name in re.findall(r"%([\w.]+)", defined.group(3) or "")
                        ] if defined else []
        result = self._define(defined.group(1)) if defined else None
        if site and result is not None:
            self.sites.append((site.group(1), site.group(2), result, dependencies))
        for dependency in dependencies:
            self.dependents.setdefault(dependency, set()).add(result)
        if line.endswith("{"):
            self._scopes.append(({}, result if defined and defined.group(2) == "scf.if"
                                 else None))

    def reachable(self, seeds: list[int]) -> set[int]:
        """Every value transitively derived from one of `seeds`."""
        out: set[int] = set()
        frontier = list(seeds)
        while frontier:
            for consumer in self.dependents.get(frontier.pop(), ()):
                if consumer not in out:
                    out.add(consumer)
                    frontier.append(consumer)
        return out


@pytest.mark.requires_air_opt
@pytest.mark.fr("FR-M4")
def test_I_w2_no_put_get_token_edge(tmp_path):
    """**Experiment E1, on our own module**: no halo get waits on a halo put's token.

    `hackathon/HANDOFF.md`'s recipe: run `air-opt -pass-pipeline='builtin.module(air-dependency)'`
    and check that no token edge joins a PE's put to its own get. That is what makes the
    put-before-get order of §3.6.1 safe rather than a rendezvous — a token from the puts into the
    gets would serialise the exchange and reintroduce exactly the deadlock VF §C cleared it of,
    and `air.api` offers no asynchronous form to select (B-P12), so the safety rests entirely on
    what this pass builds.

    The check is mechanical: taint every value transitively derived from a halo put's token —
    through dependency lists and through the `scf.if` result its `scf.yield` feeds — and assert
    no halo get's dependency list names a tainted value. A *negative* cannot be built through
    the plan: `ChannelSite.depends_on` reaches `air.api` as `dependency=`, which is validated and
    then unused by `_emit` (B-P12), so the puts and the gets are emitted identically and this is
    a measurement of the **lowering**, which is exactly what E1 asks about.
    """
    _require_air_opt()
    out = tmp_path / "dependency.mlir"
    pipeline = "builtin.module(air-dependency)"
    m6.verdict(m6.invoke([m6.tool("air-opt"), w2_module(tmp_path),
                          f"-pass-pipeline={pipeline}", "-o", out], cwd=tmp_path))
    flow = _Flow(out.read_text(encoding="utf-8"))

    halo = ("ToNorth", "ToSouth")
    puts = [s for s in flow.sites if s[0] == "put" and s[1] in halo]
    gets = [s for s in flow.sites if s[0] == "get" and s[1] in halo]
    assert len(puts) == len(gets) == 4, flow.sites   # two links x two phases, unrolled by two
    assert len(flow.sites) == 15, "2 UIn puts, 1 UIn get, 8 halo, 2 UOut puts, 2 UOut gets"
    assert all(dependencies for _kind, _channel, _result, dependencies in gets), \
        "a get with no dependency at all would make this vacuous"

    tainted = flow.reachable([result for _kind, _channel, result, _deps in puts])
    assert len(tainted) >= 2 * len(puts), (
        "each put's token should reach at least its air.wait_all and its scf.if result; "
        "a taint that stops at the put means the walk is not following anything")
    for _kind, channel, result, dependencies in gets:
        assert not set(dependencies) & tainted, (
            f"a {channel} get waits on a value derived from a halo put's token — "
            f"E1's rendezvous, which VF §C's verdict says the lowering does not build")


@pytest.mark.requires_air_opt
@pytest.mark.fr("FR-E5", "FR-M4", "FR-T6")
def test_I_facts_golden_w2(tmp_path):
    """W2's §3.8 facts, with the pipeline each came from, frozen as one small file.

    `pingpong_unroll` is asserted **0** rather than omitted: `double_buffer("U")` is the explicit
    `cur`/`next` pair (D-5's second meaning), the pair sits outside the timestep loop, and
    `isPingPongCandidate` therefore has no candidate loop to label. A number appearing here later
    would mean the pass started labelling something we did not ask it to.
    """
    _require_air_opt()
    require_pin()
    facts = m6.ir_facts(str(w2_module(tmp_path)), "npu1", W2_FACTS, workdir=tmp_path)
    assert facts["pingpong_unroll"] == 0, "the cur/next pair is not an isPingPongCandidate"
    assert facts["broadcast_pattern_count"] == 0, "no W2 channel carries a broadcast_shape"
    assert sum(facts["lock_init_histogram"].values()) > 0
    assert_golden("w2.base.npu1.ir_facts.json", facts, kind="json")


# --------------------------------------------------------------------------------------------
# W1-flip — the cascade chain. Spec: design/03-lld-M5-emitter.md §3.8. Added by B at P6.
# --------------------------------------------------------------------------------------------


def flip_module(tmp_path: Path, target: str = "npu1") -> Path:
    """The flip's emitted module, on disk, which is what `air-opt` reads."""
    from spatial import m4_mapping as m4, m5_emit
    from tests.fixtures.mappings import w1flip_legal

    path = tmp_path / f"w1.flip.{target}.air.mlir"
    path.write_text(m5_emit.emit(m4.plan(w1flip_legal.legal(target)), target).mlir,
                    encoding="utf-8")
    return path


FLIP_FACTS = ("broadcast_pattern_count", "cascade_channels", "hoist_alloc_count",
              "lock_init_histogram", "pingpong_unroll")
"""§3.8's fields for the flip. `pingpong_iter_args` is left out for the same reason W3 leaves
the ping-pong facts out: the `transform` pipeline's number is W1's fact, and what the flip adds
is the chain."""


@pytest.mark.requires_air_opt
@pytest.mark.fr("FR-E8", "FR-M6")
def test_I_cascade_channels(tmp_path):
    """FR-E8's honest form: **three** `aie.cascade_flow` ops after `air-to-aie` (ruling R-F-3).

    One bundle of `size=(PK-1,)` prints `channel_type = "npu_cascade"` once before lowering and
    `air-to-aie` splits it into three `@channel_N [1, 1]` bundles, so the attribute occurs
    **four** times in the lowered text and counting the string would answer a different
    question. The links themselves are what FR-M6 asks for, and they ascend.
    """
    _require_air_opt()
    facts = m6.ir_facts(str(flip_module(tmp_path)), "npu1", ["cascade_channels"],
                        workdir=tmp_path)
    assert facts["cascade_channels"] == 3
    assert facts["_pipeline_aie"] == m6.PIPELINES["aie"].format(
        target="npu1", **m6.AIE_GEOMETRY["npu1"])
    lowered = (tmp_path / "aie.mlir").read_text(encoding="utf-8")
    assert lowered.count('channel_type = "npu_cascade"') == 4, "the string is not the fact"
    assert [line.strip() for line in lowered.splitlines() if "aie.cascade_flow" in line] == [
        "aie.cascade_flow(%tile_2_2, %tile_3_2)",
        "aie.cascade_flow(%tile_1_2, %tile_2_2)",
        "aie.cascade_flow(%tile_0_2, %tile_1_2)"]
    # ...and W3, which has no chain, reads 0 off the same pipeline
    assert m6.ir_facts(str(w3_module(tmp_path)), "npu1", ["cascade_channels"],
                       workdir=tmp_path)["cascade_channels"] == 0


@pytest.mark.requires_air_opt
@pytest.mark.fr("FR-E3", "FR-M6")
def test_I_pingpong_labels_flip(tmp_path):
    """The `i0` loop is labelled `unroll = 2 : i32`, and **one** alloc is hoisted, not two.

    That difference is the flip, measured in the pass's own output: `b` is allocated above the
    streaming loop and never re-fetched, so it is not a candidate, while `a` is a direct child
    of the 2-trip `i0` loop whose first touch is a `get`. W1 reads `hoist_alloc_count = 2`
    because both its tiles stream.
    """
    _require_air_opt()
    facts = m6.ir_facts(str(flip_module(tmp_path)), "npu1",
                        ["pingpong_unroll", "hoist_alloc_count"], workdir=tmp_path)
    assert facts["pingpong_unroll"] == 2, "the i0 loop has 2 trips and is labelled"
    assert facts["hoist_alloc_count"] == 1, "only `a` is a ping-pong candidate (§6.2)"


@pytest.mark.requires_air_opt
@pytest.mark.fr("FR-T6")
def test_trace_requires_launch(tmp_path):
    """A module with no `air.launch` raises naming Runner.cpp, never segfaults (M6 §3.8)."""
    _require_air_opt()
    mod = tmp_path / "no_launch.mlir"
    mod.write_text("module { func.func @f() { func.return } }", encoding="utf-8")
    with pytest.raises(ToolchainError) as excinfo:
        m6.trace(str(mod), str(tmp_path / "arch.json"), "f", workdir=tmp_path)
    assert_diagnostic(excinfo, code="TOOL-AIRCC-FAILED", clause="build()",
                      mentions=("air.launch", "Runner.cpp:547-551"),
                      details_keys=("citation",))


@pytest.mark.requires_air_opt
@pytest.mark.fr("FR-E5", "FR-E8", "FR-M6", "FR-T6")
def test_I_facts_golden_flip(tmp_path):
    """The flip's §3.8 facts, with the pipeline each came from, frozen as one small file."""
    _require_air_opt()
    require_pin()
    facts = m6.ir_facts(str(flip_module(tmp_path)), "npu1", FLIP_FACTS, workdir=tmp_path)
    assert facts["cascade_channels"] == 3
    assert facts["broadcast_pattern_count"] == 0, "no flip channel carries a broadcast_shape"
    assert sum(facts["lock_init_histogram"].values()) > 0
    assert_golden("w1.flip.npu1.ir_facts.json", facts, kind="json")
