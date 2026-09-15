"""M0 — the frozen shared contract. Spec: design/06-interfaces.md §1-§6; design/04-test-plan.md §2.

One case per enforced invariant of `spatial.model`'s docstring, plus the frozen-ness, JSON,
rendering and catalogue guards. `test_invariants_match_the_docstring` asserts that the set of
invariant ids exercised here equals the set the module documents.
"""

from __future__ import annotations

import dataclasses
import json
import re

import pytest

from spatial import m6_tools
from spatial import model as m
from spatial.model import (
    AccessMap,
    Axis,
    AxisRef,
    BinOp,
    BranchNode,
    BufferPlan,
    ChannelPlan,
    ChannelSite,
    ClauseError,
    Const,
    Dependence,
    Diagnostic,
    Dtype,
    EmissionError,
    EmitResult,
    ExchangeClause,
    Expr,
    GrammarError,
    Guard,
    HerdPlan,
    KernelModel,
    LegalityError,
    LegalMapping,
    Load,
    LoopPlan,
    MappingError,
    MappingPlan,
    MappingSummary,
    MaxMin,
    Neg,
    Param,
    ReductionSpec,
    Region,
    ScheduleModel,
    Select,
    SpatialError,
    Statement,
    StoreNode,
    StreamClause,
    ToolchainError,
    WindowClause,
    from_json,
    to_json,
)
from tests.helpers.determinism import in_fresh_process
from tests.helpers.diagnostics import CATALOGUE as DOC_CATALOGUE
from tests.helpers.diagnostics import assert_diagnostic

# ---------------------------------------------------------------------------------------------
# One minimal valid construction per class (design/06-interfaces.md §1-§6)
# ---------------------------------------------------------------------------------------------

E0 = Expr((), 0)
E1 = Expr((), 1)
E4 = Expr((), 4)

PARAM = Param("A", Dtype.f32, (4, "M"), True)
AXIS = Axis("i", E0, E4, E1, 4, None, 0)
ACCESS = AccessMap("A", ((1, 0),), (E0,), True)
# §2.4 at CONTRACT_VERSION 5: the value stored, over kernel-level operands — `A[i] = A[i]`
STATEMENT = Statement("assign", ACCESS, (ACCESS,), Load("A", (E0,)), None, ("i",), 7)
DEPENDENCE = Dependence((1, 0), "RAW", "A")
REDUCTION = ReductionSpec("A", ((1, 0),), ((0, 1),), "+")
KERNEL = KernelModel("k", "def k(): ...", (PARAM,), ("M",), (("M", 4),), (AXIS,), (STATEMENT,),
                     (DEPENDENCE,), None)

STREAM = StreamClause("A", "broadcast", "py", None, None)
WINDOW = WindowClause("A", ("i",), (1,))
EXCHANGE = ExchangeClause("A", "i", 1)
SCHEDULE = ScheduleModel("npu1", (2, 2), (("i", 2),), ("i", "j"), (("k", "+"),), ("A",),
                         (STREAM,), (("A", "L1"),), ("A",), ("i",), ("k",), (WINDOW,),
                         (EXCHANGE,), ("i",))

LEGAL = LegalMapping(KERNEL, SCHEDULE, (AXIS,), ((1, 0),), ((0, 1),), ((1, 0),), ((0, 1),),
                     ((1, 0),), (), ((0, 1),), ("A",), (2, 2), (1, 1), 1024,
                     (("A", (1, 1)),))

TENSOR = BufferPlan("A", "A", "L3", "tensor", (4, 4), Dtype.f32, 64, 0, False)
BUFFER = BufferPlan("A_L1", "A", "L1", "herd.private", (4, 4), Dtype.f32, 64, 1, True)
REGION = Region((E0,), (4,), (1,))
GUARD = Guard("tx", "==", E0)
PUT = ChannelSite("A2L1.put.0@segment", "put", "A2L1", (E0,), "A", REGION, "segment", None,
                  True, (), 0)
GET = ChannelSite("A2L1.get.0@herd", "get", "A2L1", (E0,), "A_L1", REGION, "herd", GUARD,
                  True, ("A2L1.put.0@segment",), 0)
CHANNEL = ChannelPlan("A2L1", (1,), None, None, None, Dtype.f32, (PUT, GET))
HERD = HerdPlan("herd", (2, 2), (1, 2), None, ("tx", "ty"))

LOAD = Load("A_L1", (E0,))
CONST = Const(0, "0", Dtype.f32)
BINOP = BinOp("+", LOAD, CONST)
NEG = Neg(CONST)
MAXMIN = MaxMin("maximum", (LOAD, CONST))
SELECT = Select("<", LOAD, CONST, LOAD, CONST)
STORE = StoreNode("A_L1", (E0,), BINOP)
LOOP = LoopPlan("k", E0, E4, E1, "sequential", 0, (STORE,))
BRANCH = BranchNode(GUARD, (STORE,), ())

SUMMARY = MappingSummary(("herd: 2x2 logical",), (("A", "resident for the whole run"),),
                         (2, 2), (1, 2), (2, 1), ("{}", "{e_k}"), 1024, 65536,
                         (("A2L1", (1,), None),))
PLAN = MappingPlan(LEGAL, (TENSOR,), "launch", "segment", HERD, (BUFFER,), (CHANNEL,),
                   (PUT, HERD), (GET, LOOP, BRANCH), (("A", "MULTICAST", "py", True),), SUMMARY)
EMIT = EmitResult("module {}\n", "npu1", PLAN, SUMMARY, 1024)
DIAGNOSTIC = Diagnostic("CLAUSE-BAD-ENUM", "clause", 'stream("A", pattern="nope")',
                        "pattern 'nope' is not one of broadcast/forward/cascade",
                        'use pattern="broadcast"', None, {"given": "nope", "vector": (1, 0)})

INSTANCES: dict[type, object] = {
    Expr: Expr((("i", 1),), 3),
    Param: PARAM,
    Axis: AXIS,
    AccessMap: ACCESS,
    Statement: STATEMENT,
    Dependence: DEPENDENCE,
    ReductionSpec: REDUCTION,
    KernelModel: KERNEL,
    AxisRef: AxisRef("i", 1),
    StreamClause: STREAM,
    WindowClause: WINDOW,
    ExchangeClause: EXCHANGE,
    ScheduleModel: SCHEDULE,
    LegalMapping: LEGAL,
    BufferPlan: BUFFER,
    Region: REGION,
    Guard: GUARD,
    ChannelSite: PUT,
    ChannelPlan: CHANNEL,
    HerdPlan: HERD,
    Load: LOAD,
    Const: CONST,
    BinOp: BINOP,
    Neg: NEG,
    MaxMin: MAXMIN,
    Select: SELECT,
    StoreNode: STORE,
    LoopPlan: LOOP,
    BranchNode: BRANCH,
    MappingSummary: SUMMARY,
    MappingPlan: PLAN,
    EmitResult: EMIT,
    Diagnostic: DIAGNOSTIC,
}

BASE: dict[type, dict] = {
    cls: {f.name: getattr(obj, f.name) for f in dataclasses.fields(cls)}
    for cls, obj in INSTANCES.items()
}


def build(cls: type, **override):
    """Construct `cls` from its minimal valid keywords with `override` applied."""
    return cls(**{**BASE[cls], **override})


# ---------------------------------------------------------------------------------------------
# Every class is a frozen, hashable, value-equal dataclass (design/06-interfaces.md §2 preamble)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cls", list(INSTANCES), ids=lambda c: c.__name__)
def test_every_class_is_a_frozen_dataclass(cls):
    obj = INSTANCES[cls]
    assert dataclasses.is_dataclass(cls) and cls.__dataclass_params__.frozen
    field = dataclasses.fields(cls)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(obj, field, getattr(obj, field))


@pytest.mark.parametrize("cls", list(INSTANCES), ids=lambda c: c.__name__)
def test_every_class_is_hashable_and_equal_by_value(cls):
    obj = INSTANCES[cls]
    twin = build(cls)
    assert obj == twin and obj is not twin
    assert hash(obj) == hash(twin)
    assert len({obj, twin}) == 1


@pytest.mark.parametrize("cls", list(INSTANCES), ids=lambda c: c.__name__)
def test_no_field_holds_a_list(cls):
    for field in dataclasses.fields(cls):
        assert not isinstance(getattr(INSTANCES[cls], field.name), list)


# ---------------------------------------------------------------------------------------------
# One case per enforced invariant. `(id, cls, override, exception, message fragment)`
# ---------------------------------------------------------------------------------------------

CASES: list[tuple] = [
    ("I01", Param, dict(shape=[4, "M"]), TypeError, "expected a tuple"),
    ("I01", Param, dict(name=1), TypeError, "expected str"),
    ("I01", Param, dict(is_written=1), TypeError, "expected bool"),
    ("I01", Axis, dict(depth=True), TypeError, "expected int"),
    ("I01", BufferPlan, dict(dtype="f32"), TypeError, "expected Dtype"),
    ("I01", Guard, dict(value=0), TypeError, "expected Expr"),
    ("I01", HerdPlan, dict(at=(0, 0, 0)), TypeError, "does not match"),
    ("I02", BufferPlan, dict(scope="herd.shared"), ValueError, "is not one of"),
    ("I02", ScheduleModel, dict(target="npu3"), ValueError, "is not one of"),
    ("I02", Statement, dict(kind="store"), ValueError, "is not one of"),
    ("I02", Guard, dict(relation="=<"), ValueError, "is not one of"),
    ("I02", ChannelPlan, dict(channel_type="npu_stream"), ValueError, "is not one of"),
    ("I03", Expr, dict(coeffs=(("2i", 1),)), ValueError, "valid identifier"),
    ("I03", Expr, dict(coeffs=(("i", "1"),)), TypeError, "expected int"),
    ("I04", Param, dict(name="2A"), ValueError, "valid identifier"),
    ("I05", Param, dict(shape=()), ValueError, "rank must be >= 1"),
    ("I06", Param, dict(shape=(0,)), ValueError, "must be >= 1"),
    ("I06", Param, dict(shape=("2M",)), ValueError, "valid identifier"),
    ("I07", Axis, dict(name=""), ValueError, "non-empty"),
    ("I08", Axis, dict(step=Expr((("i", 1),), 1)), ValueError, "positive constant"),
    ("I08", Axis, dict(step=E0), ValueError, "positive constant"),
    ("I09", Axis, dict(depth=-1), ValueError, "must be >= 0"),
    ("I10", AccessMap, dict(matrix=((1, 0), (0, 1))), ValueError, "one row per offset"),
    ("I10", AccessMap, dict(matrix=((1, 0), (0,)), offsets=(E0, E0)), ValueError,
     "rectangular"),
    ("I11", Statement, dict(kind="assign", op="+"), ValueError, "if and only if"),
    ("I11", Statement, dict(kind="accumulate", op=None), ValueError, "if and only if"),
    ("I12", Statement, dict(line=0), ValueError, "must be >= 1"),
    ("I13", Dependence, dict(vector=()), ValueError, "non-empty"),
    ("I13", Dependence, dict(vector=(0, 0)), ValueError, "all zero"),
    ("I14", ReductionSpec, dict(projection=((1, 0), (1,))), ValueError, "rectangular"),
    ("I14", ReductionSpec, dict(space=((1, 0), (1,))), ValueError, "rectangular"),
    ("I15", KernelModel, dict(params=(PARAM, PARAM)), ValueError, "one entry per name"),
    ("I16", KernelModel, dict(params=(Param("A", Dtype.f32, (4,), False),)), ValueError,
     "at least one param must be written"),
    ("I17", KernelModel, dict(shape_params=("N", "M")), ValueError, "sorted by name"),
    ("I17", KernelModel, dict(shape_params=("M", "M")), ValueError, "sorted by name"),
    ("I18", KernelModel, dict(axes=(AXIS, AXIS)), ValueError, "one entry per name"),
    ("I19", KernelModel, dict(statements=()), ValueError, "at least one statement"),
    ("I20", KernelModel, dict(dependences=(Dependence((1,), "RAW", "B"), DEPENDENCE)),
     ValueError, "sorted by .operand, vector."),
    ("I21", KernelModel, dict(reduction=REDUCTION), ValueError, "if and only if"),
    ("I21", KernelModel, dict(statements=(Statement("accumulate", ACCESS, (ACCESS,),
                                                    Load("A", (E0,)), "+", ("i",), 7),)),
     ValueError, "if and only if"),
    ("I22", AxisRef, dict(name=""), ValueError, "non-empty"),
    ("I23", WindowClause, dict(halo=(1, 1)), ValueError, "one entry per dim"),
    ("I24", ScheduleModel, dict(grid=(2, 0)), ValueError, "must be >= 1"),
    ("I25", ScheduleModel, dict(tiles=(("i", 0),)), ValueError, "must be >= 1"),
    ("I26", ScheduleModel, dict(place=("i", "i")), ValueError, "one entry per axis"),
    ("I27", ScheduleModel, dict(stationary=("B", "A")), ValueError, "sorted by operand"),
    ("I28", ScheduleModel, dict(residency=(("B", "L1"), ("A", "L1"))), ValueError,
     "sorted by operand"),
    ("I29", ScheduleModel, dict(double_buffer=("B", "A")), ValueError, "sorted by operand"),
    ("I30", ScheduleModel, dict(sequential=("k", "j")), ValueError, "sorted by axis"),
    ("I31", ScheduleModel, dict(streams=(STREAM, STREAM)), ValueError, "one entry per operand"),
    ("I32", ScheduleModel, dict(windows=(WINDOW, WINDOW)), ValueError, "one entry per operand"),
    ("I33", ScheduleModel, dict(exchanges=(EXCHANGE, EXCHANGE)), ValueError,
     "one entry per operand"),
    ("I34", LegalMapping, dict(sigma=((1, 0), (1,))), ValueError, "rectangular"),
    ("I34", LegalMapping, dict(r_space=((1, 0), (1,))), ValueError, "rectangular"),
    ("I35", LegalMapping, dict(pi_u=()), ValueError, "row count of pi"),
    ("I36", LegalMapping, dict(stationary_ops=("B", "A")), ValueError, "sorted by operand"),
    ("I37", LegalMapping, dict(l1_bytes=-1), ValueError, "must be >= 0"),
    ("I38", BufferPlan, dict(shape=(0, 4)), ValueError, "must be >= 1"),
    ("I39", BufferPlan, dict(bytes=63), ValueError, r"prod\(shape\) \* dtype.sizeof"),
    ("I40", BufferPlan, dict(level="L3", scope="herd.private"), ValueError,
     "requires scope 'tensor'"),
    ("I41", BufferPlan, dict(loop_depth=-1), ValueError, "must be >= 0"),
    ("I41", BufferPlan, dict(loop_depth=0, ping_pong_candidate=True), ValueError,
     "inside a loop"),
    ("I42", Region, dict(sizes=(4, 4)), ValueError, "rank of offsets"),
    ("I42", Region, dict(strides=()), ValueError, "rank of offsets"),
    ("I43", ChannelSite, dict(id=""), ValueError, "non-empty"),
    ("I43", ChannelSite, dict(order=-1), ValueError, "must be >= 0"),
    ("I44", ChannelPlan, dict(name="2A"), ValueError, "valid MLIR symbol"),
    ("I45", ChannelPlan, dict(size=()), ValueError, "rank must be >= 1"),
    ("I45", ChannelPlan, dict(size=(0,)), ValueError, "must be >= 1"),
    ("I46", ChannelPlan, dict(broadcast_shape=(2, 2)), ValueError, "rank of size"),
    ("I46", ChannelPlan, dict(size=(2,), broadcast_shape=(3,)), ValueError, "multiple of size"),
    ("I47", ChannelPlan, dict(channel_type="npu_cascade", broadcast_shape=(2,),
                              chain_direction="ascending"),
     ValueError, "cascade channel cannot broadcast"),
    ("I48", ChannelPlan, dict(chain_direction="ascending"), ValueError, "if and only if"),
    ("I48", ChannelPlan, dict(channel_type="npu_cascade"), ValueError, "if and only if"),
    ("I49", HerdPlan, dict(grid=(2, 2, 2)), ValueError, "rank must be 1 or 2"),
    ("I49", HerdPlan, dict(grid=(2, 0)), ValueError, "must be >= 1"),
    ("I50", HerdPlan, dict(shape=(1,)), ValueError, "rank of grid"),
    ("I50", HerdPlan, dict(shape=(1, 3)), ValueError, "divide grid"),
    ("I51", HerdPlan, dict(at=(-1, 0)), ValueError, "must be >= 0"),
    ("I52", HerdPlan, dict(coords=("tx",)), ValueError, "one name per grid entry"),
    ("I52", HerdPlan, dict(coords=("tx", "tx")), ValueError, "one entry per name"),
    ("I52", HerdPlan, dict(coords=("tx", "2ty")), ValueError, "valid identifier"),
    ("I53", LoopPlan, dict(step=E0), ValueError, "positive constant"),
    ("I53", LoopPlan, dict(depth=-1), ValueError, "must be >= 0"),
    ("I54", MaxMin, dict(operands=()), ValueError, "non-empty"),
    ("I55", Const, dict(text=""), ValueError, "non-empty"),
    ("I56", MappingPlan, dict(launch_name="2launch"), ValueError, "valid MLIR symbol"),
    ("I56", MappingPlan, dict(segment_name="seg ment"), ValueError, "valid MLIR symbol"),
    ("I57", MappingPlan, dict(tensors=(BUFFER,)), ValueError, "must have level 'L3'"),
    ("I58", MappingPlan,
     dict(buffers=(BUFFER, dataclasses.replace(BUFFER, shape=(4, 8), bytes=128))),
     ValueError, "names two different plans"),
    ("I59", MappingPlan, dict(channels=(CHANNEL, dataclasses.replace(CHANNEL, name="B2L1"),
                                        CHANNEL)),
     ValueError, "sorted by name"),
    ("I60", MappingPlan,
     dict(herd_body=(GET, dataclasses.replace(GET, kind="put"))),
     ValueError, "names two different sites"),
    ("I61", MappingPlan, dict(delivery=(("A", "MULTICAST", "py", True),
                                        ("A", "STATIONARY", None, False))),
     ValueError, "one entry per operand"),
    ("I62", MappingSummary, dict(residency=(("B", "x"), ("A", "y"))), ValueError,
     "sorted by operand"),
    ("I63", MappingSummary, dict(l1_bytes=-1), ValueError, "must be >= 0"),
    ("I63", MappingSummary, dict(l1_budget=-1), ValueError, "must be >= 0"),
    ("I64", EmitResult, dict(target="auto"), ValueError, "never 'auto'"),
    ("I65", EmitResult, dict(l1_peak=-1), ValueError, "must be >= 0"),
    ("I66", Diagnostic, dict(code="Clause-Bad-Enum"), ValueError, "uppercase and hyphenated"),
    ("I67", Diagnostic, dict(code="CLAUSE-NO-SUCH-CODE"), ValueError, "§6.3"),
    ("I68", Diagnostic, dict(stage="mapping"), ValueError, "must be 'clause'"),
    ("I69", Diagnostic, dict(reason=""), ValueError, "non-empty"),
    ("I69", Diagnostic, dict(reason="it ends with a period."), ValueError, "trailing period"),
    ("I70", Diagnostic, dict(fix=""), ValueError, "non-empty"),
    ("I71", Diagnostic, dict(code="GRAMMAR-BAD-RANGE", stage="grammar", location=None),
     ValueError, "required for a grammar diagnostic"),
    ("I72", Diagnostic, dict(clause=None), ValueError, "required for a clause diagnostic"),
    ("I73", Diagnostic, dict(details={"x": object()}), TypeError, "not JSON-serialisable"),
    ("I73", Diagnostic, dict(details={1: "x"}), TypeError, "keys must be str"),
    ("I74", KernelModel, dict(bindings=()), ValueError, "one entry per shape_params name"),
    ("I74", KernelModel, dict(bindings=(("N", 4),)), ValueError,
     "one entry per shape_params name"),
    ("I74", KernelModel, dict(shape_params=("M", "N"), bindings=(("N", 4), ("M", 4))),
     ValueError, "one entry per shape_params name"),
    ("I75", MappingPlan, dict(segment_body=(PUT,)), ValueError,
     "exactly one HerdPlan node at top level"),
    ("I75", MappingPlan, dict(segment_body=(PUT, HERD, HERD)), ValueError,
     "exactly one HerdPlan node at top level"),
    ("I76", MappingPlan,
     dict(segment_body=(PUT, dataclasses.replace(HERD, name="not_the_herd"))),
     ValueError, "must equal MappingPlan.herd"),
    ("I77", MappingPlan, dict(herd_body=(GET, HERD)), ValueError, "must hold no HerdPlan node"),
    ("I77", MappingPlan,
     dict(segment_body=(PUT, HERD, LoopPlan("k", E0, E4, E1, "sequential", 0, (HERD,)))),
     ValueError, "must not be nested inside a LoopPlan"),
    # I78: the Load is reached through a BinOp and a MaxMin, so the whole tree is walked
    ("I78", KernelModel,
     dict(statements=(dataclasses.replace(
         STATEMENT, expr=MaxMin("maximum", (CONST, BinOp("+", Load("Z", (E0,)), CONST)))),)),
     ValueError, "must name a param of this kernel"),
]


@pytest.mark.parametrize(
    "invariant, cls, override, exception, fragment", CASES,
    ids=[f"{c[0]}-{c[1].__name__}-{i}" for i, c in enumerate(CASES)])
def test_invariant_rejects(invariant, cls, override, exception, fragment):
    with pytest.raises(exception, match=fragment):
        build(cls, **override)


def test_invariants_match_the_docstring():
    documented = set(re.findall(r"^(I\d\d) ", m.__doc__, re.MULTILINE))
    assert documented, "spatial.model's docstring lists no invariants"
    assert {case[0] for case in CASES} == documented


@pytest.mark.parametrize("cls", sorted({c[1] for c in CASES}, key=lambda c: c.__name__),
                         ids=lambda c: c.__name__)
def test_the_minimal_construction_of_every_checked_class_is_valid(cls):
    assert build(cls) == INSTANCES[cls]


# ---------------------------------------------------------------------------------------------
# Q-M2-3: the semantic cross-checks belong to M2/M3/M4, so M0 must accept them
# ---------------------------------------------------------------------------------------------


def test_M0_accepts_a_rank_3_grid_so_HERD_RANK_stays_reachable():
    # design/03-lld-M3-checker.md §9: test_L11_rank builds this and expects HERD-RANK from M3.
    assert build(ScheduleModel, grid=(2, 2, 2)).grid == (2, 2, 2)


def test_M0_accepts_place_and_grid_of_different_rank():
    assert build(ScheduleModel, grid=(2,), place=("i", "j")).place == ("i", "j")


def test_M0_accepts_a_placed_axis_that_is_also_sequential():
    assert build(ScheduleModel, place=("i", "j"), sequential=("i",)).sequential == ("i",)


def test_M0_accepts_a_tile_factor_that_does_not_divide_the_extent():
    assert build(ScheduleModel, tiles=(("i", 3),)).tiles == (("i", 3),)


def test_M0_accepts_an_over_budget_l1_and_an_unbalanced_channel():
    assert build(LegalMapping, l1_bytes=1 << 20).l1_bytes == 1 << 20
    assert build(ChannelPlan, sites=(PUT,)).sites == (PUT,)


# ---------------------------------------------------------------------------------------------
# §1 Expr and Dtype
# ---------------------------------------------------------------------------------------------


def test_expr_is_structural_across_dict_order_and_zero_coefficients():
    a = Expr({"i": 1, "j": 2}, 3)
    b = Expr({"j": 2, "i": 1, "k": 0}, 3)
    assert a == b and hash(a) == hash(b)
    assert a.coeffs == (("i", 1), ("j", 2))
    assert Expr(a.coeffs, 3) == a
    assert Expr() == Expr({}, 0) and Expr().is_constant
    assert not a.is_constant


def test_dtype_table():
    assert [d.value for d in Dtype] == ["f32", "f16", "bf16", "i32", "i8"]
    assert {d.value: (d.bits, d.sizeof, d.mlir) for d in Dtype} == {
        "f32": (32, 4, "f32"),
        "f16": (16, 2, "f16"),
        "bf16": (16, 2, "bf16"),
        "i32": (32, 4, "i32"),
        "i8": (8, 1, "i8"),
    }
    import numpy

    assert [Dtype.f32.numpy, Dtype.f16.numpy, Dtype.i32.numpy, Dtype.i8.numpy] == [
        numpy.float32, numpy.float16, numpy.int32, numpy.int8]
    assert numpy.dtype(Dtype.bf16.numpy).name == "bfloat16"


def test_the_domain_frozensets_match_their_literal_aliases():
    assert m.LEVELS == {"L1", "L2", "L3"}
    assert m.TARGETS == {"npu1", "npu2", "auto"}
    assert m.CHANNEL_TYPES == {None, "npu_cascade", "npu_dma_packet"}
    assert m.STAGES == {"grammar", "clause", "legality", "mapping", "emission", "toolchain"}
    assert m.REDUCE_OPS == {"+", "max", "min"}
    assert m.PATTERNS == {"broadcast", "forward", "cascade"}
    assert m.DIRECTIONS == {"W->E", "E->W", "N->S", "S->N"}
    assert m.DELIVERIES == {"STATIONARY", "MULTICAST", "FORWARD", "CASCADE", "NONE"}
    assert m.SCOPES == {"herd.private", "segment.private", "segment.shared",
                        "segment.per_core", "tensor"}


# ---------------------------------------------------------------------------------------------
# §8 Canonical JSON
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cls", list(INSTANCES), ids=lambda c: c.__name__)
def test_json_round_trip_is_exact_and_idempotent(cls):
    obj = INSTANCES[cls]
    text = to_json(obj)
    assert from_json(text, cls) == obj
    assert to_json(from_json(text, cls)) == text


@pytest.mark.parametrize("cls", list(INSTANCES), ids=lambda c: c.__name__)
def test_json_is_canonical(cls):
    text = to_json(INSTANCES[cls])
    assert text.endswith("\n") and not text.endswith("\n\n")
    assert json.dumps(json.loads(text), sort_keys=True, indent=2) + "\n" == text
    body = [line for line in text.splitlines() if line.startswith(" ")]
    assert all((len(line) - len(line.lstrip(" "))) % 2 == 0 for line in body)


def test_json_is_the_same_under_a_different_hash_seed():
    # design/03-lld-M7-tests.md §3.5: the canonical form must not depend on dict iteration.
    assert in_fresh_process(to_json, PLAN) == to_json(PLAN)


def test_json_tags_the_expr_and_plan_node_unions():
    assert json.loads(to_json(E0))["__type__"] == "Expr"
    body = json.loads(to_json(PLAN))["herd_body"]
    assert [node["__type__"] for node in body] == ["ChannelSite", "LoopPlan", "BranchNode"]
    # §5.6 invariant 8: the herd marker is a tagged node, so segment_body round-trips (v4).
    assert [node["__type__"] for node in json.loads(to_json(PLAN))["segment_body"]] \
        == ["ChannelSite", "HerdPlan"]
    assert json.loads(to_json(STORE))["expr"]["__type__"] == "BinOp"


def test_json_decoding_reports_corrupt_input_instead_of_crashing():
    with pytest.raises(ValueError, match="unknown __type__"):
        from_json('{"__type__": "NotAClass"}', Expr)
    with pytest.raises(ValueError, match="the JSON object has no const"):
        from_json('{"coeffs": []}', Expr)


def test_details_survives_a_key_that_looks_like_a_type_tag():
    diagnostic = build(Diagnostic, details={"__type__": "Expr", "n": 1})
    assert from_json(to_json(diagnostic), Diagnostic) == diagnostic


def test_details_is_a_canonical_immutable_mapping():
    diagnostic = build(Diagnostic, details={"b": 2, "a": [1, 0]})
    assert list(diagnostic.details) == ["a", "b"]
    assert diagnostic.details["a"] == (1, 0)
    assert diagnostic.details == build(Diagnostic, details={"a": (1, 0), "b": 2}).details
    with pytest.raises(TypeError):
        diagnostic.details["c"] = 1
    assert json.loads(to_json(diagnostic))["details"] == {"a": [1, 0], "b": 2}


# ---------------------------------------------------------------------------------------------
# §6 Diagnostics (FR-D1, FR-D3) and the error hierarchy (design/02-hld.md §4)
# ---------------------------------------------------------------------------------------------


@pytest.mark.fr("FR-D3")
def test_catalogue_matches_the_frozen_document():
    assert len(DOC_CATALOGUE) == 43
    assert m.CATALOGUE == DOC_CATALOGUE


@pytest.mark.fr("FR-D1")
def test_diagnostic_schema_is_complete():
    assert [f.name for f in dataclasses.fields(Diagnostic)] == [
        "code", "stage", "clause", "reason", "fix", "location", "details"]
    assert DIAGNOSTIC.stage == m.CATALOGUE[DIAGNOSTIC.code]


@pytest.mark.fr("FR-D1")
def test_error_rendering_has_all_four_parts():
    error = ClauseError(DIAGNOSTIC)
    rendered = str(error)
    assert error.diagnostic is DIAGNOSTIC
    assert rendered.splitlines()[0] == f"{DIAGNOSTIC.code}: {DIAGNOSTIC.reason}"
    assert f"  in clause: {DIAGNOSTIC.clause}" in rendered
    assert f"  fix:       {DIAGNOSTIC.fix}" in rendered
    assert "  because:   " in rendered
    assert "given='nope'" in rendered and "vector=(1, 0)" in rendered


def test_error_rendering_prints_a_location_for_a_grammar_diagnostic():
    diagnostic = build(Diagnostic, code="GRAMMAR-BAD-RANGE", stage="grammar",
                       location=("gemm.py", 12))
    assert "  at:        gemm.py:12" in str(GrammarError(diagnostic))


@pytest.mark.parametrize("cls", [GrammarError, ClauseError, LegalityError, MappingError,
                                 EmissionError, ToolchainError], ids=lambda c: c.__name__)
def test_the_six_leaf_classes_subclass_SpatialError(cls):
    assert issubclass(cls, SpatialError) and issubclass(cls, Exception)
    assert cls(DIAGNOSTIC).diagnostic is DIAGNOSTIC
    assert cls.__doc__


def test_SpatialError_refuses_anything_but_a_diagnostic():
    with pytest.raises(TypeError, match="expected a Diagnostic"):
        ClauseError("CLAUSE-BAD-ENUM")


# ---------------------------------------------------------------------------------------------
# FR-T6 — the pin (design/03-lld-M6-toolchain.md §3.9)
# ---------------------------------------------------------------------------------------------


@pytest.mark.fr("FR-T6")
def test_check_pin_passes_in_this_venv():
    assert m6_tools.check_pin() is None


@pytest.mark.fr("FR-T6")
def test_check_pin_raises_on_a_mismatch(monkeypatch):
    from importlib import metadata

    real = metadata.version
    monkeypatch.setattr(metadata, "version",
                        lambda name: "9.9.9" if name == "mlir_air" else real(name))
    with pytest.raises(ToolchainError) as excinfo:
        m6_tools.check_pin()
    assert_diagnostic(excinfo, code="TOOL-VERSION-PIN", clause="build()",
                      mentions=["9.9.9", m6_tools.PIN["mlir_air"]],
                      details_keys=["expected", "installed", "mismatched"])
    assert excinfo.value.diagnostic.details["mismatched"] == {
        "mlir_air": (m6_tools.PIN["mlir_air"], "9.9.9")}


@pytest.mark.fr("FR-T6")
def test_check_pin_reports_a_missing_distribution_as_a_diagnostic(monkeypatch):
    from importlib import metadata

    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)
    with pytest.raises(ToolchainError) as excinfo:
        m6_tools.check_pin()
    assert excinfo.value.diagnostic.details["installed"]["mlir_air"] is None


def test_schedule_model_is_pure_data():
    # design/06-interfaces.md §3.2: no unhashable object and no reference to the kernel function.
    for field in dataclasses.fields(ScheduleModel):
        value = getattr(SCHEDULE, field.name)
        assert not callable(value)
        hash(value)
    assert hash(SCHEDULE)


def test_contract_version_is_five():
    assert m.CONTRACT_VERSION == 6
