"""M5-TT — a second emitter: `MappingPlan` → a TT-Metalium program. Owner: Person B.

State file: `design/PROGRESS-TT.md`. This is the **T1** scope: one data-movement kernel per core
over a `ttnn.generic_op` program descriptor, executed on Tenstorrent's functional simulator
`ttsim` by `spatial.m6tt_run`. It shares nothing with `spatial.m5_emit` but the plan: the plan is
backend-neutral, and that is the claim this module exists to test.

**The emitter decides nothing** — D-14, exactly as for M5. It dispatches on the *type* of a plan
node and on whether an optional field is `None`; every name, extent, bound, index and literal is
read out of the plan. It reads **no field of `plan.mapping`**: no `LegalMapping`, no
`ScheduleModel`, no `KernelModel`, no `physical_herd`, no `repeats`. `tests/tt/test_m5tt_emit.py`
lints the source for those names.

Nothing outside the standard library and `spatial.model` is imported, so this module runs in the
project's own `.venv` and its unit tests join the default suite. `ttnn` appears only in
`spatial.m6tt_run`.

---

## The mapping table (AIR construct → TT-Metalium construct)

| Plan construct | TT-Metalium |
|---|---|
| `MappingPlan.tensors` (L3) | a DRAM `ROW_MAJOR` tensor each, in order, reached through `TensorAccessorArgs`; **one page is one row** |
| `HerdPlan.grid` | the core range `(0,0)..(gx-1, gy-1)`, one `RISCV_0` kernel per core |
| `HerdPlan.shape`, `.at`, `repeats` | **ignored** — AIE strip-mining means nothing on a Tensix grid |
| `HerdPlan.coords` | runtime args, after one base-address arg per L3 tensor |
| `BufferPlan` (L1) | a circular buffer, `total_size == page_size == bytes`; `get_write_ptr(index)` |
| `BufferPlan.ping_pong_candidate` | **ignored** — a functional target does not double-buffer |
| an L3↔L1 `ChannelPlan` | **the herd side transfers it itself**: row-wise `noc_async_read` / `noc_async_write` of the *segment* site's `Region`, plus a barrier (see `_bundle_env`) |
| a segment loop wrapping only L3 sites | nothing of its own (see `_segment_env`) |
| a core↔core `ChannelPlan` | **not in T1** — `TTNotImplemented("… T2/T3/T4")` |
| `LoopPlan`, either `kind` | a C++ `for`; `"unrolled"` is an AIE tracing distinction |
| `StoreNode` / `ExprNode` | scalar C++ over `volatile tt_l1_ptr <ctype>*`; `Const` from `Const.text`, never a float round-trip; `MaxMin`/`Select` as ternaries |
| `BranchNode`, `ChannelSite.guard` | a C++ `if` on the runtime-arg coordinates |

`design/PROGRESS-TT.md` §3 is the same table with the reasoning, §5 the list of what T1 refuses.

The target string (`"npu1"` / `"npu2"`) is an **AIR** target and is irrelevant here: it selects an
AIE generation for `air.api`'s `build()`, and nothing in a `MappingPlan` downstream of M4 depends
on it. `emit()` therefore takes no target.

## The kernel-side API this emits, as measured in the pinned wheel

Paths are relative to `<site-packages>/ttnn/`, wheel `ttnn==0.78.0`:

* `tt_metal/hw/inc/api/dataflow/dataflow_api.h:552` — `noc_async_read(uint64_t src_noc_addr,
  uint32_t dst_local_l1_addr, uint32_t size, ...)`
* `tt_metal/hw/inc/api/dataflow/dataflow_api.h:828` — `noc_async_write(uint32_t src_local_l1_addr,
  uint64_t dst_noc_addr, uint32_t size, ...)`
* `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1750`, `:1780` — `noc_async_read_barrier`,
  `noc_async_write_barrier`
* `tt_metal/hw/inc/api/tensor/tensor_accessor.h:129` — `TensorAccessor::get_noc_addr(page_id,
  offset = 0, noc = noc_index)`; the two-argument constructor (deduction guide at `:514`) takes
  the page size from the `TensorAccessorArgs` the host built, so the kernel never spells it
* `tt_metal/hw/inc/api/dataflow/dataflow_api.h:323` — `get_write_ptr(operand)`

The include list and the `KernelDescriptor.SourceType.SOURCE_CODE` shape are the gate report's
(`/home/adi/Projects/Honours/tt-probe/t3_generic_op.py`, `t6_multicore.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

from spatial.model import (BinOp, BranchNode, BufferPlan, ChannelPlan, ChannelSite, Const, Dtype,
                           Expr, Guard, HerdPlan, Load, LoopPlan, MappingPlan, MaxMin, Neg,
                           Region, Select, StoreNode)

_CTYPE = {Dtype.f32: "float", Dtype.i32: "int32_t", Dtype.i8: "int8_t"}
"""The dtypes a scalar C++ kernel can hold. `f16`/`bf16` have no RISC-V scalar type, so a plan
that asks for one is refused rather than silently widened."""

_NOC_ALIGN = 32
"""The transfer alignment T1 requires of every DRAM offset, L1 offset and length, in bytes.

Wormhole's DRAM alignment is 32 B and its L1 alignment is 16 B; 32 satisfies both and is the
conservative choice for a functional target. A misaligned NoC transfer corrupts silently rather
than failing, so it is checked **statically and completely** before any code is emitted: each
free name in an affine offset ranges over an arithmetic progression `lo, lo+step, …` known from
the plan (a loop's bounds, a herd coordinate, an inverted bundle index), so the byte offset
`width * (const + Σ c_i·(lo_i + step_i·t_i))` is a multiple of the alignment for **every** trip
exactly when `width·(const + Σ c_i·lo_i)` and each `width·c_i·step_i` is. A plan that needs a
finer alignment is refused, not mis-transferred.
"""

_MAX_CBS = 32
"""tt-metal's circular-buffer count per core."""

_RELATION = {"==": "==", "!=": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">="}
_BINOP = {"+": "+", "-": "-", "*": "*", "/": "/"}

_RESERVED = frozenset("""
alignas alignof and asm auto bool break case catch char class const constexpr continue decltype
default delete do double dynamic_cast else enum explicit export extern false float for friend
goto if inline int long mutable namespace new noexcept not nullptr operator or private protected
public register reinterpret_cast return short signed sizeof static static_cast struct switch
template this throw true try typedef typeid typename union unsigned using virtual void volatile
while xor int32_t uint32_t int8_t uint8_t kernel_main main
""".split())
"""C++ keywords and the identifiers this emitter itself writes. A plan name that collides with
one would generate code that does not mean what the plan says."""

_HEADER = """\
// Generated by spatial.m5tt_emit from MappingPlan {launch!r}. Do not edit.
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {{
"""


class TTEmitError(Exception):
    """Every failure out of this emitter. A plan that reaches here has passed M4's self-check,
    so all but `TTNotImplemented` mean a defect in M4 or in this module, never in the user's
    program."""


class TTNotImplemented(TTEmitError, NotImplementedError):
    """A plan construct the T1 scope does not implement."""


@dataclass(frozen=True)
class TTTensor:
    """One L3 tensor, as a DRAM row-major `ttnn` tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: Dtype
    page_bytes: int
    """One page is one row: `shape[-1] * dtype.sizeof`. `m6tt_run` checks the device agrees."""
    cta_define: str
    """The preprocessor name the kernel reads this tensor's `TensorAccessorArgs` offset from."""


@dataclass(frozen=True)
class TTBuffer:
    """One L1 buffer, as a circular buffer. One page is the whole buffer."""

    index: int
    name: str
    bytes: int
    page_bytes: int
    dtype: Dtype


@dataclass(frozen=True)
class TTProgram:
    """One plan, as a TT-Metalium program: what `m6tt_run` needs and nothing else."""

    launch_name: str
    grid: tuple[int, ...]
    core_range: tuple[tuple[int, int], tuple[int, int]]
    """`((x0, y0), (x1, y1))`, inclusive, as `ttnn.CoreRange` takes it."""
    source: str
    """The kernel C++, for every core in `core_range`."""
    cbs: tuple[TTBuffer, ...]
    io_tensors: tuple[TTTensor, ...]
    runtime_args: tuple[tuple[tuple[int, int], tuple[tuple[str, Any], ...]], ...]
    """`((x, y), args)` per core. Each arg is `("addr", <tensor name>)` — the tensor's DRAM base
    address, which only the runner knows — or `("const", <int>)`. The ABI is the emitter's; the
    runner substitutes and decides nothing."""


def row_major(shape: tuple[int, ...]) -> tuple[int, ...]:
    """The row-major strides of `shape` — the only layout a `Region` may describe."""
    strides = [1] * len(shape)
    for dim in range(len(shape) - 2, -1, -1):
        strides[dim] = strides[dim + 1] * shape[dim + 1]
    return tuple(strides)


class _Emitter:
    """One emission of one plan: the name environment, the text, the checks."""

    def __init__(self, plan: MappingPlan) -> None:
        self.plan = plan
        self.tensors = {tensor.name: tensor for tensor in plan.tensors}
        self.buffers = {buffer.name: buffer for buffer in plan.buffers}
        self.channels = {channel.name: channel for channel in plan.channels}
        self.env: dict[str, str] = {}       # index name -> the C++ expression it stands for
        self.ranges: dict[str, tuple[int, int] | None] = {}   # -> (lo, step), None if unknown
        self.lines: list[str] = []
        self.depth = 1
        self.dma = 0                        # counter making generated loop names unique
        self.seg_sites: dict[str, list[tuple[ChannelSite, tuple[LoopPlan, ...]]]] = {}
        self.herd_loops: tuple[LoopPlan, ...] = ()

    # -- diagnostics ---------------------------------------------------------

    def _bug(self, reason: str) -> TTEmitError:
        return TTEmitError(f"{self.plan.launch_name}: {reason}")

    # -- names ---------------------------------------------------------------

    def _identifier(self, name: str, what: str) -> str:
        if not name.isidentifier() or name.startswith("_") or name in _RESERVED:
            raise self._bug(f"{what} {name!r} is not a C++ identifier this emitter may use "
                            f"(it must be an identifier, must not start with '_', and must not "
                            f"be a C++ keyword)")
        return name

    def _lookup(self, name: str) -> str:
        bound = self.env.get(name)
        if bound is None:
            raise self._bug(f"the index name {name!r} is not bound at this point in the plan; "
                            f"bound here: {sorted(self.env)}")
        return bound

    # -- text ----------------------------------------------------------------

    def _emit(self, text: str) -> None:
        self.lines.append("    " * self.depth + text)

    def _open(self, text: str) -> None:
        self._emit(text + " {")
        self.depth += 1

    def _close(self) -> None:
        self.depth -= 1
        self._emit("}")

    # -- expressions ---------------------------------------------------------

    @staticmethod
    def _scale(text: str, factor: int) -> str:
        """`text * factor`, without writing the identity — generated C++ has to stay readable."""
        return text if factor == 1 else f"({text} * {factor})"

    @staticmethod
    def _sum(terms: list[str]) -> str:
        kept = [term for term in terms if term != "0"]
        if not kept:
            return "0"
        return kept[0] if len(kept) == 1 else "(" + " + ".join(kept) + ")"

    def _affine(self, expr: Expr, env: dict[str, str]) -> str:
        """`EVAL_EXPR`: `sum(coeff * env[name]) + const`, in the `Expr`'s own sorted order."""
        terms = [self._scale(env[name], coeff) if name in env else self._unbound(name, env)
                 for name, coeff in expr.coeffs]
        return self._sum(terms + [str(expr.const)])

    def _unbound(self, name: str, env: dict[str, str]) -> str:
        raise self._bug(f"the index name {name!r} is not bound at this point in the plan; "
                        f"bound here: {sorted(env)}")

    def _value(self, node: Any) -> str:
        """`EMIT_EXPR`: one branch per `ExprNode` case, and no case reads the kernel."""
        if isinstance(node, Load):
            return self._element(node.buffer_id, node.subscripts)
        if isinstance(node, Const):
            ctype = self._ctype(node.dtype)
            try:                            # validated, but emitted verbatim: `Const.text` is
                float(node.text)            # the source token and never a float round-trip
            except ValueError:
                raise self._bug(f"Const.text {node.text!r} is not a numeric literal") from None
            return f"(({ctype})({node.text}))"
        if isinstance(node, BinOp):
            return f"({self._value(node.lhs)} {_BINOP[node.op]} {self._value(node.rhs)})"
        if isinstance(node, Neg):
            return f"(-{self._value(node.operand)})"
        if isinstance(node, MaxMin):
            relation = ">" if node.op == "maximum" else "<"
            folded = self._value(node.operands[-1])
            for operand in reversed(node.operands[:-1]):
                value = self._value(operand)
                folded = f"(({value} {relation} {folded}) ? {value} : {folded})"
            return folded
        if isinstance(node, Select):
            return (f"(({self._value(node.lhs)} {_RELATION[node.cmp_op]} "
                    f"{self._value(node.rhs)}) ? {self._value(node.then)} : "
                    f"{self._value(node.otherwise)})")
        raise self._bug(f"{type(node).__name__} is not an ExprNode "
                        f"(design/06-interfaces.md §5.5)")

    def _ctype(self, dtype: Dtype) -> str:
        ctype = _CTYPE.get(dtype)
        if ctype is None:
            raise TTNotImplemented(
                f"{self.plan.launch_name}: dtype {dtype.value!r} has no scalar C++ type on a "
                f"Tensix data-movement core; T1 emits scalar compute only")
        return ctype

    def _element(self, name: str, subscripts: tuple[Expr, ...]) -> str:
        """`<buffer>[<row-major flattening of the subscripts>]`."""
        buffer = self.buffers.get(name)
        if buffer is None:
            raise self._bug(f"{name!r} is not one of plan.buffers ({sorted(self.buffers)}); "
                            f"a herd body may only load from and store to L1")
        if len(subscripts) != len(buffer.shape):
            raise self._bug(f"a rank-{len(subscripts)} subscript of {name!r}, which has shape "
                            f"{buffer.shape}")
        flat = self._sum([self._scale(self._affine(sub, self.env), stride)
                          for sub, stride in zip(subscripts, row_major(buffer.shape))])
        return f"{name}[{flat}]"

    def _predicate(self, guard: Guard) -> str:
        return (f"({self._lookup(guard.coord)} {_RELATION[guard.relation]} "
                f"{self._affine(guard.value, self.env)})")

    # -- the segment body: an index of L3 sites, and nothing emitted ---------

    def _index_segment(self, nodes: tuple[Any, ...], loops: tuple[LoopPlan, ...]) -> int:
        """Record every segment-scope site with the loops around it. Emits no code: on TT the
        herd side does its own DRAM transfer, so a loop that only wraps an L3 site has nothing
        of its own to run. Anything else at segment scope is out of T1's scope."""
        marks = 0
        for node in nodes:
            if isinstance(node, LoopPlan):
                marks += self._index_segment(node.body, loops + (node,))
            elif isinstance(node, ChannelSite):
                if node.scope != "segment":
                    raise self._bug(f"site {node.id!r} sits in plan.segment_body but says it is "
                                    f"emitted at {node.scope!r} scope")
                if node.buffer not in self.tensors:
                    raise TTNotImplemented(
                        f"{self.plan.launch_name}: segment-scope site {node.id!r} moves "
                        f"{node.buffer!r}, which is not an L3 tensor; T1 emits only L3<->L1 "
                        f"channels (T2/T3/T4)")
                self.seg_sites.setdefault(node.channel, []).append((node, loops))
            elif isinstance(node, HerdPlan):
                if loops:
                    raise self._bug("the HerdPlan marker is nested inside a loop "
                                    "(design/06-interfaces.md §5.6 invariant 8)")
                if node != self.plan.herd:
                    raise self._bug("the HerdPlan marker differs from plan.herd "
                                    "(design/06-interfaces.md §5.6 invariant 8)")
                marks += 1
            else:
                raise TTNotImplemented(
                    f"{self.plan.launch_name}: a {type(node).__name__} at segment scope is not "
                    f"in T1's scope (T2/T3/T4)")
        return marks

    # -- the herd body -------------------------------------------------------

    def _walk(self, nodes: tuple[Any, ...]) -> None:
        for node in nodes:
            if isinstance(node, BufferPlan):
                self._declared(node)
            elif isinstance(node, LoopPlan):
                self._loop(node)
            elif isinstance(node, StoreNode):
                self._store(node)
            elif isinstance(node, BranchNode):
                self._branch(node)
            elif isinstance(node, ChannelSite):
                self._site(node)
            elif isinstance(node, HerdPlan):
                raise self._bug("a HerdPlan marker inside the herd body "
                                "(design/06-interfaces.md §5.6 invariant 8)")
            else:
                raise self._bug(f"{type(node).__name__} is not a PlanNode "
                                f"(design/06-interfaces.md §5.5)")

    def _declared(self, buffer: BufferPlan) -> None:
        """An `air.alloc` has no counterpart: every L1 buffer is a circular buffer declared in
        the program descriptor, so the allocation site only has to be the buffer the plan says."""
        if self.buffers.get(buffer.name) != buffer:
            raise self._bug(f"the herd body allocates {buffer.name!r}, which is not the "
                            f"identical entry in plan.buffers")

    def _loop(self, loop: LoopPlan) -> None:
        axis = self._identifier(loop.axis, "LoopPlan.axis")
        lo, hi, step = (self._affine(bound, self.env) for bound in (loop.lo, loop.hi, loop.step))
        self._open(f"for (int32_t {axis} = {lo}; {axis} < {hi}; {axis} += {step})")
        shadowed = (self.env.get(axis), self.ranges.get(axis))
        self.env[axis] = axis
        self.ranges[axis] = ((loop.lo.const, loop.step.const) if loop.lo.is_constant else None)
        outer, self.herd_loops = self.herd_loops, self.herd_loops + (loop,)
        try:
            self._walk(loop.body)
        finally:
            self.herd_loops = outer
            if shadowed[0] is None:
                del self.env[axis]
                del self.ranges[axis]
            else:
                self.env[axis], self.ranges[axis] = shadowed
        self._close()

    def _branch(self, branch: BranchNode) -> None:
        self._open(f"if ({self._predicate(branch.predicate)})")
        self._walk(branch.then)
        self._close()
        if branch.otherwise:
            self._open("else")
            self._walk(branch.otherwise)
            self._close()

    def _store(self, store: StoreNode) -> None:
        self._emit(f"{self._element(store.buffer_id, store.subscripts)} = "
                   f"{self._value(store.expr)};")

    def _site(self, site: ChannelSite) -> None:
        if site.scope != "herd":
            raise self._bug(f"site {site.id!r} sits in plan.herd_body but says it is emitted at "
                            f"{site.scope!r} scope")
        if site.guard is None:
            self._transfer(site)
            return
        self._open(f"if ({self._predicate(site.guard)})")
        self._transfer(site)
        self._close()

    # -- the one transfer T1 implements: L3 <-> L1, done by the herd side ----

    def _paired(self, site: ChannelSite) -> tuple[ChannelSite, tuple[LoopPlan, ...]]:
        """The segment-scope site at the other end of this channel."""
        channel = self.channels.get(site.channel)
        if channel is None:
            raise self._bug(f"site {site.id!r} names channel {site.channel!r}, which is not in "
                            f"plan.channels")
        pairs = [pair for pair in self.seg_sites.get(site.channel, ())
                 if pair[0].kind != site.kind]
        if len(pairs) != 1:
            raise TTNotImplemented(
                f"{self.plan.launch_name}: channel {site.channel!r} has {len(pairs)} "
                f"segment-scope {'get' if site.kind == 'put' else 'put'}(s) against the "
                f"herd-scope {site.kind} {site.id!r}; a core-to-core channel is T2/T3/T4")
        return pairs[0]

    def _bundle_env(self, channel: ChannelPlan, site: ChannelSite,
                    segment: ChannelSite) -> dict[str, tuple[str, tuple[int, int]]]:
        """Invert the fan-in rule: the segment site's bundle index, as an expression in this
        core's coordinates.

        A put at bundle index `i` feeds every destination `d` of its fan-out set, which is
        `d == i` for a plain channel and `d[k] % size[k] == i[k]` for a broadcast one
        (design/06-interfaces.md §5.6 invariant 1). Read the other way, the core at destination
        `d` is fed by source index `i[k] = d[k] % size[k]`, and `d` is this site's own index.
        """
        if len(segment.indices) != len(channel.size) or len(site.indices) != len(channel.size):
            raise self._bug(f"channel {channel.name!r} has size {channel.size} but sites of "
                            f"rank {len(segment.indices)} and {len(site.indices)}")
        env: dict[str, tuple[str, tuple[int, int]]] = {}
        for dim, (source, destination) in enumerate(zip(segment.indices, site.indices)):
            here = self._affine(destination, self.env)
            fan_in = here if channel.broadcast_shape is None \
                else f"({here} % {channel.size[dim]})"
            # A bundle index runs over `0 .. size[dim]-1` (or, unbroadcast, over the fan-out
            # extent), consecutively either way: lo 0, step 1.
            if not source.coeffs:
                if channel.size[dim] != 1 or source.const != 0:
                    raise self._bug(
                        f"channel {channel.name!r} dimension {dim} has the constant source "
                        f"index {source.const} against size {channel.size[dim]}: the fan-in "
                        f"rule then feeds only some of the cores that get from it, which "
                        f"contradicts the plan's own balance invariant")
                continue
            if len(source.coeffs) != 1 or source.coeffs[0][1] != 1 or source.const != 0:
                raise TTNotImplemented(
                    f"{self.plan.launch_name}: channel {channel.name!r} dimension {dim} has the "
                    f"source index {source}; T1 inverts a bundle index only when it is a bare "
                    f"loop variable")
            env[source.coeffs[0][0]] = (fan_in, (0, 1))
        return env

    def _segment_env(self, site: ChannelSite, segment: ChannelSite,
                     loops: tuple[LoopPlan, ...]) -> dict[str, tuple[str, tuple[int, int] | None]]:
        """Every name the segment site's region may mention, as an expression this core can
        evaluate: a bundle index inverted through the fan-in rule, or a temporal loop that the
        herd body runs too — with the same trip count, which is checked here."""
        env = self._bundle_env(self.channels[site.channel], site, segment)
        herd = {loop.axis: loop for loop in self.herd_loops}
        for loop in loops:
            if loop.axis in env:
                continue
            twin = herd.get(loop.axis)
            if twin is None:
                raise TTNotImplemented(
                    f"{self.plan.launch_name}: the segment-scope loop over {loop.axis!r} around "
                    f"site {segment.id!r} is neither a bundle-index loop nor a loop the herd "
                    f"body runs around site {site.id!r} (it runs {sorted(herd)}); T1 has no "
                    f"segment-side program to run it in (T2/T3/T4)")
            if (twin.lo, twin.hi, twin.step) != (loop.lo, loop.hi, loop.step):
                raise self._bug(
                    f"the segment-scope loop over {loop.axis!r} runs "
                    f"[{loop.lo}, {loop.hi}) step {loop.step} but the herd's loop over the same "
                    f"axis runs [{twin.lo}, {twin.hi}) step {twin.step}; the two must have the "
                    f"same trip count for the herd side to do the transfer itself")
            env[loop.axis] = (self._lookup(loop.axis), self.ranges.get(loop.axis))
        return env

    def _slab(self, region: Region, shape: tuple[int, ...], what: str
              ) -> tuple[tuple[Expr, ...], tuple[int, ...]]:
        """A `Region` as (offsets, sizes) over `shape`. An empty region is the whole buffer."""
        if not region.offsets:
            return tuple(Expr() for _ in shape), shape
        if len(region.offsets) != len(shape):
            raise self._bug(f"{what} has a rank-{len(region.offsets)} region over a rank-"
                            f"{len(shape)} buffer of shape {shape}")
        expected = row_major(shape)
        if tuple(region.strides) != expected:
            raise self._bug(f"{what} has strides {tuple(region.strides)} where row-major over "
                            f"{shape} is {expected}; T1 transfers row-major slabs only")
        return region.offsets, region.sizes

    def _multiple(self, value: int, what: str, part: str) -> None:
        if value % _NOC_ALIGN:
            raise TTNotImplemented(
                f"{self.plan.launch_name}: {what} has a {part} of {value} B, which is not a "
                f"multiple of the {_NOC_ALIGN} B NoC alignment T1 requires")

    def _aligned(self, expr: Expr, width: int, spans: dict[str, tuple[int, int] | None],
                 what: str) -> None:
        """Prove statically that `expr * width` bytes is a multiple of `_NOC_ALIGN` on every
        trip, by the arithmetic-progression argument `_NOC_ALIGN` records."""
        base = expr.const
        for name, coeff in expr.coeffs:
            span = spans.get(name)
            if span is None:                # the name's lowest value is not known statically:
                self._multiple(coeff * width, what, f"step in {name!r}")  # every value must do
                continue
            lo, step = span
            base += coeff * lo
            self._multiple(coeff * step * width, what, f"step in {name!r}")
        self._multiple(base * width, what, "base offset")

    def _transfer(self, site: ChannelSite) -> None:
        """One row-wise DRAM<->L1 transfer, plus its barrier."""
        segment, loops = self._paired(site)
        tensor = self.tensors[segment.buffer]
        buffer = self.buffers.get(site.buffer)
        if buffer is None:
            raise self._bug(f"herd-scope site {site.id!r} moves {site.buffer!r}, which is not "
                            f"one of plan.buffers")
        if segment.kind == site.kind:
            raise self._bug(f"sites {site.id!r} and {segment.id!r} are both {site.kind}s")

        bound = self._segment_env(site, segment, loops)
        env = {name: text for name, (text, _) in bound.items()}
        spans = {name: span for name, (_, span) in bound.items()}
        dram_offsets, sizes = self._slab(segment.region, tensor.shape, f"site {segment.id!r}")
        l1_offsets, l1_sizes = self._slab(site.region, buffer.shape, f"site {site.id!r}")
        if prod(sizes) != prod(l1_sizes):
            raise self._bug(f"site {segment.id!r} moves {prod(sizes)} elements of "
                            f"{segment.buffer!r} but site {site.id!r} moves {prod(l1_sizes)} "
                            f"of {site.buffer!r} ({tuple(sizes)} against {tuple(l1_sizes)})")
        if tuple(sizes) != tuple(l1_sizes):
            raise TTNotImplemented(
                f"{self.plan.launch_name}: site {segment.id!r} moves a {tuple(sizes)} slab of "
                f"{segment.buffer!r} into the {tuple(l1_sizes)} buffer {site.buffer!r}; T1 "
                f"transfers a slab row for row and does not reshape one (T2/T3/T4)")
        if tensor.dtype != buffer.dtype:
            raise self._bug(f"channel {site.channel!r} moves {tensor.dtype.value} at its L3 end "
                            f"and {buffer.dtype.value} at its L1 end")

        width = tensor.dtype.sizeof
        run = sizes[-1] * width
        self._aligned(dram_offsets[-1], width, spans, f"site {segment.id!r}'s innermost offset")
        self._aligned(l1_offsets[-1], width, self.ranges, f"site {site.id!r}'s innermost offset")
        self._multiple(run, f"site {site.id!r}", f"{sizes[-1]}-element contiguous run")
        self._multiple(tensor.shape[-1] * width, f"tensor {tensor.name!r}", "row")
        self._multiple(buffer.shape[-1] * width, f"buffer {buffer.name!r}", "row")

        self.dma += 1
        names = [f"_r{self.dma}_{dim}" for dim in range(len(sizes) - 1)]
        for name, size in zip(names, sizes):
            self._open(f"for (int32_t {name} = 0; {name} < {size}; {name} += 1)")
        # A DRAM page is one row, so the page index flattens the *leading* dimensions only.
        page = self._flatten(dram_offsets[:-1], names, row_major(tensor.shape[:-1]), env)
        byte = self._affine(dram_offsets[-1], env)
        l1 = self._sum([self._flatten(l1_offsets[:-1], names,
                                      row_major(buffer.shape)[:-1], self.env),
                        self._affine(l1_offsets[-1], self.env)])
        local = (f"{site.buffer}_l1" if l1 == "0" else
                 f"{site.buffer}_l1 + (uint32_t){self._scale(l1, width)}")
        noc = (f"{tensor.name}_ta.get_noc_addr((uint32_t)({page}), "
               f"(uint32_t){self._scale(byte, width)})")
        if site.kind == "get":
            self._emit(f"noc_async_read({noc}, {local}, {run});")
        else:
            self._emit(f"noc_async_write({local}, {noc}, {run});")
        for _ in names:
            self._close()
        self._emit(f"noc_async_{'read' if site.kind == 'get' else 'write'}_barrier();")

    def _flatten(self, offsets: tuple[Expr, ...], names: list[str], strides: tuple[int, ...],
                 env: dict[str, str]) -> str:
        """The leading dimensions of a slab index, flattened row-major: for each, the region's
        own offset plus the generated loop variable that walks it."""
        return self._sum([self._scale(self._sum([self._affine(offset, env), name]), stride)
                          for offset, name, stride in zip(offsets, names, strides)])

    # -- the whole of it -----------------------------------------------------

    def run(self) -> TTProgram:
        plan = self.plan
        herd = plan.herd
        if self._index_segment(plan.segment_body, ()) != 1:
            raise self._bug("plan.segment_body does not contain exactly one HerdPlan marker "
                            "(design/06-interfaces.md §5.6 invariant 8)")
        if len(plan.buffers) > _MAX_CBS:
            raise TTNotImplemented(
                f"{plan.launch_name}: {len(plan.buffers)} L1 buffers; a Tensix core has "
                f"{_MAX_CBS} circular buffers")

        tensors = tuple(TTTensor(name=self._identifier(tensor.name, "an L3 tensor"),
                                 shape=tensor.shape, dtype=tensor.dtype,
                                 page_bytes=tensor.shape[-1] * tensor.dtype.sizeof,
                                 cta_define=f"TA_{tensor.name}")
                        for tensor in plan.tensors)
        cbs = tuple(TTBuffer(index=index,
                             name=self._identifier(buffer.name, "an L1 buffer"),
                             bytes=buffer.bytes, page_bytes=buffer.bytes, dtype=buffer.dtype)
                    for index, buffer in enumerate(plan.buffers))
        self._check_names(herd, tensors, cbs)

        args: list[tuple[str, Any]] = [("addr", tensor.name) for tensor in tensors]
        self.lines = [_HEADER.format(launch=plan.launch_name).rstrip("\n")]
        for slot, tensor in enumerate(tensors):
            self._emit(f"constexpr auto {tensor.name}_args = "
                       f"TensorAccessorArgs<{tensor.cta_define}>();")
            self._emit(f"const auto {tensor.name}_ta = TensorAccessor({tensor.name}_args, "
                       f"get_arg_val<uint32_t>({slot}));")
        for slot, coord in enumerate(herd.coords, start=len(tensors)):
            self.env[coord] = coord
            self.ranges[coord] = (0, 1)     # a herd coordinate runs 0 .. grid[d]-1
            self._emit(f"const int32_t {coord} = "
                       f"(int32_t)get_arg_val<uint32_t>({slot});")
        for cb in cbs:
            ctype = self._ctype(cb.dtype)
            self._emit(f"const uint32_t {cb.name}_l1 = get_write_ptr({cb.index});")
            self._emit(f"volatile tt_l1_ptr {ctype}* {cb.name} = "
                       f"(volatile tt_l1_ptr {ctype}*){cb.name}_l1;")
        self._walk(plan.herd_body)
        self.depth -= 1
        self._emit("}")

        return TTProgram(
            launch_name=plan.launch_name, grid=herd.grid,
            core_range=self._core_range(herd.grid), source="\n".join(self.lines) + "\n",
            cbs=cbs, io_tensors=tensors,
            runtime_args=tuple(
                (core, tuple(args) + tuple(("const", value) for value in coord))
                for core, coord in self._cores(herd.grid)))

    def _check_names(self, herd: HerdPlan, tensors: tuple[TTTensor, ...],
                     cbs: tuple[TTBuffer, ...]) -> None:
        """Every generated identifier must be distinct, or the kernel would not say what the
        plan says: an L1 buffer sharing a name with a loop axis would shadow it."""
        for coord in herd.coords:
            self._identifier(coord, "a herd coordinate")
        axes = set(self._axes(self.plan.herd_body)) | set(herd.coords)
        clash = axes & ({cb.name for cb in cbs} | {tensor.name for tensor in tensors})
        if clash:
            raise self._bug(f"{sorted(clash)} name both a buffer or tensor and a loop axis or "
                            f"herd coordinate; the emitted C++ would shadow one with the other")

    def _axes(self, nodes: tuple[Any, ...]) -> list[str]:
        found: list[str] = []
        for node in nodes:
            if isinstance(node, LoopPlan):
                found += [node.axis] + self._axes(node.body)
            elif isinstance(node, BranchNode):
                found += self._axes(node.then) + self._axes(node.otherwise)
        return found

    @staticmethod
    def _core_range(grid: tuple[int, ...]) -> tuple[tuple[int, int], tuple[int, int]]:
        extents = (grid[0], grid[1] if len(grid) > 1 else 1)
        return ((0, 0), (extents[0] - 1, extents[1] - 1))

    @staticmethod
    def _cores(grid: tuple[int, ...]) -> list[tuple[tuple[int, int], tuple[int, ...]]]:
        """Every core of the range, with the herd coordinate it carries, in `coords` order."""
        extents = (grid[0], grid[1] if len(grid) > 1 else 1)
        return [((x, y), (x, y) if len(grid) > 1 else (x,))
                for x in range(extents[0]) for y in range(extents[1])]


def emit(plan: MappingPlan) -> TTProgram:
    """Translate the plan mechanically into a TT-Metalium program. Raises `TTEmitError`.

    No target argument: `"npu1"`/`"npu2"` names an AIE generation for `air.api`'s `build()` and
    means nothing to a Tensix core. The plan it selected is the same plan either way.
    """
    return _Emitter(plan).run()


__all__ = ["TTBuffer", "TTEmitError", "TTNotImplemented", "TTProgram", "TTTensor", "emit",
           "row_major"]
