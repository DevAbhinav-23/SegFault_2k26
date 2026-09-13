"""M5-TT — a second emitter: `MappingPlan` → a TT-Metalium program. Owner: Person B.

State file: `design/PROGRESS-TT.md`. Scope **T1 + T2**: one data-movement kernel per core over a
`ttnn.generic_op` program descriptor, executed on Tenstorrent's functional simulator `ttsim` by
`spatial.m6tt_run`. It shares nothing with `spatial.m5_emit` but the plan: the plan is
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
| `MappingPlan.tensors` (L3) | a DRAM `ROW_MAJOR` tensor each, in order, reached through `TensorAccessorArgs`; **one page is one padded row** (ruling R-TT-A′) |
| `HerdPlan.grid` | the core range `(0,0)..(gx-1, gy-1)`, one `RISCV_0` kernel per core |
| `HerdPlan.shape`, `.at`, `repeats` | **ignored** — AIE strip-mining means nothing on a Tensix grid |
| `HerdPlan.coords` | runtime args, after one base-address arg per L3 tensor |
| `BufferPlan` (L1) | a circular buffer, `total_size == page_size == bytes` rounded up to 32 B; `get_write_ptr(index)` |
| `BufferPlan.ping_pong_candidate` | **ignored** — a functional target does not double-buffer |
| an L3↔L1 `ChannelPlan` | **the herd side transfers it itself**: row-wise `noc_async_read` / `noc_async_write` of the *segment* site's `Region`, plus a barrier (see `_bundle_env`) |
| a segment loop wrapping only L3 sites | nothing of its own: a herd twin of the same name and bounds, else the site's own occurrence counter (see `_segment_env`) |
| a core↔core `ChannelPlan` | a **depth-1 FIFO**: a direct remote L1 write plus `full`/`empty` counting semaphores (see `_link`) |
| `ChannelPlan.size` | one link per concrete bundle index; two semaphore ids each, `full` then `empty`, in `plan.channels` order |
| `LoopPlan`, either `kind` | a C++ `for`; `"unrolled"` is an AIE tracing distinction |
| `StoreNode` / `ExprNode` | scalar C++ over `volatile tt_l1_ptr <ctype>*`; `Const` from `Const.text`, never a float round-trip; `MaxMin`/`Select` as ternaries |
| `BranchNode`, `ChannelSite.guard` | a C++ `if` on the runtime-arg coordinates |

`design/PROGRESS-TT.md` §3 is the same table with the reasoning, §5 the list of what it refuses.

## Alignment — ruling R-TT-A′, measured

The NoC constrains the **difference** of a transfer's two byte offsets, not either address and
not the size: a write needs them congruent mod 16, a read mod 32, and a mismatch is
`UndefinedBehavior`, which aborts the simulation rather than corrupting a number. `_READ_ALIGN`
carries the proof obligation; `_leading_pad` chooses the per-tensor leading pad that discharges
it; `TTAlignmentError` (`TT-ALIGNMENT`) is the refusal when no pad can.

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

_READ_ALIGN = 32
_WRITE_ALIGN = 16
"""What the NoC requires of a transfer, **measured** on ttsim at T2 (ruling R-TT-A′,
`design/08-tt-backend.md` §3.3): the two ends' byte offsets must be **congruent**, modulo 16 for
a write and modulo 32 for a read. Neither address on its own, and not the transfer size: a 4-byte
DRAM read at offset 28 into L1 offset 28 is accepted and the same read into L1 offset 0 is
`UndefinedBehavior`, which aborts the simulation.

Congruence is proved **statically and completely** before any code is emitted, by the same
arithmetic-progression argument T1 used for absolute alignment: each free name in an affine offset
ranges over `lo, lo+step, …` known from the plan (a loop's bounds, a herd coordinate, an inverted
bundle index), so `width·(const + Σ c_i·(lo_i + step_i·t_i))` has a residue that is the same on
every trip exactly when every `width·c_i·step_i` is a multiple of the modulus, and that residue is
`width·(const + Σ c_i·lo_i)`. A plan whose two ends cannot be made congruent is refused with
`TT-ALIGNMENT`, not mis-transferred.
"""

_CB_ALIGN = 32
"""Every circular buffer's `total_size` is rounded up to this, so that consecutive CB base
addresses stay 32 B aligned and an L1 byte offset's residue is the region's own offset. Measured:
CBs of 32/16/16 B landed at `0x19ce0`, `0x19d00`, `0x19d20` on every core."""

_MAX_CBS = 32
"""tt-metal's circular-buffer count per core."""

TT_SEM_LIMIT = 16
"""Semaphore ids per core, `0 .. 15`. **Measured** at T2 by over-allocating: 8 accepted, 32
refused by the host with `TT_FATAL @ tt_metal/impl/program/program.cpp:2001:
semaphore_id < NUM_SEMAPHORES — Semaphore id 16 exceeds max value 15`. No constant for it exists
anywhere under the wheel's `tt_metal/hw/inc/`."""

TT_L1_USABLE = 1_499_136 - 32_768
"""Usable L1 per worker core, in bytes: `worker_l1_size` 1 499 136 (the soc descriptor, equal to
`MEM_L1_SIZE` in `dev_mem_map.h:33`) less the 32 KB `MEM_MAP_END` system reservation
(`dev_mem_map.h:71-72`). An **estimate** of the circular-buffer ceiling, not a measured one — the
kernel binary also lives in L1. `design/08-tt-backend.md` §4."""

_SEM_BYTES = 16
"""`get_semaphore(id)` spaces ids `L1_ALIGNMENT = 16` B apart (`dataflow_api.h:1501-1503`);
measured `get_semaphore(0) = 0x88f0`, `get_semaphore(1) = 0x8900`."""

_RELATION = {"==": "==", "!=": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">="}
_GUARD = {"==": lambda a, b: a == b, "!=": lambda a, b: a != b, "<": lambda a, b: a < b,
          "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b}
"""The same relations, evaluated rather than emitted — `_static_guard` folds a guard over a
concrete core exactly as `tests/helpers/plan_interp.py` does."""
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
    """A plan construct this emitter does not implement."""


class TTAlignmentError(TTEmitError):
    """`TT-ALIGNMENT`: a transfer whose two ends cannot be made congruent (R-TT-A′). A proposed
    error code — `design/08-tt-backend.md` §6.1 — so the message, not a catalogue entry, carries
    it."""


@dataclass(frozen=True)
class TTTensor:
    """One L3 tensor, as a DRAM row-major `ttnn` tensor, under the host-side layout policy
    R-TT-A′ (`design/08-tt-backend.md` §3.3)."""

    name: str
    shape: tuple[int, ...]
    dtype: Dtype
    page_bytes: int
    """One page is one **padded** row, `row_stride_bytes`. `m6tt_run` checks the device agrees."""
    cta_define: str
    """The preprocessor name the kernel reads this tensor's `TensorAccessorArgs` offset from."""
    pad_elems: int = 0
    """Leading pad, in elements: element `[.., j]` sits at byte `pad_elems + j` of its row. The
    smallest value making every transfer on this tensor congruent (R-TT-A′)."""

    @property
    def row_elems(self) -> int:
        """Elements per padded row — what the runner uploads and slices back."""
        return self.page_bytes // self.dtype.sizeof


@dataclass(frozen=True)
class TTSemaphore:
    """One end of one core↔core link's depth-1 FIFO (`design/08-tt-backend.md` §3.5)."""

    id: int
    name: str
    """`<channel>.full[<link>]` or `<channel>.empty[<link>]` — diagnostic only."""
    initial_value: int = 0


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
    address, which only the runner knows — `("const", <int>)`, or `("noc_x", (x, y))` /
    `("noc_y", (x, y))` — the NoC coordinate of the logical core `(x, y)`, which only a live
    device knows. The ABI is the emitter's; the runner substitutes and decides nothing."""
    semaphores: tuple[TTSemaphore, ...] = ()
    """Two per core↔core link, `full` then `empty`, over the whole core range (A-TT1)."""


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def _row_bytes(elems: int, pad: int, width: int) -> int:
    """One DRAM row of an L3 tensor under R-TT-A′: the row's own elements plus the leading pad,
    rounded up so that every row begins on a `_READ_ALIGN` boundary and the page index can never
    move a transfer's residue."""
    return _round_up((elems + pad) * width, _READ_ALIGN)


def _leading_pad(width: int, wants: list[tuple[int, int]]) -> int | None:
    """The smallest leading pad, in elements, satisfying every `(modulus, delta)` a tensor's
    transfers ask of it: `delta` is the DRAM byte offset minus the L1 byte offset with no pad,
    and the pad has to close that gap. `None` when no pad in range does (`TT-ALIGNMENT`).

    The range is `[0, _READ_ALIGN / width)`: a pad of `_READ_ALIGN / width` elements shifts every
    offset by a whole `_READ_ALIGN` bytes and so repeats the residues of a pad of 0.
    """
    for pad in range(_READ_ALIGN // width):
        if all((delta + pad * width) % modulus == 0 for modulus, delta in wants):
            return pad
    return None


def row_major(shape: tuple[int, ...]) -> tuple[int, ...]:
    """The row-major strides of `shape` — the only layout a `Region` may describe."""
    strides = [1] * len(shape)
    for dim in range(len(shape) - 2, -1, -1):
        strides[dim] = strides[dim + 1] * shape[dim + 1]
    return tuple(strides)


class _Emitter:
    """One emission of one plan: the name environment, the text, the checks."""

    def __init__(self, plan: MappingPlan, pads: dict[str, int] | None = None,
                 counters: tuple[str, ...] = ()) -> None:
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
        self.pads = pads                    # None on the discovery pass: assume 0, check nothing
        self.wants: list[tuple[str, int, int]] = []    # (tensor, modulus, delta) for _leading_pad
        self.counters: list[str] = list(counters)   # C++ occurrence counters, plan order
        self.sem_base: dict[str, int] = {}  # core-to-core channel -> its first semaphore id
        self.peers: dict[tuple[str, str, tuple[int, int]], tuple[int, int]] = {}
        self.tt_tensors: dict[str, TTTensor] = {}
        self.pending: str | None = None     # the counter this transfer advances

    @property
    def checking(self) -> bool:
        """The discovery pass collects; the second pass, which has the pads, enforces."""
        return self.pads is not None

    def _pad(self, name: str) -> int:
        return 0 if self.pads is None else self.pads[name]

    def _counter(self, channel: str, kind: str) -> str:
        """The C++ variable counting this channel end's occurrences, in program order."""
        name = f"{channel}_{kind}_n"
        if name not in self.counters:
            self.counters.append(name)
        return name

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

    # -- the concrete grid: who talks to whom, and how often ------------------

    def _is_c2c(self, channel: str) -> bool:
        """A channel with no segment-scope site is core↔core: both its ends are herd sites."""
        return not self.seg_sites.get(channel)

    def _cores_of(self) -> list[tuple[tuple[int, int], dict[str, int]]]:
        """Every core of the range with its herd coordinates bound (D-12: the grid is
        concrete, so guards and index expressions are enumerable at plan time)."""
        return [(core, dict(zip(self.plan.herd.coords, coord)))
                for core, coord in self._cores(self.plan.herd.grid)]

    def _static(self, expr: Expr, env: dict[str, int], what: str) -> int:
        """An affine `Expr` over bound coordinates. Raises if a name is not bound: the emitter
        may not guess a value it cannot read out of the plan."""
        total = expr.const
        for name, coeff in expr.coeffs:
            if name not in env:
                raise TTNotImplemented(
                    f"{self.plan.launch_name}: {what} depends on {name!r}, which is not a herd "
                    f"coordinate ({sorted(env)}); this emitter enumerates the grid at plan time "
                    f"and cannot evaluate it")
            total += coeff * env[name]
        return total

    def _static_guard(self, guard: Guard, env: dict[str, int], what: str) -> bool:
        if guard.coord not in env:
            raise TTNotImplemented(
                f"{self.plan.launch_name}: the guard on {what} tests {guard.coord!r}, which is "
                f"not a herd coordinate ({sorted(env)})")
        return _GUARD[guard.relation](env[guard.coord], self._static(guard.value, env, what))

    def _static_trips(self, loop: LoopPlan, env: dict[str, int]) -> int:
        lo, hi, step = (self._static(bound, env, f"the bounds of loop {loop.axis!r}")
                        for bound in (loop.lo, loop.hi, loop.step))
        if step <= 0:
            raise self._bug(f"loop {loop.axis!r} has step {step}")
        return max(0, -(-(hi - lo) // step))

    def _static_sites(self, nodes: tuple[Any, ...], env: dict[str, int], trips: int,
                      out: list[tuple[ChannelSite, tuple[int, ...], int]]) -> None:
        """Every site this core reaches, with its concrete channel index and how many times the
        enclosing loops run it.

        `env` holds **only** the herd coordinates. A loop axis is deliberately left unbound, so a
        guard or a channel index that mentions one raises rather than being folded at the loop's
        lower bound and silently standing for every trip: a guard that varies with a loop is a
        different site on different trips, and a channel index that does is forbidden anyway
        (`03-lld-M4-mapping.md` §3.6.2, H-3).
        """
        for node in nodes:
            if isinstance(node, LoopPlan):
                self._static_sites(node.body, env, trips * self._static_trips(node, env), out)
            elif isinstance(node, BranchNode):
                taken = node.then if self._static_guard(
                    node.predicate, env, "a BranchNode") else node.otherwise
                self._static_sites(taken, env, trips, out)
            elif isinstance(node, ChannelSite):
                if node.guard is not None and not self._static_guard(node.guard, env, node.id):
                    continue
                out.append((node, tuple(self._static(index, env, node.id)
                                        for index in node.indices), trips))

    def _segment_trips(self, site: ChannelSite, loops: tuple[LoopPlan, ...]) -> int:
        """How many times one *concrete* channel index is transferred at segment scope: the
        product of the temporal loops only, since a bundle-index loop enumerates indices rather
        than repeating one."""
        bundle = {name for index in site.indices for name, _ in index.coeffs}
        total = 1
        for loop in loops:
            if loop.axis not in bundle:
                total *= self._static_trips(loop, {})
        return total

    def _survey(self) -> None:
        """Walk the concrete grid once: allocate semaphore ids, find each core's peer on each
        core↔core channel, and check that the herd runs each channel end as often as the
        segment side expects."""
        base = 0
        for channel in self.plan.channels:                 # plan order, never sorted (TP-3)
            if self._is_c2c(channel.name):
                self.sem_base[channel.name] = base         # full = base + 2k, empty = +1
                base += 2 * prod(channel.size)

        ends: dict[tuple[str, int, str], tuple[int, int]] = {}    # link end -> its core
        held: dict[tuple[str, str, tuple[int, int]], int] = {}     # core's link, per channel end
        counts: dict[tuple[tuple[int, int], str, str], int] = {}
        for core, env in self._cores_of():
            reached: list[tuple[ChannelSite, tuple[int, ...], int]] = []
            self._static_sites(self.plan.herd_body, env, 1, reached)
            for site, index, trips in reached:
                key = (core, site.channel, site.kind)
                counts[key] = counts.get(key, 0) + trips
                if not self._is_c2c(site.channel):
                    continue
                flat = self._flat_index(self.channels[site.channel], index, site)
                if ends.setdefault((site.channel, flat, site.kind), core) != core:
                    raise self._bug(
                        f"channel {site.channel!r} link {flat} has {site.kind}s on cores "
                        f"{ends[(site.channel, flat, site.kind)]} and {core}; a depth-1 FIFO "
                        f"has one end of each kind")
                if held.setdefault((site.channel, site.kind, core), flat) != flat:
                    raise TTNotImplemented(
                        f"{self.plan.launch_name}: core {core} holds links "
                        f"{held[(site.channel, site.kind, core)]} and {flat} of channel "
                        f"{site.channel!r} as a {site.kind}; this emitter carries one peer "
                        f"coordinate per channel end (design/08-tt-backend.md §3.4 block B)")

        other = {"put": "get", "get": "put"}
        for (channel, kind, core), flat in held.items():
            peer = ends.get((channel, flat, other[kind]))
            if peer is None:
                raise self._bug(f"channel {channel!r} link {flat} has a {kind} on core {core} "
                                f"and no {other[kind]}; the plan's put/get balance says it must")
            self.peers[(channel, kind, core)] = peer
        self._check_counts(counts)

    def _check_counts(self, counts: dict[tuple[tuple[int, int], str, str], int]) -> None:
        """The herd runs a channel end exactly as often as the segment side transfers one
        concrete index of it — the check that makes the occurrence counter of `_segment_env` a
        derivation rather than a guess."""
        for channel, pairs in self.seg_sites.items():
            for site, loops in pairs:
                want = self._segment_trips(site, loops)
                kind = "get" if site.kind == "put" else "put"
                for core, _ in self._cores_of():
                    got = counts.get((core, channel, kind), 0)
                    if got and got != want:
                        raise self._bug(
                            f"the segment side of channel {channel!r} transfers one index "
                            f"{want} time(s) but core {core}'s herd body runs {got} {kind}(s) "
                            f"on it; the two must agree for the herd side to do the transfer")

    def _flat_index(self, channel: ChannelPlan, index: tuple[int, ...],
                    site: ChannelSite) -> int:
        """A concrete bundle index, flattened row-major over `ChannelPlan.size`."""
        if len(index) != len(channel.size):
            raise self._bug(f"site {site.id!r} has a rank-{len(index)} index on channel "
                            f"{channel.name!r}, whose size is {channel.size}")
        flat = 0
        for value, extent, stride in zip(index, channel.size, row_major(channel.size)):
            if not 0 <= value < extent:
                raise self._bug(f"site {site.id!r} resolves to index {index} on channel "
                                f"{channel.name!r}, outside its size {channel.size}")
            flat += value * stride
        return flat

    def _link_text(self, channel: ChannelPlan, site: ChannelSite) -> str:
        """The same flattening, as C++ over this core's coordinate runtime args."""
        return self._sum([self._scale(self._affine(index, self.env), stride)
                          for index, stride in zip(site.indices, row_major(channel.size))])

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
        transfer = self._link if self._is_c2c(site.channel) else self._transfer
        if site.guard is None:
            transfer(site)
            return
        self._open(f"if ({self._predicate(site.guard)})")
        transfer(site)
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
                # No herd loop of that name: the channel is a FIFO, so this site's n-th transfer
                # is the segment loop's n-th trip. `_check_counts` holds the two to the same
                # count. design/08-tt-backend.md §3.3 rule 5.
                counter = self._counter(site.channel, site.kind)
                axis = self._affine(loop.lo, {}) if loop.lo.is_constant else None
                if axis is None or not loop.step.is_constant:
                    raise TTNotImplemented(
                        f"{self.plan.launch_name}: the segment-scope loop over {loop.axis!r} "
                        f"around site {segment.id!r} has no herd twin and non-constant bounds "
                        f"[{loop.lo}, {loop.hi}) step {loop.step}; the occurrence counter can "
                        f"only stand in for an arithmetic progression")
                if self.pending is not None:
                    raise TTNotImplemented(
                        f"{self.plan.launch_name}: site {segment.id!r} sits inside two "
                        f"twinless segment loops; one occurrence counter cannot stand in for "
                        f"two axes at once")
                self.pending = counter
                env[loop.axis] = (self._sum([axis, self._scale(counter, loop.step.const)]),
                                  (loop.lo.const, loop.step.const))
                continue
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

    def _fixed(self, value: int, modulus: int, what: str, part: str) -> None:
        """A byte quantity that must not move the residue on any trip (R-TT-A′)."""
        if self.checking and value % modulus:
            raise TTAlignmentError(
                f"TT-ALIGNMENT: {self.plan.launch_name}: {what} advances by {value} B per {part}, "
                f"which is not a multiple of {modulus}; the transfer's two ends would stop being "
                f"congruent part-way through (design/08-tt-backend.md §3.3, R-TT-A′)")

    def _offset(self, expr: Expr, width: int, scale: int,
                spans: dict[str, tuple[int, int] | None], modulus: int, what: str) -> int:
        """`scale * expr` in bytes at its lowest trip, having checked every step leaves the
        residue alone — the arithmetic-progression argument of `_READ_ALIGN`'s docstring."""
        base = expr.const
        for name, coeff in expr.coeffs:
            span = spans.get(name)
            if span is None:                # lowest value unknown: every value must be harmless
                self._fixed(coeff * scale * width, modulus, what, f"unit of {name!r}")
                continue
            lo, step = span
            base += coeff * lo
            self._fixed(coeff * step * scale * width, modulus, what, f"step in {name!r}")
        return base * scale * width

    @staticmethod
    def _strip(sizes: tuple[int, ...]) -> tuple[int, ...]:
        """A slab's shape with its **leading** unit dimensions dropped. They pick one row rather
        than iterating, so they are an offset, not a loop; a *trailing* unit dimension is not
        dropped, because it changes what is contiguous."""
        kept = list(sizes)
        while len(kept) > 1 and kept[0] == 1:
            kept.pop(0)
        return tuple(kept)

    def _common(self, sizes: tuple[int, ...], other: tuple[int, ...], one: ChannelSite,
                two: ChannelSite) -> tuple[int, ...]:
        """The slab both ends of a transfer iterate. The two `Region`s may differ in rank —
        W3's `WestIn` names `S[i, 0:1]` at one end and the whole of `edge_in` at the other —
        but once the leading unit dimensions are dropped they must be the same slab, because the
        transfer is row for row and nothing reshapes it."""
        stripped, alike = self._strip(sizes), self._strip(other)
        if stripped != alike:
            raise TTNotImplemented(
                f"{self.plan.launch_name}: site {one.id!r} moves a {tuple(sizes)} slab of "
                f"{one.buffer!r} and site {two.id!r} a {tuple(other)} slab of {two.buffer!r}; "
                f"they are {stripped} and {alike} once the leading unit dimensions are dropped, "
                f"and this emitter transfers a slab row for row without reshaping one")
        return stripped

    @staticmethod
    def _walked(sizes: tuple[int, ...], common: tuple[int, ...],
                names: list[str]) -> tuple[str | None, ...]:
        """One entry per dimension of this side: the generated loop variable that walks it, or
        `None` for a leading unit dimension and for the innermost (contiguous) run."""
        lead = len(sizes) - len(common)
        return tuple(names[dim - lead] if lead <= dim < lead + len(names) else None
                     for dim in range(len(sizes)))

    def _side_text(self, offsets: tuple[Expr, ...], strides: tuple[int, ...],
                   walked: tuple[str | None, ...], env: dict[str, str]) -> str:
        """One side of a transfer as a row-major flat index: each dimension's own offset plus
        the loop variable walking it, if any."""
        return self._sum([
            self._scale(self._sum([self._affine(offset, env)] + ([name] if name else [])), stride)
            for offset, stride, name in zip(offsets, strides, walked)])

    def _side_base(self, offsets: tuple[Expr, ...], strides: tuple[int, ...],
                   walked: tuple[str | None, ...], width: int,
                   spans: dict[str, tuple[int, int] | None], modulus: int, what: str) -> int:
        """The same side's byte offset at its lowest trip, having checked that neither the
        region's own names nor the generated loops move the residue."""
        total = 0
        for offset, stride, name in zip(offsets, strides, walked):
            if name is not None:
                self._fixed(stride * width, modulus, what, f"trip of {name!r}")
            total += self._offset(offset, width, stride, spans, modulus, what)
        return total

    def _congruent(self, dram: int, l1: int, tensor: TTTensor, modulus: int,
                   what: str) -> None:
        """The two ends of a DRAM↔L1 transfer, held to R-TT-A′. On the discovery pass this
        records what the leading pad would have to be; on the second it enforces it."""
        delta = dram - l1
        if not self.checking:
            self.wants.append((tensor.name, modulus, delta))
            return
        if (delta + tensor.pad_elems * tensor.dtype.sizeof) % modulus:
            raise TTAlignmentError(
                f"TT-ALIGNMENT: {self.plan.launch_name}: {what} moves DRAM byte "
                f"{dram + tensor.pad_elems * tensor.dtype.sizeof} against L1 byte {l1}; a NoC "
                f"transfer needs them congruent mod {modulus} and no leading pad of "
                f"{tensor.name!r} makes them so (design/08-tt-backend.md §3.3, R-TT-A′)")

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

        self.pending = None
        bound = self._segment_env(site, segment, loops)
        counter = self.pending
        env = {name: text for name, (text, _) in bound.items()}
        spans = {name: span for name, (_, span) in bound.items()}
        dram_offsets, sizes = self._slab(segment.region, tensor.shape, f"site {segment.id!r}")
        l1_offsets, l1_sizes = self._slab(site.region, buffer.shape, f"site {site.id!r}")
        if tensor.dtype != buffer.dtype:
            raise self._bug(f"channel {site.channel!r} moves {tensor.dtype.value} at its L3 end "
                            f"and {buffer.dtype.value} at its L1 end")
        common = self._common(sizes, l1_sizes, segment, site)

        width = tensor.dtype.sizeof
        run = common[-1] * width
        spec = self.tt_tensors[tensor.name]
        modulus = _READ_ALIGN if site.kind == "get" else _WRITE_ALIGN
        what = f"site {site.id!r} against {segment.id!r}"
        self.dma += 1
        names = [f"_r{self.dma}_{dim}" for dim in range(len(common) - 1)]
        dram_walk = self._walked(sizes, common, names)
        l1_walk = self._walked(l1_sizes, common, names)
        # A DRAM page is one padded row, a multiple of _READ_ALIGN, so the page index never
        # moves the residue; only the innermost offset and the leading pad do.
        self._fixed(spec.page_bytes, _READ_ALIGN, f"tensor {tensor.name!r}", "row")
        self._congruent(self._offset(dram_offsets[-1], width, 1, spans, modulus, what),
                        self._side_base(l1_offsets, row_major(buffer.shape), l1_walk, width,
                                        self.ranges, modulus, what),
                        spec, modulus, what)

        for name, size in zip(names, common):
            self._open(f"for (int32_t {name} = 0; {name} < {size}; {name} += 1)")
        page = self._side_text(dram_offsets[:-1], row_major(tensor.shape[:-1]),
                               dram_walk[:-1], env)
        byte = self._sum([self._affine(dram_offsets[-1], env), str(spec.pad_elems)])
        l1 = self._side_text(l1_offsets, row_major(buffer.shape), l1_walk, self.env)
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
        if counter is not None:            # program order is the counter (S3.3 rule 5)
            self._emit(f"{counter} += 1;")

    # -- core to core: a depth-1 FIFO over a direct remote L1 write -----------

    def _link_end(self, channel: ChannelPlan, kind: str) -> ChannelSite:
        """The far end of a core↔core channel: where the payload lands. Every site of that kind
        must name the same buffer and region, or one write address would not serve them all."""
        ends = [end for end in channel.sites if end.kind == kind and end.scope == "herd"]
        if not ends:
            raise self._bug(f"channel {channel.name!r} has no herd-scope {kind}; a core-to-core "
                            f"channel has both ends in the herd body")
        for end in ends[1:]:
            if (end.buffer, end.region) != (ends[0].buffer, ends[0].region):
                raise TTNotImplemented(
                    f"{self.plan.launch_name}: channel {channel.name!r} has {kind}s landing in "
                    f"both {ends[0].buffer!r}{ends[0].region} and {end.buffer!r}{end.region}; "
                    f"the producer writes one address and cannot serve two")
        return ends[0]

    def _sem(self, channel: ChannelPlan, site: ChannelSite, which: str) -> str:
        """`get_semaphore(...)` for this link's `full` or `empty`, as C++ over the coordinates.
        Ids run `base + 2k` (full) and `base + 2k + 1` (empty), `k` the flat link index."""
        offset = self.sem_base[channel.name] + (0 if which == "full" else 1)
        link = self._sum([self._scale(self._link_text(channel, site), 2), str(offset)])
        return f"get_semaphore((uint32_t)({link}))"

    def _link(self, site: ChannelSite) -> None:
        """One end of a depth-1 FIFO (`design/08-tt-backend.md` §3.5)."""
        channel = self.channels[site.channel]
        counter = self._counter(channel.name, site.kind)
        peer = f"{channel.name}_{site.kind}"
        if site.kind == "get":
            # Release the previous payload's slot first: every read of it lies between the
            # previous get and this one in plan order (§3.5 point 5, operational form).
            self._open(f"if ({counter} > 0)")
            self._emit(f"noc_semaphore_inc(get_noc_addr({peer}_x, {peer}_y, "
                       f"{self._sem(channel, site, 'empty')}), 1);")
            self._close()
            self._emit(f"noc_semaphore_wait_min((volatile tt_l1_ptr uint32_t*)"
                       f"{self._sem(channel, site, 'full')}, {counter} + 1);")
            self._emit(f"{counter} += 1;")
            return

        far = self._link_end(channel, "get")
        src, dst = self.buffers.get(site.buffer), self.buffers.get(far.buffer)
        if src is None or dst is None:
            raise self._bug(f"channel {channel.name!r} moves {site.buffer!r} into {far.buffer!r}, "
                            f"and one of them is not in plan.buffers")
        if src.dtype != dst.dtype:
            raise self._bug(f"channel {channel.name!r} moves {src.dtype.value} at {site.id!r} "
                            f"and {dst.dtype.value} at {far.id!r}")
        src_offsets, sizes = self._slab(site.region, src.shape, f"site {site.id!r}")
        dst_offsets, dst_sizes = self._slab(far.region, dst.shape, f"site {far.id!r}")
        common = self._common(sizes, dst_sizes, site, far)
        for offset in dst_offsets:
            if offset.coeffs:
                raise TTNotImplemented(
                    f"{self.plan.launch_name}: the landing region of channel {channel.name!r} "
                    f"has the offset {offset}, which the *producing* core cannot evaluate; a "
                    f"remote write needs a destination offset that is the same on both cores")

        width = src.dtype.sizeof
        what = f"site {site.id!r} writing {far.id!r}"
        self.dma += 1
        names = [f"_r{self.dma}_{dim}" for dim in range(len(common) - 1)]
        src_walk = self._walked(sizes, common, names)
        dst_walk = self._walked(dst_sizes, common, names)
        here_base = self._side_base(src_offsets, row_major(src.shape), src_walk, width,
                                    self.ranges, _WRITE_ALIGN, what)
        there_base = self._side_base(dst_offsets, row_major(dst.shape), dst_walk, width, {},
                                     _WRITE_ALIGN, what)
        if self.checking and (there_base - here_base) % _WRITE_ALIGN:
            raise TTAlignmentError(
                f"TT-ALIGNMENT: {self.plan.launch_name}: {what} sends L1 byte {here_base} of "
                f"{src.name!r} to L1 byte {there_base} of {dst.name!r}; a NoC write needs the "
                f"two congruent mod {_WRITE_ALIGN} (design/08-tt-backend.md §3.3, R-TT-A′)")

        self._emit(f"noc_semaphore_wait_min((volatile tt_l1_ptr uint32_t*)"
                   f"{self._sem(channel, site, 'empty')}, {counter});")
        for name, size in zip(names, common):
            self._open(f"for (int32_t {name} = 0; {name} < {size}; {name} += 1)")
        here = self._side_text(src_offsets, row_major(src.shape), src_walk, self.env)
        there = self._side_text(dst_offsets, row_major(dst.shape), dst_walk, {})
        local = (f"{src.name}_l1" if here == "0" else
                 f"{src.name}_l1 + (uint32_t){self._scale(here, width)}")
        remote = (f"{dst.name}_l1" if there == "0" else
                  f"{dst.name}_l1 + (uint32_t){self._scale(there, width)}")
        self._emit(f"noc_async_write({local}, get_noc_addr({peer}_x, {peer}_y, {remote}), "
                   f"{common[-1] * width});")
        for _ in names:
            self._close()
        self._emit("noc_async_write_barrier();")
        self._emit(f"noc_semaphore_inc(get_noc_addr({peer}_x, {peer}_y, "
                   f"{self._sem(channel, site, 'full')}), 1);")
        self._emit(f"{counter} += 1;")

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
                                 page_bytes=_row_bytes(tensor.shape[-1],
                                                       self._pad(tensor.name),
                                                       tensor.dtype.sizeof),
                                 cta_define=f"TA_{tensor.name}",
                                 pad_elems=self._pad(tensor.name))
                        for tensor in plan.tensors)
        self.tt_tensors = {tensor.name: tensor for tensor in tensors}
        cbs = tuple(TTBuffer(index=index,
                             name=self._identifier(buffer.name, "an L1 buffer"),
                             bytes=_round_up(buffer.bytes, _CB_ALIGN),
                             page_bytes=_round_up(buffer.bytes, _CB_ALIGN), dtype=buffer.dtype)
                    for index, buffer in enumerate(plan.buffers))
        self._check_names(herd, tensors, cbs)
        self._survey()
        links = [channel for channel in plan.channels if self._is_c2c(channel.name)]

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
        slot = len(tensors) + len(herd.coords)
        for channel in links:               # block B: the peer this core talks to, §3.4
            for kind in ("put", "get"):
                for axis in ("x", "y"):
                    self._emit(f"const uint32_t {channel.name}_{kind}_{axis} = "
                               f"get_arg_val<uint32_t>({slot});")
                    slot += 1
        for cb in cbs:
            ctype = self._ctype(cb.dtype)
            self._emit(f"const uint32_t {cb.name}_l1 = get_write_ptr({cb.index});")
            self._emit(f"volatile tt_l1_ptr {ctype}* {cb.name} = "
                       f"(volatile tt_l1_ptr {ctype}*){cb.name}_l1;")
        for counter in self.counters:       # program order is the counter (§3.3 rule 5, §3.5)
            self._emit(f"int32_t {counter} = 0;")
        known = tuple(self.counters)
        self._walk(plan.herd_body)
        if self.checking and tuple(self.counters) != known:
            raise self._bug(f"the emission needed the counters "
                            f"{sorted(set(self.counters) - set(known))}, which the discovery "
                            f"pass did not find; the two passes must walk the same plan")
        self.depth -= 1
        self._emit("}")

        semaphores = tuple(
            TTSemaphore(id=self.sem_base[channel.name] + 2 * link + (0 if which == "full" else 1),
                        name=f"{channel.name}.{which}[{link}]")
            for channel in links for link in range(prod(channel.size))
            for which in ("full", "empty"))
        self._check_resources(cbs, semaphores)
        return TTProgram(
            launch_name=plan.launch_name, grid=herd.grid,
            core_range=self._core_range(herd.grid), source="\n".join(self.lines) + "\n",
            cbs=cbs, io_tensors=tensors, semaphores=semaphores,
            runtime_args=tuple(
                (core, tuple(args) + tuple(("const", value) for value in coord)
                 + tuple(
                     (axis, self.peers.get((channel.name, kind, core), core))
                     for channel in links for kind in ("put", "get")
                     for axis in ("noc_x", "noc_y")))
                for core, coord in self._cores(herd.grid)))

    def _check_resources(self, cbs: tuple[TTBuffer, ...],
                         semaphores: tuple[TTSemaphore, ...]) -> None:
        """TT-P3 (`design/08-tt-backend.md` §4), before `m6tt_run` can open a device."""
        if len(semaphores) > TT_SEM_LIMIT:
            raise TTNotImplemented(
                f"{self.plan.launch_name}: {len(semaphores)} semaphores; a Tensix core has "
                f"{TT_SEM_LIMIT} (ids 0..{TT_SEM_LIMIT - 1}, measured: the host refuses id "
                f"{TT_SEM_LIMIT} with 'Semaphore id {TT_SEM_LIMIT} exceeds max value "
                f"{TT_SEM_LIMIT - 1}')")
        total = sum(cb.bytes for cb in cbs) + _SEM_BYTES * len(semaphores)
        if total > TT_L1_USABLE:
            raise TTNotImplemented(
                f"{self.plan.launch_name}: {total} B of circular buffers and semaphores per "
                f"core against {TT_L1_USABLE} B usable L1 (design/08-tt-backend.md §4)")

    def _check_names(self, herd: HerdPlan, tensors: tuple[TTTensor, ...],
                     cbs: tuple[TTBuffer, ...]) -> None:
        """Every generated identifier must be distinct, or the kernel would not say what the
        plan says: an L1 buffer sharing a name with a loop axis would shadow it."""
        for coord in herd.coords:
            self._identifier(coord, "a herd coordinate")
        for channel in self.plan.channels:
            self._identifier(channel.name, "a channel")
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

    **Two passes over the same plan.** The first discovers what the second needs to know before
    it writes a line: each tensor's leading pad (R-TT-A′ — the congruence a transfer asks for is
    only known once its two byte offsets are), and the occurrence counters, which are declared at
    the top of the kernel and found at the sites. The first pass emits text that is thrown away
    and checks no alignment; the second is authoritative and `run()` refuses to finish if the two
    disagreed about the counters.
    """
    discovery = _Emitter(plan)
    discovery.run()
    pads = {}
    for tensor in plan.tensors:
        wants = [(modulus, delta) for name, modulus, delta in discovery.wants
                 if name == tensor.name]
        pad = _leading_pad(tensor.dtype.sizeof, wants)
        if pad is None:
            raise TTAlignmentError(
                f"TT-ALIGNMENT: {plan.launch_name}: no leading pad of tensor {tensor.name!r} "
                f"makes every transfer on it congruent; the transfers ask for "
                f"{sorted({(modulus, delta % modulus) for modulus, delta in wants})} as "
                f"(modulus, DRAM byte - L1 byte) and no pad in "
                f"[0, {_READ_ALIGN // tensor.dtype.sizeof}) satisfies them all "
                f"(design/08-tt-backend.md §3.3, R-TT-A′)")
        pads[tensor.name] = pad
    return _Emitter(plan, pads=pads, counters=tuple(discovery.counters)).run()


__all__ = ["TTBuffer", "TTAlignmentError", "TTEmitError", "TTNotImplemented", "TTProgram",
           "TTSemaphore", "TTTensor", "TT_L1_USABLE", "TT_SEM_LIMIT", "emit", "row_major"]
