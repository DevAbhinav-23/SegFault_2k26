"""M5 — AIR emitter. Owner: Person B. LLD: design/03-lld-M5-emitter.md.

Entry point per design/06-interfaces.md §7.2. `air` is imported lazily, inside the function, so
the surface, the checker and the mapper stay usable on a machine with no toolchain (FR-S20).

**The emitter decides nothing** (D-14, LLD §4 P-1). It dispatches on the *type* of a plan node
and on whether an optional field is `None`; every name, extent, bound, index and literal is read
out of the plan. The only arithmetic is `_index`, which evaluates an `Expr` — `sum(coeff *
env[name]) + const` — over herd coordinates, `air.sequential` induction variables and the
Python ints an unrolled loop binds.

**`is_async` and `depends_on` against what `air.api` actually offers.** Measured in the pinned
wheel's source (`python/air/api/`, commit ff95a9b):

* There is **no asynchronous form to select**. `Channel.put` and `Channel.get`
  (`_channel.py:511`, `:524`) take `obj, indices, dependency` (`put` also `dest`) and nothing
  else, and **every** call returns a `Token` — `_channel.py:471`, `:485`, `:507` return
  `Token(...)` on all three paths. A `Token` "carries no SSA value … AIR's own asynchrony is
  built by the `air-dependency` pass from the program order this tracer emits"
  (`_value.py:26-33`). So `ChannelSite.is_async` has no counterpart in `air.api`: asynchrony is
  a property of the pass pipeline, not of the traced call, and M5 emits the same call either
  way. The field is recorded in the plan and read by nothing here.
* `dependency=` takes a `Token` or a list/tuple of them and is **validated only** —
  `_check_dependency` (`ops.py:82-91`) raises `TypeError` for a non-`Token` and the value is
  otherwise unused by `_emit`. M5 therefore keeps the `Token` each site returned, keyed by
  `ChannelSite.id`, and passes `dependency=None` for `depends_on == ()` and the list of those
  tokens otherwise (LLD §3.2 rows 9-10).

**A `build()`-time verify failure** surfaces as a `RuntimeError` whose message starts
`air.api emitted invalid IR` — `LaunchContext._verify` catches the MLIR exception and re-raises
it in that form (`_compile.py:174-180`). That marker, and only that marker, is `EMIT-VERIFY`;
every other exception out of `air.api` is `EMIT-AIR-API`, with the original text kept verbatim
in `details["air_api_message"]` (design/02-hld.md §4.3, NFR-4).
"""

from __future__ import annotations

import operator
from contextlib import contextmanager
from typing import Any

from spatial.model import (BinOp, BranchNode, BufferPlan, ChannelSite, Const, Diagnostic, Dtype,
                           EmissionError, EmitResult, Expr, Guard, HerdPlan, Load, LoopPlan,
                           MappingPlan, MaxMin, Neg, Select, StoreNode, Target)

_CLAUSE = "build()"
"""Every emission `Diagnostic` needs a clause (design/03-lld-M7-tests.md §3.2); this is it."""

_TARGETS = ("npu1", "npu2")
"""The two resolved targets. `"auto"` is never resolved here (D-13, LLD §4 P-7)."""

_VERIFY_MARK = "air.api emitted invalid IR"
"""`LaunchContext._verify`'s own prefix for a `module.operation.verify()` failure
(`python/air/api/_compile.py:174-180`). It is the only signal that tells a verify failure apart
from any other `RuntimeError` out of `air.api`."""

_FLOAT_DTYPES = (Dtype.f32, Dtype.f16, Dtype.bf16)
"""Which `Const.text` tokens are read with `float(...)` rather than `int(...)` (LLD §3.6)."""

_HERD_PRIVATE = "herd.private"
"""The one allocation scope LLD §3.2 row 6 gives a call for; §8 lists the rest as not emitted."""

_BUG_FIX = ("this is a defect in the compiler, not in your program: please report it with the "
            "plan JSON (spatial.model.to_json(plan)) and this message")

_BINOPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}
_RELATIONS = {"==": operator.eq, "!=": operator.ne, "<": operator.lt,
              "<=": operator.le, ">": operator.gt, ">=": operator.ge}
"""`Guard.relation` → the Python operator on an `IndexExpr`, which builds a `Condition`."""

_ORDERINGS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}
"""The four `Select.cmp_op`s that **are** operators on a buffer value.

`==` and `!=` are not, and this is a measured asymmetry rather than an oversight: `BufferExpr`
and `BufferSlice` deliberately leave `__eq__`/`__ne__` undefined, because defining `__eq__` sets
`__hash__` to `None` and changes what `slice == slice` means for ordinary Python
(`python/air/api/_value.py:795-805`). `x == y` on two buffer values is therefore Python's
identity comparison and evaluates to a `bool` before `ops.select` ever sees it, which
`ops.select` detects and rejects by name (`ops.py:809-816`). `ops.equal` / `ops.not_equal`
(`ops.py:779-792`) are the spelling, and they build the same `arith.cmpi`/`cmpf` the operators
build. W3's substitution score is the first `Select` in the project and the first to need them.
"""


def _fail(code: str, reason: str, fix: str, details: dict[str, Any]) -> EmissionError:
    """Build the `EmissionError` for `code`; every one carries all four message parts."""
    return EmissionError(Diagnostic(code=code, stage="emission", clause=_CLAUSE, reason=reason,
                                    fix=fix, location=None, details=details))


def _row_major(shape: tuple[int, ...]) -> tuple[int, ...]:
    """The row-major strides of `shape`, which is what `air.api` derives (LLD §3.7)."""
    strides = [1] * len(shape)
    for dim in range(len(shape) - 2, -1, -1):
        strides[dim] = strides[dim + 1] * shape[dim + 1]
    return tuple(strides)


class _Emitter:
    """One emission of one plan: the `air.api` handles, the name environment, the tokens."""

    def __init__(self, plan: MappingPlan, target: Target, air: Any, coerce: Any) -> None:
        self.plan = plan
        self.target = target
        self.air = air
        self.coerce = coerce                       # BufferExpr.coerce, see `_expr`'s Load arm
        self.node: Any = plan                      # the node being emitted, for diagnostics
        self.values: dict[str, Any] = {}           # buffer / tensor name -> air.api value
        self.shapes: dict[str, tuple[int, ...]] = {}   # the same names -> declared shape
        self.channels: dict[str, Any] = {}         # channel name -> air.api Channel
        self.tokens: dict[str, Any] = {}           # ChannelSite.id -> the Token it returned
        self.env: dict[str, tuple[Any, str]] = {}  # index name -> (value, "coord"|"iv"|"int")
        self.herd: Any = None
        self.dtypes = {Dtype.f32: air.f32, Dtype.f16: air.f16, Dtype.bf16: air.bf16,
                       Dtype.i32: air.i32, Dtype.i8: air.i8}

    # -- diagnostics ---------------------------------------------------------

    def _bug(self, reason: str, **details: Any) -> EmissionError:
        """An input guarantee M4 owed us was broken: our defect, never the user's (LLD §5)."""
        return _fail("EMIT-AIR-API", reason, _BUG_FIX,
                     {"internal_consistency": True, "node": repr(self.node),
                      "workload": self.plan.launch_name, **details})

    def _from_air_api(self, exc: BaseException) -> EmissionError:
        """Wrap an exception raised inside `air.api`, keeping its own text verbatim."""
        message = str(exc)
        where = f"{type(self.node).__name__} of {self.plan.launch_name!r}"
        if isinstance(exc, RuntimeError) and _VERIFY_MARK in message:
            return _fail(
                "EMIT-VERIFY",
                f"the emitted module failed module.operation.verify() while emitting the "
                f"{where}",
                "read details['mlir_diagnostic']; " + _BUG_FIX,
                {"mlir_diagnostic": message, "air_api_message": message,
                 "workload": self.plan.launch_name, "node": repr(self.node)})
        return _fail(
            "EMIT-AIR-API",
            f"air.api rejected the {where}",
            "read details['air_api_message'] — it is air.api's own explanation — and change "
            "the clause that produced this node; the summary names the buffer or channel",
            {"air_api_message": message, "air_api_error": type(exc).__name__,
             "workload": self.plan.launch_name, "node": repr(self.node)})

    # -- the name environment (LLD §3.4) -------------------------------------

    @contextmanager
    def _bound(self, name: str, value: Any, kind: str):
        """Bind one index name for the duration of a body, then restore what it shadowed."""
        previous = self.env.get(name)
        self.env[name] = (value, kind)
        try:
            yield
        finally:
            if previous is None:
                del self.env[name]
            else:
                self.env[name] = previous

    @contextmanager
    def _herd_scope(self, coords: tuple[str, ...], values: tuple[Any, ...]):
        """The herd body sees its own coordinates and nothing else: it is `IsolatedFromAbove`."""
        outer = self.env
        self.env = {name: (value, "coord") for name, value in zip(coords, values)}
        try:
            yield
        finally:
            self.env = outer

    def _lookup(self, name: str) -> tuple[Any, str]:
        entry = self.env.get(name)
        if entry is None:
            raise self._bug(f"the index name {name!r} is not bound at this point in the plan; "
                            f"bound here: {sorted(self.env)}", name=name)
        return entry

    def _index(self, expr: Expr) -> Any:
        """`EVAL_EXPR`: `sum(coeff * env[name]) + const`, in the `Expr`'s own sorted order."""
        value: Any = expr.const
        for name, coeff in expr.coeffs:
            value = value + coeff * self._lookup(name)[0]
        return value

    def _bundle_index(self, expr: Expr) -> Any:
        """A bundle index, checked against `AIRDialect.cpp:3589-3592` (LLD §3.4, §5)."""
        for name, _ in expr.coeffs:
            if self._lookup(name)[1] == "iv":
                raise self._bug(
                    f"the bundle index references {name!r}, which is a sequential loop's "
                    f"induction variable; M4's BUNDLE-INDEX-IS-IV check should have rejected "
                    f"this plan (design/06-interfaces.md §5.6 invariant 3)", axis=name)
        return self._index(expr)

    def _value_of(self, name: str) -> Any:
        value = self.values.get(name)
        if value is None:
            raise self._bug(f"the buffer or tensor {name!r} is used before it is declared; "
                            f"declared here: {sorted(self.values)}", buffer=name)
        return value

    def _channel_of(self, name: str) -> Any:
        channel = self.channels.get(name)
        if channel is None:
            raise self._bug(f"the channel {name!r} is not in plan.channels", channel=name)
        return channel

    # -- the translation table (LLD §3.2) ------------------------------------

    def _declare(self, air: Any) -> None:
        """Rows 1 and 5: the L3 interface in plan order, then the channels in name order."""
        for tensor in self.plan.tensors:                        # row 1
            self.node = tensor
            self.values[tensor.name] = air.tensor(list(tensor.shape),
                                                  self.dtypes[tensor.dtype], name=tensor.name)
            self.shapes[tensor.name] = tensor.shape
        for channel in self.plan.channels:                      # rows 5a, 5b, 5c
            self.node = channel
            self.channels[channel.name] = air.channel(
                channel.name, size=list(channel.size),
                broadcast_shape=channel.broadcast_shape, channel_type=channel.channel_type)

    def _walk(self, nodes: tuple[Any, ...], *, in_loop_body: bool) -> None:
        """Emit a plan body in plan order (LLD §3.3 rule 1)."""
        for node in nodes:
            self.node = node
            if isinstance(node, BufferPlan):
                self._alloc(node, in_loop_body=in_loop_body)
            elif isinstance(node, ChannelSite):
                self._site(node)
            elif isinstance(node, LoopPlan):
                self._loop(node)
            elif isinstance(node, StoreNode):
                self._store(node)
            elif isinstance(node, BranchNode):
                self._branch(node)
            elif isinstance(node, HerdPlan):
                self._herd(node)
            else:
                raise self._bug(f"{type(node).__name__} is not a PlanNode "
                                f"(design/06-interfaces.md §5.5)")

    def _alloc(self, buffer: BufferPlan, *, in_loop_body: bool) -> None:
        """Row 6: `air.alloc(shape, dtype, scope=<herd>.private())`."""
        if buffer.scope != _HERD_PRIVATE:
            raise self._bug(
                f"buffer {buffer.name!r} asks for scope {buffer.scope!r}; M5 emits only "
                f"{_HERD_PRIVATE!r} (design/03-lld-M5-emitter.md §3.2 row 6, §8)",
                buffer=buffer.name, scope=buffer.scope)
        if buffer.ping_pong_candidate and not in_loop_body:
            raise self._bug(
                f"buffer {buffer.name!r} is a ping-pong candidate but is not a direct child of "
                f"a LoopPlan body (design/06-interfaces.md §5.6 invariant 4)", buffer=buffer.name)
        if self.herd is None:
            raise self._bug(f"buffer {buffer.name!r} is allocated outside the herd body")
        self.values[buffer.name] = self.air.alloc(list(buffer.shape),
                                                  self.dtypes[buffer.dtype],
                                                  scope=self.herd.private())
        self.shapes[buffer.name] = buffer.shape

    def _herd(self, herd: HerdPlan) -> None:
        """Row 4: open the herd where its marker node sits, and walk `plan.herd_body`."""
        air = self.air
        keywords: dict[str, Any] = {"name": herd.name, "shape": herd.shape}
        if herd.at is not None:
            keywords["at"] = herd.at
        rank = len(herd.coords)
        if rank not in (1, 2):
            raise self._bug(f"the herd has {rank} coordinate(s); air.api supports 1-D and 2-D "
                            f"herds only (python/air/api/_trace.py:1288-1291)")

        def run(values: tuple[Any, ...]) -> None:
            with self._herd_scope(herd.coords, values):
                self._walk(self.plan.herd_body, in_loop_body=False)

        # The body's positional arity is the herd's rank: `_positional_arity`
        # (python/air/api/_trace.py:725-731) counts declared positional parameters, so `*args`
        # would read as arity 0 and fail air.api's own check.
        if rank == 1:
            def body(c0):                                        # noqa: ANN001 — traced body
                run((c0,))
        else:
            def body(c0, c1):                                    # noqa: ANN001 — traced body
                run((c0, c1))

        with air.herd([range(extent) for extent in herd.grid], **keywords) as handle:
            outer, self.herd = self.herd, handle
            try:
                handle.body(body)
            finally:
                self.herd = outer

    def _loop(self, loop: LoopPlan) -> None:
        """Rows 7 and 8: one `scf.for`, or a trace-time Python loop."""
        lo, hi, step = (self._index(bound) for bound in (loop.lo, loop.hi, loop.step))
        if loop.kind == "sequential":
            for induction in self.air.sequential(lo, hi, step):
                with self._bound(loop.axis, induction, "iv"):
                    self._walk(loop.body, in_loop_body=True)
                    self.node = loop
            return
        if not all(type(bound) is int for bound in (lo, hi, step)):
            raise self._bug(
                f"the unrolled loop over {loop.axis!r} has a bound that is not a trace-time "
                f"constant (design/06-interfaces.md §5.5)", axis=loop.axis)
        for value in range(lo, hi, step):
            with self._bound(loop.axis, value, "int"):
                self._walk(loop.body, in_loop_body=True)
                self.node = loop

    def _branch(self, branch: BranchNode) -> None:
        """Row 11: `ops.branch` plus its `otherwise()`; conjunction is nesting."""
        with self.air.ops.branch(self._predicate(branch.predicate)) as handle:
            self._walk(branch.then, in_loop_body=False)
        self.node = branch
        if branch.otherwise:
            with handle.otherwise():
                self._walk(branch.otherwise, in_loop_body=False)
            self.node = branch

    def _predicate(self, guard: Guard) -> Any:
        coordinate, kind = self._lookup(guard.coord)
        if kind == "iv":
            raise self._bug(f"the guard compares {guard.coord!r}, which is a sequential loop's "
                            f"induction variable, not a herd coordinate", axis=guard.coord)
        return _RELATIONS[guard.relation](coordinate, self._index(guard.value))

    def _site(self, site: ChannelSite) -> None:
        """Rows 9, 10 and 11b: one `put` or `get`, under its guard when it has one."""
        if site.guard is None:
            self._transfer(site)
            return
        with self.air.ops.branch(self._predicate(site.guard)):
            self._transfer(site)

    def _transfer(self, site: ChannelSite) -> None:
        channel = self._channel_of(site.channel)
        obj = self._slice(site)
        indices = [self._bundle_index(index) for index in site.indices]
        dependency = None
        if site.depends_on:
            dependency = [self._token(site, other) for other in site.depends_on]
        move = channel.put if site.kind == "put" else channel.get
        self.tokens[site.id] = move(obj, indices=indices, dependency=dependency)

    def _token(self, site: ChannelSite, other: str) -> Any:
        token = self.tokens.get(other)
        if token is None:
            raise self._bug(f"site {site.id!r} depends on {other!r}, which has not been emitted "
                            f"yet (design/06-interfaces.md §5.2)", depends_on=other)
        return token

    def _slice(self, site: ChannelSite) -> Any:
        """LLD §3.7: an empty `Region` is the whole object, else a numpy-style subscript."""
        obj = self._value_of(site.buffer)
        region = site.region
        if not region.offsets:
            return obj
        shape = self.shapes[site.buffer]
        if len(shape) != len(region.sizes):
            raise self._bug(
                f"site {site.id!r} has a rank-{len(region.sizes)} region over {site.buffer!r}, "
                f"which has shape {shape} (design/06-interfaces.md §5.2)", buffer=site.buffer)
        expected = _row_major(shape)
        if tuple(region.strides) != expected:
            raise self._bug(
                f"site {site.id!r} has strides {tuple(region.strides)} where row-major over "
                f"{shape} is {expected}; air.api derives the strides itself "
                f"(design/03-lld-M5-emitter.md §3.7)",
                strides=list(region.strides), row_major=list(expected))
        subscript = []
        for offset, size in zip(region.offsets, region.sizes):
            start = self._index(offset)
            subscript.append(slice(start, start + size))
        return obj[tuple(subscript)]

    # -- compute nodes (LLD §3.6) -------------------------------------------

    def _store(self, store: StoreNode) -> None:
        """Row 12: `<buffer>[<subs>] = EMIT_EXPR(node.expr)`."""
        target = self._value_of(store.buffer_id)
        subscript = tuple(self._index(index) for index in store.subscripts)
        target[subscript] = self._expr(store.expr)

    def _expr(self, node: Any) -> Any:
        """`EMIT_EXPR`: one branch per `ExprNode` case, and no case reads the kernel."""
        if isinstance(node, Load):
            # `buf[subs]` is a `BufferSlice`, and `ops.maximum`/`minimum` refuse one: their
            # `_elementwise` guard admits `(Buffer, BufferExpr, int, float)` only
            # (`python/air/api/ops.py:384-391`), although `BufferExpr.coerce` on the next line
            # handles a `BufferSlice` (`python/air/api/_value.py:1046-1049`) and `_comparison`
            # and `select` both list it. So the coercion is applied here, through the API's own
            # entry point, rather than by wrapping the load in arithmetic — `load + 0` would put
            # an `arith.addi` in the IR that the plan does not ask for. It is a no-op for every
            # other consumer: `BufferSlice.__add__` and friends call `_as_leaf()` first
            # (`_value.py:749-751`), which is what `coerce` calls, so W1's text is unchanged.
            # Recorded as the closure of **B-P16**.
            return self.coerce(
                self._value_of(node.buffer_id)[tuple(self._index(i) for i in node.subscripts)])
        if isinstance(node, Const):
            return float(node.text) if node.dtype in _FLOAT_DTYPES else int(node.text)
        if isinstance(node, BinOp):
            return _BINOPS[node.op](self._expr(node.lhs), self._expr(node.rhs))
        if isinstance(node, Neg):
            return -self._expr(node.operand)
        if isinstance(node, MaxMin):
            fold = self.air.ops.maximum if node.op == "maximum" else self.air.ops.minimum
            value = self._expr(node.operands[-1])
            for operand in reversed(node.operands[:-1]):
                value = fold(self._expr(operand), value)
            return value
        if isinstance(node, Select):
            return self.air.ops.select(self._compare(node), self._expr(node.then),
                                       self._expr(node.otherwise))
        raise self._bug(f"{type(node).__name__} is not an ExprNode "
                        f"(design/06-interfaces.md §5.5)")

    def _compare(self, node: Select) -> Any:
        """A `Select`'s predicate: an `arith.cmpi`/`cmpf` over two buffer values (`_ORDERINGS`)."""
        lhs, rhs = self._expr(node.lhs), self._expr(node.rhs)
        if node.cmp_op == "==":
            return self.air.ops.equal(lhs, rhs)
        if node.cmp_op == "!=":
            return self.air.ops.not_equal(lhs, rhs)
        return _ORDERINGS[node.cmp_op](lhs, rhs)

    # -- the whole of it (LLD §3.1) -----------------------------------------

    def run(self) -> EmitResult:
        """Declare, trace, build, and read the text and the L1 peak back."""
        air = self.air
        plan = self.plan
        try:
            self._declare(air)
            with air.launch(name=plan.launch_name) as launch:

                @launch.body
                def _() -> None:
                    with air.segment(name=plan.segment_name) as segment:

                        @segment.body
                        def _() -> None:
                            self._walk(plan.segment_body, in_loop_body=False)

                self.node = plan
                module = launch.build(target=self.target)       # row 13; runs verify()
        except EmissionError:
            raise
        except Exception as exc:                                # noqa: BLE001 — NFR-7, §5
            raise self._from_air_api(exc) from None
        if len(launch.tensors) != len(plan.tensors):
            raise self._bug(
                f"the launch claimed {len(launch.tensors)} tensor(s) where the plan declares "
                f"{len(plan.tensors)}; a previous failed emission left tensors pending",
                claimed=len(launch.tensors), declared=len(plan.tensors))
        peak = int(launch._l1_peak)                             # _compile.py:74, :150
        self._check_l1(peak)
        return EmitResult(mlir=str(module), target=self.target, plan=plan,
                          summary=plan.summary, l1_peak=peak)

    def _check_l1(self, peak: int) -> None:
        """LLD §5: the plan's L1 total may exceed `air.api`'s peak only by the ping-pong pair."""
        total = self.plan.summary.l1_bytes
        if peak == 0 and total == 0:
            return
        if not peak <= total <= 2 * peak:
            raise self._bug(
                f"air.api reports an L1 peak of {peak} B where the plan's summary says "
                f"{total} B; the two may differ only by the ping-pong doubling "
                f"(design/03-lld-M5-emitter.md §5)", l1_peak=peak, plan_l1_bytes=total)


def emit(plan: MappingPlan, target: Target) -> EmitResult:
    """Translate the plan mechanically into air.api. Raises EmissionError."""
    if target not in _TARGETS:
        raise _fail(
            "EMIT-AIR-API",
            f"target {target!r} is not a resolved AIR target",
            'pass target="npu1" or target="npu2"',
            {"given": target, "accepted": list(_TARGETS),
             "why": "M5 never resolves \"auto\": it shells out to xrt-smi "
                    "(python/air/api/_trace.py:164-183), and decision D-13 keeps that in M6"})
    from air import api as air                                  # noqa: PLC0415 — lazy, FR-S20
    # `BufferExpr` is not re-exported by `air.api.__init__` (it exports `Buffer`, `BufferSlice`,
    # `Tensor`, `TensorSlice`, `Token`), so its own module is the only route to the coercion
    # `ops.maximum` needs — §3.2's table lists the constructs M5 *emits*, and this emits none.
    from air.api._value import BufferExpr                        # noqa: PLC0415 — lazy, FR-S20

    return _Emitter(plan, target, air, BufferExpr.coerce).run()
