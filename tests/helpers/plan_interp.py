"""A numpy interpreter for a `MappingPlan`. Owner: Person B. Spec: `04-test-plan.md` §3.4.

**It executes the plan, not the emitted IR.** `air.api` has no interpreter (VF §D.10),
`air-runner` is a timing model rather than a functional one (VF §S7), and
`air/backend/cpu_backend.py` JITs a *lowered* module through torch-mlir's RefBackend, which tests
the compiler's output and not the user's source (PC §1.4 item 1). So this closes the gap between
"the kernel is right" and "the plan says the same thing" — decision **D-9** — and the
honest-limits slide still says only a device run establishes end-to-end semantics.

What it models, and how faithfully:

* one coroutine for the segment body and one per herd coordinate, interleaved round-robin —
  the launch, the segment and the herd are concurrent async regions, so the `HerdPlan` marker is
  a no-op here and the herd does **not** wait for the segment's puts;
* one FIFO per `(channel, concrete index)`; a put on a broadcast channel copies its slab into
  **every** fan-out index's FIFO (decision D-2); a get blocks — the coroutine yields — while its
  FIFO is empty, so a full scheduling round with no completed transfer is a **deadlock**, which
  is a dynamic P2 check the static one cannot be checked against;
* `air.alloc` inside a loop re-allocates per trip, which is what a fresh array per allocation
  occurrence models;
* every region is a row-major slab: the strides are asserted, never used to index.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from spatial.model import (BranchNode, BufferPlan, ChannelSite, Const, Expr, Guard, HerdPlan,
                           Load, LoopPlan, MappingPlan, Region, StoreNode)

_RELATION = {"==": lambda a, b: a == b, "!=": lambda a, b: a != b, "<": lambda a, b: a < b,
             "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b}
_BINOP = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
          "*": lambda a, b: a * b, "/": lambda a, b: a / b}


def evaluate(expr: Expr, env: dict[str, int]) -> int:
    """An affine `Expr` over bound coordinates and loop variables."""
    return expr.const + sum(coeff * env[name] for name, coeff in expr.coeffs)


def row_major(shape: tuple[int, ...]) -> tuple[int, ...]:
    """The row-major strides of `shape` — the only layout a `Region` may describe (§3.4)."""
    strides = [1] * len(shape)
    for dim in range(len(shape) - 2, -1, -1):
        strides[dim] = strides[dim + 1] * shape[dim + 1]
    return tuple(strides)


def slab(array: np.ndarray, region: Region, env: dict[str, int]) -> np.ndarray:
    """The view a `Region` names. An empty region is the whole buffer (`03-lld-M4-mapping.md` §3.4)."""
    if not region.offsets:
        return array
    assert tuple(region.strides) == row_major(array.shape), (
        f"region strides {tuple(region.strides)} are not row-major over {array.shape}")
    index = tuple(slice(start, start + size) for start, size in
                  ((evaluate(offset, env), size)
                   for offset, size in zip(region.offsets, region.sizes)))
    return array[index]


def region_indices(region: Region, env: dict[str, int]) -> set[tuple[int, ...]]:
    """The set of element coordinates a `Region` covers — what `test_sem_access_regions` compares."""
    from itertools import product
    starts = [evaluate(offset, env) for offset in region.offsets]
    return set(product(*(range(start, start + size)
                         for start, size in zip(starts, region.sizes))))


class _State:
    """The shared world: the L3 tensors, one FIFO per `(channel, index)`, and a progress count."""

    def __init__(self, plan: MappingPlan, tensors: dict[str, np.ndarray]) -> None:
        self.plan = plan
        self.tensors = tensors
        self.channels = {channel.name: channel for channel in plan.channels}
        self.fifos: dict[tuple[str, tuple[int, ...]], list[np.ndarray]] = {}
        self.ops = 0

    def key(self, site: ChannelSite, env: dict[str, int]) -> tuple[str, tuple[int, ...]]:
        return (site.channel, tuple(evaluate(index, env) for index in site.indices))

    def destinations(self, site: ChannelSite,
                     env: dict[str, int]) -> list[tuple[str, tuple[int, ...]]]:
        """Every FIFO one put feeds: its own index, or each index of its fan-out set (D-2)."""
        channel = self.channels[site.channel]
        name, index = self.key(site, env)
        if channel.broadcast_shape is None:
            return [(name, index)]
        from itertools import product
        return [(name, destination)
                for destination in product(*(range(extent)
                                             for extent in channel.broadcast_shape))
                if all(destination[k] % channel.size[k] == index[k]
                       for k in range(len(channel.size)))]


class _Actor:
    """One concurrent region: its own L1 buffers, its own environment, one coroutine."""

    def __init__(self, name: str, body: tuple[Any, ...], env: dict[str, int],
                 state: _State) -> None:
        self.name = name
        self.state = state
        self.buffers: dict[str, np.ndarray] = {}
        self.blocked: ChannelSite | None = None
        self.steps = self._exec(body, dict(env))

    def array(self, name: str) -> np.ndarray:
        """The buffer or L3 tensor a site or a `Load` names."""
        if name in self.buffers:
            return self.buffers[name]
        return self.state.tensors[name]

    def _holds(self, guard: Guard, env: dict[str, int]) -> bool:
        return _RELATION[guard.relation](env[guard.coord], evaluate(guard.value, env))

    def _value(self, node: Any, env: dict[str, int]) -> Any:
        """One `ExprNode`, elementwise (`06-interfaces.md` §5.5)."""
        if isinstance(node, Load):
            return self.array(node.buffer_id)[
                tuple(evaluate(sub, env) for sub in node.subscripts)]
        if isinstance(node, Const):
            return node.value
        kind = type(node).__name__
        if kind == "BinOp":
            return _BINOP[node.op](self._value(node.lhs, env), self._value(node.rhs, env))
        if kind == "Neg":
            return -self._value(node.operand, env)
        if kind == "MaxMin":
            values = [self._value(item, env) for item in node.operands]
            return max(values) if node.op == "maximum" else min(values)
        if kind == "Select":
            taken = _RELATION[node.cmp_op](self._value(node.lhs, env),
                                           self._value(node.rhs, env))
            return self._value(node.then if taken else node.otherwise, env)
        raise TypeError(f"plan_interp: no rule for {kind}")

    def _exec(self, nodes: tuple[Any, ...], env: dict[str, int]) -> Iterator[ChannelSite]:
        """Execute one body, yielding the site it is blocked on whenever a get must wait."""
        for node in nodes:
            if isinstance(node, BufferPlan):
                self.buffers[node.name] = np.zeros(node.shape, dtype=node.dtype.numpy)
            elif isinstance(node, LoopPlan):
                for value in range(evaluate(node.lo, env), evaluate(node.hi, env),
                                   evaluate(node.step, env)):
                    yield from self._exec(node.body, {**env, node.axis: value})
            elif isinstance(node, BranchNode):
                taken = node.then if self._holds(node.predicate, env) else node.otherwise
                yield from self._exec(taken, env)
            elif isinstance(node, StoreNode):
                self.array(node.buffer_id)[
                    tuple(evaluate(sub, env) for sub in node.subscripts)] = self._value(
                        node.expr, env)
            elif isinstance(node, ChannelSite):
                if node.guard is not None and not self._holds(node.guard, env):
                    continue
                yield from self._transfer(node, env)
            elif isinstance(node, HerdPlan):
                continue                      # the herd is a concurrent region, not a call

    def _transfer(self, site: ChannelSite, env: dict[str, int]) -> Iterator[ChannelSite]:
        """One put or get. A get yields until its FIFO has a payload (the dynamic P2 check)."""
        state = self.state
        if site.kind == "put":
            payload = np.array(slab(self.array(site.buffer), site.region, env), copy=True)
            for destination in state.destinations(site, env):
                state.fifos.setdefault(destination, []).append(payload)
        else:
            key = state.key(site, env)
            while not state.fifos.get(key):
                self.blocked = site
                yield site
            self.blocked = None
            target = slab(self.array(site.buffer), site.region, env)
            target[...] = state.fifos[key].pop(0).reshape(target.shape)
        state.ops += 1


def run(plan: MappingPlan, tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Execute the plan's semantics over numpy, returning the (mutated) L3 tensors.

    Deterministic: the segment actor first, then the herd coordinates in lexicographic order,
    round-robin until every actor has finished. A whole round with no completed transfer and no
    actor finishing is a deadlock, and the error names every blocked site.
    """
    from itertools import product

    state = _State(plan, tensors)
    actors = [_Actor("segment", plan.segment_body, {}, state)]
    for coord in product(*(range(extent) for extent in plan.herd.grid)):
        actors.append(_Actor(f"herd{list(coord)}", plan.herd_body,
                             dict(zip(plan.herd.coords, coord)), state))
    live = list(actors)
    while live:
        before, finished = state.ops, False
        for actor in list(live):
            try:
                next(actor.steps)
            except StopIteration:
                live.remove(actor)
                finished = True
        if live and state.ops == before and not finished:
            blocked = ", ".join(f"{a.name} on {a.blocked.id}" for a in live if a.blocked)
            raise RuntimeError(f"deadlock: no actor can make progress — blocked: {blocked}")
    return tensors
