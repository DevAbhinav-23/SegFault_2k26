"""M1 -- frontend / kernel capture. Owner: Person A. LLD: design/03-lld-M1-frontend.md.

Entry point per design/06-interfaces.md §7.2. `capture(fn)` turns the source text of a
`@sp.kernel`-decorated function into a `KernelModel`. `kernel(fn)` is the `sp.kernel` decorator
itself (§7.1): it calls `capture` and wraps the result, with the original function, in a
`Kernel` that **is** the CPython oracle -- `Kernel.__call__` is exactly `fn.__call__`, nothing
else.

M1 never imports `air`, `numpy`, or anything outside the standard library (FR-S20's spirit: this
half of the surface must work with no toolchain installed at all).
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, Callable, NoReturn

from spatial import intlin
from spatial.model import (
    AccessMap, Axis, BinOp, Const, Dependence, Diagnostic, Dtype, Expr, ExprNode, GrammarError,
    KernelModel, Load, MaxMin, Neg, Param, ReductionSpec, Select, Statement,
)

_DTYPES = {"f32": Dtype.f32, "f16": Dtype.f16, "bf16": Dtype.bf16, "i32": Dtype.i32,
           "i8": Dtype.i8}


# ------------------------------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------------------------------


def _loc(node: ast.AST | None, filename: str) -> tuple[str, int] | None:
    if node is None or not hasattr(node, "lineno"):
        return None
    return (filename, node.lineno)


def _raise(code: str, reason: str, fix: str, node: ast.AST | None, filename: str,
           **details: Any) -> NoReturn:
    raise GrammarError(Diagnostic(
        code=code, stage="grammar", clause=None, reason=reason, fix=fix,
        location=_loc(node, filename), details=details))


# ------------------------------------------------------------------------------------------
# §3.4 -- parameters and loop-domain extraction
# ------------------------------------------------------------------------------------------


def _render_annotation(ann: ast.AST) -> str:
    try:
        return ast.unparse(ann)
    except Exception:  # pragma: no cover -- defensive only
        return "<unparsable>"


def _try_resolve_shape_expr(node: ast.AST, globals_: dict[str, Any]) -> int | None:
    """Fully resolve a shape entry to an int via module globals, or None if it can't be."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.Name):
        v = globals_.get(node.id)
        return v if type(v) is int else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left = _try_resolve_shape_expr(node.left, globals_)
        right = _try_resolve_shape_expr(node.right, globals_)
        if left is None or right is None:
            return None
        return left + right if isinstance(node.op, ast.Add) else left - right
    return None


def _parse_params(args: ast.arguments, filename: str,
                  globals_: dict[str, Any]) -> tuple[tuple[Param, ...], set[str]]:
    params: list[Param] = []
    shape_param_names: set[str] = set()
    for a in args.args:
        ann = a.annotation
        ok = (isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Attribute)
              and isinstance(ann.value.value, ast.Name) and ann.value.value.id == "sp"
              and ann.value.attr in _DTYPES)
        if not ok:
            _raise("GRAMMAR-BAD-ANNOTATION",
                   f"parameter '{a.arg}' is annotated "
                   f"'{_render_annotation(ann) if ann else None}'; kernel parameters must be "
                   f"annotated sp.<dtype>[<dim>, ...]",
                   f"write {a.arg}: sp.f32[M, K]", a, filename,
                   param=a.arg, seen=_render_annotation(ann) if ann else None,
                   dtypes=sorted(_DTYPES))
        dtype = _DTYPES[ann.value.attr]  # type: ignore[union-attr]
        idx = ann.slice  # type: ignore[union-attr]
        entries = list(idx.elts) if isinstance(idx, ast.Tuple) else [idx]
        shape: list[int | str] = []
        for e in entries:
            if isinstance(e, ast.Constant) and isinstance(e.value, int):
                shape.append(e.value)
            elif isinstance(e, ast.Name):
                shape.append(e.id)
                shape_param_names.add(e.id)
            elif isinstance(e, ast.BinOp):
                # A shape entry may itself be an affine form over shape params, e.g. `T + 1`.
                # Every bare name referenced here is a shape parameter (03-lld-M1-frontend.md
                # §3.4 line 2), regardless of whether the whole entry resolves to an int for
                # Param.shape's stored value: `T + 1` -> Param.shape gets 5, but `T` is still a
                # shape parameter (it appears again, unresolved, in `range(0, T)`).
                names = [n.id for n in ast.walk(e) if isinstance(n, ast.Name)]
                shape_param_names.update(names)
                resolved = _try_resolve_shape_expr(e, globals_)
                if resolved is not None:
                    shape.append(resolved)
                    continue
                names = [n.id for n in ast.walk(e) if isinstance(n, ast.Name)]
                if len(names) == 1:
                    shape.append(names[0])
                else:
                    _raise("GRAMMAR-BAD-ANNOTATION",
                           f"parameter '{a.arg}' has a shape entry that is not a single int or "
                           f"identifier: '{_render_annotation(e)}'",
                           "use a plain shape parameter name or an integer literal",
                           a, filename, param=a.arg, seen=_render_annotation(e))
            else:
                _raise("GRAMMAR-BAD-ANNOTATION",
                       f"parameter '{a.arg}' has an unsupported shape entry "
                       f"'{_render_annotation(e)}'",
                       "use a plain shape parameter name or an integer literal",
                       a, filename, param=a.arg, seen=_render_annotation(e))
        params.append(Param(name=a.arg, dtype=dtype, shape=tuple(shape), is_written=False))
    return tuple(params), shape_param_names


def _mark_written(params: tuple[Param, ...], written: set[str]) -> tuple[Param, ...]:
    return tuple(Param(name=p.name, dtype=p.dtype, shape=p.shape,
                       is_written=p.name in written) for p in params)


def _resolve(e: Expr, globals_: dict[str, Any]) -> int | None:
    """§3.4 RESOLVE: an Expr's value if every free name is bound to a Python int in globals."""
    if e.is_constant:
        return e.const
    total = e.const
    for name, coeff in e.coeffs:
        val = globals_.get(name)
        if type(val) is not int:
            return None
        total += coeff * val
    return total


def _affine_bound(node: ast.AST, allowed: set[str], enclosing: set[str], filename: str) -> Expr:
    """AFFINE restricted to §3.4's `for`-bound context: only shape params / constants."""
    return _affine(node, allowed, filename, enclosing_for_message=enclosing)


def _extract_domain(body: list[ast.stmt], shape_params: set[str],
                    globals_: dict[str, Any], filename: str) -> tuple[Axis, ...]:
    axes: list[Axis] = []
    axis_names: set[str] = set()

    def walk(stmts: list[ast.stmt], depth: int) -> None:
        for s in stmts:
            if isinstance(s, ast.For):
                if not isinstance(s.target, ast.Name) or not (
                        isinstance(s.iter, ast.Call) and isinstance(s.iter.func, ast.Name)
                        and s.iter.func.id == "range"):
                    _raise("GRAMMAR-BAD-RANGE",
                           f"'for' must iterate a plain name over range(...); got "
                           f"{type(s.iter).__name__}",
                           "use for <name> in range(...)", s, filename,
                           construct=type(s.iter).__name__)
                nargs = s.iter.args
                if not (1 <= len(nargs) <= 3):
                    _raise("GRAMMAR-BAD-RANGE", "range() takes 1 to 3 arguments",
                           "use range(hi), range(lo, hi) or range(lo, hi, step)", s, filename)
                if len(nargs) == 1:
                    lo_node, hi_node, step_node = None, nargs[0], None
                elif len(nargs) == 2:
                    lo_node, hi_node, step_node = nargs[0], nargs[1], None
                else:
                    lo_node, hi_node, step_node = nargs
                lo = (_affine_bound(lo_node, shape_params, axis_names, filename)
                      if lo_node is not None else Expr((), 0))
                hi = _affine_bound(hi_node, shape_params, axis_names, filename)
                if step_node is not None:
                    if not (isinstance(step_node, ast.Constant)
                            and type(step_node.value) is int and step_node.value >= 1):
                        _raise("GRAMMAR-BAD-RANGE",
                               "the step of range(...) must be a positive integer literal",
                               "use a positive integer step, or omit it for step 1",
                               step_node, filename)
                    step = Expr((), step_node.value)
                else:
                    step = Expr((), 1)
                name = s.target.id
                if name in axis_names:
                    _raise("GRAMMAR-BAD-RANGE", f"axis name '{name}' is already in use",
                           "use a distinct loop-variable name", s, filename, axis=name)
                lo_val, hi_val, step_val = (_resolve(lo, globals_), _resolve(hi, globals_),
                                           step.const)
                extent = None
                if lo_val is not None and hi_val is not None:
                    extent = -(-(hi_val - lo_val) // step_val)  # ceil division
                axes.append(Axis(name=name, lo=lo, hi=hi, step=step, extent=extent,
                                 parent=None, depth=depth))
                axis_names.add(name)
                walk(s.body, depth + 1)
            elif isinstance(s, (ast.Assign, ast.AugAssign)):
                continue  # leaf statements; handled by §3.5
            else:
                # Unknown at this stage; §3.5's classify pass gives the precise message.
                continue

    walk(body, 0)
    if not axes:
        _raise("GRAMMAR-UNSUPPORTED-STMT", "a kernel must have at least one 'for' loop",
              "wrap the body in at least one for ... in range(...) loop",
              body[0] if body else None, filename)
    return tuple(axes)


# ------------------------------------------------------------------------------------------
# §3.6 -- affine access-map extraction (also used for `for` bounds, §3.4)
# ------------------------------------------------------------------------------------------


def _render_expr(e: Expr) -> str:
    parts = [f"{c}*{n}" if c != 1 else n for n, c in e.coeffs]
    if e.const or not parts:
        parts.append(str(e.const))
    return " + ".join(parts)


def _affine(node: ast.AST, allowed: set[str], filename: str,
           enclosing_for_message: set[str] | None = None) -> Expr:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return Expr((), node.value)
    if isinstance(node, ast.Name):
        if node.id in allowed:
            return Expr({node.id: 1}, 0)
        if enclosing_for_message is not None and node.id in enclosing_for_message:
            _raise("GRAMMAR-BAD-RANGE",
                  f"bound names the enclosing loop variable '{node.id}', so the domain is not "
                  f"rectangular",
                  "use range(0, N) and guard nothing -- the accepted domain is a box",
                  node, filename, bound=node.id, enclosing=sorted(enclosing_for_message))
        _raise("GRAMMAR-NONAFFINE-SUBSCRIPT", f"unknown name '{node.id}'",
              "use a loop variable or a shape parameter", node, filename, name=node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _affine(node.operand, allowed, filename, enclosing_for_message)
        return Expr({n: -c for n, c in inner.coeffs}, -inner.const)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left = _affine(node.left, allowed, filename, enclosing_for_message)
        right = _affine(node.right, allowed, filename, enclosing_for_message)
        sign = 1 if isinstance(node.op, ast.Add) else -1
        coeffs: dict[str, int] = dict(left.coeffs)
        for n, c in right.coeffs:
            coeffs[n] = coeffs.get(n, 0) + sign * c
        return Expr({n: c for n, c in coeffs.items() if c != 0},
                   left.const + sign * right.const)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left = _affine(node.left, allowed, filename, enclosing_for_message)
        right = _affine(node.right, allowed, filename, enclosing_for_message)
        if left.is_constant:
            return Expr({n: left.const * c for n, c in right.coeffs}, left.const * right.const)
        if right.is_constant:
            return Expr({n: right.const * c for n, c in left.coeffs}, right.const * left.const)
        _raise("GRAMMAR-NONAFFINE-SUBSCRIPT",
              f"subscript is a product of two non-constant terms {_render_expr(left)} * "
              f"{_render_expr(right)}",
              "subscripts must be affine: c1*i + c2*j + c0 with integer c", node, filename)
    _raise("GRAMMAR-NONAFFINE-SUBSCRIPT", f"unsupported subscript node {type(node).__name__}",
          "subscripts must be affine: c1*i + c2*j + c0 with integer c", node, filename,
          construct=type(node).__name__)


def _full_subscript(sub: ast.Subscript, axes: tuple[str, ...], shape_params: set[str],
                    params: dict[str, Param], filename: str) -> tuple[str, tuple[Expr, ...]]:
    """The operand name and the FULL per-dim `Expr` (axis coefficients included)."""
    if not isinstance(sub.value, ast.Name) or sub.value.id not in params:
        _raise("GRAMMAR-NONAFFINE-SUBSCRIPT",
              f"subscript target '{_render_annotation(sub.value)}' is not a kernel parameter",
              "subscript a declared kernel parameter", sub, filename)
    operand = sub.value.id  # type: ignore[union-attr]
    idx = sub.slice
    entries = list(idx.elts) if isinstance(idx, ast.Tuple) else [idx]
    rank = len(params[operand].shape)
    if len(entries) != rank:
        _raise("GRAMMAR-BAD-ANNOTATION",
              f"'{operand}' is accessed with {len(entries)} indices but has rank {rank}",
              f"use exactly {rank} indices", sub, filename, operand=operand,
              seen=len(entries), rank=rank)
    all_names = set(axes) | shape_params
    exprs = tuple(_affine(e, all_names, filename) for e in entries)
    return operand, exprs


def _extract_access(sub: ast.Subscript, axes: tuple[str, ...], shape_params: set[str],
                    params: dict[str, Param], is_write: bool, filename: str) -> AccessMap:
    operand, exprs = _full_subscript(sub, axes, shape_params, params, filename)
    return _build_access_map(operand, list(exprs), axes, is_write)


def _build_access_map(operand: str, exprs: list[Expr], axes: tuple[str, ...],
                      is_write: bool) -> AccessMap:
    coeff_maps = [dict(e.coeffs) for e in exprs]
    matrix = tuple(tuple(cm.get(a, 0) for a in axes) for cm in coeff_maps)
    axis_set = set(axes)
    offsets = tuple(Expr({n: c for n, c in e.coeffs if n not in axis_set}, e.const)
                   for e in exprs)
    return AccessMap(operand=operand, matrix=matrix, offsets=offsets, is_write=is_write)


# ------------------------------------------------------------------------------------------
# §3.5 -- statement classification, with scalar forward substitution
# ------------------------------------------------------------------------------------------


class _Ctx:
    """Threads the read-only context through the recursive-descent value/statement walk."""

    def __init__(self, axes: tuple[str, ...], shape_params: set[str],
                params: dict[str, Param], globals_: dict[str, Any], filename: str) -> None:
        self.axes = axes
        self.axis_set = set(axes)
        self.shape_params = shape_params
        self.params = params
        self.globals = globals_
        self.filename = filename
        self.scalars: dict[str, ast.expr] = {}   # name -> its (unsubstituted) bound AST expr


def _subst(node: ast.expr, ctx: "_Ctx", seen: frozenset[str] = frozenset()) -> ast.expr:
    """Forward-substitute every scalar Name in `node` with its recorded binding (§3.5)."""
    if isinstance(node, ast.Name) and node.id in ctx.scalars:
        if node.id in seen:
            _raise("GRAMMAR-UNSUPPORTED-STMT",
                  f"scalar binding of '{node.id}' forms a cycle", "remove the circular binding",
                  node, ctx.filename, cycle=sorted(seen | {node.id}))
        return _subst(ctx.scalars[node.id], ctx, seen | {node.id})
    return node


def _value_expr(node: ast.expr, ctx: "_Ctx") -> ExprNode:
    """Build the frozen `ExprNode` tree for a value-level expression (§3.5/§3.6)."""
    node = _subst(node, ctx)
    if isinstance(node, ast.Constant):
        if type(node.value) is int:
            return Const(value=node.value, text=repr(node.value), dtype=Dtype.i32)
        if type(node.value) is float:
            return Const(value=node.value, text=repr(node.value), dtype=Dtype.f32)
        _raise("GRAMMAR-UNSUPPORTED-EXPR", f"unsupported literal {node.value!r}",
              "use an int or float literal", node, ctx.filename)
    if isinstance(node, ast.Subscript):
        operand, exprs = _full_subscript(node, ctx.axes, ctx.shape_params, ctx.params,
                                         ctx.filename)
        return Load(buffer_id=operand, subscripts=exprs)
    if isinstance(node, ast.Name):
        val = ctx.globals.get(node.id)
        if type(val) is int and type(val) is not bool:
            return Const(value=val, text=str(val), dtype=Dtype.i32)
        _raise("GRAMMAR-UNSUPPORTED-EXPR", f"name '{node.id}' does not resolve to a usable value",
              "use a kernel parameter subscript, or a module-level int constant", node,
              ctx.filename, name=node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return Neg(operand=_value_expr(node.operand, ctx))
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}[type(node.op)]
        return BinOp(op=op, lhs=_value_expr(node.left, ctx), rhs=_value_expr(node.right, ctx))
    if isinstance(node, ast.IfExp):
        cmp = node.test
        if not isinstance(cmp, ast.Compare) or len(cmp.ops) != 1 or len(cmp.comparators) != 1:
            _raise("GRAMMAR-UNSUPPORTED-EXPR",
                  "the condition of an if-expression must be a single comparison",
                  "use a single comparison, e.g. a == b", cmp, ctx.filename)
        cmp_op = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
                 ast.Gt: ">", ast.GtE: ">="}.get(type(cmp.ops[0]))
        if cmp_op is None:
            _raise("GRAMMAR-UNSUPPORTED-EXPR", "unsupported comparison operator",
                  "use ==, !=, <, <=, >, or >=", cmp, ctx.filename)
        return Select(cmp_op=cmp_op, lhs=_value_expr(cmp.left, ctx),
                     rhs=_value_expr(cmp.comparators[0], ctx),
                     then=_value_expr(node.body, ctx), otherwise=_value_expr(node.orelse, ctx))
    if isinstance(node, ast.Compare):
        _raise("GRAMMAR-UNSUPPORTED-EXPR", "a comparison is only accepted inside 'a if c else b'",
              "move the comparison into an if-expression", node, ctx.filename)
    if isinstance(node, ast.Call):
        fname = node.func.id if isinstance(node.func, ast.Name) else None
        if fname not in ("max", "min"):
            _raise("GRAMMAR-UNSUPPORTED-CALL",
                  f"call to '{fname or _render_annotation(node.func)}' is not allowed; only "
                  f"max and min", "rewrite with max/min, or precompute into an input array",
                  node, ctx.filename, callee=fname)
        if len(node.args) < 2:
            _raise("GRAMMAR-UNSUPPORTED-CALL", f"{fname}(...) needs 2 or more arguments",
                  f"pass 2 or more arguments to {fname}", node, ctx.filename)
        return MaxMin(op="maximum" if fname == "max" else "minimum",
                     operands=tuple(_value_expr(a, ctx) for a in node.args))
    _raise("GRAMMAR-UNSUPPORTED-EXPR", f"'{type(node).__name__}' is not in the accepted "
          "kernel subset", "the subset has for/range, x[...] = e, x[...] += e and max/min only; "
          "move the condition into the data or precompute it", node, ctx.filename,
          construct=type(node).__name__)


def _classify(body: list[ast.stmt], axes: tuple[str, ...], shape_params: set[str],
             params: dict[str, Param], globals_: dict[str, Any],
             filename: str) -> tuple[Statement, ...]:
    ctx = _Ctx(axes, shape_params, params, globals_, filename)
    statements: list[Statement] = []

    def enclosing_axes(depth: int) -> tuple[str, ...]:
        return axes[:depth]

    def walk(stmts: list[ast.stmt], depth: int) -> None:
        for s in stmts:
            if isinstance(s, ast.For):
                walk(s.body, depth + 1)
            elif isinstance(s, ast.AugAssign):
                if not isinstance(s.op, ast.Add) or not isinstance(s.target, ast.Subscript):
                    _raise("GRAMMAR-UNSUPPORTED-STMT",
                          f"'{type(s.op).__name__}=' is not in the accepted kernel subset",
                          "only += is accepted as an accumulation", s, filename,
                          construct=type(s.op).__name__)
                target = _extract_access(s.target, axes, shape_params, params, True, filename)
                _, target_exprs = _full_subscript(s.target, axes, shape_params, params, filename)
                value = _value_expr(s.value, ctx)
                expr = BinOp(op="+",
                            lhs=Load(buffer_id=target.operand, subscripts=target_exprs),
                            rhs=value)
                statements.append(Statement(kind="accumulate", target=target, reads=(),
                                            expr=expr, op="+", axes=enclosing_axes(depth),
                                            line=s.lineno))
            elif isinstance(s, ast.Assign):
                if len(s.targets) != 1:
                    _raise("GRAMMAR-UNSUPPORTED-STMT", "only single-target assignment is "
                          "accepted", "assign one name or subscript at a time", s, filename)
                tgt = s.targets[0]
                if isinstance(tgt, ast.Name):
                    ctx.scalars[tgt.id] = s.value
                    continue
                if not isinstance(tgt, ast.Subscript):
                    _raise("GRAMMAR-UNSUPPORTED-STMT",
                          f"assignment target must be a name or a subscript, got "
                          f"{type(tgt).__name__}", "assign to x[...] or bind a scalar name",
                          s, filename, construct=type(tgt).__name__)
                target = _extract_access(tgt, axes, shape_params, params, True, filename)
                _, target_exprs = _full_subscript(tgt, axes, shape_params, params, filename)
                is_acc, op = _is_max_min_accumulate(s.value, target, ctx)
                if is_acc:
                    call = s.value
                    assert isinstance(call, ast.Call)
                    rest = [_value_expr(a, ctx) for a in call.args[1:]]
                    expr = MaxMin(op="maximum" if op == "max" else "minimum",
                                 operands=(Load(buffer_id=target.operand,
                                              subscripts=target_exprs), *rest))
                    statements.append(Statement(kind="accumulate", target=target, reads=(),
                                                expr=expr, op=op, axes=enclosing_axes(depth),
                                                line=s.lineno))
                else:
                    expr = _value_expr(s.value, ctx)
                    statements.append(Statement(kind="assign", target=target, reads=(),
                                                expr=expr, op=None, axes=enclosing_axes(depth),
                                                line=s.lineno))
            else:
                _raise("GRAMMAR-UNSUPPORTED-STMT",
                      f"'{type(s).__name__}' is not in the accepted kernel subset",
                      "the subset has for/range, x[...] = e, x[...] += e and max/min only; "
                      "move the condition into the data or precompute it", s, filename,
                      construct=type(s).__name__)

    walk(body, 0)
    if not statements:
        return ()
    out = []
    for st in statements:
        loads = _iter_loads(st.expr)
        target_matrix = st.target.matrix
        target_offsets = st.target.offsets
        skip_one_self_read = st.kind == "accumulate"
        reads = []
        for ld in loads:
            cand = _load_to_access(ld, axes)
            if (skip_one_self_read and cand.operand == st.target.operand
                    and cand.matrix == target_matrix and cand.offsets == target_offsets):
                skip_one_self_read = False  # the target's own re-read in the desugared expr
                continue
            reads.append(cand)
        out.append(Statement(kind=st.kind, target=st.target, reads=tuple(reads), expr=st.expr,
                             op=st.op, axes=st.axes, line=st.line))
    return tuple(out)


def _is_max_min_accumulate(value: ast.expr, target: AccessMap, ctx: "_Ctx") -> tuple[bool, str]:
    if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
        return False, ""
    fname = value.func.id
    if fname not in ("max", "min") or not value.args:
        return False, ""
    first = value.args[0]
    if not isinstance(first, ast.Subscript):
        return False, ""
    try:
        cand = _extract_access(first, ctx.axes, ctx.shape_params, ctx.params, False,
                               ctx.filename)
    except GrammarError:
        return False, ""
    return (cand.operand == target.operand and cand.matrix == target.matrix), fname


def _iter_loads(node: ExprNode) -> list[Load]:
    if isinstance(node, Load):
        return [node]
    out: list[Load] = []
    if isinstance(node, BinOp):
        out += _iter_loads(node.lhs); out += _iter_loads(node.rhs)
    elif isinstance(node, Neg):
        out += _iter_loads(node.operand)
    elif isinstance(node, MaxMin):
        for o in node.operands:
            out += _iter_loads(o)
    elif isinstance(node, Select):
        out += _iter_loads(node.lhs); out += _iter_loads(node.rhs)
        out += _iter_loads(node.then); out += _iter_loads(node.otherwise)
    return out


def _load_to_access(ld: Load, axes: tuple[str, ...]) -> AccessMap:
    exprs = list(ld.subscripts)
    return _build_access_map(ld.buffer_id, exprs, axes, False)


# ------------------------------------------------------------------------------------------
# §3.7 -- uniform dependence vectors
# ------------------------------------------------------------------------------------------


def _dependences(statements: tuple[Statement, ...]) -> tuple[Dependence, ...]:
    out: dict[tuple[str, tuple[int, ...]], Dependence] = {}
    written: dict[str, AccessMap] = {}
    for st in statements:
        if st.target.is_write:
            written[st.target.operand] = st.target
    for st in statements:
        reads_group = list(st.reads)
        if st.kind == "accumulate":
            reads_group.append(st.target)
        for r in reads_group:
            if r.operand not in written:
                continue
            w = written[r.operand]
            if r.matrix != w.matrix:
                raise GrammarError(Diagnostic(
                    code="GRAMMAR-NONUNIFORM-DEP", stage="grammar", clause=None,
                    reason=f"{r.operand} is read at a different linear access than it is "
                           f"written; the dependence distance is not constant",
                    fix="make every access to a written array a constant shift of the write",
                    location=None,
                    details={"operand": r.operand, "write_matrix": [list(row) for row in
                             w.matrix], "read_matrix": [list(row) for row in r.matrix]}))
            if any(o.coeffs for o in w.offsets) or any(o.coeffs for o in r.offsets):
                raise GrammarError(Diagnostic(
                    code="GRAMMAR-NONUNIFORM-DEP", stage="grammar", clause=None,
                    reason=f"{r.operand}'s offset has a surviving shape-parameter "
                           f"coefficient; the dependence distance is not constant",
                    fix="make every access to a written array a constant shift of the write",
                    location=None, details={"operand": r.operand}))
            c_w = tuple(o.const for o in w.offsets)
            c_r = tuple(o.const for o in r.offsets)
            rhs = tuple(cw - cr for cw, cr in zip(c_w, c_r))
            d = _solve_dependence(w.matrix, rhs)
            if d is not None and any(d):
                key = (r.operand, d)
                out[key] = Dependence(vector=d, kind="RAW", operand=r.operand)
    return tuple(sorted(out.values(), key=lambda dd: (dd.operand, dd.vector)))


def _solve_dependence(matrix: tuple[tuple[int, ...], ...],
                      rhs: tuple[int, ...]) -> tuple[int, ...] | None:
    """§3.7 line 12-17: full column rank -> the unique solution; else the lex-smallest
    lexicographically-positive element of d0 + ker(M), walking the kernel basis over
    {-1, 0, 1} (small dimension in practice: <= 3 tiled/dependent axes)."""
    n = len(matrix[0]) if matrix else 0
    if intlin.rank(matrix) == n:
        return intlin.solve(matrix, rhs)
    d0 = intlin.solve(matrix, rhs)
    if d0 is None:
        return None
    ker = intlin.kernel_basis(matrix)
    if not ker:
        return d0 if any(d0) else None
    candidates = [d0]
    import itertools
    for coeffs in itertools.product((-1, 0, 1), repeat=len(ker)):
        if not any(coeffs):
            continue
        v = list(d0)
        for c, basis in zip(coeffs, ker):
            for i, x in enumerate(basis):
                v[i] += c * x
        candidates.append(tuple(v))
    positive = [c for c in candidates if intlin.lexpos(c)]
    if not positive:
        return None
    return min(positive, key=lambda v: (tuple(abs(x) for x in v), v))


# ------------------------------------------------------------------------------------------
# §3.8 -- reduction-function derivation
# ------------------------------------------------------------------------------------------


def _reduction(statements: tuple[Statement, ...]) -> ReductionSpec | None:
    acc = [s for s in statements if s.kind == "accumulate"]
    if not acc:
        return None
    target = acc[0].target.operand
    sf = acc[0].target.matrix
    r = intlin.kernel_basis(sf)
    # `op` stays None here: it is the A/C tag M2's reduce(ax, op=...) fills in, and M3 checks
    # (03-lld-M1-frontend.md §3.8 -- the schedule has not been built yet at M1 time).
    return ReductionSpec(target=target, projection=sf, space=r, op=None)


# ------------------------------------------------------------------------------------------
# Top level
# ------------------------------------------------------------------------------------------


def _fdef_of(tree: ast.Module) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node
    raise GrammarError(Diagnostic(
        code="GRAMMAR-UNSUPPORTED-STMT", stage="grammar", clause=None,
        reason="no function definition found in the decorated source",
        fix="decorate a single def", location=None, details={}))


def capture(fn: Callable) -> KernelModel:
    """Walk `fn`'s AST into a `KernelModel`. Raises `GrammarError`."""
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except OSError as exc:
        raise GrammarError(Diagnostic(
            code="GRAMMAR-UNSUPPORTED-STMT", stage="grammar", clause=None,
            reason="the source of the kernel could not be read; kernels must live in a file",
            fix="define the kernel in a .py file, not interactively", location=None,
            details={"error": str(exc)}))
    code = getattr(fn, "__code__", None)
    filename = code.co_filename if code is not None else "<kernel>"
    tree = ast.parse(src)
    fdef = _fdef_of(tree)
    globals_ = fn.__globals__

    params, shape_param_names = _parse_params(fdef.args, filename, globals_)
    shape_params = sorted(shape_param_names)

    axes = _extract_domain(fdef.body, set(shape_params), globals_, filename)
    axis_names = tuple(a.name for a in axes)

    param_by_name = {p.name: p for p in params}
    statements = _classify(fdef.body, axis_names, set(shape_params), param_by_name, globals_,
                           filename)
    if not statements:
        raise GrammarError(Diagnostic(
            code="GRAMMAR-UNSUPPORTED-STMT", stage="grammar", clause=None,
            reason="a kernel must have at least one store or accumulate statement",
            fix="write x[...] = e or x[...] += e in the innermost loop body", location=None,
            details={}))

    written = {st.target.operand for st in statements}
    params = _mark_written(params, written)

    deps = _dependences(statements)
    red = _reduction(statements)

    bindings = tuple((n, v if type(v := globals_.get(n)) is int else 0) for n in shape_params)

    return KernelModel(
        name=fdef.name,
        source=src,
        params=params,
        shape_params=tuple(shape_params),
        bindings=bindings,
        axes=axes,
        statements=statements,
        dependences=deps,
        reduction=red,
    )


class Kernel:
    """The decorated kernel: callable as the oracle, plus its parsed `KernelModel`."""

    def __init__(self, fn: Callable, source: str, model: KernelModel) -> None:
        self.fn = fn
        self.source = source
        self.model = model

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the original, unmodified function object. Raises nothing of ours."""
        return self.fn(*args, **kwargs)


def kernel(fn: Callable) -> Kernel:
    """The `sp.kernel` decorator (design/06-interfaces.md §7.1). Raises `GrammarError`."""
    model = capture(fn)
    return Kernel(fn=fn, source=model.source, model=model)


__all__ = ["capture", "kernel", "Kernel"]
