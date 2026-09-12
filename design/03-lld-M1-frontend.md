# LLD M1 — Frontend / kernel capture

*Phase 1 low-level design, 2026-09-12. Owner: **Person A**. Reads on top of
[`01-requirements.md`](01-requirements.md) §3.1, [`02-hld.md`](02-hld.md) §2 (M1) and §7, and
[`06-interfaces.md`](06-interfaces.md) §2 and §7 — which is **frozen** and is the single source
of truth for every field and signature named here. This document never restates a field list;
it cites the section.*

*Citation convention as in `01-requirements.md` §0: `VF §X` = `../hackathon/VERIFIED-AIR-FACTS.md`,
`SD-0n §X` = `../spatial-dsl/0n-*.md`, a bare `path:line` = the mlir-air clone at `ff95a9b`.*

---

## 1. Purpose and the FR IDs it satisfies

M1 turns the *source text* of a `@sp.kernel`-decorated function into a `KernelModel`
(`06-interfaces.md` §2.7) and hands back a callable that **is** the CPython oracle. It owns the
accepted Python subset and every rejection message about it. It knows nothing of PEs, memory
levels, channels or AIR, and never imports `air`.

| FR | What M1 does for it |
|---|---|
| **FR-S1** | the decorator; the callable passthrough that makes the decorated function its own oracle |
| **FR-S2** | parameter annotations, inert at run time; `Param.dtype`/`.shape` |
| **FR-S3** | the accepted grammar (§3.1), enforced on the parsed AST |
| **FR-S4** | the three-part `GrammarError` message |
| **FR-S18** | ignorability — M1's half: nothing is mutated at decoration time (§3.2) |
| **FR-D1** | every M1 rejection is a `GrammarError` carrying a full `Diagnostic` |
| **NFR-5, NFR-7** | docstrings on the public surface; no exception leaves M1 that is not a `SpatialError` |

M1 also *produces the inputs* to FR-L2 (`dependences`) and FR-L4 (`reduction`), but does not
satisfy those FRs — M3 does.

---

## 2. Public entry points

| Name | Contract (one line) | Signature |
|---|---|---|
| `sp.kernel(fn)` | parse and validate `fn`'s source, return a `Kernel`; raise `GrammarError` at **decoration** time | `06-interfaces.md` §7.1 |
| `Kernel.__call__(*args)` | call the **original, unmodified** function object; raise nothing of ours | §7.1 |
| `Kernel.model` | the `KernelModel` built at decoration time; the same object every time | §7.1 |
| `m1.capture(fn)` | the module entry point `sp.kernel` is a thin wrapper over | §7.2 |

`Kernel` additionally exposes `Kernel.source` (the dedented source text) because `KernelModel`
carries it (§2.7) and the diagnostics render source lines from it.

---

## 3. Internal design

### 3.1 The accepted subset, as a grammar

This is FR-S3 / brief D9 written as EBNF. Terminals in `"…"`; `NAME`, `INT`, `NUMBER` are the
Python tokens. Everything not derivable here is rejected, with the construct named.

```ebnf
kernel        ::= decorator NEWLINE "def" NAME "(" params ")" ":" suite
decorator     ::= "@" ("sp" ".")? "kernel"
params        ::= param ("," param)*
param         ::= NAME ":" "sp" "." dtype "[" shape_entry ("," shape_entry)* "]"
dtype         ::= "f32" | "f16" | "bf16" | "i32" | "i8"
shape_entry   ::= INT | NAME                       (* NAME = a shape parameter *)

suite         ::= stmt+
stmt          ::= for_stmt | store | accumulate | scalar_bind
for_stmt      ::= "for" NAME "in" "range" "(" affine [ "," affine [ "," INT ] ] ")" ":" suite
store         ::= subscript "=" expr
accumulate    ::= subscript "+=" expr
                | subscript "=" ("max"|"min") "(" subscript "," expr ("," expr)* ")"
                                                   (* the first argument is the same subscript *)
scalar_bind   ::= NAME "=" expr

subscript     ::= NAME "[" affine ("," affine)* "]"
expr          ::= select | sum
sum           ::= term (("+" | "-" | "*" | "/") term)*
select        ::= sum "if" cmp "else" sum          (* a value, not control flow *)
cmp           ::= sum ("==" | "!=" | "<" | "<=" | ">" | ">=") sum
term          ::= "-" term | "(" expr ")" | call | subscript | NAME | NUMBER
call          ::= ("max" | "min") "(" expr ("," expr)+ ")"

affine        ::= aterm (("+" | "-") aterm)*
aterm         ::= [ INT "*" ] avar | avar [ "*" INT ] | INT
avar          ::= NAME                             (* an enclosing loop variable or a shape param *)
```

Three restrictions the EBNF cannot carry, enforced by the checker and each with its own message:

* **R-a (rectangularity).** A `for` bound (`affine` in `for_stmt`) may name **shape parameters
  and integer constants only** — never an enclosing loop variable. FR-S3's item 1 permits
  enclosing loop variables in a bound *and* then demands "rectangular after substitution of
  parameters"; the two halves conflict, and rectangularity wins because every downstream check
  (tile divisibility FR-S7, extent-vs-grid FR-L8(c), the vertex evaluation of §3.6 here and of
  M3) evaluates affine forms over a **box**. Rejected as `GRAMMAR-BAD-RANGE`.
* **R-b (integer coefficients).** In `aterm`, the multiplying factor is an **integer literal**.
  `i * S` with `S` a shape parameter is rejected: `AccessMap.matrix` is `tuple[tuple[int,…],…]`
  (§2.3), so a symbolic coefficient has nowhere to live. Rejected as
  `GRAMMAR-NONAFFINE-SUBSCRIPT`. (None of W1/W2/W3 needs it: all tiling is done by the
  *schedule*, so every kernel subscript is `v`, `v ± INT` or `INT`.)
* **R-d (`select` is an expression).** `a if cmp else b` (Python `IfExp`) is accepted **only**
  in the `select` production: both arms and both comparison operands must be `sum`s, and the
  result is one `ExprNode.Select` (`06-interfaces.md` §5.5), lowering to `arith.select`. Both
  arms are evaluated; nothing branches; no PE diverges. A `Compare` anywhere else, and any `if`
  **statement**, are still `GRAMMAR-UNSUPPORTED-EXPR` / `GRAMMAR-UNSUPPORTED-STMT`. This is
  FR-S3 item 8, added by REVIEW-round1 RULING 2 so that W3's
  `sub = MATCH if q[i-1] == r[j-1] else MISMATCH` is inside the subset.
* **R-e (module-level integer constants).** A bare `NAME` in an expression that is neither an
  axis, a shape parameter nor a kernel parameter is resolved from `fn.__globals__` when it is
  bound to a Python `int` (not a `bool`), and becomes a literal of that value; otherwise it is
  `GRAMMAR-UNSUPPORTED-EXPR` naming the name. This is the same read as §3.4 line 18 and the same
  determinism argument applies: the binding is recorded in the captured `source`, the read is
  idempotent, and no other global is touched. W3's `MATCH`, `MISMATCH` and `GAP` reach the model
  this way (FR-S3 item 7).
* **R-c (uniform linear parts).** Every access to a *written* operand inside one statement must
  share the same `matrix` (§2.3). `S[i,j] = S[j,i] + 1` is affine but has no uniform dependence
  distance; see §5 for the code this raises and §10 for the interface proposal it forces.

Explicitly rejected, each naming its Python grammar name: `If`, `While`, `Break`, `Continue`,
`Return`, `With`, `Try`, `Global`, `Nonlocal`, `Assert`, tuple/starred assignment, `ListComp`,
`SetComp`, `DictComp`, `GeneratorExp`, `Lambda`, `Attribute`, `BoolOp`,
`Pow` (`**`), `Slice`, a `Compare` outside the `select` production, an `IfExp` whose arms or
comparison are not `sum`s, any `Call` whose function is not the bare name `max` or `min`, any
subscript whose index is itself a subscript (`A[B[i]]`), and `for` over anything but `range`.

### 3.2 What `@sp.kernel` does — and does not — do (FR-S1, FR-S2, FR-S18)

This is the ignorability contract. It is short on purpose.

```pseudo
CAPTURE(fn):
 1  src   := textwrap.dedent(inspect.getsource(fn))
 2  tree  := ast.parse(src)                       # ast.Module with one FunctionDef
 3  fdef  := the single FunctionDef; strip its decorator_list from consideration
 4  model := BUILD_MODEL(fdef, src, fn.__globals__)      # §3.3 … §3.8; raises GrammarError
 5  k     := Kernel(fn=fn, source=src, model=model)
 6  return k

Kernel.__call__(*args, **kwargs):
 7  return self.fn(*args, **kwargs)               # the original object, nothing else
```

**Does at decoration time**: read the source with `inspect.getsource`; `ast.parse` it; walk the
tree; read *integer* bindings of loop-bound names out of `fn.__globals__` (line 4, and only
those — see §3.4); construct frozen dataclasses.

**Does not, ever**: `exec` or `compile` the source; rewrite, wrap or copy `fn`; write to
`fn.__globals__`, `fn.__dict__` or any module attribute; call `typing.get_type_hints` (the
annotation objects are read from the **AST**, never evaluated by us — Python itself evaluates
them at `def` time, which is why FR-S2 requires `sp.f32[…]` to return an inert descriptor);
import `numpy`, `air`, or anything outside `ast`/`inspect`/`textwrap`/`dataclasses`; convert,
validate or touch the call arguments (line 7 is the whole of `__call__`).

Consequence, which is FR-S18: deleting every `sp.schedule` line from a program changes nothing
about what `Kernel.__call__` does, because `Kernel.__call__` never consulted a schedule. The
oracle is `fn` itself, byte for byte.

### 3.3 Order of the build

`BUILD_MODEL` runs the passes below in order; the first failure raises. The order matters only
in that it puts the cheap, most-legible rejections first.

```pseudo
BUILD_MODEL(fdef, src, globals):
 1  params      := PARSE_PARAMS(fdef.args)                     # §3.4
 2  shape_params:= sorted(names appearing as shape_entry NAMEs)
 3  axes        := EXTRACT_DOMAIN(fdef.body, shape_params, globals)   # §3.4
 4  CHECK_GRAMMAR(fdef.body, axes, params)                     # §3.1, one walk, every reject
 5  stmts       := CLASSIFY(fdef.body, axes, params)           # §3.5
 6  accesses    := for each stmt, EXTRACT_ACCESS(sub, axes, shape_params)  # §3.6
 7  deps        := DEPENDENCES(stmts)                          # §3.7
 8  red         := REDUCTION(stmts)                            # §3.8
 9  return KernelModel(...)                                    # §2.7
```

Complexity: one bounded-depth walk per pass over an AST of a few dozen nodes. Nothing here is
measurable; no pass is worth optimising.

### 3.4 Loop-domain extraction

```pseudo
EXTRACT_DOMAIN(body, shape_params, globals):
 1  axes := []
 2  walk the body depth-first, outermost first; for each For node F at depth d:
 3      require F.target is a plain Name, F.iter is Call to the bare name `range`
 4          else raise GRAMMAR-BAD-RANGE naming the actual node type
 5      require 1 <= len(F.iter.args) <= 3
 6      (lo, hi, step) := (0, a0, 1) | (a0, a1, 1) | (a0, a1, a2)
 7      for each of lo, hi: e := AFFINE(a, allowed = shape_params)     # NOT enclosing axes: R-a
 8          if AFFINE reports a name that is an enclosing axis:
 9              raise GRAMMAR-BAD-RANGE, reason "bound names the enclosing loop variable <v>,
10                  so the domain is not rectangular"
11      require step is an integer literal >= 1 else GRAMMAR-BAD-RANGE
12      extent := RESOLVE(hi) - RESOLVE(lo), rounded up by step, or None if unresolvable
13      require the loop variable name is not already an axis name else GRAMMAR-BAD-RANGE
14      axes.append(Axis(name=F.target.id, lo, hi, step, extent, parent=None, depth=d))
15  require len(axes) >= 1
16  return tuple(axes)                                          # source order, outermost first

RESOLVE(e):                                    # e is an Expr: coeffs over shape params + const
17  if e.coeffs is empty: return e.const
18  if every name in e.coeffs is bound in `globals` to a Python int (not bool):
19      return sum(c * globals[n] for n, c in e.coeffs) + e.const
20  return None
```

Line 18 is the one place M1 reads a *value* rather than the source. It refines HLD §2's
"`KernelModel` is a pure function of the source text" to: **a pure function of the source text
and of the integer bindings of the shape-parameter names that source declares.** Both inputs are
recorded (`source` in §2.7, the resolved `extent` in §2.2), the read is idempotent, and no other
global is ever touched — so determinism (NFR-1) is preserved. The refinement is necessary:
`tile` must check divisibility against a concrete extent (FR-S7) and `place` against a concrete
grid extent (FR-L8(c)), and the shipped kernels write `for i in range(M)` with `M` a module-level
`int`. An axis whose extent does not resolve keeps `extent = None`, and FR-S7's reading adopted
here is that **such an axis cannot be tiled or placed** — M2 rejects it at clause time.

### 3.5 Statement classification

```pseudo
CLASSIFY(body, axes, params):
 1  for each leaf statement s in body, in source order:
 2      if s is AugAssign with op Add and target a subscript:
 3          kind := "accumulate"; op := "+"
 4      elif s is Assign, target a subscript T, value a Call to max/min
 5           whose FIRST argument is a subscript structurally equal to T:
 6          kind := "accumulate"; op := "max" | "min"
 7      elif s is Assign and target is a subscript:
 8          kind := "assign"; op := None
 9      elif s is Assign and target is a plain Name:
10          record a scalar binding; see SCALARS below; emit no Statement yet
11      else: raise GRAMMAR-UNSUPPORTED-STMT naming type(s).__name__
12      target := EXTRACT_ACCESS(s.target, is_write=True)
13      reads  := EXTRACT_ACCESS(x, is_write=False) for each subscript x in s.value, source order
14      emit Statement(kind, target, reads, op, axes=enclosing axis names, line=s.lineno)

SCALARS:
15  a scalar name bound and read inside the SAME loop body is forward-substituted into the
15a     statement that reads it, then discarded; it never reaches the model.
15b     Substitution is TRANSITIVE and is applied in BINDING ORDER: a bind may read names
15c     bound earlier in the same body, and each is substituted before the next.  W3 binds
15d     `sub` from `q`, `r`, MATCH and MISMATCH, then reads `sub` in the store.  A cycle
15e     (a bind that reads, directly or transitively, a name bound later or itself) raises
15f     GRAMMAR-UNSUPPORTED-STMT naming the cycle's names and line numbers.
17  the triple  [ acc = <literal> ; for … : acc <op>= e ; X[…] = acc ]  where the three sit in
18      one body and the middle is a contiguous group of For nodes, is folded into ONE
19      Statement(kind="accumulate", op=<op>, target=X's access, reads=e's accesses) whose
20      `axes` are the enclosing axes plus the folded loop group. (This is the conv2d shape of
21      SD-03 §2; none of W1/W2/W3 uses it, and it is the only imperfect nesting M1 accepts.)
22  any other scalar use — read outside the body that binds it, rebound in two bodies, read
23      before binding — raises GRAMMAR-UNSUPPORTED-STMT naming the scalar and both line numbers.
```

The `+=` form and the `max`/`min` form are the same thing (FR-S3 items 2 and 5; `06-interfaces.md`
§2.4's `kind` domain), which is why both land on `"accumulate"`. `op` is what SD-02 §4 calls the
A/C tag; M3 checks the tag against `R_space` (FR-L5).

### 3.6 Affine access-map extraction

An index expression becomes an `Expr` (`06-interfaces.md` §1): integer coefficients over loop
variables and shape parameters, plus a constant.

```pseudo
AFFINE(node, axes, shape_params) -> Expr:
 1  Constant int          -> ({}, value)
 2  Name n:
 3      if n in axes or n in shape_params: return ({n: 1}, 0)
 3a     # NOTE: AFFINE runs on SUBSCRIPTS only.  A module-level int constant in a subscript is
 3b     # still a shape parameter or a literal; R-e's fn.__globals__ lookup belongs to the
 3c     # EXPRESSION walk (VALUE_EXPR, §3.5), not here.
 4      raise GRAMMAR-NONAFFINE-SUBSCRIPT, reason "unknown name <n>"
 5  UnaryOp USub x        -> negate(AFFINE(x))
 6  BinOp Add/Sub l, r    -> add/sub of AFFINE(l), AFFINE(r)
 7  BinOp Mult l, r:
 8      a := AFFINE(l); b := AFFINE(r)
 9      if a.coeffs empty: return scale(b, a.const)            # INT * var
10      if b.coeffs empty: return scale(a, b.const)            # var * INT
11      raise GRAMMAR-NONAFFINE-SUBSCRIPT, reason "product of two non-constant terms
12          <render(a)> * <render(b)>"                          # covers i*j and i*S (R-b)
13  Subscript / Call / Pow / Compare / anything else:
14      raise GRAMMAR-NONAFFINE-SUBSCRIPT naming type(node).__name__
15      (Subscript here is exactly the data-dependent A[B[i]] case; the message says so)

EXTRACT_ACCESS(sub, axes, shape_params, is_write) -> AccessMap:
16  operand := sub.value.id; require it is a Param name else GRAMMAR-NONAFFINE-SUBSCRIPT
17  idx := the index tuple; require len(idx) == rank of that Param else GRAMMAR-BAD-ANNOTATION
18  for each dimension d: e_d := AFFINE(idx[d], axes, shape_params)
19  matrix[d][a] := e_d.coeffs.get(axes[a].name, 0)             # integer part, over the axes
20  offsets[d]   := e_d with the axis coefficients removed      # shape params + const survive
21  return AccessMap(operand, matrix, offsets, is_write)
```

Line 19/20 is the split the contract asks for: `matrix` carries the loop-variable coefficients
as integers, `offsets` carries everything parametric (`06-interfaces.md` §2.3). `U[t+1, i, j]`
gives `matrix = ((1,0,0),(0,1,0),(0,0,1))` and `offsets = ((·,1), (·,0), (·,0))`.

### 3.7 Uniform dependence vectors

The scope is exactly what W1/W2/W3 need, and the limits are stated rather than hidden.

```pseudo
DEPENDENCES(stmts) -> tuple[Dependence, ...]:
 1  out := set()
 2  for each statement s and each operand a written by s:
 3      W := s.target (the write access on a)
 4      for each read access R on the same operand a, in s and in every statement of the
 5          same nest (the three kernels have one statement each; the loop is over all of them):
 6          if R.matrix != W.matrix:
 7              raise NON-UNIFORM (see §5 and §10) naming a, W.matrix, R.matrix
 8          # iteration x writes M·x + c_W ; iteration y reads M·y + c_R ; RAW when equal
 9          rhs := c_W - c_R                      # constant part only; parametric parts must cancel
10          if rhs has any surviving shape-parameter coefficient:
11              raise NON-UNIFORM naming the parametric offset
12          solve  M · d = rhs   over the integers, by the §3.9 routine:
13              if M has full column rank: d is unique; require it integral
14              else: the solution set is d0 + ker M. Take d := the LEX-SMALLEST
15                  lexicographically-positive element of { d0 + z : z in a canonical basis
16                  walk of ker M over {-1,0,1} }, which for rank-deficient M with rhs = 0
17                  is the first canonical kernel basis vector, made lex-positive.
18          if d != 0: out.add(Dependence(vector=d, kind="RAW", operand=a))
19  also emit WAW between two writes to the same operand with the same matrix (none of the
20      three kernels has one) and WAR symmetrically; both use the same solve.
21  return tuple(sorted(out, key=(operand, vector)))             # §2.7 requires this order
```

**Worked, so line 14–17 is not mysterious.** W1: `C[i,j] += …` has `M_C = ((1,0,0),(0,1,0))`,
`c_W = c_R = 0`, so `M·d = 0` and `ker M_C = span{e_k}`; the lex-smallest positive element is
`d = (0,0,1)`. W2: `M_U = I₃`, `c_W = (1,0,0)`, and the read at `U[t,i-1,j]` has
`c_R = (0,-1,0)`, so `d = (1,1,0)` directly. W3: `M_S = I₂`, `c_W = 0`, reads at
`(-1,-1), (-1,0), (0,-1)` give `d = (1,1), (1,0), (0,1)`.

**Stated limits.** M1 computes *uniform* dependences only: one constant vector per (write,
read) pair on one operand, with identical linear parts and a constant right-hand side. It does
**not** do: non-uniform or parametric distances (line 7, 11); dependences through a scalar
temporary (those are forward-substituted away in §3.5); dependences between different operands
(there are none — a dependence is same-array by definition); loop-carried dependences that exist
only for part of the domain. This is enough for W1 (one vector), W2 (five) and W3 (three), and
anything else is rejected legibly rather than analysed wrongly. That is the whole of the
contract: over-approximating a dependence set would make M3's causality check unsound in the
direction that matters.

### 3.8 Reduction-function derivation

```pseudo
REDUCTION(stmts) -> ReductionSpec | None:
 1  acc := the statements with kind == "accumulate"
 2  if acc is empty: return None
 3  require all of acc write the same operand and carry the same op
 4      else GRAMMAR-UNSUPPORTED-STMT ("two different accumulations in one kernel")
 5  Sf := acc[0].target.matrix        # the projection's linear part IS the target access map
 6  R  := KERNEL_BASIS(Sf)            # §3.9, canonical row-echelon, primitive, sign-normalised
 7  return ReductionSpec(target=..., projection=Sf, space=R, op=acc[0].op)
```

Line 5 is the content: `C[i,j] += A[i,k]*B[k,j]` *is* `C[c] = REDUCE(+, f, E)` with
`f : (i,j,k) ↦ (i,j)` (SD-02 §4), and the linear part of `f` is exactly the linear part of the
target's access map. So `R = ker Sf = span{e_k}` for W1 falls out with no dependence analysis —
which is SD-02 §4's stated reason for representing the reduction explicitly. `op` is the A/C tag;
it stays `None` here whenever the source says `+=` but the schedule has not yet tagged it — M2
fills the tag from `reduce(ax, op=)` and M3 checks it (FR-L5).

### 3.9 Integer linear algebra

M1 needs exactly two operations — an integer kernel basis (§3.8 line 6) and an integer linear
solve (§3.7 line 12). Both come from the shared pure-Python helper `intlin`, specified in
[`03-lld-M3-checker.md`](03-lld-M3-checker.md) §3.2 and **owned there**. M1 imports it and adds
nothing. `intlin` is stdlib-only (Python `int`, fraction-free Bareiss elimination), so M1's
"stdlib only" dependency claim (§8) holds.

---

## 4. Invariants, pre/postconditions

**Preconditions of `capture(fn)`**: `fn` is a Python function whose source is retrievable by
`inspect.getsource` (not defined in a REPL, not a C function, not a lambda). A failure here is
a `GrammarError` with `GRAMMAR-UNSUPPORTED-STMT` and a reason saying the source could not be
read and that kernels must live in a file.

**Postconditions of `capture(fn)`** — each is asserted once, at the end of `BUILD_MODEL`, and a
violation is an internal error, not a user error:

1. Every `AccessMap.matrix` has `len(axes)` columns and `rank(operand)` rows (§2.3).
2. Every `Dependence.vector` has `len(axes)` entries and is not all-zero (§2.5).
3. `dependences` is sorted by `(operand, vector)`; `shape_params` is sorted (§2.7).
4. `reduction is not None` **iff** some statement has `kind == "accumulate"` (§2.7).
5. Every name in every `Param.shape` that is a `str` appears in `shape_params` (§2.1).
6. At least one `Param` has `is_written` (§2.1).
7. The model is hashable and equality-comparable by value; two `capture` calls on the same
   function in the same process return **equal** models (NFR-1).

**Invariant across the module**: M1 holds no mutable state between calls, imports no module
outside §8's list, and never touches its arguments' contents.

---

## 5. Error paths

Every rejection is `GrammarError(Diagnostic(...))` with `stage="grammar"` and `location` set to
`(file, lineno)` — `06-interfaces.md` §6.1 makes `location` mandatory for this stage. The four
parts FR-S4 and HLD §4.2 demand are the `reason` (construct + what was seen), the `location`,
the `details` (the actual numbers or names), and the `fix`. `clause` is `None` for M1: there is
no clause, there is a source line.

| Code (`06-interfaces.md` §6.3) | Raised at | The four parts, by example |
|---|---|---|
| `GRAMMAR-BAD-ANNOTATION` | §3.4 line 17, `PARSE_PARAMS` | reason `parameter 'A' is annotated 'int'; kernel parameters must be annotated sp.<dtype>[<dim>, ...]` · at `w1_gemm.py:4` · details `{param: "A", seen: "int", dtypes: ["f32","f16","bf16","i32","i8"]}` · fix `write A: sp.f32[M, K]` |
| `GRAMMAR-UNSUPPORTED-STMT` | §3.5 line 11, 23 | reason `'if' is not in the accepted kernel subset` · at `:12` · details `{node: "If"}` · fix `the subset has for/range, x[...] = e, x[...] += e and max/min only; move the condition into the data or precompute it` |
| `GRAMMAR-UNSUPPORTED-EXPR` | §3.1 walk | reason `'a.b' (attribute access) is not in the accepted kernel subset` · details `{node: "Attribute"}` · fix `pass the value as a kernel parameter` |
| `GRAMMAR-UNSUPPORTED-CALL` | §3.1 walk | reason `call to 'abs' is not allowed; only max and min` · details `{callee: "abs"}` · fix `rewrite with max/min, or precompute into an input array` |
| `GRAMMAR-NONAFFINE-SUBSCRIPT` | §3.6 lines 4, 11, 14 | reason `subscript A[i*j] is a product of two non-constant terms i * j` · at `:9` · details `{operand: "A", dim: 0, expr: "i*j"}` · fix `subscripts must be affine: c1*i + c2*j + c0 with integer c` |
| `GRAMMAR-BAD-RANGE` | §3.4 lines 4, 10, 11, 13 | reason `bound of 'for j' names the enclosing loop variable 'i', so the domain is not rectangular` · at `:7` · details `{axis: "j", bound: "i", enclosing: ["i"]}` · fix `use range(0, N) and guard nothing — the accepted domain is a box` |
| `GRAMMAR-NONUNIFORM-DEP` | §3.7 lines 7, 11 | reason `S is read at S[j,i] and written at S[i,j]; the dependence distance is not constant` · details `{operand: "S", write_matrix: ..., read_matrix: ...}` · fix `make every access to a written array a constant shift of the write` |

The last row's code landed in `06-interfaces.md` v2 before the D0 freeze (REVIEW-round1
RULING 5), closing Q-M1-1: `GRAMMAR-NONUNIFORM-DEP`, stage `grammar`, one of the catalogue's 43.
No `GRAMMAR-UNSUPPORTED-EXPR` fallback remains, so `test_D3_catalogue_complete` and FR-D3's
"never reused" rule both hold.

Nothing else escapes M1. A `SyntaxError` from `ast.parse` cannot occur (the function already
compiled); an `OSError` from `inspect.getsource` is caught and re-raised as above (NFR-7).

---

## 6. Worked examples — M1's exact output for W1, W2, W3

The three sources are Person A's D0 deliverable. They are printed here in the form M1 accepts,
because the rest of this document set quotes their models and nothing else fixes the text.

### 6.1 W1 — GEMM (`kernels/w1_gemm.py`)

```python
M = N = K = 64

@sp.kernel
def gemm(A: sp.f32[M, K], B: sp.f32[K, N], C: sp.f32[M, N]):
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]
```

`KernelModel`: `name = "gemm"`; `shape_params = ("K", "M", "N")`;
`params = [A f32 ("M","K") not written, B f32 ("K","N") not written, C f32 ("M","N") written]`;
`axes = [i(0,M,1,extent 64,depth 0), j(…64…,1), k(…64…,2)]`;
one `Statement(kind="accumulate", op="+", line=8, axes=("i","j","k"))` with

| access | `matrix` (rows = array dims, columns = `(i, j, k)`) | `offsets` |
|---|---|---|
| `C[i,j]` (write) | `((1,0,0), (0,1,0))` | `(0, 0)` |
| `A[i,k]` | `((1,0,0), (0,0,1))` | `(0, 0)` |
| `B[k,j]` | `((0,0,1), (0,1,0))` | `(0, 0)` |

`dependences = (Dependence((0,0,1), "RAW", "C"),)`.
`reduction = ReductionSpec(target="C", projection=((1,0,0),(0,1,0)), space=((0,0,1),), op=None)`
— so `R = ker Sf = span{e_k}`, exactly SD-02 §9's line.

### 6.2 W2 — Jacobi 5-point (`kernels/w2_jacobi.py`)

```python
T = 4; H = W = 16                 # H, W are the INTERIOR extents; U carries the halo

@sp.kernel
def jacobi(U: sp.f32[T + 1, H + 2, W]):
    for t in range(0, T):
        for i in range(1, H + 1):
            for j in range(1, W - 1):
                U[t + 1, i, j] = 0.2 * (U[t, i, j] + U[t, i - 1, j] + U[t, i + 1, j]
                                        + U[t, i, j - 1] + U[t, i, j + 1])
```

`axes = [t(0,T,1,extent 4), i(1,H+1,1,extent 16), j(1,W-1,1,extent 14)]`; one
`Statement(kind="assign", op=None)`; every access to `U` has
`matrix = ((1,0,0),(0,1,0),(0,0,1))`; the offsets are `(1,0,0)` for the write and
`(0,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)` for the five reads.
`dependences = ((1,0,0),(1,-1,0),(1,0,-1),(1,0,1),(1,1,0))` all RAW on `U`, sorted by vector —
exactly HLD §7.2's list. `reduction = None`, so `R = {}`.

> **The W2 contract, settled (REVIEW-round1 RULING 1).** `U` is **one rank-3 parameter**
> `sp.f32[T + 1, H + 2, W]`; the write domain is planes `1..T`, rows `1..H`, cols `1..W-2`; rows
> `0` and `H + 1` and columns `0` and `W - 1` are read-only Dirichlet boundary. `H = W = 16`,
> `PI = 2`, `HS = 8`, so the tiling invariant `PI·HS == H` holds **exactly** and `tile(ax.i, HS)`
> divides the `i` extent of 16. The earlier `H = W = 18` proposal fixed divisibility but left
> `PI = 4`, which is measured to fail `aie.connect` on the first interior PE (REVIEW-round1
> P-R2); `PI = 4` is now a `DMA-CHANNELS` negative fixture, not a positive one. The per-core L1
> figure is **1 280 B** (`02-hld.md` §7). Note the `j` extent is `W - 2 = 14`, which is why `j`
> is **not** tiled or placed in W2's schedule — only `i` is.

### 6.3 W3 — Smith-Waterman wavefront (`kernels/w3_sw.py`)

```python
MQ = NR = 32; MATCH = 2; MISMATCH = -1; GAP = 1

@sp.kernel
def sw(q: sp.i32[MQ], r: sp.i32[NR], S: sp.i32[MQ + 1, NR + 1]):
    for i in range(1, MQ + 1):
        for j in range(1, NR + 1):
            sub = MATCH if q[i - 1] == r[j - 1] else MISMATCH
            S[i, j] = max(0, S[i - 1, j - 1] + sub,
                          S[i - 1, j] - GAP, S[i, j - 1] - GAP)
```

`params = [q i32 ("MQ",) not written, r i32 ("NR",) not written,
S i32 ("MQ+1","NR+1") written]` — **read-only params first**, which `MappingPlan.tensors` must
preserve (`06-interfaces.md` §5.6 invariant 6).
`axes = [i(1,MQ+1,1,extent 32), j(1,NR+1,1,extent 32)]`; one `Statement(kind="assign")` —
**not** `accumulate`, because the target `S[i,j]` does not appear among the `max` arguments, so
`reduction = None` and `R = {}`, which is what HLD §7.3 asserts. Every access to `S` has
`matrix = ((1,0),(0,1))`; offsets `(0,0)` write, `(-1,-1)`, `(-1,0)`, `(0,-1)` reads;
`q[i-1]` has `matrix = ((1,0),)`, offset `(-1,)`; `r[j-1]` has `matrix = ((0,1),)`, offset
`(-1,)`. `dependences = ((0,1),(1,0),(1,1))` RAW on `S`, sorted. Row 0 and column 0 of `S` are
zero-initialised read-only boundary.

The scalar `sub` is bound and read in the same body, so §3.5 forward-substitutes it and it never
reaches the model: the statement's expression is
`MaxMin("maximum", (Const 0, BinOp("+", Load(S,(i-1,j-1)), Select("==", Load(q,(i-1,)), Load(r,(j-1,)), Const 2, Const −1)), BinOp("−", Load(S,(i-1,j)), Const 1), BinOp("−", Load(S,(i,j-1)), Const 1)))`
(`06-interfaces.md` §5.5). `MATCH`, `MISMATCH` and `GAP` resolve from `fn.__globals__` by R-e;
the conditional is one `Select` by R-d; and `max` with four arguments is FR-S3 item 5 ("2 or more
scalar arguments"), so the four-way recurrence needs no rewriting. This is the kernel — there is
no precomputed `Sub` array and no fixture-side comparison.

---

## 7. Unit tests

Marks: `fr(...)` per `04-test-plan.md` §7. Level U unless stated. Every negative asserts the
`code`, a non-empty `reason` containing the named construct, a non-`None` `location`, and a
non-empty `fix` (FR-S4, FR-D1).

| Test id | Input (concrete) | Expected |
|---|---|---|
| `test_M1_w1_model` | §6.1 source | the model of §6.1, asserted **field by field** (axes, three matrices, offsets, the one dependence, the reduction) |
| `test_M1_w2_model` | §6.2 source | the model of §6.2, including all five dependence vectors in sorted order |
| `test_M1_w3_model` | §6.3 source | the model of §6.3, `reduction is None` |
| `test_kernel_call_is_oracle` *(O)* | decorated `gemm`, integer-valued `f32` fixture | `C == A @ B` exactly (FR-S1) |
| `test_annotations_inert` | `gemm` called with plain Python lists | no error from the annotation machinery; `model.params[0].dtype == f16`-style assertion on a `sp.f16[M,K]` variant; `shape == ("M","K")` (FR-S2) |
| `test_ignorability_m1` *(O)* | each of W1/W2/W3, run with no schedule ever constructed | output bytes equal the run with a full schedule constructed but un-built (FR-S18) |
| `test_M1_no_mutation` | capture `gemm`, then compare `gemm.__globals__`, `fn.__code__`, `fn.__dict__` before/after | identical objects; `Kernel.__call__` is `fn` |
| `test_M1_capture_is_pure` | capture the same source twice, and once with `PYTHONHASHSEED` changed | models compare equal; `shape_params` and `dependences` in the same order (NFR-1) |
| **property** `test_M1_oracle_vs_numpy` | 20 random integer fixtures per workload, fixed seed | `kernel(*fixture)` equals the independent numpy reference elementwise (W1 `A @ B`, W2 a two-loop numpy Jacobi, W3 a textbook DP) |
| `test_grammar_select_expr` | W3's `sub = MATCH if q[i-1] == r[j-1] else MISMATCH`, plus the same comparison written as an `if` **statement** | the expression form parses to one `Select` whose arms are `Const 2` / `Const −1` resolved from `fn.__globals__` (R-d, R-e); the statement form raises `GRAMMAR-UNSUPPORTED-STMT` naming `If` (FR-S3 items 7-8) |
| `test_grammar_transitive_binds` | a body binding `d`, then `sub` from `d`, then reading `sub` | both binds are forward-substituted in binding order and neither reaches the model; a body whose binds form a cycle raises `GRAMMAR-UNSUPPORTED-STMT` naming both lines (§3.5 lines 15b-15f) |
| **property** `test_M1_dep_solves` | for each workload, each computed `Dependence` `d` and its originating pair | `M @ d == c_W - c_R` exactly, over the integers |

Negative corpus (`04-test-plan.md` §2 M1 lists the minimum set; these are the cases, one test
each, parametrised as `test_grammar_rejects[...]`):

| # | Source fragment | Code |
|---|---|---|
| 1 | `if i > 0: ...` | `GRAMMAR-UNSUPPORTED-STMT` |
| 2 | `while i < N: ...` | `GRAMMAR-UNSUPPORTED-STMT` |
| 3 | `break` | `GRAMMAR-UNSUPPORTED-STMT` |
| 4 | `continue` | `GRAMMAR-UNSUPPORTED-STMT` |
| 5 | `return C` | `GRAMMAR-UNSUPPORTED-STMT` |
| 6 | `a, b = 1, 2` | `GRAMMAR-UNSUPPORTED-STMT` |
| 7 | `global g` | `GRAMMAR-UNSUPPORTED-STMT` |
| 8 | `acc` bound in the `j` body, read in the `i` body | `GRAMMAR-UNSUPPORTED-STMT` (§3.5 line 23) |
| 9 | `C[i,j] = sum(x for x in ...)` | `GRAMMAR-UNSUPPORTED-EXPR` (`GeneratorExp`) |
| 10 | `f = lambda x: x` | `GRAMMAR-UNSUPPORTED-EXPR` |
| 11 | `C[i,j] = A.shape` | `GRAMMAR-UNSUPPORTED-EXPR` (`Attribute`) |
| 12 | `C[i,j] = A[i,k] if k else 0` | `GRAMMAR-UNSUPPORTED-EXPR` (`IfExp`/`Compare`) |
| 13 | `C[i,j] = A[i,k] ** 2` | `GRAMMAR-UNSUPPORTED-EXPR` (`Pow`) |
| 14 | `C[i,j] = abs(A[i,k])` | `GRAMMAR-UNSUPPORTED-CALL` |
| 15 | `C[i,j] = A[B[i], k]` | `GRAMMAR-NONAFFINE-SUBSCRIPT` (data-dependent) |
| 16 | `C[i,j] = A[i*j]` | `GRAMMAR-NONAFFINE-SUBSCRIPT` |
| 17 | `C[i,j] = A[i**2]` | `GRAMMAR-NONAFFINE-SUBSCRIPT` |
| 18 | `C[i,j] = A[i*M]` with `M` a shape param | `GRAMMAR-NONAFFINE-SUBSCRIPT` (R-b) |
| 19 | `C[i,j] = A[0:4]` | `GRAMMAR-NONAFFINE-SUBSCRIPT` (`Slice`) |
| 20 | `for x in [1,2]:` | `GRAMMAR-BAD-RANGE` |
| 21 | `for j in range(i, N):` | `GRAMMAR-BAD-RANGE` (R-a, non-rectangular) |
| 22 | `for j in range(i*j):` | `GRAMMAR-BAD-RANGE` |
| 23 | `for j in range(0, N, 0):` | `GRAMMAR-BAD-RANGE` (step) |
| 24 | `def k(A, B):` no annotation | `GRAMMAR-BAD-ANNOTATION` |
| 25 | `A: int` | `GRAMMAR-BAD-ANNOTATION` |
| 26 | `A: sp.f64[M]` | `GRAMMAR-BAD-ANNOTATION` |
| 27 | `C[i,j] = A[i]` on a rank-2 `A` | `GRAMMAR-BAD-ANNOTATION` (rank mismatch, §3.6 line 17) |
| 28 | `S[i,j] = S[j,i] + 1` | non-uniform dependence — the code decided by Q-M1-1 |

28 cases against the test plan's floor of 12 (FR-S3) / 16 (test plan §2).

---

## 8. Dependencies

**On other modules**: `model` (M0) for every dataclass and for `GrammarError`/`Diagnostic`; the
`intlin` helper owned by [`03-lld-M3-checker.md`](03-lld-M3-checker.md) §3.2 for the two
integer-algebra calls of §3.7 and §3.8. Nothing else, in either direction — M2, M3, M4, M5 and
M6 all read `KernelModel`, none of them is read by M1.

**On Python**: standard library only — `ast` (parse and walk), `inspect` (`getsource`),
`textwrap` (`dedent`), `dataclasses`, `typing`, `fractions` is **not** used (see M3 §3.2).
**No numpy** and **no `air`** (FR-S20's spirit; M1 is the half of the surface that must work with
no toolchain installed at all).

**Stubs M1 needs from others**: none. M1 is the first module that can be written, and its output
is what B's hand-written `LegalMapping` literal (HLD §8, D0) must agree with — so the three
models of §6 are A's D0 deliverable and are what B builds against.

---

## 9. Implementation order, effort, definition of done

Order: (1) the `Expr`/`AFFINE` core of §3.6, because everything else calls it; (2)
`PARSE_PARAMS` + `EXTRACT_DOMAIN`; (3) `CHECK_GRAMMAR`'s reject walk with its message table;
(4) `CLASSIFY` + `EXTRACT_ACCESS`; (5) `DEPENDENCES`; (6) `REDUCTION`; (7) the decorator and
`Kernel`, which is fifteen lines and should be written last so it is never tempted to grow.

**Effort — estimate, not measured: 5 hours**, of which ~1.5 h is the reject walk and its
messages. Fits D1's morning after M0 lands (`05-work-breakdown.md` §2, D1).

**Definition of done.** All ten tests of §7 plus all 28 negatives pass; the three models of §6
are asserted field by field, not by `repr`; `test_M1_no_mutation` and
`test_ignorability_m1` pass; every public name has a docstring (NFR-5); the module imports
nothing outside §8; and B has consumed `KernelModel` for W1 without asking for a field that is
not in `06-interfaces.md` §2.

---

## 10. Open questions owned by M1

| # | Question | Resolution |
|---|---|---|
| **Q-M1-1** *(closed)* | Which error code covers an affine-but-non-uniform dependence (`S[i,j] = S[j,i]`)? | **Closed by REVIEW-round1 RULING 5**: `GRAMMAR-NONUNIFORM-DEP` (stage `grammar`) landed in `06-interfaces.md` v2 **before** the D0 freeze, so the `GRAMMAR-UNSUPPORTED-EXPR` fallback is retired. The catalogue is now 43 codes. |
| **Q-M1-2** | Is `KernelModel` a pure function of the source text, as HLD §2 says, when loop extents come from module-level `int`s? | **Resolved here** (§3.4 line 18): it is a pure function of *(source text, the integer bindings of the shape-parameter names the source declares)*. Both are recorded in the model; no other global is read. HLD §2's phrasing is a simplification, not a contradiction. |
| **Q-M1-3** | May a `for` bound name an enclosing loop variable (FR-S3 item 1) given the same FR demands rectangularity? | **Resolved here** (§3.1 R-a): no. Rectangularity wins, because FR-S7, FR-L7, FR-L8 and FR-L9 all evaluate affine forms at the vertices of a box. Rejected as `GRAMMAR-BAD-RANGE` with a message that says why. |
| **Q-M1-4** *(closed)* | Where do the W2 fixture's numbers come from, given `14 % 4 != 0`? | **Closed by REVIEW-round1 RULING 1**: the fixture is `H = W = 16`, `PI = 2`, `HS = 8`, `U: sp.f32[T + 1, H + 2, W]`, with `PI·HS == H` exact and the `i` extent 16. `H = W = 18` is **not** adopted, because it leaves `PI = 4`, which is measured to fail `aie.connect` (P-R2). See §6.2. |
