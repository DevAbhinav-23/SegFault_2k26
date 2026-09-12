# Requirements — Spatial DSL (P1 surface over MLIR-AIR)

*Phase 1 design, 2026-09-12. Binding inputs: the architect's design brief
(module decomposition, 3-person split, decisions D1–D11),
[`../hackathon/VERIFIED-AIR-FACTS.md`](../hackathon/VERIFIED-AIR-FACTS.md) (every AIR fact
relied on here), [`../hackathon/01-paradigm-comparison.md`](../hackathon/01-paradigm-comparison.md)
r3 §§1–5 P1 material, and `../spatial-dsl/01`–`05`.*

**Citation convention.** `VF §X` = a section of `VERIFIED-AIR-FACTS.md`. `PC §X` = a section
of `01-paradigm-comparison.md` r3. `SD-0n §X` = `spatial-dsl/0n-*.md`. A bare `path:line` is
the mlir-air clone at commit `ff95a9b35b692b4cfbdf1d52bf69e6ffe01facde`
(`/home/adi/Projects/Honours/mlir-air`, 2026-09-11). `[UNVERIFIED — owner, due day]` marks
anything not sourced.

**No performance claim appears in this document.** Nothing here has been built or run except
the probes recorded in `VERIFIED-AIR-FACTS.md`.

---

## 1. Goals

**G1 — An ignorable schedule surface.** A `@sp.kernel` plain-Python loop nest plus a separate
`sp.schedule` object whose method calls are the pragmas. Deleting every schedule call leaves a
program that runs in CPython and **is** the specification (SD-01 §1, SD-03 §0, PC §3.0 P1).

**G2 — A pre-codegen legality check.** Every illegal schedule is rejected with a named clause,
a reason and a fix hint, *before* any AIR is emitted. This is the entry's differentiator:
mlir-air ships an unimplemented *plan* for the equivalent checker
(`docs/AIRCorrectnessChecker.md`, VF §B.4) and `air-opt` prints an error while exiting 0
(VF §B.6, §S11).

**G3 — Emission through `air.api`, inheriting the whole aircc pipeline.** The emitter targets
`air.api` constructs only; the optimisation pipeline is the stock C++ one inside `aircc`
(VF §D.9, §H.3).

**G4 — Three kernels end to end.** W1 GEMM output-stationary, W2 Jacobi 5-point with halo
exchange, W3 Smith-Waterman wavefront — each from Python source to AIR text to an `aircc`
artifact, with the CPython oracle as the correctness reference.

**G5 — Honest scope.** Every limitation is stated on the demo's honest-limits slide: one
lowering path, two device flags, scalar compute bodies, `air-runner` is a timing model and not
an oracle (VF §S7).

## 2. Non-goals

| Non-goal | Why | Where it is recorded |
|---|---|---|
| Performance-optimisation passes, cost model, autotuner, DSE, II search | Out of scope by `CLAUDE.md` and PC §1.3 | honest-limits slide |
| Vectorised compute (extern `.cc` kernels via `air.extern`) | Extern kernels buy *vectorisation*, not expressiveness (VF §D.11); and an extern call as a buffer's first toucher **silently disables ping-pong** (VF §E.2 item 3) | §9 future work, FR-E2 |
| W4 FFT | `p ↦ p ⊕ 2^s` is affine in neither `p` nor `s` (PC §2); admitted only if W1–W3 pass on device (D8) | FR-K5 |
| Multi-kernel composition / fusion (`SD-05`) | Needs a second graph-level vocabulary; out of the hackathon cut | §9 future work |
| GPUs, Qualcomm Hexagon, Tenstorrent | Never in scope (`CLAUDE.md`); not AIR targets (PC §1.2) | — |
| Versal (`xcvc1902`) end to end | `air.api`'s `resolve_target` accepts only `npu1`/`npu2` (`_trace.py:172-177`, VF §F.3/§S5) | FR-T1, R-14 |
| Solving for `(σ,π)` | The user supplies `place`/`stationary`/`skew`; step 2 is a *check*, not a solve (PC §1.3) | FR-L1…L11 |
| Data-dependent control flow, sparsity, softmax | Expressiveness cliff, stated openly (SD-01 §5, SD-03 §4) | FR-S3 rejects it |
| Re-deriving NumPy broadcast rules | `ChannelOp::verify()` already enforces them (`AIRDialect.cpp:3777-3806`, VF §A.4) | FR-M2 |

---

## 3. Functional requirements

Each FR carries: **Shall** (the requirement), **Rationale**, **Source**, **Acceptance** (a
concrete checkable test). Owner is the module owner from §7 of the HLD (Person A: M1–M3;
Person B: M4–M5; Person C: M6–M8).

### 3.1 Surface (M1 frontend, M2 schedule IR)

**FR-S1 — kernel decorator.**
*Shall*: `@sp.kernel` shall accept a Python function whose parameters are annotated with
`sp.<dtype>[<dim>, ...]` and shall return an object that (a) is callable with the same
positional arguments as the undecorated function and executes the original body unchanged in
CPython, and (b) exposes the parsed `KernelModel`.
*Rationale*: the oracle and the compiler front end must read the same source text.
*Source*: D1; SD-03 §0 ("a plain typed Python nested-loop kernel **executes in CPython as-is**").
*Acceptance*: `test_kernel_call_is_oracle` — call the decorated `gemm` on numpy arrays;
compare element-wise against `A @ B` computed by numpy; assert exact equality for an integer
fixture and `allclose` for float.

**FR-S2 — parameter types.**
*Shall*: the annotation forms `sp.f32`, `sp.f16`, `sp.bf16`, `sp.i32`, `sp.i8` subscripted by
a tuple of shape entries shall be accepted; a shape entry shall be a positive Python `int` or
a named shape parameter. Annotations shall be inert at run time (subscripting returns a
descriptor object; it must not wrap or copy the argument).
*Rationale*: the emitter needs dtype and rank for `air.tensor(shape, dtype)` (`_trace.py:1803`);
the oracle needs the annotation to cost nothing.
*Source*: VF §D.2 (L3 is `air.tensor`, not `alloc`).
*Acceptance*: `test_annotations_inert` — a decorated kernel called with plain Python lists
raises no type error from the annotation machinery; `KernelModel.params` reports
`dtype="f16", shape=("M","K")`.

**FR-S3 — kernel body grammar (the accepted subset).**
*Shall*: the body grammar shall be exactly:
1. `for <v> in range(<lo>, <hi>[, <step>])` — `lo`/`hi`/`step` each a Python integer literal,
   a shape parameter, or an affine expression over enclosing loop variables, shape parameters
   and integer constants; loops perfectly or imperfectly nested, rectangular after
   substitution of parameters;
2. assignment `x[<subs>] = <expr>` and augmented assignment `x[<subs>] += <expr>`;
3. subscripts `<subs>` affine in enclosing loop variables, shape parameters and constants;
4. arithmetic on scalars: `+ - * /`, unary `-`, integer and float literals;
5. calls to `max` and `min` only, with 2 or more scalar arguments;
6. a local scalar name bound by simple assignment (`acc = 0.0`) and read later in the same loop
   body; substitution is **transitive** and applied in binding order (W3 binds `d`, then `sub`
   from `d`), and a binding cycle is `GRAMMAR-UNSUPPORTED-STMT`;
7. a bare `NAME` in an expression that is neither an axis, a shape parameter nor a kernel
   parameter, resolved from `fn.__globals__` when it is bound to a Python `int` and treated as a
   literal of that value (W3's `MATCH`, `MISMATCH`, `GAP`); otherwise `GRAMMAR-UNSUPPORTED-EXPR`;
8. a **value-level conditional expression** `a if <cmp> else b`, where `<cmp>` compares two
   scalar expressions with `== != < <= > >=` and `a`, `b` are scalar expressions. This is a
   `SELECT` (`arith.select`, `06-interfaces.md` §5.5 `ExprNode.Select`), **not** control flow:
   both arms are evaluated and no branch is emitted.
Everything else — `if`/`while` **statements**, `break`, `continue`, comprehensions, attribute
access, any other call, data-dependent subscripts, tuple assignment, `global`/`nonlocal`,
lambdas, slicing, `**` — shall be rejected. A comparison outside the conditional-expression form
of item 8 is still `GRAMMAR-UNSUPPORTED-EXPR`.
*Rationale*: the `(σ,π)` machinery is defined over affine domains (SD-02 §1); accepting more
than it can analyse produces silent mis-compilation.
*Source*: D9 as amended by REVIEW-round1 RULING 2 (items 6-8 added so that W3's
`sub = MATCH if q[i-1] == r[j-1] else MISMATCH` is inside the subset); SD-02 §1; SD-01 §5 open
problem 2.
*Acceptance*: `test_grammar_accepts` — the three kernel sources of W1/W2/W3 parse without
error; `test_grammar_rejects[...]` — a parameterised negative test with one source per rejected
construct (≥ 12 cases), each asserting `GrammarError` and that the message text contains the
construct's Python name and the source line number; `test_grammar_select_expr` — W3's
`sub = MATCH if q[i-1] == r[j-1] else MISMATCH` parses to one `Select` with `MATCH`/`MISMATCH`
resolved from `fn.__globals__`, while the same comparison used as a statement `if` is rejected.

**FR-S4 — grammar rejection message.**
*Shall*: a `GrammarError` shall name (a) the rejected construct by its Python grammar name,
(b) the source file and line, (c) one sentence saying what the accepted subset allows instead.
*Rationale*: judges see rejections; a rejection with no fix hint reads as a crash (brief §5,
"error-message quality").
*Source*: D9; brief must-not-surprise list.
*Acceptance*: covered by `test_grammar_rejects[...]` above, which asserts all three parts are
present.

**FR-S5 — schedule object.**
*Shall*: `sp.schedule(kernel, target=...)` shall return a `Schedule` whose `axes()` returns an
axis-handle namespace with one attribute per loop variable of the kernel, and which accumulates
clauses as pure data. `target` shall accept `"npu1"`, `"npu2"` and `"auto"`.
*Rationale*: D1; D7. `"auto"` shells out to `xrt-smi` and falls back to `npu2` with no device,
so tests must pass an explicit target.
*Source*: SD-03 §0; VF §F.3 (`_trace.py:164-183`).
*Acceptance*: `test_schedule_is_pure_data` — building a schedule with every clause emits no
MLIR, imports no `air` module, and leaves `ScheduleModel` equal to a stored fixture.

**FR-S6 — `grid(PI[, PJ])`.**
*Shall*: declare a *logical* PE grid of rank 1 or 2. Arguments are positive ints. Rank > 2
shall be rejected at clause time with the reason "air.api supports 1-D and 2-D herd grids".
Default: no grid ⇒ the schedule is rejected at legality time (a schedule with `place` and no
`grid` is inconsistent).
*Rationale*: `air.herd` is 1-D or 2-D only.
*Source*: VF §D.1 (`_trace.py:1288-1291`).
*Acceptance*: `test_grid_rank3_rejected` asserts `ClauseError` naming `grid` and the 2-D cap.

**FR-S7 — `tile(ax, F)`.**
*Shall*: strip-mine axis `ax` by factor `F` (positive int), creating handles `ax0` (outer) and
`ax1` (inner) that are valid arguments to later clauses. `F` shall divide the axis extent
exactly when the extent is a literal; otherwise the schedule shall be rejected.
*Rationale*: `air.herd`'s own strip-mining requires exact division (`_trace.py:1363-1368`), and
a partial tile has no `air.api` spelling in our emission shape.
*Source*: PC §3.0 P1 glossary; VF §D.1.
*Acceptance*: `test_tile_must_divide` — `tile(ax.i, 3)` on `M = 64` raises `ClauseError` naming
`tile`, `i`, `64`, `3`.

**FR-S8 — `place(px=ax[, py=ax])`.**
*Shall*: set the rows of `Sπ` to the named axes, in order. Each argument shall be an axis
handle of the kernel (possibly a tile handle). An axis may be placed at most once. Rank of
`place` shall equal rank of `grid`.
*Rationale*: `π` is the allocation map (SD-02 §1); PC §3.1 W1·P1.
*Source*: SD-02 §7.
*Acceptance*: `test_place_rank_matches_grid`; `test_place_duplicate_axis_rejected`.

**FR-S9 — `reduce(ax, op=)`.**
*Shall*: declare `ax` a reduction axis and tag the accumulation operator. `op` ∈ `{"+", "max",
"min"}`; all three are associative and commutative. Default when `op` is omitted: `"+"`.
Declaring `reduce` on an axis the kernel body does not accumulate over shall be rejected.
*Rationale*: the A/C tag is what unlocks reassociation, hence spatial reduction (SD-02 §4).
*Source*: SD-02 §4, §5; SD-03 §4 item 3 (`max` is A/C).
*Acceptance*: `test_reduce_op_domain` (bad `op` string rejected, naming the three legal values);
`test_reduce_axis_must_accumulate`.

**FR-S10 — `stationary(name)`.**
*Shall*: assert that operand `name` (a kernel parameter name) is stationary. Checked, not
solved (FR-L3). An unknown operand name shall be rejected at clause time listing the kernel's
parameter names.
*Rationale*: "naming it is ergonomic sugar … it *is* a clause of the map's definition"
(SD-02 §3).
*Source*: SD-02 §3.
*Acceptance*: `test_stationary_unknown_operand` asserts the message lists `A, B, C`.

**FR-S11 — `stream(name, pattern=, along=ax)` and `forward(name, along=ax, dir=, depth=)`.**
*Shall*: `pattern` ∈ `{"broadcast", "forward", "cascade"}`; `along` is a *placed* axis handle.
The clause **overrides** the delivery mode the mapper would derive (FR-M1). `forward(...)` is
sugar for `stream(name, pattern="forward", along=ax)` plus a direction; `dir` ∈ `{"W->E",
"E->W", "N->S", "S->N"}`; `depth` is a hint recorded in the plan and **not** emitted
(FR-E4). Default: no clause ⇒ delivery is derived.
*Rationale*: SD-01 §3's vocabulary table has `stream(buf: pattern)` with exactly these three
patterns; SD-04 §3 asks for a derived-by-default, overridable lever.
*Source*: SD-01 §3; SD-04 §3; PC §3.0 P1 glossary.
*Acceptance*: `test_stream_overrides_derivation` — a W1 schedule with
`stream("A", pattern="forward", along=ax.j0)` produces a `MappingPlan` whose A-operand
delivery is `FORWARD` where the default plan has `MULTICAST`; `test_stream_along_must_be_placed`.

**FR-S12 — `reside(name=level, ...)`.**
*Shall*: `level` ∈ `{"L1", "L3"}` **in this cut**. `L1` maps to a herd-private allocation, `L3`
to the launch's `air.tensor` interface; `L3` shall not be `alloc`-able. `"L2"` is accepted by the
clause (it is in the `Level` enumeration) and rejected by the mapper with
`PROTOCOL-UNSUPPORTED` naming `reside`, because no segment-private staging protocol is
synthesised for this cut — REVIEW-round1 RULING 7. L2 residency is §9 future work.
*Rationale*: the four `air.api` allocation scopes and the fact that L3 is `air.tensor`, not
`alloc` (VF §D.2).
*Source*: VF §D.2 (`_trace.py:1845`, `:1803`).
*Acceptance*: `test_reside_l3_alloc_rejected`; `test_reside_maps_to_scope` asserts the
`BufferPlan` scope string for `L1` (`"herd.private"`) and `L3` (`"tensor"`), and asserts that
`reside(x="L2")` raises `MappingError(code="PROTOCOL-UNSUPPORTED")` whose `clause` is `reside`.

**FR-S13 — `double_buffer(name)`.**
*Shall*: assert that operand `name`'s L1 tile is allocated inside the innermost streamed loop
in the shape `air-label-scf-for-to-ping-pong` requires, and record that assertion in the plan.
The emitter shall **not** pass `buffer_resources` to `air.api`. If the emitted loop shape does
not satisfy the pattern, the *checker* shall reject the schedule with a message naming which of
the pattern's conditions fails.
*Rationale*: D3. `air.api` raises `NotImplementedError` on `buffer_resources`
(`_channel.py:563`, VF §D.4); the ping-pong passes are in the default `aircc` pipeline
(`aircc.cpp:966-980`) and fired unprompted on an `air.api` GEMM (VF §E.5).
*Source*: D3; VF §E.2, §E.5, §D.4.
*Acceptance*: `test_double_buffer_ir_shape` (integration) — emit W1, run
`air-opt -air-dependency,…,air-label-scf-for-to-ping-pong{device=npu1}` and assert the K loop
carries `unroll = 2 : i32`; `test_no_buffer_resources_arg` — grep the emitter's `air.api` call
sites for `buffer_resources` and assert zero hits.

**FR-S14 — `pipeline(ax)` and `sequential(ax)`.**
*Shall*: `sequential(ax)` shall force `ax` into `ker Sπ` (temporal) and shall be emitted as
`air.sequential`. `pipeline(ax)` shall be **recorded and read by nothing**; it is reserved for a
future vectorisation pass, changes no `σ` row, and emits nothing of its own.
*Rationale*: `air.sequential` is the only scalar loop construct and carries **no loop-carried
values** (VF §D.5); software pipelining is the backend's job.
*Source*: VF §D.5 (`_loop.py:89`, `:180`).
*Acceptance*: `test_sequential_emits_scf_for` — the emitted text contains one `scf.for` per
`sequential` axis; `test_pipeline_is_hint` — two schedules differing only in `pipeline` emit
byte-identical AIR text.

**FR-S15 — `window(name, dims, halo=h)`.**
*Shall*: declare an `h`-wide ghost region on operand `name` along `dims`. `h` is a
non-negative int or a tuple with one entry per dim. The *boundary rows are never written by
the user*: they are derived from the access map (FR-M4).
*Rationale*: eliminates the halo off-by-one class by construction (PC §3.2 W2·P1 (e)).
*Source*: SD-03 §1; PC §3.0 P1 glossary.
*Acceptance*: `test_window_halo_derived` — the W2 `ChannelPlan` names row indices `1`, `HS`
(sent) and `0`, `HS+1` (received) with no user-supplied index anywhere in the schedule text.

**FR-S16 — `exchange(name, along=ax0, halo=h)`.**
*Shall*: declare that neighbouring PEs along `ax0` swap their `h`-wide boundary each step. The
protocol is put-first (FR-M4) and is synthesised, not written.
*Source*: PC §3.0 P1 glossary; D6.
*Acceptance*: `test_exchange_protocol_shape` (see FR-M4).

**FR-S17 — `skew(time=(a, b))`.**
*Shall*: `skew(time=(a, b))` **defines `σ`'s leading row** as the sum of the named axes, in
order (`skew` is a sum, so the order of the terms does not change the row). The **remaining `σ`
rows are the default loop order restricted to axes that are neither named in `skew` nor placed**;
causality is then checked lexicographically over those rows (FR-L2, `03-lld-M3-checker.md` §3.3
line 24). Excluding the placed axes is load-bearing: it is what makes W3's dropped-term schedule
a rejection rather than an acceptance (`03-lld-M8-kernels-demo.md` §3.5).
*Source*: SD-02 §2; PC §3.3 W3·P1; REVIEW-round1 RULING 6.
*Acceptance*: `test_skew_sets_sigma` — `ScheduleModel.skew == ("i", "j0")`, and the
`LegalMapping.sigma` it produces for W3 is `[e_i + e_j0; e_j1]` — leading row the sum, then `j1`
only, with `j0` excluded as skewed and `i` retained only inside the leading row.

**FR-S18 — ignorability (hard requirement).**
*Shall*: no clause shall mutate the kernel function, its module, or any global state observable
to the kernel. Removing every `sp.schedule`/`s.*` line from a test program shall leave a
program whose output is bit-identical to the same program with the schedule present but
un-built.
*Rationale*: D1; the defining property of the column (PC §3.0).
*Source*: D1; SD-01 §1.
*Acceptance*: `test_ignorability` — for each of W1/W2/W3, run the kernel (i) with no schedule
constructed and (ii) after constructing the full schedule, and assert the two output arrays are
bit-identical (`numpy.array_equal` on the raw bytes).

**FR-S19 — clause argument validation.**
*Shall*: every clause shall validate its arguments at call time and raise `ClauseError` naming
the clause, the offending argument, its value, and the accepted domain. Unknown axis handles,
unknown operand names, out-of-domain enum strings, non-positive factors and rank mismatches are
all clause-time errors.
*Rationale*: a schedule that fails late fails in the middle of emission, where the message is
about MLIR, not about the user's clause.
*Source*: brief must-not-surprise item "error-message quality"; mirrors `air.api`'s own style
(`_channel.py:127-131`).
*Acceptance*: `test_clause_errors[...]` — one case per clause per argument-domain violation
(≥ 20 cases), each asserting the four message parts.

**FR-S20 — the schedule does not lower.**
*Shall*: constructing a `Schedule` and calling any clause shall not import `air`, shall not
create an MLIR context, and shall not run the legality checker. Lowering begins only at
`s.check()`, `s.mlir()` or `s.build(...)`.
*Rationale*: keeps the surface usable with no toolchain installed, and keeps M2 pure data (brief §2).
*Source*: brief §2 M2.
*Acceptance*: `test_no_air_import_until_build` — build a full schedule inside a process whose
`sys.modules` is asserted not to contain `air` afterwards.

### 3.2 Legality (M3)

All checks operate on small integer matrices; `Sσ` and `Sπ` are the linear parts of `σ` and `π`
over the `n`-dimensional iteration domain (SD-02 §1).

**FR-L1 — (L1) no structural conflict.**
*Shall*: reject unless `ker Sσ ∩ ker Sπ = {0}`.
*Rationale*: "a PE does one thing per cycle" (SD-02 §2 L1).
*Source*: SD-02 §2.
*Acceptance*: `test_L1_conflict` — W1 with `skew(time=(ax.i1, ax.j1))`, whose
`ker Sσ ∩ ker Sπ = span{e_i1 − e_j1} ≠ {0}`, is rejected with
`LegalityError(code="L1-CONFLICT")` naming `skew`, `place` and the offending direction.
(A schedule that merely places `i` and leaves `i` out of `skew` is **legal**: `03-lld-M3-checker.md`
§3.3 line 27 gives every unplaced, unskewed axis its own `σ` row, so `ker Sσ` is trivial.)

**FR-L2 — (L2) causality.**
*Shall*: for every uniform dependence vector `d` of the kernel, reject unless `Sσ · d ⪰ 1`
lexicographically.
*Rationale*: SD-02 §2 L2; this is the check that makes an illegal `skew` a compile-time
rejection, which PC §3.3 W3·P1 (e) identifies as the one safety property that is neither
upstream nor in any neighbour.
*Source*: SD-02 §2.
*Acceptance*: `test_L2_causality` — W3 with `skew(time=(ax.i,))`, the **dropped `j0` term**, is
rejected with `LegalityError(code="L2-CAUSALITY")` printing the violating dependence vector among
`(1,0)`, `(0,1)`, `(1,1)` and the product that came out `⪯ 0`. (Reversing the two terms to
`skew(time=(ax.j0, ax.i))` is a **no-op** — `skew` is a sum, FR-S17 — so the dropped term is the
only rejection this demo can use.) Under FR-S17's rule the remaining `σ` rows exclude both the
skewed axis `j0` and the placed axis `j0`, leaving `Sσ = [e_i; e_j1]`, and the tile-crossing
representative of `d = (0,1)` gives `Sσ·d = (0, −7) ⪯ 0`.

**FR-L3 — stationarity.**
*Shall*: for each `stationary(a)`, reject unless `ker M_a ⊆ ker Sπ`.
*Rationale*: "Operand `a` is stationary ⟺ `L_a = ker M_a ⊆ ker Sπ`" (SD-02 §3).
*Source*: SD-02 §3.
*Acceptance*: `test_L3_stationarity` — W1 with `place(px=ax.i0, py=ax.k0)` and
`stationary("C")` is rejected with `code="STATIONARITY"`, the message naming `C`, `ker M_C =
span{e_k}` and the placed axes.

**FR-L4 — reduction split.**
*Shall*: compute `R = ker Sf`, `R_time = R ∩ ker Sπ`, `R_space = R / R_time`, and record both
in the `LegalMapping`.
*Source*: SD-02 §5.
*Acceptance*: `test_L4_split` — W1 output-stationary gives `R_time = span{e_k}`,
`R_space = {}`; the weight-stationary flip gives `R_space = span{e_k}`.

**FR-L5 — A/C operator requirement for spatial reduction.**
*Shall*: if `R_space ≠ {}`, reject unless the reduction operator was declared via `reduce(ax,
op=)` with an op in the A/C set.
*Rationale*: "without that tag the accumulation chain is a *rigid* sequential dependence"
(SD-02 §4).
*Source*: SD-02 §4, §5 R2.1.
*Acceptance*: `test_L5_ac_required` — the W1 flip with the `reduce` clause deleted is rejected
with `code="RSPACE-NO-AC-OP"` and a fix hint naming `reduce(ax.k, op="+")`.

**FR-L6 — cascade rider (AIE-specific).**
*Shall*: if `R_space ≠ {}` and the declared realisation is a cascade, reject unless
(a) `rank(R_space) = 1`, (b) the spatial axis carrying it is mapped to a **contiguous line** of
PEs, and (c) the herd for that kernel is 1-D or has extent 1 on the other axis. The plan shall
record the carrying line; **no `at=` pinning is recorded** — `HerdPlan.at` stays `None`.
*Rationale*: "Cascade is unidirectional and linear … prefer `R_space` of rank 1 mapped along a
contiguous PE line" (SD-02 §6); `air.api`'s own text: "a cascade is a physical link between
neighbouring cores"; `at=` exists for pinning such a line (`_trace.py:1774-1778`) but is **not**
used — Q-1 measured that `air-place-herds` picks a contiguous line on its own, and pinning only
removes its freedom to avoid an occupied column (`03-lld-B-open-questions.md` §4); and
`air.channel(channel_type="npu_cascade")` rejects `broadcast_shape` (`_channel.py:170-177`).
*Source*: SD-02 §6; `_channel.py:170-177`; `_trace.py:1771-1778`.
*Acceptance*: `test_L6_cascade_rank` — a rank-2 `R_space` with cascade is rejected with
`code="CASCADE-RANK"`; `test_L6_cascade_no_broadcast` — a cascade channel with a broadcast
pattern on the same operand is rejected before `air.api` would raise.

**FR-L7 — halo ≥ dependence footprint.**
*Shall*: for each `window(name, dims, halo=h)`, compute the maximum absolute offset of `name`'s
accesses along each of `dims` after `π` is applied, and reject unless `h` is at least that
footprint on every dim.
*Rationale*: an under-declared halo is a silent wrong answer, which is the bug class this
surface exists to remove (PC §3.2 W2·P1 (e)).
*Source*: SD-03 §1; PC §3.2.
*Acceptance*: `test_L7_halo_too_small` — W2 with `halo=0` is rejected with
`code="HALO-TOO-SMALL"`, the message printing the derived footprint `1` and the declared `0`.

**FR-L8 — grid / place / tile consistency.**
*Shall*: reject unless (a) `rank(place) == rank(grid)`, (b) every placed axis exists in the
(possibly tiled) axis set, (c) the extent of each placed axis equals the corresponding grid
extent, (d) no axis is both placed and declared `sequential`.
*Source*: D1; SD-02 §7.
*Acceptance*: `test_L8_consistency[...]` — four cases, one per sub-clause, each with its own
error code.

**FR-L9 — L1 capacity.**
*Shall*: compute the per-core L1 working set as the sum over herd-private buffers of
`prod(shape) × sizeof(dtype)`, **doubled for every buffer marked `double_buffer`**, and reject
unless the total is ≤ 65 536 bytes.
*Rationale*: `L1_BYTES = 65536` is `air.api`'s own trace-time budget (`_trace.py:100`); the
figure that has to fit is the ping-ponged one (`_compile.py`'s `_annotate_l1_failure`
docstring: "the declared buffers can fit in L1 and the design still not place, because the
pipeline ping-pongs L1 buffers"); and `air-label-scf-for-to-ping-pong` itself declines on the
L1 budget (`exceedsL1Budget`, VF §E.2 item 6).
*Source*: `_trace.py:100`; `_compile.py` `_annotate_l1_failure`; VF §D.2, §E.2.
*Acceptance*: `test_L9_capacity` — a W1 schedule with `M=N=K=192`, `TM=TN=96`, `TK=32` in `f32`
and `double_buffer("A","B")` is rejected with `code="L1-CAPACITY"`: the undoubled working set is
`96·96·4 + 96·32·4 + 32·96·4 = 36 864 + 12 288 + 12 288 = 61 440` B, which **fits**; doubling
`A` and `B` charges `36 864 + 2·12 288 + 2·12 288 = 86 016` B, `20 480` over the 65 536 budget.
The message prints the computed bytes, the budget, and the per-buffer breakdown.
(`03-lld-M8-kernels-demo.md` §3.5. This case is used rather than `TM=TN=TK=128` in `bf16`, which
is `3 × 32 768 = 98 304` B **before** doubling and therefore never demonstrates the doubling.)

**FR-L10 — herd shape vs physical array.**
*Shall*: reject unless every logical grid extent has a divisor ≤ the target's physical cap:
`npu1` = `{1-D: 4, 2-D: (1, 4)}`, `npu2` = `{1-D: 8, 2-D: (2, 4)}`. The plan shall record the
resolved physical shape (the largest divisor of each extent not exceeding the cap) and the
repeat factors.
*Rationale*: `air.api` strip-mines a larger logical grid and requires exact division
(`_trace.py:88-91`, `:1297`, `:1355-1372`).
*Source*: VF §D.1; `_trace.py:88-91`, `:1345-1372`.
*Acceptance*: `test_L10_physical` — `grid(3, 5)` on `npu1` resolves to physical `(1, 1)` with
repeats `(3, 5)` and the plan says so; `grid(4)` on `npu1` resolves to physical `(4,)`,
repeats `(1,)`.

**FR-L11 — grid rank ≤ 2.**
*Shall*: reject a grid of rank > 2 at legality time even if the clause was constructed
programmatically.
*Source*: VF §D.1 (`_trace.py:1288-1291`).
*Acceptance*: `test_L11_rank`.

**FR-L12 — rejection diagnostic.**
*Shall*: every `LegalityError` shall carry a stable `code`, the **clause** that is implicated,
the **reason** stated in the terms of the check (the matrix identity or the inequality that
failed, with the actual numbers), and a **fix hint** naming a concrete clause edit.
*Source*: G2; brief must-not-surprise item "error-message quality".
*Acceptance*: `test_legality_error_schema` — for every legality test above, assert the
`Diagnostic` has all four fields non-empty and that `code` is in the catalogue of §3.6.

**FR-L13 — check before emit.**
*Shall*: `s.mlir()` and `s.build()` shall run the full legality check first and shall not
import `air` or construct any AIR op if the check fails.
*Rationale*: G2; the whole differentiator is *pre-codegen* rejection.
*Acceptance*: `test_no_emission_on_illegal` — an illegal schedule's `build()` raises
`LegalityError` and a spy on the emitter records zero `air.api` calls.

**FR-L14 — swap parity.**
*Shall*: when a protocol realises a buffer swap by unrolling a temporal loop by two (the `u`/`v`
pair in W2, the `prev`/`cur` pair in W3), reject unless that loop's trip count is a compile-time
constant `T ≥ 1`, naming the loop, the trip count, and the clause that sets it. **An odd trip
count is legal**: the mapper emits `floor(T/2)` unrolled pairs inside one `air.sequential` and
**peels the final timestep** after the loop, so the swap is never carried across trips.
*Rationale*: `air.sequential` emits `scf.for` with `yield_([])` and has **no `iter_args`
anywhere in `air.api`** (`_loop.py:180`; VF §D.5), so a swap cannot be carried across trips; and
a plain Python loop is the wrong tool because it unrolls and strands the acquire/release pairs,
computing "with stale operands" (`_loop.py:14-19`). Unrolling by two inside one `air.sequential`
trip is the remaining option; the residual odd timestep is **peeled by M4**, not rejected
(architect adjudication 2026-09-12, overriding decision D-4 — `00-README.md` §7).
*Source*: VF §D.5; `_loop.py:14-19`, `:180`; architect adjudication, `00-README.md` §7.
*Acceptance*: `test_L14_swap_parity` — W2 with `T = 5` is **accepted**, and the resulting
`MappingPlan` contains an `air.sequential` of trip count 2 (two unrolled pairs) followed by one
peeled timestep body; W2 with `T = 0` is rejected with `code="SWAP-PARITY"` naming `t`, `0`, and
the fixture parameter that sets it.

### 3.3 Mapping and protocol synthesis (M4)

**FR-M1 — reuse trichotomy.**
*Shall*: for each operand `a` and each loop direction `r`, classify delivery as
**stationary** (`r ∈ ker M_a` and `π(r) = 0`), **multicast** (`r ∈ ker M_a` and `π(r) ≠ 0`),
or **stream/flow** (`r ∉ ker M_a` and `π(r) ≠ 0`), and record one classification per operand in
the plan.
*Source*: SD-04 §3 (the trichotomy table).
*Acceptance*: `test_M1_trichotomy` — W1 output-stationary yields `C: STATIONARY`,
`A: MULTICAST along py`, `B: MULTICAST along px`, matching SD-04 §5's worked derivation.

**FR-M2 — multicast emission.**
*Shall*: a multicast operand shall be emitted as one `air.channel(name, size=S,
broadcast_shape=B)` where `len(S) == len(B)`, `B[axis] % S[axis] == 0`, `S` has extent 1 on the
broadcast axis and the herd extent on the others; the put site carries `indices` into `S` and
the get site carries the herd coordinates.
*Rationale*: this is `air.api`'s own fan-out idiom, e.g.
`air.channel("Q2L1", size=[NR,1], broadcast_shape=[NR,2])`
(`programming_examples/flash_attention/kernel_fusion_based/attn_npu2_temporal_causal.py:305`),
and `broadcast_selective_capture.py:56-57` states the rule: "broadcast_shape is the consumer
grid the get indexes".
*Source*: VF §A.4, §D.4; `_channel.py:120-145`; the example lines above.
*Acceptance*: `test_M2_broadcast_shape` — the W1 plan has
`ChannelPlan(name="A2L1", size=[PI,1], broadcast_shape=[PI,PJ])` and the emitted text contains
`air.channel @A2L1 [2, 1] {broadcast_shape = [2 : index, 2 : index]}` for the `PI=PJ=2`
fixture — the attribute is printed with `: index` on every entry (measured, `$PROBE/q7_a.mlir:4`).

**FR-M3 — declared override.**
*Shall*: a `stream(name, pattern=…)` clause shall override the derived classification for that
operand, and the mapping summary shall mark the operand `declared` rather than `derived`.
*Source*: FR-S11; SD-04 §3.
*Acceptance*: covered by `test_stream_overrides_derivation` (FR-S11).

**FR-M4 — halo protocol (W2).**
*Shall*: for each `exchange(name, along=ax0, halo=h)` the plan shall contain, per PE per
timestep and **in this order**: `put` of the north boundary rows, `put` of the south boundary
rows, `get` of the north ghost rows, `get` of the south ghost rows — with the two puts in
asynchronous form and the two gets carrying **no token dependency** on them. There shall be no
prologue, no peeled iteration and no seeded ghost value. Boundary PEs shall have their missing
side guarded so that every channel index still sees equal puts and gets.
*Rationale*: D6. Per-iteration balance is required by AIR's own model (PC §1.1 fact 2, bullet
2), a prologue put is therefore illegal; the `t=0` ghost is the neighbour's *initial* boundary
row, not zero (PC §3.2). E1 is **resolved SAFE**: no producer lock is ever initialised to 0
(VF §C.7), the lock allocator does that by construction
(`AIRToAIESchedulingUtils.cpp:1230-1246`, `:502-503`), and
`programming_examples/channel_examples/worker_to_worker/` is a hardware-CI-run ring of exactly
this protocol (`run_makefile_peano.lit:4-9`).
*Source*: D6; VF §C, §C.7.
At `PI = 2` there is one neighbour per PE, so the per-timestep sequence is
`PUT(boundary) → GET(ghost) → compute → PUT(strip to UOut)`; the trailing drain put is what makes
the L3 image cover every plane (FR-K3), and it is the second and last outbound endpoint on the
core, which is the budget `DMA-CHANNELS` enforces.
*Acceptance*: `test_M4_halo_protocol` — the W2 plan's site list for an interior PE is exactly
`[PUT(north), PUT(south), GET(north_ghost), GET(south_ghost), …, PUT(UOut)]` with `async=True` on
the boundary puts and `dependency=None` on the ghost gets; `test_M4_balanced` — the self-check of
FR-M9 reports balance for every channel index including both boundary PEs
(`03-lld-M4-mapping.md` §7).

**FR-M5 — wavefront forward protocol (W3).**
*Shall*: for each `forward(name, along=ax0, dir=)` the plan shall contain one scalar `get` at
the start of each row body and one scalar `put` at the end, spread across **three homogeneous
channels** — `WestIn` (L3→L1, `size=[1]`, the constant-zero source into PE 0), `West` (L1→L1,
`size=[PJ−1]`, the PE-to-PE links) and `EastOut` (L1→L3, `size=[1]`, the drain from the last
PE) — so that the `PJ+1` channel indices are balanced and **no bundle mixes an L3 endpoint with
a core-to-core member**, which `AIRLoweringPass.cpp:798` rejects with `failed to specialize
channel bundle indices` (measured). Guards shall be emitted with `ops.branch`, never a Python
`if` on a coordinate.
*Rationale*: D6; PC §3.3 W3·P1 (d); "balance must hold independently on each branch"
(PC §1.1 fact 2, bullet 4). A Python `if` on a herd coordinate raises, because `bool()` on a
`Condition` is refused (`_cond.py:41-45`), and `air-to-aie` folds the branch away once the
coordinate is a literal (`_cond.py:49-54`).
*Source*: D6; VF §D.6.
*Acceptance*: `test_M5_wavefront_balance` — the W3 plan for `PJ=4` has 5 channel indices across
the three channels (`WestIn[0]`, `West[0..2]`, `EastOut[0]`), each with
`put_count == get_count == MQ`; `test_M5_uses_branch` — the emitted text contains
`scf.if`, and the emitter source contains no `if tx ==` construct.

**FR-M6 — cascade plan for `R_space`.**
*Shall*: when `R_space ≠ {}`, the plan shall contain a chain of `PK-1` cascade channels
(`channel_type="npu_cascade"`, no `broadcast_shape`) along the contiguous PE line, with
per-stage `get`/accumulate/`put` and a head that skips the `get` and a tail that writes the
result out, all guarded by `ops.branch`.
*Rationale*: `npu_cascade` **is implemented** in `air.api` (`_channel.py:547`
`_IMPLEMENTED_TYPES = ("npu_cascade", "npu_dma_packet")`) and has a worked example
(`programming_examples/cascade_reduction/cascade_reduction.py`, whose docstring describes
exactly this head/middle/tail shape and says the discrimination must be `ops.branch`, not a
Python `if`). No MLIR-text fallback is needed.
*Source*: `_channel.py:547`, `:170-177`; `cascade_reduction.py:1-35`; SD-02 §5 R2, §6.
*Acceptance*: `test_M6_cascade_chain` — the W1-flip plan for `PK=4` contains **`PK−1 = 3`
cascade links on one `npu_cascade` bundle** and 4 guarded stages, with
`chain_direction == "ascending"` (measured: on a 1-D `grid(4)` herd the descending chain fails
`'aie.cascade_flow' op source tile must be to the North or West of the destination tile`,
REVIEW-round1 P-R3); and `ir_facts.cascade_channels == 3` — three `aie.cascade_flow` ops after
`air-to-aie`.

**FR-M7 — buffer plan.**
*Shall*: the plan shall list every buffer with name, memory level, `air.api` scope spelling
(`h.private()` / `seg.private()` / `seg.shared()` / `seg.per_core()`), shape, dtype, byte size
and the loop nesting depth at which it is allocated.
*Rationale*: the allocation scope decides whether a buffer is per-core, and `<herd>.shared()`
raises (`_trace.py:1405-1414`); the nesting depth decides whether ping-pong fires (VF §E.2).
*Source*: VF §D.2, §E.2.
*Acceptance*: `test_M7_buffer_plan` — the W1 plan's `acc` is `h.private()` at depth 0 and
`a`,`b` are `h.private()` at depth 1 (inside the K loop).

**FR-M8 — channel plan.**
*Shall*: every channel shall be planned with: symbol name, `size`, optional `broadcast_shape`,
optional `channel_type`, and an ordered list of put/get sites each carrying its indices, the
buffer or tensor region (offsets, sizes, strides), the enclosing scope, and whether it is
asynchronous.
*Rationale*: `air_ChannelPutOp` requires `static_src_offsets`/`_sizes`/`_strides`
(PC §3.0 floor step F2); planning them explicitly is what makes the emitter a translation
rather than a derivation.
*Source*: PC §3.0 F2; VF §D.3, §D.4.
*Acceptance*: `test_M8_channel_plan_complete` — for each of W1/W2/W3, every `ChannelPlan` has
every field populated and every site's region rank matches the buffer rank.

**FR-M9 — balance by construction, plus self-check.**
*Shall*: the mapper shall construct balanced protocols, and a **self-check** shall then verify
the finished `MappingPlan` against AIR's own P1 definition before emission: for each key
`(channel_name, indices)`, `put_count == get_count`, evaluated per loop body (per iteration),
per `scf.if` branch independently, and with the cross-iteration-space product rule for channels
joining scopes of different extent. **Broadcast channels take a modified rule** (see §6,
Decision D-4): a put at index `i` on a channel with `broadcast_shape` must be matched by
exactly one get at each index of `i`'s fan-out set.
*Rationale*: D5. mlir-air has only the unimplemented spec
(`docs/AIRCorrectnessChecker.md:15-20` — P1 channel balance, P2 deadlock freedom, P3 resource
constraints, P4 token-constraint consistency; §4.1 "For each channel index, along every
possible execution path, the number of `put` operations equals the number of `get`
operations"; §4.3 per-iteration and per-branch transfer rules), and
`mlir/lib/Analysis/` does not exist (VF §B.4). `air-opt` prints
`'air.channel.put' op found channel op not in pairs` **and exits 0** (VF §B.6, §S11), and
`AIRToAIEPass.cpp:4484-4491` silently *repairs* L2 imbalance with dummy ops (VF §S4).
*Source*: D5; `docs/AIRCorrectnessChecker.md:15-20`, §4; VF §B.4, §B.6, §S4, §S11.
*Acceptance*: `test_M9_selfcheck_accepts` — the three kernel plans pass;
`test_M9_selfcheck_rejects[...]` — three hand-corrupted plans (drop one get; add a put inside a loop body; guard a
put but not its get) are each rejected with `code="BALANCE"`, a per-key count table, and the
site locations, mirroring the spec's §4.4 error shape.

**FR-M10 — acyclicity by construction, plus self-check.**
*Shall*: the self-check shall build the put→get graph plus the intra-body program-order edges,
exclude loop-carried back edges, and reject any strongly connected component containing a
channel edge.
*Rationale*: D5. This is the spec's P2b verbatim (`docs/AIRCorrectnessChecker.md` §5.2, Tarjan
SCC over the extended graph, `StructuralBack` edges excluded — §5.3). Nothing in mlir-air
implements it: a `get`-before-`put` cycle across two herds is silent through `air-to-aie`,
`air-enforce-channel-fifo-order` and `air-verify-hierarchy-locality` (VF §B.6 row (iii)).
*Source*: D5; `docs/AIRCorrectnessChecker.md` §5.2–5.3; VF §B.6.
*Acceptance*: `test_M10_cycle_rejected` — a hand-built plan where PE `p` gets before putting on
both halo channels is rejected with `code="CHANNEL-CYCLE"` printing the cycle's sites;
`test_M10_w2_acyclic` — the real W2 plan has no such SCC.

**FR-M11 — mapping summary.**
*Shall*: every accepted schedule shall produce a human-readable summary naming, per operand,
its delivery (stationary / multicast along which PE axis / stream / cascade) and whether that
was **derived** or **declared**; plus the herd logical and physical shape, the reduction split,
the per-buffer L1 bytes and the total against the 65 536 budget, and the channel list with
sizes.
The summary shall additionally name, **per operand, on its own line**, the operand's *residency
duration*: how long its L1 tile stays put, either `resident for the whole run` or
`[resident across <axes>, ]re-fetched per <axis>`. The stationarity predicate FR-L3 checks is
*spatial* — no reuse crosses a PE — and is silent about time, so "stationary" alone cannot
distinguish an operand that is fetched once from one re-fetched every trip (RULING 9). The
duration is computed from `ker M_a` and the temporal tile axes in σ order
(`03-lld-M4-mapping.md` §3.9) and is carried in `MappingSummary.residency`
(`06-interfaces.md` §5.7, `CONTRACT_VERSION = 3`).
*Rationale*: "the reader of the program cannot tell output-stationary from weight-stationary
without re-deriving it from slice arithmetic" is the ARIES criticism this surface answers
(PC §1.4 item 2); the summary is what a judge reads in five minutes.
*Source*: PC §1.4; SD-01 §0.
*Acceptance*: `test_M11_summary_golden` — the summary text for W1 matches a stored golden file
byte for byte, and contains the literal strings `C: stationary (derived)`,
`A: multicast along py (derived)`, `B: multicast along px (derived)`;
`test_M11_residency_line` — W1 prints `C: stationary (spatial), resident for the whole run` and
`A: multicast along py, re-fetched per k0`, and the W1-flip prints
`B: stationary (spatial), resident for the whole run` and
`A: stationary (spatial), re-fetched per i0`.

**FR-M12 — plan determinism.**
*Shall*: channel names, buffer names and site ordering shall be a pure function of the
`KernelModel` and `ScheduleModel`; no dictionary-iteration order, no `id()`, no timestamps, no
random names.
*Source*: NFR-1.
*Acceptance*: `test_M12_plan_stable` — building the same plan twice in one process and once in
a fresh process with `PYTHONHASHSEED` varied yields equal plans.

### 3.4 Emission (M5)

**FR-E1 — hierarchy.**
*Shall*: emit `air.tensor(...)` for each L3 operand, `air.launch(name=)` around everything,
`air.segment(name=)` when any L2 buffer or any L3-endpoint channel op sits inside a herd, and
`air.herd(iterable, name=, shape=)` for the PE grid, with the herd body taking the coordinates
as positional arguments.
*Rationale*: the verified `air.api` idiom, e.g.
`worker_to_worker.py:73-108`; a segment is required when an L3 endpoint is inside a herd body
(`_channel.py`'s implemented check; PC §1.1 fact 4 — cite the code, not the docstring).
*Source*: VF §D.1, §D.4; `worker_to_worker.py:73-108`.
*Acceptance*: `test_E1_hierarchy` — the emitted W1 text contains `air.launch`, `air.segment`
and `air.herd` in that nesting, and `module.operation.verify()` succeeds.

**FR-E2 — native compute body.**
*Shall*: per-PE compute shall be emitted as `air.sequential` loops with fully-integer
subscripts on L1 buffers, producing `memref.load` / `memref.store` / `arith` in the herd body.
No `air.extern`, no `func.call`, and no `link_with` shall appear.
*Rationale*: D2. A fully-integer subscript is a rank-0 access that emits no loop of its own
(`_value.py:590-593`, `_emit.py:647-650`), and the tree's own test writes
`dst[i, j, 0] = src[i + 3, j, 1] + 1` under two `air.sequential` loops with FileCheck expecting
`scf.for`/`affine.apply` (`python/test/api/partial_assign.py:79-92`). Accumulation into the
same element also works (`partial_assign.py:120-127`). An extern kernel buys vectorisation, not
expressiveness (`_extern.py:11-14`) — and an opaque callee as a buffer's first toucher
**disqualifies ping-pong** (VF §E.2 items 3–4).
*Source*: D2; VF §D.11, §E.2.
*Acceptance*: `test_E2_native_body` — the W1 emitted text contains ≥ 3 nested `scf.for` inside
the herd body and `memref.load`/`memref.store`, and contains zero occurrences of `func.call`
and `link_with`.

**FR-E3 — ping-pong loop shape.**
*Shall*: for every operand marked `double_buffer`, the emitter shall place its L1
`air.alloc(...)` as a **direct child** of the streamed `air.sequential` loop body, with the
buffer's **first** touch being the `air.channel.get` that fills it, **at most one** get per
buffer per iteration, static trip counts on every intervening loop, and no opaque callee.
*Rationale*: D3. These are `isPingPongCandidate`'s conditions verbatim
(`AIRDependencyScheduleOpt.cpp:1604`, `:1620-1630`, `:1690`; `Transform/Passes.td:970-971`),
and the shape was **measured** to fire on an `air.api` GEMM with `a` and `b` allocated inside
the `air.sequential` K loop (VF §E.5).
*Source*: D3; VF §E.2, §E.5.
*Acceptance*: `test_E3_alloc_is_direct_child` — a regex over the emitted text asserts the
`memref.alloc` lines occur at the indentation level directly inside the K `scf.for`;
`test_E3_pingpong_fires` (integration, off-device) — `air-opt` with the labelling pipeline
reports `unroll = 2 : i32` on the K loop.

**FR-E4 — no `buffer_resources`.**
*Shall*: the emitter shall never pass `buffer_resources` to `air.api`.
*Rationale*: D3; `air.api` raises `NotImplementedError` on it (`_channel.py:563`, VF §D.4), and
the depth is realised anyway as the lock slot count set by the ping-pong unroll factor
(VF §E.3, §E.5 — "8 locks with `init = 2`").
*Source*: D3; VF §D.4, §E.3, §E.5.
*Acceptance*: `test_no_buffer_resources_arg` (shared with FR-S13).

**FR-E5 — broadcast channels.**
*Shall*: emit multicast operands as `air.channel(name, size=, broadcast_shape=)` per FR-M2,
with the put at segment scope and the get inside the herd body indexed by the herd coordinates.
*Rationale*: D4. Declaring `broadcast_shape` bypasses `air-broadcast-detection` entirely, which
only walks `air.dma_memcpy_nd` and never `air.channel.put/get`
(`AIRDependencyScheduleOpt.cpp:3328`, VF §E.4, §S9).
*Source*: D4; VF §E.4, §S9.
*Acceptance*: `test_E5_broadcast_emitted` (shared with FR-M2), plus `test_E5_detector_count` —
run `air-opt -air-broadcast-detection` on the emitted W1 and record the number of
`broadcast_pattern` attributes; assert it equals the number recorded in the golden file (see
R-04).

**FR-E6 — memory spaces.**
*Shall*: L1 buffers shall be `air.alloc(..., scope=h.private())`, per-core L1 buffers that must
outlive the herd body shall be `seg.per_core()`, L2 buffers `seg.private()`, and L3 operands
`air.tensor(...)`. `<herd>.shared()` shall never be emitted.
*Rationale*: `<herd>.shared()` raises (`_trace.py:1405-1414`); the four scopes and their
lifetimes are VF §D.2's table.
*Source*: VF §D.2.
*Acceptance*: `test_E6_memory_spaces` — the emitted text's memrefs carry `2 : i32` for L1 and
`1 : i32` for L2, matching the golden file.

**FR-E7 — AIR text output.**
*Shall*: `s.mlir()` shall return the AIR module as text via `LaunchContext.mlir()`
(`str(self.build())`, `_compile.py:242`) and shall write it to a caller-named path on
`s.emit(path)`.
*Source*: `_compile.py:106-158`, `:242`.
*Acceptance*: `test_E7_text_roundtrip` — the written file re-parses through
`air-opt` with exit code 0 and no `error:` on stderr (note FR-T5: the exit code alone is not a
verdict).

**FR-E8 — cascade emission.**
*Shall*: cascade channels shall be emitted with `channel_type="npu_cascade"`, without
`broadcast_shape`, on a herd whose grid is a row, and **without `at=` pinning** — `HerdPlan.at`
is `None` for every variant, so the emitter passes no `at=` (Q-1 measured; the field stays in the
contract as the documented escape hatch, R-03).
*Source*: FR-M6; `_channel.py:170-177`; `_trace.py:1771-1778`.
*Acceptance*: `test_E8_cascade_text` — the W1-flip emitted text contains
`channel_type = "npu_cascade"` on one bundle of `size=[PK−1] = [3]`, and no `broadcast_shape` on
it; `ir_facts.cascade_channels == 3` after `air-to-aie`.

**FR-E9 — the emitted module verifies.**
*Shall*: emission shall call `build()`, which runs `module.operation.verify()`
(`_compile.py:156-158`), and shall surface a verification failure as an `EmissionError` naming
the kernel and the MLIR diagnostic.
*Rationale*: `air.api`'s own docstring: "A DSL whose whole premise is that misuse raises should
not hand back IR it has never checked" (`_compile.py` `_verify`).
*Source*: `_compile.py:106-170`.
*Acceptance*: `test_E9_verify_surfaced` — a deliberately broken plan (segment coordinate used
inside an `IsolatedFromAbove` herd body) raises `EmissionError`, not a bare MLIR exception.

**FR-E10 — deterministic text.**
*Shall*: emitting the same `(KernelModel, ScheduleModel)` twice shall produce byte-identical
AIR text.
*Source*: NFR-1.
*Acceptance*: `test_E10_byte_identical` — two emissions in one process and one in a fresh
process with a different `PYTHONHASHSEED` compare equal.

### 3.5 Toolchain and runtime (M6)

**FR-T1 — build targets.**
*Shall*: `s.build(target=...)` shall accept `"npu1"`, `"npu2"` and `"auto"` and shall pass the
value straight to `LaunchContext.build(target=)`. `"auto"` shall be documented as probing
`xrt-smi` and falling back to `npu2` with no device.
*Source*: D7; VF §F.3 (`_trace.py:164-183`, `:96`, `:182`).
*Acceptance*: `test_T1_targets` — `build(target="npu1")` and `build(target="npu2")` both
succeed on a machine with no NPU; `build(target="xcvc1902")` raises `ValueError` from `air.api`
and is re-raised as `ToolchainError` naming the two accepted values.

**FR-T2 — off-device artifacts.**
*Shall*: the driver shall run `aircc --device <target> --output-format=none|pdi <module.mlir>`
and shall treat both as success criteria for the off-device path.
*Rationale*: measured — `--output-format=none` and `pdi` both exit 0 on an NPU-free machine
(VF §G.5); `matrix_multiplication/i8/run.py:573` does exactly this.
*Source*: D7; VF §G.5.
*Acceptance*: `test_T2_aircc_none` (integration, marked slow) — W1's module through
`aircc --device npu1 --output-format=none` exits 0.

**FR-T3 — xclbin requires XRT.**
*Shall*: requesting `output_format="xclbin"` without `xclbinutil` on `PATH` shall fail with a
`ToolchainError` that names `xclbinutil`, says XRT provides it, and points at
`--output-format=none|pdi`.
*Rationale*: measured — `--output-format=xclbin` fails at step 40/41 with
`tool 'xclbinutil' not found in search paths or PATH` (VF §G.5).
*Source*: D7; VF §G.5.
*Acceptance*: `test_T3_xclbin_message` — with `xclbinutil` absent, the error text contains all
three parts. (Skipped when XRT is present.)

**FR-T4 — device run and oracle diff.**
*Shall*: when a device is present, the driver shall compile with `output_format="xclbin"`, run
the kernel on the device fixtures, and diff against the CPython oracle with an explicit
tolerance per dtype (exact for integer fixtures; a stated absolute tolerance for float).
*Source*: D7; G4.
*Acceptance*: `test_T4_device_diff` — marked `requires_device`, skipped otherwise; asserts the
diff is within tolerance and prints max absolute error.

**FR-T5 — never trust `air-opt`'s exit code.**
*Shall*: every invocation of `air-opt` or `aircc` shall capture stderr and shall treat any line
matching `error:` as a failure **regardless of exit status**.
*Rationale*: `air-opt` prints `error: 'air.channel.put' op found channel op not in pairs` and
**exits 0** (VF §B.6, §S11) — the emitting site calls `emitOpError` without
`signalPassFailure()` (`mlir/lib/Util/Dependency.cpp:2063-2066`).
*Source*: D5; VF §B.6, §S11.
*Acceptance*: `test_T5_error_without_exit_code` — feed the driver a module known to produce that
diagnostic (an unmatched put) and assert the driver reports failure even though the process
exit code is 0.

**FR-T6 — pinned toolchain.**
*Shall*: the project shall pin `mlir_air[aie]` to the wheel version
`0.0.1.2026091204+ff95a9b` (commit `ff95a9b`) with `mlir-aie 1.4.3.dev55+g10767b5` and
`llvm-aie 22.0.0.2026091201+386ca5c6`, installed from the three GitHub release index URLs, and
shall record the exact recipe in `07-environment.md`.
*Rationale*: D7; the wheel is built from the exact commit every fact here cites (VF §G.3), and
`pruneAIRReleaseAssets.yml:95-97` prunes release assets — so a `latest-*` tag is not
reproducible (VF "Install path" notes).
*Source*: D7; VF §G.2, §G.3, "Install path".
*Acceptance*: `test_T6_versions` — a test asserts `air.__version__`-equivalent metadata (from
`importlib.metadata.version("mlir_air")`) matches the pin, and fails loudly with the upgrade
instruction if not.

### 3.6 Diagnostics (M3 catalogue, used by M1/M2/M4/M5/M6)

**FR-D1 — every rejection is structured.**
*Shall*: every user-facing rejection shall be raised as a subclass of `SpatialError` carrying a
`Diagnostic` with fields `code`, `stage`, `clause`, `reason`, `fix`, `location` (source file and
line where available), and `details` (a mapping of the concrete numbers involved).
*Source*: G2; brief §5.
*Acceptance*: `test_D1_schema` — a parametrised test over every negative test in the suite
asserts every field is populated and `code` is in the catalogue.

**FR-D2 — accepted schedules produce a summary.**
*Shall*: see FR-M11.

**FR-D3 — stable error-code catalogue.**
*Shall*: a single table in `06-interfaces.md` shall list every `code`, its stage, and one
example message. A code shall never be reused for a different condition.
*Rationale*: the negative test suite keys on codes, not on message prose, so messages can be
improved without breaking tests.
*Acceptance*: `test_D3_catalogue_complete` — every code raised anywhere in the package appears
in the table, and every table entry has at least one test that raises it.

### 3.7 Kernels (M8)

**FR-K1 — W1 GEMM output-stationary (required).**
*Shall*: ship `gemm` (three nested loops, `C[i,j] += A[i,k]*B[k,j]`) with the output-stationary
schedule of PC §3.1, a small CPU fixture, the golden AIR text, and a passing oracle diff.
*Source*: D8; PC §3.1; SD-02 §9.
*Acceptance*: `test_W1_end_to_end` — oracle matches numpy; emitted text matches golden;
`aircc --output-format=none` exits 0.

**FR-K2 — W1 weight-stationary flip (required stretch).**
*Shall*: ship the same `gemm` kernel text with `grid(PK)`, `tile(ax.i, 32)`, `tile(ax.k, 16)`,
**`j` left untiled** (`TN = N = 64`), `reduce(ax.k, "+")`, `place(px=ax.k0)`, `stationary("B")`,
`reside(A="L1", B="L1", C="L1")` and `double_buffer("A")` **only** — a **1-D** herd of `PK = 4`
— producing `R_space = span{e_k}` and a cascade reduction along the PE row, with **no edit to
the kernel**. `j` is untiled *so that the flip is genuinely weight-stationary* (RULING 9): each
PE's `B` tile is `B[kchunk, :]` = `[TK, N]` = `[16, 64]` and is constant over the whole temporal
sweep, so the weights are fetched once and stay; `A`'s `[32,16]` tile changes with `i0` and is
re-fetched on each of the two `i0` trips, which is what makes it the ping-pong candidate.
`double_buffer("B")` would be a `PINGPONG-SHAPE` rejection (`03-lld-M3-checker.md` §3.13
condition 2: a resident tile has no streaming loop to be allocated inside).
*Rationale*: D8; "Flip one pragma … and the *same algorithm* re-lowers to `R_space = span{e_k}`,
a **cascade reduction** down a PE column … No kernel edit" (SD-02 §9).
*Dependency*: `npu_cascade` through `air.api`. **Resolved available**:
`_IMPLEMENTED_TYPES = ("npu_cascade", "npu_dma_packet")` (`_channel.py:547`), with the worked
example `programming_examples/cascade_reduction/cascade_reduction.py`. No MLIR-text fallback is
required. *Contingency if the cascade lowering fails off-device*: emit the same head/middle/tail
chain on the **default `npu_dma_stream`** channel type — identical protocol, identical balance,
different physical link — owned by **Person B**, decision point **D6**.
*Acceptance*: `test_W1_flip` — the same kernel source with the flipped schedule produces a plan
with `R_space = span{e_k}`, `PK−1 = 3` cascade **links** on one `npu_cascade` bundle for `PK=4`,
an **ascending** chain (`chain_direction == "ascending"`), and text that
`aircc --output-format=none` accepts; **and the summary asserts both residency facts** — the
lines `B: stationary (spatial), resident for the whole run` and
`A: stationary (spatial), re-fetched per i0` (FR-M11).

**FR-K3 — W2 Jacobi 5-point with halo exchange (required).**
*Shall*: ship `jacobi(U: sp.f32[T + 1, H + 2, W])` — one rank-3 parameter, the 0.2 × 5-point
update, write domain planes `1..T` × rows `1..H` × cols `1..W-2`, with rows `0`/`H+1` and cols
`0`/`W-1` read-only Dirichlet boundary — scheduled with `grid(PI)`, `tile(ax.i, HS)`,
`place(px=ax.i0)`, `sequential(ax.t)`, `window(U, dims=(ax.i, ax.j), halo=1)`,
`exchange(U, along=ax.i0, halo=1)`, `reside(U="L1")`, `double_buffer("U")`. The fixture is
`H = W = 16`, `PI = 2`, `HS = 8` (invariant `PI·HS == H`), `T = 4` with an odd-`T` variant
`T = 5`. **Every plane the kernel writes is drained**: one `UOut` put of the computed strip per
PE per timestep, so the L3 image equals the oracle over the whole write domain
(`04-test-plan.md` §3.4 `test_sem_coverage`).
*Source*: D8; PC §3.2 W2·P1; REVIEW-round1 RULING 1.
*Acceptance*: `test_W2_end_to_end` — the oracle matches a two-loop numpy Jacobi over the whole
write domain within `1e-5`; plus `test_W2_oracle_is_timestep_outermost` — asserts the kernel's
own outermost loop is `t`, so the P5 lockstep problem measured in VF §I does not arise for this
surface (the un-annotated nest is already timestep-outermost).

**FR-K4 — W3 Smith-Waterman wavefront (required).**
*Shall*: ship `sw(q: sp.i32[MQ], r: sp.i32[NR], S: sp.i32[MQ + 1, NR + 1])` — the substitution
score computed from `q` and `r` by `sub = MATCH if q[i-1] == r[j-1] else MISMATCH` (FR-S3 item 8)
and `S[i,j] = max(0, S[i-1,j-1] + sub, S[i-1,j] - GAP, S[i,j-1] - GAP)`, with row 0 and column 0
of `S` a zero-initialised read-only boundary — scheduled with `grid(PJ)`, `tile(ax.j, CW)`,
`place(px=ax.j0)`, `skew(time=(ax.i, ax.j0))`, `forward(S, along=ax.j0, dir="W->E")`,
`reside(S="L1")`. The fixture is `MQ = NR = 32`, `PJ = 4`, `CW = 8`, `MATCH = +2`,
`MISMATCH = -1`, `GAP = 1`. **Every row of `S` is drained** (one `SOut` put per row per PE), so
the L3 image equals the oracle over the whole write domain.
*Source*: D8; PC §3.3 W3·P1; REVIEW-round1 RULING 2.
*Acceptance*: `test_W3_end_to_end` — the oracle matches a textbook two-loop DP in numpy exactly;
plus the negative `test_L2_causality` above, which is the demo's headline rejection.

**FR-K5 — W4 FFT status.**
*Shall*: W4 is **out of scope** unless W1–W3 all pass on device. The documentation shall state
why: the stage-parameterised partner map `p ↦ p ⊕ 2^s` is affine in neither `p` nor `s`, so the
surface cannot state it without an escape hatch.
*Source*: D8; PC §2 W4.
*Acceptance*: `test_W4_absent` is not a test; the acceptance is a line in the honest-limits
slide and in `00-README.md`'s status table.

---

## 4. Non-functional requirements

**NFR-1 — determinism.** The same `(kernel source, schedule)` shall produce byte-identical AIR
text and a byte-identical mapping summary across processes and across `PYTHONHASHSEED` values.
*Acceptance*: FR-E10, FR-M12.

**NFR-2 — pinned runtime dependencies.** Exactly four third-party runtime dependencies:
`mlir_air[aie]` (pinned, FR-T6), which brings `mlir_aie` and `llvm-aie`, and `numpy`. No other
dependency shall be added without all three owners agreeing.
*Acceptance*: `test_NFR2_deps` — the package's declared dependencies equal the pinned list.

**NFR-3 — CPU test runtime.** The default `pytest` run (unit + integration, excluding the
`slow` and `requires_device` marks) shall complete in **under 3 minutes** on a laptop CPU.
Fixtures shall be sized to that budget (§4 of the test plan).
*Acceptance*: CI records wall-clock time and fails the build over 3 minutes.

**NFR-4 — error messages.** Every user-facing error shall name the clause or construct, the
concrete numbers involved, and one concrete fix. No error shall surface a raw MLIR diagnostic or
a Python traceback into the top-level message.
*Acceptance*: FR-D1's schema test plus a lint test asserting no `raise` of a bare `Exception`,
`AssertionError` or `RuntimeError` in the package's user-facing paths.

**NFR-5 — documentation.** Every public entry point in `06-interfaces.md` shall have a
docstring; the three kernels shall each have a runnable script that prints the mapping summary.
*Acceptance*: `test_NFR5_docstrings` — every name in the frozen public API has a non-empty
`__doc__`.

**NFR-6 — no network at test time.** The test suite shall not reach the network. The toolchain
is installed once, at day 0.
*Acceptance*: CI runs with networking disabled after the install step.

**NFR-7 — the checker never crashes on user input.** Any input that the grammar accepts shall
produce either a `MappingPlan` or a `SpatialError` — never an unhandled exception.
*Acceptance*: a smoke test that runs the full negative corpus and asserts every failure is a
`SpatialError`.

---

## 5. Assumptions

Each assumption is linked to the fact that justifies it. If a fact is wrong, the linked
requirement is void.

| # | Assumption | Source | If false |
|---|---|---|---|
| A-1 | A herd body may contain an arbitrary scalar loop nest over L1 buffers, emitting `memref.load`/`store` | VF §D.11 (`partial_assign.py:79-92`, `_value.py:590-593`) | FR-E2 is void; compute must go through `ops.dot`/elementwise or `air.extern`, and ping-pong is then at risk (VF §E.2 item 3) |
| A-2 | A put into an empty depth-1 channel does not rendezvous | VF §C.7 (measured lock inits), §C.3, §C.5 | FR-M4's put-first halo protocol is void; W2 leaves the demo |
| A-3 | Nothing in mlir-air enforces put/get balance or channel acyclicity | VF §B.1–B.6 | FR-M9/M10's self-check becomes redundant, and G2 loses its differentiator |
| A-4 | The ping-pong passes are in the default `aircc` pipeline and fire on an `air.api`-emitted loop of the required shape | VF §E.1, §E.5 (measured) | FR-S13/E3 become unverifiable assertions; `double_buffer` becomes a pure hint |
| A-5 | `air.api` accepts `channel_type="npu_cascade"` | `_channel.py:547`; `cascade_reduction.py` | FR-K2 falls back to the `npu_dma_stream` chain (its stated contingency) |
| A-6 | `build()` needs no device, no XRT and no aircc — it traces, verifies, and returns a module | VF §D.9 (`_compile.py:128-158`; zero `PassManager` hits) | the whole CI plan changes; emission tests need a device |
| A-7 | `aircc --output-format=none\|pdi` works with no XRT and no device | VF §G.5 (measured) | FR-T2 is void; off-device integration stops at AIR text |
| A-8 | `air.herd` is 1-D or 2-D, with per-target physical caps and exact strip-mining | VF §D.1 (`_trace.py:88-91`, `:1288-1291`, `:1345-1372`) | FR-L10/L11 change |
| A-9 | L1 is 65 536 bytes per core as `air.api` itself budgets | `_trace.py:100` | FR-L9's constant changes |
| A-10 | `air-broadcast-detection` never sees our channel ops | VF §E.4, §S9 (`AIRDependencyScheduleOpt.cpp:3328`) | derived multicast may fire unasked on our `ops.load`s; R-04 |
| A-11 | The install is six minutes from pip wheels on any team laptop | VF §G.3 (measured, 5 min 43 s) | D0 grows; the whole plan slips |
| A-12 | `air-opt` may print `error:` and exit 0 | VF §B.6, §S11 | FR-T5 is over-cautious but harmless |
| A-13 | Team size is 3, and D0 is 2026-09-12 with final evaluation 2026-09-19/20 | Architect brief D10, D11; `CLAUDE.md` | §5 of the work-breakdown re-plans |

---

## 6. Risk register

Likelihood/Impact: L = low, M = medium, H = high. Every must-not-surprise item from the brief
§5 appears here. Owner is the person who acts; trigger is the observable that fires the
mitigation.

| # | Risk | L | I | Mitigation | Owner | Trigger |
|---|---|---|---|---|---|---|
| R-01 | **Compute-body emission** — the native scalar loop nest does not lower for some access shape we need (e.g. a subscript the tracer will not take) | M | H | D2 fixes native emission and VF §D.11 measures it on a shifted-window access; the fallback (`ops.dot` + `convert-linalg-to-loops`) is documented in §9 but costs ping-pong (VF §E.2) | B | any W1/W2/W3 body fails to trace by D2 |
| R-02 | **L1 capacity** — a fixture that looks small overflows once ping-pong doubles the tiles | M | M | FR-L9 charges the doubled figure; fixtures are sized in the test plan; `_annotate_l1_failure` already explains it | A (check), C (fixtures) | `test_L9_capacity` or an `aircc` "exceeded available memory" |
| R-03 | **Cascade for `R_space`** — the cascade lowering rejects our chain shape (adjacency, pinning) | M | M | FR-M6 restricts to a contiguous line, without pinning (Q-1 measured: `air-place-herds` finds the line, and `at=` would only remove its freedom); `at=` stays available as the escape hatch if a device run shows a bad placement; contingency is the `npu_dma_stream` chain (FR-K2) | B | W1-flip `aircc` failure at D6 |
| R-04 | **`air-broadcast-detection` fires unasked** on our remaining `ops.load`s and interacts with our declared broadcast channels | M | M | Emit multicast operands as explicit channels (D4, FR-E5), which the detector never walks (VF §S9); pin the detector's behaviour with `test_E5_detector_count` against a golden count | B | the golden count changes |
| R-05 | **Ping-pong declines silently** — one of `isPingPongCandidate`'s eight conditions fails and `double_buffer` becomes a no-op with no diagnostic | M | M | FR-S13 makes the checker assert the shape, and `test_E3_pingpong_fires` inspects the post-`-air-dependency` IR for `unroll = 2` | A (check), C (IR test) | the IR test finds no `unroll` attribute |
| R-06 | **No balance/acyclicity verifier in AIR** — an emitted program is wrong and nothing downstream says so | H (certain) | H | FR-M9/M10: balanced and acyclic **by construction** plus a self-check against the spec's P1/P2 before emission; FR-T5 parses diagnostics rather than exit codes | B (construction), A (checker) | `test_M9`/`test_M10` |
| R-07 | **Channel lowering depth / rendezvous** — the put-first halo protocol stalls on real hardware at `T` timesteps even though E1 is resolved for one exchange | L | H | VF §C's residual-risk note is BD/lock **capacity**, not rendezvous; keep `T` small in the demo fixture; measure on device at D5 | C | a device run of W2 hangs |
| R-08 | **Python-subset grammar** — a kernel we want to demo needs a construct the grammar rejects | M | M | D9 fixes the grammar now; W1/W2/W3 sources are written against it at D0 and parsed by a stub parser before M1 exists | A | any of the three sources fails to parse |
| R-09 | **Device availability / late arrival** | H | M | Every requirement above is satisfiable off-device (A-6, A-7); device tests are marked `requires_device` and skipped; the demo's primary artifact is AIR text + `aircc --output-format=none` | C | no device by D5 |
| R-10 | **mlir-air build time** | L | L | Retired: the install is a pip wheel, measured at 5 min 43 s, no source build (VF §G.3) | C | the wheel index 404s |
| R-11 | **XRT on team machines** | M | L | Only `--output-format=xclbin` needs `xclbinutil`; `none`/`pdi` work without XRT (VF §G.5); FR-T3 makes the failure legible | C | `test_T3` fires on a machine that was believed to have XRT |
| R-12 | **Error-message quality** — judges see a rejection that reads as a crash | M | H | FR-D1's schema, FR-D3's catalogue, and a negative test per FR that asserts all four message parts | A | `test_D1_schema` |
| R-13 | **Wheel index pruning** — `latest-air-wheels` loses the asset mid-week and the team cannot reinstall | M | H | Pin a `v*.*.*` tag (VF Install-path note); **cache the four wheels locally at D0** and record their sha256 in `07-environment.md` | C | a reinstall on D3+ resolves a different version |
| R-14 | **Versal creep** — someone tries `target="xcvc1902"` in the demo | L | L | FR-T1 rejects it with the two accepted values; §2 records it as a non-goal | C | — |
| R-15 | **Interface churn** — a frozen dataclass changes mid-week and two people's work diverges | M | H | `06-interfaces.md` is frozen at D0; changes need all three signatures and a version bump recorded in `00-README.md` | all | any PR touching `model.py` |
| R-16 | **Scope creep into performance** — a reviewer asks for numbers and the team starts optimising | M | M | §2 non-goals; the honest-limits slide says so out loud; `air-runner` is named as a timing model, not an oracle (VF §S7) | C | any commit adding a benchmark |
| R-17 | **Demo depends on a 3-person integration that lands late** | M | H | Build order and stubs are fixed in the HLD §8; every module is testable against a stub from D1 | all | a D3 checkpoint slips |
| R-18 | **Per-core circuit-switched DMA budget** — a core needs more than 2 inbound (S2MM) or 2 outbound circuit-switched channels and `aie.connect` fails late, inside `aircc` | H (measured at `PI=4`) | H | `03-lld-M4-mapping.md` §3.8's `DMA-CHANNELS` check rejects **before** emission, counting circuit-switched core-to-core and non-packet L3 endpoints only; W2's fixture is `PI=2` and `PI=4` is a negative test | B | any `'aie.connect' op … targets same dst` in an `aircc` smoke run |
| R-19 | **Silent MM2S merge at > 2 outbound endpoints** — `air-to-aie` merges two outbound streams onto one shim MM2S channel with no diagnostic, and whether both destinations receive correct data is **unverifiable off-device** | M | H | Keep every core at ≤ 2 outbound endpoints (W1, W1-flip, W3 and W2 at `PI=2` all are); `DMA-CHANNELS` warns at 3; the honest-limits slide names it (B-O1) | B (analysis), C (device) | a device run, or a third outbound endpoint appearing in any plan |
| R-20 | **Channel bundles may not mix L3 and core-to-core members** — a single `PJ+1` bundle for W3's wavefront fails to lower | M (measured) | M | FR-M5's three homogeneous channels (`WestIn`/`West`/`EastOut`); `AIRLoweringPass.cpp:798` is the failure and the reason | B | `failed to specialize channel bundle indices` |
| R-21 | **`npu2` goldens are unbuilt** — four of the eight golden modules target a device no probe has touched; every shape in this set was proved on `npu1` only | M | M | Run all four variants through `aircc --device npu2 --output-format=none` at D2 and record; `npu2`'s physical caps are strictly larger (`_trace.py:88-91`), so `npu1` passing is the binding case, but that is an argument, not a measurement (B-O4) | B | D2's `npu2` smoke |

---

## 7. Open questions

None are unowned. Each is assigned with a due day; **an unanswered question at its due day
becomes a risk with the stated fallback.**

| # | Question | Owner | Due | Fallback if unanswered |
|---|---|---|---|---|
| Q-1 | Does the cascade chain of FR-M6 survive `aircc --output-format=none` on `npu1` with a 4-wide row herd, and does it need `at=` pinning? | B | D2 | **Answered (P-R3/P-R4, `03-lld-B-open-questions.md` §4):** it survives — exit 0, zero `error:` lines — and **`at=` pinning is not needed**; the chain's direction is fixed by `aie.cascade_flow`'s verifier (ascending on the 1-D herd), carried by `ChannelPlan.chain_direction`. FR-K2's `npu_dma_stream` contingency is not taken |
| Q-2 | Does the W2 halo channel bundle need `size=[PI-1]` (one index per link) or `size=[PI]` with a guarded boundary? Fix the convention and write it into the plan. | B | D1 | `size=[PI-1]` with both boundary PEs guarded by `ops.branch`, which is the shape FR-M5 already requires for W3 |
| Q-3 | Is the golden AIR text stable across a `--upgrade` of the wheel within the pinned version? (Governs whether goldens are asserted byte-for-byte or by FileCheck-style patterns.) | C | D1 | Byte-for-byte for the pinned wheel, with a single `--update-goldens` switch and a CI guard on the pin |
| Q-4 | What absolute tolerance does the device diff use for `bf16` W1? | C | D3 | Integer fixtures only for the device diff; float fixtures stay on the oracle path |
| Q-5 | Does `test_E3_pingpong_fires` need the full `aircc` pipeline prefix, or does `-air-dependency,-air-dma-to-channel,-air-label-scf-for-to-ping-pong` suffice as in VF §E.5? | C | D2 | Use VF §E.5's exact pipeline string, which is recorded as having worked |
| Q-6 | Which of `seg.per_core()` vs `h.private()` does the W2 strip buffer need, given it must live across the `t` loop? | B | D1 | `h.private()` with the `t` loop **inside** the herd body (the herd body is entered once), matching `worker_to_worker.py`'s shape |
| Q-7 | Does a segment-scope loop of `K/TK` puts, issued *before* the `air.herd` op that consumes them one per iteration, make progress — or must the herd sit **inside** the producer loop as in `programming_examples/matrix_multiplication/bf16/run.py:161-165`? | B | D1 | **Answered (P-R1, re-measured):** the producer-loop-before-herd shape builds (127 lines), verifies and `aircc`s at exit 0 with zero `error:` lines. Kept as a record; no fallback needed |
| B-O1 | Does the N-3 outbound MM2S merge deliver correct data to both destinations? Only a device run settles it (R-19) | B (analysis), C (device) | D5 | Keep every core at ≤ 2 outbound endpoints; `DMA-CHANNELS` warns at 3 |
| B-O4 | Does everything proved on `npu1` also hold for `--device npu2`? Only `npu1` was probed (R-21) | B | D2 | Run the four modules through `aircc --device npu2 --output-format=none` at D2 and record. `npu2`'s caps are larger (`_trace.py:88-91`), so `npu1` is the binding case |
| B-O5 | Does `stationary("B")` on W1-flip assert temporal residency rather than `ker M_B ⊆ ker Sπ`? | A (check), B (plan) | D2 | **Closed by RULING 4**: under the 1-D `grid(PK)`, `place(px=ax.k0)` form, `ker M_B = span{e_i} ⊆ ker Sπ_u = span{e_i, e_j}` holds, so the declared clause and the derived classification agree and FR-L3 accepts. **RULING 9** keeps that predicate and answers the other half of the question separately: temporal residency is *reported*, not asserted, as `MappingSummary.residency`, and the flip fixture leaves `j` untiled so `B` is in fact resident for the whole run |
| B-O6 | `DMA-CHANNELS` needs adding to the frozen `06-interfaces.md` §6.3 | B | D1 | **Closed**: the code landed in `06-interfaces.md` v2 before the D0 freeze (REVIEW-round1 RULING 5). No `PROTOCOL-UNSUPPORTED` fallback remains |
| B-O7 | Is a `PI ≥ 3` halo reachable at all on `npu1` — via `npu_dma_packet` channels or a seeding wave? | B | after the freeze | Out of scope for the week. `PI = 2` is the fixture, `PI = 4` is a negative test, and the honest-limits slide names the limit and the two candidates |
| Q-C4 | Is `--placed-ir-verifiers=error` (the `aircc` default) ever what fails one of our four variants? | C | D2 | Fix the plan, never lower the verifier to `warn` — silencing one of the two real upstream checks would undo G2 (`03-lld-M6-toolchain.md` §8) |
| Q-C8 | Is the 180 s budget achievable once all eight golden modules exist? | C | D3 | Build each module once in a session-scoped fixture so the three assertions share it, cutting eight builds to four (`03-lld-M7-tests.md` §8) |
| Q-C15 | Does `skew(time=…)` replace `σ` outright or prefix a default `σ`? | A | D4 | **Answered by RULING 6**: it defines the leading row and the remaining rows come from loop order excluding placed **and** skewed axes. The headline rejection survives (FR-L2) |
| Q-C16 | Does the D-4 override make `w2_odd` accept-and-peel, retiring the old `test_L14_swap_parity` expectation? | A + architect | D5 | **Answered**: `T = 5` is accepted and peeled; `T = 0` is the `SWAP-PARITY` negative (`03-lld-M8-kernels-demo.md` §4, fixture `w2_zero_t`) |
| Q-C17 | Is the PACT 2026 attribution confirmed by the user? | C | rehearsal #2 | Unconfirmed ⇒ the sentence is deleted from the deck, not hedged |

---

## 8. Traceability

Every FR above names at least one acceptance test in its **Acceptance** clause, and
`04-test-plan.md` §7 carries the FR → test → owner table. `test_traceability` asserts that every
FR id appears in **at least one** test's `@pytest.mark.fr` marker and that no marker names an FR
that does not exist — three tests legitimately carry two FR ids (`03-lld-M7-tests.md` §6.3), so
"exactly one" would fail on a correct suite.

Counts: **70 functional requirements** (S: 20, L: 14, M: 12, E: 10, T: 6, D: 3 — note FR-D2
delegates to FR-M11 — K: 5), **7 non-functional requirements**, **13 assumptions**, **21 risks**,
**18 open questions**, all assigned. Every FR has an acceptance test; **70 of 70** — FR-D2
delegates to FR-M11's `test_M11_summary`, and FR-K5 is covered by `test_K5_scope_documented`
(`03-lld-M7-tests.md` §7.7), which asserts the honest-limits text exists rather than asserting
the absence of a test.

Every test id in this document is written on one line, because `test_traceability`'s parser
reads them out of this file line by line.

## 9. Future work (named, so the pitch can be honest about it)

1. **Vectorised compute** via `air.extern` and a hand-written `.cc` per kernel — buys
   vectorisation, costs ping-pong on any buffer the extern call touches first (VF §D.11,
   §E.2).
2. **Composition and fusion** (`SD-05`): the graph-level vocabulary above the per-kernel
   schedule.
3. **Solving for `(σ,π)`** instead of checking it — the LP/ILP of SD-02 §8 item 2.
4. **W4 FFT** and the escape hatch for non-affine partner maps.
5. **Versal** through `aircc --device xcvc1902` on a module built by hand or through
   `air.dialects`, bypassing `LaunchContext.compile` (VF §F.3, E5 status).
6. **Upstreaming** the balance/acyclicity checker as `-air-verify-program`, which AMD has
   specified and not built (`docs/AIRCorrectnessChecker.md`; VF §S3). The pitch must say the
   *idea* is upstream and public; only the implementation is ours.
