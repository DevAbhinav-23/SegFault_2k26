# Design review, round 1 — adversarial pass over `design/`

*2026-09-12. Reviewer: design-review agent. Scope: every file in `design/` plus
`hackathon/VERIFIED-AIR-FACTS.md`. Method: cross-document consistency, interface conformance,
pipeline trace per workload, plan realism, requirements hygiene, and re-verification of
`path:line` citations against the mlir-air clone at `ff95a9b` plus four new probes run this
session from the scratchpad venv (`mlir_air 0.0.1.2026091204+ff95a9b`).*

**Counts: 13 blockers, 29 drift rows, 26 gaps, 8 plan risks, 62 edit-list items.**
Nothing outside this file was modified.

**New measurements made during this review** (probe sources in
`$SCRATCH/air-probe/review/`, nothing written into the repo):

| # | Probe | Result |
|---|---|---|
| **P-R1** | `q7_a.py` re-run → `aircc --device npu1 --output-format=none` | exit 0, **0** `error:` lines, 127-line module. Q-7 **confirmed** |
| **P-R2** | `q2_w2.py` (W2, `PI=4`) → `aircc` | exit 1, `'aie.connect' op … TileID(1, 2) targets same dst`. N-1 **confirmed** |
| **P-R3** | **new**: W1-flip on a **1-D** `grid(PK)` herd, both chain directions | **ascending** (head `tx=0`) → exit 0, `aie.cascade_flow(%tile_0_2,%tile_1_2)`, `(1,2)→(2,2)`, `(2,2)→(3,2)`. **descending** → exit 1, `source tile must be to the North or West`. **Opposite of the 2-D case B probed** |
| **P-R4** | **new**: `w3b.py` + `q`/`r` L3→L1 staging (3 logical inbound channels/core) | (a) `_check_interface` raised *"output tensors must be declared after all input tensors"*; after reordering, **exit 0**. `aie.q7a.mlir` has **3 `aie.packet_flow`** and `air_A2L1_0`/`air_B2L1_0`/`air_B2L1_1` all on `shim_noc_tile_0_0, MM2S, 0`; `aie.w2pi4.mlir` has **0 packet flows**. The DMA budget binds on **circuit-switched** flows only |

---

## 1. BLOCKERS

Each: file:section → verbatim sentence → evidence → exact edit.

### B-1 — W2's emitted program is not W2's kernel

`03-lld-M5-emitter.md` §6.3 line 1: *"`U := air.tensor([18, 16], f32);  Uout := air.tensor([18, 16], f32)`"*.
`03-lld-M1-frontend.md` §6.2: *"`def jacobi(U: sp.f32[T1, H, W])`"* — **one** rank-3 parameter whose
write domain is planes `1…T`. `06-interfaces.md` §5.6 makes `MappingPlan.tensors` *"the L3
interface"*, and the L3 interface is `KernelModel.params`. Two rank-2 tensors named `U`/`Uout`
are neither. Consequences: `test_sem_coverage` (`04-test-plan.md` §3.4 — *"the union over PEs of
the written index sets equals the kernel's full write domain"*) fails by construction, because
the plan drains only the final strip; `m6.diff` has no array to compare; `test_W2_end_to_end`
cannot pass. R-B adds a **third** shape (`U[T+1, H+2, W]`).
**Edit**: decide the W2 tensor contract once (see EDIT-11) and rewrite `03-lld-M4-mapping.md`
§6.3, `03-lld-M5-emitter.md` §6.3 and `03-lld-M1-frontend.md` §6.2 against it.

### B-2 — W3 exists as three mutually incompatible kernels

`03-lld-M1-frontend.md` §6.3: `sw(Sub: sp.i32[MQ1, NR1], S: sp.i32[MQ1, NR1])`.
`03-lld-M8-kernels-demo.md` §3.4: `sw(q: sp.i32[MQ], r: sp.i32[NR], S: sp.i32[MQ+1, NR+1])`.
`03-lld-M5-emitter.md` §6.4 line 1: *"`Zrow := air.tensor([33], i32);  Sink := air.tensor([33], i32);  Out := air.tensor([4, 8], i32)`"* and line r5:
*"`c[j] = air.ops.maximum(p[j-1] + 2, air.ops.maximum(p[j] - 1, c[j-1] - 1))`"* — the
substitution score is a **literal `+2`**, `Sub`/`q`/`r` are never read, `GAP` never appears, and
`Out` is `[PJ, CW] = 4×8` where the kernel writes `S` as `33×33`. The probe this is copied from
(`$PROBE/w3b.py`) has the same three properties, so "W3 compiles" is a statement about a
different program. `test_W3_end_to_end`'s *"oracle exactly matches the textbook DP"* is
unreachable.
**Edit**: adopt M8 §3.4's source as canonical, then rewrite M1 §6.3, M4 §6.4 and M5 §6.4 to
stage `q`/`r` into L1, compute `sub` from them, and drain all of `S`.

### B-3 — `ComputeNode` cannot express any of the four kernels; M5 must decide

`06-interfaces.md` §5.5: *"`ComputeNode`: `(statement_index: int, subscripts: tuple[Expr, ...], op: str)` — a single scalar `memref.load`/`arith`/`memref.store` triple"*.
Four things M5 emits that this cannot carry:
1. W2's `dst[i,j] = (src[i-1,j]+src[i+1,j]+src[i,j-1]+src[i,j+1]) * 0.25` (M5 §6.3 s7) — a
   four-term sum, a literal, **and** the wrong stencil (see D-19);
2. W3's nested 4-ary `maximum` (M5 §6.4 r5);
3. the accumulator zeroing (`acc[m,n] = 0.0`, M5 §6.1 line 22) — there is **no kernel statement**
   for `statement_index` to point at;
4. the flip's whole-tile `acc[:] = acc[:] + recv[:]` (M5 §6.2 line 28) — not a scalar triple.
To emit any of them M5 must re-walk `KernelModel.statements`, map operands to buffers and
subtract per-PE offsets. That is deciding, which `00-README.md` §5 D-14 forbids and
`test_emitter_makes_no_decisions` is supposed to enforce.
**Edit**: replace `ComputeNode` with an expression tree (EDIT-1).

### B-4 — `PlanNode` has no branch region, but three protocols need one

`06-interfaces.md` §5.5: *"`PlanNode` is the union `BufferPlan | ChannelSite | LoopPlan | ComputeNode`"*,
and §5.2 gives `ChannelSite.guard: Guard | None`, a single predicate on a single site.
`03-lld-M4-mapping.md` §3.7.1 line 25 nevertheless walks *"`elif n is Guard-region`"*, §3.7.2
rules 2–3 reason about *"guard regions"*, and §3.6.3 lines 19–23 need an `ops.branch` whose
`otherwise` body contains a get, a compute node and a **nested** branch. A per-site optional
guard cannot express that, and M5 §3.2 row 11 emits `with air.ops.branch(...) as h: … with h.otherwise(): …`
from nothing in the contract.
**Edit**: add `BranchNode` to `PlanNode` (EDIT-2).

### B-5 — M4 reads `LegalMapping` in the wrong basis, and the basis it needs is not a field

`03-lld-M3-checker.md` §3.1: *"`LegalMapping.sigma`, `.pi`, `.ker_pi` → `Coord` … `AccessMap.matrix` (from M1) → `UCoord`"*.
`03-lld-M4-mapping.md` §3.2 line 4: *"`M := access matrix of a over mapping.axes`"* — `mapping.axes`
is `Coord`, `AccessMap.matrix` is `UCoord`. Line 11 then evaluates `M · e[axis_d]` where
`axis_d = mapping.schedule.place[d]` is a post-tiling name (`"i0"`), which has no column in
`UCoord`; line 15 tests `ker(M) ⊆ mapping.ker_pi`, mixing `UCoord` and `Coord`. M3 computes the
object M4 actually needs — `ker_pi_u` (§3.3 line 20, §3.6 line 1) — but `06-interfaces.md` §4.1
has **no field for it**, so M4 cannot get it without re-deriving `root(place)` itself.
Every delivery classification (FR-M1, FR-M2, FR-M3) rests on this.
**Edit**: add `pi_u` and `ker_pi_u` to `LegalMapping` (EDIT-3) and rewrite M4 §3.2 in `UCoord`.

### B-6 — W1-flip: four incompatible specifications, and the cascade direction in the LLDs is measured wrong for the adopted form

| Source | grid | place | chain |
|---|---|---|---|
| `01-requirements.md` FR-K2, `02-hld.md` §7.1 | — | `place(px=ax.i0, py=ax.k0)` | "3 `npu_cascade` channels along a 1-D row herd" |
| `04-test-plan.md` §4 | `(4, 2)` | — | — |
| `03-lld-M2-schedule.md` §6.2, `03-lld-M3-checker.md` §6.2, `03-lld-M8` §3.1, architect **R-D** | `(4,)` | `place(px=ax.k0)` | — |
| `03-lld-M4-mapping.md` §6.2, `03-lld-M5-emitter.md` §6.2 | `(2, 4)` | `place(px=ax.i0, py=ax.k0)` | **descending** (`ty == 3` head) |

`03-lld-M4-mapping.md` §3.6.3 lines 6–7 get the *rule* right (*"if `len(herd.grid) == 1`: direction := ASCENDING"*),
but §6.2, M5 §6.2 and `test_M6_cascade_chain` (*"chain descends"*) are all written for the 2-D
form that R-D has just abolished, and the only cascade probe B ran (`$PROBE/flip2.py`) is 2-D.
**Measured this session (P-R3)**: on a 1-D `grid(4)` herd the chain must **ascend**
(`aie.cascade_flow(%tile_0_2, %tile_1_2)`, `(1,2)→(2,2)`, `(2,2)→(3,2)`, `aircc` exit 0); the
descending variant of the same module fails with
`'aie.cascade_flow' op source tile must be to the North or West of the destination tile`.
So the adopted flip is buildable, but M4 §6.2 / M5 §6.2 / `test_M6_cascade_chain` are wrong in
both the herd shape and the direction, and M4 §6.2's `TN = 64` contradicts `tile(ax.j, 32)`.
**Edit**: EDIT-12 through EDIT-16.

### B-7 — the new `DMA-CHANNELS` check is unsound as specified and will reject designs that compile

`03-lld-M4-mapping.md` §3.8: *"`inb := MAX_OVER_EXCLUSIVE_BRANCHES({ s.channel for s in herd sites if s.kind == "get" … })` … `if len(inb) > DMA_IN_MAX: FAIL("DMA-CHANNELS", …)`"*, with `DMA_IN_MAX := 2`.
**Measured (P-R4)**: W3 with `q` and `r` staged from L3 gives **three** distinct inbound channels
per core (`WestIn`/`West` on exclusive branches = 1, plus `QIn`, `RIn`) and `aircc --device npu1 --output-format=none`
**exits 0 with zero `error:` lines**. The reason is in the lowered IR: W1's and W3's L3→L1 gets
become **packet flows** (`aie.q7a.mlir`: 3 × `aie.packet_flow`; `air_A2L1_0`, `air_B2L1_0`,
`air_B2L1_1` all allocated `shim_noc_tile_0_0, MM2S, 0`), which multiplex; W2 at `PI=4` has
**zero** packet flows (all circuit-switched `aie.flow`), which is why its interior PE ran out at
three. Counting logical channels therefore produces false rejections — and a checker that
rejects a program the toolchain accepts is worse for G2 than no checker.
**Edit**: restate the check over **circuit-switched core-to-core and non-packet L3 endpoints
only**, keep it a warning for anything else, and record the packet-flow observation (EDIT-21).

### B-8 — W2's fixture parameters take five values across the set

`00-README.md` §7 D-11: *"Use `H = W = 18` (interior 16×16); the per-core L1 figure becomes 864 B, not 768"*.
`04-test-plan.md` §4: `H=W=16, PI=4, HS=4`, `768`. `03-lld-M8` §3.3: `H, W = 18, 16`, `768`.
`03-lld-M3-checker.md` §6.3: `H=W=18, PI=4, HS=4`, `864` (span `(2,6,18)`).
`03-lld-M4-mapping.md` §6.3 and `03-lld-M5-emitter.md` §6.3: `H=W=16, HS=8, PI=2`, `1280`.
Architect **R-B**: `U[T+1, H+2, W]`, `H=W=16`, `PI=2`, `HS=8`, `PI·HS == H`.
Five parameter sets, four L1 figures. No downstream artefact (fixture, golden, summary,
`ir_facts`) can be written until one wins.
**Edit**: adopt R-B verbatim everywhere (EDIT-11).

### B-9 — `ChannelSite.depends_on` refers to ids that do not exist

`06-interfaces.md` §5.2: *"`depends_on` | `tuple[str, ...]` | site ids this site takes a token from"*.
`ChannelSite` has ten fields and none is an `id`. M5 §3.2 row 9 must resolve
`dependency=<tok or None>` from them.
**Edit**: add `id: str` to `ChannelSite` (EDIT-4).

### B-10 — two error codes are raised that the frozen catalogue does not contain

Machine check over `design/*.md`: exactly two codes are used outside `06-interfaces.md` §6.3 —
`DMA-CHANNELS` (`03-lld-M4-mapping.md` §3.8, §5, §7, §10; `03-lld-B-open-questions.md` §5) and
`GRAMMAR-NONUNIFORM-DEP` (`03-lld-M1-frontend.md` §5, §10). `test_D3_catalogue_complete`
asserts both directions (`03-lld-M7-tests.md` §3.2: *"`RAISED_CODES == set(CATALOGUE)`"*), so it
fails on both until §6.3 is amended. R-G already rules the grammar code in; R-B implies the DMA
one. The catalogue count 41 then becomes **43** and three places quote 41.
**Edit**: EDIT-5, EDIT-6, EDIT-44.

### B-11 — the demo's headline rejection survives only because of one unstated rule

`03-lld-M8-kernels-demo.md` §3.5: *"If instead `skew` sets only the leading terms of a default `σ` and the checker appends `(j0, j1)`, then `Sσ·d = (0, 1, −7) ⪰ 1` lexicographically and **the schedule is accepted** — and the demo has no headline."*
Architect **R-E** adopts exactly that reading ("skew DEFINES σ's leading row; remaining rows from
loop order"). The rejection survives **only** because `03-lld-M3-checker.md` §3.3 line 24 also
excludes **placed** axes from the appended rows, so `Sσ = [e_i; e_j1]` and `Sσ·(0,1,−7) = (0,−7)`.
That exclusion appears in one pseudocode line in one LLD and in no requirement.
**Edit**: write it into FR-S17 and FR-L2 (EDIT-7).

### B-12 — M3 raises `ClauseError` codes, contradicting the error model and the test helper

`03-lld-M3-checker.md` §3.3 lines 5–7: *"raise PLACE-EXTENT-family error (reason "place without grid" …) -> code `CLAUSE-RANK`, raised as a LegalityError-shaped ClauseError"*, and line 11
*"require name in Coord else `CLAUSE-UNKNOWN-AXIS`"*. `02-hld.md` §4.1 assigns `ClauseError` to
**M2** only; `06-interfaces.md` §6.3 gives both codes `stage = clause`; and
`03-lld-M7-tests.md` §3.2's `assert_diagnostic` asserts `d.stage == CATALOGUE[d.code].stage`.
A `legality`-stage diagnostic carrying a `clause`-stage code fails that assertion.
**Edit**: EDIT-8.

### B-13 — `MappingPlan.tensors` has no construction rule and violates an upstream ordering law

`06-interfaces.md` §5.6: *"`tensors` | … | the L3 interface, in `air.tensor` declaration order"*.
`03-lld-M4-mapping.md` §3.3 `BUFFER_PLAN` line 11 *"`if mapping.schedule.residency[a] != "L1": continue`"*
— L3 buffers are never produced, and no other pass creates them.
**Measured (P-R4)**: `air.api`'s `_check_interface` (`python/air/api/_compile.py:226-240`) raises
`RuntimeError: output tensors must be declared after all input tensors; the interface order is [('Zrow','in'),('Sink','out'),('Out','out'),('Q','in'),('Rr','in')]`
— i.e. **every read-only tensor must precede every written tensor**, a hard constraint on
`MappingPlan.tensors` that no design document states. It bites W3 immediately and W2 as soon as
B-1 is fixed.
**Edit**: EDIT-9, EDIT-10.

---

## 2. DRIFT TABLE

"Adopt" is the reviewer's single value, with the reason.

| # | Quantity | Occurrences | Adopt |
|---|---|---|---|
| D-1 | W2 `H`, `W` | D-11 `18,18`; 04 §4 `16,16`; M1 §6.2 `18,18`; M3 §6.3 `18,18`; M8 §3.3 `18,16`; M4/M5 §6.3 `16,16`; R-B `16,16` with `U[T+1,H+2,W]` | **R-B**: `H = W = 16`, `U[T+1, H+2, W]`. Only form where `PI·HS == H` holds exactly *and* the halo row is in bounds; it is the architect's ruling and the only one the `PI=2` probe compiled |
| D-2 | W2 `PI`, `HS` | HLD §7.2 / 04 §4 / M1 / M3 / M8 `4, 4`; LLD-B §5 / M4 / M5 / R-B `2, 8` | **`PI = 2, HS = 8`**. `PI=4` is measured to fail `aie.connect` (P-R2); it becomes a negative fixture |
| D-3 | W2 per-core L1 | HLD §7.2 `768`; 04 §4 `768`; M8 §3.3 `768`; D-11 `864`; M3 §6.3 `864`; LLD-B §5 / M4 §6.3 `1280` | **`1280`** = `2 × (8+2) × 16 × 4`, the figure implied by D-1 + D-2 |
| D-4 | W2 domain / stencil | M1 §6.2 `range(1,H-1)`, 5-point `0.2 *`; M4 §6.3 / M5 §6.3 4-point `* 0.25`, `i` in `1..HS` | **M1's 5-point `0.2` stencil** — it is the kernel, hence the oracle |
| D-5 | W1-flip grid | FR-K2/HLD `place(i0,k0)`; 04 §4 `(4,2)`; M4/M5 `(2,4)`; M2/M3/M8/R-D `(4,)` | **`grid(4)`, `place(px=ax.k0)`** (R-D; only form with `repeats=(1,)` on both targets) |
| D-6 | W1-flip cascade direction | M4 §3.6.3 rule "1-D ⇒ ASCENDING"; M4 §6.2 + §7 + M5 §6.2 "descending" | **ASCENDING**, measured P-R3 |
| D-7 | W1-flip `TN` | M2 §6.2 `tile(ax.j,32)`; M4 §6.2 `TN=64` | **32** |
| D-8 | W1-flip `R_time`/`R_space` | M3 §6.2 `r_time = ()`, `r_space = ((0,0,1),)` (UCoord); M4 §6.2 `span{e_k0}`; M8 §6.2 demo prints `R_time = span{e_k1}, R_space = span{e_k0}` (Coord) | **Carry both**: `LegalMapping` keeps the `UCoord` split (M3 is right); `MappingSummary.reduction_split` renders the **tiled** split, because `R_time = {}` on the flip reads as "no local accumulation", which is false. See EDIT-17 |
| D-9 | W3 per-core L1 | HLD §7.3 / 04 §4 `80`; M3 §6.4 `72` (M4 → `80`); M8 §3.4 `240` | **`240`** (M8 counts `q` 128 B and `r` 32 B, which every other figure omits and which the kernel demonstrably reads) |
| D-10 | W3 channels | HLD §7.3 / FR-M5 one `West` bundle of `PJ+1`; LLD-B §6 / M4 §3.6.2 / M5 §6.4 three channels `WestIn[1]`, `West[PJ-1]`, `EastOut[1]` | **Three channels** (`AIRLoweringPass.cpp:798`; the single bundle is measured to fail) |
| D-11 | W3 kernel signature | M1 §6.3 `sw(Sub, S)`; M8 §3.4 `sw(q, r, S)`; M5 §6.4 `Zrow, Sink, Out` | **M8 §3.4's `sw(q, r, S)`** — it is the only one inside FR-S3 that also matches `04-test-plan.md` §4's `sub = +2/-1` |
| D-12 | W2 halo bundle `size` | HLD §7.2 `[4]`; LLD-B §2 / M4 §3.6.1 `[PI-1]` | **`[PI-1]`** (R-A) |
| D-13 | `broadcast_shape` golden text | FR-M2 `{broadcast_shape = [2, 2]}`; N-5 / M5 §3.8 `[2 : index, 2 : index]` | **`[2 : index, 2 : index]`** — re-verified in `$PROBE/q7_a.mlir:4` |
| D-14 | Cascade acceptance | FR-M6/FR-E8 "3 `npu_cascade` channels", attribute "three times"; LLD-B §4 / M5 §3.8 "one bundle, 3 `aie.cascade_flow`" | **"`PK-1` cascade links; `ir_facts.cascade_channels == 3`"** (R-A); confirmed 3 `aie.cascade_flow` in P-R3 |
| D-15 | Golden file names | 06 §8 `<workload>.<variant>.<target>.air.mlir`; M5 §6.1/§7 `w1.default.*`, `w1.flip.*`; M7 §6.2 / M8 §6.1 `w1.base.*`; 05 §2 D6 `w1-flip.*` | **`w1.base.*` / `w1.flip.*` / `w2.base.*` / `w3.base.*`** — pick one; `base` reads better than `default` in a gallery of four |
| D-16 | Causality negative input | FR-L2, 04 §5 item 1, M7 §6.1: `skew(time=(ax.j0, ax.i))`; M3 §6.5, M3 Q-M3-4, M8 §3.5: `skew(time=(ax.i,))` | **`skew(time=(ax.i,))`** — the reversal is a no-op because `skew` is a sum (FR-S17) |
| D-17 | Stationarity negative | FR-L3 `place(px=ax.i0, py=ax.k0)` + `stationary("C")` with `grid` unstated; M3 §7 `grid(2,4)`; M8 §3.5 `grid(PI, PK)` = `(2,4)` | **`grid(2,4)`, `place(px=ax.i0, py=ax.k0)`, `stationary("C")`** |
| D-18 | `L1-CAPACITY` demo case | FR-L9 `TM=TN=TK=128` `bf16` (98 304 B before doubling); M8 §3.5 `M=N=K=192, TM=TN=96, TK=32` `f32` (61 440 → 86 016) | **M8's**, as FR-L9's acceptance; the FR's own numbers never demonstrate the doubling (R-G) |
| D-19 | `test_traceability` strength | 04 §7 "exactly one test"; M7 §6.3 / 04 §8 item 1 "at least one" | **"at least one"** (R-G) |
| D-20 | FR count with a test | 01 §8 "70 of 70"; M7 §7.9 "69 of 70" | **70 of 70** after `test_K5_scope_documented` lands (R-G) |
| D-21 | W2 oracle tolerance | 04 §8 item 5 "all three exactly, `tol = 0.0`"; M6 §3.6 / M8 §4 W2 `1e-5` | **`1e-5` for W2**, exact for W1/W1-flip/W3 |
| D-22 | Error-code count | 06 §6.3 41; 04 §5 "41 codes, 41 tests"; M3 §7 "15 of the catalogue's 41" | **43** after EDIT-5/EDIT-6 |
| D-23 | `aircc` ping-pong opt-out flag | VF §E.1 (line 844) `--omit-pingpong`; M6 §3.10 / §8.1 `--omit-ping-pong-transform` | **`--omit-ping-pong-transform`** — re-verified via `aircc --help` this session |
| D-24 | ping-pong `air-opt` pipeline | 04 §3.2 / 07 §4 the 16-pass VF §E.5 prefix; M6 §3.7 (measured) the 2-pass form | **`builtin.module(air-dependency,air-label-scf-for-to-ping-pong{device=<t>})`** (R-G); keep the long one as the named fallback |
| D-25 | W1-large dtype | 04 §4 silent; M8 §3.2 `bf16` in, `f32` acc | **`bf16`/`f32`**; `f32` throughout would be 81 920 B and rejected by FR-L9 |
| D-26 | Effort, per module | 05 §7: M1 1.0 pd, M2 0.5, M3 1.5, M4 2.5, M5 1.0, M6 1.0, M7 1.5, M8 1.0; LLD §9s: M1 5 h, M2 4 h, M3 10 h (+3 h corpus), M4 **3.3 pd**, M5 1.0, M6 1.0, M7 **1.7** (steps sum), M8 1.0 | **The LLD figures**, and state that 1 pd = 8 h. New total ≈ **11.1 pd** against 05 §7's 10.5 pd budget |
| D-27 | Kernel-source ownership | 00-README §2 M8→C; 00-README §3 "Kernel sources … A"; 05 §2 D0 A; M8 §9 "A writes the sources, C owns the schedules" | **M8 §9's split**, written into 00-README §2 |
| D-28 | `LegalMapping` literal owner | 05 §2 D0 "B … and one `LegalMapping` literal"; HLD §8 "A → B … from D0"; M7 §3.8 owner **A**; M4 §8 "A's hand-written literal" | **A** |
| D-29 | Demo beat 3 line count | 05 §5 "Change two clause lines"; M8 §5.1 "three schedule lines — `grid(PK)`, `place`, `stationary`" | **Three** |

---

## 3. GAPS — where a document is silent and an implementer must guess

| # | Where | Missing | Sentence(s) to add |
|---|---|---|---|
| G-1 | `06-interfaces.md` §5.6 | how `tensors` is built and its ordering law | *"`tensors` is one `BufferPlan` per `KernelModel.param`, `level="L3"`, `scope="tensor"`, **every read-only param before every written param** — `air.api`'s `_check_interface` raises otherwise (`_compile.py:226-240`, measured)."* |
| G-2 | `06-interfaces.md` §5.6 | launch and segment names | *"`launch_name: str` and `segment_name: str`. M5 emits them verbatim; it derives no name."* |
| G-3 | `06-interfaces.md` §5.3 | the cascade chain's direction | *"`chain_direction: "ascending" \| "descending" \| None` — set only on a cascade channel; `None` otherwise."* |
| G-4 | `06-interfaces.md` §5.5 | synthetic loop axes | *"`LoopPlan.axis` is a kernel axis name, or a plan-synthesised name for a bundle-index or drain loop (`<operand>_bundle`, `<operand>_drain`), which no `KernelModel.axes` entry matches."* |
| G-5 | `03-lld-M4-mapping.md` §3.3 | `reside(x="L2")` | line 11 currently `continue`s, so an L2 residency is **silently ignored**; FR-S12's `test_reside_maps_to_scope` cannot pass. Add: *"a residency other than `L1` or `L3` raises `PROTOCOL-UNSUPPORTED` naming `reside`."* |
| G-6 | `03-lld-M1-frontend.md` §3.1/§3.6 | module-level `int`s inside an **expression** (`GAP`, W3) | §3.4 line 18 resolves globals only for loop **bounds**; §3.6's `AFFINE` only runs on subscripts. Add a rule: *"a bare `NAME` in an expression that is neither an axis nor a parameter is resolved from `fn.__globals__` if it is bound to a Python `int`, and is otherwise `GRAMMAR-UNSUPPORTED-EXPR`."* |
| G-7 | `03-lld-M1-frontend.md` §3.5 | chained scalar binds | W3 binds `d`, then `sub` from `d`, then `diag` from `sub`. §3.5 line 15 substitutes "a scalar name bound and read inside the SAME loop body", singular. Add: *"substitution is transitive and is applied in binding order; a cycle is `GRAMMAR-UNSUPPORTED-STMT`."* |
| G-8 | `03-lld-M4-mapping.md` §3.6.2 | W3's `Sub`/`q`/`r` staging | no buffer, no channel, no DMA count. Add the two L3→L1 channels and recount §3.8 |
| G-9 | `03-lld-M4-mapping.md` §3.6.2 | draining all of `S` | `SOut` carries one row per PE. Add a per-row `SOut` put inside `ROW`, or state that `S`'s L3 image is the last row and change FR-K4 |
| G-10 | `03-lld-M4-mapping.md` §3.6.1 | W2's per-plane drain | same shape of gap as G-9, for `U[t+1,…]` |
| G-11 | `03-lld-M3-checker.md` §3.12 | reaching `SWAP-PARITY` after the D-4 override | the only reachable condition is `T < 1` or a non-constant trip count. Add a `w2_zero_t` negative fixture to `03-lld-M8` §4 or `test_D3_catalogue_complete` fails |
| G-12 | `03-lld-M6-toolchain.md` §3.3 | `"auto"` at the toolchain boundary | D-13 says the checker resolves `"auto"` internally without a subprocess; `artifact()` calls `detect_target_device`. Add: *"only `m6.artifact` may resolve `"auto"`, and only outside a test (I-6)."* |
| G-13 | `07-environment.md` §2 | `vendor/wheels/` is 2.3 GB | add: *"`vendor/wheels/` is git-ignored; `vendor/wheels/SHA256SUMS` is committed and the wheels are distributed out of band"* (M6 §3.10 flags it) |
| G-14 | `07-environment.md` §2 | wheels are Python- and platform-tagged | add M6 §3.10's note, and reconcile with M7 §3.9's CI matrix (see P-6) |
| G-15 | `07-environment.md` §4 | the correct opt-out flag | add `--omit-ping-pong-transform` and a note that VF §E.1 spells it wrong |
| G-16 | `07-environment.md` §6 | repo layout | add `demo/`, `tests/helpers/`, `tests/integration/`, `kernels/rejections.py` (M7 §2.1, M8 §2) |
| G-17 | `01-requirements.md` §6 | four new risks | LLD-B's DMA budget, N-3's silent MM2S merge, mixed-bundle L3/core-to-core, unbuilt `npu2` goldens (R-C, lead 22) |
| G-18 | `01-requirements.md` §7 | the new open questions | B-O1, B-O4…B-O7, Q-C4, Q-C8, Q-C15…Q-C17 exist only inside LLDs |
| G-19 | `04-test-plan.md` §3.4 | `test_sem_coverage` vs partial drains | as written it fails for W2 and W3 (B-1, B-2). Either fix the plans or restate the check as "over the drained sub-domain the plan declares" |
| G-20 | `03-lld-M4-mapping.md` §7 | `test_M4_balanced` (FR-M4) | named in FR-M4 and M7 §7.3, defined nowhere |
| G-21 | `03-lld-M5-emitter.md` §7 | `test_sequential_emits_scf_for` (FR-S14) | named in FR-S14 and M7 §7.1, defined nowhere |
| G-22 | `01-requirements.md` FR-M9 | a test id split across a line | `test_M9_selfcheck_\nrejects[...]` — M7 §6.3's parser reads FR ids from this file; keep ids on one line |
| G-23 | `03-lld-M3-checker.md` §3.11 | the `(2,)` grid row | W2 at `PI=2` needs it; the table has `(4,)` and `(8,)` only |
| G-24 | `06-interfaces.md` §5.1 | `BufferPlan.scope` for L3 | invariant says `L3 ⟹ scope == "tensor"` but §5.6 never creates one (see G-1) |
| G-25 | `00-README.md` §1 | the LLDs | *"The per-module LLDs … are **not** in this set"* is stale; §7 D-15 overrides it. The reading-order table has no LLD rows |
| G-26 | `03-lld-M8` §3.3 | a stale claim about FR-L14 | *"This contradicts FR-L14's acceptance test (`test_L14_swap_parity` — "W2 with `T = 5` is rejected")"* — FR-L14 was already edited under the D-4 override and now says `T = 5` is **accepted**. Delete the contradiction paragraph |

---

## 4. PLAN RISKS

| # | Person / day | Problem | Fix |
|---|---|---|---|
| P-1 | **A, D1** | M0 `0.3 pd` + M1 `5 h` + M2 `4 h` ≈ **11.4 h**, and 05 §2's sequencing note requires M0 in the first two hours because B and C block on it | Move M1's 28-case and M2's 28-case negative corpora (≈3 h) to D3, where 05 §2 already schedules "the negative corpus". Restate M1 §9 and M2 §9 DoD as "D1 positives, D3 negatives" |
| P-2 | **A, D2** | 05 §2 D2 asks for "L1 capacity" and asserts `l1_bytes == 12288`, but `03-lld-M3-checker.md` §9 puts `FOOTPRINT`/capacity in step 6, on **D4–D5** | Move `FOOTPRINT` + capacity into D2 (it is ~1 h and W1's figure is the D2 gate), or drop the byte assertion from G2 |
| P-3 | **B, whole week** | M4 §9 totals **3.3 pd** (0.8 over 05 §7's 2.5) and M5 1.0 pd → 4.3 pd for the critical-path owner against ≈3.5 pd at 50 % utilisation. B-6 adds a **rewrite** of the flip plan on D6, the stretch day | Carry the 3.3 pd figure into 05 §7; move M4 §3.9's summary renderer to A (05 §2 already gives A the renderer on D5); accept that G5 will most likely cut the flip, and say so at the D2 stand-up rather than at D6 midday |
| P-4 | **C, whole week** | M6 1.0 + M7 **1.7** (its own §9 steps sum) + M8 1.0 = 3.7 pd vs 05 §7's 3.5, plus D0 installs on three machines and a 2.3 GB wheel cache | Cut M7 step 10 (0.2 pd of budget tuning) to D7-morning, or move `test_K5_scope_documented` + the NFR tests to A |
| P-5 | **D0, stub ownership** | 05 §2 gives **B** both the `MappingPlan` **and** the `LegalMapping` literal; HLD §8, M7 §3.8 and M4 §8 give the `LegalMapping` to **A**. And B's D0 checkpoint is *"the plan literal validates against the M0 invariants"* — **M0 does not exist until D1** | 05 §2 D0: B writes the `MappingPlan` literal **as data, unvalidated**; A writes the `LegalMapping` literal; both are validated against M0 on D1 morning |
| P-6 | **C, D0/CI** | M6 §3.10 records that wheels are Python- and platform-tagged; M7 §3.9 pins CI to **3.12**; the measured install is **3.14.7** | Fix one interpreter version for the three machines **and** CI, record it in `07-environment.md` §1, and download the cache on that version |
| P-7 | **All, D0** | `06-interfaces.md` is "frozen at D0" but this review requires **eleven** changes to it (EDIT-1…EDIT-10, EDIT-44), plus two new codes | Apply every `06-interfaces.md` edit **before** the D0 signature and bump `CONTRACT_VERSION` to 2 in the same commit. A freeze over a contract with known holes is worse than a one-hour delay |
| P-8 | **A, D0 (missing task)** | M8 §8.1 needs A's ruling on module-level `int` constants in an expression (`GAP`) **at D0**; 05 §2's D0 row does not list it | Add it to 05 §2 D0, Person A, with the fallback "the kernel inlines `1`" |

**Critical path, as reviewed.** 05 §3's chain is right, but it understates two boxes: M4's W1
path now also carries the tensor-ordering rule (B-13) and the expression-tree `ComputeNode`
(B-3), both of which are on the D2 gate. There is **no slack** on B before D6.

---

## 5. LEAD VERDICTS

| # | Verdict | Evidence | Edit |
|---|---|---|---|
| 1 | **Confirmed** | `M_B = (k,j)`, `ker M_B = span{e_i}`; with `i0` placed, `e_i0 ∉ ker Sπ` (`03-lld-M2-schedule.md` §6.2, `03-lld-M8` §3.1's two independent arguments) | EDIT-12…16 |
| 2 | **Confirmed** | `01-requirements.md` FR-S14 vs its own `test_pipeline_is_hint`; `02-hld.md` §7.1 prints `σ` with `k0` third under `pipeline(ax.k0)` | EDIT-22 (R-F: pure hint) |
| 3 | **Confirmed** | FR-L1's schedule is legal because §3.3 line 27 gives every unplaced axis its own σ row; FR-L2's reversal is a no-op because `skew` is a sum | EDIT-23, EDIT-24 |
| 4 | **Confirmed, but superseded** | `14 % 4 ≠ 0` is real; R-B replaces it with `PI=2, HS=8, H=W=16, U[T+1,H+2,W]` — which also satisfies the P-R2 DMA measurement, so adopt R-B, **not** `H=W=18` | EDIT-11 |
| 5 | **Confirmed** | machine check: `GRAMMAR-NONUNIFORM-DEP` appears only in `03-lld-M1-frontend.md`; the catalogue has 41 codes and not this one | EDIT-5 |
| 6 | **Partially confirmed — and it exposes B-5** | The two-frame rule is internally coherent (`03-lld-M3-checker.md` §3.1) and Bareiss-over-`int` is justified (`§3.2`: `256⁸ ≈ 1.8 × 10¹⁹` > `int64`). But `06-interfaces.md` §4.1 carries no `ker_pi_u`, and `03-lld-M4-mapping.md` §3.2 reads `M_a` "over `mapping.axes`" — the wrong frame | EDIT-3, EDIT-18 |
| 7 | **Confirmed, and now measured** | P-R3: 1-D `grid(4)` compiles with an **ascending** chain and fails descending. Lead 7's "2-D folds the cascade axis" is also right: `(4,2)` → npu1 physical `(1,2)` repeats `(4,1)` | EDIT-12…16 |
| 8 | **Refuted in favour of R-B** | `H=W=18` fixes divisibility but leaves `PI=4`, which P-R2 measures as a hard `aie.connect` failure | EDIT-11 |
| 9 | **Confirmed** | `03-lld-M8` §3.4: `q` is `MQ×4 = 128 B`, `r` is `CW×4 = 32 B`; HLD §7.3, 04 §4 and M3 §6.4 all omit them | EDIT-25 |
| 10 | **Confirmed** | FR-L9's `TM=TN=TK=128` `bf16` is `3 × 32 768 = 98 304` B **before** doubling; M8 §3.5's `61 440 → 86 016` is the case that shows the doubling | EDIT-26 |
| 11 | **Confirmed** | `04-test-plan.md` §7 "exactly one"; M7 §6.3 names three tests that legitimately carry two FR ids | EDIT-27 |
| 12 | **Confirmed** | FR-K5's acceptance is *"`test_W4_absent` is not a test"*; `test_traceability` would fail on it | EDIT-28 |
| 13 | **Confirmed, re-measured** | `aircc --help` this session: `--omit-ping-pong-transform[=<string>]`; VF line 844 says `--omit-pingpong` | EDIT-29, EDIT-30 |
| 14 | **Confirmed, re-measured** | `find <purelib> -name 'arch*.json'` → nothing. `mlir/test/Util/Runner/arch.json` → `"devicename": "testdevice"`, `L1 = 32768`. `Runner.cpp:543-551` dereferences a `dyn_cast_if_present<air::LaunchOp>` with no null check | EDIT-31, EDIT-32 |
| 15 | **Confirmed** | M6 §3.7's three-row table, and P-R1's `$PROBE/q7a_lab.mlir:111` `{unroll = 2 : i32}` with two `hoist_alloc = true` | EDIT-33 |
| 16 | **Ruled (R-E) — and it needs one more sentence to work** | Under R-E alone, M8 §3.5's arithmetic says the bad schedule is **accepted**. It is rejected only because M3 §3.3 line 24 excludes **placed** axes from the appended σ rows | EDIT-7 |
| 17 | **Confirmed, re-measured** | P-R1: the producer-loop-before-herd shape builds (127 lines), verifies, `aircc` exit 0 / 0 errors. Q-2 `size=[PI-1]`: `$PROBE/q2_w2.mlir` declares `@ToNorth [3]`, `@ToSouth [3]`. Q-6 `h.private()`. Q-1: `$PROBE/q/flip2d/air_project/aie.flip2.mlir:619-621` — three `aie.cascade_flow`, descending, on a **2-D** herd | — |
| 18 | **Confirmed as an observation; the generalisation is wrong** | P-R2 reproduces the `aie.connect` failure verbatim. But P-R4 shows three inbound channels per core compile fine when they lower to **packet** flows: `aie.q7a.mlir` has 3 `aie.packet_flow` and shares one shim MM2S across `A2L1`/`B2L1`; `aie.w2pi4.mlir` has **zero** packet flows. The budget binds on circuit-switched flows | EDIT-21, EDIT-34 |
| 19 | **Confirmed** | `AIRLoweringPass.cpp:798` `"failed to specialize channel bundle indices"`, reached from `getIndexToMetadataArrayFromChannelIndices`; `$PROBE/w3b.py`'s three-channel form compiles | EDIT-19, EDIT-20 |
| 20 | **Confirmed** | `$PROBE/q7_a.mlir:4` `air.channel @A2L1 [2, 1] {broadcast_shape = [2 : index, 2 : index]}`; `q6_w2_3herd` ELF lists (`elfs_jacobi_seg_core_{0..3}_{2,3,4}` — twelve cores) | EDIT-35, EDIT-36 |
| 21 | **Confirmed; settled as R-B** | see D-1…D-3 | EDIT-11 |
| 22 | **Confirmed** | no `npu2` module has been built; B-O4 is open at D2. N-3's merge is unverifiable without a device | EDIT-37, EDIT-38 |

---

## 6. SPOT-CHECKS

| # | Claim | Source cited | Verdict |
|---|---|---|---|
| 1 | `PHYSICAL_HERD = {"npu1": {1:(4,), 2:(1,4)}, "npu2": {1:(8,), 2:(2,4)}}` | `_trace.py:88-91` | **PASS** (exact lines) |
| 2 | `L1_BYTES = 65536` | `_trace.py:100` | **PASS** |
| 3 | `_IMPLEMENTED_TYPES = ("npu_cascade", "npu_dma_packet")` | `_channel.py:547` | **PASS** |
| 4 | `npu_cascade` rejects `broadcast_shape` | `_channel.py:170-177` | **PASS** |
| 5 | `broadcast_shape[d] % size[d] == 0` enforced | `_channel.py:137-145` | **PASS** (137-145) |
| 6 | bundle index must not be an `scf.for` IV | `AIRDialect.cpp:3586-3593` | **PASS** (3586-3593; the message is verbatim) |
| 7 | `"failed to specialize channel bundle indices"` | `AIRLoweringPass.cpp:798` | **PASS** |
| 8 | `emitOpError` without `signalPassFailure()` | `Dependency.cpp:2063-2066` | **PASS** (2064) |
| 9 | `air-runner` dereferences the `LaunchOp` unguarded | `Runner.cpp:547-551` | **PASS** (`dyn_cast_if_present` at 547-548, `.getSizeOperands()` at 549) |
| 10 | `mlir/lib/Analysis/` does not exist | VF §B.4 | **PASS** (`ls` fails) |
| 11 | `docs/AIRCorrectnessChecker.md:15-20` = P1…P4 | 01-req, HLD, M4 | **PASS**; file is **511** lines, as 05 §5 claims |
| 12 | `<herd>.shared()` raises | `_trace.py:1405-1414` | **PASS** (1405-1413) |
| 13 | `air.api` supports 1-D/2-D herds only | `_trace.py:1288-1291` | **PASS** (raise at 1287-1289 — off by one) |
| 14 | `resolve_target` accepts `None/auto/npu1/npu2` | `_trace.py:164-183` | **PASS** (164-181) |
| 15 | `mlir()` is `str(self.build())` | `_compile.py:242` | **PASS** |
| 16 | `_l1_peak` exists on `LaunchContext` | `_compile.py:74`, `:150` | **PASS** |
| 17 | a written tensor becomes `is_output` | `ops.py:310` | **PASS**, and **also** `_channel.py:427` for a channel `get` into a tensor — the path our emitter actually uses, cited nowhere |
| 18 | `ops.maximum` exists | finding N-9 | **PASS** (`ops.py:709`) |
| 19 | `cascade_reduction.py` uses the dead-index form `size=[NUM_TILES]` | LLD-B §2 | **PASS** (`:60-62`) |
| 20 | `worker_to_worker` lit is hardware-CI | `run_makefile_peano.lit:4-9` | **PARTIAL** — `REQUIRES: ryzen_ai, peano` is line 4, `CHECK: PASS!` is line **10** |
| 21 | `AIRToAIEPass.cpp:4484-4491` silently repairs L2 imbalance | VF §S4 | **PASS** (the doc comment begins at 4482) |
| 22 | no arch JSON ships in the wheel | M6 §3.8 | **PASS** (`find <purelib> -name 'arch*.json'` → empty) |
| 23 | `aircc` flag is `--omit-ping-pong-transform` | M6 §3.10 | **PASS** |
| 24 | `broadcast_pattern_count == 0` on our W1 | N-8 | **PASS** (`$PROBE/q7a_bc.mlir`, count 0) |

24 of 24 pass; two are off by one line. Citation discipline in this set is good.

---

## 7. CONSOLIDATED EDIT LIST

Each item is self-contained.

### `design/06-interfaces.md`

1. **§5.5, replace `ComputeNode`.** Replace *"`ComputeNode`: `(statement_index: int, subscripts: tuple[Expr, ...], op: str)` — a single scalar `memref.load`/`arith`/`memref.store` triple, which is what a fully-integer subscript emits (`_value.py:590-593`, VF §D.11)."* with:
   `ComputeNode`: `(target: str, target_subscripts: tuple[Expr, ...], expr: ExprTree)`, where
   `ExprTree` is the frozen union `Load(buffer: str, subscripts: tuple[Expr, ...]) | Const(value: int | float, text: str) | Bin(op: "+"|"-"|"*"|"/", lhs: ExprTree, rhs: ExprTree) | Neg(ExprTree) | Call(op: "maximum"|"minimum", args: tuple[ExprTree, ...])`.
   Add: *"`Const.text` is the source token, never a float round-trip (HLD §5 rule 4). Every `buffer` names a `BufferPlan` in this plan, so M5 resolves it by lookup and never by derivation. Whole-buffer forms (`b[:] = …`) are expressed with an empty `target_subscripts` tuple."*
2. **§5.5, extend `PlanNode`.** Replace the union with
   `BufferPlan | ChannelSite | LoopPlan | ComputeNode | BranchNode`, and add:
   `BranchNode`: `(condition: Guard, body: tuple[PlanNode, ...], otherwise: tuple[PlanNode, ...])` — *"emitted as `with air.ops.branch(condition) as h: <body>` then `with h.otherwise(): <otherwise>`. Conjunction is nesting; there is no `and` (`_cond.py:57-60`). `otherwise` may be empty, which leaves an `scf.if` with an empty else that `RemoveEmptyElseBranch` deletes (`_cond.py:61-66`)."*
3. **§4.1, add two `LegalMapping` fields** after `ker_pi`:
   `pi_u` | `tuple[tuple[int, ...], ...]` | *"`Sπ` in the **untiled** frame: row `r` is `e_{root(place[r])}`"* | `len(pi_u) == len(pi)`, `len(kernel.axes)` columns;
   `ker_pi_u` | `tuple[tuple[int, ...], ...]` | *"a canonical basis of `ker Sπ_u`"* | deterministic.
   Add a note: *"`sigma`, `pi`, `ker_pi` are over the post-tiling axis order (`axes`); `pi_u`, `ker_pi_u`, `r_time`, `r_space` and every `AccessMap.matrix` are over `kernel.axes` (`03-lld-M3-checker.md` §3.1)."*
4. **§5.2, add a field** to `ChannelSite`: `id` | `str` | *"a plan-unique site identifier, `f"{channel}.{kind}.{order}@{scope}"`"* | unique in the plan. Then `depends_on` is well defined.
5. **§6.3, add a row** (stage `grammar`): `` `GRAMMAR-NONUNIFORM-DEP` | grammar | a written array is read at an affine but non-uniform distance (`S[i,j] = S[j,i]`) ``.
6. **§6.3, add a row** (stage `mapping`): `` `DMA-CHANNELS` | mapping | a core needs more circuit-switched inbound or outbound DMA channels than the target provides ``.
7. **§5.6, add invariant 6**: *"**Tensor order**: every `BufferPlan` in `tensors` that the kernel only reads precedes every one it writes. `air.api`'s `_check_interface` raises otherwise (`_compile.py:226-240`)."*
8. **§5.6, add two `MappingPlan` fields**: `launch_name: str` and `segment_name: str`.
9. **§5.3, add a field** to `ChannelPlan`: `chain_direction` | `"ascending" \| "descending" \| None` | *"the cascade chain's orientation in the carrying herd coordinate; `None` on a non-cascade channel"*.
10. **§5.5, footnote `LoopPlan.axis`**: *"a kernel axis name, or a plan-synthesised name for a bundle-index or drain loop; no `KernelModel.axes` entry matches the latter."*
11. **Header**, bump `CONTRACT_VERSION = 2` and add a change-log row in `00-README.md` §4 recording edits 1–10 with forcing requirement *"REVIEW-round1 B-3, B-4, B-5, B-9, B-10, B-13"*.

### Fixture and kernel parameters

12. **`04-test-plan.md` §4, W2 row** — replace `` `H=W=16`, `PI=4`, `HS=4`, `T=4` `` / `` `2 × 6 × 16 × 4 = 768` `` with
    `` `T=4`, `H=W=16`, `U: [T+1, H+2, W]`, `PI=2`, `HS=8` `` / `` `2 × 10 × 16 × 4 = 1280` ``, and add a note *"`PI·HS == H` is an invariant of this fixture; `PI=4` is a `DMA-CHANNELS` negative (measured: `aie.connect` failure on the first interior PE)."*
13. **`02-hld.md` §7.2** — same numbers; replace `768` with `1280`, `size=[4]` with `size=[PI-1]`, and the index sentence with `03-lld-M4-mapping.md` §3.6.1's table.
14. **`03-lld-M1-frontend.md` §6.2** — change the source to `T = 4; H = W = 16` and `def jacobi(U: sp.f32[T + 1, H + 2, W])` with `for i in range(1, H + 1)`; delete the §6.2 "Finding" block and Q-M1-4, replacing them with a pointer to R-B.
15. **`03-lld-M3-checker.md` §6.3** — retitle to `T=4, H=W=16, U[T+1,H+2,W], PI=2, HS=8`; extents become `(4, 2, 8, 14)`; span `(2, 10, 16)`; `l1_bytes = 1280`; delete the `864`/`768` footnote.
16. **`03-lld-M8-kernels-demo.md` §3.3, §4** — same source and numbers; `w2` and `w2_odd` L1 become `1280`; delete the "`PI = 2` was considered and rejected" paragraph and replace it with *"`PI = 2` is forced by the measured 2-S2MM circuit budget; FR-M4's interior-PE assertion moves to the `PI=4` negative fixture, which is checked but never emitted."*
17. **`03-lld-M4-mapping.md` §6.3 / `03-lld-M5-emitter.md` §6.3** — make `tensors` the single kernel parameter `U: [T+1, H+2, W]`; put/get regions index plane `t` and plane `t+1`; the drain writes **every** plane the kernel writes, or FR-K3 is restated to compare only the final plane (say which).
18. **`03-lld-M4-mapping.md` §6.3 / `03-lld-M5-emitter.md` §6.3, stencil** — replace the 4-term `* 0.25` update with the kernel's five-term `0.2 * (src[i,j] + src[i-1,j] + src[i+1,j] + src[i,j-1] + src[i,j+1])`.

### W1-flip

19. **`01-requirements.md` FR-K2** — replace *"with `place(px=ax.i0, py=ax.k0)` and `stationary("B")`"* with *"with `grid(PK)`, `place(px=ax.k0)` and `stationary("B")`"*, and the acceptance with *"…produces a plan with `R_space = span{e_k}`, `PK−1 = 3` cascade **links** on one `npu_cascade` bundle, an **ascending** chain, and text that `aircc --output-format=none` accepts"*.
20. **`02-hld.md` §7.1, flip paragraph** — replace `place(px=ax.i0, py=ax.k0)` with `grid(PK)`, `place(px=ax.k0)`; replace *"each guarded by `ops.branch(tx == 0)` / `ops.branch(tx == PK-1)`"* with *"head at `tx == 0`, tail at `tx == PK-1`, the chain ascending in `tx` — measured: a descending chain on a 1-D herd fails `'aie.cascade_flow' op source tile must be to the North or West of the destination tile`"*.
21. **`04-test-plan.md` §4, W1-flip row** — replace `` `PK=4`, `PJ=2`, grid `(4, 2)` `` with `` `PK=4`, grid `(4,)`, `TM=TN=32`, `TK=16` ``, L1 `12 288`.
22. **`03-lld-M4-mapping.md` §6.2** — rewrite for `grid=(4,)`, `place=("k0",)`, `coords=("tx",)`, `physical_herd=(4,)`, `repeats=(1,)`, `TN=32`; delivery `A: STATIONARY (derived)`, `B: STATIONARY (declared)`, `C: CASCADE along px (derived)`; channels `A2L1 (4,)`, `B2L1 (4,)`, `CascadeK (3,) npu_cascade chain_direction="ascending"`, `C2L3 (1,)`; head at `tx == 0` putting at `[tx]`, middle getting at `[tx-1]`, tail at `tx == PK-1` putting to L3. Buffers `acc [32,32]`, `recv [32,32]`, `a [32,16]`, `b [16,32]`; the `(i0, j0)` tile pair is a 4-trip `air.sequential` inside the herd body. Delete the `(PI=2, PK=4)` paragraph and the `stationary("B")` footnote — under `place(px=ax.k0)`, `ker M_B = span{e_i} ⊆ ker Sπ_u = span{e_i, e_j}` holds, so **B-O5 is closed**.
23. **`03-lld-M4-mapping.md` §7** — `test_M6_cascade_chain`: *"chain **ascends**"*; `test_M6_cascade_orientation`: *"chain **descends** when `len(grid) == 2`"*. Add: *"both directions measured on the pinned wheel (REVIEW-round1 P-R3, `$PROBE/q/flip2.py`)."*
24. **`03-lld-M5-emitter.md` §6.2** — rewrite the call sequence for the 1-D herd, one coordinate `tx`, `ops.branch(tx == 0)` head, `ops.branch(tx == PK-1)` tail, puts at `[tx]`, gets at `[tx-1]`.
25. **`03-lld-M4-mapping.md` §3.6.3** — add after line 7: *"Measured on the pinned wheel: a 1-D `grid(4)` herd places as four columns on one row and the chain must ascend; a 2-D `(1,4)` herd is one column of four rows and the chain must descend. The direction is a `ChannelPlan.chain_direction` field, never an emitter choice."*
26. **`03-lld-M8-kernels-demo.md` §6.2** — the printed summary's `R_time = span{e_k1} R_space = span{e_k0}` is the **tiled** split; add a sentence saying so and cross-reference EDIT-30.

### W3

27. **`03-lld-M1-frontend.md` §6.3** — replace the `sw(Sub, S)` source with `03-lld-M8-kernels-demo.md` §3.4's `sw(q, r, S)` verbatim, and recompute the `KernelModel`: `params = [q i32 (MQ,) read, r i32 (NR,) read, S i32 (MQ+1,NR+1) written]`; scalar binds `d`, `sub`, `diag` forward-substituted; `dependences = ((0,1),(1,0),(1,1))` RAW on `S`.
28. **`03-lld-M1-frontend.md` §3.1 / §3.6** — add G-6's module-constant rule and G-7's transitive-substitution rule.
29. **`02-hld.md` §7.3** — replace the single `West` bundle of `PJ+1` with `WestIn [1]` / `West [PJ-1]` / `EastOut [1]`, delete *"**No per-core guard is needed**"* and replace it with *"the head and tail are guarded by `ops.branch`; a bundle may not mix L3 and core-to-core members (`AIRLoweringPass.cpp:798`, measured)"*, and change the L1 figure from `80` to `240 B` with the `q` 128 B / `r` 32 B breakdown.
30. **`01-requirements.md` FR-M5** — replace *"on a channel bundle of extent `PJ+1`, plus a constant source at index 0 and a drain at index `PJ`"* with *"across three homogeneous channels — `WestIn` (L3→L1, `size=[1]`), `West` (L1→L1, `size=[PJ−1]`), `EastOut` (L1→L3, `size=[1]`) — so that the `PJ+1` channel indices are balanced and no bundle mixes an L3 with a core-to-core member (`AIRLoweringPass.cpp:798`)"*.
31. **`04-test-plan.md` §4, W3 row** — L1 `2 × 9 × 4 + 2 × 4 = 80` becomes `240` with the `q`/`r` breakdown.
32. **`03-lld-M3-checker.md` §6.4** — add `q` and `r` to the L1 calculation; `l1_bytes` becomes `232` at M3's scope (M4 adds the two staging scalars → `240`).
33. **`03-lld-M4-mapping.md` §6.4 / `03-lld-M5-emitter.md` §6.4** — `tensors = (q, r, S)` in that order (reads first, EDIT-7); add `QIn`/`RIn` channels and `qb`/`rb` buffers; make the compute read `sub` from them instead of the literals `+2`/`-1`; drain **every** row of `S` (a `SOut` put per row inside `ROW`, `SOut size=[PJ]`), or restate FR-K4 to compare only the last row (say which). Add: *"measured — with `q`/`r` staged, `aircc --device npu1 --output-format=none` exits 0 with zero `error:` lines (REVIEW-round1 P-R4)."*

### Checker, mapper, emitter

34. **`03-lld-M4-mapping.md` §3.2** — replace line 4 with *"`M := the AccessMap matrix of a, over `mapping.kernel.axes` (UCoord)`"*, line 11 with *"`if M · e[root(axis_d)] == 0`"* (`root` = the untiled parent of a placed tile handle), and line 15 with *"`elif contains(mapping.ker_pi_u, kernel_basis(M))`"*. Add: *"every classification is computed in `UCoord`; `mapping.pi`/`ker_pi` are `Coord` and are used only for the herd geometry (`03-lld-M3-checker.md` §3.1)."*
35. **`03-lld-M3-checker.md` §3.3** — replace lines 5-7's *"raised as a LegalityError-shaped ClauseError"* with *"`raise LegalityError(PLACE-EXTENT)` with the reason `place without grid`"*; replace line 11's `CLAUSE-UNKNOWN-AXIS` with an internal assertion (M2 §4 postcondition 1 already guarantees it). No `clause`-stage code is ever raised at `legality` stage.
36. **`03-lld-M3-checker.md` §3.3, after line 24** — add: *"**Placed axes are excluded from the appended σ rows.** This is what makes the demo's headline rejection a rejection: with `skew(time=(ax.i,))` on W3, `Sσ = [e_i; e_j1]` and the tile-crossing representative of `d = (0,1)` gives `Sσ·d = (0, −7)`. If `j0` were appended instead, `Sσ·d = (0, 1, −7)` would be lexicographically positive and the schedule would be **accepted** (`03-lld-M8-kernels-demo.md` §3.5)."*
37. **`01-requirements.md` FR-S17** — after *"set `σ` to the sum of the named axes, in order"* add: *"The remaining σ rows are the default loop order restricted to axes that are **neither named in `skew` nor placed**."* Acceptance: replace `ScheduleModel.sigma_terms` with `ScheduleModel.skew` (Q-M2-1).
38. **`01-requirements.md` FR-L2 acceptance** — replace *"W3 with `skew(time=(ax.j0, ax.i))` reversed, or with only `i` in `σ`"* with *"W3 with `skew(time=(ax.i,))` — the dropped term. (`skew` is a **sum**, so reversing the two terms is a no-op.)"*
39. **`01-requirements.md` FR-L1 acceptance** — replace *"a schedule placing `i` and leaving `i` out of `σ`"* with *"W1 with `skew(time=(ax.i1, ax.j1))`, whose `ker Sσ ∩ ker Sπ = span{e_i1 − e_j1}`"* (`03-lld-M3-checker.md` Q-M3-5).
40. **`01-requirements.md` FR-S14** — delete *"shall place `ax` innermost in `σ`"*; replace with *"shall be recorded and read by nothing; it is reserved for a future vectorisation pass"* (R-F).
41. **`01-requirements.md` FR-L9 acceptance** — replace the `TM=TN=TK=128` `bf16` case with `M=N=K=192, TM=TN=96, TK=32, f32, double_buffer("A","B")`: `61 440` undoubled, `86 016` charged, over by `20 480` (`03-lld-M8-kernels-demo.md` §3.5).
42. **`01-requirements.md` FR-M2 acceptance** — replace `` `air.channel @A2L1 [2, 1] {broadcast_shape = [2, 2]}` `` with `` `air.channel @A2L1 [2, 1] {broadcast_shape = [2 : index, 2 : index]}` `` (measured).
43. **`01-requirements.md` FR-M6 and FR-E8 acceptance** — replace *"the emitted text contains `channel_type = "npu_cascade"` three times"* with *"the plan contains `PK−1 = 3` cascade **links** on one bundle, and `ir_facts.cascade_channels == 3` (three `aie.cascade_flow` ops after `air-to-aie`)"* (R-A).
44. **`03-lld-M4-mapping.md` §3.8** — retitle *"P3 — the per-core **circuit-switched** DMA-channel check"* and rewrite the counting rule: *"count only channels whose members are core-to-core, or L3 endpoints the design cannot expect to be packet-switched. **Measured**: W1's and W3's L3→L1 gets lower to `aie.packet_flow` and share one shim MM2S channel (`aie.q7a.mlir`: 3 packet flows; `air_A2L1_0`/`air_B2L1_0`/`air_B2L1_1` all on `shim_noc_tile_0_0, MM2S, 0`), so three logical inbound channels compile cleanly; W2 at `PI=4` has zero packet flows and fails at three. A check that counts logical channels would reject W3-with-`q`/`r`, which compiles. Exceeding the circuit budget is a `DMA-CHANNELS` **error**; exceeding it counting packet channels is a **warning** naming the uncertainty."*
45. **`03-lld-M4-mapping.md` §5, §10** — delete the *"until the catalogue is amended, raise `PROTOCOL-UNSUPPORTED`"* fallback and close **B-O6**; EDIT-6 lands the code at D0.
46. **`03-lld-M4-mapping.md` §3.3** — add G-5's L2 rejection line; add a `TENSOR_PLAN` pass producing `MappingPlan.tensors` from `kernel.params` with EDIT-7's ordering law.
47. **`03-lld-M4-mapping.md` §3.9** — `SUMMARY` line 10 renders the **tiled** reduction split: *"`R_time` and `R_space` are rendered over the post-tiling axes, so the flip prints `R_time = span{e_k1}, R_space = span{e_k0}`; the `LegalMapping` fields stay untiled (`03-lld-M3-checker.md` §3.1). The two are not the same object and the summary says which it is showing."*
48. **`03-lld-M5-emitter.md` §3.1** — replace `air.launch(name=plan.mapping.kernel.name)` and `air.segment(name=f"{kernel.name}_seg")` with `plan.launch_name` / `plan.segment_name`; add to §4's P-1 list: *"M5 reads no field of `LegalMapping` or `KernelModel` at all."*
49. **`03-lld-M5-emitter.md` §3.2** — add a row for `BranchNode` and rewrite row 11 to read the node, not `ChannelSite.guard`; add a row for the L3 `BufferPlan` ordering assertion.

### Tests, requirements hygiene, plan

50. **`04-test-plan.md` §5** — *"41 codes, 41 tests minimum"* → *"43 codes, 43 tests minimum"*; item 1's schedule → `skew(time=(ax.i,))`; item 3's numbers → EDIT-41's.
51. **`03-lld-M3-checker.md` §7** — *"15 of the catalogue's 41"* → *"16 of the catalogue's 43"*; add a `(2,)` row to §3.11's table; add a `T = 0` W2 case so `SWAP-PARITY` stays reachable.
52. **`03-lld-M8-kernels-demo.md` §4** — add a `w2_zero_t` negative fixture (`T = 0`, `"expect": "reject"`, `SWAP-PARITY`), levels N only.
53. **`04-test-plan.md` §7** — *"appears in exactly one test's marker"* → *"appears in **at least one** test's marker, and no marker names an FR that does not exist"* (R-G).
54. **`04-test-plan.md` §8 item 5** — *"The three oracle diffs pass exactly (`tol = 0.0`)"* → *"W1, W1-flip and W3 diff exactly (`tol = 0.0`); W2 diffs within `1e-5` absolute, because the kernel multiplies by `0.2`"*.
55. **`04-test-plan.md` §3.2** — replace the 16-pass pipeline string for `test_I_pingpong_labels` with `builtin.module(air-dependency,air-label-scf-for-to-ping-pong{device=npu1})` and note the long form is the recorded fallback (M6 §3.7, measured).
56. **`04-test-plan.md` §3.4** — restate `test_sem_coverage` as *"the union over PEs of the written index sets equals the write domain the plan claims to drain, and the plan's claimed drain domain equals the kernel's write domain"*, so a partial drain fails **loudly** rather than by accident.
57. **`04-test-plan.md` §4, W1-large row** — add *"`A`,`B` `bf16`, `C` `f32`; in `f32` the same shape is 81 920 B and FR-L9 rejects it"*.
58. **`01-requirements.md` §6** — add four risks: **R-18** per-core circuit DMA budget (owner B, trigger: any `aie.connect` failure in an `aircc` smoke run); **R-19** silent MM2S merge at >2 outbound endpoints, correctness unverified without a device (owner B/C, trigger: a device run); **R-20** channel bundles may not mix L3 and core-to-core members (owner B, trigger: `failed to specialize channel bundle indices`); **R-21** `npu2` goldens are unbuilt and all four shapes were proved on `npu1` only (owner B, trigger: D2's `npu2` smoke).
59. **`01-requirements.md` §7** — add B-O1, B-O4, B-O5 *(closed by EDIT-22)*, B-O6 *(closed by EDIT-6)*, B-O7, Q-C4, Q-C8, Q-C15, Q-C16, Q-C17, each with owner, due day and fallback, so the open-question table is the single list.
60. **`01-requirements.md` §8** — *"Every FR has an acceptance test; 70 of 70"* stays, but add *"FR-D2 delegates to FR-M11 and FR-K5 is covered by `test_K5_scope_documented` (`03-lld-M7-tests.md` §7.7)."* Put FR-L14 back in numeric order between FR-L13 and the §3.3 heading. Keep every test id on one line (FR-M9 currently splits `test_M9_selfcheck_rejects`).
61. **`00-README.md` §1** — delete the paragraph *"The per-module LLDs (`03-lld-M1..M8.md`) named in the architect's brief are **not in this set**…"* and add nine rows to the reading-order table for `03-lld-*.md`, marked "read your own, plus M0's contract".
62. **`00-README.md` §2, §3** — set M8's owner to *"A (kernel sources, D0), C (schedules, fixtures, demo)"* per M8 §9; update §3's status table W2/W1-flip rows to the adopted parameters.
63. **`05-work-breakdown.md` §2 D0** — B writes the `MappingPlan` literal **as data**, A writes the `LegalMapping` literal, both validated on D1 once M0 lands; add "A: rule on module-level `int` constants in an expression (`GAP`)".
64. **`05-work-breakdown.md` §2 D1/D2** — move M1's and M2's negative corpora to D3; move `FOOTPRINT` + L1 capacity into D2 so the G2 checkpoint's `l1_bytes == 12288` is reachable.
65. **`05-work-breakdown.md` §5** — beat 3: *"three schedule lines — `grid(PK)`, `place(px=ax.k0)`, `stationary("B")`"*; beat 4: *"drop the `j0` term from the skew"*, not "reverse".
66. **`05-work-breakdown.md` §7** — replace the per-module estimates with the LLD figures (M4 3.3, M7 1.7), state *"1 pd = 8 hours"*, and set the total to ≈11.1 pd against 10.5 available — with the sentence *"the plan is 0.6 pd over budget before the first line is written; the flip (G5) is the designated cut."*
67. **`07-environment.md` §2** — add G-13's gitignore sentence, G-14's Python-tag sentence, and pin one interpreter version.
68. **`07-environment.md` §4** — add `--omit-ping-pong-transform` with the note that VF §E.1 spells it `--omit-pingpong`; replace the long `air-opt` ping-pong invocation with the measured two-pass form and keep the long one below it labelled "fallback".
69. **`07-environment.md` §3, §6** — add an `air-runner` row: *"works, but **no** resource model ships in the wheel; the only ones in the project are under `mlir/test/Util/Runner/`, and the canonical `arch.json` is a `testdevice` with a 32 KB L1, not `npu1`/`npu2`"*; extend §6's layout with `demo/`, `tests/helpers/`, `tests/integration/`, `kernels/rejections.py`.
70. **`hackathon/VERIFIED-AIR-FACTS.md` line 844** *(outside `design/`, flagged not edited)* — `--omit-pingpong` → `--omit-ping-pong-transform`.
71. **`03-lld-M8-kernels-demo.md` §3.3** — delete the paragraph beginning *"**This contradicts FR-L14's acceptance test**"*; FR-L14 already accepts `T = 5` under the D-4 override.
72. **`03-lld-M7-tests.md` §6.1** — change `test_L2_causality`'s input from `skew(time=(ax.j0, ax.i))` to `skew(time=(ax.i,))` and its `mentions` to `[(0, 1), (0, -7)]`.
73. **`03-lld-M7-tests.md` §6.2, `03-lld-M8` §6.1, `03-lld-M5-emitter.md` §6.1/§7** — settle golden names on `w1.base.*`, `w1.flip.*`, `w2.base.*`, `w3.base.*`; state the variant vocabulary in `06-interfaces.md` §8.
74. **`03-lld-M4-mapping.md` §7 / `03-lld-M5-emitter.md` §7** — add `test_M4_balanced` (FR-M4) and `test_sequential_emits_scf_for` (FR-S14), which every other document names.

---

## 8. VERDICT

**Not yet.** After the 74 edits above the set is close, and the architectural spine — the
`(σ,π)` checker, the two-frame rule, the balance/acyclicity self-check, D-14, the golden and
diagnostic conventions — is sound and unusually well evidenced. But four things would stop an
implementer on day two and cannot be resolved by an editor alone:

1. **W2 and W3 have no agreed program.** Three kernel signatures for W3, two stencils for W2,
   and in both cases the planned L3 interface is not the kernel's (B-1, B-2). Somebody must
   decide, and it is a design decision, not a transcription.
2. **`ComputeNode` and `PlanNode` cannot carry the four kernels** (B-3, B-4). Until they do,
   D-14 is aspirational and M5's one-day estimate is fiction.
3. **The flip's worked plan is written for a herd shape the architect has abolished**, and the
   chain direction in it is measured wrong for the shape that replaces it (B-6).
4. **`06-interfaces.md` is about to be frozen with eleven known holes** (P-7).

**Residual unknowns that need a device or a build, and cannot be closed by editing:**

* **N-3 / R-19** — whether a merged MM2S stream is correctly demultiplexed at two destinations.
  Unverifiable off-device; keep every core at ≤2 outbound endpoints.
* **The packet-vs-circuit rule** (B-7) is now measured for four module shapes; it is **not** a
  documented upstream contract, so `DMA-CHANNELS` must stay a warning wherever packet flows may
  appear. A device run is the only thing that turns "it compiled" into "it is correct".
* **`npu2`** — nothing in the set has been built for it. Four of the eight goldens are for a
  target no probe has touched (B-O4, due D2).
* **Whether ping-pong fires on the flip's `(i0, j0)` loop** (M8 §3.1) — one `air-opt` run at D6
  settles it; the fallback (`PINGPONG-SHAPE` at check time, drop `double_buffer`) is correct and
  already written.
* **W2 and W3 end-to-end numerics.** Every structural check in `04-test-plan.md` §3.4 is a proxy;
  `air.api` has no interpreter and `air-runner` is a timing model. Only level D closes it, and
  the honest-limits slide already says so.
