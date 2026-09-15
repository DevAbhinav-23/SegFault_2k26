# Design — Spatial DSL for NPUs, lowering to MLIR-AIR

*Phase 1 (design) document set, 2026-09-12. **As of 2026-09-13, Person B's modules — M0, M4, M5
and M6's off-device half — are built, verified and merged to `main` (`f7ecd70`, 2026-09-13) (see
[`PROGRESS-B.md`](PROGRESS-B.md) §P7), the stretch TT backend of
[`08-tt-backend.md`](08-tt-backend.md) is built and merged to `main` (`f7ecd70`, 2026-09-13)
with gates T1–T4 green (see [`PROGRESS-TT.md`](PROGRESS-TT.md)), while Person A's M1/M2/M3 and Person C's M6 device path,
M7 harness and M8 fixtures are still stubs.** Every AIR fact this set rests on was verified or
measured in
[`../hackathon/VERIFIED-AIR-FACTS.md`](../hackathon/VERIFIED-AIR-FACTS.md) and is cited
`path:line` at mlir-air commit `ff95a9b`.*

---

## 1. Reading order

Read in this order. Each document assumes the ones above it.

| # | Document | What it settles | Read it if… |
|---|---|---|---|
| 1 | [`01-requirements.md`](01-requirements.md) | goals, non-goals, **70 functional requirements** each with an acceptance test, 7 NFRs, 13 assumptions, a **21**-entry risk register, **18** assigned open questions | you are about to build, test, or argue with anything |
| 2 | [`02-hld.md`](02-hld.md) | the eight modules, the data flow, the error model, determinism rules, ten hot spots, the build order, and the **concrete W1/W2/W3 walk-through** (channels, sizes, `broadcast_shape`, put/get sites, herd shape, loop shape) | you need to know what your module receives and hands on |
| 3 | [`06-interfaces.md`](06-interfaces.md) | **FROZEN**: every shared dataclass with fields, types and invariants; every public entry point; the diagnostics schema and the 43-code error catalogue; the golden-file and fixture conventions | you are writing code that another person's code calls |
| 4 | [`04-test-plan.md`](04-test-plan.md) | seven test levels, fixtures, the negative corpus, CI, and the ten-point definition of "thoroughly tested" | you are writing a test, or deciding whether something is done |
| 5 | [`05-work-breakdown.md`](05-work-breakdown.md) | D0–D7 per person, the critical path, the stretch items, the five go/no-go gates, the demo script and the honest-limits slide | it is a morning and you want to know what to do |
| 6 | [`07-environment.md`](07-environment.md) | the pinned install, what works with no XRT and no device, `air-opt`/`aircc`/`air-runner` invocations, the repo layout | you are setting up, or a tool is misbehaving |
| 7 | [`03-lld-M1-frontend.md`](03-lld-M1-frontend.md) | the accepted grammar as EBNF, the AST walk, access maps, dependence vectors; W1/W2/W3's exact `KernelModel` | **read your own, plus M0's contract** — A |
| 8 | [`03-lld-M2-schedule.md`](03-lld-M2-schedule.md) | the 14 clauses, argument validation, the post-tiling axis order, why `pipeline` is inert | read your own, plus M0's contract — A |
| 9 | [`03-lld-M3-checker.md`](03-lld-M3-checker.md) | the two frames, integer linear algebra, the eleven legality checks, the physical-herd table, swap parity | read your own, plus M0's contract — A |
| 10 | [`03-lld-M4-mapping.md`](03-lld-M4-mapping.md) | the reuse trichotomy in `UCoord`, buffer/channel/loop plans, the three protocols, the P1′/P2b/P3 self-check | read your own, plus M0's contract — B |
| 11 | [`03-lld-M5-emitter.md`](03-lld-M5-emitter.md) | the translation table, one row per plan element; the expression walk; the four call sequences | read your own, plus M0's contract — B |
| 12 | [`03-lld-M6-toolchain.md`](03-lld-M6-toolchain.md) | `aircc`/`air-opt` wrappers, the stderr parser, the device path, the pin check | read your own, plus M0's contract — C |
| 13 | [`03-lld-M7-tests.md`](03-lld-M7-tests.md) | the harness, marks, golden helper, the traceability gate, the budget | read your own, plus M0's contract — C |
| 14 | [`03-lld-M8-kernels-demo.md`](03-lld-M8-kernels-demo.md) | the three kernel sources verbatim, the fixtures, the three demo rejections, the five-minute script | read your own, plus M0's contract — C |
| 15 | [`03-lld-B-open-questions.md`](03-lld-B-open-questions.md) | B's probe log: what was measured, what it settled, what is still open | read it before arguing with a channel shape — B |
| 16 | [`REVIEW-round1.md`](REVIEW-round1.md) + [`RESPONSE-review1.md`](RESPONSE-review1.md) | the adversarial review of this set and what was done about every item | you want to know why something reads the way it does |
| 17 | [`08-tt-backend.md`](08-tt-backend.md) | **stretch**: the second emitter, `MappingPlan` → TT-Metalium, **executed** on Tenstorrent's functional simulator ttsim — the translation table, the depth-1 FIFO protocol over counting semaphores, the TT-P3 resource model, FR-TT1…FR-TT13, gates T1–T4, and the honest-limits text | you are arguing about whether the plan is backend-neutral — B |

**Upstream context, not part of this set but load-bearing:**
`../hackathon/VERIFIED-AIR-FACTS.md` (every AIR fact, with the E1–E5 status table),
`../hackathon/01-paradigm-comparison.md` r3 (why P1), `../spatial-dsl/02` (the `(σ,π)` maths the
checker implements), `../CLAUDE.md` (scope rules).

---

## 2. Ownership

| Module | Name | Owner | Primary documents |
|---|---|---|---|
| M0 | `model` — shared contract | **all three** (frozen; see §4) | `06-interfaces.md` §2–§6 |
| M1 | Frontend / kernel capture | **A** | `01-req` §3.1 FR-S1…S4, `02-hld` §2 |
| M2 | Schedule builder / schedule IR | **A** | `01-req` §3.1 FR-S5…S20 |
| M3 | Legality checker | **A** | `01-req` §3.2 FR-L1…L14 |
| M4 | Mapping / protocol synthesis | **B** | `01-req` §3.3 FR-M1…M12, `02-hld` §7 |
| M5 | AIR emitter | **B** | `01-req` §3.4 FR-E1…E10 |
| M6 | Toolchain & runtime driver | **C** | `01-req` §3.5 FR-T1…T6, `07-environment.md` |
| M7 | Tests & CI harness | **C** | `04-test-plan.md` |
| M8 | Kernels, fixtures, demo | **A** (the three kernel sources, at D0) + **C** (schedules, fixtures, demo) | `01-req` §3.7 FR-K1…K5, `05-wbs` §5, `03-lld-M8-kernels-demo.md` §9 |

Person A is "Front & Law", Person B is "Mapping & Emit" and owns the critical path, Person C is
"Toolchain, Tests, Kernels, Demo". The split is the architect's (brief D10) and is not
negotiable inside the week.

---

## 3. Status

| Item | Status | Owner | Gate |
|---|---|---|---|
| Requirements | **drafted** | — | — |
| HLD | **drafted** | — | — |
| Interfaces | **drafted at `CONTRACT_VERSION = 6`, v4, v5 and v6 pending signatures**; every REVIEW-round1 edit and RULING 9's `MappingSummary.residency` applied *before* the freeze | all three | **D0** |
| Per-module LLDs (M1…M8 + B's probe log) | **drafted**, revised per REVIEW-round1 | per module | — |
| Design review round 1 | **answered** — `RESPONSE-review1.md` | — | — |
| Test plan | **drafted** | — | — |
| Work breakdown | **drafted** | — | — |
| Environment | **drafted**; wheel sha256s **to be filled at D0** | C | **D0** |
| Toolchain installed on 3 machines | not started | C | **D0 / gate G1** |
| Kernel sources W1/W2/W3 | not started | A | D0 |
| Schedules, fixtures, demo | not started | C | D0–D6 |
| M0 `model` | not started | A | D1 |
| M1, M2 | not started | A | D1 |
| M3 | not started | A | D2–D3 |
| M4 (W1 / W3 / W2 / flip) | **built (B side)** — 2026-09-13, see `PROGRESS-B.md` §P7; gate **G2/G3/G4/G5** B-half green | B | D2 / D4 / D5 / D6 |
| M5 | **built (B side)** — 2026-09-13, see `PROGRESS-B.md` §P7 | B | D1 skeleton, D2 real |
| M6 | off-device path (`invoke`, `verdict`, `artifact`, `ir_facts`, `check_pin`) **written by B at P0c/P4/P6** — flagged for C in `PROGRESS-B.md`; **device path not started** | C | D2 |
| M7 harness, M8 fixtures | harness shell + helpers (`conftest`, `golden`, `diagnostics`, `determinism`, `plan_interp`) **written by B** — flagged, **C to own**; M8 fixtures not started | C | D1 |
| **W1 end to end** | **built (B side)** — 2026-09-13, see `PROGRESS-B.md` §P7 — gate **G2** B-half: `test_W1_legal_to_text` | B, C | **gate G2, D2** |
| **W3 end to end** | **built (B side)** — 2026-09-13, see `PROGRESS-B.md` §P7 — gate **G3** B-half: `test_golden_w3`, `test_W3_aircc_none` | B, C | **gate G3, D4** |
| **W2 end to end** | **built (B side)** — 2026-09-13, see `PROGRESS-B.md` §P7 — gate **G4** B-half: `test_golden_w2`, `test_W2_aircc_none` | B, C | **gate G4, D5** |
| **W1 weight-stationary flip** (stretch) — 1-D `grid(PK=4)`, `place(px=ax.k0)`, `j` untiled, ascending cascade | **built (B side)** — 2026-09-13, see `PROGRESS-B.md` §P7 — gate **G5** B-half: `test_golden_w1_flip`, `test_W1_flip_aircc_none`, `cascade_channels == 3` | B | **gate G5, D6 midday**; the designated cut if the plan runs over |
| Device run (stretch) | not started | C | D5 |
| **TT backend (B, stretch)** — second emitter, `MappingPlan` → TT-Metalium on ttsim; spec [`08-tt-backend.md`](08-tt-backend.md) | **T1–T4 GREEN (2026-09-13), close-out T5; see [`PROGRESS-TT.md`](PROGRESS-TT.md); merged to main** | B | **gates T1–T4**; abandoned if a gate is not green after two agent-days |
| W4 FFT | **out of scope** | — | — |
| Freeze | — | all | **end of D6** |

**The design set is ready for D0 handoff.** The design loop is closed: REVIEW-round1 answered,
rulings 1–9 applied, and the final consistency sweep of 2026-09-13 recorded in
`RESPONSE-review1.md` under *Final sweep checks (2026-09-13)*. Nothing above the D0 line is
waiting on a design decision; what is outstanding is the three signatures on `06-interfaces.md`
v3 and the D0 task list in `05-work-breakdown.md` §2.

Open questions: **18**, all assigned with due days (`01-requirements.md` §7) — the original 7
plus B-O1, B-O4…B-O7 and Q-C4, Q-C8, Q-C15…Q-C17, which previously existed only inside LLDs.
Risks: **21**, all with an owner and a trigger (`01-requirements.md` §6) — R-18…R-21 were added
by the round-1 review.

**Fixture parameters, so there is one place to look**: W1 `M=N=K=64, TM=TN=32, TK=16, PI=PJ=2`;
W1-flip `PK=4`, grid `(4,)`, `TM=32`, `TK=16`, **`j` not tiled** (`TN = N = 64`); W2
`T=4, H=W=16, U[T+1,H+2,W], PI=2, HS=8`; W3 `MQ=NR=32, PJ=4, CW=8`. Per-core L1:
12 288 / **24 576** / 1 280 / 240 B, arithmetic in `02-hld.md` §7.

---

## 4. How to propose an interface change

`06-interfaces.md` is frozen at D0. It is frozen because three people write against it in
parallel from D1, and a field rename costs two of them a rebase (risk R-15).

**The process, which takes fifteen minutes and is not optional:**

1. Open a one-paragraph proposal naming: the field or signature, the change, **which FR forces
   it**, and which tests change.
2. All three owners reply. **All three must agree** — not a majority. If one disagrees, the
   change does not happen and the proposer works around it; if the workaround is worse than the
   change, say so in the proposal and re-run the step.
3. On agreement: bump `CONTRACT_VERSION` in `06-interfaces.md`, edit the document, and add a row
   to the table below **in the same commit** as the code change.
4. Whoever merges it tells the other two in the channel, with the version number.

**A change to `06-interfaces.md` after the end of D5 requires a stated reason why the demo
fails without it.** By then the cost of churn exceeds the cost of an ugly interface.

### Signature block (fill at D0)

| Document | Person A | Person B | Person C | Date |
|---|---|---|---|---|
| `06-interfaces.md` v3 | ☐ | ☐ | ☐ | |
| `06-interfaces.md` v4 | ☐ | ☐ | ☐ | |
| `06-interfaces.md` v5 | ☐ | ☐ | ☐ | |
| `06-interfaces.md` v6 | ☐ | ☐ | ☐ | |

*Sign **v3**, not v1 or v2: the round-1 review found eleven holes in the contract and RULING 9
found a twelfth, and a freeze over a contract with known holes is worse than a one-hour delay
(REVIEW-round1 P-7). Every edit below is already in the document.*

### Change log

| Version | Date | Change | Forcing requirement | Signed by |
|---|---|---|---|---|
| 1 | 2026-09-12 | initial | — | superseded before D0 |
| 2 | 2026-09-12 | §5.5 `ComputeNode` → `StoreNode` + the `ExprNode` tree (`Load`/`Const`/`BinOp`/`Neg`/`MaxMin`/`Select`); §5.5 `BranchNode` added to `PlanNode`; §5.5 `LoopPlan.axis` synthetic-name footnote; §4.1 `LegalMapping.pi_u`, `.ker_pi_u` and the two-frame note; §5.2 `ChannelSite.id`; §5.3 `ChannelPlan.chain_direction`; §5.6 `MappingPlan.launch_name`, `.segment_name`, the `tensors` construction rule and invariants 6-7; §6.3 `GRAMMAR-NONUNIFORM-DEP` and `DMA-CHANNELS` (catalogue 41 → **43**); §8 the closed `<variant>` vocabulary | REVIEW-round1 B-3, B-4, B-5, B-9, B-10, B-13 | *(pending D0)* |
| 3 | 2026-09-12 | §5.7 `MappingSummary.residency` — one `(operand, duration)` pair per operand — and the residency line each pair renders into `lines` (`03-lld-M4-mapping.md` §3.9 computes it) | architect **RULING 9**: the stationarity predicate is spatial, so "stationary" alone does not say how long an operand stays in L1 (FR-M11, FR-D2) | *(pending D0)* |
| 4 | 2026-09-13 | §5.5 `HerdPlan` ∈ `PlanNode`, §5.6 `segment_body` carries exactly one `HerdPlan` == `herd` (invariant 8); §2.7 `KernelModel.bindings`; §2.1 expression shape entries resolve to ints at capture; §5.2 `order` per body | FR-E1 + D-14 (B-P7); FR-M7 + Q-M1-2 (B-P9) | *proposed by B's architect 2026-09-13; pending A, B, C signatures* |
| 4 | 2026-09-13 | §8 summary golden path gains `<target>` (erratum, 2026-09-13) | architect ruling on **B-P14**: the herd line carries the physical shape and the repeats, which differ per target, so the target-less path of §8 contradicted `04-test-plan.md` §3.1's "stored per target too". Documentation only — no `CONTRACT_VERSION` bump | *architect, 2026-09-13* |
| 5 | 2026-09-13 | §2.4 `Statement.expr` (`ExprNode`, the desugared right-hand side over kernel-level operands) and §2.7's invariant that its `Load`s name `Param`s; §5.6 `declared` := named by a clause; §5.5 the loop-axis naming rule | FR-M8 + FR-E2 (B-P19); FR-M1/FR-M11 (B-P17); (B-P18) | *proposed by B's architect 2026-09-13; pending A, B, C* |
| 6 | 2026-09-13 | §7.2 `m6.run` gains `target`, `kernel_name`, optional `workdir`; `m6.trace` gains `function`, optional `workdir`; §7.2 `DiffReport` is a frozen dataclass in `spatial/model.py` | the frozen two-arg shapes cannot construct `XRTBackend` nor name the function for `air-runner -f` (C, `progress.md` §3) | *proposed by C 2026-09-13, bumped by the architect 2026-09-15; pending A, B, C* |

---

## 5. Decisions taken during design that the architect's brief did not fix

Listed so the architect can adjudicate any of them. Each says what was decided and why.

| # | Decision | Why | Reversible? |
|---|---|---|---|
| **D-1** | **`npu_cascade` is available through `air.api`, so FR-K2 needs no MLIR-text fallback.** `_channel.py:547` lists `_IMPLEMENTED_TYPES = ("npu_cascade", "npu_dma_packet")` and `programming_examples/cascade_reduction/cascade_reduction.py` is a worked head/middle/tail chain. The brief anticipated it being `_UNSUPPORTED` | the brief asked us to check; the check came out positive | n/a — this is a fact, not a choice. The *contingency* (an `npu_dma_stream` chain, same protocol) remains |
| **D-2** | **Balance rule P1′ diverges from the upstream spec's literal P1 for broadcast channels.** The spec keys on `(channel_name, indices)` and requires `put_count == get_count` per key; that would reject `air.api`'s own fan-out idiom (one put at `[pi,0]`, `PJ` gets at `[pi,0..PJ-1]`). Our rule: a put at index `i` must be matched by exactly one get at **each** index of `i`'s fan-out set | the literal rule is wrong for broadcast; ours reduces to the literal one when `broadcast_shape is None` | yes, but the alternative is to forbid broadcast channels |
| **D-3** | **Loop-kind rule**: a loop whose index is a channel bundle index is a **Python** loop; a loop whose body reuses an L1 buffer across trips is an **`air.sequential`**. Never the other way | `ChannelPutOp::verify` rejects an `scf.for` IV as a bundle index (`AIRDialect.cpp:3589-3592`); a Python loop unrolls and strands acquire/release pairs, computing "with stale operands" (`_loop.py:14-19`) | no — both halves are upstream constraints |
| **D-4** | **Buffer swaps are realised by unrolling a temporal loop by two**, which requires an even trip count, enforced as `SWAP-PARITY` (FR-L14) | `air.sequential` has no `iter_args` anywhere in `air.api` (`_loop.py:180`), so a swap cannot be loop-carried; the alternative is a copy per timestep | yes — the copy is the alternative and costs bandwidth, which is out of scope anyway |
| **D-5** | **`double_buffer` means two different things and says which.** On W1 it asserts the `isPingPongCandidate` shape and the pass does the work. On W2 the strips live outside the `t` loop, so it means "the emitter writes an explicit `u`/`v` pair" — and the checker's message says that rather than promising the pass will fire | promising a pass will fire when it structurally cannot is exactly the silent-no-op failure R-05 is about | yes |
| **D-6** | **W3 uses a segment-scope constant source and drain rather than per-core boundary guards**; W2 uses `ops.branch` guards. Both achieve the brief's "every channel index is balanced" | the source/drain is the mechanism PC §3.3 W3·P1 (d) names, and it needs no branch; W2's boundaries have no analogous source | yes |
| **D-7** | **Six leaf error classes; all specificity in a 43-code catalogue.** Negative tests key on codes, not on message prose | a new check then adds a code and a test, not a class, and messages can be improved without breaking tests | yes |
| **D-8** | **Golden AIR text is compared byte for byte; post-pass facts are extracted into a small `ir_facts.json` instead of a second golden module** | an upstream pass change then shows as one changed number rather than a 400-line diff — which matters most for the `air-broadcast-detection` count (R-04) | yes |
| **D-9** | **Off-device semantics are established structurally, not by executing the emitted IR** (three checks: access-region reconstruction, compute-node replay, write-domain coverage). Only the device run closes the loop, and the honest-limits slide says so | `air.api` has no interpreter (VF §D.10), `air-runner` is a timing model (VF §S7), and the CPU backend JITs the *lowered* module, testing the compiler's output rather than the user's source (PC §1.4) | no — this is a property of the toolchain |
| **D-10** | **Freeze at the end of D6 (Thu 18 Sep); D7 is rehearsal only.** The evaluation is listed 19–20 Sept | a freeze on the evaluation day is not a freeze | yes, if the date moves |
| **D-11** | **Fixture sizes** `W1 64³ / PI=PJ=2`, `W2 16×16 / PI=4 / T=4` *(the `PI = 4` half is **superseded by RULING 1** — the adopted W2 fixture is `PI = 2`, `HS = 8`; see §7's D-11 row)*, `W3 32×32 / PJ=4`, plus a larger `W1 256³ / 4×4` used only for toolchain-smoke and IR-fact tests (the shape VF §E.5 measured) | small enough for a 3-minute CPU suite, and the large one matches the measured probe so the ping-pong facts are asserted on comparable IR | yes |
| **D-12** | **Balance and acyclicity are checked by enumerating the concrete herd coordinates**, not by symbolic reasoning over guards | grid extents are compile-time constants and small (≤ 8 per axis), so enumeration is exact and trivial; symbolic reasoning over `ops.branch` guards is a research problem we do not need | yes |
| **D-13** | **The surface's `target` accepts `"auto"` but tests always pass an explicit target** | `"auto"` shells out to `xrt-smi` (`_trace.py:182`); a test must not spawn a subprocess or depend on a device's presence | yes |
| **D-14** | **The emitter makes no decisions.** If M5 would have to choose something, the plan is under-specified and the choice moves to M4 | it is what keeps the goldens meaningful and M5 a one-day task | yes |
| **D-15** | **The per-module LLDs (`03-lld-M1..M8.md`) were not written as separate documents.** `02-hld.md` §2 (responsibilities and non-responsibilities), `02-hld.md` §7 (a fully concrete walk-through per kernel) and `06-interfaces.md` (signatures, fields, invariants, error codes) together carry what the brief asked the LLDs to carry | with 7 days and 3 people, a second full pass over the same material costs a day of the critical path and adds a third place for the contracts to drift out of sync. **Flagged for the architect**: if the LLDs are wanted as separate documents, say so and they are half a day each | yes — say the word |

---

## 6. One-paragraph summary, for someone with two minutes

A `@sp.kernel` plain-Python loop nest plus an *ignorable* `sp.schedule` — delete the schedule and
the program runs in CPython and **is** the specification. The schedule's clauses are a space-time
map `(σ, π)`; a checker verifies causality, stationarity, the reduction split, halo width, grid
consistency and L1 capacity **before any IR exists**, and rejects with the clause, the failing
identity with its actual numbers, and a fix. A mapper then derives the reuse trichotomy
(stationary / multicast / stream), synthesises balanced, acyclic protocols for the halo exchange,
the wavefront and the cascade reduction, self-checks them against AIR's own P1/P2 definitions,
and an emitter translates the plan mechanically into `air.api`. Three kernels: GEMM
output-stationary (plus a weight-stationary flip with no kernel edit), Jacobi with halo exchange,
Smith-Waterman wavefront. The differentiator is the pre-codegen legality check: AMD specified
exactly this checker in their own repository (`docs/AIRCorrectnessChecker.md`) and has not built
it — `mlir/lib/Analysis/` does not exist, and `air-opt` prints its one channel diagnostic while
exiting 0.

---

## 7. Architect adjudication (2026-09-12)

Every decision in §5 has been ruled on. Two are overridden; the rest stand as written and are
binding from D0.

| # | Ruling | Note |
|---|---|---|
| **D-1** | **accepted** | `npu_cascade` is available through `air.api`; FR-K2 needs no MLIR-text fallback. The `npu_dma_stream` contingency remains as written. |
| **D-2** | **accepted** | Balance rule P1′ for broadcast channels stands; it is the rule `06-interfaces.md` §5.6 invariant 1 states. |
| **D-3** | **accepted** | The loop-kind rule (bundle index ⇒ Python loop; buffer reused across trips ⇒ `air.sequential`) stands; both halves are upstream constraints. |
| **D-4** | **OVERRIDDEN** | Buffer swaps with an **odd** timestep count are **not** rejected. The mapper peels the last timestep after the unrolled-by-two loop. `SWAP-PARITY` therefore fires only on a non-constant or `< 1` trip count. **FR-L14 and its acceptance test are edited accordingly**; the peel is M4's, and M3 records nothing extra (`03-lld-M3-checker.md` §3.12). |
| **D-5** | **accepted** | `double_buffer` means two things and must say which. The predicate that decides between them is `m3.pingpong_mode` (`03-lld-M3-checker.md` §3.13), exported so M3 and M4 share one definition. |
| **D-6** | **accepted** | W3 uses a segment-scope constant source and drain; W2 uses `ops.branch` guards. |
| **D-7** | **accepted** | Six leaf error classes, all specificity in the 43-code catalogue. Negative tests key on codes. |
| **D-8** | **accepted** | Golden AIR text byte for byte; post-pass facts in `ir_facts.json`. |
| **D-9** | **accepted** | Off-device semantics established structurally, not by executing the emitted IR. The honest-limits slide says so. |
| **D-10** | **accepted** | Freeze at the end of D6; D7 is rehearsal only. |
| **D-11** | **accepted, then superseded by REVIEW-round1 RULING 1** | Fixture sizes stand. W2's numbers are **not** `H = W = 18` / `864 B`: that fixes the `14 % 4 ≠ 0` divisibility failure but keeps `PI = 4`, which is *measured* to fail `aie.connect` on the first interior PE. The adopted fixture is `U: sp.f32[T+1, H+2, W]` with `H = W = 16` interior, **`PI = 2`, `HS = 8`** (`PI·HS == H` exactly), per-core L1 **1 280 B**. `PI = 4` is a `DMA-CHANNELS` negative fixture. |
| **D-12** | **accepted** | Balance and acyclicity by enumerating concrete herd coordinates, not symbolic reasoning. |
| **D-13** | **accepted** | `"auto"` is accepted by the surface; tests always pass an explicit target, and the checker resolves `"auto"` to `npu2` internally rather than shelling out. |
| **D-14** | **accepted** | The emitter makes no decisions. |
| **D-15** | **OVERRIDDEN** | The per-module LLDs **are** written as separate documents, `design/03-lld-*.md`, one per module, **owned per person**. Each references `06-interfaces.md` **by section number** and never restates a field list or a signature, so there is no third place for the contracts to drift. Person A's three — `03-lld-M1-frontend.md`, `03-lld-M2-schedule.md`, `03-lld-M3-checker.md` — are written; M4–M8 follow the same ten-section template. |

**Consequential edits made under this adjudication**: `01-requirements.md` FR-L14 (text and
acceptance test) — nothing else in that file at the time.

### 7.1 REVIEW-round1 rulings 1–9 (2026-09-12/13, after the adversarial review)

`REVIEW-round1.md` raised 13 blockers, 29 drift rows, 26 gaps and 8 plan risks, and proposed a
74-item edit list. The architect ruled on it; the rulings are binding, they win wherever they
conflict with an edit-list item, and `RESPONSE-review1.md` records every item's disposition.

| # | Ruling | What it settles |
|---|---|---|
| **RULING 1** | **W2's canonical contract.** `def jacobi(U: sp.f32[T + 1, H + 2, W])`, one rank-3 parameter; the `0.2 ×` **five**-point stencil (every 4-term/`0.25` variant deleted); write domain planes `1..T` × rows `1..H` × cols `1..W-2`, with rows `0`/`H+1` and cols `0`/`W-1` read-only boundary. Fixture `H = W = 16`, `PI = 2`, `HS = 8` (`PI·HS == H`), `T = 4` and `T = 5`. Protocol per PE per timestep: put boundary row(s), then get ghost(s), compute, then **one** `UOut` put of the computed strip for plane `t+1` — every plane drained. `UIn` stages plane 0 once. 2 in / 2 out at `PI = 2`; `PI ≥ 3` is a `DMA-CHANNELS` rejection naming the 2-S2MM budget | B-1, B-8, G-10, G-19 |
| **RULING 2** | **W3's canonical contract.** `def sw(q: sp.i32[MQ], r: sp.i32[NR], S: sp.i32[MQ + 1, NR + 1])` with `sub = MATCH if q[i-1] == r[j-1] else MISMATCH`. **Grammar D9 amended**: a value-level conditional is a `SELECT` (`arith.select`), module-level `int`s resolve from `fn.__globals__`, scalar binds substitute transitively in binding order. Fixture `MQ = NR = 32`, `PJ = 4`, `CW = 8`. `q` broadcast to every PE, `r` sliced per PE, the west boundary as **three** channels, `S` drained every row via `SOut`. Three inbound per core compiles because L3→L1 lowers to packet flows (measured) | B-2, G-6…G-9 |
| **RULING 3** | **Plan expressiveness.** `ComputeNode` → `StoreNode(buffer_id, subscripts, expr)` over an `ExprNode` tree; `BranchNode(predicate, then, otherwise)` joins `PlanNode`; accumulator zeroing is an explicit M4-produced `LoopPlan` of `StoreNode(Const 0)`; whole-tile ops are expanded by M4; `ChannelSite.id`; `MappingPlan.tensors` = one L3 `BufferPlan` per param, **all read-only params before all written params** (`_compile.py:226-240`); `launch_name`/`segment_name`; `ChannelPlan.chain_direction`; the synthetic-axis naming rule. **D-14 stands** | B-3, B-4, B-9, B-13, G-1…G-4, G-24 |
| **RULING 4** | **The flip is 1-D.** `grid(PK=4)`, `place(px=ax.k0)`, `stationary("B")`, cascade **ASCENDING** (measured P-R3). *(`TN = 32` superseded by **RULING 9**: `j` is not tiled.)* FR-K2, HLD §7.1, 04-test-plan's grid `(4,)`, M4 §6.2, M5 §6.2 and `test_M6_cascade_chain` all rewritten | B-6 |
| **RULING 5** | **Interfaces first.** Every `06-interfaces.md` edit lands **before** the D0 freeze, at `CONTRACT_VERSION = 2`. Catalogue = **43** codes (`DMA-CHANNELS`, `GRAMMAR-NONUNIFORM-DEP` added; every "41" fixed). `LegalMapping.pi_u`/`.ker_pi_u` added and M4 §3.2 rewritten in the untiled basis. M3 raises only `legality`-stage codes — the `CLAUSE-RANK`/`CLAUSE-UNKNOWN-AXIS` checks move to M2 | B-5, B-10, B-12, P-7 |
| **RULING 6** | **The skew rule is a requirement, not an implementation accident.** `skew(time=(a,b))` defines `σ`'s leading row `a+b`; the remaining `σ` rows are the loop order **excluding placed and skewed axes**; causality is lexicographic over those rows. The W3 illegal-skew demo (`skew(time=(ax.i,))`, the *dropped* term) is rejected with `L2-CAUSALITY`. Written into FR-S17 and FR-L2 | B-11 |
| **RULING 7** | **`reside(x="L2")` → `PROTOCOL-UNSUPPORTED` naming `reside`.** FR-S12 is restricted to L1/L3; L2 is documented future work | G-5 |
| **RULING 8** | Every other item of the 74-item edit list is **accepted as written**; where an item conflicts with rulings 1–7 the ruling wins and the conflict is logged in `RESPONSE-review1.md` | — |

**Items previously flagged and now closed by these rulings**: FR-K2's inconsistent
`place(px=ax.i0, py=ax.k0)` + `stationary("B")` (RULING 4 — under `place(px=ax.k0)` the
containment holds, closing B-O5); FR-S14's "`pipeline` places `ax` innermost in `σ`" (deleted —
`pipeline` is recorded and read by nothing); FR-L1's and FR-L2's acceptance prose (rewritten to
name schedules that are in fact rejected). All of these **are** now edited into
`01-requirements.md`.

**RULING 9 (2026-09-12) — the W1-flip must be genuinely weight-stationary.** SD-02 §3's spatial
predicate `ker M_a ⊆ ker Sπ` **stands**, but it does not say how long an operand stays in L1, and
under RULING 4's `tile(ax.j, 32)` the declared-stationary `B` was re-fetched every `j0` trip — the
demo sentence "the weights stay put" would have been false. Two consequences, both binding:
(a) `MappingSummary` gains a **residency duration** per operand (`06-interfaces.md` §5.7 at
`CONTRACT_VERSION = 3`, algorithm in `03-lld-M4-mapping.md` §3.9, test `test_M11_residency_line`);
(b) the flip fixture leaves **`j` untiled** (`TN = N = 64`), tiles `TM = 32` / `TK = 16` over
`PK = 4`, and declares `double_buffer("A")` only — `B` is resident for the whole run, so
`double_buffer("B")` would be `PINGPONG-SHAPE`. Per-core L1 is **16 384 B** at M3's scope and
**24 576 B** once M4 adds the cascade `recv` tile. This supersedes `RESPONSE-review1.md` §3's
conflict C-2 (the `12 288` figure and "the flip declares no `double_buffer`"). W1's own
output-stationary figures are untouched.
