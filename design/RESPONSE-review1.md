# Response to `REVIEW-round1.md`

*2026-09-12. Editor: consolidation pass. Scope: every file in `design/`, plus one flag-name
correction in `hackathon/VERIFIED-AIR-FACTS.md` §E.1 and a state update in `hackathon/HANDOFF.md`.
Nothing was committed. **No implementation code was written anywhere**; every change is
specification text, pseudocode, a kernel-surface listing, or a shell/IR transcript.*

**Inputs.** `REVIEW-round1.md` (13 blockers, 29 drift rows, 26 gaps, 8 plan risks, 22 lead
verdicts, 24 spot-checks, and §7's consolidated 74-item edit list) and the architect's **RULINGS
1–8**, now recorded in `00-README.md` §7.1. Where an edit-list item conflicts with a ruling, the
ruling won and the conflict is logged in §3 below.

**Outcome: 74 of 74 items dispositioned** — 62 applied as written, 11 applied with a ruling
modifying them, 1 applied with a corrected number (EDIT-21's L1 figure, §3). **13 of 13 blockers
closed.** All six mechanical checks pass; §4 has the pasted output.

---

## 1. Edit-list disposition, items 1–74

Legend: **A** = applied as written · **A/R** = applied, modified by a ruling · **A/C** = applied
with a correction recorded in §3 · **N** = not applied.

### `design/06-interfaces.md` (applied *before* the D0 freeze, per RULING 5 / plan risk P-7)

| # | Item | Status | Where it landed |
|---|---|---|---|
| 1 | replace `ComputeNode` with an expression tree | **A/R** | §5.5. RULING 3's naming wins: the node is **`StoreNode(buffer_id, subscripts, expr)`** and the union is `ExprNode = Load \| Const \| BinOp \| Neg \| MaxMin \| Select`. The item's `ComputeNode`/`ExprTree`/`Call` spelling lacked `Select`, which W3's kernel needs. The item's three added sentences (`Const.text`, buffer lookup, whole-buffer form) are all kept; the whole-buffer rule is strengthened to "there is no whole-buffer form — M4 expands it". Renamed at all 26 call sites in M4, M5, 04-test-plan and LLD-B |
| 2 | extend `PlanNode` with `BranchNode` | **A/R** | §5.5. Field names are RULING 3's `(predicate, then, otherwise)`, not the item's `(condition, body, otherwise)`. The item's `_cond.py` notes are kept verbatim; added that `ChannelSite.guard` is exactly `BranchNode(p, (site,), ())` |
| 3 | add `LegalMapping.pi_u`, `.ker_pi_u` | **A** | §4.1, plus the two-frame note the item asks for |
| 4 | add `ChannelSite.id` | **A** | §5.2; `depends_on` restated as "`ChannelSite.id` values" |
| 5 | catalogue row `GRAMMAR-NONUNIFORM-DEP` | **A** | §6.3 |
| 6 | catalogue row `DMA-CHANNELS` | **A** | §6.3, worded "**circuit-switched**" per EDIT-44/B-7 |
| 7 | §5.6 invariant 6, tensor order | **A** | §5.6 invariant 6, with the measured `_check_interface` message |
| 8 | `MappingPlan.launch_name`, `.segment_name` | **A** | §5.6 + invariant 7 |
| 9 | `ChannelPlan.chain_direction` | **A** | §5.3, with the `iff npu_cascade` invariant |
| 10 | footnote `LoopPlan.axis` | **A** | §5.5 |
| 11 | `CONTRACT_VERSION = 2` + `00-README.md` §4 change-log row | **A** | §header; `00-README.md` §4 change log and signature block (now "sign **v2**") |

### Fixture and kernel parameters

| # | Item | Status | Where it landed |
|---|---|---|---|
| 12 | `04-test-plan.md` §4 W2 row → `T=4, H=W=16, U[T+1,H+2,W], PI=2, HS=8`, `1280` | **A** | 04 §4, plus the `PI·HS == H` / `PI=4`-is-a-negative note |
| 13 | `02-hld.md` §7.2 same numbers, `size=[PI-1]`, M4's index table | **A/R** | §7.2 **rewritten whole** under RULING 1 — it also needed `UIn`/`UOut`, the per-plane drain, the 5-point stencil and the endpoint budget |
| 14 | `03-lld-M1-frontend.md` §6.2 source + delete the Finding and Q-M1-4 | **A** | §6.2 source is now `def jacobi(U: sp.f32[T + 1, H + 2, W])`, `for i in range(1, H + 1)`; the Finding block is replaced by the RULING-1 pointer; Q-M1-4 marked closed |
| 15 | `03-lld-M3-checker.md` §6.3 retitle, extents `(4,2,8,14)`, span `(2,10,16)`, `1280` | **A** | §6.3 rewritten; boundary lifts corrected `±3 → ±7`; the `864`/`768` footnote deleted |
| 16 | `03-lld-M8` §3.3/§4 same source and numbers; replace the `PI = 2` paragraph | **A** | §3.3 source and prose; §4 fixture rows `w2`/`w2_odd` → `1 280` |
| 17 | M4 §6.3 / M5 §6.3: `tensors` = the single `U`; drain every plane | **A/R** | Both rewritten. RULING 1's "ONE put of the computed strip per timestep" is the form taken, so FR-K3 is **not** restated — the whole write domain is compared |
| 18 | M4 §6.3 / M5 §6.3 stencil → five-term `0.2 *` | **A** | Both; each section says the four-term `0.25 ×` form is deleted |

### W1-flip

| # | Item | Status | Where it landed |
|---|---|---|---|
| 19 | FR-K2 → `grid(PK)`, `place(px=ax.k0)`, `stationary("B")`; new acceptance | **A** | `01-requirements.md` FR-K2, with `PK−1 = 3` **links**, ascending, `aircc` |
| 20 | `02-hld.md` §7.1 flip paragraph | **A** | Rewritten, with the measured ascending/descending evidence and the B-O5 closure |
| 21 | `04-test-plan.md` §4 W1-flip row → grid `(4,)`, L1 `12 288` | **A/C** | Applied; the **L1 arithmetic is corrected** — see §3 conflict C-2 |
| 22 | M4 §6.2 rewritten for the 1-D herd | **A** | §6.2 rewritten: `grid=(4,)`, `coords=("tx",)`, `TN=32`, the four channels, the ascending sites, the explicit `(m,n)` accumulate nest, the 4-trip `(i0,j0)` `air.sequential`, B-O5 closed |
| 23 | M4 §7 `test_M6_cascade_chain` ascends / `test_M6_cascade_orientation` descends | **A** | §7, both rows, with the P-R3 citation |
| 24 | M5 §6.2 rewritten for one coordinate | **A** | §6.2 rewritten; also shows the expanded accumulate loop (D-14) |
| 25 | M4 §3.6.3 measured-direction note | **A** | §3.6.3, plus `chain_direction` as a plan field |
| 26 | `03-lld-M8` §6.2: the printed split is the **tiled** one | **A** | §6.2, with the "`R_time = {}` reads as no local accumulation" explanation and a pointer to M4 §3.9 |

### W3

| # | Item | Status | Where it landed |
|---|---|---|---|
| 27 | M1 §6.3 → M8 §3.4's `sw(q, r, S)`; recompute the model | **A/R** | §6.3 rewritten. RULING 2's body is used — `sub = MATCH if q[i-1] == r[j-1] else MISMATCH` — not M8's older `2 - 3*min(d*d,1)`, so `params`, the access maps for `q`/`r`, and the substituted expression tree are all given |
| 28 | M1 §3.1/§3.6: the module-constant and transitive-bind rules | **A/R** | §3.1 gains **R-d** (`select` is an expression) and **R-e** (`fn.__globals__` ints) and the EBNF gains the `select`/`cmp` productions; §3.5 lines 15b-15f give transitive substitution in binding order with a cycle rule; §3.6 gains a note that `AFFINE` runs on subscripts only. FR-S3 gains items 6-8 |
| 29 | `02-hld.md` §7.3: three channels, delete "no per-core guard", L1 `240` | **A/R** | §7.3 **rewritten whole** under RULING 2 — it also needed `q`/`r` staging, `SOut`, the σ construction and the packet-flow endpoint argument |
| 30 | FR-M5 → three homogeneous channels | **A** | `01-requirements.md` FR-M5 text and acceptance |
| 31 | `04-test-plan.md` §4 W3 row → `240` with the breakdown | **A** | 04 §4 |
| 32 | M3 §6.4: add `q`/`r`; `l1_bytes = 232` | **A** | §6.4 rewritten; `q` is also shown **not** stationary (`ker M_q ⊄ ker Sπ_u`), so `stationary_ops = ("S","r")` |
| 33 | M4 §6.4 / M5 §6.4: `tensors=(q,r,S)`, `QIn`/`RIn`, `sub` from them, drain every row | **A/R** | Both rewritten. RULING 2's `SOut size=[PJ]` per-row drain is taken, so **FR-K4 is not restated**. The measured P-R4 sentence is added to both |

### Checker, mapper, emitter

| # | Item | Status | Where it landed |
|---|---|---|---|
| 34 | M4 §3.2 in `UCoord` | **A** | §3.2 lines 4, 11, 15 rewritten; a new paragraph states the two frames and that `pi_u`/`ker_pi_u` are now fields |
| 35 | M3 §3.3: no `clause`-stage code from a `legality`-stage check | **A** | §3.3 lines 5-7 → `LegalityError(PLACE-EXTENT)`; line 11 → an in-module `assert` (M2 §4 postcondition 1); a new paragraph states the rule |
| 36 | M3 §3.3: the placed-axis exclusion, worked | **A** | §3.3, with the `(0,−7)` vs `(0,1,−7)` arithmetic |
| 37 | FR-S17: the remaining σ rows; acceptance asserts `ScheduleModel.skew` | **A/R** | FR-S17 rewritten to RULING 6's wording (leading row = the sum; remaining rows exclude **placed and skewed** axes; lexicographic causality) |
| 38 | FR-L2 acceptance → `skew(time=(ax.i,))` | **A** | FR-L2, with the "reversal is a no-op" note and the `Sσ·d = (0,−7)` arithmetic |
| 39 | FR-L1 acceptance → `skew(time=(ax.i1, ax.j1))` | **A** | FR-L1, with the reason the old prose described a legal schedule |
| 40 | FR-S14: `pipeline` is recorded and read by nothing | **A** | FR-S14; M2 §3.7 and Q-M2-2 marked closed |
| 41 | FR-L9 acceptance → `192/96/32 f32` | **A** | FR-L9 with the full arithmetic; propagated to M3 §7, M8 §3.5, 04 §5 item 3 |
| 42 | FR-M2 acceptance → `[2 : index, 2 : index]` | **A** | FR-M2 |
| 43 | FR-M6 / FR-E8 → `PK−1` links + `ir_facts.cascade_channels == 3` | **A** | Both; FR-M6 also asserts `chain_direction == "ascending"` |
| 44 | M4 §3.8 retitled and rewritten over circuit-switched endpoints | **A** | §3.8: new `CIRCUIT()` predicate, hard-fail vs warn split, the full P-R2/P-R4 evidence, and the "a checker that rejects what the toolchain accepts is worse than no checker" argument |
| 45 | M4 §5, §10: delete the `PROTOCOL-UNSUPPORTED` fallback, close B-O6 | **A** | §3.8, §5 and §10; `PROTOCOL-UNSUPPORTED` now means only "no synthesis rule", including `reside(x="L2")` |
| 46 | M4 §3.3: the L2 rejection line and a `TENSOR_PLAN` pass | **A** | §3.3 lines 11a-11d and 24a-24n, plus notes 5 and 6; §3.1 gains the pass and the two names |
| 47 | M4 §3.9: the summary renders the **tiled** split | **A** | §3.9 line 10 and a new paragraph |
| 48 | M5 §3.1 uses `plan.launch_name`/`segment_name`; §4 P-1 strengthened | **A** | §3.1 lines 4-5, §6.1/§6.2/§6.3/§6.4 call sequences, §4 P-1 |
| 49 | M5 §3.2: a `BranchNode` row, row 11 rewritten, an L3 ordering assertion | **A** | §3.2 rows 1, 11, 11b, 12; §3.6 gains the whole `EMIT_EXPR` walk |

### Tests, requirements hygiene, plan

| # | Item | Status | Where it landed |
|---|---|---|---|
| 50 | 04 §5: 43 codes; item 1 → the dropped term; item 3 → EDIT-41's numbers | **A** | 04 §5 |
| 51 | M3 §7: "16 of the catalogue's 43"; a `(2,)` row; a `T = 0` case | **A** | §7 and §3.11 (two new rows, `npu1` and `npu2`) |
| 52 | M8 §4: a `w2_zero_t` negative fixture | **A** | §4, plus a `w2_pi4` `DMA-CHANNELS` negative and a paragraph saying why both exist |
| 53 | 04 §7: "at least one" | **A** | 04 §7, with the three-tests-carry-two-FRs reason |
| 54 | 04 §8 item 5: W2 within `1e-5` | **A** | 04 §8 |
| 55 | 04 §3.2: the two-pass ping-pong pipeline | **A** | 04 §3.2, with the 16-pass form named as the recorded fallback |
| 56 | 04 §3.4: restate `test_sem_coverage` | **A** | 04 §3.4, as a two-part assertion |
| 57 | 04 §4 W1-large: `bf16`/`f32` and the `f32` figure | **A** | 04 §4's larger-fixture paragraph (`49 152` undoubled, `81 920` charged) |
| 58 | `01-requirements.md` §6: R-18…R-21 | **A** | §6 |
| 59 | `01-requirements.md` §7: the ten LLD-local open questions | **A** | §7 now lists B-O1, B-O4…B-O7, Q-C4, Q-C8, Q-C15…Q-C17 with owner, due day and fallback; Q-7 marked answered by P-R1 |
| 60 | `01-requirements.md` §8: 70/70, FR-L14 in order, ids on one line | **A** | §8 rewritten; FR-L14 moved between FR-L13 and §3.3; FR-M9's split id joined |
| 61 | `00-README.md` §1: delete the stale paragraph, add the LLD rows | **A** | §1 gains rows 7–16 (nine LLDs + the review pair); the paragraph is gone |
| 62 | `00-README.md` §2, §3: M8 ownership; status rows | **A** | §2 M8 = "A (kernel sources, D0) + C (schedules, fixtures, demo)"; §3 gains LLD, review, flip and schedule rows and a fixture-parameter summary |
| 63 | 05 §2 D0: B's literal as data, A's `LegalMapping`, the `GAP` ruling | **A** | 05 §2 D0 row (tasks, outputs, checkpoint) |
| 64 | 05 §2 D1/D2: negatives to D3; `FOOTPRINT` into D2 | **A** | 05 §2 sequencing note and the D2 task cell |
| 65 | 05 §5: beat 3 "three schedule lines"; beat 4 "drop", not "reverse" | **A** | 05 §5 beats 3, 4 and 5 |
| 66 | 05 §7: LLD figures, 1 pd = 8 h, ≈11.1 pd, the flip is the cut | **A** | 05 §7 rewritten, including B's 4.3 pd and the two mitigations |
| 67 | 07 §2: gitignore, Python tag, one pinned interpreter | **A** | 07 §1 (the pin row) and §2 (items 3 and 4) |
| 68 | 07 §4: the correct flag; the two-pass pipeline with the long one as fallback | **A** | 07 §4 |
| 69 | 07 §3, §6: the `air-runner` model row; the repo layout | **A** | 07 §3 (`arch.json`, `Runner.cpp:543-551`) and §6 (`demo/`, `tests/helpers/`, `tests/integration/`, `kernels/rejections.py`) |
| 70 | `hackathon/VERIFIED-AIR-FACTS.md` line 844 flag name | **A** | Corrected to `--omit-ping-pong-transform`, with a one-line note that the earlier spelling was wrong |
| 71 | M8 §3.3: delete the FR-L14 contradiction paragraph | **A** | Replaced by the accept-and-peel statement plus the `w2_zero_t` sentence |
| 72 | M7 §6.1: `test_L2_causality` input and `mentions` | **A** | §6.1, `mentions=[(0, 1), (0, -7)]`, `details_keys` gains `representative` |
| 73 | golden names settle on `w1.base.*` etc.; vocabulary in 06 §8 | **A** | M5 §6.1/§7, M7 §6.2, M8 §6.1, 05 §2 D6; `06-interfaces.md` §8 states the closed `<variant>` vocabulary |
| 74 | add `test_M4_balanced` and `test_sequential_emits_scf_for` | **A** | M4 §7 and M5 §7. Nine further tests were added alongside them for the new behaviour — see §2, B-1/B-2/B-3/B-13 |

**Counts: 62 A, 11 A/R (items 1, 2, 13, 17, 27, 28, 29, 33, 37), 1 A/C (item 21), 0 N.**
*(Items 1, 2, 13, 17, 27, 28, 29, 33 and 37 are the nine A/R rows; items 12 and 19 were checked
against RULING 1 and RULING 4 and needed no modification, so they count as A.)*

---

## 2. Blockers B-1 … B-13

| # | Blocker | Closed by | How |
|---|---|---|---|
| **B-1** | W2's emitted program is not W2's kernel | EDIT-11/12/13/14/15/16/17/18 under **RULING 1** | `MappingPlan.tensors` for W2 is the kernel's **one** parameter `U: f32[T+1, H+2, W]`; the `U`/`Uout` rank-2 pair is deleted from M5 §6.3 and M4 §6.3. `UIn` stages plane 0 once; `UOut` drains the computed strip **every timestep**, so the L3 image covers planes `1..T` × rows `1..H` and `test_sem_coverage` compares the whole write domain. The stencil is the kernel's five-term `0.2 ×`. New tests: `test_M4_drains_every_plane`, `test_M4_stencil_is_five_point` |
| **B-2** | W3 exists as three incompatible kernels | EDIT-27/28/29/31/32/33 under **RULING 2** | One source, M8 §3.4's `sw(q, r, S)` with the `Select` body, is now in M1 §6.3, HLD §7.3, M3 §6.4, M4 §6.4, M5 §6.4 and 04 §4. `q`/`r` are staged (`QIn` broadcast, `RIn` per-PE slice) and read by the compute; `GAP` is a named constant; `SOut` drains every row. The `Zrow`/`Sink`/`Out` triple is gone. New tests: `test_M5_stages_q_and_r`, `test_M5_drains_every_row`, `test_E_select_emitted`, `test_grammar_select_expr` |
| **B-3** | `ComputeNode` cannot express any of the four kernels | EDIT-1 under **RULING 3** | `StoreNode` + the `ExprNode` tree carries all four: W2's five-term sum with a literal, W3's 4-ary `MaxMin` with a nested `Select`, the accumulator zeroing (an M4-produced `LoopPlan` of `StoreNode(Const 0)`, no `statement_index`), and the flip's whole-tile accumulate (expanded by M4 into an explicit loop nest). M5 §3.6 gives the structural `EMIT_EXPR` walk, so `test_emitter_makes_no_decisions` is satisfiable and D-14 is real |
| **B-4** | `PlanNode` has no branch region | EDIT-2 under **RULING 3** | `BranchNode(predicate, then, otherwise)` is a `PlanNode`; M5 §3.2 row 11 reads it; M4 §3.7.1's balance walk recurses into both arms (line 25-26a); §3.7.2's rules now say "`BranchNode` arm". Nested branches (the cascade's tail inside the middle's `otherwise`) are expressible. New test: `test_E_branch_node` |
| **B-5** | M4 reads `LegalMapping` in the wrong basis | EDIT-3 + EDIT-34 | `pi_u` and `ker_pi_u` are `LegalMapping` fields (`06-interfaces.md` §4.1) with a two-frame note; M4 §3.2 lines 4/11/15 are rewritten in `UCoord` with `root(axis_d)`; a new paragraph states that `pi`/`ker_pi` are used only for herd geometry. M3 §10 Q-M3-1 closed |
| **B-6** | W1-flip: four incompatible specs, wrong chain direction | EDIT-19…26 under **RULING 4** | One spec everywhere: `grid(PK=4)`, `place(px=ax.k0)`, `stationary("B")`, `TM=TN=32`, `TK=16`, chain **ascending**. Carried in FR-K2, HLD §7.1, 04 §4, M2 §6.2, M3 §6.2, M4 §3.6.3/§6.2/§7, M5 §6.2, M8 §3.1/§6.2 and 05 §2/§5 |
| **B-7** | `DMA-CHANNELS` is unsound as specified | EDIT-21(44) | M4 §3.8 counts **circuit-switched** core-to-core and non-packet L3 endpoints as errors and warns otherwise, with the P-R2/P-R4 measurements in the section; the catalogue row says "circuit-switched"; W3's three inbound is a warning and W2 at `PI=4` an error. LLD-B §5 amended. New test: `test_P3_dma_packet_warns` |
| **B-8** | W2's fixture takes five values | EDIT-11(12)…16 under **RULING 1** | One set: `T=4` (and `5`), `H=W=16`, `U[T+1,H+2,W]`, `PI=2`, `HS=8`, L1 `1 280`. Check (d) below prints every occurrence |
| **B-9** | `depends_on` refers to ids that do not exist | EDIT-4 | `ChannelSite.id = f"{channel}.{kind}.{order}@{scope}"`; M4 §3.1's name table carries the format |
| **B-10** | two codes raised that the catalogue lacks | EDIT-5, EDIT-6, EDIT-44 | Both are in §6.3; the catalogue is **43**; every "41" is fixed (check (e)). M1 §5's error table and §10 Q-M1-1, and M4 §3.8/§5/§10 B-O6, are updated |
| **B-11** | the headline rejection survives on one unstated rule | EDIT-7(36, 37, 38) under **RULING 6** | The placed-axis exclusion is in FR-S17, FR-L2, M3 §3.3 (with the worked `(0,−7)` vs `(0,1,−7)` arithmetic) and M8 §3.5; Q-C15 and Q-M3-4 closed |
| **B-12** | M3 raises `ClauseError` codes | EDIT-8(35) | M3 §3.3 raises `LegalityError(PLACE-EXTENT)` and asserts the rest; a new paragraph states "M3 raises only `legality`-stage codes"; the clause checks stay in M2 §3.4, which already had them. `assert_diagnostic`'s stage assertion now holds |
| **B-13** | `MappingPlan.tensors` has no construction rule | EDIT-9(7), EDIT-10(8), EDIT-46 | `06-interfaces.md` §5.6 invariant 6 states the law with the measured `_check_interface` message; M4 §3.3's `TENSOR_PLAN` pass constructs it (reads first, then writes) and is the only creator of an L3 `BufferPlan`; M5 §3.2 row 1 asserts it before emitting. New tests: `test_M4_tensor_order`, `test_M4_tensor_order_rejected`, `test_E_tensor_order` |

---

## 3. Conflicts between the review and a ruling, and one corrected number

| # | Conflict | Resolution |
|---|---|---|
| **C-1** | **EDIT-1** names the node `ComputeNode` with `ExprTree = Load \| Const \| Bin \| Neg \| Call`; **RULING 3** names it `StoreNode` with `ExprNode = Load \| Const \| BinOp \| MaxMin \| Select` | **Ruling wins.** The item's union has no `Select`, and RULING 2's W3 kernel requires one. Adopted: RULING 3's names and members, plus the item's `Neg` (harmless, and the grammar admits unary minus) and all three of its added sentences |
| **C-2** | **EDIT-21** gives the W1-flip fixture L1 as `12 288`, carried from M8 §3.1's `acc + 2a + 2b` | **Applied with a corrected derivation, same number.** M8 §3.1's figure omitted the cascade's `recv [32,32] f32` tile that M4 §6.2 synthesises, and counted `a`/`b` as doubled. Under `place(px=ax.k0)` **both** `A` and `B` are stationary (delivery depth 0), so neither can satisfy `isPingPongCandidate`: declaring `double_buffer` would earn a `PINGPONG-SHAPE` rejection (FR-S13, M4 §3.3 `PP()`). `schedule_ws` therefore **drops `double_buffer`**, and the figure is `acc 4096 + recv 4096 + a 2048 + b 2048 = 12 288` — the same total, now internally consistent. M3's own scope figure is `8 192` (it does not know about `recv`), stated in M3 §6.2. Recorded because it changes one line of a committed kernel source and the D6 ping-pong question it retires |
| **C-3** | **EDIT-17** offers "drain every plane **or** restate FR-K3"; **EDIT-33** offers the same choice for FR-K4 | **Rulings choose.** RULING 1 and RULING 2 both mandate the full drain, so neither FR is restated and `test_sem_coverage` keeps its strong form (EDIT-56) |
| **C-4** | **EDIT-13 / EDIT-29** are described as local edits to HLD §7.2 / §7.3 | **Applied as full rewrites.** Under RULINGS 1 and 2 those sections needed the staging and drain channels, the σ construction, the endpoint budgets and the packet-flow argument, none of which a numeric substitution reaches |
| **C-5** | **EDIT-27** says "recompute the model" against M8 §3.4's *existing* source (`2 - 3*min(d*d,1)`) | **Ruling wins.** RULING 2 replaces that body with the `Select` form, so M8 §3.4 was rewritten too and the model in M1 §6.3 is computed against the new body |
| **C-6** | **EDIT-21 / D-5** call the W2 strips `u`/`v` | **Renamed `cur`/`next`** throughout M4 §6.3, M5 §6.3 and HLD §7.2, per RULING 1's wording. `double_buffer("U")`'s D-5 meaning is unchanged |
| **C-7** | **RULING 7** (`reside(x="L2")` → `PROTOCOL-UNSUPPORTED`) has no edit-list item | Applied anyway: FR-S12 restricted to L1/L3 with L2 as future work, M4 §3.3 lines 11a-11d and note 5, new test `test_M4_reside_l2`, and the catalogue row reworded. This closes gap **G-5** |
| **Ruling 9** *(2026-09-13, after this response; supersedes **C-2**)* | C-2 left the flip spatially stationary but **not resident**: under `tile(ax.j, 32)` the declared `B` satisfied `ker M_B ⊆ ker Sπ`, yet its `[16,32]` tile moved with `j0`, so the weights were re-fetched every inner trip and "the weights stay put" would have been false on stage | **Predicate unchanged** (SD-02 §3). Two edits instead: (a) `MappingSummary` gains a **residency duration** per operand — `06-interfaces.md` §5.7 at `CONTRACT_VERSION = 3`, algorithm in M4 §3.9, test `test_M11_residency_line`, `00-README.md` §4 change-log row 3; (b) the flip fixture leaves **`j` untiled** (`TN = N = 64`), tiles `TM = 32` / `TK = 16` over `PK = 4`, and declares **`double_buffer("A")` only** — `A` streams over `i0` and is the one ping-pong candidate, `B` is resident so `double_buffer("B")` is a `PINGPONG-SHAPE`. C-2's `12 288` and "the flip declares no `double_buffer`" are **withdrawn**: L1 is **16 384** at M3's scope and **24 576** in the plan (`acc [32,64] 8192` + `recv [32,64] 8192` + `a [32,16] ×2 4096` + `b [16,64] 4096`). Cascade payload `8 192 B` per link per `i0` trip. W1's own output-stationary figures are untouched. Carried in FR-K2, FR-M11, HLD §7 and §7.1, M2 §3.6/§6.2, M3 §3.6/§3.13/§6.2, M4 §3.9/§6.1/§6.2/§7, M5 §6.2, M8 §3.1/§4/§5.1/§6.2, 04 §2/§4, 05 §5 and `00-README.md` §3/§4/§7 |

**Nothing in the edit list was declined.**

---

## 4. Checks

Six mechanical checks, run over `design/*.md` (excluding `REVIEW-round1.md` and this file) after
the last edit. Output pasted verbatim.

```text
(a) catalogue = 43 codes; used outside 06-interfaces §6.3 = 43; in catalogue but never used = none;
    used but not in catalogue = none
(b) 91 test ids named in 01-requirements; 0 with no definition in an LLD §7 or 04-test-plan: none
(c) 9 LLD §1 FR lists checked against 01-requirements' 77 FR/NFR ids; not a subset: none
(d) fixture parameters, every `NAME = value` occurrence across design/:
    M   : =64 (2x)
    TM  : =32 (1x)
    TN  : =32 (14x); =96 (5x); =64 (1x); =2 (1x)
    TK  : =16 (12x); =32 (6x); =64 (4x); =128 (3x)
    PI  : =2 (40x); =4 (36x); =3 (1x)
    PJ  : =4 (15x); =2 (12x); =1 (1x)
    PK  : =4 (15x)
    H   : =16 (6x); =18 (1x)
    W   : =16 (19x); =18 (5x); =10 (1x)
    HS  : =8 (18x); =4 (3x); =16 (1x)
    T   : =4 (18x); =5 (17x); =0 (14x)
    MQ  : =32 (3x)
    NR  : =32 (11x); =1 (1x)
    CW  : =8 (10x)
(e) stale-string sweep:
    0.25 (as a stencil coefficient)         03-lld-M4-mapping.md:1040, 03-lld-M5-emitter.md:550
    grid (4, 2)                             03-lld-M2-schedule.md:369, 03-lld-M3-checker.md:758,
                                            03-lld-M3-checker.md:891
    "H = 18" adopted anywhere               none
    "41 codes" / 41-code / catalogue's 41   none
    --omit-pingpong named as the flag       none
(f) fenced blocks by language: {untagged: 47, pseudo: 70, python: 13, bash: 5, mlir: 3, text: 2,
    ebnf: 1} — outside the allowed set (pseudo/text/mlir/ebnf/bash/json/untagged/kernel-surface
    python): none
```

**Reading (d).** Every W1/W2/W3/flip *positive* fixture parameter has exactly one value; the
second and third values are deliberate and each is a named negative fixture or a second fixture:

* `TN=96`, `TK=32`, `K=192` — `w1_l1_overflow`, the `L1-CAPACITY` negative (FR-L9, M8 §3.5).
* `TK=64`, `K=256` — `w1_large`, the smoke/IR-facts fixture (D-11, M8 §3.2).
* `TK=128` (3×) — three sentences saying the old FR-L9 case is **not** used.
* `PI=4`, `HS=4` — `w2_pi4`, the `DMA-CHANNELS` negative, plus LLD-B's probe log and the
  "not adopted" notes in M1 §6.2, M3 §6.3, M8 §3.3 and `00-README.md` §7 D-11.
* `PI=3` — one sentence in LLD-B saying `PI=3` would fail too.
* `PJ=2` — W1's `PI=PJ=2`, a different workload's parameter of the same name.
* `PJ=1` — `PI=PJ=1`, the one-PE grid `test_sem_compute_nodes` replays on.
* `H=18`, `W=18` (6×) — every one is a "the earlier `H = W = 18` proposal is **not** adopted"
  sentence in M1 §6.2, M1 §10, M3 §6.3, M8 §3.3, M8 §10 and `00-README.md` §7 D-11.
* `W=10` — `(HS + 2) × W = 10 × 16`, an arithmetic expression, not a parameter.
* `HS=16` / `NR=1` / `TN=2` / `TN=64` — `PI·HS = 16`, `MQ·NR = 1 024`, `M/TM × N/TN = 2 × 2`, and
  one sentence saying the earlier `TN = 64` is deleted: all arithmetic, not assignments.
* `T=5` — the `w2_odd` fixture; `T=0` — the `w2_zero_t` `SWAP-PARITY` negative.

**Reading (d) is as of this pass.** **Ruling 9** (§3, above) later changed the W1-flip's own
parameters — `TM = 32`, `TK = 16`, `j` untiled so `TN = N = 64`, `double_buffer("A")` — so the
flip's row of this check reads differently now: `TN = 64` is the flip's **value**, not an
arithmetic coincidence, and `TN = 32` belongs to W1 and to the `test_L6_cascade_rank` negative
only. The checks were not re-run.

**Reading (e).** The three remaining literal hits are all negative references:
`03-lld-M4-mapping.md:1040` and `03-lld-M5-emitter.md:550` are the sentences *"the earlier
four-term `0.25 ×` form in this section was a different program and is deleted"*;
`03-lld-M2-schedule.md:369` is the historical finding that concluded `grid (4, 2)` must be
rejected (now annotated as adopted under RULING 4); `03-lld-M3-checker.md:758` and `:891` are the
`test_L6_cascade_rank` **negative** fixture, which is deliberately `grid(4,2)`.
`--omit-pingpong` survives twice in `03-lld-M6-toolchain.md` (§3.2 and §8) and once in
`hackathon/VERIFIED-AIR-FACTS.md` §E.1, in each case as *"**not** `--omit-pingpong`"* — the note
G-15 asked for. No document names it as the flag to pass.

**Reading (f).** Every `python` fence is a kernel-surface listing (the three DSL kernel sources
and their schedules, M1 §6, M2 §6, M8 §3) or an upstream usage snippet (`07-environment.md`'s
`air-runner` invocation). Every untagged fence is a diagram, an IR dump, tool stderr, a
signature list, a plan listing or a shell transcript. **No module implementation code exists in
any fenced block in `design/`.**

---

## 5. What the review said still needs a device or a build, and still does

Unchanged by this pass, and now recorded in `01-requirements.md` §6 as risks R-18…R-21 and in §7
as open questions:

1. **N-3 / R-19 / B-O1** — whether a merged MM2S stream is correctly demultiplexed at two
   destinations. Unverifiable off-device; every core is kept at ≤ 2 outbound endpoints.
2. **The packet-vs-circuit rule (B-7)** — measured on four module shapes, not a documented
   upstream contract. `DMA-CHANNELS` stays a *warning* wherever packet flows may appear.
3. **`npu2` (B-O4 / R-21)** — nothing in the set has been built for it; four of the eight goldens
   target a device no probe has touched. Due D2.
4. **Ping-pong on the flip's streaming loop** — **re-opened by Ruling 9**, and narrower than
   before: there is no `j0` loop (`j` is untiled) and `B` is resident, so the only candidate is
   `A`'s alloc inside the 2-trip `i0` loop, which has the same structure as W1's `k0` loop.
   Like the base W1 case it rests on one `air-opt` run; if it does not fire, `schedule_ws` drops
   `double_buffer("A")` and `PINGPONG-SHAPE` says so at check time (M8 §7).
5. **W2 and W3 end-to-end numerics** — every structural check in `04-test-plan.md` §3.4 is a
   proxy. Only level D closes it, and the honest-limits slide says so.

---

## Final sweep checks (2026-09-13)

Run after the last edit of the sweep, over every `design/*.md`. `REVIEW-round1.md` is excluded
from (d) only — it is the review's own text and quotes the pre-ruling numbers by design.

**Two stale spots were fixed first**, both architect-verified:

1. **`03-lld-M2-schedule.md` §6.3** carried the pre-RULING-1 W2 schedule `grid(4)` / `tile(ax.i, 4)`.
   It is now `grid(2)` / `tile(ax.i, 8)`, with `grid = (2,)`, `tiles = (("i",8),)`, the RULING 1
   fixture stated (`U: sp.f32[T+1, H+2, W]`, `H = W = 16`, `PI = 2`, `HS = 8`, `T = 4`, odd-`T`
   variant `T = 5`), the derived extents `(4, 2, 8, 14)` and a pointer to M3 §6.3 for the
   `1 280 B` L1 figure. **§3.6's W2 row** was `(4, 4, 4, 16)`; it is now `(4, 2, 8, 14)` over the
   axis order `(t, i0, i1, j)` — M3 §6.3's values, adopted verbatim — with a paragraph under the
   table saying where each extent comes from (`T` written planes, `PI` strips, `HS` rows per
   strip, `W − 2 = 14` interior columns).
2. **`at=` pinning in the flip/cascade context.** Q-1 was *measured* (`03-lld-B-open-questions.md`
   §4): the chain compiles with no `at=`, and pinning only removes `air-place-herds`' freedom to
   avoid an occupied column. `03-lld-M3-checker.md` §6.2's cascade row now reads *"no `at=` pinning
   needed (Q-1 measured); `HerdPlan.at = None`"*, and §3.8's pseudo line 33 with it. Aligned in the
   same pass: **FR-L6**'s last sentence and its rationale, **FR-E8** (was "with `at=(column, row)`
   pinning recorded in the plan"), **R-03** ("restricts to a contiguous line" stays; "with `at=`
   pinning" becomes "without pinning", with `at=` kept as the named escape hatch), the **Q-1** row
   in §7 (now marked answered), `05-work-breakdown.md` D6 (`at=` pinning → `chain_direction`), and
   `03-lld-M5-emitter.md` §2 row 4 ("it always is" → "it is always `None`"). `03-lld-M4-mapping.md`
   §3.6.3/§5, which already said `at=` is not planned, needed no change.

```text
(a) catalogue = 43 codes; used outside 06-interfaces §6.3 = 43;
    in catalogue but never used elsewhere = none;
    code-shaped tokens used but not in the catalogue = none
      (the only non-catalogue UPPER-HYPHEN tokens anywhere in design/ are GIT-IGNORED,
       LEX-SMALLEST, MLIR-AIE, MLIR-AIR, NON-UNIFORM, PING-PONG, READ-ONLY, SPOT-CHECKS,
       VERIFIED-AIR-FACTS — none is an error code)
(b) 92 test ids named in 01-requirements.md; with no definition in an LLD §7 or
    04-test-plan.md: none.
    Four are prose prefixes of a defined id, not ids of their own:
      test_M9  -> test_M9_selfcheck_accepts / _rejects        (R-06)
      test_M10 -> test_M10_cycle_rejected / _w2_acyclic       (R-06)
      test_M11_summary -> test_M11_summary_golden             (FR-M11 prose)
      test_T3  -> test_T3_xclbin_message                      (R-11)
(c) 9 LLD §1 FR lists checked against 01-requirements' 77 FR/NFR ids:
      M1 10, M2 18, M3 20, M4 13, M5 10, M6 11, M7 6, M8 8, B 1 -> not a subset: none
(d) fixture parameters, every `NAME = value` occurrence across design/ (fences included):
    M =64(2)  N =64(12)  K =64(14)  TM =32(9)  TN =32(11)/=64(3)  TK =16(18)
    PI =2(47) PJ =4(15)/=2(14)  PK =4(20)  H =16(6)  W =16(22)  HS =8(24)
    T =4(23)/=5(19)  MQ =32(3)  NR =32(11)  CW =8(10)
    every other value is a named negative fixture, a second fixture, an arithmetic
    expression or a "not adopted / deleted" note -- itemised below. Conflicting
    occurrences without such a marker: 1 found, 1 fixed (00-README §5 D-11).
(e) "43" wherever the catalogue is counted (00-README ×3, 04 §5, 05 ×2, M7 ×2, M3 §7,
      M1 ×2, M4 B-O6); "41" only inside REVIEW-round1.md and this file's earlier
      check output, both historical records
    CONTRACT_VERSION = 3 in 06-interfaces.md §header and in every forward-looking
      quotation (00-README §3/§4, 02-hld D0, 05 D0, 01-req FR-M11, M4 §3.9);
      the surviving "= 2" quotations are historical (RULING 5's own text, M4 B-O6,
      this file's rows 11 and Ruling 9) and say which version a field landed in
    "H = 18" adopted anywhere .................. none (6 "not adopted / retired" notes)
    flip `TN = 32` ............................. none (flip is `TN = N = 64`, j untiled)
    `tile(ax.j, 32)` in a flip context ......... none (W1 base, the
      test_L6_cascade_rank negative, and "used to do / restored" regression prose only)
    `double_buffer("A", "B")` in a flip context  none (W1 base and the
      L1-CAPACITY negative only; M8 §3.1 annotates the flip line `was double_buffer("A","B")`)
(f) fenced blocks: {untagged 49, pseudo 71, python 13, bash 5, mlir 3, text 3, ebnf 1}.
    All 13 ```python fences, classified:
      03-lld-M1-frontend.md:429  kernel-surface listing (W1 source)
      03-lld-M1-frontend.md:457  kernel-surface listing (W2 source)
      03-lld-M1-frontend.md:488  kernel-surface listing (W3 source)
      03-lld-M2-schedule.md:325  schedule listing (W1)
      03-lld-M2-schedule.md:359  schedule listing (W1-flip)
      03-lld-M2-schedule.md:400  schedule listing (W2)
      03-lld-M2-schedule.md:429  schedule listing (W3)
      03-lld-M8-kernels-demo.md:70   kernel-surface listing (kernels/w1_gemm.py)
      03-lld-M8-kernels-demo.md:203  kernel-surface listing (kernels/w1_gemm_bf16.py)
      03-lld-M8-kernels-demo.md:231  kernel-surface listing (kernels/w2_jacobi.py)
      03-lld-M8-kernels-demo.md:312  kernel-surface listing (kernels/w3_sw.py)
      03-lld-M8-kernels-demo.md:395  schedule listing (kernels/rejections.py)
      07-environment.md:186          upstream usage snippet (air-runner)
    OTHER: none. No module implementation code exists in any fence in design/.
(g) 00-README §1 reading order: 16 rows covering all 9 LLDs (M1-M8 + B), REVIEW-round1
      and RESPONSE-review1 (row 16 carries both)
    00-README §3 status table: now states "The design set is ready for D0 handoff"
    00-README §7 adjudication: rulings 1-9 present (1-8 as table rows in §7.1, 9 as the
      binding paragraph below it; §7.1's title now reads "rulings 1-9")
```

**Reading (d).** The off-fixture values, every one accounted for:

* `M=N=K=192`, `TN=96`, `TK=32` — `w1_l1_overflow`, the `L1-CAPACITY` negative (FR-L9, M8 §3.5).
* `M=N=K=256`, `TM=TN=TK=64`, `PI=PJ=4` — `w1_large` / `w1_gemm_bf16`, the smoke and IR-facts
  fixture (D-11, M8 §3.2).
* `TK=128` (4×) — sentences saying the old `bf16` FR-L9 case is **not** used.
* `PI=4`, `HS=4` — `w2_pi4`, the `DMA-CHANNELS` negative fixture, plus LLD-B's probe log
  (finding N-1) and the "`PI = 4` is the negative, `PI = 2` is the fixture" notes in
  `01-requirements.md` §6/§7, M1, M3 §6.3, M4, M5 §6.3, M8 and `00-README.md` §5/§7.
* `PI=3` — two sentences saying `PI = 3` would fail for the same reason.
* `PJ=2` — W1's own `PI=PJ=2`. `PJ=1` — the one-PE grid `test_sem_compute_nodes` replays on.
* `H=18`, `W=18` — every occurrence is a "the earlier `H = W = 18` proposal is **not** adopted /
  is retired" sentence.
* `T=0` — `w2_zero_t`, the only reachable `SWAP-PARITY` negative once FR-L14 peels an odd `T`.
* `W=10`, `HS=16`, `NR=1`, `TN=2`, `TM=2` — arithmetic, not assignments: `(HS+2) × W = 10 × 16`,
  `PI·HS = 16`, `MQ·NR = 1 024`, `M/TM × N/TN = 2 × 2`, `M/TM = 2`.
* `TK=32` at `03-lld-M8-kernels-demo.md:77` is a scanner artefact: the line is the tuple assignment
  `TM, TN, TK = 32, 32, 16`, i.e. `TK = 16`.
* **The one real conflict, now fixed**: `00-README.md` §5's D-11 row stated the W2 fixture as
  `16×16 / PI=4 / T=4` with no supersession marker, while §7's D-11 row says RULING 1 replaced it.
  §5's row now carries the marker.

**Also corrected in this pass, outside (a)–(g) but found by them:**
`hackathon/HANDOFF.md` gave the W1-flip per-core L1 as `12 288` (the withdrawn C-2 figure) and the
contract as v2; both now read `24 576` / `16 384` at M3's scope and v3.

**One inconsistency left deliberately, because it needs an owner, not a sweep.** W3's `reside`
clause differs between documents: FR-K4 and `03-lld-M2-schedule.md` §6.4 say `reside(S="L1")`,
while `03-lld-M8-kernels-demo.md` §3.4 and §3.5 say `reside(S="L1", q="L1", r="L1")`. Nothing
downstream breaks — `03-lld-M4-mapping.md` §6.4 allocates `qb`/`rb` from the access maps and not
from `reside`, so the L1 total is `240 B` either way — but the committed kernel source and the
requirement should say the same thing. It is a one-line decision for A (does `reside` gate L1
staging for a read-only operand, or only declare a level?) and is recorded here rather than
guessed at.
