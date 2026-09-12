# Work breakdown — 3 people, D0 → D7

*Phase 1, 2026-09-12. Scope is **not** reduced (architect brief D10). Effort figures are
**estimates**, labelled as such; nothing here is measured.*

## 0. Calendar

| Plan day | Calendar | Note |
|---|---|---|
| **D0** | Fri 12 Sep (evening) + Sat 13 Sep (morning) | install, freeze, sources |
| **D1** | Sat 13 Sep | modules start against stubs |
| **D2** | Sun 14 Sep | **W1 end to end** |
| **D3** | Mon 15 Sep | self-check + negative corpus |
| **D4** | Tue 16 Sep | **W3 end to end** |
| **D5** | Wed 17 Sep | **W2 end to end**; device window |
| **D6** | Thu 18 Sep | flip, goldens, **freeze at end of day** |
| **D7** | Fri 19 Sep | rehearsal + evaluation. **No code.** |

The final evaluation is listed as **19–20 Sept** (`CLAUDE.md`). The freeze is therefore at the
**end of D6**, not D7 — D7 is a rehearsal day that may already be an evaluation day. Anything
not green at the end of D6 is cut from the demo, not fixed on D7.

## 1. People

- **Person A — "Front & Law"**: M1 frontend, M2 schedule IR, M3 legality checker, the legality
  error-message catalogue, and the negative-test corpus for M1–M4.
- **Person B — "Mapping & Emit"**: M4 mapping/protocol synthesis, M5 AIR emitter, the AIR golden
  snapshots, and the `air.api` integration. **B is the critical-path owner** (§3).
- **Person C — "Toolchain, Tests, Kernels, Demo"**: M6 toolchain/runtime, M7 harness/CI, M8
  kernels/fixtures/demo, day-0 install on all three machines, device runs, the pitch.

---

## 2. Day by day

### D0 — install, freeze, sources

| | Person A | Person B | Person C |
|---|---|---|---|
| **Tasks** | Write the three kernel sources against the FR-S3 grammar; write a 20-line throwaway `ast.parse` script that proves each parses; write, by hand, the W1 `LegalMapping` literal; **rule on module-level `int` constants in an expression (`GAP`, `MATCH`, `MISMATCH`) — FR-S3 items 7-8, fallback "the kernel inlines the literals"**; co-sign `06-interfaces.md` **with every REVIEW-round1 edit and RULING 9's `MappingSummary.residency` already applied, at `CONTRACT_VERSION = 3`** | Write, by hand, one `MappingPlan` literal for W1 (the table in HLD §7.1) **as data, unvalidated** — M0 does not exist until D1; co-sign `06-interfaces.md` | Install the pinned toolchain on **all three** machines per `07-environment.md`; **cache the four wheels locally and record their sha256** (R-13); create the repo skeleton (`spatial/`, `tests/`, `fixtures/`); co-sign `06-interfaces.md` |
| **Inputs** | `01-requirements.md` §3.1, `06-interfaces.md` | `06-interfaces.md` §5, HLD §7 | `07-environment.md`, VF "Install path" |
| **Outputs** | `kernels/w1_gemm.py`, `w2_jacobi.py`, `w3_sw.py` (kernel bodies only); `tests/fixtures/plans/w1_mapping.py` | `tests/fixtures/plans/w1_plan.py` | three working installs; wheel cache; repo skeleton; `pytest` runs zero tests green |
| **Checkpoint** | all three sources parse; the `GAP` ruling is written into FR-S3 | the plan literal exists and reads as data (it is **validated against M0 on D1 morning**, not at D0 — M0 does not exist yet) | `air-opt --version`, `aircc --help`, `import air` exit 0 on 3/3 machines |

**Go/no-go G1 (end of D0)**: interfaces signed by all three **and** the toolchain works on at
least 2 of 3 machines. *If not*: C spends D1 on installs and the shared machine becomes the
build host; A and B continue, since M0–M4 need no toolchain (A-6).

### D1 — modules start

| | Person A | Person B | Person C |
|---|---|---|---|
| **Tasks** | M0 `model` dataclasses; M1 frontend (AST → `KernelModel`) for all three sources; M2 clause API with argument validation | M5 emitter skeleton driving **B's hand-written W1 plan** through `air.api` to text | M7 harness (marks, golden helper, `--update-goldens`); M8 fixture generator with fixed seeds; `expected.npz` from independent numpy |
| **Inputs** | — | M0 (from A, first thing) | M0 |
| **Outputs** | `model.py`, `m1_frontend.py`, `m2_schedule.py` + their unit tests | `m5_emit.py` producing AIR text for the stub plan | `conftest.py`, `tests/fixtures/**` |
| **Checkpoint** | the three `KernelModel`s match hand-written expectations field by field | `launch.build("npu1")` succeeds on the stub plan and `str(module)` is non-empty | `pytest` green; fixtures committed |

*Sequencing note*: M0 must land in the first two hours of D1, because both B and C block on it.
A writes M0 first, pushes, then starts M1. **A's D1 is positives only**: M0 (0.3 pd) + M1 (5 h) +
M2 (4 h) is ≈ 11.4 h against a day, so M1's and M2's 28-case negative corpora move to **D3**,
where §2's D3 row already schedules "the negative corpus". M1 §9 and M2 §9's definition of done
reads "D1 positives, D3 negatives". Both plan literals are validated against M0 this morning
(plan risk P-5).

### D2 — W1 end to end

| | Person A | Person B | Person C |
|---|---|---|---|
| **Tasks** | M3 checks L1, L2, stationarity, grid/place/tile consistency, herd physical shape, **and `FOOTPRINT` + L1 capacity — moved here from M3 §9 step 6 (D4-D5), because G2's checkpoint asserts `l1_bytes == 12288` and cannot be reached without it (≈ 1 h)** | M4 W1 path: trichotomy, broadcast channels, buffer plan, loop plan with the ping-pong shape; wire M5 to the real plan | M6 `aircc --output-format=none` wrapper with the **stderr** parser (FR-T5); the golden pipeline test |
| **Inputs** | M1, M2 | M3's `LegalMapping` (or A's hand-written one if M3 slips) | B's AIR text |
| **Outputs** | `m3_legality.py` (six checks) | `m4_mapping.py` (W1) | `m6_tools.py`, `test_golden.py` |
| **Checkpoint** | W1's `LegalMapping` matches the HLD §7.1 numbers (`l1_bytes == 12288`, `physical_herd == (1,2)` on npu1) | W1 from **source** to AIR text | `aircc --device npu1 --output-format=none` exits 0 with no `error:` line |

**Go/no-go G2 (end of D2)**: **W1 end to end**. *If not*: W1 becomes the entire emitted demo and
W2/W3 degrade to *checker-only* (their schedules are checked and rejected/accepted, with the
legality diagnostics as the demo) — which still carries the differentiator (G2) but loses two
lowerings. Decide at the D2 stand-up; do not let it slide to D3.

### D3 — self-check and the negative corpus

| | Person A | Person B | Person C |
|---|---|---|---|
| **Tasks** | The negative corpus: one test per `GRAMMAR-*`, `CLAUSE-*`, `L*` code; hand-tune the three demo rejections (§5) | M4 **self-check**: balance P1′ and acyclicity P2b over the plan, plus the corrupted-plan negatives | The IR-inspection tests (`unroll = 2`, broadcast count, the unpaired-channel characterisation test); W1 oracle diff |
| **Inputs** | M3 | M0 §5.6 invariants; `docs/AIRCorrectnessChecker.md` §4–§5 | B's emitted W1 |
| **Outputs** | `tests/negative/**` | `m4_selfcheck.py` | `tests/test_ir_facts.py`, `w1.ir_facts.json` |
| **Checkpoint** | every code in the catalogue fires with all four message parts | all six corrupted plans are rejected with the right code | `unroll = 2 : i32` is present on W1's K loop |

### D4 — W3

| | Person A | Person B | Person C |
|---|---|---|---|
| **Tasks** | The causality check's message (the demo's headline rejection); `skew` handling in `σ` | M4/M5 **W3**: the three homogeneous channels `WestIn[1]`/`West[PJ-1]`/`EastOut[1]`, `q`/`r` staging, segment-scope source and drain, per-row get/put and per-row `SOut` drain, `prev`/`cur` swap | W3 fixture + oracle + golden + `aircc` smoke; start the device attempt if hardware exists |
| **Inputs** | M1's dependence vectors | M3's W3 `LegalMapping` | B's W3 text |
| **Outputs** | `L2-CAUSALITY` with the violating vector printed | W3 plan + text | `w3.*` goldens |
| **Checkpoint** | the reversed-skew schedule is rejected and the message names the vector | W3 self-check reports 5 balanced channel indices | W3 `aircc --output-format=none` exits 0; oracle diff exact |

**Go/no-go G3 (end of D4)**: W3 end to end. *If not*: W3 degrades to checker-only and the
causality rejection is still the demo's headline; B moves straight to W2 on D5.

### D5 — W2, and the device window

| | Person A | Person B | Person C |
|---|---|---|---|
| **Tasks** | The halo-footprint derivation and `HALO-TOO-SMALL`; the mapping-summary renderer (data-level, unblocks B) | M4/M5 **W2**: `ToNorth`/`ToSouth` bundles, guarded put-first protocol with `ops.branch`, async puts and dependency-free gets, the `u`/`v` pair | W2 fixture + oracle + golden + `aircc` smoke; **the device window** — if hardware exists, `output_format="xclbin"` and run W1 |
| **Inputs** | M1's access maps | M3's W2 `LegalMapping` | B's W2 text |
| **Outputs** | `m4_summary.py` | W2 plan + text | `w2.*` goldens; a device result or a recorded reason for its absence |
| **Checkpoint** | `halo=0` is rejected with the derived footprint printed | the W2 site order is `[PUT, PUT, GET, GET]` with `depends_on == ()` on both gets | W2 oracle diff exact; device result or documented skip |

**Go/no-go G4 (end of D5)**: W2 end to end **and** the device question answered either way.
*If W2 fails*: it degrades to checker-only. *If the device is absent*: the demo's end-to-end
claim stops at `aircc --output-format=none`, and the honest-limits slide says so — this is
planned for, not a failure (R-09).

### D6 — flip, goldens, freeze

| | Person A | Person B | Person C |
|---|---|---|---|
| **Tasks** | Finish the negative corpus; `test_D3_catalogue_complete`; `test_traceability`; message polish | **W1-flip**: cascade chain, `R_space` path, `chain_direction` (no `at=` pinning); regenerate all goldens | Full-suite green under 3 min; determinism gate; the pitch script; the honest-limits slide; rehearsal #1 |
| **Inputs** | everything | A's `R_space` split | everything |
| **Outputs** | corpus complete | `w1.flip.*` goldens | the deck, the demo script, CI green |
| **Checkpoint** | 43/43 codes covered | the flip emits `PK−1 = 3` ascending cascade links on one `npu_cascade` bundle (`ir_facts.cascade_channels == 3`) and `aircc` accepts | **FREEZE at end of day** |

**Go/no-go G5 (D6 midday)**: the flip. It is a **stretch item** — if it is not green by midday
on D6, cut it and use the remaining D6 to harden the three kernels and the messages. Cutting it
costs one slide, not the entry. *Contingency before cutting*: re-emit the same chain on the
default `npu_dma_stream` channel type (FR-K2's contingency) — identical protocol and balance,
different physical link, one-line change in the plan.

### D7 — rehearsal and evaluation. No code.

All three: rehearsal #2 and #3; Q&A drill against the questions in §6; submission.

---

## 3. Critical path

```
M0 frozen ─▶ M4 W1 ─▶ M5 W1 ─▶ aircc smoke ─▶ M4/M5 W3 ─▶ M4/M5 W2 ─▶ M1-flip ─▶ freeze
  D0           D2       D2        D2             D4            D5          D6        D6
```

**Person B owns every box on it after D0.** Three consequences, all acted on in the plan above:

1. **B starts on D1 against a hand-written plan**, not against A's modules, so an M1/M2/M3 slip
   does not move the critical path.
2. **Work that is not on the path is moved off B**: the mapping-summary renderer (A, D5), the
   golden harness and every IR-inspection test (C, D2–D3), the corrupted-plan negatives (B
   writes the check, A writes the corpus).
3. **The emitter makes no decisions** (HLD §2, M5). This is what keeps M5 a one-day task rather
   than a second mapping module.

A's path (`M1 → M2 → M3 → messages`) has two days of slack before it blocks B at D4; C's path
runs in parallel throughout and blocks only on B's text.

---

## 4. Stretch items

| Item | Owner | Decision point | Cut cost | Contingency |
|---|---|---|---|---|
| **W1 weight-stationary flip** (cascade, `R_space`, `j` untiled so the weights are resident) | B | D6 midday | one slide, and the "flip the clauses, same kernel" story — the most persuasive 30 seconds of the pitch | `npu_dma_stream` chain instead of `npu_cascade` (FR-K2) |
| **Device run** | C | D5 | the end-to-end claim shortens to `aircc --output-format=none` | say so on the honest-limits slide |
| **`air-runner` trace** for one kernel | C | D6 | a nice visual, nothing more | omit; it is a timing model, not an oracle (VF §S7), and claiming otherwise is worse than omitting it |
| **W4 FFT** | — | not attempted | none — it is a declared non-goal | §2 of the requirements says why |
| **`test_I_lock_inits`** (lock histogram after `air-to-aie`) | C | D6 | one number in `ir_facts.json` | omit |

---

## 5. What is demoed on D7

**Five minutes, in this order.**

1. **(45 s) The claim.** A plain Python loop nest and a separate list of schedule clauses.
   Delete the clauses, the program runs in CPython, and that run **is** the specification.
2. **(60 s) W1.** Show `gemm`, show the schedule, run `s.summary()` — the audience reads
   `C: stationary (derived)`, `A: multicast along py (derived)`, `B: multicast along px
   (derived)` off the screen. That is the ARIES criticism answered out loud: the dataflow is
   named, not emergent.
3. **(45 s) The flip.** Change **four schedule lines** — `grid(PK)`, `place(px=ax.k0)`,
   `stationary("B")`, `double_buffer("A")` — and drop `tile(ax.j, TN)` so each PE holds the
   whole `B` row-block, **with no edit to the kernel**. The summary now says
   `B: stationary (declared)`, `B: stationary (spatial), resident for the whole run`,
   `A: stationary (spatial), re-fetched per i0`, `R_time = span{e_k1}, R_space = span{e_k0}`,
   and `CascadeK size=[3] type=npu_cascade`: `PK−1 = 3` cascade **links** on one bundle,
   ascending. The line to say out loud: **the weights stay put; `A` streams; partial sums
   cascade** — and the residency line is the checker saying so, not us (RULING 9).
   *(Cut if G5 said so.)*
4. **(90 s) The differentiator — rejection before codegen.** **Drop the `j0` term from the
   `skew` on W3** — `skew(time=(ax.i,))`. (Do **not** say "reverse": `skew` is a sum, so
   reversing the terms is a no-op and the schedule stays legal. The dropped term is also the
   mistake a user actually makes.) The
   checker prints the code, the clause, the violating dependence vector, and the fix. Then say
   the sentence that matters: **mlir-air ships a 511-line specification for this checker
   (`docs/AIRCorrectnessChecker.md`) and has not implemented it — `mlir/lib/Analysis/` does not
   exist, and `air-opt` prints its one channel diagnostic while exiting 0.** The idea is
   upstream and public; the implementation is ours.
5. **(60 s) It lowers.** W2's halo exchange at `PI = 2`: two link bundles, put-first,
   per-iteration balanced, no prologue, every plane drained, `aircc --output-format=none`
   exits 0. Show the `unroll = 2 : i32` the ping-pong
   labeller wrote on W1's K loop — our emitter produced the shape the pass requires.
6. **(30 s) Honest limits.** The slide below, read, not skipped.

### The honest-limits slide (verbatim content)

- **One lowering path, two device flags.** `air-to-aie` → MLIR-AIE; `npu1` (Phoenix, AIE2) and
  `npu2` (Strix, AIE2P). Versal values are accepted by the pass but **not** by `air.api`'s
  `resolve_target`, so we do not claim them.
- **Compute is scalar.** Per-PE bodies are native `air.sequential` loops with element
  load/store. An `air.extern` vectorised kernel would be faster and is future work — and would
  *disable* the ping-pong passes on any buffer it touches first.
- **No performance numbers.** Optimisation passes are out of scope; we measured nothing.
- **`air-runner` is a timing model, not a correctness oracle.** We do not use it as one.
- **Semantics off-device are established structurally**, not by executing the emitted IR:
  `air.api` has no interpreter, and the CPU backend JITs the *lowered* module, which tests the
  compiler's output rather than the user's source. Only the device run closes that loop.
  *(Adjust this line to what D5 actually achieved.)*
- **The checker's idea is not ours; its implementation is.** AMD specified it
  (`docs/AIRCorrectnessChecker.md`, properties P1–P4) and did not build it.
- **Nearest neighbours, named**: `amd/Triton-XDNA` (SPMD → MLIR-AIR for AIE2/AIE2P), **Dato**
  (typed streams → MLIR-AIE, already rejects deadlock and inconsistent put/get), **AIEHalide**
  (PACT 2026, ignorable directives and derived halos, targets MLIR-AIE), **IRON/ObjectFIFO**,
  **ARIES**.
- **Out of scope**: W4 FFT (the partner map `p ↦ p ⊕ 2^s` is affine in neither `p` nor `s`),
  multi-kernel fusion, autotuning, GPUs.

---

## 6. Q&A drill (2–3 minutes, prepared answers)

| Question | The answer, in one breath |
|---|---|
| *"Why not just use `air.api`?"* | Two things it does not give: the program does not run as its own specification, and nothing in it says *why* a transfer is shaped that way. Everything else — channels, broadcast, ping-pong, scope validation — is upstream and we use it unchanged. |
| *"Doesn't AIR already check balance?"* | Its compute model says a violation is a compile-time error. No pass implements it. We fed three violations through the real `air-opt`: one printed an error **and exited 0**; two were silent through `air-to-aie`. |
| *"Is the halo protocol safe?"* | Yes, and it was checked four ways, one of them measured: every producer lock `air-to-aie` emits starts at ≥ 1, never 0; the lock allocator does that by construction; the ObjectFIFO path acquires an *empty* buffer, not a consumer; and upstream ships a hardware-CI ring of exactly this protocol. |
| *"How is this different from AIEHalide?"* | It derives halos too, and it ships ignorable directives — so declared intent is not the delta. The delta is the pre-codegen legality check: an illegal wavefront skew is rejected with the violating dependence vector before any IR exists. |
| *"Performance?"* | We measured none and claim none. Optimisation is out of scope; the compute bodies are scalar. |
| *"Does it run on hardware?"* | *(Answer from D5's actual result. If no device: "It compiles through `aircc` end to end off-device; we have not run it on silicon, and the slide says so.")* |

---

## 7. Effort estimates (labelled estimates, not measurements)

**1 person-day = 8 hours.** These are each module's own LLD §9 figure, not a second independent
estimate; where the two disagreed, the LLD's wins, because it is the one derived from a step list.

| Module | Estimate | Basis (the LLD §9 step list) |
|---|---|---|
| M0 `model` | 0.3 pd | ~15 frozen dataclasses with invariants |
| M1 frontend | 0.6 pd (5 h) | AST walk + affine-expression extraction + the rejection corpus |
| M2 schedule | 0.5 pd (4 h) | 14 clauses, argument validation |
| M3 legality | 1.6 pd (10 h) + 0.4 pd (3 h) corpus | 11 checks over small integer matrices, plus messages |
| M4 mapping | **3.3 pd** | three protocols + the self-check + the tensor-ordering and expression-tree work; **the critical path** |
| M5 emitter | 1.0 pd | mechanical translation, no decisions |
| M6 toolchain | 1.0 pd | subprocess wrappers, the stderr parser, the device path |
| M7 tests/CI | **1.7 pd** | harness + goldens + the 43-code corpus scaffolding (its own §9 steps sum to 1.7, not 1.5) |
| M8 kernels/demo | 1.0 pd | three sources, fixtures, the deck |
| **Total** | **≈ 11.1 pd** | against 3 people × 7 days ≈ 21 pd at a conservative 50 % utilisation ≈ **10.5 pd** |

**The plan is 0.6 pd over budget before the first line is written, and the flip (G5) is the
designated cut.** Say that at the D2 stand-up, not at D6 midday. Two further moves are already
scheduled: M4 §3.9's summary renderer goes to **A** (§2 already gives A the renderer on D5), and
M7 step 10 (0.2 pd of budget tuning) slides to the D7 morning. B's own load is M4 3.3 + M5 1.0 =
**4.3 pd** against ≈ 3.5 pd at 50 % utilisation, and B-6's rewrite of the flip plan lands on D6,
the stretch day — which is the concrete reason G5 exists.
