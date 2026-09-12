# ML Compilers Reading Group — Syllabus (Fall 2026)

**Goal.** Not "learn ML compilers." The goal is: **by Week 8, every student can write a
Spatial IR sketch for a real mapping and defend it; by Week 11, each student owns a work
package** from [`../proposals/spatial_dsl_project_proposal.md`](../proposals/spatial_dsl_project_proposal.md) §15
or from the [reviewer-handed openings](#3-research-openings-the-reviews-handed-us) below.

The reading group is the *ramp*, and it produces research artifacts as a side effect. We do
not do a "foundations semester followed by a research semester" — they run in parallel from
Week 1.

- **Cadence:** weekly, 90 minutes.
- **Start:** week of Mon 17 Aug 2026. 16 sessions ⇒ ends week of Mon 30 Nov 2026.
- **Two tracks per week:** a *paper* and a *build task*, ~2–3 h prep each. Nobody reads a
  paper they can't run something against.

---

## 1. The organizing principle: the PACT reviews are the syllabus

[`../PACT/56.txt`](../PACT/56.txt) contains four reviews of AIEHalide (accepted, PACT 2026)
plus our rebuttal. Read as a research plan rather than as feedback, it says:

| Reviewer demand | Our rebuttal committed to | Becomes |
|---|---|---|
| A-Q3: compare with **Interstellar** (ASPLOS'20) | "We will add the comparison." | **Wk 5** |
| A-Q2: why not a hybrid cost model like **TileLoom**? | "integrates naturally… future work" | **Wk 7** + build task |
| B-Q1, C: why not Halide beam search / **TVM–Ansor**? | "will position against Halide beam search and TVM/Ansor" | **Wk 6** |
| B-Q2: schedules that pass the model but **crash the backend** (NoC routing, L1 fragmentation) | "an explicit feedback loop is future work" | **Wk 12–13** |
| A-Q1, D: is **DAG recovery** fundamental? Halide flattens, we un-flatten. | defended the lowering window | **Wk 2, 8** |
| C, A: eval is **single-operator**; deeply fused pipelines untested | "we agree… will tone claims" | **Wk 14** |
| A, C-Q1: **15–30 % microkernel gap** (AIEVec `exp`, register tiling) | "limits of the shared vectorization path" | **Wk 15** |
| C-Q3: generalize to **Cerebras / Groq**-class arrays? | "a path exists" via the device model | **Wk 10** = the whole Spatial DSL proposal |
| B: "arguably a novel spatial NPU compiler that merely **uses Halide as a frontend**" | — | the strongest argument *for* Spatial IR |

That last one deserves emphasis. Reviewer B's sharpest criticism — that Halide is doing
less work than we claim and the real content is the spatial mapping layer — is precisely the
Spatial DSL thesis. **A reviewer independently arrived at our next project's premise.** Say
this in Week 1; it is the most motivating thing a student can hear.

---

## 2. Session format (90 min, fixed)

| Time | What | Who |
|---|---|---|
| 0–10 | **Build-task demo.** Last week's hands-on, screen-shared, working or broken. | rotating |
| 10–55 | **Paper presentation** — see the contract below. | rotating |
| 55–75 | **Ledger session.** Fill this paper's row in [the ledger](01-abstraction-ledger.md), out loud, together. | everyone |
| 75–90 | **Next build task + blockers.** | you |

### Presenter contract
Five questions. No paper summary, no related-work slide.

1. **What decision does this paper make explicit that others leave implicit?**
2. **What is its unit of work?** (loop nest? tile? tensor op? PE? actor?)
3. **Where does architecture leak in?** Point at the specific construct.
4. **Sharpest one-slide example** — the smallest program in the paper's own notation, and
   what the machine actually does.
5. **What would Spatial IR have to steal, and what would it have to refuse?**

Q5 is the point. The presenter drafts the ledger row before the meeting; the group edits it
live.

---

## 3. Research openings the reviews handed us

Each of these is a student-sized project that a PACT reviewer has already certified as
worth doing. Pair each with its week.

1. **Closed-loop autoscheduling** (B-Q2, Wk 12–13). Today the flow is open-loop: the model
   says "fits," the backend fails on routing congestion or L1 fragmentation, and we fall
   back to the next candidate blindly. Build the feedback edge — model the router, or learn
   from failures. Directly proposal RQ5.
2. **Hybrid analytical + measured ranking** (A-Q2, Wk 7). Our search returns a *ranked
   feasible list*; TileLoom-style top-k profiling bolts on cheaply and we already have ρ=0.9
   / τ=0.8 evidence. Cheapest publishable delta in the whole list.
3. **Deep fusion evaluation** (C, Wk 14). Two-stage blur and three-pass softmax are not
   "deeply fused ML pipelines." Attention blocks, fused transformer layers.
4. **Closing the microkernel gap** (C-Q1, Wk 15). AIEVec lowering of `exp`/reciprocal is
   incomplete; register tiling on irregular shapes is heuristic. Upstream-able work.
5. **The second architecture** (C-Q3, Wk 10). The device-model claim ("new tile, memory and
   channel parameters, not new passes") is *asserted* in the rebuttal and *unproven*. A
   Tenstorrent backend is the experiment that makes it true or false. This is proposal
   Phase 3 and the reason the Spatial DSL project exists.

---

## 4. The 16 weeks

**[B]** basics · **[P]** project-critical · **[F]** frontier. 📄 = PDF already in `../papers/`.

### Phase I — Ramp (Weeks 1–5)

| Wk | Track | Reading | Build task (due next week) |
|---|---|---|---|
| 1 | **[P]** Internal | **(a)** [`02-spatial-accelerator-landscape.md`](02-spatial-accelerator-landscape.md) — the six-machine comparison; sets the vocabulary and seeds the ledger. **(b) AIEHalide (PACT'26) + all four reviews + our rebuttal** (`../PACT/56.txt`). You present the compiler; a student presents the reviews. End on §1's table: every criticism is a project. | Install IRON/MLIR-AIE; run vector-scalar-mul and passthrough end-to-end on hardware. |
| 2 | **[B]** | **Halide** (PLDI'13) + scheduling-language reference. Read with A-Q1 in hand: *where exactly does Halide dissolve the producer–consumer graph?* | 3-stage blur, 4 schedules, measure, explain the crossover. |
| 3 | **[P]** | **IRON** 📄 `iron.pdf` + ISCA'25 tutorial 📄 (`../tutorials/`). Focus: ObjectFIFO as *the* abstraction. | ObjectFIFO producer→consumer chain across 3 AIE tiles. Undersize the FIFO deliberately; observe. |
| 4 | **[B]** | **MLIR** (CGO'21) + Toy tutorial ch. 1–5. Dialects, regions, progressive lowering. | A toy dialect: 2 ops + a verifier. Student A's core skill, learned in Week 4. |
| 5 | **[F]** | **Interstellar** (ASPLOS'20) — Halide's schedule language used to *classify* accelerator dataflows. The comparison we owe the camera-ready, and independent validation of the Level-2 thesis. | Express WS / OS / RS dataflows as schedules. Then: which of them can AIEHalide currently express? |

**Exit check:** everyone has run code on the NPU, can read a Halide schedule, and can state
the ObjectFIFO ↔ circular-buffer correspondence.

### Phase II — The core question (Weeks 6–11)
*Every paper here is a prior attempt at "decouple the computation from its spatial mapping,"
or a rival autoscheduler. Read as competitors, not as background.*

| Wk | Track | Reading | Build task |
|---|---|---|---|
| 6 | **[B/P]** | **Autoscheduling I.** Adams et al., *Learning to Optimize Halide Schedules* (SIGGRAPH'19) + **Ansor** (OSDI'20). The rebuttal's B-Q1 argument — soft learned cost vs. hard compile-or-fail constraints — must be defensible in detail. | Run Halide's autoscheduler on the Wk-2 blur; inspect its search space vs. ours. |
| 7 | **[P]** | **Autoscheduling II.** **TileLoom** 📄 `tileloom.pdf` + **Timeloop** (ISPASS'19) + **MAESTRO** 📄 `maestro.pdf` (notes started in `../papers/maestro.md`). | 🔴 **Deliverable:** add top-k hardware profiling to our ranked feasible list. This is opening #2 — a real result by Week 8. |
| 8 | **[P]** | **T2S-Tensor** (FCCM'19) + **SuSy** (ICCAD'20). Closest prior art to Spatial IR, six years older than ARIES. | 🔴 **Milestone (proposal §19):** take 3 mappings from AIEHalide, rewrite in a hypothetical Spatial IR, classify every decision domain / spatial / AMD-specific. |
| 9 | **[B]** | **Space-time mapping + polyhedral scheduling.** Two presenters: (a) Quinton (ISCA'84) & Darte/Robert/Vivien ch. 4; (b) Feautrier '92 Part I + Pluto (PLDI'08). Guide: `../spatial-dsl/02-spacetime-ir-and-reduction-legality.md` §2. | Derive (σ,π) for a 1-D systolic FIR by hand, then GEMM. |
| 10 | **[P]** | **Tenstorrent.** No paper — TT-Metalium docs + one tt-mlir pass, presented *as if* a paper. This is C-Q3 and proposal Phase 3. | Get TT hardware/simulator running; run `matmul_multi_core`. **Unblocks Student C.** |
| 11 | **[P]** | **The AIE compiler landscape:** **ARIES** (FPGA'25) 📄 (notes in `../docs/notes.html`) + **Allo** (PLDI'24) 📄 + **CHARM** (FPGA'23) 📄. | Same GEMM in Spatial-Triton (`../spatial-dsl/04-spatial-triton-dsl.md`) vs. raw IRON. Diff the concept count. **Assign work packages this week.** |

**Exit check:** the ledger has ~13 rows. The union of the "Spatial" column *is* the first
draft of the Spatial IR op set.

### Phase III — The open problems (Weeks 12–16)
*Straight from the reviews. Everything here is unclaimed.*

| Wk | Track | Reading | Build task |
|---|---|---|---|
| 12 | **[B/F]** | **Bounded buffers & deadlock.** Lee & Messerschmitt, *Synchronous Dataflow* (Proc. IEEE'87) + Kahn Process Networks. The formal foundation for RQ5 and for B-Q2. | Channel-dependency graph for the Wk-3 broken FIFO chain; exhibit the cycle. **Unblocks Student E.** |
| 13 | **[F]** | **Routing, placement & the open-loop failure.** Read the mlir-aie router (source, not a paper) + one NoC mapping/placement paper. The gap: "the model cannot model physical wire routing" (B). | Construct a schedule that passes C1–C6 and *fails* to route. Then characterize the failure boundary. Opening #1. |
| 14 | **[F]** | **Fusion & deep pipelines.** **Stream** (Mei et al.) or **FLAT**/**Chimera**; plus `../spatial-dsl/05-composition-and-fusion.md`. Answers C's "single-operator evaluation." | Fused attention block (QKᵀ→softmax→V) as Spatial IR; estimate, then measure. Opening #3. |
| 15 | **[P]** | **Vector microkernels — the 15–30 % gap.** AIEVec lowering + 📄 `gemm-npu-taka.pdf`, 📄 `spatial-matmul.pdf`, 📄 `systolic-aie.pdf`. | Profile one autogenerated vs. hand-tuned microkernel; attribute the gap instruction by instruction. Opening #4. |
| 16 | — | **Writing session.** No new paper. Read the ledger end to end; pick the venue; draft the abstract and the contribution list for the Spatial IR paper. | — |

**Spares** for a substitution or a 17th week: 📄 `asymmetric-buffering-npus.pdf`,
📄 `spatial-transformers.pdf`, 📄 `loopnest2silicon.pdf`, 📄 `aie4ml.pdf`,
📄 `Spatial_vs_GPU_effort_tradeoff.pdf`, **Spatial** (Koeplinger, PLDI'18 — best source on
escape-hatch design), **Exo** (PLDI'22).

---

## 5. The Ledger

One shared table, one row per system, filled live in the last 20 minutes of every session.
See [`01-abstraction-ledger.md`](01-abstraction-ledger.md).

**The rule:** a construct enters the *Spatial* column only if someone in the room can state,
out loud, how it is realized on **both** AMD AIE **and** Tenstorrent. If nobody can, it goes
in *Leakage*. That is proposal §14.1's test, executed 16 times with evidence — and the
Leakage column is the answer to RQ3.

---

## 6. Work packages and which weeks are mandatory

Assign in **Week 11**, not Week 1 — by then students have signal about what they're good at.
Until then everyone reads everything.

| Package | Source | Critical weeks | Owns in the ledger |
|---|---|---|---|
| **A** — Spatial IR design | proposal §15 | 2, 4, 5, 8, 9 | the whole thing |
| **B** — AMD backend refactor | proposal §15 | 1, 3, 11, 15 | leakage, AMD side |
| **C** — Tenstorrent backend | proposal §15 · opening #5 | 3, 10, 11 | leakage, TT side |
| **D** — Spatial optimization / DSE | proposal §15 · openings #1, #2 | 6, 7, 12, 13 | cost-model rows |
| **E** — Analysis & correctness | proposal §15 · opening #1 | 3, 12, 13 | buffer/deadlock rows |
| **F** — Microkernels & fusion | openings #3, #4 | 14, 15 | — |

Package F is not in the proposal; the reviews argue it into existence, and it is the most
self-contained entry point for a newer student.

---

## 7. Design rationale

You asked for a mix that doesn't cost research time. Concretely:

- **The "basics" are load-bearing.** Halide (2), MLIR (4), autoscheduler theory (6),
  space-time maps + polyhedral (9), SDF/deadlock (12) — each is a prerequisite for a
  *specific* work package and lands just before it's needed. Karp–Miller–Winograd, Lamport,
  and the rest of `../spatial-dsl/REFERENCES.md` cluster A stay off the syllabus; they are
  Student A's reference shelf.
- **Hardware access is Week 1.** NPU toolchain setup eats 2–3 weeks of calendar time
  regardless — start it while reading Halide, not after.
- **The reviews front-load the competition.** Interstellar (5), Halide-autoscheduler/Ansor
  (6), TileLoom (7), T2S-Tensor (8) are all systems a reviewer either named or would name.
  Meeting them in the first half is what protects the project from a late "this exists."
- **Two publishable deliverables inside the reading group itself:** the Week-7 hybrid
  ranking build task, and the Week-8 §19 exercise. Neither is homework — both are camera-
  ready or next-paper material.

### If you only have 10 weeks
Keep 1, 2, 3, 5, 7, 8, 10, 11, 12, 16. You lose the polyhedral foundations and the
microkernel work; RQ4 becomes future work.

### If a student joins late
Weeks 1, 2, 3, 8 in that order. Nothing else is a hard prerequisite.
