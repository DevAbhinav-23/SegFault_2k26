# RED TEAM — Round 2 against `01-paradigm-comparison.md` rev. 2026-09-12 r2

*2026-09-12. Attack only. Every finding quotes the sentence attacked, gives location, evidence
and why it matters. Hats: **[J]** venue jury · **[D]** domain expert · **[M]** methodologist ·
**[P]** prior art · **[C]** mechanical. Scope: NPUs only; the GPU deletion is not attacked.
Budget: 2,407 words of my own prose; a further 978 are verbatim quotation required by the
brief; tables excluded.*

**Headline.** The reviser did the work: 18 of 27 round-1 findings are fully discharged, 9
partially, **0 escaped**. But the repairs created new load-bearing failures, and two of them
are not fixable by wording. The r2 fix for K1/K2 — put-before-get — deadlocks under AIR's own
default channel depth, and P5's oracle (the property the whole recommendation rests on) does
not hold on W2.

---

## 1. KILLS

### K1 [D][J] The put-before-get halo protocol deadlocks at AIR's default depth 1.

**Attacked** (§3.2 preamble, L766–771): *"Both are satisfied by one discipline, used in all
five sketches: **each PE puts both of its boundary rows, and only then gets both of its ghost
rows, every timestep.**"*

**Evidence.** `xilinx.github.io/mlir-air/dev/AIRComputeModel/` fetched today and cross-checked
byte-for-byte against `raw.githubusercontent.com/Xilinx/mlir-air/main/docs/AIRComputeModel.md`.
Three sections the document quotes only in part. *Capacity and depth*: *"A channel has a finite
buffer capacity set by the `depth` attribute (default **1**)"*; depth 1 = *"Rendezvous: each
`put` must be consumed by a `get` before the next `put` can proceed."* *Flow control
semantics*: *"`put` and `get` are **blocking** at the channel boundary"*. And the page's own
*Minimal deadlock* example, on a fresh `air.channel @C [] {depth = 1}`, comments verbatim
*"// put blocks: channel is at capacity, waiting for a get to free space"* — i.e. the **first**
put on an empty depth-1 channel blocks. Under that reading the protocol is the textbook send-send deadlock. PE `p`'s `put(down)`
blocks until PE `p+1` issues its `get` — but PE `p+1`'s first action is its own blocking put.
Induct to PE `PI-1`, whose only put blocks on the already-blocked PE `PI-2`. No PE reaches a
`get`. All five W2 sketches have this shape.

**No escape exists inside the document's own findings.** §1.1 fact 1 establishes that
`air_ChannelOp` has no `depth` argument and that `air.api` rejects `buffer_resources` — *"the
objectFifo depth knob"* — by name, so no surface can raise depth to 2. The only route left is
the ping-pong passes, whose applicability the document marks `[UNVERIFIED]` (B15).

**The counter-reading is not adjudicated.** The stall rule — *"A `put` issued when the channel
already holds `depth` unread transfers stalls"* — implies the first put into an empty depth-1
channel does **not** stall, making the protocol safe. The page supports both readings and
contradicts its own rule with its own example; §1.1 fact 1 quotes *both* halves and reconciles
neither.

**Why it matters.** Put-before-get is the whole r2 answer to K1 and K2: the sole content of
W2·P1's step (6) and W2·P5's step (4), the basis of W2·P1 (e)'s *"strongest single argument for
P1 on this workload"* and of the M4 P5 cell's *"deadlock is not user-reachable while
`sp.exchange` is used"*. B12 hedges only **acyclicity**; **capacity** appears in no (d) or (e)
section.

### K2 [D] W2·P5's *"sequential oracle: yes, unconditionally"* is false.

**Attacked** (W2·P5 (c), L1145–1147): *"**(c) Sequential oracle: yes, unconditionally.**
`pe_id` iterates, `exchange` is a no-op (the strips are contiguous rows of one flat array in
the fallback), and the nest runs in CPython as a plain Jacobi."*

**Evidence.** `spatial-dsl/04` §4 fixes the fallback of `@sp.kernel(grid=(PI,PJ))` as *"loop
over all `(pi,pj)`"* — the grid loop **wraps the body** — and §3.0's P5 table repeats it.
W2·P5's body holds `for t in sp.stream(0, T, 1)` with `sp.exchange` inside, so in the fallback
`p` is outermost: PE 0 runs all `T` timesteps before PE 1 runs any, and PE 0's south ghost at
timestep `t` is a row PE 1 has not computed. The flat-array escape fails twice: the halo
dependence is bidirectional *within* each timestep, so no PE-outermost serialisation satisfies
it, and `u, v = v, u` swaps two named buffers, which views cannot be. Correct sequential
execution needs all PEs in lockstep — exactly what §3.0 charges against P2: *"**No plain
sequential fallback exists in this paradigm.** ... running it as a specification requires a
**deterministic actor interpreter**."*

**Why it matters.** The oracle is half of §1.4's delta over `air.api`, the whole of Candidate
B's reason to exist, the M4 P5 cell's claim to *"the 5-anchor's oracle clause outright and
unconditionally"*, and the demo's first 90 seconds (§6.5 D1 has W2 *"passing against numpy"*).
**P1's W2 oracle is genuinely unconditional** — `t` outermost, schedule strippable — so this is
a real P1-vs-P5 discriminator scored backwards.

### K3 [J] By §1.4's own test, the fallback the pick degenerates to is *"not worth a week"*.

**Attacked** (§1.4, L274–276): *"**So the defensible delta of any surface in this document over
`air.api` is exactly two things: (a) the program runs in CPython as its own specification, and
(b) spatial intent is declared rather than emergent.** Everything else is already upstream. Any
column that cannot claim both of those is not worth a week."*

**Evidence, all internal.** §6.2: *"**Candidate B** ... The P5 core alone, with delivery
*derived* and **nothing declared**"*. §6.3's `air.api` row, B column: *"the CPython oracle only
(§1.4)"*. §6.7: *"At **N = 1 this plan degenerates to Candidate B by design** ... and that is
the correct outcome, not a failure of the plan."* And §6.7 again: *"B's honest delta over
`amd/Triton-XDNA` is a Python oracle, and Triton has an interpreter mode."*

B claims one of two required properties, and that one the nearest shipped neighbour already
has — as `spatial-dsl/04` §2 concedes, calling the oracle *"(Triton's interpreter-mode
trick)"*. §8 quotes the venue FAQ: *"solo entries did fine — three of the six finalists were
teams of one."* **The modal case ships what §1.4 disqualifies.**

### K4 [M] M1 inverts between P1 and P5 against the document's own counting rule.

**Attacked** (§5, M1 P5 cell): *"Two capabilities missing from the stated vocabulary → 3, not
4"* — against (§5, M1 P1 cell): *"multicast is only *derived*, so the multicast⇄systolic lever
of `spatial-dsl/04` §3 has no surface, and non-neighbour communication (W4) needs an escape
hatch. ... nowhere near the 1-clause, which needs two absent"*.

**Evidence.** (i) P1's primitive table (§3.0) contains **no multicast construct**, and §2 says
P1's affine surface *"cannot state it without an escape hatch"* for W4 — two of six not
directly statable, the same count that sends P5 to 3. (ii) P5's §3.0 table **does** contain
`sp.exchange` (halo) and `sp.load(..., bcast=?, flow=?)` — precisely the multicast⇄systolic
lever P1 is docked for lacking (`spatial-dsl/04` §3: *"the surface must let you **default to
derived multicast but override to systolic forward**"*) — and §2 says P5 states non-neighbour
communication directly: *"P5 by indexing `A[pe_id() ^ (1 << s)]`"*. By the document's own
six-capability list, **P5 states 5 of 6, P1 states 4 of 6**; only skew is missing from P5.
(iii) P5's halo penalty is *"needed a primitive the user's own doc does not have"* — but §3's
preamble says *"**All surface syntax below is invented for this comparison**"*, and P1's
`s.window`/`s.exchange`/`s.forward`/`s.skew` are equally absent from `spatial-dsl/01`, whose
vocabulary table (L94–98) lists only `place`, `stream(broadcast|forward|cascade)`,
`stationary`, `reside`, `double_buffer`. The invention penalty is charged to one column.

**Why it matters.** M1 is half of §6.6's *"**M3 + M1 (thesis alignment)** → **P1 — Candidate
A**"* row — the weighting that produces the pick. P1 = 3, P5 = 4 flips it, and §6.7 with it.

### K5 [P][J] §6.3's AIEHalide row is built from 5 lines of a 304-line local file whose unread remainder contradicts it.

**Attacked** (§7.2): *"Source is a **local reviews file**, not independently verifiable online
... **Treat as the user's own in-submission work.**"* and (§6.3): *"**nothing novel — and that
is the point.** AIEHalide *is* the P1 design point ... the pitch should say 'our group
published the autoscheduled version; this is the ignorable-pragma surface over MLIR-AIR' and
not one word more"*.

**Evidence — `PACT/56.txt`, read in full, all verbatim.** **L300:** *"We are pleased to inform
you that your paper has been accepted."* — accepted at PACT 2026, not "in-submission".
**L249:** *"Rebuttal Response by Author [Abnikant singh <abnikant.singh@research.iiit.ac.in>]"*
— same institution as the user, different person; nothing in the file establishes *"the user's
own PACT '26 paper #56"*, yet §6.3 advises saying *"our group published"* on that basis.
**L256:** *"Standard directives still define the mapping, and the autoscheduler emits its
decisions through them; we add only three optional expert directives (`aie_dataflow`,
`aie_fuse_with`, `aie_kernel`)."* — **an ignorable-directive surface for AIE already exists and
is accepted**; that is Candidate A's surface layer, shipped. **L256/L259:** *"Bounds inference
yields the producer regions, **halos**, and per-tile working-set sizes our constraints need"* —
halos already derived, not written, which is exactly W2·P1 (e)'s *"strongest single argument
for P1 on this workload"*. **L256:** *"AIEHalide synthesizes this dataflow instead of
hand-writing ObjectFIFOs, DMA descriptors, placement, and host code in **MLIR-AIE**"* — it
bypasses AIR, like Dato; the document applies the "where does it lower" test to Dato (N9) and
never here. **L51–53:** *"53% peak on XDNA, 62% peak on XDNA 2 for int8 GEMM"* against
hand-tuned StB at *"66% on XDNA and 93% on XDNA 2"* — measured, on both target generations.

**Why it matters.** Candidate A then adds over AIEHalide **only** the AIR-vs-AIE target — the
same differentiator the document gives against Dato. "Declared spatial intent" is not available
as a delta against the user's own group's accepted paper.

---

## 2. WOUNDS

**W1 [M] M5 = 5 for P5 is unreachable from its own anchor.** §4: *"A **5** means ... ≤3 local,
syntactic column-specific steps"*. §5's own cell: *"**5** — 3 local syntactic steps on W1,
**4 on W2/W3**"* — it fails that clause on two of three workloads. M5 is what §6.6's *"**M5 +
M7 (the deadline binds)**"* row turns on, and a bolded cell in the executive summary.

**W2 [M] The F1–F3 floor is still not uniform — in both directions.** F1 is *"emit the
`air.launch`/`air.segment`/`air.herd` nesting and **memory-space annotations**"*. Yet P3's
counted step (1) is *"assign stages to herd/segment/launch scope by which memory level they
touch"*, P4's is *"identify the DATAFLOW region's task instances and their scope"*, P5's is
*"map `grid=`/`pe_id()` onto `air.herd`"* — all F1, none charged to P1 or P2. Token-edge
emission is likewise counted for P3/P4/P5 and free for P1/P2. Conversely **channel
materialisation is counted for P1 (step 5) and P3 (step 2) and appears nowhere in P5's three
steps.** Net noise ±2 on counts of 3–6 — the whole quantitative basis of an M5 spread of 2→5.

**W3 [J][P] Triton-XDNA ships a schedule surface; §6.3's claim holds of the language and fails
of the toolchain.** Attacked: *"Triton's SPMD model ... has no way to say what is
**stationary**, what is **multicast versus forwarded**, what the **halo** is, or what the
**wavefront skew** is"*. Repo enumerated (47 example kernels, 1.24 MB): decorators are
`@triton.jit`/`@triton.autotune` only, and **multicast / stationary / wavefront / skew /
objectfifo each occur zero times** — the four named words check out. But every example ships
`transform_aie2.mlir` / `transform_aie2p.mlir`, selected by `AIR_TRANSFORM_TILING_SCRIPT=`,
with phase headers verbatim *"PHASE 2: PROMOTE OUTPUT TO L2"* and *"PHASE 5: TILE FOR
MULTI-CORE PARALLELISM — Tile [16, 16, 0] for herd distribution"*; `memory_space = 1`/`2`
appears 146 times; `matmul_transform.py` exposes `l1_m`, `l1_n`, `l2_k`, documenting that
*"Parameter names align with mlir-air's programming_examples/matrix_multiplication
conventions"*. Residency and herd width **are** user-settable. The real delta is *binding*
intent to a tensor or loop rather than to a detached script — thinner. §6.7's *"a jury that
finds Triton-XDNA during Q&A ... has a two-sentence objection to B and a much harder one to A"*
is backwards for the flip demo: the answer is "swap the transform script".

**W4 [P] SpaDA — nearest neighbour to the document's own novelty claim — is still unread.**
§7.3: *"**Abstract level only — not read in primary form** ... Unassessed."* Abstract
(arXiv 2511.09447; Gianinazzi, Ben-Nun, Hoefler): *"precise control over **data placement,
dataflow patterns**, and asynchronous operations"*, *"demonstrated here with the **GT4Py
stencil DSL**"* — placement, dataflow and stencils: the claimed delta and W2. §7.4's own lesson
was *"A negative-results section is only as good as the column list it was written against"*;
the failure repeated one level down. Mitigation verified: SpaDA targets Cerebras CSL, not AIE.

**W5 [M][J] §6.1's exclusion arithmetic contradicts itself.** Verbatim: *"**P1-full (3–6 pw)
needs N ≥ 3.** ... P2 (3–5 pw) and P4 (4–7 pw) are out at any plausible *N*."* By §6.1's own
table N = 3 → 4.2 pw and N = 4 → 5.6 pw: P2's range fits **entirely** at N = 4 and at its
optimistic end at N = 3; P1-full's fits at neither. P2 is strictly cheaper at the pessimistic
end, equal at the optimistic end, yet excluded **on cost** while P1-full is carried. §6.2 gives
P2 non-cost reasons, so the conclusion survives; the sentence does not.

**W6 [D] `sp.flow` was redefined, and its stated fallback does not work.** §3 preamble:
*"including P5's `sp.exchange` and the receiving form of `sp.flow` (which `spatial-dsl/04` does
not have)"*. But `spatial-dsl/04` §4 defines `sp.flow(slice, dir="W→E")` as *"force systolic
forward instead of multicast"* — a delivery-mode override on an **operand load**, fallback
*"slice"*. W3·P5 uses `sp.flow(cur[CW], dir="W->E")` to send an arbitrary **computed scalar**
and `sp.flow(dir="W->E")` to receive one, fallback *"shift along the PE loop index"*: different
arity, object and fallback. (§3.0's P5 table marks **both** forms ✚ while the preamble marks
only the receiving one.) And the fallback is wrong: W3·P5 (c) claims *"`sp.flow` is a shift
along that loop index, and the result is the plain row-major DP"*, but with `p` outermost PE 0
emits all `MQ` boundary values before PE 1 reads the first — so it must buffer `MQ` per link,
and the order is **column-block-major**. Values come out correct (acyclic chain), so this is a
wound not a kill; but the table that makes P5's oracle "for free" does not describe what has to
happen.

**W7 [D][M] Every *"AIR catches it at compile time"* rests on a checker the primary source
never names.** The compute model says *"The compiler enforces the **static balance
condition**"* and *"A violation of the balance condition is a compile-time error"* with **no
pass and no verifier named** — while the same page names passes elsewhere and marks
unimplemented ones *"(planned: `air-cross-rank-dma-to-mgpu`)"*, and its one use of "verifier"
concerns `air.rank`, not channels. The document leans on *"AIR re-checks"* / *"compile time —
and AIR itself rejects it"* in at least eight (e) cells and in the M4 justifications for P1,
P2, P3 and P5. B12 hedges only whether the checker accepts a *shape*, never whether it exists.

**W8 [D][C] The stated fix for K2 violates a balance sub-bullet the document never quotes.**
§3.0: *"The lowering bug is **fixable** — peel the loop (prologue `put` + `T−1` in-loop puts +
`T` gets ...)"*. The balance requirement's second bullet, quoted nowhere in the document: *"For
channels inside loop bodies, balance must hold **per iteration** (equal puts and gets in the
loop body)."* `T−1` puts against `T` gets is unbalanced per iteration. Two further bullets are
also unquoted, including *"For channels inside conditional branches, balance must hold
independently on each branch"* — which bears on W3·P5's `if p < PJ - 1: sp.flow(...)`. N4's fix
reached the deadlock list and not the balance list.

---

## 3. NITs

| # | Verbatim / location | Evidence | Why |
|---|---|---|---|
| N1 [C] | §0: *"returns only this file, `REDTEAM-round1.md` and `HANDOFF.md`"* | Re-ran: returns `01-paradigm-comparison.md`, `REDTEAM-round1.md`, `RESPONSE-round1.md`. `grep -c -i akka hackathon/HANDOFF.md` → **0** | Count right, membership wrong; §0's one verified-by-command claim |
| N2 [C] | L851, W2·P1: *"the shortest W2 sketch"* (16 lines) | W2·P5 = 13, W2·P3 = 15, both counted by the document itself | A false comparative against its own numbers, two pages apart |
| N3 [C] | L1497: *"**Nine cells moved from r1**"* | `RESPONSE-round1.md` W2 lists **eight** (P4 M2, P2 M5, P3 M4/M5/M6/M7, P4 M5, P1 M7); column deltas confirm eight (P1 +1, P2 −1, P3 −4, P4 −2) | The two companion documents disagree on the size of the r2 rescore |
| N4 [P] | A51's Triton-XDNA quotes, presented as verbatim | README carries inline markdown links inside the first (*"using [Triton](…) and [MLIR-AIR](…)"*) and bold inside the performance claim | Mark markup-stripped; everything else in A51 is exact |
| N5 [D] | §1.1 fact 4 quotes both *"An endpoint in L3 ... has to sit inside an `air.segment`."* and the herd-body rule | `_channel.py`'s docstring is stricter than its own implemented check, which requires a segment only inside a herd body and says launch scope is *"fine with no segment at all"* | The two quotes disagree and the document presents them as one rule |
| N6 [D] | §1.4: *"there is no CPython run to diff against"* | mlir-air ships `python/air/backend/cpu_backend.py` (`AirCpuBackend`, LLVM-JIT refbackend, `air-to-async` → async-to-LLVM). Nothing in `python/air/api/` references it; it serves the legacy torch path | §1.4 survives as a claim about `air.api`, but a jury may not grant the distinction between `air.api` and mlir-air |

**No flaw found [D], B14.** §1.4's *"No sequential oracle"* claim is **correct** and should be
upgraded out of Appendix B. All 12 files of `python/air/api/` were enumerated and read.
`_compile.py`: *"The body is not executed when ``@launch.body`` runs -- it is recorded, and
replayed later"*. `_trace.py`: *"Nothing here emits IR until a herd body is registered"*.
`_value.py`'s `BufferExpr.__bool__` raises because *"a comparison against a coordinate has no
value at trace time"*. The only execution path is `compile()` → `XRTBackend` → xclbin →
hardware; `jit` is a raising stub; no `_interpret*`/`_sim*`/`_eval*` file exists. **Candidate
B's oracle delta over `air.api` is real** — which makes K3, not this, the live objection.

---

## 4. Escape table (27 rows)

| # | r2 disposition claimed | Verified | Evidence |
|---|---|---|---|
| **K1** cyclic graphs | fact 5 added; r1 text deleted; W2 redefined; all 5 sketches put-before-get | **TRUE** (new defect) | Fact 5 present §1.1; W2·P4 (e) correction present; all five W2 sketches put-then-get. But the replacement protocol is this round's K1 |
| **K2** `initial_tokens` | dropped from surface; balance quoted; "fixable by peeling" | **PARTIAL** | Dropped ✓ (§3.0, §6.4); `t=0` ghost bug named ✓. The stated peel fix violates *"balance must hold per iteration"* — an unquoted bullet of the same requirement (W9) |
| **K3** `depth` not an attribute | exact arg list; fact 1 rewritten; B15 added | **TRUE** | §1.1 table gives `$sym_name`/`$size`/`$channel_type`; *"There is no `depth` argument"*; A46/A47/A48 present |
| **K4** `air.api` has channels | fact 4 rewritten; new §1.4; B4 RESOLVED | **TRUE** | §1.4 present and correct; signature and scope docstrings verified against `_channel.py` today |
| **K5** deadline | new §6.1, division done, N cases, plans | **PARTIAL** | §6.1 present; arithmetic done. But it excludes P2 (3–5 pw) *"at any plausible N"* while carrying P1-full (3–6 pw) — W5 |
| **K6** third backend | GPU deleted; M6 anchor reworded; "no verified backend" | **TRUE** | 3 "GPU" lines remain, all scope/quote; M6 anchor reworded and flagged; §5 M6 note states it |
| **K7** P3 reduction hazard | `iters` mandatory; M4 5→4; Dato comparison; W3·P3 drain | **TRUE** | `@g.sink(inp=c_ch, iters=1)` present; §3.0 mandates it; `g.drain(w_ch[PJ], count=MQ)` present |
| **K8** M7 P1 mispriced | §1.3 re-prices; step 2 = check; 3–6 pw; 2→3 | **TRUE** | §1.3, W1·P1 (d) step 2 *"**check** `(σ,π)`"*, M7 cell all present |
| **K9** fifth column | P5 added, 3 sketches, scored, Triton-XDNA §7.2, §7.4 query | **TRUE** | All present. The new column carries this round's K2/K4 |
| **W1** ARIES fairness | applied uniformly; withdrawn as discriminator | **PARTIAL** | Fairness note present ✓. Problem moved: P5 is docked in M1 for inventing `sp.exchange` vs doc 04 while P1's `s.window`/`s.exchange`/`s.forward`/`s.skew` are equally absent from doc 01 and undocked (K4) |
| **W2** anchors decorative | every cell re-derived, naming its clause | **PARTIAL** | Every cell now names a clause ✓. ≥9 of 40 still unreachable — see §5 below |
| **W3** laundering + TAPA | qualifier restored **and carried into §5**; TAPA→P4 | **TRUE** | §0.1 present with counterfactual; W2·P4 (e) and the M1 P4 cell both carry *"in the canonical DATAFLOW form under the fixed mapping"* |
| **W4** floor not like-for-like | F1–F3 defined and excluded everywhere | **PARTIAL** | Floor defined ✓. F1 and token-edge emission still counted for P3/P4/P5 only; channel materialisation counted for P1/P3 only — W2 above |
| **W5** SDF unbounded buffers | both caveats stated; moot under AIR | **TRUE** | §3.0 and §7.1 both carry necessary-not-sufficient and the bounded-buffer obligation |
| **W6** equal-weighted total | relabelled; M5/M7 dependence stated 3× | **TRUE** | Row relabelled; paragraph present; §4 note and §6.6 row both repeat it. Totals re-added: 29/19/30/20/31 all correct |
| **W7** `Feeder` hoisting | corrected with verbatim source; P2 6→4 | **TRUE** | W1·P2 (d) quotes the real error text; *"Hoisting is a **choice**, not a legality requirement"* |
| **W8** occupancy test | AIEHalide added, §6.3 + §7.2 + A57 | **PARTIAL** | Added ✓. Built on 5 lines of a 304-line file whose remainder contains the acceptance notice, three `aie_*` directives, halo derivation and an MLIR-AIE backend — K5 |
| **N1** grep akka | §0 names the three files | **PARTIAL** | Count 3 correct; names `HANDOFF.md` (0 hits) instead of `RESPONSE-round1.md` |
| **N2** A12 URL/quote | versioned URL; quote re-verified | **TRUE** | Quote confirmed present at `docs.amd.com/r/2022.2-English/ug1399-vitis-hls/pragma-HLS-dataflow` |
| **N3** depth sourced to AIR.td | A4 relabelled prose; A46 gives TableGen | **TRUE** | A4 reads *"depth **semantics** (prose, not a dialect attribute — see A46)"* |
| **N4** A5 dropped a bullet | A5 carries **all** deadlock bullets | **PARTIAL** | Deadlock list complete and verbatim-correct (both bullets, confirmed live). Same behaviour recurs on the **balance** list, three of whose four bullets are unquoted — W9 |
| **N5** Spatial 2.9× | removed | **TRUE** | `grep "2\.9"` → one hit, inside B11's own record of the removal |
| **N6** `gpu_symmetric_heap` stale | dropped as out of scope | **TRUE** | Retained in one clause of §1.1 and A46, both marked out of scope; nothing depends on it |
| **N7** missing neighbours | Ripple + IRONSmith verified; SpaDA/TileLoom listed unread | **PARTIAL** | Ripple: authors, PACMPL 9 PLDI 2025 Art. 157, June 2025 and the abstract sentence all confirmed against the publisher PDF (DOI 10.1145/3729256). IRONSmith: arXiv 2607.10944, 2026/07/12, Sorenson/Ali/Bansil/Arora, both quotes verbatim in `citation_abstract`. But SpaDA is the nearest neighbour to the doc's own claim and is still unread — W4 |
| **N8** B3 Dato oracle | B3 RESOLVED; novelty withdrawn | **TRUE** | §6.4 reason 4 and B3 both present; A53 Allo quotes present |
| **N9** Dato lowers to AIE | quoted in §6.3 and §7.2 | **TRUE** | *"we leverage MLIR-AIE [53] as the backend"* quoted twice; A54 present |
| **N10** confirmed-not-findings | retained; flow-control cited | **TRUE** | W3·P1 (d) now cites the blocking-`get` semantics |

**Totals: 18 TRUE · 9 PARTIAL · 0 ESCAPED.**

---

## 5. Spot-checks

| Check | Result | Evidence |
|---|---|---|
| 16 fenced blocks; 15 in §3 + 1 in §6.4 | **PASS** | scripted extraction |
| All 16 per-block line counts | **PASS** | 18/24/19/31/9 · 16/19/15/27/13 · 14/16/16/26/14 · 20 — every stated figure exact |
| Column totals 48/59/50/84/36 | **PASS** | sums of the above |
| Algorithm/schedule splits (7+11, 7+9, 6+8) | **PASS** | counted by hand |
| *"the shortest W2 sketch"* (W2·P1 = 16) | **FAIL** | W2·P5 = 13, W2·P3 = 15 |
| *"exactly half of W1·P1"* (W1·P5 = 9) | **PASS** | 18/2 |
| §5 totals 29/19/30/20/31 | **PASS** | re-added |
| §5 M5 step counts vs the 15 (d) sections | **PASS** | P1 5/6/6 · P2 4/3/3 · P3 4/5/4 · P4 4/4/4 · P5 3/4/4 — all match |
| F1–F3 applied uniformly across 15 (d) sections | **FAIL** | W2 above |
| *"Nine cells moved from r1"* | **FAIL** | eight, per `RESPONSE-round1.md` and the column deltas |
| §0 `grep -ril akka` | **FAIL** | third file is `RESPONSE-round1.md`, not `HANDOFF.md` |
| §6.1 day count (12–18 Sept = 7) | **PASS** | inclusive count |
| §6.1 `7·N/5 = 1.4·N` pw; 10 h/day ≈ 1.75·N | **PASS** | 7/5; 70/40 |
| §6.1 P5/P3/P1 fit statements | **PASS** | consistent with the N table |
| §6.1 *"P2 … out at any plausible N"* | **FAIL** | 3–5 pw fits N = 4's 5.6 pw entirely |
| Dates (2026-09-12; 19–20 Sept; 15 Aug; 2–3 Oct) | **PASS** | internally consistent, matches §8 |
| A5 acyclicity bullet verbatim | **PASS** | matches raw `docs/AIRComputeModel.md` exactly |
| A5 covers the **balance** bullets | **FAIL** | three of four unquoted; one breaks K2's stated fix |
| A46/A47/A48 (`AIR.td`, `_channel.py`, `Passes.td`) | **PASS** | signature, `_UNSUPPORTED` keys, scope docstrings re-read today |
| A50 air-runner performance-simulator-only | **PASS** | `docs/AIRRunner.md` + `air/compiler/util.py` |
| A51 Triton-XDNA quotes | **PASS (markup)** | verbatim modulo inline links / bold |
| A51 "no spatial vocabulary" in Triton-XDNA | **PASS for the language, FAIL for the toolchain** | W3 above |
| A55 Ripple metadata + abstract quote | **PASS** | publisher PDF; only `Ripple`/`RIPPLE` small-caps differs |
| A56 IRONSmith arXiv id, date, authors, quotes | **PASS** | `citation_*` meta tags |
| A57 AIEHalide title + reviewer summary | **PASS** | `PACT/56.txt` L3–4, L15–24 |
| A57 *"in-submission"* / *"not independently verifiable"* | **FAIL** | L300: *"your paper has been accepted"* |
| B14 (`air.api` has no oracle) | **PASS — upgrade from `[UNVERIFIED]`** | 12 files enumerated; pure tracing builder |
| B12 (acyclicity applies to put→get edges) | **UNREACHABLE** | the page never defines *"nodes = ops"*; it names a separate *"async dependency graph"* for tokens and never relates the two |
| *"AIR itself rejects it"* (checker exists) | **UNREACHABLE** | no pass or verifier named on the page for balance or acyclicity |
| depth-1 halo protocol legality | **FAIL / self-contradictory source** | K1 |
| §5 M1/M2/M4/M5/M6 cells reachable from §4 | **FAIL on ≥9 of 40** | M1 P1=4, M1 P5=3, M2 P1=4 vs P5=4, M4 P5=3, M5 P5=5, M5 P1=2 vs P2=2, M6 P5=4, M7 P1=3, M8 P4=4 |

---

## 6. Rival statement

**Do not build a surface. Build the *check* nobody has, on top of the surface AMD already
ships, and make the demo a rejection.** Take `amd/Triton-XDNA` as given — SPMD onto MLIR-AIR
for AIE2 and AIE2P with a parity claim, its schedule already an editable `transform_aie2.mlir`
chosen by `AIR_TRANSFORM_TILING_SCRIPT`. Take AIEHalide as given — your own group's accepted
PACT '26 paper already has the ignorable-directive surface (`aie_dataflow`, `aie_fuse_with`,
`aie_kernel`) and already derives halos from bounds inference. Neither, nor Dato, nor IRON, can
tell you *before codegen* that a mapping is illegal: AIEHalide's rebuttal concedes an open-loop
flow whose failures *"appear as a compile-time routing failure"*, and the AIR compute model
states balance and acyclicity conditions with **no pass and no verifier named anywhere**. So
ship the missing verifier: read AIR IR, build the put→get graph plus per-iteration balance
counts, and answer three questions with a reason — is the graph acyclic, is every channel index
balanced on every path and per iteration, and does the halo protocol survive `depth = 1`
rendezvous. Four reasons this beats §6.7 before *this* jury. (1) A week of work, no new
language, no oracle, no invented primitives, buildable at N = 1. (2) One demo slide: feed it
the get-before-put Jacobi and watch it print *"cycle: PE 2 get(up) → PE 3 put(up) → PE 3
get(down) → PE 2"*. A rejection is more legible in five minutes than a passing numpy diff, and
it beats IRONSmith's canvas on content rather than losing on pixels. (3) It answers "why not
just `air.api`?" without a novelty claim: `air.api` will build your program, the model calls it
a compile-time error, and nothing checks. (4) Thesis-continuous the honest way — the `(σ,π)`
legality check is the rank/kernel arithmetic §6.2 wanted, arriving as a verifier instead of a
frontend. One scope change: settle the depth-1 question on day 1, because if put-before-get
deadlocks, every stencil in every one of these designs is wrong — and that finding alone is
the entry.

---

## 7. The one question the document cannot answer

> AIR's default channel depth is 1, your own §1.1 fact 1 proves there is no `depth` argument to
> raise and that `air.api` rejects the knob by name, and the compute model's own minimal-
> deadlock example comments that a `put` on a fresh `depth = 1` channel *"blocks: channel is at
> capacity, waiting for a get to free space"*. Your W2 protocol — the one r2 adopted in all five
> columns to escape r1's cycle — has every PE issue both of its puts before either of its gets.
> **Under the page's own example, who gets first?** And if the answer is "nobody, because the
> ping-pong passes will have raised the depth by then", what have you verified about those
> passes that B15 does not already say is unverified?

---

## 8. Verdict — does the loop stop?

**No. Three of the five kills need an experiment or a build; two need a rescore that changes
the recommendation. Wording cannot close this round.**

| Kill | Fixable by wording? | What it actually needs |
|---|---|---|
| **K1** depth-1 halo deadlock | **No** | An experiment. Either run a two-PE put-before-get exchange through `aircc` on an NPU, or get a ruling from the mlir-air maintainers on whether a `depth = 1` slot is real storage or a rendezvous. The page contradicts itself; no amount of hedging resolves it, and every W2 sketch in every column depends on the answer |
| **K2** W2·P5 oracle | **No** | A build — ten lines of Python. Run the W2·P5 fallback with `p` outermost and diff against a plain Jacobi. Expect a mismatch. Then either restrict the oracle claim to workloads with no intra-timestep inter-PE dependence (which removes W2 from the D1 plan and from the demo) or accept that P5 needs an interpreter on W2, which is P2's penalty |
| **K3** §1.4's test disqualifies the fallback | Wording only in the trivial sense | A **decision**. Either §1.4's two-property test is wrong, or Candidate B is not a legitimate fallback. The document cannot keep both. Note this is the N = 1 case, and §8 says solo entries win here |
| **K4** M1 inversion | **Yes — but it flips §6.6 and therefore §6.7** | Rescore M1 from the anchor with the invention penalty applied to both columns. Expect P1 = 3, P5 = 4, which inverts *"M3 + M1 → P1 — Candidate A"* |
| **K5** AIEHalide | **Partly** | Read the remaining 299 lines of `PACT/56.txt`, then re-derive §6.3, the M1/M4 P1 cells, and W2·P1 (e)'s "strongest single argument". The three `aie_*` directives and bounds-inference halos remove "declared spatial intent" as a delta against the user's own group |

**Two wounds also need work, not words:** W1 and W2 together mean the M5 spread of 2→5 — the
number §6.6's deciding weighting rests on — is smaller than the noise in its own construction.
A rebuild, not an edit.

**Settled; do not re-litigate.** B14 resolves in the document's favour. A5's deadlock bullets,
A46–A51, A55 and A56 are verbatim-correct against live sources. All sixteen line counts are
exact. The GPU deletion is complete. Round-1 K1, K3, K4, K6–K9, W3, W5–W7 and seven nits are
discharged.

**Recommendation: one more round, narrow.** Do not re-audit prose. Resolve two empirical
questions — does put-before-get deadlock at depth 1, does the W2·P5 fallback compute Jacobi —
then rescore M1 and M5 from the anchors. Both are hours, not days, and both are cheaper than
the pitch they protect.

---

*End of round 2. 5 kills · 8 wounds · 6 nits · 1 no-flaw-found. Escape table: 18 TRUE,
9 PARTIAL, 0 ESCAPED. Nothing in this file was implemented or measured; every external claim
was fetched live on 2026-09-12 and every local claim re-read from the repository.*
