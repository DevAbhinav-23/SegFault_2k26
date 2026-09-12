# RED TEAM — Round 1 against `01-paradigm-comparison.md` + `HANDOFF.md`

*2026-09-12. Attack only. Every finding quotes the sentence it attacks and shows its evidence.
Hats: **[J]** venue jury · **[D]** domain expert · **[M]** methodologist · **[P]** prior art · **[C]** mechanical.*

---

## 1. KILLS

### K1 [D] AIR bans cyclic channel graphs. The "sharpest expressiveness result" is half a bullet list.

**Attacked** (§3.2, W2·P4 (e), L790–794): "**AIR has no such restriction**: `air.channel` may be cyclic, and the AIR compute model supplies a *condition* for when a cycle is safe … instead of banning cycles. … This is the sharpest expressiveness result in the document."

**Evidence.** Fetched `xilinx.github.io/mlir-air/dev/AIRComputeModel/` today, stripped to text. The quoted condition is item 1 of a **two-item list** under *Deadlock conditions*. Item 2, omitted, verbatim: "The communication graph (nodes = ops, edges = channel put→get dependencies) must be **acyclic**: a cycle means at least one op in the cycle is waiting on another in the same cycle, which can never be resolved."

**Kill.** (a) P4's `M1 = 2` is half-justified by "cannot express W2 at all (*Feedback between tasks*)" — the target imposes the same restriction, so it is not a paradigm difference. (b) W2 was chosen *for* the cycle (§2: "the single most instructive structure in this document"); it has no legal AIR lowering under the mapping held constant. (c) W2·P3 is the sole evidence for `M4 = 5` and is that cycle.

### K2 [D][C] `initial_tokens` lowers to a guaranteed AIR compile error.

**Attacked** (W2·P3 (d), L715): "`initial_tokens` → one prologue `air.channel.put` before the `scf.for`".

**Evidence.** Same page, *Balance requirement*: "Along every possible execution path … the number of `put` operations and the number of `get` operations at each channel index must be equal" and "A violation of the balance condition is a compile-time error." Count: `1 + T` puts against `T` gets. Unequal for all `T ≥ 0`.

**Kill.** `initial_tokens` is one of §6.1's four mandatory declarations ("how a cycle is made live") and W2·P3 (c) calls it "the safety proof". As specified it cannot be emitted.

### K3 [D][C] The second mandatory declaration, `depth`, is not an AIR attribute and is rejected by name in `air.api`.

**Attacked** (§1.1 table): "`air.channel` … Has a `depth` attribute"; §6.1: "four declarations and no others: `depth`, `rate`, `fanout`, `initial_tokens`".

**Evidence.** (i) `AIR.td` (raw, `main`, 1264 lines): `air_ChannelOp` takes only `$sym_name`, `$size`, `$channel_type`; `grep -i depth AIR.td` → **exit 1, zero hits**. (ii) `python/air/api/_channel.py` (597 lines, raw): `_UNSUPPORTED = {… "buffer_resources": "the objectFifo depth knob" …}`, raising `NotImplementedError: air.api does not implement buffer_resources=`.

**Kill.** `depth` exists only in compute-model prose, so no verifier enforces it, and the API B1 explicitly assumes ("the backend emits calls into mlir-air's existing Python API rather than generating MLIR text") rejects it. With K2: **two of four mandatory declarations have no legal realisation in the chosen target.**

### K4 [J][P] `air.api` already has channels *and* declared multicast; MLIR-AIR already has *derived* multicast.

**Attacked** (§1.1 fact 4, repeated in §7.3): "**This is the baseline any hackathon DSL must beat or sit above.** … **no notion of stationarity, multicast, reuse, or wavefront**." Plus B4, which HANDOFF calls "the single most decision-relevant unknown": "I have not verified whether mlir-air's `air.api` Python frontend exposes channels."

**Evidence.** (i) `_channel.py`: `channel(name, size=None, broadcast_shape=None, channel_type=None, attrs=None)`, `.put(obj, indices, dependency, dest)`, `.get(obj, indices, dependency)`; its validator says "a 1-to-N fan-out is `size=[1]*N`, `broadcast_shape=…`" — the declared multicast §5's M1 cell awards P3 a point for. (ii) `mlir/include/air/Transform/Passes.td` L870, pass `air-broadcast-detection`: "detects DMA broadcast opportunities by tracing the source indices' dependence to the induction variables of any parent spatial loop space" — the derived multicast §5's M1 cell credits to P1.

**Kill.** The jury's obvious question ("why not just use `air.api`?") has exactly one prepared answer in the document and it is wrong on both halves. Stripped of `depth`, `initial_tokens` and `fanout`, the recommended outer layer reduces to `rate` + `emit_every` over an API that already ships.

### K5 [J][M] The recommended option does not fit the deadline either.

**Attacked** (§6.1, reason 1): "A 6–10 person-week `(σ,π)` solver does not fit; **a 1.5–3 person-week graph builder plus a balance-equation check does.**"

**Evidence.** Doc dated 2026-09-12; §8 re-verified live at `segfault.compilertech.org` today: *"Sept 19–20: Final evaluation"* — **7 calendar days**. 1.5 pw = 7.5 person-days at 5-day weeks; 3 pw = 15. No team size is stated, and the same page's FAQ says *"solo entries did fine — three of the six finalists were teams of one."* A solo entrant has ≤7 person-days including three kernels, the backends, the slides and the pitch.

**Kill.** The load-bearing word is "**does**". At the optimistic end it fits with zero slack and no demo; at the doc's own upper end it overruns 2×. The doc names the deadline "the binding constraint" and never does the division. Compounding, both quoted in §8 and neither acted on: *"Aug 15 same day: Hacking begins"* and *"Registrations are closed"* (today: "Sign-ups ended on Aug 15, 2026") — four of five hacking weeks are gone, or there is no entry.

### K6 [J][D] The third backend cannot carry the recommended paradigm.

**Attacked** (§1.2): "a *verifiable, in-budget* '2–3 backends' claim … is **NPU1 + NPU2 + AMD GPU** … **that is the single strongest scope argument in this document**."

**Evidence.** A GPU channel lowering exists — `mlir/include/air/Conversion/GPUPasses.td` (raw), `air-gpu-channel-to-cacheline` — but its own constraints, verbatim: only `channel_type = "gpu_symmetric_heap"`; "**initial scope: exactly one of each**" put and get; both inside `air.herd` bodies; ranks "derivable from an enclosing `scf.if (arith.cmpi eq %rid, %const)` rank-dispatch nest"; and memrefs "must be **`memref<32xi32, #air.symmetric_heap>`** (initial-PR shape constraint; generalising is a follow-up)". Per `AIR.td`, `gpu_symmetric_heap` is "**Cross-GPU** messaging … enclosed by an `air.rank` op". The programming-examples dashboard lists NPU1 and NPU2 and **no GPU target**.

**Kill.** The recommendation's channels are `sp.f16[TM,TK]` tiles, `depth=2`, `fanout="cols"`, `T` put/get pairs. All five properties fall outside the only implemented GPU path. And NPU1/NPU2 are two device flags on one lowering path — §1.2's own sentence: "The mlir-air README names exactly two lowering paths." Honest count for the recommended paradigm: **one**. `M6 = 4` is unsupported for every column: the anchor for 5 reads "≥3 **verified** backends" and no backend is run anywhere in this document.

### K7 [D][M] The reduction hazard is **not** a compile error in P3, so `M4 = 5` fails.

**Attacked** (W1·P3 (e)): "becomes a *rate mismatch*: write `emit_every=1` and `c_ch` produces `K/TK`× more tokens than `drain` consumes | **compile time** (graph inconsistency). **The best safety result in the comparison**."

**Evidence.** SDF consistency is the existence of positive integer `q` with `Γq = 0`. In the doc's own sketch (L400–426) the sources carry `iters=K // TK` but `@g.sink(inp=c_ch)` carries **no `iters=`**, so `drain`'s firing count is free. With `emit_every=1`, `q_drain = (K/TK)·q_pe` is valid: the graph is **consistent**, schedules, runs, and overwrites `C[...]` `K/TK` times, leaving the last partial sum. Silent wrong answer. Theory checked against the doc's own A28 (Lee & Messerschmitt, *IEEE TC* C-36(1), Jan 1987, PDF extracted): "We now have a **necessary** condition for the existence of a PASS, that the rank of Γ be `s − 1`." Consistency yields *a* `q`, never the intended one.

**Kill.** This cell is the whole of `M4 = 5`, the matrix's highest score, and the stated reason for §6.3's "M4 alone → **P3**". P3's hazard behaviour is the same as P2's and P4's: **silent**.

### K8 [D][M] `M7 = 2` for P1 is priced with a solver §1.3 deletes and a pass that already ships.

**Attacked** (§5, M7 P1): "**est. 6–10 pw** … §3's step list is a compiler (domain extraction, `(σ,π)` **solve**, **reuse classification**, …)".

**Evidence.** (i) §1.3: "No cost model, no autotuner, no design-space exploration, no placement optimiser, no II search. That removes the single strongest argument for P1 (a pragma surface exists partly so a *solver* can later choose `(σ,π)` for you …)". Every P1 sketch hands both maps in: `s.place(px=ax.i0, py=ax.j0)`, `s.stationary("C")`, `s.skew(time=(ax.i, ax.j0))`. The doc's own step 2 hedges — "**solve/check** `(σ,π)`" — and W3·P1 (e) describes a check ("an illegal `skew` violates causality `Sσ·d ≥ 1`"). With both maps given it is a rank/kernel assertion over small integer matrices. (ii) Step 3, "classify each operand into stationary / multicast / stream", is `air-broadcast-detection`, already upstream (K4).

**Kill.** Reason 1 stages a race between 6–10 pw and 1.5–3 pw. Remove one item the doc's own scope deletes and one that ships in the target, and the gap the recommendation turns on is unpriced.

### K9 [P] A fifth column exists, is the user's own, is AMD's own, and already runs on MLIR-AIR.

**Attacked** (§0): "I resolve it by **splitting it into four distinct columns** rather than three" — and §7.4, "Negative results, with the queries that produced them", which never ran this query.

**Evidence.** (i) `spatial-dsl/04-spatial-triton-dsl.md` is an SPMD per-PE surface — "**Unit of execution = PE-tile** (locked)", `sp.pe_id()`, `sp.load(slice, bcast=?, flow=?)` with delivery "default **derived** from access", no explicit channels — whose §2 states: "**Reference oracle preserved.** Every primitive has **sequential fallback semantics** … so the kernel **runs in plain Python as the spec**." (ii) `github.com/amd/Triton-XDNA`, README fetched: "An experimental open-source project demonstrating compiler-driven kernel generation for AMD XDNA NPUs using **Triton and MLIR-AIR**"; flow `@triton.jit → triton-shared → MLIR Transform → MLIR-AIR/MLIR-AIE → XRT binary`; devices **AIE2 and AIE2P** (= NPU1 and NPU2); standard SPMD `tl.program_id`; claimed "performance parity with handwritten NPU implementations". (iii) `AIR.td`'s `air_HerdOp` is one body region plus `sizes` — SPMD over tile coordinates. (iv) Dato's tasks are `@task(mapping=[P0])` + `tid = dato.get_tid()` (Fig. 2c, PDF extracted).

**Kill.** The omitted column is the user's stated reference point (§7.3's own words), is 1:1 with `air.herd` so its `M5` is at least P3's, is lighter on `M2` (no SDF rates to teach), already has the runnable-oracle property §6.1 sells as its only novelty, and already reaches both NPU backends through MLIR-AIR in shipped AMD code. No argument that §6.1 beats it appears, because the column never does — and §7.4 is the section that was supposed to catch this.

---

## 2. WOUNDS

**W1 [D] The ARIES criticism is applied to P2 only.** W1·P2 (c): "the dataflow is also *encoded in slice arithmetic inside `Feeder`* … the ARIES criticism in `spatial-dsl/01` §0, reproduced exactly." But W1·P3's source is `A[pi*TM:(pi+1)*TM, kk*TK:(kk+1)*TK]`, §6.1's hybrid is `index=lambda pi, kk: (slice(pi*TM,(pi+1)*TM), …)`, and W3·P3's body has `j = ctx.pid * CW + c` — same arithmetic, uncriticised. §5's `M3 = 1` for P2 cites "tiling migrated into `Feeder`'s slices" while P3 scores 3 with identical slices.

**W2 [M] The §4 anchors are decorative.** M1's "1" anchor is "Two or more of {…} cannot be written at all"; §5's M1 P4 cell ends "**Two of six capabilities absent**" and scores **2**. M2's "1" anchor is ">80 lines … and a correctness burden"; P4 is 84 lines and W2·P4 (b) says the burden is "Identical to W2·P2 (the user owns … **and the send/receive order**)" — scores **3**. M2's "5" anchor is "≤50 lines": P1 = 48 scores 4, P3 = 51 scores 3. No cell is reachable from its own anchor via the doc's own justification text.

**W3 [D] "Cannot express W2 at all" is laundered, and the column boundary sets the score.** W2·P4 (e) says "in its idiomatic form at all"; §5's M1 cell and §6.3's last row drop the qualifier. Meanwhile the doc places **TAPA** in P3's lineage (§0) while describing it (§7.1) as "the research answer to exactly these restrictions: task-parallel programs with fine-grained inter-task channels made first-class in C++ HLS". Crediting TAPA to P3 and pricing P4 at stock Vitis HLS fixes `M1 = 2` and `M6 = 2` by taxonomy, not by evidence.

**W4 [M] 8-vs-3 is not like-for-like.** W1·P1 step (1) is "build iteration domain and access maps"; P3's three steps have no equivalent, yet `AIR.td`'s `air_ChannelPutOp` requires `static_src_offsets` / `_sizes` / `_strides` as `DenseI64ArrayAttr`s, so `feed_a`'s index function must still become offsets/sizes/strides — and §6.1's stage body must still be compiled for the AIE vector unit. Both charged to P1, omitted from P3. B5's admission does not repair a count that sets `M5 = 2 vs 5` and, via B1, `M7 = 2 vs 5`.

**W5 [D] SDF safety is proved at unbounded queues, applied at `depth = 2`.** §3.0: "a *consistent* SDF graph has a periodic admissible sequential schedule". Lee & Messerschmitt 1987 (extracted): consistency is "a **necessary** condition for the existence of a PASS"; admissibility is a separate class-S result, over unbounded buffers. AIR channels are bounded at `depth` (default 1), so a buffer-bounded schedulability check is additionally needed and is absent from §6.1's "balance-equation check". (W2·P3's specific graph *is* live — checked: one initial token per channel, unit rates, all stages fire at `t = 0` — but no general argument is given, and §3.0 contradicts W2·P3 (c), which states the cycle condition correctly.)

**W6 [M] "Total, for reference only" is an equal-weighted total whose weights double-count.** §5: "**No weighted total is given as a headline**, because the weighting is exactly the thing the user has to choose" — then prints 28 / 20 / 34 / 22 (all re-added: correct). Equal weights *are* a weighting. Worse, B1 says the M7 estimates come "from **the step counts in §3's (d) sections**", so M7 is a monotone function of M5 and §6.3's "M5 + M7 ≥ ~40%" weights one fact twice. "Robust across the two weightings that actually apply" tests one axis and calls it two.

**W7 [D] The stated reason for hoisting `Feeder` is false.** W1·P2 (d): "**because AIR's `put` for an L3→L2 transfer is not legal inside `air.herd`**." `_channel.py`'s check (L408–421) says the opposite — an L3 endpoint in a herd body "needs an `air.segment` around that herd", and the error offers both fixes: "Either wrap the herd in `with air.segment(...) as seg:`, **or** move this {direction} out to launch scope, where an L3 endpoint is fine with no segment at all." The hoist may still be wanted; this `M5` penalty on P2 is not earned.

**W8 [P] The "design point is occupied" test is run on one column only.** §6.2 uses Dato to retire P3's novelty, then keeps P3. `spatial-dsl/01-design-overview.md` §0 — the file §0 names as P1's source — closes: "There is a contemporary proof that clean separation is possible on this exact hardware: **AIEHalide** (PACT '26) — *"Compiling Halide to AMD NPU Spatial Dataflow with Algorithm-Schedule Separation."*" Never mentioned. *Caveat:* I could **not** confirm AIEHalide exists — a targeted search returned no match, so treat it as the user's unverified claim. Either way, a prior-art claim about the rejected column, sitting in the cited source file, went unaddressed while the same test was decisive for the chosen one.

---

## 3. NITs

- **[C]** §0: "verified: `grep -ril akka` over the whole tree returns nothing". Re-ran: returns both hackathon files. HANDOFF states it correctly; the doc does not.
- **[C]** A12's URL is **HTTP 404** (today), and the quote *"the HLS tool issues a message and does not perform DATAFLOW optimization"* is **not found** — exact-phrase search returns zero. Nearest real AMD wording is weaker: "it issues a warning, **depending on the situation**, and **might** not perform the DATAFLOW optimization"; other AMD text says the tool "issues a message and **transforms the region to apply** DATAFLOW optimization". Load-bearing in W2·P4 (e) and the M2 P4 cell.
- **[C]** §1.1 sources "Has a `depth` attribute" to rows A1/A2 = `AIR.td`, which contains "depth" **zero times**. Prose doc and dialect definition merged with no flag.
- **[C]** A5 quotes the deadlock condition and drops the next bullet of the same list — the direct cause of K1.
- **[C]** B11 cites "Spatial's reported 2.9× mean speedup over SDAccel HLS on a VU9P" as an example of numbers the doc cites. `grep` shows "2.9" appears once — inside B11. No Appendix A row.
- **[D]** `AIR.td`'s `gpu_symmetric_heap` text is **stale**: "Lowering will be added by a future GPU pass (planned: `air-gpu-channel-to-mgpu`)", but `GPUPasses.td` ships `air-gpu-channel-to-cacheline`, implemented. Flagged so round 2 does not re-derive K6 from the stale comment.
- **[P]** §7 misses four 2025–26 neighbours. **Ripple**, PLDI 2025 (Ghosh, Shi, Lucia, Beckmann), verified at `pldi25.sigplan.org` — "Ripple efficiently implements **deadlock-free**, asynchronous task communication by exposing hardware token queues in its ISA" — directly on-point for M4's framing. **IRONSmith**, arXiv:2607.10944 (12 Jul 2026), a visual AIE tile-grid canvas with "wires representing FIFOs, split/join patterns, broadcast connections" emitting IRON Python — same paradigm, same hardware, far more demoable in five minutes. **SpaDA** (arXiv:2511.09447) and **TileLoom** (arXiv:2512.22168), which I have only at abstract level, not primary read.
- **[P]** B3 is answerable and the delta is thinner than claimed. Dato PDF (extracted): task bodies **are** plain Python loop nests (Fig. 2c). I found **no** sequential/CPU/functional fallback in the body (searched "simulat", "csim", "CPU execution", "functional", "golden", "numpy"), so only the runnable-oracle half of the delta survives — and Dato "is built atop **Allo**" (§6), whose tutorial says "By default, Allo will generate a LLVM program that can be executed on the CPU", so even that half is thin.
- **[P]** The doc never says where Dato lowers. Paper §3.1: "To target AMD NPUs, we leverage **MLIR-AIE** [53] as the backend; for FPGAs, we generate C++ code for high-level synthesis (HLS)." Dato bypasses AIR entirely — both a sharper overlap statement and the one clean differentiator §6.2 could have claimed.
- **[D] Confirmed, not findings** (checked because the brief asked). W3's "one scalar per row per PE boundary" is **correct**: PE `p`'s `(-1,-1)` value `S[i-1, p·CW-1]` is the west value received at row `i-1`, retained at `prev[0]` by the `prev`/`cur` swap. W3·P1 (d)'s "the skewed `σ` → **nothing explicit in AIR at all**" is **correct**: the compute model's *Flow control semantics* make `get` block on empty, so the wavefront is emergent.

---

## 4. Escape table

**Round 1: no prior round.** No escapes to score. This section exists so round 2 can record which findings the document dodged, narrowed, or absorbed.

---

## 5. Spot-checks

| Claim | What was checked | Source fetched | Result |
|---|---|---|---|
| A1 / A2 | op mnemonics; herd/channel/put/get descriptions | `…/mlir-air/main/…/AIR.td` (raw) | **PASS** |
| §1.1 "depth attribute" | `air_ChannelOp` args; `grep -i depth` | same file | **FAIL** — args are `sym_name`, `size`, `channel_type`; zero hits |
| A3 / A4 | memory spaces; depth 1 Rendezvous / 2 Double-buffering / put stalls | `…/dev/AIRComputeModel/` | **PASS** (verbatim) |
| A5 | balance, deadlock definition, concurrent-execution-context condition | same page | **PASS but incomplete** — omits the acyclicity bullet in the same list |
| A6 / A7 | `from air import api as air`; "imperative behavioral programs…"; `air-to-aie` / `air-to-rocdl` | mlir-air `README.md` (raw) | **PASS** |
| B4 | does `air.api` expose channels? is `depth` settable? | `python/air/api/_channel.py` (597 lines, raw) | **FAIL** — channels yes, with `broadcast_shape`; `depth` explicitly unimplemented |
| §5 M1 "multicast only derived" (P1) | is derived multicast already upstream? | `…/air/Transform/Passes.td` L870 (raw) | **FAIL** — `air-broadcast-detection` does exactly that |
| W1·P2 (d) L3-put-in-herd | placement legality | `_channel.py` L408–421 | **FAIL** — legal with an enclosing `air.segment` |
| §1.2 GPU backend | can `air.channel` lower to GPU, and under what limits? | `…/air/Conversion/GPUPasses.td` (raw) | **FAIL for the recommendation** — `gpu_symmetric_heap`-only, one-put/one-get, `memref<32xi32>`-only, cross-GPU |
| A8 | NPU1 = Phoenix/AIE2, NPU2 = Strix/AIE2P | `…/dev/programming_examples/` | **PASS**; **no GPU target listed** |
| A9 | mlir-aie device list; Hexagon absent | `mlir-aie/docs/Devices.md` | **PASS** |
| A10 | title + 22-author order + arXiv 2510.14871 | local PDF (`pdfinfo`) + `arxiv.org/abs/2510.14871` | **PASS**; TRETS DOI 10.1145/3785670 **UNREACHABLE** (no journal-ref; ACM DL gated) |
| A12 | UG1399 URL + "issues a message…" quote | `docs.amd.com/…/pragma-HLS-dataflow`; tutorials; phrase search | **FAIL** — 404; quote not found; real wording weaker |
| A14 | IRON 11-author list and order + FCCM 2025 | `arxiv.org/abs/2504.18430` | **PASS** |
| A19 | hexagon-mlir arXiv:2602.19762, Feb 2026 | `arxiv.org/abs/2602.19762` | **PASS** |
| A28 | Lee & Messerschmitt, *IEEE TC* C-36(1):24–35, Jan 1987; rank(Γ)=s−1, nullspace, class-S | PDF + `pdftotext` | **PASS** — rank(Γ)=s−1 stated as **necessary only** |
| A37 | ARIES, FPGA 2025, pp. 92–102, DOI 10.1145/3706628.3708870, 8 authors | search → ACM DL record | **PASS** |
| A42 | Dato title / authors / date / abstract + body §§2–5 | `arxiv.org/abs/2509.06794` + PDF | **PASS** |
| A43–A45 | Sept 19–20 evaluation; 5-min pitch + 2–3 Q; registrations closed; ₹2,00,000; 8 themes; no rubric | `segfault.compilertech.org` (today) | **PASS** |
| §5 line counts | recounted by script, non-blank lines per fenced block | local file | **PASS** — 48 / 59 / 51 / 84; hybrid 20. Exact |
| §5 totals row | re-added all four columns | local file | **PASS** — 28 / 20 / 34 / 22 (see W6) |
| B12 word count | prose outside fences and table rows | script | **PASS** — 8,296 vs "roughly 8,200" |
| §0 `grep -ril akka` | re-ran over the tree | local repo | **FAIL** — returns both hackathon files |
| §2 W4 routing | circuit-switched, compile-time routes, router can fail | local `reading-group/02` §4 | **PASS** |
| proposal §5 | "expose architectural intent" | local proposal L184–188 | **PASS** |
| §0 four-column framing | is there an unscored fifth surface, and is it built? | local `spatial-dsl/04`; `github.com/amd/Triton-XDNA` | **FAIL** — SPMD column omitted; AMD ships it on MLIR-AIR for AIE2/AIE2P |

---

## 6. Rival statement

**Ship `air.api` plus one thin SPMD layer, and make the demo the diff — not a new graph language.** Write GEMM and Smith-Waterman directly against `from air import api as air` (`launch`/`segment`/`herd`, `channel(name, size=, broadcast_shape=)`, `put`/`get`, `alloc(scope=)`, `build(target=…)`) and add exactly one construct: a `@sp.herd(grid=…)` decorator whose body is a plain Python loop nest that (a) runs in CPython over the herd coordinates as the reference oracle and (b) emits the herd body, letting `air-broadcast-detection` derive the multicast rather than declaring it. No new IR, no SDF machinery, no `depth` and no `initial_tokens` — neither has a legal AIR realisation (K2, K3). Four reasons this beats §6.1 before *this* jury. (1) It is buildable in the seven days that remain, because the channel emission, the scope checks, the broadcast validator and the broadcast-detection pass are already written and tested upstream — the machinery §6.1 budgets 1.5–3 person-weeks to rebuild. (2) It survives the killer Q&A question with a live side-by-side instead of a claim about `air.api` that is false twice over. (3) The five-minute demo is `python gemm.py` printing a passing numpy comparison, then the same file at `target="npu1"` and `target="npu2"` — an honest **one** lowering path with two device flags, said out loud, which a compiler jury will respect more than a three-backend claim whose third leg is a fixed-shape cross-GPU primitive. (4) It keeps the thesis alive: derived-multicast-from-access-map is the first real `(σ,π)` result, not a throwaway. Two scope changes follow. Drop Jacobi, or re-map it to a one-directional halo forward, because AIR requires an acyclic communication graph (K1). And open the pitch by naming `amd/Triton-XDNA` and saying what this adds to it — a jury that finds it during Q&A is a jury you have lost.

---

## 7. The one question the document cannot answer

> Your outer layer's four mandatory declarations are `depth`, `rate`, `fanout`, `initial_tokens`. `depth` is not an argument of `air.channel` in `AIR.td` and is rejected by name in `air.api`. `initial_tokens` lowers to a prologue `put` that violates AIR's static balance condition, and the cycle it exists to make live violates AIR's stated acyclicity requirement. `fanout` is `broadcast_shape`, which shipped before you started — and `air-broadcast-detection` infers it from the access map without being told. Strip those three, and **what is left that `air.api` does not already give the user — and why is it worth a week you do not have, when AMD already ships a Triton SPMD frontend onto the same MLIR-AIR for the same two devices?**
