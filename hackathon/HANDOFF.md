# HANDOFF — paradigm comparison for the hackathon Spatial DSL

*2026-09-12, **rev. r3**. Reviser → user. Companion to
[`01-paradigm-comparison.md`](01-paradigm-comparison.md) (rev. 2026-09-12 r3),
[`REDTEAM-round1.md`](REDTEAM-round1.md) / [`RESPONSE-round1.md`](RESPONSE-round1.md) and
[`REDTEAM-round2.md`](REDTEAM-round2.md) / [`RESPONSE-round2.md`](RESPONSE-round2.md).*

## State

**The wording loop is stopped after r3, by the architect's decision.** Six documents in
`hackathon/`, nothing implemented, nothing measured, nothing committed, and **no file outside
`hackathon/` touched**. Round 1: 9 kills, 8 wounds, 10 nits — all 27 accepted, r2 written.
Round 2: 5 kills, 8 wounds, 6 nits, 1 no-flaw-found, plus a rival statement — **all 20
accepted**, r3 written as a **narrow** pass (no prose re-audit, no restructuring). Every claim
is traced in Appendix A and every gap in Appendix B or marked `[UNVERIFIED]` inline.

**Why the loop stops.** Round 2's own verdict was *"one more round, narrow"*, and its reason
was that two of its five kills *"need an experiment or a build"*, not wording. r3 did the
wording half and converted the rest: **what remains open is empirical, and is now five named
experiments with recipes in §9 of the main document.** A fourth round of reading would produce
better sentences about questions that ten lines of Python and one `air-opt` invocation answer.

## Decisions taken in r3, with reasons

1. **The K1 depth-1 question is left open, deliberately.** The AIR compute model contradicts
   itself — its stall rule says the first `put` into an empty depth-1 channel does not stall,
   its own minimal-deadlock example says it does. §1.1 fact 1 quotes both and adjudicates
   neither. §3.2's protocol is restated in terms that make the question testable (async puts,
   gets with no token dependency on them, different buffers) and marked
   `[UNVERIFIED — Experiment E1]`. **Do not let the pitch assert either way.**
2. **The fallback is redefined: B′, not B.** r2's "P5 core with nothing declared" failed §1.4's
   own two-property test and was also Triton-XDNA's shape. **Candidate B′ = the P5 core with
   declared intent** (`stationary=`, `bcast=`, `flow=`, `exchange`), which claims both
   properties and is a legitimate standalone entry rather than a degradation.
3. **"Declared spatial intent" is no longer a novelty claim.** AIEHalide is **accepted** at
   PACT 2026 with three optional ignorable directives for AIE and halos derived from bounds
   inference; SpaDA (2026, Cerebras) declares placement, neighbour streams and multicast.
   **What survives as the differentiator is the pre-codegen legality check.**
4. **Two cells moved**: M1 P5 3→4 (P1's own doc-01 `stream(broadcast|forward|cascade)` was
   missing from its glossary, and the invention penalty was charged to one column only) and
   M4 P5 3→2 (the W2 oracle is conditional). They cancel in the equal-weighted total — a
   coincidence, not a defence. **The pick keeps its name and loses two of its three reasons**;
   §6.7 is re-derived rather than protected.
5. **The M5 row was rebuilt once, uniformly** (§3.0): a fixed S1–S12 step vocabulary, F1–F4
   excluded for every column, channel materialisation charged to every column with channels.
   New maxima 5/4/3/4/3; the scores happen not to move. **§4 now says the M5 spread is ±1
   noise — do not build an argument on an M5 gap of 1.**
6. **The rival is in the document as §6.8**, stated at full strength, with the architect's
   position (A's D3 checker *is* the rival's component) and the named case where the rival
   wins: **N = 1 and E1 fails.**

## Verified facts the user can rely on

Read directly from primary sources unless noted. Appendix A row numbers in brackets.
Everything carried forward from r2's list is still good; these are the r3 additions and
corrections.

**New in r3**

- **The AIR compute model contradicts itself on depth-1 semantics.** Stall rule: *"A `put`
  issued when the channel already holds `depth` unread transfers stalls until the consumer
  issues a `get` and frees a slot."* Minimal-deadlock example on a fresh `{depth = 1}` channel:
  *"// put blocks: channel is at capacity, waiting for a get to free space"*. Async form: *"In
  the asynchronous form the operation is dispatched immediately and the returned `!air.token`
  does not signal until the blocking condition is resolved and the transfer is complete."*
  **Unresolved — E1.** [A4]
- **The balance requirement has four bullets, not one.** Per path; **per iteration for channels
  in loop bodies**; a cross-iteration-space product rule for channels between herds; and
  **independently on each branch** for channels in conditionals. *"A violation of the balance
  condition is a compile-time error."* [A5]
- **No pass or verifier enforcing balance or acyclicity exists in mlir-air, as far as r3 could
  find.** `AIRDialect.cpp` (4 069 lines): zero hits for "balanc"/"acycl". `ChannelOp::verify()`
  checks the channel-type allow-list, NumPy broadcast-shape rules, `refeed_count`,
  `packet_ids`. `ChannelPutOp::verify()` checks sizes/strides rank, `refeed_count`, the
  `npu_mmio` L3-source rule, the `gpu_symmetric_heap` `air.rank` rule, and that *"channel bundle
  indices must not be temporal `scf.for` induction variables"*. `Transform/Passes.td` has no
  such pass. **E3.** [A59, A61]
- **`air-enforce-channel-fifo-order`** = *"Serialize same-channel async ops to preserve FIFO
  order"*, description opening *"A single air.channel is an ordered FIFO, but **air-dependency
  only orders channel ops that share a buffer**."* — the sentence E1's protocol rests on.
  **`air-verify-hierarchy-locality`** *"Statically detect[s] data races on memrefs that an
  air.launch / air.segment / air.herd passes to itself as kernel operands"* — races, not
  balance. [A61]
- **`air.api` has no oracle — verified, B14 resolved.** `_compile.py`: *"The body is not
  executed when ``@launch.body`` runs -- it is recorded, and replayed later"*; `_trace.py`:
  *"Nothing here emits IR until a herd body is registered."* Twelve files; no interpreter.
  **But** mlir-air ships `python/air/backend/cpu_backend.py` — `AirCpuBackend`, *"This currently
  uses the torch-mlir linalg-on-tensors RefBackend for JIT execution."*, pipeline `air-to-async,
  canonicalize, cse` — unreferenced from `air/api/`. **An AIR *module* can be CPU-JIT'd; an
  `air.api` *program text* still does not run as its own spec.** [A59, A62]
- **AIEHalide is ACCEPTED at PACT 2026.** *"We are pleased to inform you that your paper has
  been accepted."* (`PACT/56.txt` L300). Rebuttal author **Abnikant Singh, IIIT** (L249) — a
  different person from the user, so **the reviews file establishes no authorship link**; the
  "our group" attribution is **the user's own proposal**, L9. It adds *"only three optional
  expert directives (`aie_dataflow`, `aie_fuse_with`, `aie_kernel`)"* to Halide's standard
  ones; *"Bounds inference yields the producer regions, halos, and per-tile working-set sizes"*;
  it targets **MLIR-AIE, bypassing AIR**; it is **open-loop** (*"appear as a compile-time
  routing failure"*, *"an explicit feedback loop is future work"*); measured **53%/62%** of peak
  int8 GEMM on XDNA/XDNA2 vs StB's 66%/93%. [A57]
- **Halide's schedule language has no skew, wavefront or stationarity directive.**
  `src/Func.h` (2 972 lines, `main`): zero hits for skew/wavefront/stationar/diagonal. [A64]
- **Triton-XDNA ships per-kernel transform scripts.** 53 transform paths; selected by
  `AIR_TRANSFORM_TILING_SCRIPT` (README: *"Path to the MLIR transform dialect tiling script"*);
  one read in full, *"Auto-generated by matmul_transform.py — do not edit manually."*, with
  phases including *"PHASE 2: PROMOTE OUTPUT TO L2"* and *"PHASE 5: TILE FOR MULTI-CORE
  PARALLELISM"* / *"Tile [16, 16, 0] for herd distribution."* **Memory-space promotion and herd
  width are user-settable today.** The four intent words still occur zero times (red team's
  grep, not re-run here). [A63]
- **SpaDA** (arXiv:2511.09447v2; Gianinazzi, Ben-Nun, Hoefler) declares *"data placement,
  dataflow patterns, and asynchronous operations"* via `place`/`compute`/`dataflow` blocks,
  `relative_stream(x, y)` and cardinal-direction multicast; **derives** halos one level up in a
  GT4Py stencil IR; has no stationarity or skew directive and no legality check; and **targets
  Cerebras CSL, not AIE**. [A65]
- **AIE tile DMAs are direction-specific**: `AIE.dma_start("MM2S", …)` in the source tile's
  `AIE.mem`, `AIE.dma_start("S2MM", …)` in the destination's. From **mlir-aie's own docs, not
  AMD AM020** (not fetched). Removes an engine-level deadlock mechanism; does not settle E1.
  [A60]

**Corrected in r3**

- `spatial-dsl/01` §3's vocabulary table **does** have `stream(buf: pattern)` with *"`broadcast`,
  `forward` (systolic), `cascade` (reduce)"* — r2's P1 glossary omitted it and docked P1 for it.
- `spatial-dsl/04` §4 fixes the `@sp.kernel(grid=)` fallback as *"loop over all `(pi,pj)`"* —
  **grid loop outermost**, which is why W2·P5's oracle needs a lockstep interpreter.
- `_channel.py`'s **module docstring is stricter than its implemented check**: the docstring says
  an L3 endpoint *"has to sit inside an `air.segment`"*; the code requires an `air.launch`
  always and a segment only inside a herd body. Cite the code, not the docstring.
- `grep -ril akka` returns **four** files, all under `hackathon/`; `HANDOFF.md` has zero hits.

## The two inputs only the user can supply — still open, still blocking §6

1. **Team size *N*, and days actually available.** Budget = 7·*N* person-days = 1.4·*N* person-
   weeks (≈1.75·*N* at 10 h days). **At *N* = 1 the plan degenerates to Candidate B′ by design**,
   which r3 considers an acceptable entry (r2 considered it a disqualified one — the K3 fix).
   At *N* ≥ 2 the P1 layer and its checker are affordable. **And the conditional instruction:
   if *N* = 1 and E1 fails, ship the rival of §6.8 instead of Candidate A.**
2. **Is 19–20 September your evaluation date, and do you have an entry?** The site says
   registrations closed 15 Aug and *"Aug 15 same day: Hacking begins"*. If work already exists,
   the 7-day figure is only what remains. If there is no entry, §6 is moot. If you mean a
   different, IIIT-internal event, §8's timeline — and therefore §6 entirely — does not apply
   (SegFault 2026 is an **IICT/IISc** event; no IIIT affiliation was found).

**One thing to confirm rather than supply**: the document says AIEHalide is your group's work
**on the strength of your own proposal's L9**, not of `PACT/56.txt`, whose rebuttal is signed
by a different person at your institution. **Confirm before the pitch says "our group".**

## The experiments — run E1 and E2 before day 1

Full recipes, expected outcomes and the text that changes on failure are in **§9** of the main
document. In one line each:

| # | Question | Cost | What it decides |
|---|---|---|---|
| **E1** | Does put-before-get deadlock on a depth-1 channel? Write a two-PE halo exchange via `air.api`, run `air-opt -air-dependency` and check no token edge joins a PE's put to its own get; then `aircc` and run if a device exists. | hours | Every W2 sketch in all five columns, `sp.exchange`, `s.exchange`, M4 for P1 and P5, and whether the rival becomes the recommendation. |
| **E2** | Does the W2·P5 fallback compute Jacobi? Run it with `p` outermost (expect mismatch) and with `t` outermost (expect match) against numpy. | ten lines | M4's P5 cell (2 now, 3 if it passes), and whether W2 can be in the five-minute demo at all. |
| **E3** | Does `air-opt`/`aircc` reject an unbalanced or cyclic channel program, and which pass does it? Feed it three hand-written violations. | an afternoon | Roughly ten (e) cells and four M4 justifications that r3 had to soften; and whether §6.8's rival is "an alternative" or "the obviously missing component". |
| **E4** | Do the ping-pong passes fire on an `air.api`-emitted herd loop? (B15) | an afternoon | Whether any surface `depth=`/`double_buffer` in this document has a real lowering. |
| **E5** | *(optional)* End-to-end Versal: `aircc` W1 with `device=xcvc1902`. (B13) | an hour | One sentence in the pitch's honest-limits slide. |

## Remaining technical unknowns, in decreasing decision-relevance

1. **E1 and E2** — see above. Everything else is behind these.
2. **E3 / W7** — the checker is not merely un-run, it is **unlocated**. If it does not exist,
   every "compile time" cell that cites AIR is wrong, and building it is the entry.
3. **B15 / E4** — the `depth` realisation path.
4. **B13 / E5** — end-to-end Versal.
5. **W4 (FFT) still has no sketch** (B6). The claim that P1's affine surface cannot express
   `p ↦ p ⊕ 2^s` without an escape hatch remains reasoning, not a demonstrated attempt — and
   it is now the *only* M1 capability separating P1 from P5, so it carries more weight than it
   did.
6. **TileLoom** (arXiv:2512.22168) is still unread. It is named in `PACT/56.txt` L62 as the
   hybrid-cost-model comparison AIEHalide's reviewers asked about, so it is not unrelated.
7. **Lipton & Lopresti (1985)** still cited via survey literature only (B8).
8. **Whether a derived halo beats a declared one.** Two independent systems — AIEHalide and
   SpaDA — derive halos rather than naming them, which is evidence that `s.exchange` /
   `sp.exchange` is the wrong primitive. §5's M1 does not capture this, and a fourth round
   would have to decide whether P1's derived-halo story is a *scoring* advantage or just a
   design one.

## If a round 3 red team is run anyway

Do not re-audit prose; it has been through three passes. Attack, in order: **(a)** the two cell
moves and whether §6.7 should have flipped to B′ or to the rival once M1 tied and "declared
intent" stopped being novel; **(b)** §3.0's rebuilt step table, which is the quantitative basis
for M5 and is a decomposition of an unimplemented lowering (B5, B19); **(c)** §9's experiments —
whether E1's recipe actually distinguishes the two readings, since a passing `air-opt` run
proves only that no *token* edge exists, not that no *channel-slot* stall occurs at run time.
