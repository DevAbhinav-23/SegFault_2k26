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

---

## Integration — state at 2026-09-15

*Appended by the integration pass, on branch `integration` off `main` at `78f308c` (all three
roles merged). Companions: C's [`../design/PROGRESS-C.md`](../design/PROGRESS-C.md),
[`../design/PROGRESS-B.md`](../design/PROGRESS-B.md),
[`../design/PROGRESS-TT.md`](../design/PROGRESS-TT.md).*

### What the architect verified on `main` before this pass

* `.venv` default suite: **694 passed, 1 failed, 4 skipped**. The failure was
  `tests/unit/test_nfr.py::test_NFR4_no_bare_raise`.
* `-m slow` (with `source scripts/airenv.sh .venv/bin/python`): **12 passed, 1 skipped**.
* `.venv-tt`, `source scripts/tt_env.sh`, `-m requires_ttsim tests/tt -rA`: **23 PASSED,
  2 FAILED** — both force-failed by the harness's own 3 s per-test budget, not by ttsim.
* A's live surface (`sp.kernel` → `sp.schedule` → `.check()/.plan()/.mlir()/.summary()`)
  reproduces B's goldens **byte for byte** for W1, W1-flip and W2 on npu1 and npu2, and
  `m5tt_emit.emit(live_plan) == m5tt_emit.emit(fixture_plan)` for the same three.
* **W3 broke at M4 on the live path**: `s.plan()` raised `PROTOCOL-UNSUPPORTED`, the plan
  charging 232 B for the staged buffers where `LegalMapping.l1_bytes` said 108 B.

### Defects found, and the fix in this pass

| # | Defect | Owner of the code | Fix |
|---|---|---|---|
| 1 | M3 charged W3's read-only `q` **4 B** (span 1 along the skewed axis `i`) where `03-lld-M3-checker.md` §6.4 says 128; `l1_bytes` 108 vs M4's 232, so `s.plan()` refused every W3 schedule | **A** | `_pinned` gains `skew_pins=`; `_check_l1_capacity` passes `skew_pins=False` for a read-only operand. Ruling **R-L9-1** below |
| 2 | `tests/conftest.py::_EXEMPT_MARKS` lacked `requires_ttsim`, so the 3 s per-test budget force-failed two ttsim tests that take ~48 s each | **C** | `requires_ttsim` added; `test_NFR3_exempts_every_requires_marker` pins the rule against the next marker |
| 3 | Three bare raises in `spatial/` failed NFR-4: two `raise AssertionError("unreachable")` sentinels in `m1_frontend` and `m3_legality._operand_matrix`'s bare `AssertionError` | **A** | `_raise` annotated `-> NoReturn` and the sentinels deleted; `_operand_matrix` raises `_fail("STATIONARITY", ...)`. **The condition is user-reachable** — a kernel parameter the body never touches — so the message is written for a user, not as a bug report |
| 4 | `kernels/w2_jacobi.py`, `w3_sw.py`, `w1_gemm_bf16.py` and `rejections.py` still stubbed the surface; `demo/run_demo.py` printed nothing for three beats and `test_demo_rejections` skipped | **C** | All wired to the live surface; three rejection goldens captured; the demo runs every beat |
| 5 | `tests/unit/test_a_smoke.py` carried no `fr()` marks, so the traceability gate reported 28 of 44 FRs uncovered | **A** | Marks for the ten FRs those tests genuinely exercise: residual **18** |
| 6 | C changed §7.2's `m6.run` / `m6.trace` shapes and added `DiffReport` with no version bump | **C** | Contract **v6** below |
| 7 | `kernels/w1_gemm_bf16.py` (256³, `bf16` in, `f32` out) is accepted by M1, M2, M3 and M4 (`l1_bytes` 49 152) and **rejected by M5**: `EMIT-AIR-API`, `air.api` saying *"dtype mismatch in elementwise assignment: destination is air.api.f32 but operand is air.api.bf16"* | B's emitter, `air.api`'s limit | Not forced. `schedule_os` raises `NotImplementedError` naming that stage and code; `w1_large` stays an inputs-only fixture at levels I and S |

### Rulings

**R-L9-1 (architect, 2026-09-15) — a skew pins a written operand's L1 footprint, never a
read-only one.** In `spatial/m3_legality.py`'s L1 accounting, and **only** there, a
skewed-but-unplaced axis is unpinned for an operand the kernel does not write. A written
operand under a skew is produced and forwarded step by step (the swap pair, §3.12 swap parity),
so only the pinned-axis span is resident; a read-only operand with no dependence on a placed
axis is delivered **whole** by M4's multicast (HLD §7.3 — `q` is *"multicast along px"*, FR-M1),
and a per-step stream of one is not a delivery M4 implements. `_pinned` itself is unchanged: it
still pins the skewed axis for `_check_halo` and `_check_swap_parity`.
Measured after the fix: **W3 232** (`q` 128 + `r` 32 + `S` 72, §6.4's own arithmetic), W1
**12 288**, W1-flip **16 384**, W2 **1 280** — the last three unchanged.
`design/03-lld-M3-checker.md` §3.10 carries the erratum: pseudo-code line 2 charges **every
operand the kernel references** (not only those `reside()` names, which §6.4 and M4 already
contradicted), line 3's pinned box is per operand, and the scope note's "for W3 they are 72 and
80 bytes" is corrected to **232 and 240**.

**Contract v6 (architect, 2026-09-15).** `CONTRACT_VERSION = 6`. `06-interfaces.md` §7.2 now
states what `spatial/m6_tools.py` implements: `m6.run(artifact, inputs, target, kernel_name,
workdir=None)`, `m6.trace(mlir_path, model_json, function, workdir=None)`, `m6.has_device()`,
and `DiffReport` as the **frozen dataclass in `spatial/model.py`** it has always been. Forcing
requirement: the frozen two-argument shapes cannot construct an `XRTBackend` nor name the
function for `air-runner -f` (C, `design/PROGRESS-C.md` §3). No field of any §2–§5 record changes.
`00-README.md` §4 carries change-log row 6 and the v6 signature row.

### What the suite says now

`.venv` default: **723 passed, 3 skipped, 37 deselected**. The three skips are each a
deliverable, not a gap: no device (`/dev/accel*` absent), the traceability residual, and
`test_NFR7_all_errors_are_spatial` (the negative corpus is empty). `pytest -m slow`: 12 passed,
1 skipped. `.venv-tt -m requires_ttsim tests/tt`: **25 PASSED, exit 0**.
`python demo/run_demo.py`: exit 0 in ~1.4 s, with and without `scripts/airenv.sh`.

*After the B-side closing pass (2026-09-15), measured on that branch: `.venv` default **746
passed, 3 skipped, 40 deselected in 15.9 s** — same three skips; `pytest -m slow` **13 passed, 2
skipped, 774 deselected** (the second skip is W2's ttsim deadlock demonstration, which needs
`.venv-tt` and therefore skips under `.venv` — it runs, and passes, in the TT suite);
`.venv-tt -m requires_ttsim tests/tt` **26 PASSED, exit 0**; `python demo/run_demo.py` **exit 0
in 3.7 s** with `.venv-tt` and the simulator present (2.0 s on a warm JIT cache), which now
includes a real W3 execution on ttsim. The A-side pass ran concurrently and its figures are its
own.*

### Tenstorrent in the project

The second backend is no longer a stretch item sitting beside the entry; it is on the demo path
and on the live test path, and the two rulings it was waiting on are closed.

* **A demo beat.** `[4:05] Second backend — Tenstorrent Wormhole on ttsim, same plan`, between
  "It lowers" and "Honest limits". It takes W3's plan from the **live surface**
  (`kernels.w3_sw.schedule("npu1").plan()` — the same object the 3:40 beat lowered through
  `air.api`), emits the TT-Metalium program in the demo's own interpreter, prints the four-core
  range, the one data-movement kernel, the six semaphores, the circular buffers and the io
  tensors, and then **runs it**: `.venv-tt` on `demo/tt_w3_on_ttsim.py`, compared against
  `kernels.w3_sw.sw` executed in CPython. Measured: `W3 on ttsim: EXACT (1089 cells compared)`.
  With `.venv-tt` or `vendor/tt/libttsim_wh.so` absent it prints which one is missing and points
  at `design/PROGRESS-TT.md` §T5.3 — it never claims a run it did not get.
* **Live tests, both halves.** `test_live_plan_emits_the_same_tt_program` (default suite, no
  device) asserts `m5tt_emit.emit(live_plan) == m5tt_emit.emit(m4.plan(fixture_literal))` for
  W1-os, W1-ws, W2 and W3 — one structural `==` over the kernel C++, the core range, the buffers,
  the semaphores and the runtime args; `tests/tt/test_tt_live.py` runs the same live path through
  the simulator. Until now every TT test began at a hand-written `LegalMapping`.
* **R-TT-B and R-TT-B′ are adjudicated** (architect, 2026-09-15): *"Accepted as measured:
  occurrence pairing (k-th put ↔ k-th get) and the credit = smallest gap between aliasing landing
  regions (1 for W3 and the cascade, 2 for W2). The depth-1 deadlock on W2 was reproduced, the
  credit rule removed it, and all four variants are exact on ttsim with negative controls. Not a
  general theorem: holds for the four plans measured; any new plan shape must re-run
  `tests/tt`."* That scope clause is the sentence to say out loud, and it is honest limit 9 in
  `design/08-tt-backend.md` §8.

What is **not** claimed is unchanged: no Tenstorrent silicon was touched at any point, ttsim is a
**functional** simulator of one Wormhole B0 part, the compute is scalar C++ on one data-movement
RISC-V, there is no timing claim, and **Tenstorrent is not an AIR target**.

### Open items, by owner

*Rewritten **2026-09-15**, after the two closing passes that followed the integration pass: a
**B-side** pass (the demo, the second backend, the emitter's dtype cast, CI, the state files —
this list's B and C entries) and an **A-side** pass running concurrently. The two could not see
each other's results, so every A item below says only that it is being closed by that pass, and
its outcome is `design/PROGRESS-A.md` (to be linked when the A-side pass lands it) rather than a
claim made here.*

**Person A — being closed by the A-side pass; read `design/PROGRESS-A.md` (to be linked) for
what it actually did.** The list the integration pass left: the **negative corpus**
(`tests/negative/` holds only `__init__.py`, 30 of the 43 catalogue codes raised by nothing,
`test_D1_schema` / `test_D3_catalogue_complete` absent, `test_NFR7_all_errors_are_spatial`
skipping); the **18 uncovered FRs** (`FR-S1, FR-S4, FR-S5, FR-S6, FR-S9, FR-S10, FR-S11, FR-S15,
FR-S16, FR-S17, FR-S19, FR-L1, FR-L5, FR-L6, FR-L8, FR-L10, FR-L11, FR-L13`); **FR-L9's
per-buffer breakdown** in `details`; **`GrammarError` with `location=None`** raising `ValueError`
instead of the rejection it meant; M1/M2 diagnostics carrying **absolute** `location` paths; and
the **unused kernel parameter rejected under `STATIONARITY`**, which the architect accepted on
2026-09-15 as the closest code in the frozen catalogue. A's **signatures** on `06-interfaces.md`
v3–v6 are still A's to give (B's are in, 2026-09-15).

**Person B — done in this pass.**

1. ~~The Tenstorrent backend is a stretch item beside the project~~ — **done.** It is in the
   demo (beat 4:05, W3 executed on ttsim against the CPython kernel) and on the live test path
   (`test_live_plan_emits_the_same_tt_program` in the default suite, `tests/tt/test_tt_live.py`
   on the simulator). `design/PROGRESS-TT.md` §T6.
2. ~~`kernels/w1_gemm_bf16` is not emittable~~ — **done, B-P32.** M5 widens each load to the
   destination's dtype with `air.api.ops.cast`, so "bf16 in, f32 out" emits; every golden is
   byte-identical. `aircc` still refuses W1-large **for its 256³/4×4 shape, not for its dtype** —
   a cast-free f32 control at the same grid and the same L1 draws the same two diagnostics, and
   the `slow` test asserts exactly that.
3. ~~B-P29, B-O8~~ — **closed.** `tests/unit/test_nfr5_constants.py` is the NFR-5 AST lint (90
   module constants, 16 undocumented, each listed with its owner); B-O8 is answered — `check_pin`
   plus byte-for-byte goldens are the standing answer, and a pin change *is* a golden
   regeneration. **B-P26 stays open and documented** (`PIPELINES["aie"]` cannot lower W1;
   nothing is blocked).

**Person C**

1. ~~The CI wheel cache cannot be primed from the workflow~~ — **the job exists now** (B's pass):
   `.github/workflows/ci.yml`'s `prime-wheels`, `workflow_dispatch` with `confirm: prime`,
   downloads the 16 pinned wheels, `sha256sum -c`, saves the cache under the key the other jobs
   restore with `fail-on-cache-miss: true`. **Still not run: nobody has dispatched it, and CI
   has never run on a runner at all** — no `gh` and no token on this machine, so it could not be
   triggered from here. `design/07-environment.md` §2 carries the one-line recipe.
2. **The device run — the one item no machine here can close.** `m6.run` / `m6.diff` /
   `m6.trace` are built and **untested on hardware**; `test_T4_device_diff` skips because
   `/dev/accel*` is absent. It needs an XDNA1 (Phoenix) laptop and XRT; until someone runs it,
   the entry's end-to-end claim stops at `aircc --output-format=none`, and the honest-limits
   slide says so.
3. ~~`progress.md` sits at the repository root~~ — **moved** to
   [`../design/PROGRESS-C.md`](../design/PROGRESS-C.md) on 2026-09-15, references updated.

**User / architect**

1. **A's and C's signatures** on `06-interfaces.md` v3–v6. B's four are signed (2026-09-15,
   architect on B's behalf); a version counts as signed only when all three are.
2. ~~The pitch decision on a Tenstorrent beat~~ — **taken: there is one**, at 4:05, and it costs
   ~2 s of the five minutes. Cut it by deleting one `beat(...)` line if the rehearsal says the
   five minutes cannot afford it.
3. ~~Adjudicate R-TT-B and its credit~~ — **done, 2026-09-15**: accepted as measured, and
   explicitly *not* a general theorem (`design/08-tt-backend.md` §3.5, `PROGRESS-TT.md` §T6.1).

---

## Design phase (2026-09-12)

Phase-1 design is drafted under [`../design/`](../design/) — start at
[`../design/00-README.md`](../design/00-README.md), which carries the reading order, the
module→person ownership map, the status table, the interface-change process, and the fifteen
design decisions the architect's brief did not fix. **No implementation code exists yet**;
every document in `design/` is specification, and every AIR fact it rests on is cited from
`VERIFIED-AIR-FACTS.md` or `path:line` at mlir-air commit `ff95a9b`.

**State: the design set has been revised per `design/REVIEW-round1.md`.** The adversarial review
raised 13 blockers, 29 drift rows, 26 gaps and 8 plan risks with a 74-item edit list; the
architect ruled on it (rulings 1–9, recorded in `design/00-README.md` §7.1); all 74 items are
applied and all 13 blockers are closed. `design/RESPONSE-review1.md` has the item-by-item
disposition, the conflicts between an edit and a ruling, and the mechanical-check output.

**Handoff state (2026-09-13).** The design loop is closed by architect decision, after
`design/REVIEW-round1.md` and rulings 1–9; no further review round is planned and no design
question is blocking D0. `design/06-interfaces.md` stands at `CONTRACT_VERSION = 3` and is
waiting on the three co-signatures at D0 — that signature is the only gate between this document
and implementation. A final consistency sweep of the whole set on 2026-09-13 ran seven mechanical
checks — error-code catalogue both ways, test-id definitions, per-LLD FR subsets, fixture
parameters, the stale-string list, fenced-block classification, and the README's own reading
order and status — and came back clean after two stale spots (W2's schedule listing in M2 §6.3
and the cascade `at=`-pinning claims) were fixed; the output is in `design/RESPONSE-review1.md`
under *Final sweep checks (2026-09-13)*. The per-person D0 task list is
`design/05-work-breakdown.md` §2.

What changed that anyone reading the older documents must know:

* **W2's contract**: `def jacobi(U: sp.f32[T + 1, H + 2, W])` — **one** rank-3 parameter, the
  `0.2 ×` **five**-point stencil, write domain planes `1..T` × rows `1..H` × cols `1..W-2`.
  Fixture `H = W = 16`, **`PI = 2`, `HS = 8`** (`PI·HS == H`), `T = 4` (and `T = 5`). Every plane
  is drained. `PI = 4` is a `DMA-CHANNELS` **negative** — it is measured to fail `aie.connect`.
* **W3's contract**: `def sw(q: sp.i32[MQ], r: sp.i32[NR], S: sp.i32[MQ + 1, NR + 1])` with
  `sub = MATCH if q[i-1] == r[j-1] else MISMATCH`. The grammar gained a value-level `SELECT`,
  module-level `int` constants and transitive scalar substitution. `q`/`r` are staged into L1;
  every row of `S` is drained.
* **The flip is 1-D**: `grid(PK=4)`, `place(px=ax.k0)`, `stationary("B")`, cascade **ascending**
  (the opposite of the 2-D case, and measured both ways).
* **`06-interfaces.md` is at `CONTRACT_VERSION = 3` and is signed as v3, not v1 or v2** — the
  eleven round-1 contract edits and RULING 9's twelfth land *before* the D0 freeze.
  `ComputeNode` became `StoreNode` over an `ExprNode` tree; `BranchNode` joined `PlanNode`;
  `LegalMapping` gained `pi_u`/`ker_pi_u`; `MappingSummary` gained `residency`; the error
  catalogue is **43** codes.
* **Per-core L1**: W1 `12 288`, W1-flip `24 576` (`16 384` at M3's scope, before M4 adds the
  cascade `recv` tile), W2 `1 280`, W3 `240` (`232` at M3's scope) bytes. The arithmetic is in
  `design/02-hld.md` §7 and nowhere else.

**Residual unknowns that need a device or a build, and cannot be closed by editing** (now risks
R-18…R-21 and open questions B-O1/B-O4 in `design/01-requirements.md` §6–§7):

1. Whether a **merged MM2S stream** is correctly demultiplexed at two destinations (N-3). Every
   core is kept at ≤ 2 outbound endpoints as the mitigation.
2. The **packet-vs-circuit DMA rule** is measured on four module shapes and is **not** a
   documented upstream contract, so `DMA-CHANNELS` stays a warning wherever packet flows may
   appear. A device run is the only thing that turns "it compiled" into "it is correct".
3. **`npu2`** — nothing has been built for it; four of the eight goldens target a device no probe
   has touched. Due D2.
4. Whether **ping-pong fires** on W1's K loop in the real emitted module (one `air-opt` run).
5. **W2 and W3 end-to-end numerics** — every structural check is a proxy; `air.api` has no
   interpreter and `air-runner` is a timing model. Only a device run closes it.

**Next action: D0.** Install the pinned toolchain on all three machines and cache the four wheels
on the pinned interpreter (`design/07-environment.md` §1–§2); **sign `06-interfaces.md` v3**; and
write the three kernel sources — `kernels/w1_gemm.py`, `w2_jacobi.py`, `w3_sw.py` — verbatim from
`design/03-lld-M8-kernels-demo.md` §3.1, §3.3 and §3.4. `design/05-work-breakdown.md` §2 D0 has
the per-person task list.

---

## Person B implementation — state at 2026-09-13

*Appended by Person B at close-out (phase P7). Full detail, phase by phase, is in
[`../design/PROGRESS-B.md`](../design/PROGRESS-B.md); this section is the part someone who is
not B needs. All of it is on **`main`** at **`f7ecd70`**, pushed 2026-09-13; `role-b` is the
same commit.*

### State

**M0, M4, M5 and the off-device half of M6 are built and verified; M1, M2, M3 and M6's device
path are not.** On `role-b`: `spatial/model.py` at **`CONTRACT_VERSION = 5`** (33 dataclasses,
78 enforced invariants, the 43-code catalogue, canonical JSON), `spatial/m4_mapping.py` (the ten
passes of M4 §3.1 and all four protocol builders — fill/compute/drain, halo, wavefront,
cascade), `spatial/m4_selfcheck.py` (P1′ balance, P2b acyclicity, the §3.7.3 structural
invariants, the P3 DMA-channel budget), `spatial/m5_emit.py` (one public name, all thirteen
§3.2 translation rows), and the off-device half of `spatial/m6_tools.py`. The M7 harness shell
and its helpers (`conftest`, `golden`, `diagnostics`, `determinism`, `plan_interp`) were written
by B to unblock the above and are **flagged in every file's docstring for Person C**.

**All four variants run end to end on B's side** — `LegalMapping → m4.plan → m5.emit → AIR
text → aircc`: W1 GEMM, the W1 weight-stationary flip, W2 Jacobi and W3 Smith-Waterman, on
**both npu1 and npu2**. Gates **G2, G3, G4 and G5 are green on B's half only.**

Tests: **611 passed, 1 skipped, 12 deselected in 12.4 s** (`pytest`), **12 passed in 5.9 s**
(`pytest -m slow`), and the `PYTHONHASHSEED=1` vs `=2` `-vv` id/outcome diff over 612 lines is
**empty**. Architect re-run on `main`, 2026-09-13: **662 passed, 1 skipped, 37 deselected in
15.9 s** — the rise over 611 is B's `tests/tt` unit tests and the T5 traceability additions.
The one skip is the traceability gate for A's and C's FR groups, and it prints the
31 uncovered ids as its reason.

**`main` has all of this.** The architect fast-forwarded `main` to `role-b` (`b220f24`) and then
to `tt-backend` (`f7ecd70`) and pushed on 2026-09-13; all three branches name the same commit.

**The load-bearing caveat.** Every `LegalMapping` in the suite is a **hand-written literal**
transcribed from Person A's LLDs, because M1/M2/M3 do not exist. **No test here proves the
legality checker will produce them.** That is the single largest open risk to G2–G5 as a chain,
and it is not a risk B can close.

### Decisions taken, with reasons

1. **`CONTRACT_VERSION = 4`** (architect's ruling, P0d). Two additions, each forced by a
   requirement rather than by convenience. **`HerdPlan ∈ PlanNode`**: M4 §6.1 printed a
   `<HERD>` marker in `segment_body` that the frozen `PlanNode` union had no member for, so M5
   would have had to *infer* where the herd opens — which is exactly what D-14 forbids. It is
   now a node, at top level, equal to `MappingPlan.herd`. **`KernelModel.bindings`**: FR-M7's
   `TENSOR_PLAN` needs concrete L3 shapes, and M1 §6.3 prints shapes like `("MQ+1","NR+1")`
   that M0 rejects two ways; §2.1 now says normatively that a shape entry which is not a bare
   NAME resolves to the int it evaluates to under `bindings` at capture.
2. **`CONTRACT_VERSION = 5`** (three rulings, P2b). **`Statement.expr: ExprNode`** — the frozen
   `Statement` carried `kind`, `target`, `reads` and `op` and nothing else, from which M4 can
   rebuild `acc = acc + <product>` and **cannot** rebuild W2's five-point stencil or W3's
   `max`/`Select` recurrence. It was a contract gap, not an implementation gap, and it blocked
   W2 and W3 outright. M4 now rewrites the kernel's own tree load by load and synthesises no
   arithmetic of its own except the accumulator zeroing and the cascade accumulate, neither of
   which corresponds to a kernel statement.
3. **Ruling R2 — `declared` means "a clause names this operand's delivery"**, not "the
   derivation disagreed". The documents wanted W1's `C: stationary (derived)` *and* the flip's
   `B: stationary (declared)` out of one rule, and no rule produced both; R2 makes it a fact
   about the schedule, so W1 now renders `C: stationary (declared)` too. It moves one flag and
   never the delivery — the `*.air.mlir` goldens were byte-identical across the change, which
   is the check that proves it.
4. **Ruling R3 — the loop-axis naming rule.** `p<root>_bundle`, `<axis>_drain`, `<axis>_source`
   (added at P4 for W3's row loop), and a compute or zeroing nest named by the **post-tiling
   axis it realises** (`i1`, `j1`, `k1`), never positionally. `06-interfaces.md` §5.5's old
   `<operand>_bundle` footnote matched neither worked example.
5. **Ruling R-W3-1 — W3 keeps its source and drain on `S`'s own boundary columns.** The
   fallback of dropping `EastOut` and the tail PE's put was available and was **not taken**:
   the source puts `S[i, 0:1]` and the drain gets `S[i, NR:NR+1]`, so `MappingPlan.tensors` is
   exactly `(q, r, S)` and no synthetic tensor exists. It is a measurement, not an argument —
   `aircc` runs `air-verify-hierarchy-locality{strict=true}` on the **placed** IR by default
   (`tools/aircc/aircc.cpp:1213-1218`, `:264` is `cl::init(PIV_error)`), and both it and an
   explicit run of that pass alone are clean.
6. **W2's swap pair is seeded, not just `cur`.** M4 §3.6.1 stages plane `lo` into `cur` alone;
   §6.3 then claims of **every** drained plane that "the value written back is the one that was
   staged in". The planes alternate between `cur` and `next` and the drain puts the strip whole,
   so three values would be undefined — the two Dirichlet columns, and the domain-edge ghost
   rows whose guards are false exactly there — and the plan would write an alternating boundary
   into L3 that no oracle matches. The plan therefore carries a `StoreNode` copy
   `next[i1,j] = cur[i1,j]` right after the `UIn` get, the same device the accumulator zeroing
   and the cascade accumulate already use. `UIn` stays 1 put / 1 get per index.
7. **P2b channel edges are FIFO-paired.** §3.7.2's all-pairs rule — "every put node and every
   get node whose `(channel, concrete index)` match" — reports a **cycle on W2's real plan**,
   and every channel edge in that cycle pairs a *later* put with an *earlier* get, which a FIFO
   never does. `_pairs` zips the k-th put to the k-th get where the pairing is defined and falls
   back to all-pairs otherwise. It is strictly fewer edges, so it cannot turn a cyclic plan
   acyclic where the pairing does not apply; the reversed halo still raises `CHANNEL-CYCLE`.
8. **The packet-vs-circuit DMA rule was derived from measurement and then found in the source.**
   `mlir/lib/Transform/AIRDmaToChannel.cpp:1598-1740`, pass option `shim-dma-channels-per-col`,
   default 2 (`mlir/include/air/Transform/Passes.td:1808-1812`), wheel
   `0.0.1.2026091204+ff95a9b`. Per segment and per direction the per-column shim pressure is
   `#non-broadcast + Σ_span ceil(members_span / span)`; above 2, **every** L3-attached channel of
   that direction becomes `npu_dma_packet`. Five probes reproduce it and the pass prints its own
   arithmetic (`warning: auto-upgrading 3 input channels to dma_packet (per-column pressure 3
   exceeds shim DMA limit of 2)`). **Labelled measured, not contractual** (R-19/R-21): it is a
   pass option's default on a pinned wheel. That is why `03-lld-M4-mapping.md` §3.8 splits the
   verdict — a circuit-switched overflow is a `DMA-CHANNELS` **error** (W2 at `PI = 4`), an
   overflow only packet-capable channels cause is a **warning** (W3's three L3 inputs).
9. **B-P23 — a self-check failure that can only be M4's own bug reuses `PROTOCOL-UNSUPPORTED`**,
   with `details["internal_consistency"] = True`, `details["invariant"] = <§5.6 number>`, a
   `reason` beginning `internal:` and a `fix` that asks for a bug report rather than blaming a
   clause the user wrote. The honest spelling would be a dedicated code; `06-interfaces.md` §6.3
   is frozen at 43 codes and M0 enforces the per-stage set, so the catalogue was not grown
   mid-flight. The same reasoning applies to M5's `EMIT-AIR-API` (**B-P13**, **B-P20**).
10. **B-P25 — the W2 fixture invariant.** *Planes `1..T` of the input `U` must carry plane 0's
    boundary rows (`0`, `H+1`) and columns (`0`, `W-1`).* The plan carries plane 0's boundary
    **forward**; the kernel text reads plane `t`'s **own** boundary out of the one rank-3 array.
    The two agree exactly when the input's boundary is time-invariant, which is what "read-only
    Dirichlet boundary" (`02-hld.md` §7.2) means for a one-array kernel. The old test fixture
    violated it silently. **Person C must satisfy it in `make_fixture.py`.**
11. **B-P26 — `PIPELINES["aie"]` is not usable on every module we emit.** W1's fails: its
    `C2L3` bundle index goes through the `repeats` strip-mine `affine_map` (npu1 runs a 2×2
    logical grid on a 1×2 physical herd), `air-to-aie` cannot fold it to a constant, and the
    pipeline dies. `aircc` reaches `air-to-aie` with the dependency and dma-to-channel passes
    already run; our pipeline does not. Consequence, recorded rather than worked around: two
    `ir_facts` goldens each lost **one key** — `w1.base.npu1` lost `cascade_channels` and
    `w3.base.npu1` lost `_pipeline_pingpong` — with **no measured number altered**. The cascade
    fact is asserted on W3 (0) and the flip (3) instead. A GEMM with no cascade has nothing to
    say about cascade flows.

### Verified facts

Re-measured in the close-out session of 2026-09-13 unless the row says otherwise. Every wall
clock and every lowering count is **this machine, this pin** (R-19/R-21).

**`aircc --device <target> --output-format=none`, eight runs, one at a time, scratch `--tmpdir`
and `cwd` outside the repository.** `error:` lines counted on stderr, because the exit code
alone is never trusted (FR-T5):

| variant | npu1 | npu2 |
|---|---|---|
| W1 | exit **0**, **0** `error:`, 0.88 s | exit **0**, **0** `error:`, 0.31 s |
| W1-flip | exit **0**, **0** `error:`, 0.56 s | exit **0**, **0** `error:`, 0.36 s |
| W2 | exit **0**, **0** `error:`, 0.21 s | exit **0**, **0** `error:`, 0.21 s |
| W3 | exit **0**, **0** `error:`, 0.30 s | exit **0**, **0** `error:`, 0.31 s |

**Per variant:**

* **W1.** `physical_herd` is `(1,2)` on npu1 and `(2,2)` on npu2, so npu1's text carries
  `air.api`'s strip-mine loop and `#map1 = affine_map<()[s0, s1] -> (s0 * 2 + s1)>` and npu2's
  does not — two core ELFs against four, which is what proves the npu2 compile is real and not
  a short circuit. `ir_facts`: `pingpong_unroll 2`, `hoist_alloc_count 2`,
  `broadcast_pattern_count 0`, `pingpong_iter_args 4`. The `air-opt` transform pipeline doubles
  the K loop's step from 16 to 32 with 4 `!air.async.token` iter-args. **The emitted text is
  byte-identical to a hand-written upstream-API probe of the same shape** (P1's probe oracle).
* **W1-flip.** **9 `aie.flow`, 0 `aie.packet_flow`, 3 `aie.cascade_flow`** on both targets,
  ascending: `(0,2)→(1,2)`, `(1,2)→(2,2)`, `(2,2)→(3,2)`, asserted as an exact line list. The
  **2-D descending** variant compiles too — `(0,5)→(0,4)`, `(0,4)→(0,3)`, `(0,3)→(0,2)`.
  `ir_facts`: `cascade_channels 3`, `pingpong_unroll 2`, `hoist_alloc_count **1**` (`b` is
  hoisted above `i0` and never re-fetched, so there is one ping-pong candidate and `a` is it),
  `broadcast_pattern_count 0`, `lock_init_histogram {0: 9, 1: 9}` — 18 locks over four cores.
  The `npu_dma_stream` contingency of FR-K2 was **not** needed: `aircc` accepted `npu_cascade`
  on both generations at the first attempt.
* **W2.** **6 `aie.flow`, 0 `aie.packet_flow`** — `UIn` ×2, `UOut` ×2, `ToNorth`, `ToSouth` —
  so every core sits at exactly 2 S2MM and 2 MM2S, which is §3.8's prediction measured.
  `ir_facts`: `broadcast_pattern_count 0`, `pingpong_unroll 0` (the `cur`/`next` pair sits
  outside the timestep loop, so there is no candidate loop), `lock_init_histogram
  {0: 8, 1: 2, 2: 6}` — 16 locks over two cores. `PI = 4` is rejected by `m4.plan` with
  `DMA-CHANNELS` naming PE `[1]` and the three circuit-switched inbound channels, **before**
  `aircc` — which is measured to fail that shape with `'aie.connect' op … targets same dst`.
* **W2, experiment E1.** `air-opt -pass-pipeline='builtin.module(air-dependency)'` on **our own
  module**, then a mechanical walk of the token graph:
  **no token edge joins a PE's halo put to its own get.** Phase 0's four ops all take the
  timestep loop's iter-arg token; phase 1's all take the compute nest's token; the `scf.if`
  results the puts yield are dead. The taint set is 8 values, the gets' dependency lists are two
  values, the intersection is empty. The non-vacuity guard asserts each put's token *does* reach
  its `air.wait_all` and its `scf.if` result. **E1 is answered for the dependency pass; it is
  not a statement about run-time channel-slot stalls, which only a device closes.**
* **W3.** `air-verify-hierarchy-locality` clean at `strict=false` **and** `strict=true`, run
  explicitly rather than inferred from `aircc`'s exit code. `ir_facts`:
  `broadcast_pattern_count 0`, `cascade_channels 0`, `lock_init_histogram
  {0: 20, 1: 16, 2: 4}` — 40 locks over four cores. `scf.if` 4, `arith.select` **2**,
  `arith.maxsi` 6.

**Interpreter results** (`tests/helpers/plan_interp.py`, green this session). It executes the
**plan**, not the emitted IR — D-9, and the honest-limits slide is unchanged:

| variant | oracle | result |
|---|---|---|
| W1 | `A @ B`, `numpy.random.default_rng(0)` | **exactly equal**, both targets |
| W1-flip | `A @ B`, integer-valued `f32` in `[-8, 8)` | **exactly equal**, both targets and the 2-D variant. Dropping only the cascade accumulate gives `A[:, 48:] @ B[48:, :]` — the tail PE's own slice — so the check is not vacuous |
| W2 | two-loop numpy Jacobi, `T = 4` and `T = 5` | **max abs error 0.0** in all four runs (`tol` is `1e-5`); boundaries and every drained plane checked |
| W3 | textbook two-loop Smith-Waterman DP, `MQ = 32` and the peeled `MQ = 31` | **exactly equal**; row 0 and column 0 stay zero, `q` and `r` untouched, `dtype == int32` |

**Probe comparisons** (measured at P4/P5/P6, **not** re-run in the close-out session). Each
compares our emitter's lowered output against a hand-written upstream-API probe of the same
shape: **W3 vs `review/w3c.py`** — 8 circuit `aie.flow` **byte-identical** tile for tile and DMA
channel for DMA channel, 4 `aie.core`, 40 `aie.lock` identical; 6 `aie.packet_flow` against the
probe's 9, because `QIn`'s declared `broadcast_shape` makes one flow with two `packet_dest`s
where the probe needs four. **W2 vs `q/pi2/w2_pi2.py`** — 6 `aie.flow` / 0 `aie.packet_flow` /
2 cores / 8 `scf.if` / 7 gets all identical. **W1-flip vs `review/flip1d.py asc`** — 3
`aie.cascade_flow` identical tile for tile, 9/0 flows, 4 cores, 18 locks, 11 puts / 4 gets, 4
allocs, all identical. **Every remaining difference is a difference in the program, not in the
routing** — a five-point stencil against a four-point one, a per-timestep drain against a single
one, a declared multicast against a replicated put.

**Error-code coverage, measured this session**: **12 of the catalogue's 43 codes are raised** —
mapping **5/5**, emission **2/2**, toolchain **5/6** (only `TOOL-NO-DEVICE` is short, and it
needs a device). The 31 unraised are clause 0/8, grammar 0/7 and legality 0/15 — every one of
them M1/M2/M3's.

### Open items, by owner

**Person A**

1. **Sign `06-interfaces.md` v4 and v5.** Neither has a signature.
2. **M1 must emit `KernelModel.bindings`** (v4 — it already reads the integer bindings at
   M1 §3.4 line 18) **and `Statement.expr`** (v5 — the tree M1 §6.1–§6.3 already prints). No
   `KernelModel` constructs without either.
3. **The four `LegalMapping` literals under `tests/fixtures/mappings/` are B's transcription of
   A's LLDs and must be replaced by M3's output.** `test_M4_plan_equals_literal` then becomes
   the D6 test M7 §3.8 names.
4. **B-P10** — M1 §6.2's dependence list is not in M0's sort order (five lines to re-order).
5. The clause, grammar and legality negatives: 30 of the 43 error codes are raised by nothing,
   and `test_D1_schema` / `test_D3_catalogue_complete` do not exist. The `w2_zero_t` (`T = 0`)
   `SWAP-PARITY` fixture is the one that keeps a legality code reachable.

**Person C**

1. **Review B's edits to `spatial/m6_tools.py`** — `ERROR_LINE` (widened by one `loc(...)`
   alternative, because M6 §3.2's pattern as written matches **no** `aircc` diagnostic at all),
   `PIPELINES["aie"]` (prefixed with `air-place-herds` using the geometry `aircc` itself
   resolves, without which `lock_init_histogram` is unreadable), and
   `EXTRACTORS["cascade_channels"]` (counts `aie.cascade_flow`, which is 3 where the
   `npu_cascade` string is 4).
2. **Replace the harness stubs** written by B: `tests/conftest.py`, `tests/helpers/golden.py`,
   `diagnostics.py`, `determinism.py`, `plan_interp.py`, and the three test files whose
   docstrings say "Person C owns this file". The traceability gate's `_UNBUILT` list is the one
   line C deletes when A's and C's FRs are covered.
3. **Honour B-P25 in `make_fixture.py`** (the W2 boundary invariant above).
4. **Update the demo line**: `05-work-breakdown.md` §5 step 2 and the M8 demo screen now read
   `C: stationary (declared)`, not `(derived)` — ruling R2. The channel line renders
   `CascadeK size=(3,)`, a tuple, not `[3]`. **`demo/` is empty**; B has produced every line
   steps 2 and 3 read off the screen, but the script is unwritten.
5. **The device path**: `m6.has_device` / `run` / `diff` / `trace`, and every `requires_device`
   test — there is currently **not one** in the suite.
6. **B-P26**, **B-P29** (NFR-5 covers functions and classes; module-level constants need an AST
   lint), **B-O8** (`str(module)` stability across an upgrade within the pin).

**User / architect**

1. **Fast-forward `main` to `role-b`** — **done 2026-09-13**: `main` is at `f7ecd70`, pushed.
2. **Sign v4 and v5 as B** — B's rows in `00-README.md` §4's signature block.
3. Rule on **B-P12** (`ChannelSite.is_async` cannot be honoured — `air.api` has no asynchronous
   form to select), **B-P13** / **B-P20** (no error code for an internal-consistency failure),
   **B-P15** (M5 §7's `test_sequential_emits_scf_for` row is unimplementable as written),
   **B-P27** (the LLDs cite an `FR-D9` that does not exist; they mean FR-S3 item 8), **B-P28**
   (`00-README.md` §3's M0 row still says "not started"), **B-P30** (test-id drift), **B-P31**
   (M4 §8 says M4 uses numpy; it does not), **B-P3** (the "four wheels" line).

### How to resume

```bash
git checkout main                                 # f7ecd70; role-b is the same commit
uv venv --python python3.12 --seed .venv          # or reuse the existing .venv
.venv/bin/python -m pip install --no-index --find-links vendor/wheels 'mlir_air[aie]' pytest
.venv/bin/python -m pip install -e . --no-deps

.venv/bin/python -m pytest                        # 611 passed, 1 skipped, 12 deselected, ~12 s
source scripts/airenv.fish                        # bash/zsh: source scripts/airenv.sh
.venv/bin/python -m pytest -m slow                # 12 passed, ~6 s — needs aircc on PATH
```

`scripts/airenv.{fish,sh}` is the only thing that puts `aircc`, `aiecc` and Peano on `PATH`;
without it every `slow` test **skips with a reason** rather than failing. `vendor/wheels/` is
git-ignored and ~644 MB — check it with `sha256sum -c vendor/wheels/SHA256SUMS`.

**Goldens** live in `tests/golden/`: eight `*.air.mlir`, ten `*.plan.json`, eight
`*.summary.txt`, four `*.ir_facts.json`. Regenerate with **`pytest --update-goldens`**, which
has three guards and refuses outright (a) in CI, (b) when `check_pin()` says the installed
toolchain is not the pinned one, and (c) when `tests/golden` has uncommitted changes — so the
regeneration diff is the only diff. It also re-renders `tests/README.md`, the FR → test inverse
index. **A golden is only valid for the pin**; a pin mismatch **skips** the golden tests rather
than failing them.

Person B's phase-by-phase record, every spec reading taken and every number measured, is
`design/PROGRESS-B.md` — start at its **Status** block, then **§P7** for the definitions of
done and the consolidated open-items table.

### Not verified

* **Anything on a device.** No hardware has been touched. The plan interpreter executes the
  **plan**, never the emitted IR (D-9); `air-runner` is a timing model, not a correctness
  oracle; every structural check is a proxy. Only a device run closes W1/W2/W3/flip numerics,
  and the honest-limits slide is unchanged.
* **Any other machine or any other wheel.** Every wall clock, every flow and lock count, and
  the whole packet-vs-circuit rule are this machine at wheel `0.0.1.2026091204+ff95a9b`. The DMA
  rule is a pass option's **default**, not a documented upstream guarantee (R-19, R-21).
* **That the legality checker will produce the four `LegalMapping` literals** the whole suite
  rests on. M1/M2/M3 do not exist.
* **Shapes that raise by name rather than being built**: a rank-2 halo (`ToWest`/`ToEast`), a
  halo at `PI ≥ 3`, a wavefront with a rank-2 herd or a row span other than 2, a cascade with a
  chain axis of extent 2 or a reduction operator other than `+`, and `r_space` of rank > 1. Each
  raises `NotImplementedError` naming the phase; none guesses.
* Everything already marked `[UNVERIFIED]` in this document and in
  `VERIFIED-AIR-FACTS.md` remains so.

### Tenstorrent backend (B, stretch) — state at 2026-09-13

*Appended by Person B at the TT close-out (phase T5). Full detail, phase by phase, is in
[`../design/PROGRESS-TT.md`](../design/PROGRESS-TT.md); the spec is
[`../design/08-tt-backend.md`](../design/08-tt-backend.md). Merged to **`main`** at **`f7ecd70`**,
pushed 2026-09-13; `tt-backend` is the same commit. This is the stretch item and the designated
cut: the AIE path above is unaffected by anything in it.*

#### State

**All four gates are green.** The same, unchanged `MappingPlan` that the AIR emitter consumes is
turned by a **second** emitter into a TT-Metalium program and **executed** on Tenstorrent's
functional simulator `ttsim`:

| Gate | Workload | Result | ttsim's own cycle counter | wall, one `run()` | Where |
|---|---|---|---|---|---|
| **T1** | W1 GEMM | **exact** | 11 509 553 | ≈50 s | `PROGRESS-TT.md` §4 |
| **T2** | W3 wavefront | **exact** | 24 426 | ≈2 s | §T2.1 |
| **T4** | W1-flip cascade | **exact** | 12 129 177 | ≈55 s | §T2.2 |
| **T3** | W2 Jacobi, `T = 4` / `T = 5` | **exact** | 183 914 / 229 133 | ≈1.3 s | §T3.1 |

T4 was reached **in T2's own session with no emitter change** — the emitter dispatches on plan
shape and never on a workload. Each of the four is green under §7's three-part rule: the
execution is exact against its own numpy oracle under `np.array_equal`, a **negative control** (a
one-line mutation of the emitted kernel) returns a different answer, and the result equals
`tests/helpers/plan_interp.py`'s interpretation of the same plan **element for element**.

**On `tt-backend`**: `spatial/m5tt_emit.py` (`MappingPlan` → `TTProgram`), `spatial/m6tt_run.py`
(executes it through `ttnn.generic_op`), `tests/tt/` (5 modules), `scripts/tt_env.sh`,
`scripts/tt_probe_align.py`, `scripts/tt_probe_coords.py`, `vendor/tt/SHA256SUMS`,
`design/08-tt-backend.md` and `design/PROGRESS-TT.md`. **No file owned by A or C is touched**,
and `06-interfaces.md` stays frozen at `CONTRACT_VERSION = 5` — the whole point is that the plan
did not have to change.

**Suites.** The default `.venv` suite is green and carries the emitter's own unit tests (it needs
no simulator: `m5tt_emit` imports nothing but the standard library and `spatial.model`). The
device suite is `.venv-tt`, `-m requires_ttsim tests/tt`: **25 tests, all `PASSED`, exit status
0** at T3, and again on the architect's re-run of 2026-09-13 — **25 `PASSED`, 0 failed, exit 0,
47 577 057 ttsim cycles, 262 s**; see `PROGRESS-TT.md` §T5.3.

**What is `[not run]`, and none of it is closeable here**: real Tenstorrent **silicon** (no
hardware was touched at any point); the Tensix **compute engines** (matrix/vector — everything
runs as scalar C++ on one data-movement RISC-V); **double buffering** (`ping_pong_candidate` is
read and ignored); **`f16`/`bf16`** plans (no scalar C++ type on a data-movement core); and any
**timing** claim of any kind (`ttsim` is a functional simulator, not a timing model).
`PROGRESS-TT.md` §5 is the table with the exact refusals.

#### Decisions taken, with reasons

1. **Tenstorrent, not Qualcomm Hexagon, as the second backend.** Hexagon is a **single DSP core**
   with HVX/HMX over a shared scratchpad — no PE array and no channel model, so `grid`, `place`
   and the put/get protocols have nothing to map onto; and Qualcomm already ships
   `qualcomm/hexagon-mlir` (Triton/PyTorch → linalg → Hexagon), which needs their proprietary
   SDK. Tensix is a grid of cores with private L1 and a NoC, which is the shape the plan already
   describes, and it runs **device-free** on an Apache-2.0 functional simulator. `CLAUDE.md`'s
   scope rules record Hexagon as out and staying out.
2. **`MappingPlan` → TT-Metalium directly, through `ttnn.generic_op`, rather than through
   `tt-mlir` or `tt-lang`.** Going through the TTIR/TTNN/D2M/TTKernel/TTMetal dialects would mean
   **building a compiler** — a second toolchain to pin, install and version-check — to reach the
   same `ttnn` program descriptors we build from Python in one step. The plan already carries
   explicit buffers, regions, loops and expression trees, so there is nothing an MLIR round trip
   would infer that we do not already have, and **D-14 forbids the emitter inferring anything**.
   Reversible: if TT ever became a *performance* target, `tt-mlir` is the right entry point
   (`08-tt-backend.md` §10).
3. **R-TT-A′ — alignment is a *relative* congruence, not an absolute one.** T1's rule (every DRAM
   offset, L1 offset and length a multiple of 32 B) was measured wrong-because-too-strong: over
   23 single-transfer cases, one simulator process each, exactly the **mismatched** pairs fail,
   and what the NoC constrains is the **difference** of the two byte offsets — mod 16 for a
   write, mod 32 for a read — never the size. Consequence: `pad_elems = 0` for every tensor of
   all four workloads, and the scratch buffer, read-modify-write and single-writer machinery that
   the absolute rule's `p = 7` would have forced **does not exist** (§T2.4).
4. **R-TT-B — a link's k-th put occurrence pairs with its k-th get occurrence**, and a remote
   write's destination is read off the **paired get**, not off a single landing region per
   channel. This is what lets one link land its payload in `cur` on one occurrence and in `next`
   on the next — T2's blocker on W2 (§T3.2).
5. **R-TT-B's credit rule, which is the one amendment measurement forced on the ruling.** A
   depth-1 FIFO is correct for a chain and **deadlocks W2** (measured: 180 s and killed, against
   ≈1.3 s clean), because two PEs that each put on their outbound link *before* they get on their
   inbound one make the **wait-for** graph cyclic even where the channel graph is acyclic — which
   is why P1′ and P2b do not exclude it. The credit is the smallest gap between two
   get-occurrence landing regions that alias, computed in **dynamic** order (the static
   occurrence list gives 1 at `T = 5` and hangs): **1** for W3 and the cascade, **2** for W2
   (§T3.3).
6. **The `empty` release goes at the top of the *next* get on the same link, before its
   `wait_min`** — not at the end of the enclosing body. W3's herd body is one `i` loop of **step
   2** holding two rows, so the literal reading deadlocks; the operational form is the same rule
   wherever the body is unambiguous, and only W3 can tell the two apart (`08-tt-backend.md`
   §3.5).
7. **`f32` constants are narrowed at the point of use**, `((float)(0.2))`. Measured: the `f`
   suffix is byte-for-byte identical, and the **unsuffixed `double` literal is not** — 358 of 896
   interior elements one ulp off and 38 % more simulated cycles, because `0.2 * <float>` is then
   evaluated in `double`. The cast is kept over the suffix because it is also correct for a
   `Const.text` with no decimal point (§T3.4).

#### Verified facts

Every number here was measured in this repository on 2026-09-13; the section of
`design/PROGRESS-TT.md` that records it is named.

* **The four gates**, as tabulated above — exact, negative control, and element-for-element
  equality with `plan_interp` for each (§4, §T2.1, §T2.2, §T3.1).
* **16 semaphores per core**, ids 0..15: 8 accepted, 32 refused with `Semaphore id 16 exceeds max
  value 15`. Now `m5tt_emit.TT_SEM_LIMIT`, checked **before** a device is opened; W3 and the flip
  use 6, W2 uses 4 (§T2.3, Q-TT5).
* **A-TT1 holds.** One `CBDescriptor` list over one `CoreRangeSet` gives every core the *same* L1
  addresses, so a producer can compute the consumer's buffer address as its own — which is what
  makes a remote write addressable at all. Measured with a probe kernel on W3's own CB table; all
  six CB bases ≡ 0 mod 32 (§T2.3, Q-TT1).
* **`worker_core_from_logical_core` returns *virtual* coordinates and they work.** It gives
  `(18..21, 18)` where the soc descriptor's physical row is `1-1 … 4-1`; a four-core chain
  delivers with **either** on an unharvested Wormhole B0, and an off-by-one mis-delivers (the
  negative control). The emitter passes the virtual value — the one that survives harvesting,
  which is reasoning and not a measurement (§T2.3, Q-TT2/C-TT3).
* **Alignment is a relative congruence** (R-TT-A′): 23 cases, one process each; mod 16 for a
  write, mod 32 for a read, on the *difference* of the two byte offsets. ttsim's
  `UndefinedBehavior` is **fatal** — it prints and the host exits 1 — which is why the sweep had
  to be split (§T2.4).
* **ttsim does not flag a reversed barrier / increment.** Move `noc_semaphore_inc` *before*
  `noc_async_write_barrier()` at both `West` put sites and the answer is **unchanged** with **no
  `UndefinedBehavior` printed**: the simulator is **not an oracle for ordering hazards**. The
  emitter's order is unchanged; the reversal lives inside one test (§T2.3, Q-TT3).
* **Soft-float scalar `f32` is bit-exact against numpy** — max abs error **0.0** over the whole
  tensor at `T = 4` and `T = 5` — under two conditions: exact left-nested parenthesisation, and
  the narrowing cast. A bare `double` literal costs **358 of 896** elements one ulp (§T3.4,
  Q-TT4).
* **The depth-1 credit deadlocks W2, and R-TT-B's credit rule fixes it** — 180 s and killed
  against ≈1.3 s clean; kept as `test_one_credit_per_link_deadlocks`, marked `slow`, capped at
  60 s (§T3.3).
* **Environment, pinned and checksummed**: `ttnn==0.78.0`, ttsim **v1.10.7** (`libttsim_wh.so`),
  **sfpi 7.75.1**, and a `soc_descriptor.yaml` taken out of the wheel. `.venv-tt` is built from
  the `vendor/tt/` cache (git-ignored, like `vendor/wheels/`) whose **`SHA256SUMS` is committed**;
  `scripts/tt_env.sh` **unsets `TT_METAL_HOME`** — a stale value sends `tt_metal` looking for a
  source tree that does not exist (§2).
* **Neighbours, architect-verified 2026-09-13** and to be named in any pitch:
  `qualcomm/hexagon-mlir` (Triton/PyTorch → linalg → Hexagon, BSD-3, needs the proprietary SDK;
  Hexagon is a single DSP core, not a PE grid), **TileLoom** (arXiv 2512.22168, Triton →
  Tenstorrent), `tenstorrent/tt-lang`, `kernelize-ai/triton-tenstorrent`,
  `triton-lang/triton-ascend`. **No project was found targeting both AIE and Tensix from one
  spatial DSL — "none found", which is not "none exists".**

#### Open items

1. **The proposed TT error codes are deliberately *not* in the frozen 43-code catalogue.**
   `TT-ALIGNMENT` and the `EMIT-TT-*` names exist only in message text: `m5tt_emit` raises
   `TTEmitError` / `TTNotImplemented` / `TTAlignmentError` and `m6tt_run` raises `TTRunError`,
   none of which subclasses `SpatialError` or carries a `Diagnostic`, so none can reach
   `RAISED_CODES`. `test_TT_ALIGNMENT_is_not_in_the_frozen_error_catalogue` is the standing check.
   **If the architect wants them in the catalogue, that is a `06-interfaces.md` change and all
   three owners must agree** (`00-README.md` §4) — B did not make it unilaterally.
2. **ttsim is not an oracle for ordering hazards** (Q-TT3 above, O-TT2 in `08-tt-backend.md` §9).
   The write-barrier-before-increment rule stands on the `noc_async_write` contract, **not** on a
   measurement, and only silicon can close it.
3. **The TT suite's pytest summary line is lost.** ttsim's exit ends the process without flushing
   Python's stdout, so `N passed in Ns` never reaches a pipe or a file — reproduced with
   `PYTHONUNBUFFERED=1`, which does not help, because the loss is in the C++ exit and not in the
   buffer policy. **Judge that run by its exit status and its `-rA` `PASSED` lines**, never by a
   summary line (§T3.6).
4. **Two rulings want the architect's adjudication**, both from T3 and both already implemented
   and measured: **R-TT-B** as written into `08-tt-backend.md` §3.5, and its **credit**, the
   amendment the measured deadlock forced on it (§T3.2, §T3.3).
5. **Person C**: the slide text is written out ready to paste in `08-tt-backend.md` §8 and
   condensed to three bullets plus one Q&A row in `PROGRESS-TT.md` §T5.4. `05-work-breakdown.md`
   §5 and §6 are C's and were **not** edited by B.
6. **Everything is pushed.** The architect merged `tt-backend` into `main` (`f7ecd70`) and pushed
   it on 2026-09-13.

#### How to resume

```bash
git checkout main                                         # f7ecd70; tt-backend is the same commit
source scripts/tt_env.sh                                  # bash/zsh; builds .venv-tt on first use
.venv-tt/bin/python -m pytest -rA -q -m requires_ttsim tests/tt   # 25 PASSED, exit 0, ~286 s
```

`scripts/tt_env.sh` is the only thing that makes the simulator path work: it builds `.venv-tt`
from `vendor/tt/` on first use, names `libttsim_wh.so` in `TT_METAL_SIMULATOR`, sets
`TT_METAL_SLOW_DISPATCH_MODE=1` and **unsets `TT_METAL_HOME`**. Check the cache with `sha256sum -c
vendor/tt/SHA256SUMS`; `vendor/tt/`'s artefacts are git-ignored, its `SHA256SUMS` is committed.

**Two virtualenvs, and they cannot be merged**: the `ttnn` wheel pins `numpy<2` and the project
pins `numpy==2.5.3`. The emitter's own unit tests need no simulator and run in the default
`.venv` suite; only `tests/tt/test_tt_*.py` need `.venv-tt`, and without the environment they
**skip with a reason naming the missing variable** rather than failing.

The state file is **`design/PROGRESS-TT.md`** — start at its **Status** block, then **§T5** for
the FR-TT traceability result (14 of 14 covered), the consolidated Q-TT / C-TT / R-TT ledger, and
both suites' skip and deselect lists. The spec is `design/08-tt-backend.md`; §5 is the pin, §7 the
gates, §8 the honest-limits text, §9 the five open items and §10 the neighbours.

#### For the pitch

What may be claimed, in one sentence: **the same `MappingPlan` — unchanged, and with
`06-interfaces.md` still frozen — executes bit-exactly on a second spatial NPU's functional
simulator**, on four workloads (a GEMM, a wavefront with a real core-to-core protocol, a cascade,
and a bidirectional halo whose arithmetic is floating-point), each against numpy, each with a
negative control that makes the answer change, and each equal element for element to our own plan
interpreter — so the plan and the P1′/P2b checks are demonstrably target-neutral rather than
asserted to be, and the TT side's semantics are *executed* rather than argued. What may **not** be
claimed: anything about **silicon** (no Tenstorrent hardware was touched at any point, and only
the value of `TT_METAL_SIMULATOR` separates the two paths — an untested equivalence); anything
about **performance** (ttsim is functional, the cycle counts are its own counter and not a timing
model, the compute is scalar C++ on one data-movement RISC-V with the Tensix matrix and vector
engines untouched, and there is no double buffering); that Tenstorrent is an **AIR target** (it is
not — it is a second, separate emitter behind the same plan); or **Hexagon**, which is out of
scope and stays out. And if asked what is novel: no project was found targeting both AIE and
Tensix from one spatial DSL — *none found*, which is not *none exists*.
