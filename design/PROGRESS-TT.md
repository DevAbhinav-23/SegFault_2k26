# PROGRESS — the Tenstorrent backend

*State file for the second emitter. Phases **T1**, **T2** and **T3**, 2026-09-13, branch
`tt-backend`. Nothing pushed. The AIR side's state file is `design/PROGRESS-B.md`; nothing in the
M5 LLDs changes.*

## Status

**T1, T2 and T3 are done and measured. All four workloads execute on `ttsim` and are exact**
— W1 (GEMM), W3 (Smith-Waterman wavefront), W1-flip (cascade) and, since T3, W2 (Jacobi halo) at
both parities of `T` — **from the same `MappingPlan` the AIR emitter consumes.**
`spatial/m5tt_emit.py` turns a `MappingPlan` into a `TTProgram` — a core range, one data-movement
kernel's C++, a circular-buffer table, a semaphore table, an io-tensor list and per-core runtime
args — and `spatial/m6tt_run.py` executes it through `ttnn.generic_op` on Tenstorrent's functional
simulator. Each of the four is exact against its own oracle under `np.array_equal`, equal element
for element to `tests/helpers/plan_interp.py`'s numpy interpretation of the same plan, and changed
by a deliberate one-line mutation of the emitted kernel. **Every gate — T1, T2, T3, T4 — is
green.**

**T2's subject is the first core↔core protocol.** W1 has no core↔core channel at all, so its
whole program is DRAM traffic on four independent cores. W3's `West` and the flip's `CascadeK`
are three-link chains, and every value that crosses one does so through the depth-1 FIFO of
`design/08-tt-backend.md` §3.5: a direct remote L1 write, a barrier, and a pair of counting
semaphores per link. The wavefront is **emergent** — nothing in the emitted kernel expresses the
skew; each PE blocks on its own `noc_semaphore_wait_min` and the diagonal order follows.

**T3's subject is the first link graph that is not a path.** W2's two PEs each put on their
outbound link before they get on their inbound one, so unlike a chain the *wait-for* graph has a
cycle. Two things came out of it, both rulings and both measured: **R-TT-B**, pairing a link's
put and get **occurrences** positionally rather than pairing channel ends — which is what lets one
link land its payload in `cur` on one occurrence and in `next` on the next, T2's blocker — and
R-TT-B's **credit**, which is what keeps the pair from deadlocking. W2 is also the only workload
whose arithmetic is not integer-valued, so it is the one that answers **Q-TT4**: soft-float scalar
`f32` on a Tensix data-movement core reproduces numpy **bit for bit**.

The claim this supports is therefore: one plan, two unrelated backends, the same arithmetic — on
**all four** workloads, two of which (W1-flip, and every structural mechanism W2 shares with W3)
needed no workload-specific code to reach.

No Tenstorrent hardware was touched. `ttsim` is a **functional** simulator of a Wormhole B0
chip; it is not a timing model and its wall-clock numbers say nothing about a real part.

---

## 1. What the gate (G-TT0) established

From `/home/adi/Projects/Honours/tt-probe/` — `REPRO.sh`, `t3_generic_op.py`, `t6_multicore.py`,
`t6_neg.py`. Reused here verbatim rather than rediscovered:

1. A `ttnn` **wheel** plus a `ttsim` shared object is a complete device-free stack: no tt-metal
   source tree, no `TT_METAL_HOME`, no hardware.
2. `TT_METAL_SIMULATOR` names the `.so`; **`soc_descriptor.yaml` must sit beside it under
   exactly that name** (the gate copied `wormhole_b0_80_arch.yaml` out of the wheel).
3. `TT_METAL_SLOW_DISPATCH_MODE=1` — there is no fast-dispatch firmware for the simulator.
4. **`TT_METAL_HOME` must be unset.** The wheel is self-rooted; a stale value sends `tt_metal`
   looking for a source tree that does not exist.
5. **sfpi is not in the wheel.** The device C++ toolchain must be unpacked at exactly
   `<site-packages>/ttnn/runtime/sfpi`.
6. `ttnn.generic_op` takes a `ProgramDescriptor` whose `KernelDescriptor` carries the kernel
   **C++ as an inline string** (`SourceType.SOURCE_CODE`) — which is what an emitter produces —
   with per-core `RuntimeArgs` over a `CoreRangeSet`, proven over an 8×8 grid.

## 2. The environment

`scripts/tt_env.sh`. Sourcing it builds `.venv-tt` on first use from `vendor/tt/`, then exports
the three facts above. A second venv is not a convenience: the `ttnn` wheel pins `numpy<2` and
the project pins `numpy==2.5.3`, so they cannot share `.venv`.

```bash
source scripts/tt_env.sh
.venv-tt/bin/python -m pytest -m requires_ttsim tests/tt
```

`vendor/tt/` follows `vendor/wheels/`'s R-13 discipline: the artefacts are git-ignored,
`vendor/tt/SHA256SUMS` is committed.

| Artefact | Version | sha256 |
|---|---|---|
| `ttnn-0.78.0-cp312-cp312-manylinux_2_34_x86_64.whl` (90.3 MB) | 0.78.0 | `0617c1167b1206d256f6045c0b3e36c4bdc8c38a6e36c0f5ac903251b2e59b1c` |
| `libttsim_wh.so` (219 KB) | ttsim 1.10.7 | `d2b7cb4ffcefe06cffe218bc32de36f4ca0755966e31ce9e69c4c9df73528197` |
| `soc_descriptor.yaml` (3.1 KB) | `wormhole_b0_80_arch.yaml`, out of the wheel | `24fd3dfae80435a7d9113e255d6d6af9cdf83f6ae3b1c385e1d48a24795ccf49` |
| `sfpi_7.75.1_x86_64_debian.txz` (79.6 MB) | sfpi 7.75.1 | `7ec141c063eaf1290359ecce23dff3133574371f64a24cc0cce1619b8756ba28` |

The sfpi hash is the one **upstream publishes** for `x86_64_debian` at release 7.75.1
(`sfpi_x86_64_debian_txz_hash` in the release's `sfpi-version.txt`) — an independent check, not
a hash of whatever happened to download.

`.venv-tt`: 26 packages, **965 MB**, built in **8 s** (the wheel comes from `vendor/tt/` and the
transitive dependencies from a warm pip cache; a cold network install is a few minutes).
`numpy 1.26.4` there against `numpy 2.5.3` in `.venv`.

## 3. The mapping (AIR construct → TT-Metalium construct)

The module docstring of `spatial/m5tt_emit.py` carries this table; it is the contract.

| Plan construct | TT-Metalium |
|---|---|
| `MappingPlan.tensors` (L3) | one DRAM `ROW_MAJOR` tensor each, in `plan.tensors` order, addressed through `TensorAccessorArgs` compile-time args. **One page is one row** (`shape[-1] * dtype.sizeof`) — measured, and re-checked against `Tensor.buffer_page_size()` at run time |
| `HerdPlan.grid` | the core range `(0,0)..(gx-1, gy-1)`; one data-movement kernel on `RISCV_0` per core carrying the whole herd body. `HerdPlan.shape` / `at` / `repeats` are AIE strip-mining and are **ignored** (W1's `(1, 2)` physical herd becomes 4 real cores) |
| `HerdPlan.coords` | runtime args, after one base-address arg per L3 tensor |
| `BufferPlan` (L1) | one circular buffer, `total_size == page_size == bytes`; the kernel takes its L1 address from `get_write_ptr(index)`. `ping_pong_candidate` is **ignored** — T1 is functional and does not double-buffer |
| an L3↔L1 `ChannelPlan` | **the herd side does the DRAM transfer itself**: a get from L3 is a row-wise `noc_async_read` of the *segment* site's `Region`, a put to L3 a row-wise `noc_async_write`, each with its barrier. The segment site's bundle index is inverted through the channel's `size`/`broadcast_shape` fan-in rule into an expression in this core's coordinates, so a broadcast needs no multicast: each destination reads independently |
| the segment loops that only wrap L3 sites | nothing. A segment loop that is not a bundle-index loop (W1's `k0`) must be matched by an enclosing herd loop of the same axis and the same `(lo, hi, step)` — **checked**, and an internal-consistency error otherwise |
| a core↔core `ChannelPlan` | `TTNotImplemented("… T2/T3/T4")` |
| `LoopPlan` (either `kind`) | a C++ `for`; `"unrolled"` is an AIE tracing distinction with no counterpart |
| `StoreNode` / `ExprNode` | scalar C++ over `volatile tt_l1_ptr float*`; `Const` from `Const.text` verbatim, `MaxMin`/`Select` as ternaries |
| `BranchNode`, `ChannelSite.guard` | a C++ `if` on the runtime-arg coordinates |

**The target string is irrelevant.** `"npu1"`/`"npu2"` picks an AIE generation for `air.api`'s
`build()`; `m5tt_emit.emit` takes no target, and
`test_target_does_not_reach_the_tt_program` asserts the two produce the identical `TTProgram`.

**D-14 holds for this emitter exactly as for M5.** It reads no field of `plan.mapping` — no
`LegalMapping`, `ScheduleModel` or `KernelModel`, no `physical_herd`, no `repeats` — and
`tests/tt/test_m5tt_emit.py::test_emitter_makes_no_decisions` lints the code (docstrings
stripped) for those names. The runner decides nothing either: the runtime-argument ABI is the
emitter's (`("addr", <tensor>)` / `("const", <int>)`) and `m6tt_run` only substitutes.

Two things the emitter refuses rather than guesses, because both fail silently on a NoC:

* **Alignment.** *(T1's rule; **superseded at T2** — see §T2 and ruling R-TT-A′ in
  `design/08-tt-backend.md` §3.3.)* T1 required every DRAM offset, L1 offset and transfer length
  to be a multiple of 32 B. That is stricter than the device: what the NoC enforces is a
  **relative congruence** between the two ends of a transfer. The arithmetic-progression proof
  survives unchanged; only what it proves changed.
* **Names.** Every buffer, tensor, coordinate and axis must be a C++ identifier that does not
  start with `_`, is not a keyword, and does not collide across the four sets. *(T2 adds channel
  names to that set: a core↔core channel's name becomes C++ identifiers.)*

### 3.1 Reconciliation with `design/08-tt-backend.md` (done at T2)

The spec governs. One line per difference between the table above (written at T1) and the spec:

| # | Difference | Resolution |
|---|---|---|
| 1 | T1's runtime-arg vector is **addresses, then coordinates**; the spec §3.4 lists blocks **A** (coordinates), **B** (neighbour NoC coordinates), **C** (DRAM base addresses) in that order | T1's third block *is* the spec's block C — it was never an extra invention. The **order** differs, and the spec was amended to C, A, B rather than renumbering a verified T1 artifact for an ordering that carries no meaning; T2 appends block B. Flagged for the architect |
| 2 | T1's table says "a core↔core `ChannelPlan` → `TTNotImplemented`" | T2 implements it, exactly as spec §3.5: depth-1 FIFO, `full`/`empty` counting semaphores, `wait_min`, remote `noc_async_write` + barrier + `noc_semaphore_inc` |
| 3 | T1's table says a segment loop that is not a bundle-index loop "must be matched by an enclosing herd loop of the same axis and the same `(lo, hi, step)`" | Still the first rule. T2 adds the fallback the spec now carries as §3.3 rule 5: no twin ⇒ bind the axis to the site's own occurrence counter, and check the occurrence count against the segment trip count |
| 3a | T1/T2 pair a core↔core channel's two ends by **site**: every get of a channel must name the same buffer and region | Superseded at T3 by ruling **R-TT-B** (spec §3.5): pair by **occurrence**, the k-th put with the k-th get, and take the destination from the paired get. W3's and the cascade's emitted kernels are byte-identical either way, because their gets *do* all land in one region |
| 3b | T1/T2 fix the FIFO at **depth 1** — the producer's put `n` waits `empty ≥ n` | Amended at T3 by measurement: correct for a chain, and it **deadlocks** on W2, whose two PEs each put before they get on links running in opposite directions. The credit is derived from R-TT-B's pairing — the smallest gap between two landing regions that alias — and is 1 for W3 and the cascade, 2 for W2 (§T3.3) |
| 4 | T1 rounds CB sizes to 16 B (spec §4's original wording) | T2 rounds to **32 B**: measured, CB bases are 32 B aligned and R-TT-A′'s L1 residues depend on it |
| 5 | T1's alignment rule (absolute, 32 B) | Superseded by R-TT-A′ (relative congruence), measured. The T1 rule is strictly stronger, so every W1 transfer still passes and W1's emitted kernel is byte-identical |

## 4. T1 result

Commands are `source scripts/tt_env.sh` first, from the repo root.

| Measurement | Result |
|---|---|
| `.venv-tt/bin/python -m pytest -m requires_ttsim tests/tt` | **4 passed, 19 deselected in 104.44 s** (two simulator runs; 99.45 s on an earlier, identical-output build) |
| W1 exact (`np.array_equal(C, A @ B)`) | **yes** — inputs are `test_semantics.inputs()`: `rng(0)`, integer-valued `f32` in `[-8, 8)`, so `==` is the right comparison |
| equality with `tests/helpers/plan_interp.py` on the same inputs | **yes**, element for element |
| inputs `A`, `B` read back unchanged | yes |
| one `run()`, **cold** JIT (`~/.cache/tt-metal-cache` moved away; `JIT cache stats: 0/9 hits`) | **50.9 s** wall |
| one `run()`, **warm** JIT (`JIT cache stats: 10/10 hits`) | **49.9 s** wall |
| ttsim's own report for one `run()` | `[11509553] 50.0 seconds (230.0 KHz)` — **identical cycle count on every run**, cold or warm |
| device open | ≈ 2.0 s of the above |

The JIT build is ≈ 1 s of a ≈ 50 s run: the simulator's execution of 4 cores × (32×32 zeroing +
4 × 32×32×16 scalar FMAs + 4 × 48 row DMAs + 32 row DMAs) dominates, at ~230 kHz simulated.

**The negative control.** `tests/tt/test_tt_w1.py::test_mutating_the_kernel_changes_the_answer`
replaces the compute node's `+` with `-` in the emitted string — `acc[…] = (acc[…] + (a[…] *
b[…]))` becomes `… - (a[…] * b[…])` — and asserts the device returns neither `A @ B` nor the
clean result. It does not: **4086 of 4096 elements differ**, `C[0, 0]` is `-185.0` against
`185.0` (the mutation computes `-(A @ B)`, so the ten zero entries coincide), and ttsim reports
a *different* cycle count for the run, `[11736922]` against the clean `[11509553]`. Without this
the exactness test would also pass against a simulator that never ran our kernel and left `C`
holding something that happened to be right.

**The rest of the suite is untouched.** `.venv/bin/python -m pytest --ignore=tests/tt` is
**611 passed, 1 skipped, 12 deselected**, the same as before this branch; with `tests/tt` it is
**630 passed, 1 skipped, 16 deselected** (the 19 new emitter unit tests join the default run
because `spatial/m5tt_emit.py` imports nothing but the standard library and `spatial.model`).

## T2 — W3 on ttsim: the first core↔core protocol

*2026-09-13. Every number below was run; each probe script is named where it produced one.*

### T2.1 W3, the wavefront

`tests/tt/test_tt_w3.py`, seven tests, all `requires_ttsim`.

| Measurement | Result |
|---|---|
| **W3 exact** — `np.array_equal(S, textbook two-loop DP)` | **yes.** Inputs `test_semantics.w3_inputs()`: `rng(0)`, `q, r ∈ [0, 4)`, `i32`, `S` zeroed. Integers throughout, so `==` is the right comparison and a tolerance would hide a real difference |
| equality with `tests/helpers/plan_interp.py` on the same inputs | **yes**, element for element |
| boundary and inputs | row 0 and column 0 stay zero; `q` and `r` read back unchanged; `S[1:, 1:]` is non-zero, so the test is not vacuous |
| wall time, one `run()` | **2.2 s** (`(4,)` cores, 6 semaphores). The whole module, four simulator sessions, is ≈ 9 s |
| ttsim's own report, one `run()` in its own process | **`[24426] 2.1 seconds (11.6 KHz)`** — 24 426 simulated cycles, against W1's 11 509 553. W3 is 4 PEs × 32 rows × 8 columns of scalar `i32`; W1 is 65 536 scalar FMAs |
| `slow` marker | **not needed** — 2.2 s is far under the suite budget (Q-TT6's concern was W2) |
| UndefinedBehavior lines | **none** (and UB is fatal, §T2.4 — a silent one is not possible) |
| semaphores | **6**, ids 0..5: `West.full[k]` = `2k`, `West.empty[k]` = `2k+1` for the three links of `West`. `WestIn` and `EastOut` have an L3 end and carry none |

**The negative control.** `test_mutating_the_match_score_changes_the_answer` rewrites the one
`Select` — `q[i-1] == r[j-1] ? MATCH : MISMATCH` — so that MATCH scores `-1` like MISMATCH, at
all **eight** places the `MaxMin` fold spells it, and asserts the device returns neither the
textbook DP nor the clean result. It does not. Without this the exactness test would also pass
against a simulator that never ran our kernel and left `S` holding the zeros it was uploaded
with.

**The first failure was real and is worth recording**: the first W3 run returned a score matrix
whose row 1 held the *correct row 32* and whose other rows were zero. The occurrence counter of
§3.3 rule 5 was being read but never incremented on the L3 path, so all 32 `SOut` drains landed
on page 1. Every row had in fact been computed correctly — the semaphore chain and the
`prev`/`cur` swap were already right — which is why the symptom was a pure addressing one.

### T2.2 W1-flip, the cascade — gate T4, reached without an emitter change

`tests/tt/test_tt_flip.py`, four tests. **Not in T2's brief**; it is here because nothing in
`spatial/m5tt_emit.py` mentions a workload, so the same code path carried it.

| Measurement | Result |
|---|---|
| **exact** — `np.array_equal(C, A @ B)` on `test_semantics.inputs()` | **yes** |
| equality with `plan_interp.run` | **yes** |
| negative control — drop the cascade accumulate (`acc += recv` → `acc = acc`) | the product is wrong, as designed |
| wall time, one `run()` | **54.9 s**; ttsim reports **`[12129177]`** cycles |

What it adds over W3: the payload is a **32 × 64 partial tile** (8 192 B) moved as thirty-two
256-byte remote writes inside one generated loop nest under a single barrier and a single `full`
increment — the multi-row shape of §3.5 that W3's four-byte score column never reaches. By §7's
rule (exact, negative control fails as designed, equal to the interpreter, no UB) **gate T4 is
green**.

### T2.3 The open questions T2 owned

| # | Answer | How it was measured |
|---|---|---|
| **Q-TT1 / A-TT1** | **Holds.** On W3's own CB table, every core reports the same addresses: CBs at `105696, 105824, 105856, 105920, 105984, 106016` and semaphores 0..5 at `35056 … 35136`, 16 B apart. All six CB bases are ≡ 0 mod 32, which is what R-TT-A′'s L1 residues rest on. No fourth runtime-arg block is needed | `test_A_TT1_every_core_sees_the_same_addresses`: a probe kernel replaces W3's kernel and L3 tensors, keeps its CBs and semaphores, and writes `get_write_ptr`/`get_semaphore` into a debug DRAM tensor, one row per core |
| **Q-TT2 / C-TT3** | **The design's choice works, and this part cannot discriminate.** `worker_core_from_logical_core(CoreCoord(x, 0))` returns `(18,18) (19,18) (20,18) (21,18)` — the **virtual** space; the soc descriptor's physical `functional_workers` row is `1-1 2-1 3-1 4-1`. A four-core chain delivers correctly with **either**. The emitter passes the virtual coordinates, which are the ones that stay correct under harvesting | `scripts/tt_probe_coords.py`: run A virtual, run B physical, run C virtual **+1** (the negative control) — C mis-delivers, core 1's semaphore never fires |
| **Q-TT3** | **ttsim does not notice the reversed order.** With `noc_semaphore_inc` moved *before* `noc_async_write_barrier()` at both `West` put sites, the answer is **unchanged** and **no `UndefinedBehavior` is reported**. So the simulator is not an oracle for this class of bug: it does not prove the barrier unnecessary, only that this model does not model the hazard. **The spec's order is what the emitter emits**; the reversal exists only inside one test | `test_Q_TT3_reversing_the_barrier_and_the_increment` — a fourth simulator run, then the emitted order is restored (it was never changed: the test mutates a copy) |
| **Q-TT5** | **16 semaphores per core**, ids 0..15. 8 accepted, 32 refused: `TT_FATAL @ tt_metal/impl/program/program.cpp:2001: semaphore_id < NUM_SEMAPHORES — Semaphore id 16 exceeds max value 15`. Now `m5tt_emit.TT_SEM_LIMIT`, checked before a device is opened; W3 and the flip use 6 | a throwaway program built on the coordinate probe, over-allocating deliberately: `ProgramDescriptor(semaphores=[SemaphoreDescriptor(id=i, …) for i in range(n)])` for `n = 8, 32` |
| **Q-TT7** | **The question's premise was wrong** — see §T2.4. 4-byte DRAM transfers are legal; what is illegal is a *mismatched* one | `scripts/tt_probe_align.py` |

### T2.4 What the NoC actually requires — ruling R-TT-A′

23 single-transfer cases, **one simulator process each** (the sweep had to be split: ttsim's
`UndefinedBehavior` is **fatal** — it prints, the host exits 1, and the remaining launches in
that session never run). Reproduce with `scripts/tt_probe_align.py`:

```bash
source scripts/tt_env.sh
for i in $(seq 0 22); do .venv-tt/bin/python scripts/tt_probe_align.py $i; done
```
 Source and destination byte offsets are relative to a 256 B L1 CB and a
256 B DRAM page.

| kind | src | dst | size | verdict |
|---|---|---|---|---|
| L1 → L1 write | 0 | 0 | 32 | ok |
| L1 → L1 write | 4 | 4 | 4 | ok |
| L1 → L1 write | 4 | 0 | 4 | **UB** |
| L1 → L1 write | 16 | 0 | 16 | ok |
| L1 → L1 write | 16 | 32 | 16 | ok |
| L1 → L1 write | 8 | 8 | 8 | ok |
| L1 → DRAM write | 0 | 0 | 32 | ok |
| L1 → DRAM write | 4 | 4 | 4 | ok |
| L1 → DRAM write | 4 | 0 | 4 | **UB** |
| L1 → DRAM write | 4 | 36 | 32 | ok |
| L1 → DRAM write | 16 | 0 | 16 | ok |
| L1 → DRAM write | 16 | 32 | 16 | ok |
| L1 → DRAM write | 0 | 16 | 16 | ok |
| L1 → DRAM write | 0 | 156 | 4 | **UB** |
| L1 → DRAM write | 28 | 156 | 4 | ok |
| L1 → DRAM write | 0 | 144 | 16 | ok |
| DRAM → L1 read | 0 | 0 | 32 | ok |
| DRAM → L1 read | 0 | 0 | 4 | ok |
| DRAM → L1 read | 28 | 28 | 4 | ok |
| DRAM → L1 read | 28 | 0 | 4 | **UB** |
| DRAM → L1 read | 32 | 0 | 32 | ok |
| DRAM → L1 read | 16 | 0 | 16 | **UB** |
| DRAM → L1 read | 128 | 0 | 32 | ok |

Exactly the mismatched pairs fail, and the moduli are 16 for a write and 32 for a read — the row
`read 16 → 0, 16 B` is what separates them. ttsim's diagnostic names it:

```text
[4788] ERROR: UndefinedBehavior: noc_cmd_ctrl: write:
       alignment of src_addr=0x19ce4 and dst_addr=0x19de0 does not match
```

and the read form is the one that separates the two moduli — case 21, `src 16 → dst 0, 16 B`,
which is congruent mod 16 and not mod 32:

```text
[4056] ERROR: UndefinedBehavior: noc_cmd_ctrl: read:
       alignment of src_addr=0x100050 and dst_addr=0x19ce0 does not match
```

`strings vendor/tt/libttsim_wh.so` carries both forms and a `multicast:` one beside them.

**Consequence for the architect's ruling R-TT-A.** Its policy stands — DRAM layout is a host-side
policy of `m6tt_run`, derived mechanically from the plan's regions, with per-tensor `pad_elems`
and `row_stride_bytes`, padding on upload and stripping on readback, and `TT-ALIGNMENT` as the
refusal. Its *derivation* does not: it assumed each transfer's own start and size were what the
NoC checked, and on that premise W3's `S` needed `p = 7`, an L1 scratch buffer for the `WestIn`
read, a read-modify-write for the `EastOut` drain, and a static single-writer check to make the
RMW safe. With the measured rule, **`p = 0` for every tensor of all four workloads**, every W3
transfer is direct, and **none of that machinery exists**:

| site | L1 byte offset | DRAM byte offset | mod | congruent |
|---|---|---|---|---|
| `WestIn` get `S[i, 0:1]` | `edge_in + 0` → 0 | `160·i + 0` → 0 | 32 | yes |
| `SOut` put `cur[1:9]` | `cur + 4` → 4 | `160·i + 32·tx + 4` → 4 | 16 | yes |
| `EastOut` put `S[i, 32:33]` | `edge_out + 0` → 0 | `160·i + 128` → 0 | 16 | yes |

`p = 7` would put `S[i, 0]` at byte 28 against an L1 offset of 0 — `28 ≢ 0 (mod 32)` — and is
refused by the emitter. The row is still padded, 33 `i32` = 132 B → **160 B**, so that a page
index can never move a residue; with `p = 0` that padding is entirely *trailing*.

`_leading_pad` is the whole of the policy, five lines: the smallest pad in `[0, 32/width)` that
satisfies every `(modulus, DRAM byte − L1 byte)` the tensor's transfers ask of it, `None` — i.e.
`TT-ALIGNMENT` — otherwise. `test_a_plan_no_leading_pad_can_satisfy_raises_TT_ALIGNMENT` builds
the negative: shift W3's `SOut` drain one column left at its segment end only, so it wants
`p ≡ 1 (mod 4)` while `WestIn` still wants `p ≡ 0 (mod 8)`.

### T2.5 The emitter, and what made W3 possible

Four mechanisms, none of them W3-specific — the emitter dispatches on plan shape only, which is
why W1-flip fell out for free:

1. **Core↔core channels** (`_link`). A channel with no segment-scope site has both ends in the
   herd body. The producer waits on its own `empty`, writes straight into the consumer's get-site
   buffer, barriers, then increments the consumer's `full`. Semaphore ids are `base + 2k` and
   `base + 2k + 1` over `plan.channels` order and the flat bundle index, so the kernel computes
   them from its own coordinate: `get_semaphore((uint32_t)((tx * 2)))`.
2. **The `empty` release**, at the top of the *next* get on the link rather than at the end of
   the enclosing body — the spec's §3.5 point 5 made operational, because W3's herd body is one
   `i` loop of **step 2** holding two rows, and the literal reading deadlocks. The argument is in
   `design/08-tt-backend.md` §3.5.
3. **Occurrence counters** for twinless segment loops (§3.3 rule 5). `i_source` and `i_drain` run
   `[1, 33)` step 1 at segment scope against the herd's `i` over `[1, 33)` step 2; a channel is a
   FIFO, so the site's n-th transfer is the loop's n-th trip. The emitter declares one
   `int32_t <channel>_<kind>_n` per channel end and **checks it**: it enumerates the herd body
   per concrete core, evaluates the guards, multiplies the loop trips, and requires the count to
   equal the segment loop's trips — 32 against 32 for each of `WestIn`, `EastOut`, `SOut`.
4. **Decoupled slab ranks** (`_common`). `WestIn`'s two ends are `S[i, 0:1]` — rank 2 — and the
   whole of `edge_in` — rank 1. Leading unit dimensions pick a row rather than iterating one, so
   they are dropped from the iteration and kept as an offset; a *trailing* unit dimension is not
   dropped, because it changes what is contiguous.

Two deviations from the spec text, both flagged for the architect:

* **Runtime-arg block order is C, A, B**, not A, B, C. T1 fixed C (DRAM base addresses) first and
  T2 appends B (peer NoC coordinates); the order carries no meaning and renumbering would
  invalidate a verified T1 artifact for nothing. `design/08-tt-backend.md` §3.4 was amended to
  say so rather than the code changed. **Overrule this if you would rather have the spec's
  order** — it is one edit and one re-run.
* **Block B is per channel, not per link.** A core is the producer on at most one link of a given
  channel and the consumer on at most one, for all four workloads; the emitter checks that and
  refuses a channel where it fails, naming the two links.

### T2.6 Test counts, both suites

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest` (default) | **648 passed, 1 skipped, 27 deselected in 12.8 s** (T1: 630 passed, 1 skipped, 16 deselected) |
| `source scripts/tt_env.sh; .venv-tt/bin/python -m pytest -m requires_ttsim tests/tt` | **15 passed, 37 deselected in 206.9 s** (T1: 4 passed in 104.4 s). Six simulator sessions: W1 clean + negative, W3 clean + negative + Q-TT3 + A-TT1 probe, flip clean + negative |

W1's emitted kernel is **byte-identical** to T1's — every T1 structural assertion still passes
untouched — because the twin rule of §3.3 rule 4 still runs before the counter fallback, `p = 0`
adds no term to any offset, and W1's circular buffers were already multiples of 32 B.

## T3 — W2 on ttsim: occurrence pairing, and f32 exactness

*2026-09-13. Every number below was run; the command that produced it is named beside it.*

### T3.1 W2, the Jacobi halo — gate T3

`tests/tt/test_tt_w2.py`, ten tests. Inputs are `test_semantics.w2_inputs(T)`: plane 0 is
`rng(0)` integer-valued `f32`, and **B-P25** holds — plane 0's Dirichlet rows (`0`, `H+1`) and
columns (`0`, `W-1`) are copied into every plane, which is what makes the plan and the kernel
text compute the same thing.

| Measurement | `T = 4` | `T = 5` (the odd peel) |
|---|---|---|
| **exact** — `np.array_equal(U, two-loop numpy Jacobi)` over the **whole** tensor | **yes**, max abs error **0.0** | **yes**, max abs error **0.0** |
| equality with `tests/helpers/plan_interp.py` on the same inputs | **yes**, element for element | **yes** |
| boundary — plane 0 unwritten; rows `0`/`17` and columns `0`/`15` of every written plane carry plane 0's | **yes** | **yes** |
| non-vacuous — the interior is non-zero | yes | yes |
| ttsim's own counter, one `run()` per process, **identical on every run** | **183 914** | **229 133** |
| wall for the whole process, measured outside it (two repetitions each) | 2.9 s then 1.1 s | 1.3 s then 1.3 s |
| `slow` marker on the run | **not needed** (Q-TT6) | not needed |
| UndefinedBehavior lines | **none** (and UB is fatal, §T2.4) | none |
| semaphores | **4**, ids 0..3: `ToNorth.full[0]=0`, `ToNorth.empty[0]=1`, `ToSouth.full[0]=2`, `ToSouth.empty[0]=3`. `UIn`/`UOut` have an L3 end and carry none | 4, the same |
| circular buffers | `cur` 640 B, `next` 640 B — already multiples of 32, so the rounding of R-TT-A′ adds nothing | the same |
| `U`'s DRAM layout | `pad_elems = 0`, `page_bytes = 64` (16 `f32` is already 32-B aligned) | the same |

For scale: 183 914 cycles against W3's 24 426 and W1's 11 509 553. W2 is 2 PEs × 4 `STEP`s × 112
five-point updates plus a 160-element seed copy, so it sits nearer W3 than W1.

**The negative control.** `test_mutating_the_stencil_constant_changes_the_answer` rewrites the one
`f32` literal, `((float)(0.2))` → `((float)(0.25))`, at both the places the two `STEP` bodies
spell it, and asserts the device returns neither the oracle nor the clean result. It does not:
**884 of 896** interior elements differ, `U[1,1,1]` is `2.25` against `1.8000001`, and ttsim
reports a different cycle count for the run — **178 668** against the clean 183 914. A second,
independent perturbation was measured but is not kept as a test (one suffices): dropping the
fifth stencil term gives 875 of 896 differing and **170 793** cycles.

**No W2-specific code exists.** Everything except R-TT-B below came from mechanisms T1 and T2
already had: the rank-3 `UIn` slab `(1, 10, 16)` → the `(10, 16)` `cur` buffer is `_common`'s
leading-unit-dimension rule (T2, §T2.5 item 4, verified unchanged); the seed copy P5 added is an
ordinary `StoreNode` nest; the `UOut` drain's plane index is the twinless-segment-loop occurrence
counter of §3.3 rule 5 (`UOut_put_n`, incremented once per `STEP`, so the two `STEP`s of a `t`
trip write planes `t+1` and `t+2`); the 8 × 64 B drain rows and the 64 B halo rows are all
congruent under R-TT-A′ with `p = 0` and needed nothing.

### T3.2 Ruling R-TT-B — pairing by occurrence

T2's blocker, verbatim from §5 as it stood: `TTNotImplemented: jacobi: channel 'ToNorth' has gets
landing in both 'cur'Region(…) and 'next'Region(…); the producer writes one address and cannot
serve two`.

The architect's ruling (`design/08-tt-backend.md` §3.5) is that this is a limitation of pairing by
*channel*: the link is served by two put **occurrences** and two get **occurrences** per `t` trip,
every core runs the identical program, and so the producer's k-th put occurrence is consumed by
the consumer's k-th get occurrence. What the emitter does, in `_pair`:

1. `_survey` already enumerates the concrete grid; it now also records each link end's site
   occurrences **in plan order with their trip counts** (only one core reaches each end, which the
   existing `ends` check enforces, so the list is that core's program order).
2. Occurrence k of the producer's put list is paired with occurrence k of the consumer's get list.
   Unequal lengths, or unequal trip counts at the same position, are the internal-consistency
   error. A put site reached on two links must land the same way on both — one kernel text serves
   every core — and is refused otherwise.
3. `_link_end` no longer scans `channel.sites` for a single landing region; it reads the paired
   get. W2's first put of a trip therefore writes `cur_l1 + 9·16·4` and its second
   `next_l1 + 9·16·4`, out of one kernel text.

At `T = 5` the peeled `STEP` pairs positionally after the loop's occurrences: puts
`[put.0, put.6, put.5]` against gets `[get.2, get.8, get.7]`, landing `cur, next, cur`.

**W1, W3 and W1-flip are unaffected: their emitted kernels are byte-identical to T2's**, checked
by diff before and after the change. W1 has no core↔core channel; W3's two `West` get occurrences
both land in the whole of `edge_in` and the cascade's single one in the whole of `recv`, so the
pairing returns what `_link_end` returned before.

### T3.3 The credit — and the deadlock that is not in any of the other three workloads

The ruling left the depth-1 FIFO in place. **It deadlocks on W2**, and the reason is structural:
W3 and the cascade are chains, so the wait-for graph is a path; W2's two PEs each put on their
outbound link *before* they get on their inbound one, so with one credit PE0's second put blocks
on a slot only PE1's later get can free while PE1's second put blocks on one only PE0's later get
can free. P1′ and P2b do **not** exclude this: they constrain the *channel* graph, which is
acyclic here, and the cycle is in the wait-for graph the protocol builds on top of it.

Measured, one simulator process, the clean W2 program with the two credit expressions replaced by
the bare counters:

```text
$ source scripts/tt_env.sh
$ timeout 180 .venv-tt/bin/python <the one-credit variant>
exit=124 after 180s          # killed; the clean run of the same program takes ~1.3 s
```

No output past `Dispatch telemetry SMC buffer unavailable`, no `UndefinedBehavior`, no progress.
The test that keeps this is `tests/tt/test_tt_w2.py::test_one_credit_per_link_deadlocks`, marked
`slow`, which runs the variant in its own process with a **60 s** cap — 45× the clean run and
enough to be conclusive without spending three minutes.

**The credit, as implemented** (`_credit_of`): enumerate the link's get occurrences **in dynamic
order**, loops expanded, and take the smallest `d ≥ 1` for which two landing regions `d` apart
overlap. The producer's put `n` then waits `empty ≥ max(0, n − credit + 1)`. Two details are
load-bearing:

* **dynamic order, not the static occurrence list.** At `T = 5` the real sequence is
  `cur next cur next cur` while the static list spells `cur cur next next cur`; the static list
  would compute a credit of 1 and hang. The expansion prunes every subtree with no site of that
  channel (so W1-flip's 32 × 64 × 16 compute nest is never walked) and refuses a link carrying
  more than 100 000 payloads rather than expanding it.
* **disjointness, not "a different buffer".** Different buffers are different circular buffers and
  never alias; two regions of the same buffer are compared interval by interval, and an offset the
  emitter cannot fold is assumed to overlap — which costs a credit and is never wrong.

Measured consequence: **W3 and W1-flip get credit 1** and keep T2's exact text; **W2 gets 2**. The
emitted wait is the unchanged `…, West_put_n);` at credit 1 and
`…, (ToNorth_put_n < 1 ? 0 : ToNorth_put_n - 1));` at credit 2.

### T3.4 Q-TT4 — soft-float `f32` is bit-exact, and what makes it so

**Closed: yes, bit for bit.** `np.array_equal` over the whole tensor at both parities, max abs
error 0.0, against the two-loop numpy oracle *and* against `plan_interp`. The tolerance
`test_semantics.TOL = 1e-5` was not needed and was not widened.

Two conditions carry it, and the second was measured rather than assumed:

| Variant of the one `f32` literal | Result | ttsim cycles |
|---|---|---|
| `((float)(0.2))` — what the emitter emits | **exact**, max abs err 0.0 | 183 914 |
| `0.2f` | **exact**, byte-for-byte the same answer | **183 914** (identical) |
| `0.2` — unsuffixed, i.e. a C++ `double` | **not exact**: 358 of 896 interior elements one ulp off, max abs err **4.77e-07**, `U[1,1,1] = 1.8` against `1.8000001` | **253 102** |

The third row is the evidence that the narrowing is load-bearing rather than decorative: an
unsuffixed `0.2` is a `double`, `0.2 * <float>` is then evaluated in `double` and narrowed only on
the store, which is not what `numpy.float32(0.2) * <float32>` computes — and the simulator's own
cycle counter rises by 38 %, which is the soft-float `double` path showing up as work.

The `f` suffix and the cast measure **identical**, so the suffix is a spelling and not a fix. The
cast is kept: it is also correct for a `Const.text` with no decimal point, where `5f` would not
compile. `tests/tt/test_m5tt_emit.py::test_f32_constants_are_narrowed_before_they_are_used` holds
all four workloads to it and asserts no `double` token appears in any emitted kernel.

The other condition is §3.7's exact parenthesisation, which was already the rule: W2's five-term
sum is emitted left-nested, `((((a+b)+c)+d)+e)`, exactly as the `ExprNode` tree spells it and as
Python's parser built it.

### T3.5 Two defects the device found, neither of them in the emitter

1. **`tests/helpers/plan_interp.py` evaluated `Const` in Python's dtype, not the plan's.** It
   returned `node.value`, a Python float. Whether `0.2 * <np.float32>` is then computed in float32
   or in float64 is a **numpy version** question: NEP 50 (numpy ≥ 2) keeps the scalar weak and
   gives float32; numpy 1.x's value-based casting promotes to float64 and narrows on the store.
   `.venv` has numpy 2.5.3 and `.venv-tt` has 1.26.4, so the interpreter agreed with the oracle in
   the default suite and disagreed with it — by exactly the one-ulp pattern of §T3.4's third row,
   358 of 1440 elements — under the TT venv, which is where
   `test_W2_matches_the_plan_interpreter` runs. Fixed by returning `node.dtype.numpy(node.value)`:
   the plan carries the dtype, so use it. Both venvs now agree at 0.0, and the default suite is
   unchanged (659 passed). **Only W2 could have found this**: the other three workloads are `i32`
   or integer-valued `f32`, where the promotion is invisible.
2. **`ttnn.generic_op` refuses a single io tensor.** W2's plan declares one L3 tensor, `U`, read
   and written in place; the host asserts
   `TT_FATAL @ generic_op_device_operation.cpp:135: io_tensors.size() >= 2` — *"io_tensors must
   contain at least one input tensor and one output tensor, got 1 tensors"* — because it expects
   the pre-allocated output last. `m6tt_run.run` passes the same handle twice when there is only
   one. A host-API shape, not a fact about the plan: `program.io_tensors`, the
   `TensorAccessorArgs` block and the kernel's addressing are all unchanged.

### T3.6 Test counts, both suites

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest` (default) | **659 passed, 1 skipped, 37 deselected in 12.9 s** (T2: 648 passed, 1 skipped, 27 deselected) |
| `source scripts/tt_env.sh; .venv-tt/bin/python -m pytest -m requires_ttsim tests/tt` | **25 selected, 25 passed**, exit status 0, in **270 s** and **275 s** on two runs (T2: 15 in 206.9 s). Nine simulator sessions: W1 clean + negative, W3 clean + negative + Q-TT3 + A-TT1 probe, flip clean + negative, W2 `T = 4` + `T = 5` + negative, and the one-credit variant that is *meant* to hang |

*A recording note, so the count above is not taken on trust: **pytest's own final summary line is
lost** in this suite. `ttsim`'s exit path ends the process without flushing Python's stdout, so
when stdout is a file or a pipe the `N passed in Ns` line never reaches it — reproduced with
`PYTHONUNBUFFERED=1`, which does not help because the loss is in the C++ exit and not in the
buffer policy. What is recorded instead is the progress line — **25 dots, no `F` or `E`** — the
exit status (**0**, and pytest exits non-zero on any failure: the run where
`test_W2_matches_the_plan_interpreter` failed did print its `FAILED …` section and exited 1) and
`--collect-only -q`'s per-file counts, `flip 4 + w1 4 + w2 10 + w3 7 = 25`. ttsim's own total for
the suite is **47 577 057** cycles, identical on both runs.*

Of the ≈270 s, **W1 and the cascade are ≈205 s** (scalar C++ GEMM: 11.5 M and 12.1 M simulated
cycles) and the deadlock demonstration is 60 s; all four W2 runs together are under 6 s.
`-m requires_ttsim` overrides `addopts`, so the `slow` deadlock test **is** selected by the
command above; `-m 'requires_ttsim and not slow'` is the 210 s form.

---

## 5. Not run / not done

| Item | Why | The exact error |
|---|---|---|
| ~~**W2** (Jacobi) on ttsim~~ | **done at T3** — exact at `T = 4` and `T = 5`, §T3.1. T2's refusal (`channel 'ToNorth' has gets landing in both 'cur'Region(…) and 'next'Region(…); the producer writes one address and cannot serve two`) is answered by ruling **R-TT-B**, occurrence pairing; the deadlock it was hiding is answered by R-TT-B's credit | — |
| ~~**W1-flip** on ttsim~~ | **done at T2** — exact, §T2.2. T1's refusal (`channel 'CascadeK' has 0 segment-scope get(s) …; a core-to-core channel is T2/T3/T4`) no longer fires | — |
| ~~the rank-3 L3 slab of W2's `UIn`~~ | **fixed at T2** by `_common`: T1's `site 'UIn.put.0@segment' moves a (1, 10, 16) slab … into the (10, 16) buffer 'cur'` is gone; W2 now reaches the ping-pong problem above | — |
| ~~**W3** (Smith-Waterman) on ttsim~~ | **done at T2** — exact, §T2.1. T1's refusal (`the segment-scope loop over 'i_source' … is neither a bundle-index loop nor a loop the herd body runs …`) is answered by the occurrence counter of §3.3 rule 5 | — |
| real Tenstorrent silicon | no hardware. Only `TT_METAL_SIMULATOR` separates the two paths, but that is an untested claim | — |
| Tensix **compute** (matrix) engine | T1 puts everything on one data-movement RISC-V and computes in scalar C++. Nothing here uses `MATH`/`PACK`/`UNPACK` or a tile format | — |
| double buffering | `ping_pong_candidate` is read and ignored | — |
| a timing claim of any kind | `ttsim` is functional | — |
| `f16` / `bf16` plans | no scalar C++ type on a Tensix data-movement core | `TTNotImplemented: … dtype 'bf16' has no scalar C++ type …` |

## 6. What is left

**Every gate is green: T1 (W1), T2 (W3), T3 (W2) and T4 (W1-flip).** The `[not run]` list of §5
now holds **no workload**. What remains is out of scope, unreachable here, or a limit:

1. **Real Tenstorrent silicon.** No hardware was touched. Only `TT_METAL_SIMULATOR` separates the
   two paths, and that is an *untested* claim, not a measured one.
2. **The Tensix compute engine.** Everything runs on one data-movement RISC-V in scalar C++,
   which is what makes W1 and the cascade take ~50 s of simulated time against W3's 2 s and W2's
   1.3 s. A `ComputeConfigDescriptor` kernel with `mm_init`/`matmul_tiles` over `TILE_LAYOUT`
   tensors would be the real lowering — and would change the mapping table's CB row from "one
   page is the whole buffer" to a tile-paged CB with `cb_reserve_back`/`cb_push_back`. Out of
   scope: this repo does not do performance work (`CLAUDE.md`).
3. **Double buffering** (`ping_pong_candidate` is read and ignored), **`f16`/`bf16`** (no scalar
   C++ type on a data-movement core), and **any timing claim** (`ttsim` is functional).
4. **Two rulings are recorded here and need the architect's adjudication**, both from T3:
   **R-TT-B** as written into `design/08-tt-backend.md` §3.5, and its **credit**, which is the
   amendment T3's measurement forced on the ruling as briefed. Both are implemented, both are
   measured, and the one-credit form's deadlock is reproducible in 180 s (§T3.3).

A further honest note for any pitch: what this demonstrates is that **the `MappingPlan` is
backend-neutral for all four workloads** — a GEMM, a wavefront with a real core-to-core protocol,
a cascade, and a bidirectional halo exchange whose arithmetic is floating-point and bit-exact —
not that the DSL targets Tenstorrent. Tenstorrent is not an AIR target (`CLAUDE.md`'s scope rules
already say so); this is a second, separate emitter behind the same plan. And ttsim is a
functional simulator: **no Tenstorrent hardware was involved at any point**, and the one thing T2
measured about the simulator's own fidelity is a limit — it does not report a missing write
barrier before a semaphore increment (Q-TT3). T3 adds a second, more useful limit in the other
direction: a **deadlock** is a hang, not a wrong number, so the simulator does surface that class
of protocol bug unambiguously (§T3.3).
