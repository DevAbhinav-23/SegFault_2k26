# PROGRESS — the Tenstorrent backend

*State file for the second emitter. Phase **T1**, 2026-09-13, branch `tt-backend`. Nothing
pushed. The AIR side's state file is `design/PROGRESS-B.md`; nothing in the M5 LLDs changes.*

## Status

**T1 is done and measured. W1 GEMM is exact on `ttsim`, from the same `MappingPlan` the AIR
emitter consumes.** `spatial/m5tt_emit.py` turns a `MappingPlan` into a `TTProgram` — a core
range, one data-movement kernel's C++, a circular-buffer table, an io-tensor list and per-core
runtime args — and `spatial/m6tt_run.py` executes it through `ttnn.generic_op` on Tenstorrent's
functional simulator. `emit(m4.plan(w1_legal.legal(...)))` → `run(...)` gives `C == A @ B` under
`np.array_equal`, equal element for element to `tests/helpers/plan_interp.py`'s numpy
interpretation of the same plan, and a one-character mutation of the emitted kernel changes the
answer. **The claim this supports is narrow and is only about W1**: one plan, two unrelated
backends, the same arithmetic. W1-flip, W2 and W3 are refused by name (§5).

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

## 5. Not run / not done

| Item | Why | The exact error |
|---|---|---|
| **W1-flip** on ttsim | its `CascadeK` is a core↔core channel | `TTNotImplemented: gemm: channel 'CascadeK' has 0 segment-scope get(s) against the herd-scope put 'CascadeK.put.4@herd'; a core-to-core channel is T2/T3/T4` |
| **W2** (Jacobi) on ttsim | the L3 slab is rank 3 against a rank-2 L1 buffer, and `ToNorth`/`ToSouth` are core↔core | `TTNotImplemented: jacobi: site 'UIn.put.0@segment' moves a (1, 10, 16) slab of 'U' into the (10, 16) buffer 'cur'; T1 transfers a slab row for row and does not reshape one (T2/T3/T4)` |
| **W3** (Smith-Waterman) on ttsim | its `i_source` segment loop has no herd twin, and `West` is core↔core | `TTNotImplemented: sw: the segment-scope loop over 'i_source' around site 'WestIn.put.0@segment' is neither a bundle-index loop nor a loop the herd body runs around site 'WestIn.get.0@herd' …` |
| real Tenstorrent silicon | no hardware. Only `TT_METAL_SIMULATOR` separates the two paths, but that is an untested claim | — |
| Tensix **compute** (matrix) engine | T1 puts everything on one data-movement RISC-V and computes in scalar C++. Nothing here uses `MATH`/`PACK`/`UNPACK` or a tile format | — |
| double buffering | `ping_pong_candidate` is read and ignored | — |
| a timing claim of any kind | `ttsim` is functional | — |
| `f16` / `bf16` plans | no scalar C++ type on a Tensix data-movement core | `TTNotImplemented: … dtype 'bf16' has no scalar C++ type …` |

## 6. What T2–T4 need

In dependency order, each already named by an error T1 raises:

1. **T2 — core↔core channels.** `ToNorth`, `West`, `CascadeK`. A `ChannelPlan` whose sites are
   both herd-scope is a neighbour exchange: on TT that is an L1→L1 `noc_async_write` to the
   neighbour's address plus a `SemaphoreDescriptor` pair for the handshake (`ProgramDescriptor`
   already takes `semaphores=[]`; the gate never exercised them). The plan gives the direction:
   `ChannelPlan.chain_direction` for a cascade, and the two sites' `indices` for a halo.
2. **T3 — a segment-side program.** W3's `i_source` loop and W2's reshaping slab both need
   something to run at segment scope. On TT the honest shape is a second kernel on a reader core
   (or the same kernel guarded by coordinate), not a host loop.
3. **T4 — the compute engine.** Scalar C++ on `RISCV_0` is what makes W1 take 50 s of simulated
   time. A `ComputeConfigDescriptor` kernel with `mm_init`/`matmul_tiles` over `TILE_LAYOUT`
   tensors would be the real lowering — and would change the mapping table's CB row from "one
   page is the whole buffer" to a tile-paged CB with `cb_reserve_back`/`cb_push_back`.

A further honest note for any pitch: what T1 demonstrates is that **the `MappingPlan` is
backend-neutral for W1**, not that the DSL targets Tenstorrent. Tenstorrent is not an AIR target
(`CLAUDE.md`'s scope rules already say so); this is a second, separate emitter behind the same
plan.
