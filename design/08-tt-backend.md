# LLD — TT backend: a second emitter, `MappingPlan` → TT-Metalium, executed on ttsim

*Phase 2 (stretch), 2026-09-13. Owner: **Person B**. Reads on top of
[`06-interfaces.md`](06-interfaces.md) §5 (the plan contract) and
[`03-lld-M5-emitter.md`](03-lld-M5-emitter.md) §1–§4, whose structure this document mirrors:
a translation table, one row per plan element, and an emitter that makes no decisions.
Protocol shapes come from [`03-lld-M4-mapping.md`](03-lld-M4-mapping.md) §3.6–§3.8 and are
**not** re-derived here. **No implementation exists.***

**Two citation roots.** Repo paths are relative. Out-of-repo paths use two shorthands, expanded
once:

```text
$TT     = /home/adi/Projects/Honours/tt-probe
$WHEEL  = $TT/venv/lib/python3.12/site-packages/ttnn
```

Everything cited `$WHEEL/…` is inside the pinned `ttnn==0.78.0` wheel and was read on
2026-09-13. Everything cited `$TT/…` is a gate-G-TT0 probe artifact from the same day.

---

## 1. Purpose and scope

M5 turns a `MappingPlan` into `air.api` text. **Nothing executes it.** That is decision D-9
(`00-README.md` §5): `air.api` has no interpreter, `air-runner` is a timing model, and the CPU
backend JITs the *lowered* module, so off-device the AIE path establishes semantics
**structurally** — three static checks plus `tests/helpers/plan_interp.py`, which executes the
plan over numpy rather than the emitted IR.

This document specifies a **second emitter** from the same, unchanged `MappingPlan` to
**TT-Metalium**, run on Tenstorrent's functional simulator **ttsim**. It buys two things and
only two:

1. **The plan is backend-neutral, demonstrated rather than asserted.** One plan, two emitters,
   two unrelated device models — an AIE herd over `air.channel`, and a Tensix core grid over raw
   NoC writes and counting semaphores. Nothing in `MappingPlan` changes; `06-interfaces.md` stays
   frozen at `CONTRACT_VERSION = 5`.
2. **Executed semantics, closing D-9's gap on one target.** The TT side *runs*, on a simulator
   whose project claims "bit-exact numerical results relative to silicon"
   (`github.com/tenstorrent/ttsim`, Apache-2.0, v1.10.7 of 2026-09-12; architect-verified
   2026-09-13). The claim is Tenstorrent's, not ours — §8.

**Non-goals**, each of which is a thing a reader might otherwise assume:

| Not doing | Why |
|---|---|
| performance, cycle counts, roofline, any comparison to the AIE path | ttsim is functional; the repo's scope rule forbids optimisation work (`CLAUDE.md`, *DSL + lowering only*) |
| the Tensix matrix/vector engines (Unpack/Math/Pack) | compute is **scalar C++ on one data-movement RISC-V**; using the FPU would mean tile layouts and LLK intrinsics, which is a second project |
| tile layouts, `TILE_LAYOUT`, sharded tensors | the plan's `Region` is a row-major slab (`tests/helpers/plan_interp.py:54-62`); row-major is the only layout that maps onto it without an M4 change |
| multi-chip, fabric, Blackhole, silicon | one Wormhole device, `ttsim` only |
| replacing the AIE path | AIE stays primary; this is a stretch item and the designated cut |

**Ownership and blast radius.** Owner **B**. Touches no file owned by A or C: two new modules
under `spatial/`, one new test directory, one new script, one new state file, plus the two
one-line edits recorded in §9 of this document's commit. Branch `tt-backend` until every gate in
§7 is green.

**Requirements this document creates**: FR-TT1…FR-TT13 (§6). They are **not** part of
`01-requirements.md`'s 70; they are a stretch group, and the traceability gate
(`03-lld-M7-tests.md` §6.3) must not count them against that denominator until the group lands.

---

## 2. Modules, files, and where the state lives

| # | Artifact | Contents | Owner |
|---|---|---|---|
| 1 | `spatial/m5tt_emit.py` | `emit(plan) -> TTProgram`. **stdlib only** — no `ttnn` import, no numpy; importable in the default suite | B |
| 2 | `spatial/m6tt_run.py` | `run(program, tensors) -> dict[str, np.ndarray]`. **Lazy `import ttnn` inside the function body**, so importing the module costs nothing and the default suite never loads the wheel | B |
| 3 | `tests/tt/` | the FR-TT acceptance tests, every one marked `requires_ttsim` | B |
| 4 | `.venv-tt` | the TT virtualenv, separate from `.venv` (the AIR pin). Git-ignored | B |
| 5 | `vendor/tt/` | download cache: `libttsim_wh.so`, `sfpi_7.75.1_x86_64_debian.txz`. Git-ignored | B |
| 6 | `vendor/tt/SHA256SUMS` | **committed**; `scripts/tt_env.sh` verifies against it before use | B |
| 7 | `scripts/tt_env.sh` | the environment recipe of §5, idempotent, the executable form of `$TT/REPRO.sh` | B |
| 8 | `design/PROGRESS-TT.md` | the state file: one section per gate, what ran, what it printed, what is still open. Same role `design/PROGRESS-B.md` plays for the AIR side | B |

`TTProgram` is a **frozen dataclass**, stdlib only, six fields:

| Field | Type | Meaning |
|---|---|---|
| `grid` | `tuple[int, int]` | the core range `[0,gx) × [0,gy)`; a 1-D `HerdPlan.grid` gives `gy = 1` |
| `kernel_source` | `str` | the whole C++ data-movement kernel, one string, byte-identical across runs (FR-TT3) |
| `cbs` | `tuple[CBSpec, ...]` | one per non-L3 `BufferPlan`, in `plan.buffers` order: `(buffer_index, bytes, page_bytes, dtype)` |
| `semaphores` | `tuple[SemSpec, ...]` | one per core↔core link end: `(id, name, initial_value)`; `initial_value` is always `0` (§3.5) |
| `runtime_args` | `tuple[tuple[tuple[int, ...], ...], ...]` | `[x][y]` → that core's argument vector (§3.4) |
| `io_order` | `tuple[str, ...]` | the `io_tensors` order handed to `ttnn.generic_op`; equals `tuple(t.name for t in plan.tensors)` |

`m6tt_run.run` is the only place `ttnn` is named. It builds `CBDescriptor` / `CBFormatDescriptor`
/ `SemaphoreDescriptor` / `KernelDescriptor` / `ProgramDescriptor` from a `TTProgram`, calls
`ttnn.generic_op(io_tensors, program_descriptor)`, and reads the outputs back. The split exists
so that FR-TT1…FR-TT3 (structure, D-14, determinism) run in the **default** suite with no wheel
installed, and only FR-TT4…FR-TT10 need ttsim.

---

## 3. The translation table

### 3.1 Shape of the module

```pseudo
 1  function EMIT_TT(plan) -> TTProgram:                     # m5tt_emit.emit
 2      grid   := (plan.herd.grid[0], plan.herd.grid[1] if rank 2 else 1)
 3      cbs    := [CB(i, b.bytes) for i, b in enumerate(plan.buffers)]      # plan order
 4      links  := CORE_TO_CORE_LINKS(plan)                   # one per (channel, concrete index)
 5      sems   := [SEM(2k, f"full_{link}"), SEM(2k+1, f"empty_{link}") for k, link in links]
 6      src    := "".join([PROLOGUE,
 7                         WALK_SEGMENT(plan.segment_body),   # only the L3 sites the herd owns
 8                         WALK(plan.herd_body),
 9                         EPILOGUE])
10      rt     := [ARGS(coord) for coord in ENUMERATE(grid)]  # §3.4
11      return TTProgram(grid, src, cbs, sems, rt, tuple(t.name for t in plan.tensors))
```

One kernel string for the whole grid; the per-core difference is carried entirely in
`runtime_args` and in `BranchNode` predicates evaluated on them at run time. That is the SPMD
shape the plan already has, and it is why one `KernelDescriptor` over one `CoreRangeSet` suffices
(measured over an 8×8 grid: `$TT/t6_multicore.py`, `MATCH=True`).

### 3.2 The table — one row per plan element

| # | Plan element (`06-interfaces.md` §) | TT-Metalium form | Notes |
|---|---|---|---|
| 1 | `plan.tensors` (§5.6) | DRAM tensors, `ROW_MAJOR` layout, handed to `ttnn.generic_op` in `io_tensors` order | **page = one row** of the tensor's last dimension. Addressed from the kernel through `TensorAccessorArgs`: host `list(ttnn.TensorAccessorArgs(t).get_compile_time_args())` concatenated in `io_order`, kernel `constexpr auto args = TensorAccessorArgs<K>()` then `TensorAccessor(args, base_addr)` (`$TT/t3_generic_op.py`, measured working). Invariant 6's read-before-write order is carried through unchanged |
| 2 | `HerdPlan.grid` (§5.4) | one `CoreRangeSet` = `CoreRange(CoreCoord(0,0), CoreCoord(gx-1, gy-1))` | rank 1 ⇒ `y = 0`. `HerdPlan.shape` and `repeats` are **ignored**: they are AIE physical-placement facts with no TT meaning, and ignoring a field is not a decision |
| 3 | `HerdPlan` (§5.4) | **one** `KernelDescriptor`, `config=DataMovementConfigDescriptor(processor=RISCV_0, …)`, `core_ranges` = row 2's set, carrying the **whole** herd body | one processor, not five. DM1/Unpack/Math/Pack stay unused (§1 non-goals) |
| 4 | `BufferPlan`, `level == "L1"` (§5.1) | one `CBDescriptor` per buffer: `total_size = bytes`, one `CBFormatDescriptor(buffer_index=i, page_size=bytes, data_format=…)` | `i` is the buffer's index in `plan.buffers`. **`ping_pong_candidate` is ignored** — it is an AIE pass predicate (`isPingPongCandidate`), and the TT emitter allocates one slot |
| 5 | a buffer's address | `get_write_ptr(cb_id)` (`$WHEEL/tt_metal/hw/inc/api/dataflow/dataflow_api.h:323`), taken once at kernel top into a `const uint32_t` | `get_read_ptr` (`:344`) is the same address for a one-page CB and is not used |
| 6 | `ChannelPlan` with an L3 end (§5.3) | **no channel object at all** — the herd side does the DRAM transfer itself (§3.3) | there is no TT analogue of an `air.channel` whose other end is a shim DMA; the transfer is the core's own `noc_async_read`/`noc_async_write` |
| 7 | `ChannelPlan` core↔core (§5.3) | a **depth-1 FIFO over a direct remote L1 write plus two counting semaphores** (§3.5) | the whole of the protocol work |
| 8 | `LoopPlan`, either `kind` (§5.5) | `for (int32_t <axis> = lo; <axis> < hi; <axis> += step) { … }` | `kind` collapses: `"sequential"` vs `"unrolled"` is an AIE trace-time distinction (D-3) with no TT counterpart. Both become the same C++ `for`; the emitter reads `kind` for nothing |
| 9 | `StoreNode` (§5.5) | `<buf>[<flat index>] = <expr>;` over `volatile <ctype>*` views of the CB address | §3.6 |
| 10 | `BranchNode` / `ChannelSite.guard` (§5.5) | `if (<predicate over the coordinate runtime args>) { then } else { otherwise }` | the `else` is omitted when `otherwise == ()`. A plain C++ `if` is correct here — the AIE prohibition (`_cond.py:41-45`) is an `air.api` tracing artefact and does not apply |
| 11 | `MappingPlan` complete | `ProgramDescriptor(kernels=[kd], semaphores=sems, cbs=cbs)` → `ttnn.generic_op(io_tensors, prog)` | `$TT/t3_generic_op.py`, one core; `$TT/t6_multicore.py`, 8×8 |

Nothing else. No compute kernel, no `CBFormatDescriptor` beyond one per buffer, no
`common_runtime_args`, no `named_compile_time_args`, no second NoC, no multicast primitive, no
`socket_api.h`, no `DataflowBuffer`/`Noc` wrapper objects. `defines` carries exactly one entry per
`TensorAccessorArgs` block offset, as the probe does.

> **Conflict — C-TT1 (measurement vs design, kept as designed).** The only *measured-working*
> path in `$TT` uses `ttnn.TILE_LAYOUT` + `bfloat16` + a 2048-byte page (`$TT/t3_generic_op.py`,
> `PAGE = 2 * 1024`; `$TT/emitted_kernel.cpp` is its emitted form). The design here mandates
> `ROW_MAJOR` + `f32`/`i32` + page = one row, which **no probe has run**. Design kept; the
> row-major/one-row-page path is `[UNVERIFIED — settled at T1]` and is T1's first failure mode.

> **Conflict — C-TT2 (measurement vs design, kept as designed).** The probe kernel reaches the
> NoC through the wrapper objects `Noc` and `DataflowBuffer` (`$TT/t3_generic_op.py`,
> `noc.async_read(...)`, `dfb.reserve_back(1)`), not the raw calls this document emits. Both
> exist in the same wheel — `noc_async_read` at `$WHEEL/tt_metal/hw/inc/api/dataflow/dataflow_api.h:552`,
> `noc_async_write` at `:828` — but *the raw pair has not been compiled by this wheel's JIT in
> our hands*. Design kept (the raw calls take an explicit destination NoC address, which the
> wrapper does not); `[UNVERIFIED — settled at T1]`.

### 3.3 L3 ↔ L1: the herd side does the DRAM transfer, row-wise

A `ChannelPlan` with one L3 end becomes, at the herd site's position in plan order:

* for a **get** (L3 → L1): one `noc_async_read(s.get_noc_addr(page_id), l1_addr + row_off, row_bytes)`
  per row of the *segment-scope* site's `Region`, then one `noc_async_read_barrier()`
  (`dataflow_api.h:1750`);
* for a **put** (L1 → L3): one `noc_async_write` per row into the destination page, then
  `noc_async_write_barrier()` (`:1780`).

Four rules make this mechanical rather than a decision:

1. **The region is the segment site's, not the herd site's.** W1's herd get has `region=empty`
   (the whole buffer) while the matching segment put carries the real
   `Region(offsets=(pi*32, kk), sizes=(32,16), strides=(64,1))`
   (`03-lld-M4-mapping.md` §6.1, segment body line 2). The emitter pairs them by
   `(channel, concrete index)`.
2. **The bundle index is substituted by this core's coordinates.** The segment put's index
   expression is `(pi, 0)`; at the herd end the matching index is `(tx, ty)`. Substituting the
   core's own coordinate runtime args into the segment site's `Region` offsets gives that core's
   slab. No arithmetic on kernel sizes is invented: every offset is an `Expr` already in the plan.
3. **Broadcast fan-in is fan-out done backwards: each destination reads independently.** W1's
   `A2L1` has `size=(2,1)`, `broadcast_shape=(2,2)` — one put, two gets. On AIE
   `air-broadcast-detection` turns that into a multicast; on TT each of the two consumer cores
   simply issues its own `noc_async_read` of the same DRAM rows. Same values, no multicast
   primitive, and P1′ balance is untouched because the plan's put/get structure is unchanged.
4. **A segment `LoopPlan` that wraps only L3 sites emits nothing.** W1's `pi_bundle`/`k0` segment
   nest (lines 0-2 of §6.1) exists to enumerate puts; on TT the consumer core already knows its
   own `k0` from its own copy of the herd-side loop. **A segment temporal loop whose trips the
   herd loop must match — W1's `k0`, which appears at both scopes — is `assert`ed equal, and an
   inequality is an internal-consistency failure (§5's list in `03-lld-M5-emitter.md` applies
   verbatim: it is our bug, not the user's).**

### 3.4 Runtime arguments

`RuntimeArgs()` is a 2-D per-core table: `rt[x][y] = [...]` (`$TT/t6_multicore.py`, the `tid`
loop). Each core's vector is three blocks, in this order:

| Block | Contents | Fixed by |
|---|---|---|
| A | this core's herd coordinates, in `HerdPlan.coords` order (`("tx","ty")`, or `("tx",)` at rank 1) | the design |
| B | per **link**, in program order: the neighbour endpoint's NoC `x` then `y` | the design |
| C | one DRAM base address per `io_order` entry | forced: `TensorAccessor(args, base_addr)` needs it, and the probe passes `src.buffer_address()` / `dst.buffer_address()` as args 0 and 1 (`$TT/t3_generic_op.py`) |

Block B's values are computed **on the host**: for each link endpoint,
`device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))`, whose result's `.x`/`.y` go into the
vector. The kernel never converts coordinates; it only substitutes them into
`get_noc_addr(noc_x, noc_y, addr)`.

> **Conflict — C-TT3 (source vs source; the design's answer is recorded and stays
> `[UNVERIFIED]`).** Which coordinate space `get_noc_addr` expects is answered **two different
> ways inside the same wheel**:
>
> * `$WHEEL/tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h:203-204` documents the
>   parameters as `noc_x | Physical x coordinate of core | WH: 0-9` and
>   `noc_y | Physical y coordinate of core | WH: 0-11`;
> * the Python binding's own docstring for the helper this design passes reads
>   *"Convert a logical coordinate to a virtual coordinate for a worker core. … Returns:
>   CoreCoord: The virtual coordinate of the worker core."*
>   (`strings $WHEEL/_ttnn.cpython-312-x86_64-linux-gnu.so`, the `worker_core_from_logical_core`
>   docstring), and upstream's own C++ names the same value `mux_virtual_core`
>   (`$WHEEL/ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/reduce_to_root_program.cpp:736`).
>
> **Design kept: pass `worker_core_from_logical_core` results.** Supporting evidence that this is
> the value kernels are given: `$WHEEL/ttnn/cpp/ttnn/operations/ccl/sharding_addrgen_helper.cpp:49-51`
> names exactly that result `noc_core` and packs it into kernel args at `:93-95`. The *space* is
> `[UNVERIFIED — settled at T2]`; on a non-harvested Wormhole the two may coincide, which is
> precisely why T2 must check it rather than infer it.
>
> *(Citation erratum against the brief: `worker_core_from_logical_core` does **not** appear in
> `strings $WHEEL/build/lib/_ttnncpp.so` — 0 occurrences, measured. It appears 3× in
> `$WHEEL/_ttnn.cpython-312-x86_64-linux-gnu.so` and 9× in `$WHEEL/build/lib/libtt_metal.so`. The
> API exists; the evidence lives in the other two objects.)*

The kernel reads them as `get_arg_val<uint32_t>(i)` at fixed indices computed by the emitter from
the plan, and emits those indices as literals. Coordinates are also available device-side as
`get_absolute_logical_x()` / `get_absolute_logical_y()`
(`$WHEEL/tt_metal/hw/inc/api/compute/common.h:121-140`); **they are not used** — a runtime arg is
one mechanism for every coordinate rather than two, and `BranchNode` predicates then read the same
values on every target.

### 3.5 Core ↔ core: a depth-1 FIFO over a direct remote L1 write

This is the row that earns the document. Let a core↔core `ChannelPlan` have, at concrete bundle
index `k`, producer core `p` and consumer core `c` (both fixed at plan time: the plan's guards and
index expressions are enumerated over the concrete grid, D-12, exactly as
`03-lld-M4-mapping.md` §3.7 does).

**A-TT1 (assumption, verified at T2).** Every core carries the **identical** CB layout — the same
`CBDescriptor` list over one `CoreRangeSet` — therefore a given buffer's L1 address is the same
number on every core. This is what makes a remote write addressable: the producer computes the
*consumer's* buffer address as its own. If T2 shows the allocator varying the address per core,
the fallback is to pass each consumer's buffer address as a further runtime-arg block; that is a
change to §3.4 block B, nothing else.

**Two counting semaphores per link**, both created with `initial_value = 0` and **never reset**:

| Semaphore | Lives on | Counts |
|---|---|---|
| `full_k` | the **consumer** `c` | payloads delivered on link `k` |
| `empty_k` | the **producer** `p` | slots freed by the consumer on link `k` |

`SemaphoreDescriptor{.id, .core_type = WORKER, .core_ranges, .initial_value = 0}` is the exact
shape (`$WHEEL/ttnn/cpp/ttnn/operations/data_movement/move/device/move_overlap_program_factory.cpp:112-118`;
also `$WHEEL/ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:2082-2086`). Under A-TT1 the
`core_ranges` is the whole grid, so every id exists on every core.

**The producer's `n`-th put on link `k`** (`n` counted from 0, a **static** counter the emitter
increments as it walks the plan — program order is the counter):

```cpp
  noc_semaphore_wait_min(EMPTY_k, n);                       // n == 0 passes: the pre-signalled slot
  noc_async_write(src_l1 + off_src,
                  get_noc_addr(NBR_X, NBR_Y, DST_L1 + off_dst),
                  BYTES);
  noc_async_write_barrier();
  noc_semaphore_inc(get_noc_addr(NBR_X, NBR_Y, get_semaphore(FULL_k)), 1);
```

**The consumer's `n`-th get on link `k`**:

```cpp
  noc_semaphore_wait_min(FULL_k, n + 1);                    // the data is already in place
  /* … the compute that reads the received region … */
  noc_semaphore_inc(get_noc_addr(PROD_X, PROD_Y, get_semaphore(EMPTY_k)), 1);
```

Six things this fixes, each of which is otherwise a decision:

1. **The producer writes into the consumer's L1 directly.** There is no staging buffer and no
   rendezvous. The destination is the consumer's *get-site* buffer plus the get site's `Region`
   offset — so W2's `get ToNorth[tx] -> src[HS+1:HS+2, :]` (`03-lld-M4-mapping.md` §6.3, STEP
   line 2) is written by the neighbour straight into row `HS+1` of that core's `src` strip.
2. **`noc_async_write`'s contract is exactly this shape**: its documentation says the destination
   is an on-chip node "located at NOC coordinates (x,y) and a local address created using
   `get_noc_addr`" (`$WHEEL/tt_metal/hw/inc/api/dataflow/dataflow_api.h:828` and the comment block
   above it), signature `noc_async_write(uint32_t src_local_l1_addr, uint64_t dst_noc_addr,
   uint32_t size, uint8_t noc, uint32_t vc)`.
3. **`wait_min`, not `wait`.** `noc_semaphore_wait` spins on equality (`:1943`, `while ((*sem_addr) != val)`);
   `noc_semaphore_wait_min` spins on `<` (`:1969`, `while ((*sem_addr) < val)`). With
   never-reset counters only the `min` form is correct — a fast producer may already have
   incremented past `n`.
4. **Barrier before the increment, never after.** `noc_async_write_barrier()` (`:1780`) is what
   makes the payload visible before the flag that advertises it. `noc_async_writes_flushed()`
   (`:1809`) is the weaker form and is **not** used. *(Ordering is §9's Q-TT3: barrier-then-inc is
   what the design fixes; whether ttsim would also catch the reversed order is a T2 question, and
   `UndefinedBehavior` reporting is the thing that might catch it — §5.)*
5. **The empty increment goes at the end of the enclosing body in plan order** — the end of W2's
   `STEP`, W3's `ROW`, the cascade's middle-PE block — i.e. **after** the compute that reads the
   received region, never immediately after the wait. Incrementing early would let the producer
   overwrite a region still being read, which at depth 1 is the only race there is.
6. **`get_semaphore(id)` resolves an id to a local L1 address**
   (`:1501-1503`, `sem_l1_base[type] + id * L1_ALIGNMENT`); the *remote* address is that local
   address wrapped by `get_noc_addr(x, y, …)`. The object-oriented spelling of the same thing is
   `$WHEEL/tt_metal/hw/inc/api/dataflow/noc_semaphore.h:254`. `noc_semaphore_set_remote` (`:1521`)
   writes a 4-byte value rather than atomically incrementing and is **not** used;
   `noc_semaphore_set` (`:1995`) is a local store, used nowhere because the descriptor's
   `initial_value = 0` already does it; `noc_semaphore_inc` (`:2264`,
   `noc_semaphore_inc(uint64_t addr, uint32_t incr, …)`) is an atomic and is the only mutator.

### 3.6 Deadlock freedom

**P1′ and P2b remain the load-bearing conditions and are unchanged** (`03-lld-M4-mapping.md`
§3.7.1, §3.7.2). The TT protocol is a *depth-1 FIFO with a pre-signalled empty slot*, which is
the same structure VF §C found SAFE for the AIE lock protocol. The argument is written out per
workload, as M4 §6.3/§6.4 do:

**W2, `PI = 2`, two PEs, two links** (`ToSouth[0]`: PE0 → PE1; `ToNorth[0]`: PE1 → PE0). Site
order inside `STEP` is fixed and not negotiable: `PUT(north) → PUT(south) → GET(north) → GET(south)`
(`03-lld-M4-mapping.md` §3.6.1).

* Every PE issues **both** of its puts before **either** of its gets. At step `n`, PE0's put on
  `ToSouth[0]` waits `empty_ToSouth0 ≥ n`; PE1's put on `ToNorth[0]` waits `empty_ToNorth0 ≥ n`.
  At `n = 0` both pass (the pre-signalled slot).
* Both writes therefore land and both `full` counters reach `n+1`; both gets, waiting
  `full ≥ n+1`, unblock. Neither PE is ever blocked on a semaphore only the other's *later* work
  could raise.
* Inductively, PE `x` at step `n+1` waits `empty ≥ n+1`, which its partner increments at the end
  of *its* step `n` body — after the 5-point update that read the ghost row. So the producer is
  at most one step ahead: exactly depth 1.
* The guards are what keep this exact at the domain edge: PE0's `ToNorth` put is guarded `tx > 0`
  and never fires; PE1's `ToSouth` put is guarded `tx < PI-1` and never fires. The links that do
  not exist cost nothing because their sites are not emitted on those coordinates.

**W3, `PJ = 4`, a chain 0 → 1 → 2 → 3 over `West` (3 links)**. `WestIn` and `EastOut` have an L3
end, so under §3.3 they are DRAM transfers and carry **no semaphore**; only `West` does.

* PE0 never waits on a core↔core semaphore — its `edge_in` comes from DRAM. It always makes
  progress.
* PE `k > 0`'s `ROW` begins by waiting `full_{k-1} ≥ n+1`, which PE `k-1` raises at the end of
  *its* `ROW` `n`. By induction along the path, if PE `k-1` completes row `n` then PE `k` does.
* Backwards, PE `k`'s put at row `n+1` waits `empty_k ≥ n+1`, raised by PE `k+1` at the end of
  its row `n`. The dependency graph is the path itself, in both directions, with no cycle —
  which is the P2b property M4 already proved for this plan.
* The wavefront is emergent, exactly as on AIE: nothing in the TT program expresses the skew;
  each PE blocks on its own `wait_min` and the diagonal order follows.

**W1-flip, the cascade chain, `PK = 4`, ASCENDING** — the same path argument as W3 with the head
at PE0 and the tail at PE3, the middle block being `GET(recv) → acc += recv → PUT(acc)`
(`03-lld-M4-mapping.md` §3.6.3 lines 19-23). `ChannelPlan.chain_direction` is read and never
derived (D-14), and on TT it selects which endpoint of each link is producer.

**W1 has no core↔core channel at all** — `A2L1`, `B2L1`, `C2L3` all have an L3 end — so the TT
program has **zero** semaphores. That is why W1 is gate T1: it exercises §3.2–§3.4 and §3.6 and
nothing else.

### 3.7 Compute: scalar C++

A `BufferPlan` becomes one pointer at kernel top:

```cpp
  volatile tt_l1_ptr float*   acc = (volatile tt_l1_ptr float*)get_write_ptr(0);
  volatile tt_l1_ptr int32_t* qb  = (volatile tt_l1_ptr int32_t*)get_write_ptr(1);
```

| `ExprNode` (§5.5) | C++ | Rule |
|---|---|---|
| `Load(buffer_id, subs)` | `buf[<row-major flat index over BufferPlan.shape>]` | the flat index is the same row-major arithmetic `plan_interp.row_major` asserts (`tests/helpers/plan_interp.py:47-53`) |
| `Const(value, text, dtype)` | `text`, verbatim, with an `f` suffix for `f32` | never a float round-trip — `0.2` is `0.2f`, not `repr(1/5)` (HLD §5 rule 4) |
| `BinOp(op, lhs, rhs)` | `(<lhs> <op> <rhs>)` | **parenthesised exactly per the tree**, every node, no precedence shortcuts. This is what makes f32 association identical to the interpreter's |
| `Neg(operand)` | `(-<operand>)` | |
| `MaxMin("maximum", (a,b,c,d))` | nested ternary, left-folded: `((a) > (b) ? (a) : (b))` composed, **or** `std::max` if the JIT include set provides `<algorithm>` | the fold order is the tuple order, matching `plan_interp._value`'s `max(values)` over the same list |
| `Select(cmp, l, r, t, o)` | `((<l> <cmp> <r>) ? <t> : <o>)` | an expression, never control flow |
| `StoreNode(buf, subs, expr)` | `buf[<flat>] = <expr>;` | |
| `LoopPlan` | `for (int32_t <axis> = <lo>; <axis> < <hi>; <axis> += <step>) { … }` | bounds are the plan's `Expr`s, formatted by the one literal formatter |

**Two dtypes only.** `f32` → `float`, `i32` → `int32_t`. Anything else raises (§5.1). The
Wormhole scalar float path is **soft-float**: the only `-march` string in
`$WHEEL/build/lib/libtt_metal.so` is Blackhole's
`rv32im_zmmul_zaamo_zba_zbb_xtttensixbh_zve32f`, which carries no scalar `f` extension, and the
Wormhole string could not be read from the wheel at all — `[UNVERIFIED — read from a verbose JIT
log at T2]`. Soft-float is still IEEE-754 single precision, so exactness against numpy is a
reasonable requirement rather than a hope; W2 is the test that settles it (§6 FR-TT7, §9 Q-TT4).

### 3.8 Determinism and D-14

D-14 holds verbatim: **the TT emitter makes no decisions.** Its operational form here, and
`test_TT_emitter_makes_no_decisions` is the lint:

| # | Invariant |
|---|---|
| TP-1 | **No field of `plan.mapping` is read.** Not `LegalMapping`, not `ScheduleModel`, not `KernelModel`. Everything arrives through `MappingPlan`. The lint greps the module's AST for attribute chains rooted at `.mapping` |
| TP-2 | No arithmetic on kernel sizes. Extents come from `BufferPlan.shape`, `Region.sizes`, `LoopPlan` bounds |
| TP-3 | Plan order is emission order. `plan.tensors`, `plan.buffers`, `plan.segment_body`, `plan.herd_body`, `LoopPlan.body`, `ChannelSite.order` — iterated, never sorted or regrouped |
| TP-4 | No dict or set iteration reaches the output. The name → CB-index table is read by key |
| TP-5 | No wall clock, no `id()`, no `uuid`, no RNG, no absolute path in the kernel string |
| TP-6 | The kernel string is byte-identical across processes and `PYTHONHASHSEED` values (FR-TT3), the same property `test_E10_byte_identical` asserts on the AIR side |
| TP-7 | Fields deliberately **ignored**, listed so ignoring them is visible rather than accidental: `BufferPlan.ping_pong_candidate`, `BufferPlan.loop_depth`, `HerdPlan.shape`, `HerdPlan.at`, `LoopPlan.kind`, `ChannelSite.is_async`, `ChannelSite.depends_on`, `ChannelPlan.channel_type` (its *chain_direction* is read) |

TP-7 is the honest half of "the plan is backend-neutral": eight fields are AIE-specific
scheduling hints that the TT model has no use for. **None of them carries semantics** — each is a
performance or placement hint, or an AIE tracing artefact — which is the claim the executed tests
of §6 check by getting the same numbers out.

---

## 4. TT-P3 — the resource model that replaces the AIE check

`03-lld-M4-mapping.md` §3.8's `DMA-CHANNELS` check is an **AIE2 fact**: two S2MM and two MM2S
circuit-switched DMA channels per core tile. **It is not run for this target.** A Tensix core has
two NoCs and no equivalent per-core endpoint budget in anything we can cite, so there is no DMA
budget row here. P1′ and P2b are **mandatory and unchanged** — they are properties of the plan,
not of the device.

What the TT target does check, per core:

| Bound | Value | Source |
|---|---|---|
| L1 per worker core | **1 499 136 B** | `$TT/ttsim-bin/soc_descriptor.yaml:132-133` (`worker_l1_size: 1499136`), byte-identical to `$WHEEL/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml` (measured `diff`). Equals `MEM_L1_SIZE (1464 * 1024)` in `$WHEEL/tt_metal/hw/inc/internal/tt-1xx/wormhole/dev_mem_map.h:33` |
| reserved | **32 768 B** (32 KB) | `dev_mem_map.h:71-72`: *"1432 KB = 1464 KB (L1 total) − 32 KB (MEM_MAP_END system reserved)"*, `#define MEM_MAX_KERNEL_SIZE (1432 * 1024)`. **`[UNVERIFIED]` as the actual CB-arena bound** — it is the *kernel binary* limit in that header, and the kernel binary itself also occupies L1; the effective CB ceiling is measured at T2 |
| usable, as designed | **1 466 368 B** | `1 499 136 − 32 768`. An estimate until T2 |
| semaphores per core | **`[UNVERIFIED — measured at T2]`** | not derivable statically from the wheel: no `NUM_SEMAPHORES`/`MAX_SEMAPHORE` constant exists anywhere under `$WHEEL/tt_metal/hw/inc/` (grep, 0 hits), and the bound surfaces only as a host message, `Semaphore id {} exceeds max value {}` (`strings $WHEEL/build/lib/libtt_metal.so`). The allocator-side spelling is `desc.find_available_semaphore_id(core, CoreType::WORKER)` (`$WHEEL/ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:2073`) |
| alignment | `L1_ALIGNMENT = 16` B; DRAM read 32 B, write 16 B | `$WHEEL/tt_metal/hw/inc/internal/tt-1xx/wormhole/noc/noc_parameters.h:291-292` (`NOC_L1_{READ,WRITE}_ALIGNMENT_BYTES 16`), `:295-296`, `:299-302` |

**The check**: `Σ CB bytes + Σ semaphore words ≤ usable`, raised before `m6tt_run.run` opens a
device. Semaphore words are 4 B each but are spaced `L1_ALIGNMENT` apart by `get_semaphore`
(`dataflow_api.h:1501-1503`), so the figure counted is `16 · len(semaphores)`.

**The four workloads.** Note the denominator shift against the AIE figures: the plan's L1 totals
double every `ping_pong_candidate` buffer (`06-interfaces.md` §5.6 invariant 5); the TT emitter
**ignores** that field (§3.2 row 4), so its CB total is the undoubled sum.

| Variant | CB bytes (TT, undoubled) | plan L1 (AIE, doubled) | links | semaphore ids | waited per interior PE | CB + sem, % of usable |
|---|---|---|---|---|---|---|
| W1 | `4096 + 2048 + 2048` = **8 192** | 12 288 | 0 | **0** | 0 | 0.56 % |
| W1-flip | `8192 + 2048 + 4096 + 8192` = **22 528** | 24 576 | `PK−1 = 3` | 6 | **2** | 1.54 % |
| W2 | `640 + 640` = **1 280** | 1 280 | 2 (`ToNorth[0]`, `ToSouth[0]`) | **4** | 2 | 0.09 % |
| W3 | `128 + 32 + 36 + 36 + 4 + 4` = **240** | 240 | `PJ−1 = 3` | 6 | **2** | 0.02 % |

*Denominators, stated because the design's figures use two different ones.* **Semaphore ids** is
the program-wide count, `2 × links` (one `full`, one `empty` per link); under A-TT1 every id is
allocated on every core, so it is also the per-core allocated count. **Waited per interior PE** is
how many an interior core actually blocks on: its inbound link's `full` and its outbound link's
`empty` = 2. The design's headline figures — *W2 needs 4, W3 2 per interior PE, flip 2* — are the
first and second columns respectively. Percentages are against 1 466 368 B, itself an estimate.

**One concrete alignment risk, from the table above.** W3's buffers are not multiples of
`L1_ALIGNMENT = 16`: `prev`/`cur` are 36 B and `edge_in`/`edge_out` are 4 B, and W3's L3 edge
transfers are single `i32` columns — `S[i, 0:1]` = **4 B**, below the 32 B DRAM read alignment.
CB `page_size`/`total_size` are therefore rounded **up** to the next multiple of 16 by the
emitter (a padding, not a layout change — the flat index arithmetic of §3.7 is unchanged), and
whether a 4-byte DRAM transfer is accepted at all is `[UNVERIFIED — settled at T2]`, Q-TT7.

---

## 5. Environment pin

Exactly `$TT/REPRO.sh`, promoted to `scripts/tt_env.sh` and made idempotent. Everything below was
established on 2026-09-13 at gate G-TT0.

| Component | Pin | Verified |
|---|---|---|
| `ttnn` | **0.78.0**, cp312, `manylinux_2_34_x86_64`, 90.3 MB | installed into `.venv-tt` from PyPI |
| ttsim | `libttsim_wh.so` **v1.10.7** (released 2026-09-12), sha256 `d2b7cb4ffcefe06cffe218bc32de36f4ca0755966e31ce9e69c4c9df73528197` | re-measured by `sha256sum` today |
| SOC descriptor | `soc_descriptor.yaml` — **the filename is exact**, and it must sit **beside the `.so`** | byte-identical to `$WHEEL/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml` (measured `diff`, no differences) |
| sfpi | `sfpi_7.75.1_x86_64_debian.txz`, sha256 `7ec141c063eaf1290359ecce23dff3133574371f64a24cc0cce1619b8756ba28`, unpacked and symlinked at `$WHEEL/runtime/sfpi` | **a wheel gap**: the device toolchain is not bundled and the wheel will not JIT without it |
| Python | 3.12, `uv venv --seed` | same interpreter policy as the AIR venv (`design/PROGRESS-B.md` §1 row 1) |

Three environment variables, and the third is the one that costs an afternoon:

```sh
export TT_METAL_SIMULATOR=<dir>/libttsim_wh.so    # the .so; the descriptor is found beside it
export TT_METAL_SLOW_DISPATCH_MODE=1              # no fast-dispatch firmware on a simulator
unset TT_METAL_HOME                               # MUST be unset: the wheel is self-rooted
```

**The `TT_METAL_HOME` confound.** A `TT_METAL_HOME` left over from a source checkout re-roots the
wheel's include and firmware search and produces failures that look like missing headers rather
than like a misconfiguration. `scripts/tt_env.sh` unsets it and prints a line saying it did;
`m6tt_run.run` raises with that instruction if it finds it set.

**Cache and provenance.** `vendor/tt/` holds the two downloads; `vendor/tt/SHA256SUMS` is
committed and checked with `sha256sum -c -` before either is used — the same "verify the pin,
refuse otherwise" rule `TOOL-VERSION-PIN` enforces on the AIR side.

**ttsim's `UndefinedBehavior` reporting is an asset, not noise.** A functional simulator that
names undefined behaviour turns a class of protocol bug — reading a region before its `full`
increment, writing past a CB, an unaligned transfer — into a printed diagnostic instead of a
plausible wrong number. Every gate in §7 records the UB output, and a clean numeric result with UB
lines printed is **not** a pass.

---

## 6. Requirements and acceptance tests

Every test lives in `tests/tt/`, is marked `fr("FR-TTn")` in the existing style
(`tests/conftest.py`, `_MARKERS`), and those needing a device are additionally marked
`requires_ttsim`. The marker is registered in `tests/conftest.py::_MARKERS` and **added to the
default deselect** in `pyproject.toml` (`addopts = "-m 'not slow and not requires_device and not
requires_ttsim' …"`), so the default suite is unchanged (FR-TT13).

| FR | Requirement | Acceptance test |
|---|---|---|
| **FR-TT1** | `m5tt_emit.emit(plan)` returns a `TTProgram` whose `grid`, `cbs`, `semaphores`, `runtime_args` and `io_order` match §2's table for each of the four plans, and whose `kernel_source` contains: one `kernel_main`, one `for` per `LoopPlan`, one `if` per `BranchNode`, and — for W2/W3/flip — one `noc_semaphore_wait_min` per get site and one `noc_semaphore_inc` per put site plus one per consumed region | `test_TT_emit_structure[w1\|w2\|w3\|flip]` — structural assertions on the dataclass plus counted regex over the string. **No device**, runs in the default suite |
| **FR-TT2** | The TT emitter reads **no** field of `plan.mapping`, and no `LegalMapping`, `ScheduleModel` or `KernelModel` attribute (D-14, TP-1/TP-2) | `test_TT_emitter_makes_no_decisions` — AST walk of `spatial/m5tt_emit.py` for attribute chains rooted at `.mapping`, plus the M5 lint's own name list. **No device** |
| **FR-TT3** | `kernel_source` is byte-identical across two in-process emissions and one fresh process at a different `PYTHONHASHSEED` (TP-6) | `test_TT_byte_identical[w1\|w2\|w3\|flip]`, the shape of `test_E10_byte_identical`. **No device** |
| **FR-TT4** | W1 executes on ttsim and the output tensor equals the numpy oracle **exactly** | `test_TT_exec_w1` — `np.array_equal(C, A @ B)` on the `default_rng(0)` integer-valued f32 fixture the AIR side already uses (`design/PROGRESS-B.md`: `C == A @ B` exactly, `np.array_equal`) |
| **FR-TT5** | W1-flip (1-D `grid(4)`, ascending cascade) executes and matches exactly | `test_TT_exec_w1_flip` — same oracle, same exactness |
| **FR-TT6** | W3 executes and matches the textbook Smith-Waterman DP exactly (`i32`; integer arithmetic, so exactness is not in question) | `test_TT_exec_w3` — `np.array_equal` against the two-loop DP |
| **FR-TT7** | W2 executes and matches the two-loop numpy Jacobi **exactly**, at `T = 4` and `T = 5` | `test_TT_exec_w2[4\|5]` — `np.array_equal`, **not** a tolerance. The interpreter already matches at max abs error **0.0** (`design/PROGRESS-B.md`, the `default_rng(0)` run over both targets and both parities), so exact is the right bar; if soft-float moves it, **record the measured deviation in `design/PROGRESS-TT.md` and do not silently widen the test** |
| **FR-TT8** | A **negative control** per variant: one deliberate perturbation of the emitted kernel produces a *wrong* result, proving the test could fail | `test_TT_neg[w1\|w2\|w3\|flip]` — the shape of `$TT/t6_neg.py`, which perturbs one page index (`{.page_id = i}` → `{.page_id = i + 1}`) and asserts `MATCH=False`. Per variant: W1 an offset row, W2 a dropped `full` increment, W3 a shifted edge column, flip a skipped accumulate |
| **FR-TT9** | For each variant, the ttsim result equals `tests/helpers/plan_interp.run(plan, tensors)` on the same inputs | `test_TT_matches_interp[w1\|w2\|w3\|flip]` — `np.array_equal` between the two. **This is the backend-neutrality claim**: one plan, two executions, same numbers |
| **FR-TT10** | The emitted program's semaphore count is within the measured per-core limit, and the check names the limit and where it came from | `test_TT_semaphore_budget` — asserts `len(program.semaphores) <= TT_SEM_LIMIT`, with `TT_SEM_LIMIT` a module constant carrying its provenance comment. **Until T2 measures it the constant is `None` and the test `xfail`s with the reason**, rather than asserting a guessed number |
| **FR-TT11** | `Σ CB bytes + 16·len(semaphores) ≤ usable L1` (§4), checked before a device is opened | `test_TT_l1_budget[w1\|w2\|w3\|flip]` — the four rows of §4's table, asserted against the constants and their citations |
| **FR-TT12** | With `ttnn` uninstalled or `TT_METAL_SIMULATOR` unset, every device test **skips with a reason naming what is missing**, and never errors | `test_TT_skips_without_ttsim` — runs the collection with the variables cleared and asserts skip, not error. Mirrors `03-lld-M7-tests.md`'s skip discipline |
| **FR-TT13** | The default suite is unaffected: same selected count, same outcomes, no `ttnn` import | `test_TT_default_suite_unaffected` — collection-only comparison of the default `-q` selection before and after the branch, plus an assertion that `sys.modules` has no `ttnn` after the default run |

FR-TT4…FR-TT9 are `requires_ttsim`. FR-TT1, FR-TT2, FR-TT3, FR-TT11, FR-TT12 and FR-TT13 are not
and run everywhere.

### 6.1 Error paths

`m5tt_emit.emit` raises `EmissionError` and nothing else (NFR-7, `06-interfaces.md` §6.2), for:
a dtype other than `f32`/`i32`; a `Region` whose strides are not row-major; a `BufferPlan.scope`
that is not herd-private; a segment/herd temporal-loop trip-count mismatch (§3.3 rule 4); a
resource failure from §4. `m6tt_run.run` raises `EmissionError` carrying `ttnn`'s **original**
message verbatim in `details`, exactly as `EMIT-AIR-API` does for `air.api`.

**Four new error codes would be needed** — `EMIT-TT-DTYPE`, `EMIT-TT-L1`, `EMIT-TT-SEMAPHORES`,
`EMIT-TT-RUNTIME`. `06-interfaces.md` is **frozen at 43 codes**, so they are **proposed, not
adopted**: they go through `00-README.md` §4 (one-paragraph proposal, all three owners agree,
`CONTRACT_VERSION` bump, change-log row, same commit). Until then the TT emitter raises
`EmissionError` with the code field carrying the nearest adopted code and the specific condition
in `reason`, and `test_D3_catalogue_complete` must **not** see a TT code.

---

## 7. Gates

Each gate is green only when **all three** hold: the execution is exact, the negative control
fails as designed, and the result equals `plan_interp.run`. UB lines from ttsim void a pass (§5).

| Gate | Workload | What it first exercises | Status |
|---|---|---|---|
| **T1** | **W1** (GEMM output-stationary) | DRAM row-wise transfer, `TensorAccessor`, CBs, per-core runtime args, the scalar compute walk, `BranchNode`-free path. **Zero semaphores.** Also settles C-TT1 (row-major/one-row pages) and C-TT2 (the raw NoC calls) | **in progress** |
| **T2** | **W3** (wavefront) | **the first semaphores**: a 3-link chain, `full`/`empty`, `wait_min`, remote `inc`. Settles A-TT1, the `get_noc_addr` coordinate space (C-TT3), the semaphore limit, the `-march` string, the 4-byte-transfer alignment question | not started |
| **T3** | **W2** (Jacobi halo) | **bidirectional** exchange, two links per interior PE, the `cur`/`next` swap with the odd-`T` peel, and f32 exactness under soft-float | not started |
| **T4** | **W1-flip** (cascade) | `chain_direction` read rather than derived; a 1-D grid; the accumulate-in-the-middle block | not started |

**Stop rule.** *A gate that is not green after two agent-days is abandoned, not extended.* Keep
whatever runs, write what failed into `design/PROGRESS-TT.md`, and **say so on the slide** — "W1
and W3 execute on ttsim; W2 did not" is a true, useful sentence and a silently omitted workload is
not. The AIE path is unaffected either way; this whole document is the designated cut.

---

## 8. Honest limits — the pitch text

Five sentences, to be said out loud rather than implied:

1. **This is a functional simulator, not silicon.** No Tenstorrent hardware was involved at any
   point. ttsim is Apache-2.0 and its project states bit-exact numerical results relative to
   silicon — **that is Tenstorrent's claim, not our measurement.**
2. **The compute is scalar C++ on one of five RISC-V cores.** The Tensix matrix and vector engines
   are not touched. Nothing here is a performance result, and no number in this document is a
   cycle count.
3. **One generation, Wormhole.** Not Blackhole, not multi-chip.
4. **The AIE path stays primary.** The TT emitter exists to show the `MappingPlan` is
   backend-neutral and to *execute* the plan's semantics — closing D-9's gap on one target — not
   to be a second product.
5. **What it does prove is narrow and real**: the same plan, unchanged, drives two unrelated
   device models to the same numbers, and on the TT side those numbers came out of an execution
   rather than a static argument.

---

## 9. Open questions

All owned by **B**; each names the gate that closes it.

| # | Question | Due |
|---|---|---|
| **Q-TT1** | **A-TT1** — is a buffer's L1 address identical on every core when one `CBDescriptor` list covers one `CoreRangeSet`? If not, consumer buffer addresses become a fourth runtime-arg block | **T2** |
| **Q-TT2** | Which coordinate space does `get_noc_addr(noc_x, noc_y, addr)` expect — the header says *physical*, the Python helper's docstring says it returns *virtual* (C-TT3). The design passes `worker_core_from_logical_core`; confirm | **T2** |
| **Q-TT3** | Is `noc_async_write_barrier()` before `noc_semaphore_inc` sufficient ordering, or does the increment need a stronger fence? And does ttsim report the reversed order as UB? | **T2** |
| **Q-TT4** | Does soft-float scalar f32 reproduce numpy bit-for-bit on W2's 5-point stencil, given §3.7's exact parenthesisation? If not, by how much — recorded, not tolerated away | **T3** |
| **Q-TT5** | The per-core semaphore limit. Not derivable from the wheel (§4); read it from the host error by over-allocating deliberately | **T2** |
| **Q-TT6** | ttsim wall time for W2 at `T = 4` on a 2-core grid. The 8×8 probe ran in seconds (`$TT/mc8.log`), but W2 has 4 timesteps × 2 halo exchanges per step. If it exceeds the suite budget, W2 becomes `slow` | **T3** |
| **Q-TT7** | Alignment: are 36-byte and 4-byte CB pages accepted after rounding to `L1_ALIGNMENT = 16`, and is W3's 4-byte DRAM edge-column transfer legal against a 32-byte DRAM read alignment? | **T2** |
| **Q-TT8** | The row-major / one-row-page tensor path and the raw `noc_async_read`/`noc_async_write` calls — neither has been run (C-TT1, C-TT2) | **T1** |

---

## 10. Neighbours, and why not tt-mlir

**Named in any pitch**, alongside the AIE neighbours in `CLAUDE.md`:

| Work | What it is | How this differs |
|---|---|---|
| **TileLoom** (arXiv 2512.22168) | Triton → Tenstorrent | a Triton frontend and a performance story; ours is a *second* backend from one backend-neutral plan, functional only |
| **`tenstorrent/tt-lang`** | Tenstorrent's own kernel language | a first-party language for the platform; ours is a portability demonstration, not a kernel language for TT |
| **`kernelize-ai/triton-tenstorrent`** | Triton → Tenstorrent | same frontend axis as TileLoom |
| **`qualcomm/hexagon-mlir`** | Triton → Hexagon | **out of scope and staying out**: Hexagon is a single DSP core, not a PE grid, so the spatial `place`/`grid` surface has nothing to map onto (`CLAUDE.md`, scope rules) |

**Why `tt-mlir` is not used.** `docs.tenstorrent.com/tt-mlir` documents the TTIR / TTNN / D2M /
TTKernel / TTMetal dialects (architect-verified 2026-09-13). Going through them would mean
**building a compiler** — a second toolchain to pin, install and version-check on three machines —
to reach the same `ttnn` program descriptors this document builds directly from Python. For a
**functional** target that is cost with no benefit: the plan already carries explicit buffers,
regions, loops and expression trees, so there is nothing an MLIR round trip would infer that we do
not already have, and D-14 forbids inferring anything anyway. If the target ever became a
performance target, tt-mlir would be the right entry point and this decision would be reversed.
