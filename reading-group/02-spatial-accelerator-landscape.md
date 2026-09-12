# The Spatial Accelerator Landscape

*Week 1 discussion material for the [ML Compilers Reading Group](00-syllabus.md).*

**What this is for.** Before we argue about whether a common spatial abstraction is
possible ([the proposal's central hypothesis](../proposals/spatial_dsl_project_proposal.md#4-central-research-hypothesis)),
we need a shared picture of what we would be abstracting *over*. This document compares six
machines along six axes and ends with the uncomfortable finding: **the machines agree on
almost everything except the one thing our proposed abstraction assumes.**

Read it before Week 1. Come with an opinion on §8.

---

## 1. The cast

| | AMD XDNA NPU | Versal ACAP | Tenstorrent | Groq LPU | Cerebras WSE | *(Graphcore IPU)* |
|---|---|---|---|---|---|---|
| Compute unit | AIE-ML tile | AIE tile + PL | Tensix core | functional slice | PE | tile |
| Count | 20 (XDNA1) / 32 (XDNA2) | up to ~400 | 72–140 / chip | 20 superlanes × 16 slices | 850k–900k | 1472 |
| Unit ISA | 7-way VLIW SIMD | 7-way VLIW SIMD | 5× RISC-V + matrix engine | slice-specific | small dataflow core | 6-thread barrel core |
| Local memory | 64 KB | 32 KB (AIE1) | ~1.5 MB | (slice-resident) | 48 KB | 624 KB |
| Total on-chip | ~2–4 MB + memtiles | ~30 MB | ~100–200 MB | ~220 MB | 40–44 GB | ~900 MB |
| Off-chip | DDR (shared w/ CPU) | DDR/HBM | GDDR6 | **none** | **none** (streamed) | DDR (streaming) |
| Interconnect | circuit-switched AXI-Stream | AXI-Stream + PL + NoC | 2× packet-switched 2D torus NoC | statically routed E-W | wavelet mesh, 32-bit msgs | all-to-all exchange |
| Sync model | locks + backpressure | locks + backpressure | semaphores + backpressure | **cycle-exact static schedule** | **data-triggered tasks** | **BSP barriers** |
| Low-level API | IRON / MLIR-AIE | ADF / Vitis / MLIR-AIE | TT-Metalium | *(none exposed)* | CSL | Poplar |

> ⚠️ Numbers are approximate and generation-dependent. Whoever presents should verify against
> current vendor datasheets — see [§9](#9-numbers-to-verify-before-you-present).

Graphcore is in parentheses because it is not a target for us, but its synchronization model
is a useful third point that keeps the AIE-vs-Tenstorrent comparison from looking universal.

---

## 2. Axis A — The unit of compute

The single most important question for a compiler: **what is the thing you place code on?**

- **AIE tile (AMD/Xilinx).** A 7-way VLIW, 512-bit SIMD processor with its own program
  memory. You compile a C/C++ kernel to it. It is a *processor*: it has a program counter,
  a stack, and loops. Vector width and MAC counts differ across generations (AIE1 → AIE-ML
  → AIE2p) but the model is stable.
- **Tensix core (Tenstorrent).** Not one processor but **five RISC-V cores plus a matrix
  engine**, and the five are not symmetric. Convention: two are data-movement cores (one per
  NoC direction), three drive the compute pipeline (unpack / math / pack). You write *three
  separate kernels per core* — reader, compute, writer — and they run concurrently.
  **The producer/consumer split is imposed by the hardware, not chosen by the compiler.**
- **Functional slice (Groq).** The chip is sliced *by function*, not by tile: memory slices,
  vector slices, matrix slices, switch slices, laid out as columns. Data flows east–west
  through them. There is no "core" to place a kernel on; there is a pipeline stage to
  schedule an instruction into, at an exact cycle.
- **PE (Cerebras).** A very small core (48 KB) with a router. Execution is *triggered*:
  arriving wavelets on a given color activate a task. You write per-PE code in CSL, but the
  granularity is so fine that you think in terms of thousands of PEs at once.
- **PCU/PMU (SambaNova, and Koeplinger's Plasticine before it).** Reconfigurable compute and
  memory units — the closest thing in industry to the `Spatial` PLDI'18 model. Worth knowing
  because it is our proposal's intellectual ancestor.

**Compiler consequence.** "Place a computation on a core" means four different things here.
For AIE it means *emit a program*. For Tensix it means *emit three cooperating programs*.
For Cerebras it means *emit a task and a color*. For Groq it means *choose a cycle*.

Our proposal's `spatial.place` is written assuming the AIE meaning. That is assumption #1
to test.

---

## 3. Axis B — Memory: everyone agrees, and it's the good news

**Every machine here is scratchpad-only. No caches. No coherence. No hardware prefetch.**
Address translation is either absent or trivial. This is the strongest point of agreement in
the whole landscape, and it is what makes a common abstraction plausible at all.

What differs is **capacity and depth**:

| Machine | Levels | Sizes |
|---|---|---|
| Cerebras | 1 | 48 KB/PE — that's it (weights stream in from off-wafer) |
| Groq | 1 | SRAM slices only, ~220 MB aggregate |
| Tenstorrent | 2 | 1.5 MB L1/core → GDDR6 over the NoC |
| AMD XDNA / Versal | **3** | 64 KB L1/tile → 512 KB memtile/column → DDR via shim |
| Graphcore | 2 | 624 KB/tile → streaming memory |

Two structural notes that matter more than the numbers:

1. **AIE's memtile is unusual.** A software-managed staging level *between* core-local memory
   and DRAM, with its own 4D-strided DMA engines. It is the thing AIEHalide exploits for
   [zero-cost layout transforms](../PACT/56.txt) — the optimization Reviewer A called the
   paper's most distinctive. **Tenstorrent has no equivalent level.** A mapping strategy
   built around memtile staging does not have a Tenstorrent analogue; it has to be
   *re-derived* as either L1 residency or a DRAM round-trip.
2. **AIE has neighbour-shared L1.** An AIE tile can directly address its immediate
   neighbours' data memory. Combined with the dedicated **cascade stream** between
   horizontally adjacent tiles, AIE has *three* mechanisms for getting a value from one
   compute unit to another (shared memory, cascade, DMA/stream) where Tenstorrent has *one*
   (NoC). See `../spatial-dsl/04-spatial-triton-dsl.md` on the "reuse trichotomy."

**Compiler consequence.** A 30× spread in local memory (48 KB → 1.5 MB) does not change the
*form* of the capacity constraint — AIEHalide's C1, "working set ≤ local memory," is
genuinely portable. But memory *hierarchy depth* is not a parameter you can just retune; it
changes which mappings exist. Log this in the [ledger's constraint table](01-abstraction-ledger.md#the-constraint-question-seeded-wk-1-answered-by-wk-11).

---

## 4. Axis C — Interconnect

| Machine | Fabric | Routing decided | Multicast |
|---|---|---|---|
| AIE | AXI-Stream switches, **circuit-switched** | compile time (routes are configured, then fixed) | yes, native |
| Versal | AIE stream + programmable logic + hard NoC | compile time, plus whatever you synthesize in PL | yes |
| Tenstorrent | two **packet-switched** 2D tori (opposite directions) | runtime, per transfer, by the issuing core | yes, native |
| Groq | fixed east–west dataflow through slices + switch slices | compile time, cycle-exact | via switch slices |
| Cerebras | 2D mesh, 5-port router/PE, 32-bit wavelets, "colors" | compile time (color→route tables) | yes, via fanout |

The AIE/Tenstorrent split is the one that matters for us, and it is sharper than it looks:

- **AIE routes are established at compile time and are a scarce, allocatable resource.**
  The mlir-aie router can *fail*. This is exactly Reviewer B's criticism: our autoscheduler
  "verifies channel budgets but cannot model physical wire routing." Congestion is a
  compile-time error, not a slowdown.
- **Tenstorrent NoC transfers are issued at runtime by a data-movement core** naming a
  destination coordinate. There is no route allocation to fail. Congestion shows up as
  *latency*, not as a build failure.

So "communication" is a compile-time resource-allocation problem on one machine and a runtime
performance problem on the other. Any IR that makes communication first class has to decide
which of those it is modelling — or model both.

---

## 5. Axis D — Synchronization and flow control ⭐

**This is the deepest divide in the landscape, and the one our proposal is least prepared
for.** There are four genuinely different execution contracts:

```
  static / deterministic                                   dynamic / flow-controlled
  ├────────────────┬──────────────────┬──────────────────┬──────────────────────────┤
  Groq             Cerebras           Graphcore          AMD AIE, Tenstorrent
  cycle-exact      data-triggered     BSP barriers       blocking acquire/release
  schedule         tasks              (compute|sync|     on bounded buffers
                                       exchange)
  no buffers       no blocking        global barrier     backpressure
  no stalls        consumer           between phases     deadlock is possible
  no deadlock      by design
```

- **AMD AIE & Tenstorrent — bounded buffers with backpressure.** ObjectFIFO `acquire` blocks
  when empty; `cb_wait_front` blocks when empty; producers block when full. This is
  Kahn/SDF semantics realized in hardware locks and semaphores. **Deadlock is a real failure
  mode**, which is exactly why proposal RQ5 (deadlock freedom, channel dependency graphs)
  makes sense — for these two machines.
- **Groq — no flow control at all.** Everything is scheduled to the cycle by the compiler.
  There is no arbitration, no queueing, no stall, and therefore *nothing to deadlock*.
  Latency is a compile-time constant. The compiler's job is not "size the buffers" but
  "prove the schedule."
- **Cerebras — data-triggered.** A task runs when a wavelet of its color arrives. There is
  no blocking consumer waiting on a buffer; there is an activation. Backpressure exists at
  the link level but the programming contract is closer to actors than to FIFOs.
- **Graphcore — BSP.** A hard global barrier separates compute from exchange. Deadlock is
  structurally impossible; the cost is that everyone waits for the slowest tile.

**Compiler consequence — and the headline finding of this document.** The proposal's
[§6.6 pipelining](../proposals/spatial_dsl_project_proposal.md#66-concurrency-and-pipelining)
and `spatial.stream` / `spatial.buffer double` are written in the vocabulary of *bounded
buffers with backpressure*. That vocabulary is shared by AMD AIE and Tenstorrent — our two
chosen targets — and by essentially nobody else on this page.

That is not fatal. It is a **scoping result**: it tells us the abstraction we are designing
is an abstraction over *flow-controlled spatial dataflow machines*, not over spatial
accelerators in general. State that in the paper deliberately, before a reviewer states it
for us. (Reviewer C already gestured at this with "Cerebras, Groq for example.")

---

## 6. Axis E — Who moves the data

| Machine | Mover | Programmed as |
|---|---|---|
| AIE | dedicated DMA engines in tiles / memtiles / shims | descriptors (4D strided), generated from ObjectFIFO |
| Tenstorrent | **two of the five RISC-V cores** | ordinary C++ kernels calling `noc_async_read/write` |
| Groq | the slice pipeline itself | nothing separate — movement *is* the schedule |
| Cerebras | the router | route/color configuration + `@bind_local_task` |

Tenstorrent's choice is unusual and pedagogically valuable: **data movement is ordinary
software running on ordinary cores.** There is no DMA descriptor format to lower to; there is
a program to emit. Meanwhile AIE's 4D DMA descriptors are expressive enough to *do work* —
they perform layout transposes for free, which is AIEHalide's signature optimization.

**Compiler consequence.** "Emit a transfer" means "fill in a strided descriptor" on one
machine and "generate a loop nest in C++" on the other. And a mapping that leans on AIE's
free descriptor-based relayout has no zero-cost Tenstorrent equivalent — on TT the same
relayout costs either compute (`tilize`) or NoC traffic. This is a concrete, testable
mapping-portability question, and it's a good candidate for the Week 8 §19 exercise.

---

## 7. Axis F — What the programmer is allowed to see

| Machine | Lowest exposed level | Can you place a kernel on a specific core? | Can you write the data movement? |
|---|---|---|---|
| AMD AIE | IRON / MLIR-AIE / ADF | yes, explicit coordinates | yes (ObjectFIFO, or raw DMA) |
| Versal | + programmable logic | yes | yes, and you can *build* the mover |
| Tenstorrent | TT-Metalium | yes, core ranges | yes, you write the reader/writer kernels |
| Cerebras | CSL | yes, per-PE | yes, colors and routes |
| Groq | Groq compiler (ONNX/PyTorch in) | **no** | **no** |

Groq is the outlier: the whole value proposition is that the compiler owns determinism, so
user-level spatial control would break the guarantee. **A Groq backend for Spatial IR is not
"more work" — it is category-incompatible with an expert-authored Level-2 surface.** If we
ever claim generality, say this explicitly.

---

## 8. Same GEMM, three machines

The most useful ten minutes of the session. All three express *the same mapping*: tile C
across a core grid, stream A tiles in from the west, B tiles from the north, accumulate
locally, double-buffer both inputs.

**AMD AIE (IRON, Python)** — declare the channels, the hardware is implied:

```python
of_a = ObjectFifo(a_tile_ty, depth=2)          # double buffering = an integer
of_b = ObjectFifo(b_tile_ty, depth=2)
of_c = ObjectFifo(c_tile_ty, depth=2)

def core_fn(a_in, b_in, c_out, matmul):
    elem_c = c_out.acquire(1)                   # blocks if full
    for _ in range(K_tiles):
        elem_a = a_in.acquire(1)                # blocks if empty
        elem_b = b_in.acquire(1)
        matmul(elem_a, elem_b, elem_c)          # hand-tuned microkernel
        a_in.release(1); b_in.release(1)
    c_out.release(1)

workers = [Worker(core_fn, [of_a.cons(), of_b.cons(), of_c.prod(), matmul_kernel])
           for _ in range(n_cores)]
```

**Tenstorrent (TT-Metalium, C++)** — the same mapping, but you write the movers by hand and
the host wires it up:

```cpp
// host
CreateCircularBuffer(program, core_grid, cb_a_config /* depth 2 */);
CreateKernel(program, "reader.cpp",  core_grid, DataMovementConfig{RISCV_0, NOC_0});
CreateKernel(program, "compute.cpp", core_grid, ComputeConfig{});
CreateKernel(program, "writer.cpp",  core_grid, DataMovementConfig{RISCV_1, NOC_1});

// reader.cpp  — runs on its own RISC-V, concurrently
for (uint32_t k = 0; k < Kt; ++k) {
    cb_reserve_back(cb_a, 1);
    noc_async_read(get_noc_addr(a_bank, a_off), get_write_ptr(cb_a), tile_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_a, 1);
}

// compute.cpp
for (uint32_t k = 0; k < Kt; ++k) {
    cb_wait_front(cb_a, 1); cb_wait_front(cb_b, 1);
    matmul_tiles(cb_a, cb_b, 0, 0, /*dst*/0, false);
    cb_pop_front(cb_a, 1); cb_pop_front(cb_b, 1);
}
```

**Groq** — there is no third listing. You hand it an ONNX graph. The mapping above is not
something you express; it is something the compiler decided, cycle by cycle.

### What to notice

- `depth=2` and `cb_..._config /* depth 2 */` are **the same decision**, and it is
  architecture-independent. Strong candidate for `spatial.buffer double`. ✅
- `acquire`/`release` and `cb_wait_front`/`cb_pop_front` are **the same semantics** —
  blocking on a bounded buffer. Strong candidate for `spatial.stream`. ✅
- But the *reader kernel exists only on Tenstorrent*. On AIE the DMA is inferred from the
  ObjectFIFO declaration. So `spatial.stream` lowers to a **declaration** on one machine and
  a **generated program** on the other. That's fine — it's exactly the "intent vs. mechanism"
  split the proposal wants. This example is the proposal's thesis working. ✅
- And `matmul_tiles` vs. the AIE microkernel are *not* the same thing: TT's matrix engine
  wants 32×32 tiles in a specific `tilize`d layout. Layout is not neutral. ⚠️

---

## 9. What actually generalizes

Provisional verdict, to be argued in session and then transcribed into
[the ledger](01-abstraction-ledger.md):

| Proposal §6 concept | Generalizes? | Why |
|---|---|---|
| `spatial.compute` | ✅ | every machine has "code on a unit" — but the unit's shape varies (1 program vs. 3) |
| `spatial.partition` | ✅ | pure logic; no machine disagrees |
| `spatial.place` | ✅ with care | grids everywhere; Groq has no notion of it |
| `spatial.buffer` (local, depth) | ✅ | scratchpad-only is universal; **hierarchy depth is not** (AIE 3 levels, TT 2, Cerebras 1) |
| `spatial.stream` | ⚠️ AIE + TT only | assumes bounded buffer + backpressure. Groq: no buffers. Cerebras: triggers, not blocking. |
| `spatial.multicast` | ✅ | native on AIE, TT, Cerebras |
| `spatial.pipeline` | ⚠️ AIE + TT only | same assumption as `stream` |
| `spatial.replicate` | ✅ | logical |
| *(deadlock analysis, RQ5)* | ⚠️ AIE + TT only | meaningless where there is no blocking |

**The honest framing for the paper:** our abstraction targets *flow-controlled spatial
dataflow machines with software-managed scratchpads*. AMD AIE and Tenstorrent are two
substantially different members of that class — different core structure, different
interconnect discipline, different memory depth, different data-movement programming model —
which makes them a legitimate portability test. Groq and Cerebras sit outside the class, and
saying so precisely is a contribution, not a limitation.

---

## 10. Discussion questions for the session

1. AIE has three ways to get a value between compute units (shared L1, cascade, stream/DMA);
   Tenstorrent has one. Does `spatial.stream` therefore *under*-specify AIE mappings — and
   if we add `spatial.cascade`, have we already lost portability?
2. AIEHalide's zero-cost relayout depends on 4D memtile DMA descriptors. What is the honest
   Tenstorrent story: `tilize` on the compute engine, or an extra NoC round-trip? Which
   would our cost model even be able to express?
3. Is the memtile a *level* (parameter: hierarchy depth) or a *mechanism* (leakage)? The
   answer determines whether `spatial.buffer` needs a memory-space parameter.
4. Reviewer C asked about Cerebras and Groq. Given §5, what is the strongest *true* sentence
   we can write about generalizing to them?
5. Tenstorrent forces a reader/compute/writer split; AIE does not. Is "the number of
   independent programs per compute unit" a property Spatial IR must model, or one a backend
   can invent?
6. Deadlock freedom (RQ5) is only meaningful for two of six machines here. Does that make it
   a *narrower* contribution, or a *sharper* one?

---

## 11. Numbers to verify before you present

Do not present the §1 table without checking these; several are generation-dependent and I
have given them from memory:

- XDNA1 (Phoenix) and XDNA2 (Strix) tile-grid geometry and TOPS.
- AIE1 vs AIE-ML per-tile data memory, vector width, and MAC counts; memtile capacity.
- Whether AIE-ML retains the cascade stream and neighbour-shared L1 in the same form as AIE1.
- Tenstorrent Wormhole vs. Blackhole Tensix counts (harvesting makes the usable count differ
  from the physical grid) and L1 per core.
- Groq TSP SRAM capacity and superlane/slice organization.
- Cerebras WSE-2 vs WSE-3 core counts, per-PE SRAM, total SRAM.
- Graphcore Bow/IPU tile count and per-tile memory.

Primary sources: AMD AIE Architecture Manuals (AM020/AM009) and the mlir-aie device models;
Tenstorrent `tt-metal` docs and the Tensix architecture pages; Groq's ISCA'20 "Think Fast: A
Tensor Streaming Processor" paper; Cerebras SDK/CSL documentation and the WSE hot-chips
talks.

---

## 12. Further reading

- **Groq**, "Think Fast: A Tensor Streaming Processor (TSP)," *ISCA* 2020 — the best written
  account of a fully deterministic accelerator. Read §II if nothing else.
- **Koeplinger et al.**, "Spatial," *PLDI* 2018 — and Plasticine (*ISCA* 2017) for the
  PCU/PMU model behind SambaNova.
- **Jia et al.**, "Dissecting the Graphcore IPU Architecture," arXiv 2019 — for the BSP
  contrast.
- **Cerebras**, "Wafer-Scale Deep Learning" (Hot Chips) + the CSL SDK docs.
- In this repo: 📄 `../papers/iron.pdf`, 📄 `../papers/aries.pdf`, 📄 `../papers/charm.pdf`,
  📄 `../papers/systolic-aie.pdf`, and `../spatial-dsl/04-spatial-triton-dsl.md` for the
  reuse trichotomy.
