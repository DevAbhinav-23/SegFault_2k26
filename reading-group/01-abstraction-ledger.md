# The Abstraction Ledger

One row per system read in the [reading group](00-syllabus.md). Filled **live**, in the last
20 minutes of each session, by the whole group.

## The rule

> A construct may be entered in the **Spatial** column only if someone in the room can state,
> out loud, how it is realized on **both** AMD AIE **and** Tenstorrent.
> If nobody can, it goes in **Leakage**.

This is proposal §14.1's test ("Can this operation be interpreted naturally on both AMD and
Tenstorrent?"), executed 16 times with evidence. The union of the Spatial column at the end
of Week 11 is the **first draft of the Spatial IR op set**; the Leakage column is the answer
to **RQ3**.

## Ledger

| Wk | System | Unit of work | Domain-level constructs | Spatial (arch-independent) | Leakage (target-specific) | Escape hatch? | Verdict for Spatial IR |
|---|---|---|---|---|---|---|---|
| 1 | **AIEHalide** (PACT'26 — ours) | Halide `Func` → AIE tile | Halide pipeline | reconstructed producer–consumer DAG, tile geometry, buffer depth, fusion choice | L1/L2 capacity C1, 6-DMA-channel C5, 4D MemTile descriptors, SHIM column penalty, vector alignment C2/C3 | `aie_dataflow`, `aie_fuse_with`, `aie_kernel` | *the baseline: which of C1–C6 are AMD facts vs. spatial facts?* |
| 2 | Halide | `Func` / loop nest | `Func`, `Var`, pure defs, RDom | `split`/`tile`, `compute_at`/`store_at`, `reorder` | `gpu_blocks`, `hexagon`, target-specific vector widths | `define_extern` | *note where the DAG dissolves (A-Q1)* |
| 3 | IRON / MLIR-AIE | AIE tile + ObjectFIFO | — (it is Level 3) | ObjectFIFO as producer/consumer channel; `link`; tile placement | shim tiles, mem tiles, lock counts, AIE2 vs AIE2p, cascade | inline `aie.core` C++ | |
| 4 | MLIR | op / region | — | — (infrastructure, not a mapping abstraction) | — | — | *methodology row: what is a good dialect boundary?* |
| 5 | Interstellar | Halide schedule | layer | schedule ⇒ dataflow (WS/OS/RS) classification | — (it designs hardware; ours is fixed) | — | *validates the Level-2 thesis* |
| 6 | Halide autoscheduler (Adams'19) | schedule | pipeline | search space: tiling × fusion × vectorization | learned cost model features are CPU-shaped | manual schedule | |
| 6 | Ansor / TVM | tensor expr stage | ops | sketch + annotation search space | target strings, tensorize intrinsics | `tensorize`, extern | |
| 7 | TileLoom | tile config | op | analytical rank → measured top-k | — | — | *hybrid ranking = opening #2* |
| 7 | Timeloop | mapping over loop nest | layer | mapspace = loop order × tiling × spatial split | arch template YAML | — | |
| 7 | MAESTRO | data-centric directive | layer | temporal/spatial map, cluster, dataflow-as-reuse-spec | — (analytical) | — | *cost-model vocabulary* |
| 8 | T2S-Tensor | uniform recurrence (URE) | UREs (functional spec) | space-time transform, `isolate`, `scatter`/`gather`, `buffer` | Intel FPGA channels, OpenCL kernels | | |
| 8 | SuSy | URE, Halide-hosted | UREs | `space_time_transform` | FPGA / OpenCL | | |
| 9 | Quinton / systolic synthesis | index point | uniform recurrence eqns | σ (time), π (space), projection direction | — (pre-architecture) | — | *the formal core* |
| 9 | Pluto / polyhedral | statement instance | loop nest | affine schedule, tiling hyperplanes, fusion choice | — | — | |
| 10 | TT-Metalium / tt-mlir | Tensix core + kernel triple | — | circular buffer, NoC transfer, multicast, core range | reader/compute/writer RISC-V split, semaphores, L1 alloc, `tilize` | raw kernel C++ | *the row that decides the whole project* |
| 11 | ARIES | tile (task) | `task` bodies | task graph, tile placement grid, buffer decls | AIE IP selection, ADF-style graph | HLS C++ | |
| 11 | Allo | schedule primitive on op | ops | `.split`, `.buffer_at`, `.to()` (dataflow), composability | AIE / HLS pragmas | | |
| 11 | CHARM | accelerator (CU) | ops in a DNN graph | accelerator partitioning, off-chip↔on-chip planning | Versal AIE geometry, PLIO | | |
| 12 | SDF / KPN | actor + channel | — | token rates, bounded-buffer schedulability, deadlock condition | — | — | *RQ5's foundation* |
| 13 | mlir-aie router / NoC placement | route | — | channel-dependency graph, congestion | switchbox capacity, physical wire counts | — | *opening #1* |
| 14 | Stream / FLAT / Chimera | fused layer group | layer graph | inter-layer tiling, on-chip residency of intermediates | — | — | *opening #3* |
| 15 | AIEVec / microkernels | vector op | — | register tiling | AIE intrinsics, `exp`/reciprocal lowering, shape regularity | hand-written kernel | *opening #4; probably 100 % leakage — that is a finding* |

> Pre-seeded cells are **hypotheses to argue with**, not answers. Strike and rewrite them in
> session — a row that survives unedited means nobody read carefully.

## Derived artifacts

### Candidate Spatial IR op set (revisit Wk 8, 10, 11)
Compare against proposal §6 and §10 Phase 2. Track which proposed ops survive contact:

| Proposed op | Evidence from | AMD realization | TT realization | Status |
|---|---|---|---|---|
| `spatial.compute` | | AIE core body | compute kernel | |
| `spatial.partition` | | | | |
| `spatial.place` | | tile coords | core range | |
| `spatial.buffer` | | tile L1 / mem tile | L1 circular buffer | |
| `spatial.stream` | | ObjectFIFO | CB + NoC | |
| `spatial.multicast` | | ObjectFIFO broadcast | NoC multicast | |
| `spatial.pipeline` | | | | |
| `spatial.replicate` | | | | |

### The constraint question (seeded Wk 1, answered by Wk 11)
AIEHalide's C1–C6 are stated as AMD facts. Which are actually *spatial* facts with different
constants on every machine? This table alone may be a paper section.

| Constraint | AIEHalide form | Spatial-level statement? | TT form | Verdict |
|---|---|---|---|---|
| C1 local capacity | L1 ≤ 64 KB minus stack | "working set ≤ local memory" | L1 ≤ 1.5 MB/core | likely spatial, parameterized |
| C2/C3 alignment & divisibility | vector-width alignment | ? | tile granularity (32×32) | |
| C5 channel budget | ≤ 6 DMA channels per MemTile column | "≤ N channels per memory agent" | NoC ports per core | |
| C6 working-set | per-tile region from bounds inference | frontend-derived | same | likely domain-level |
| SHIM penalty | 3 %/column beyond two | ? | — | likely AMD-specific |

### Constructs we've decided *cannot* generalize (→ RQ3 answer)

- …

### Constructs nobody else has (→ novelty claim)

- …
