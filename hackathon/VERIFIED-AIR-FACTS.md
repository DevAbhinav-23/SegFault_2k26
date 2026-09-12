# Verified MLIR-AIR facts (source-level, pinned to a commit)

*Produced 2026-09-12 by the verifier pass over `01-paradigm-comparison.md` r3 and
`HANDOFF.md`. Every fact below is `path:line` at the commit named here.*

**Two things changed mid-investigation and they change what this document is.**
First, **a pip-only install works** — `air-opt`, `aircc` and `import air` are running on
this machine in under six minutes, from prebuilt wheels, with no NPU, no XRT, no Vitis and
no source build (§G). Second, the published wheel is built from **exactly the commit under
review** (`mlir_air 0.0.1.2026091204+ff95a9b`). So E3 and E4 were not read about — **they
were run**, and their results are marked *"measured"* below. Nothing was run on hardware;
no claim here depends on a device.

| | |
|---|---|
| **Repo** | `https://github.com/Xilinx/mlir-air` |
| **Clone path** | `/home/adi/Projects/Honours/mlir-air` (sibling of our repo, outside it) |
| **Commit** | `ff95a9b35b692b4cfbdf1d52bf69e6ffe01facde` |
| **Commit date** | 2026-09-11 20:58:31 +0000 |
| **Subject** | `[AIR][llms] Full-ELF fused decode with a runtime context length, and one configuration per path (#1986)` |
| **Clone kind** | `--depth 50` (shallow; sufficient — every citation is at HEAD) |
| **mlir-aie** | **not vendored, not a submodule, not present.** Pinned out-of-tree by `utils/clone-mlir-aie.sh:17` → `HASH=10767b50e5beec9b2ec97ce92298e3507a33490c`, `WHEEL_VERSION=1.4.3.dev55+g10767b5` |

All paths below are relative to the clone path unless written absolute.

---

## A. `air_ChannelOp` argument list, `channel_type` values, `broadcast_shape`

**Verdict: TRUE, with one material correction.**

### A.1 — no `depth` argument: TRUE

`mlir/include/air/Dialect/AIR/AIR.td:620-623`:

```tablegen
def air_ChannelOp : air_Op<"channel", [Symbol]>,
    Arguments<(ins SymbolNameAttr:$sym_name,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$size,
                   DefaultValuedAttr<StrAttr, "\"npu_dma_stream\"">:$channel_type)> {
```

`grep -c -i depth mlir/include/air/Dialect/AIR/AIR.td` → **0**. The argument list is
exactly `$sym_name`, `$size`, `$channel_type`, as r3 said.

### A.2 — but there IS a depth knob, spelled `buffer_resources`: **the document is wrong that nothing reads it**

`mlir/include/air/Dialect/AIR/AIR.td:738-743`:

```cpp
    int getBufferResources() {
      if(auto attr = getOperation()->getAttrOfType<IntegerAttr>("buffer_resources"))
        return llvm::dyn_cast<IntegerAttr>(attr).getInt();
      else
        return 1;
```

It is a *discardable* attribute (not in the `Arguments<>` list, so `grep depth` misses
it and `ChannelOp::verify()` never checks it), and it is consumed by the lowering —
see §C. It is settable today from `air.dialects` (**not** from `air.api`):
`python/air/dialects/_air_ops_ext.py:195` (`buffer_resources: Optional[Union[int, IntegerAttr]] = None`) and
`python/air/dialects/_air_ops_ext.py:204-205`:

```python
        # Optional objectFIFO depth knob: the `buffer_resources` attribute is
        # consumed by AIRToAIEPass (AIRToAIEPass.cpp, ChannelOp::getBufferResources)
```

Round-trip test: `python/test/dialect/channel_buffer_resources.py:25` →
`# CHECK: air.channel @chan_depth2 [] {buffer_resources = 2 : i64}`.

### A.3 — `channel_type` values: TRUE, five of them

`mlir/lib/Dialect/AIR/IR/AIRDialect.cpp:3767-3771` (`air::ChannelOp::verify()`):

```cpp
  if (chanType != "npu_dma_stream" && chanType != "npu_dma_packet" &&
      chanType != "npu_cascade" && chanType != "npu_mmio" &&
      chanType != "gpu_symmetric_heap")
```

Exactly as r3 lists them. Default `"npu_dma_stream"` (AIR.td:622).

### A.4 — `broadcast_shape`: TRUE, NumPy rules, verifier-enforced

`mlir/include/air/Dialect/AIR/AIR.td:668-670`:

```
    If a channel broadcasts to multiple destinations, the optional `broadcast_shape` attribute
    annotates the output sizes after broadcasting. Broadcasting follows NumPy's broadcasting rules.
```

`ChannelOp::verify()` (`AIRDialect.cpp:3777-3806`) checks rank equality, `bcastVal >= sizeVal`
(*"broadcasting cannot shrink a dimension"*) and `sizeVal == 1 || sizeVal == bcastVal`
(*"(NumPy broadcasting rules)"*). Two further attributes r3 did not list are also
verified there: `refeed_count` (`AIRDialect.cpp:3812`) and `packet_ids` (`:3818-3839`,
each id in `[0,31]`, unique, `npu_dma_packet` only).

> **Design consequence.** Our emitter must spell buffer depth `buffer_resources` on the
> `air.channel` op, not `depth` — and it can do so through `air.dialects.air.Channel`,
> not through `air.api`. A surface `depth=N` therefore has a *real* lowering (§C), which
> removes B15's "hint whose realisation is unverified". `broadcast_shape` is verifier-checked,
> so a bad fan-out is caught by `air-opt` with a named diagnostic; our checker need not
> re-derive NumPy broadcast rules.

---

## B. Is anything enforcing put/get balance or channel-graph acyclicity?

**Verdict: FALSE — nothing enforces it. But the repo contains an unimplemented *plan* for
exactly that checker, which the design document did not know about.**

### B.1 — the grep

`grep -rn -i "balance\|acyclic\|deadlock" mlir/lib mlir/include python` returns **57 hits**,
and **every single one is a comment, a diagnostic string, or an unrelated identifier.**
No hit is a balance or acyclicity *check*. Representative classes:

* Lock-protocol comments in the AIE lowering (37 hits across
  `mlir/lib/Conversion/AIRToAIEPass.cpp`, `AIRToAIESchedulingUtils.cpp`, `AIRRtToNpuPass.cpp`),
  all about avoiding *hardware lock* deadlock, e.g. `AIRToAIESchedulingUtils.cpp:682`:
  ```cpp
  // restore cap=N). Default init=1 would deadlock the first writer's acq>=N.
  ```
* `mlir/include/air/Util/DirectedAdjacencyMap.h:20` — *"The class implements a directed acyclic graph"* — a data-structure docstring, not a check.
* `python/air/api/_loop.py:42` and `_index.py:64-65` — a trace-time refusal of a *coordinate-dependent trip count*, which is a different property: `_loop.py:42`
  ```python
  anything with a channel operation in it would deadlock on the cores that run
  ```
* `mlir/lib/Util/Runner/RunnerNode.cpp:1447` — the **simulator**, not a checker:
  ```cpp
    // satisfy this, so an unbalanced put/get pair still stalls instead of
  ```

### B.2 — the three verifiers, read in full

* **`air::ChannelOp::verify()`** — `mlir/lib/Dialect/AIR/IR/AIRDialect.cpp:3762-3841`.
  Checks: `channel_type` allow-list; NumPy `broadcast_shape` rules; `refeed_count`;
  `packet_ids` (range, uniqueness, packet-only). **No put/get counting. No graph.**
  **It does not even validate `buffer_resources`.**
* **`air::ChannelPutOp::verify()`** — `AIRDialect.cpp:3579-3644`. Checks: src sizes/strides
  rank; `refeed_count`; bundle indices not `scf.for` IVs; `npu_mmio` src must be L3;
  `gpu_symmetric_heap` needs an enclosing `air.rank`; `pad_before`/`pad_after` agreement,
  non-negativity and `<= 65535`. Verbatim, `AIRDialect.cpp:3589-3592`:
  ```cpp
      return emitOpError()
             << "channel index " << i
             << " is an scf.for induction variable; channel bundle indices "
                "must not be temporal scf.for induction variables";
  ```
* **`air::ChannelGetOp::verify()`** — `AIRDialect.cpp:3690-3747`. The mirror image
  (dst rank; `npu_mmio` dst must be **L1**, `memory_space=2`, `:3711-3717`).

**None of the three counts puts against gets or inspects a graph.** r3's statement stands.

### B.3 — the passes whose names contain "channel" or "verify"

`mlir/include/air/Transform/Passes.td` and `Conversion/Passes.td`, complete list:

| Pass | Line | What it is |
|---|---|---|
| `air-broadcast-detection` | Transform/Passes.td:870 | derives multicast from IV dependence |
| `air-specialize-channel-wrap-and-stride` | :994 | BD shaping |
| `air-unroll-channel-by-factor` | :1008 | unrolling |
| `air-fuse-channels` | :1027 | merges channels |
| `air-enforce-channel-fifo-order` | :1186 | **ordering repair, not a check** |
| `air-verify-hierarchy-locality` | :1344 | **race check on hierarchy operands** |
| `air-specialize-dma-broadcast` | :1427 | broadcast lowering |
| `air-label-broadcast-channel-with-tile` | :1490 | labelling |
| `air-dma-to-channel` | :1727 | `air.dma_memcpy_nd` → `air.channel` |
| the four ping-pong passes | :926, :939, :950, :966 | see §E |

`air-enforce-channel-fifo-order`, verbatim (`Transform/Passes.td:1189-1191`):

```
    A single air.channel is an ordered FIFO, but air-dependency only orders
    channel ops that share a buffer. Ops on the same channel + same indices +
    same direction (put/put or get/get) that touch different buffers (e.g. two
```

and its own scope caveat, `Transform/Passes.td:1199-1201`:

```
    NOTE: it only orders same-block ops (post-fusion collapsed channel ops); it
    does not order ops still nested in distinct loops (those would need
    loop-carried deps, which canonicalize strips).
```

`air-verify-hierarchy-locality`, verbatim (`Transform/Passes.td:1350-1352`):

```
    Statically detect data races on memrefs that an air.launch /
    air.segment / air.herd passes to itself as kernel operands.
```

It is read-only, has a `strict` option, and **does run by default** — `tools/aircc/aircc.cpp:1213-1217`:

```cpp
  if (placedIrVerifiers != PIV_off) {
    std::string strict = (placedIrVerifiers == PIV_error) ? "true" : "false";
    std::string pipeline =
        "builtin.module(air-verify-hierarchy-locality{strict=" + strict + "})";
```

with `cl::init(PIV_error)` at `aircc.cpp:264`. So **strict locality verification is on by
default in `aircc`.** So is `air-enforce-channel-fifo-order` (`aircc.cpp:1169`).

### B.4 — the finding the document did not have: `docs/AIRCorrectnessChecker.md`

The repo ships a 511-line **implementation plan** for precisely the missing checker.
`docs/AIRCorrectnessChecker.md:1-8`:

```
# AIR Program Correctness Checker — Implementation Plan

This document specifies the design and implementation plan for a static
correctness checker for AIR programs.
```

Its property table (`:15-20`) is P1 channel balance, P2 deadlock freedom, P3 resource
constraints, P4 token-constraint consistency, and it names the files to create
(`mlir/lib/Analysis/ChannelBalance.cpp`, `DeadlockAnalysis.cpp`, …) and the umbrella pass
`-air-verify-program`.

**None of it exists.** `mlir/lib/Analysis/` and `mlir/include/air/Analysis/` do not exist
(`ls` fails). `grep -rn 'air-verify-program\|AIRVerifyProgram\|ChannelBalance' mlir python test tools`
returns **zero hits**. It is a design document, not code.

### B.5 — what the compiler does instead of rejecting: it *repairs*

`mlir/lib/Conversion/AIRToAIEPass.cpp:4484-4491`:

```cpp
  /// Insert dummy air.channel.put or air.channel.get operations for L2 memrefs
  /// to ensure that the number of puts and gets match for each buffer.
  /// This helps prevent the risk if a race condition due to imbalanced lock
```

and `AIRToAIEPass.cpp:4530` — `// Balance puts and gets by inserting dummy ops`,
`:4781` — `// Generate dummy air.channel ops to balance the number of BDs at either`.

### B.6 — E3 RUN, not reasoned: the three violations through the real `air-opt`

Three hand-written modules, scratchpad only, run through the installed `air-opt`
(`mlir_air 0.0.1.2026091204+ff95a9b`). Sources: `air-probe/e3_i.mlir`, `e3_ii.mlir`,
`e3_iii.mlir`.

| violation | plain parse+verify | `-air-dependency -air-dependency-canonicalize` | through `air-to-aie{device=npu1}` |
|---|---|---|---|
| **(i)** one `put`, no `get`, same index | exit 0, silent | **error printed**, `exit 0` | exit 0, silent |
| **(ii)** loop body, two `put`s vs one `get` | exit 0, silent | exit 0, silent | **exit 0, silent** |
| **(iii)** two herds, `get`-before-`put` cycle | exit 0, silent | exit 0, silent | **exit 0, silent** |

(iii) was additionally run through `-air-enforce-channel-fifo-order
-air-verify-hierarchy-locality`: **exit 0, silent**.

**The one diagnostic that exists**, for (i) only:

```
e3_i.mlir:9:7: error: 'air.channel.put' op found channel op not in pairs
```

Its source is **not a checker** — `mlir/lib/Util/Dependency.cpp:2063-2066`:

```cpp
    std::vector<air::ChannelGetOp> channel_gets =
        getTheOtherChannelOpThroughSymbol(channel_put);
    if (!channel_gets.size()) {
      op->emitOpError("found channel op not in pairs");
```

It tests `size() == 0`, i.e. *"does at least one counterpart exist anywhere in the module"* —
not counts, not per path, not per iteration. And **it does not signal pass failure**: the
process exits 0 and prints the module. The repo's own test says what it is for —
`mlir/test/Transform/AIRDependency/channel_unpaired.mlir:8-9`:

```
// Verify that air-dependency-canonicalize does not crash when a channel op
// has no matching put/get counterpart (e.g. channel.get without channel.put).
```

It is a **crash guard**, added because the empty vector used to assert.

> **Conclusion for E3 — measured, not inferred.**
> **(i) unmatched put** — *not rejected.* A diagnostic is printed by
> `air-dependency-canonicalize`, but `air-opt` **exits 0** and compilation continues to
> `air-to-aie`. It only checks existence of one counterpart. For L2 memrefs the lowering
> additionally **inserts a dummy get** to rebalance (`AIRToAIEPass.cpp:4484-4553`).
> **(ii) per-iteration imbalance** — *not rejected, not even warned.* Silent through
> `air-to-aie`.
> **(iii) get-before-put cycle across two herds** — *not rejected, not even warned.* Silent
> through `air-to-aie`, `air-enforce-channel-fifo-order` and
> `air-verify-hierarchy-locality`. No acyclicity analysis exists anywhere.
>
> The only channel legality actually enforced at IR level is *a bundle index that is a
> temporal `scf.for` IV* (`ChannelPutOp::verify`/`ChannelGetOp::verify`), and at trace level
> a coordinate-dependent trip count (`python/air/api/_loop.py:140`).
>
> **Design consequence.** §6.8's rival ("ship the missing checker") is confirmed *necessary*
> and simultaneously **loses its novelty framing**: AMD has already written the spec for it
> in their own repo (`docs/AIRCorrectnessChecker.md`) and has not implemented it. Our checker
> must be built, and the pitch cannot say the idea is ours. Our emitter cannot rely on
> `air-opt` rejecting any of (i)–(iii), and **must not read `air-opt`'s exit code as a
> verdict** — it returns 0 while printing an error.

---

## C. How channels lower — and the E1 verdict

**Verdict: PARTIAL-verified statically, and it comes out on the *stall-rule* side: a put into
an empty depth-1 channel acquires a free slot immediately. It does **not** rendezvous.**

### C.1 — two lowering paths, and the default is NOT ObjectFIFO

`mlir/include/air/Conversion/Passes.td:197-200`:

```tablegen
    Option<"clUseObjFifo", "use-objectfifo", "bool",
           /*default=*/"false",
           "Choose whether to lower data movement ops to aie.objectFifo, or "
           "directly to aie.locks.">,
```

`aircc` never sets it (confirmed: `generate-shim-dma`, `emit-herd-lock`, `use-objectfifo`
and `test-patterns` are absent from the `air-to-aie{...}` string built at
`tools/aircc/aircc.cpp:1230-1262`). **So the production path is locks + DMA BDs.**

### C.2 — ObjectFIFO path: depth comes from `buffer_resources`, default 1

`mlir/lib/Conversion/AIRToAIEPass.cpp:3095-3097`:

```cpp
    AIE::ObjectFifoCreateOp objFifo = createObjectFifo(
        rewriter, datatype, producerTile, consumers,
        channel.getBufferResources(), "air_" + channel.getName().str());
```

`createObjectFifo`'s `int depth` becomes the fifo's depth attribute
(`AIRToAIEPass.cpp:3204-3219`). With no attribute, `getBufferResources()` returns **1**
(AIR.td:742). Lit test `mlir/test/Conversion/AIRToAIE/air_channel_to_objectfifo_buffer_resources.mlir:12-13`:

```
// CHECK-DAG:   aie.objectfifo @[[VAL_2:.*]](%[[VAL_0]], {%[[VAL_1]]}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
// CHECK-DAG:   aie.objectfifo @[[VAL_3:.*]](%[[VAL_0]], {%[[VAL_1]]}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
```

(`@channel_1 [1,1] {buffer_resources = 2}` → depth 2; `@channel_0` with no attribute → depth 1.)

The emitted core code, same file `:22-27`, is **acquire-free-slot, not rendezvous**:

```
// CHECK:   %[[VAL_9:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:     %[[VAL_10:.*]] = aie.objectfifo.acquire @[[VAL_3]](Produce, 1) : memref<32xi32>
// CHECK:     %[[VAL_12:.*]] = aie.objectfifo.acquire @[[VAL_2]](Produce, 1) : memref<32xi32>
```

A producer issues `acquire(Produce, 1)` — it asks the fifo for *an empty buffer*, not for a
consumer. On a fresh depth-1 fifo the single buffer is free, so the first acquire succeeds.

### C.3 — default (lock) path: the producer's lock is initialised to the slot count

`mlir/lib/Conversion/AIRToAIESchedulingUtils.cpp:1230-1246`:

```cpp
  auto rlock = allocateLockOp(device, tile, 0);
```
```cpp
  auto wlock = UsesSemaphoreLocks
                   ? allocateLockOp(device, tile, static_cast<int>(wlockInit))
                   : rlock;
```

with `init = std::max(init_pair.first, init_pair.second)` from `getLockValuePair`
(`:1206-1212`), which returns at minimum `(1,1)` (`AIRToAIESchedulingUtils.cpp:502-503`:
`if (!read_counter || !write_counter) return std::make_pair(1, 1);`).

So the **read/full lock starts at 0** and the **write/empty lock starts at the number of
slots (≥ 1)**. A producer BD acquires the *empty* lock — which is already signalled — writes,
and releases the *full* lock. **The first put does not wait for a get.**

### C.4 — mlir-aie ObjectFIFO acquire/release semantics

**UNVERIFIED half.** mlir-aie is not present in this tree (§header). To close it, read, at
mlir-aie commit `10767b50e5beec9b2ec97ce92298e3507a33490c`:

* `include/aie/Dialect/AIE/IR/AIEObjectFifoOps.td` (or `AIE.td`) — `ObjectFifoAcquireOp` /
  `ObjectFifoReleaseOp` descriptions and the `Produce`/`Consume` port semantics;
* `lib/Dialect/AIE/Transforms/AIEObjectFifoStatefulTransform.cpp` — the lock lowering;
* `docs/AIEDesignPatterns.md` / the Object FIFO section of the programming guide.

### C.5 — the decisive empirical evidence already in-tree

`programming_examples/channel_examples/worker_to_worker/worker_to_worker.py` is **exactly the
E1 shape**: every core in a herd puts to its successor *before* getting from its predecessor,
on a depth-1 (no `buffer_resources`) channel bundle, different buffers, same sequential body
(`worker_to_worker.py:116-131`):

```python
                            chan_in.get(tile_in, indices=[th, tw])
                            tile_out[:] = tile_in[:] + tile_num
                            ring.put(tile_out, indices=[th_next, tw_next])
```
```python
                            ring.get(tile_in2, indices=[th, tw])
```

It is a **hardware-run CI test** — `worker_to_worker/run_makefile_peano.lit:4-9`:

```
// REQUIRES: ryzen_ai, peano
```
```
// RUN: make -f %S/Makefile run PEANO_INSTALL_DIR=%PEANO_INSTALL_DIR | FileCheck %s
// CHECK: PASS!
```

A full ring of N producers all putting before any gets, checked for numerical correctness on
a real NPU, upstream.

### C.6 — the compute-model "contradiction", resolved

`docs/AIRComputeModel.md:773-774` (the stall rule) and `:808-816` (the minimal-deadlock
example) are **not in contradiction** once read against the code. The minimal-deadlock
example puts and gets **on the same channel in the same sequential scope**, which the same
page separately forbids (`AIRComputeModel.md:823-825`: *"`put` and `get` on the same channel
must appear in different herds, different async branches, or different segment instances"*).
Its inline comment `// put blocks: channel is at capacity` is simply **wrong about the reason**
(0 unread transfers is not capacity on a depth-1 channel); what actually blocks there is that
a synchronous put cannot *complete* with no consumer. The stall rule is the one the lowering
implements.

Also note: **`docs/AIRComputeModel.md` uses an attribute name the dialect does not have.**
`AIRComputeModel.md:692`:

```mlir
air.channel @name [dim₀, dim₁, …] {channel_type = "npu_dma_stream", depth = <N>}
```

The real spelling is `buffer_resources` (§A.2). The compute-model page is out of sync with
`AIR.td`.

### C.7 — measured: the emitted locks on a real `air.api` GEMM

Running the §E.5 GEMM through `air-to-aie{device=npu1}` with the installed toolchain and
histogramming the emitted `aie.lock` initialisers:

```
aie.lock(%tile_0_2, 5) {init = 2 : i32}
aie.lock(%tile_0_2, 4) {init = 0 : i32}
aie.lock(%tile_0_2, 1) {init = 1 : i32}
```
```
--- lock inits histogram ---   12 × init = 0    4 × init = 1    8 × init = 2
```

Every lock is either `init = 0` (a *full* lock — the consumer waits) or `init = 1`/`init = 2`
(an *empty* lock — the producer's, pre-signalled with as many slots as the buffer has).
**No producer lock is ever initialised to 0.** That is the rendezvous question answered by
measurement: a producer never waits for a consumer to *start*; it waits only when all slots
are full.

> **E1 verdict: SAFE.** Four independent pieces of evidence agree: (1) **measured** — every
> emitted producer lock has `init >= 1`, never 0 (§C.7); (2) the lock-allocation source says
> so by construction (`AIRToAIESchedulingUtils.cpp:1230-1246`, `:502-503`); (3) the ObjectFIFO
> path emits `aie.objectfifo.acquire(Produce, 1)`, an ask for an empty buffer; (4)
> `worker_to_worker` is a hardware-CI-run ring of exactly this protocol. It is not
> *undecidable from source*, and it does not deadlock.
> The residual risk is not the semantics but the *scale*: `worker_to_worker` has one put and
> one get per core per herd entry; a T-timestep halo loop has 2T of each, and the failure mode
> there would be BD/lock capacity, not a rendezvous.
>
> **Design consequence.** §3.2's put-before-get halo protocol is buildable as written.
> `sp.exchange` / `s.exchange` survive. A surface `depth=` is real and must be emitted as
> `buffer_resources` on the channel declaration.

---

## D. Everything `air.api` can express — and it is a great deal more than §1.4 says

**Verdict: the design document materially understates `air.api`. This is the most consequential
section of this report.**

Twelve files, 9 927 lines total (`wc -l python/air/api/*.py`):
`_channel.py` 597, `_compile.py` 402, `_cond.py` 526, `_emit.py` 1 477, `_extern.py` 565,
`_index.py` 362, `__init__.py` 233, `_loop.py` 360, `ops.py` 1 735, `_trace.py` 2 147,
`types.py` 292, `_value.py` 1 231.

### D.1 — `launch` / `segment` / `herd`

* `air.launch(grid=None, name="kernel", target=None, repeat=None, attrs=None)` —
  `python/air/api/_compile.py:44`. **No rank cap**; `_compile.py:51-54`:
  ```python
        # No rank cap: air.launch's sizes are Variadic<Index>, so the op takes
        # as many axes as it is given. This carried a limit of two, then of
        # three, and both were the DSL inventing a constraint the dialect does
  ```
  `repeat=` wraps the launch in an `scf.for` and hands the body a dispatch index.
* `air.segment(grid=None, name=None)` — `_trace.py:1201`. L2 (memtile) scope; a segment
  grid **is** the launch grid, one segment instance per point (`__init__.py` docstring).
* `air.herd(iterable, name=, shape=, target=, link_with=, at=, params=)` — `_trace.py:1760`.
  * **1-D or 2-D only** — `_trace.py:1288-1291`:
    ```python
            if len(self.dims) > 2:
                raise NotImplementedError(
                    f"air.api supports 1-D and 2-D herd grids; got {len(self.dims)}-D"
    ```
  * **Physical sizes are capped per target** — `_trace.py:88-91`:
    ```python
    PHYSICAL_HERD = {
        "npu1": {1: (4,), 2: (1, 4)},
        "npu2": {1: (8,), 2: (2, 4)},
    }
    ```
    A *logical* grid larger than that is **strip-mined**: `self.repeats = tuple(g // p for g, p in zip(self.grid, self.physical))` (`_trace.py:1303`), and `shape=` must divide the logical grid exactly (`_trace.py:1363-1368`).
  * **Coordinates reach the body as positional arguments**: `@h.body def _(tx, ty)`, arity-checked.
  * `at=(column, row)` pins `x_loc`/`y_loc` instead of letting `air-place-herds` choose
    (`_trace.py:1771-1778`) — needed for cascade chains.
  * `link_with=` stamps the object file; **one object file per herd** (`_trace.py:1330-1338`).

### D.2 — `alloc` scopes, and how L1-per-tile is addressed

`air.alloc(shape, dtype, scope=None, vector=None, column=None, split=True)` — `_trace.py:1845`.
Four scopes:

| spelling | level | lifetime | per-core? |
|---|---|---|---|
| `<herd>.private()` | **L1** | the herd body | yes — one buffer per core, implicitly |
| `<segment>.private()` | **L2** (memtile) | the segment | no — segment-wide |
| `<segment>.shared()` | **L1** | the segment | sliced: leading dims are the herd's, each core takes its slab |
| `<segment>.per_core()` | **L1** | the segment | yes — every core gets the whole shape privately |

`<herd>.shared()` **raises** (`_trace.py:1405-1414`). Budgets are checked at trace time:
`L1_BYTES = 65536` (`_trace.py:100`), `L2_BYTES_PER_MEMTILE = 512 * 1024` ×
`DEVICE_COLUMNS = {"npu1": 4, "npu2": 8}` (`_trace.py:117-118`).
L3 is **not** `alloc`-able — it is `air.tensor(shape, dtype, name=, inout=)` (`_trace.py:1803`),
the kernel's interface.
`<segment>.shared()` is how L1-per-tile is addressed explicitly (`_trace.py:1069-1071`):

```python
        It is still L1, so it is still charged against the 64 KB core budget, and
        each core addresses its own slab of it -- which is why such a buffer is
        allocated with leading herd dimensions and subscripted by tile
```

### D.3 — `ops.load` / `ops.store`

`ops.load(dst, src, pad_before=None, pad_after=None, dependency=None)` — `ops.py:257`;
`ops.store(src, dst, ...)` — `ops.py:287`. Both emit `air.dma_memcpy_nd` (`ops.py:245`:
`from air.dialects.air import dma_memcpy_nd`) and return a `Token`.

Levels, verbatim `ops.py:260-262`:

```
    ``load(l1, A[i:i+n])`` is L3 to L1; ``load(l1, staged[tx, 0:n, :])`` is L2 to
    L1; ``load(l2, A[i:i+n])`` is L3 to L2. The destination is always the buffer
    being filled.
```

Regions are **strided memref regions** built from numpy-style subscripts, and
`.reshape(...)` / `.transpose(...)` are *views* that change the DMA's wraps and strides
(never a copy) — `__init__.py` docstring, and `python/test/api/tensor_views.py:74`:
`// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[0, 0] [32, 64] [1, 32])`.
Padding (`pad_before`/`pad_after`) is accepted here but **rejected on channels**.

### D.4 — `channel()` / `put` / `get`

`channel(name, size=None, broadcast_shape=None, channel_type=None, attrs=None)` —
`_channel.py:581`. `Channel.put(obj, indices=None, dependency=None, dest=None)` —
`_channel.py:511`; `Channel.get(obj, indices=None, dependency=None)` — `_channel.py:524`.

* `indices=` selects a bundle member.
* **Put and get sizes need not match** — `_channel.py:29-32`:
  ```
  **Put and get sizes need not match.** ``passthrough_channel`` puts one whole
  ``[4096]`` tensor and the herd takes four gets of ``[1024]``;
  ```
* `broadcast_shape=` requires `size=`; the error text spells the idiom
  (`_channel.py:127-131`): *"a 1-to-N fan-out is `size=[1]*N`, `broadcast_shape=…`"*.
* Implemented channel types — `_channel.py:547`: `_IMPLEMENTED_TYPES = ("npu_cascade", "npu_dma_packet")`,
  plus the default `npu_dma_stream`. `npu_mmio` and `gpu_symmetric_heap` raise.
* **`_UNSUPPORTED`, verbatim (`_channel.py:553-570`):**
  ```python
  _UNSUPPORTED = {
      "channel_type": (
          "Still unimplemented: "
          + ", ".join(_UNIMPLEMENTED_TYPES)
          + ". Each has its own lowering and its own verifier rules -- mmio, for "
          "instance, requires an L3 put, an L1 get and a constant "
          "memref.get_global source"
      ),
      # These two need a packet channel, which is now implemented, but the knobs
      # themselves are not: say so, rather than implying packet is the blocker.
      "dest": "the packet-demux destination index (the channel type it needs is "
      "implemented; this knob is not)",
      "packet_ids": "explicit packet routing ids (the channel type they need is "
      "implemented; these are not)",
      "buffer_resources": "the objectFifo depth knob",
      "pad_before": "padded transfers",
      "pad_after": "padded transfers",
  }
  ```
  **Correction to §1.4 item 3:** `dest` is in that dict but **is implemented on `put()`** —
  it is an explicit named parameter (`_channel.py:511`), never routed through
  `**unsupported`, and it is materialised at `_channel.py:459` (`extra["dest"] = coerce_index(dest).materialize()`).
  The dict entry only fires if `dest=` is passed to `channel()` or to `get()`.
  So the genuinely unsupported set is: `packet_ids`, `buffer_resources`, `pad_before`,
  `pad_after`, and the two unimplemented channel types.
* Scope rule — as r3 corrected: the **code** requires an enclosing `air.launch` for an L3
  endpoint and a segment only inside a herd body; the module docstring
  (`_channel.py:39-41`) is stricter. Cite the code.

### D.5 — `_loop.py`: what loops exist

* **`air.sequential(start, stop=None, step=None, name=None)`** → **one `scf.for`**
  (`_loop.py:89`). Bounds may be Python ints, `air.symbol`s, an enclosing loop's IV, an
  `ops.switch` result, or an `air.rtp` runtime parameter. A **coordinate-derived bound is
  refused** (`_loop.py:137-144`). Constant bounds are checked to tile the extent exactly.
* **`air.parallel(...)`** → **`scf.forall`** (`_loop.py:211`), one loop with N IVs for a grid.
  Its IV **may** index a channel bundle; `sequential`'s may not (that is the
  `ChannelPutOp::verify` rule of §B.2 showing through).
* **A plain Python `for` unrolls at trace time** and is documented as the wrong tool when
  buffers are reused across trips (`_loop.py:14-19`).
* **No loop-carried values.** Every emitted `scf.for` terminates with `yield_([])`
  (`_loop.py:180`, `_loop.py:200+`). There are **no `iter_args`** anywhere in `air.api`.
  Reduction state is carried in **memory** — that is what `<segment>.shared()` /
  `<segment>.per_core()` and `ops.dot(acc=)` exist for.
* `air.sequential` is a Python generator, so `break`/`return` out of it is detected and
  reported (`_loop.py:60-84`, `aborted_regions`).

### D.6 — `_cond.py`: conditionals on herd coordinates

`ops.branch(c)` — a **real branch** lowering to `scf.if`; `c` compares **index expressions**
(`tx == 0`, a loop IV, an `air.symbol`, a Python int, in any mix). `ops.select(c, a, b)` is
the branchless per-element counterpart lowering to `arith.select`. The table is verbatim at
`_cond.py:18-28`. `with head.otherwise():` is the else. **No `and`/`or`; conjunction is
nesting** (`_cond.py:57-60`). A Python `if` on a coordinate raises, because `bool()` on a
`Condition` is refused (`_cond.py:41-45`).
Crucially, `_cond.py:49-54`: *"`air-to-aie` folds the branch away once the herd is unrolled
and the coordinate is a literal"* — so per-core specialisation is free at runtime.
`ops.switch(index, values=None)` is `scf.index_switch`, in an expression form and a region
form (`ops.py:997`); the region form has exactly one case region plus `otherwise()`.

### D.7 — `_extern.py`: attaching a precompiled kernel

`air.extern(name, link_with=None, scalars=(), signature=None, emit_c_interface=True,
link_with_mode=None)` — `_extern.py:50`. It emits a **`func.call`** into a private
`func.func` declaration stamped with `link_with`. Verbatim shape (`_extern.py:22-28`):

```
    func.func private @matvec_vectorized_bf16_bf16(i32, i32, i32,
        memref<1x8192xbf16, 2 : i32>, memref<8192xbf16, 2 : i32>,
        memref<4xbf16, 2 : i32>)
        attributes {link_with = "mv.o", llvm.emit_c_interface}
```

**Conventions:** buffer argument types are *derived* from the `Buffer` objects; **scalar
element types must be declared** via `scalars=` because a Python `0.0` cannot pick between
`bf16` and `f32` (`_extern.py:31-37`). `link_with` is mandatory. One object file per herd.
The object file is built outside the DSL (the examples' `Makefile`s compile a `.cc` with
`xchesscc` or Peano) and reaches `aircc` via `air-linalg-to-func{link-with=...}` /
the herd's `link_with` attribute.

### D.8 — `ops.py`: what compute exists natively

`__all__` (`ops.py:50-80`): `branch, load, store, copy, fill, cast, bitcast, maximum,
minimum, equal, not_equal, select, fma, reduce_add, reduce_max, argmax, switch, relu,
tanh, exp, rsqrt, sigmoid, silu, gelu, dot`.

* **Elementwise arithmetic on whole L1 tiles** via operators on `Buffer`/`BufferExpr`
  (`b[:] = a[:] * 2 + c[:]`), lazily built then emitted as a loop nest (vectorised where the
  innermost extent allows) by `_emit.emit_elementwise` (`_emit.py:601`).
* **`ops.dot(a, b, acc=, alpha=, transpose_b=, kernel=)`** (`ops.py:1515`) — emits
  `linalg.dot`/`vecmat`/`matvec`/`matmul` by operand rank, or a 6-D blocked contraction.
  **It is a statement accumulating into a buffer**, not an expression.
* Unimplemented, raising by name (`ops.py:1732-1735`): `reduce`, `stack`, `dequant`, `atomic_add`.
* **L2 buffers cannot be read or written elementwise** — `_value.py:615-619`:
  ```python
            raise TypeError(
                f"cannot {what} an {self.space} buffer elementwise: {self.space} "
                "is a memtile, which has DMA engines but no compute core. Stage "
  ```

### D.9 — `_compile.py`: `build(target=)`, the pipeline, and whether a device is needed

* **Accepted `target=`: `None`, `"auto"`, `"npu1"`, `"npu2"`** — validated against
  `PHYSICAL_HERD` (`_trace.py:172-177`). `"auto"` shells out to `xrt-smi` via
  `detect_target_device` and **falls back to `NO_DEVICE_TARGET = "npu2"`** when there is no
  device (`_trace.py:96`, `:182`). So `build()` works with no NPU; pass `target="npu1"`
  explicitly to avoid the probe.
* **`build()` contains no pass pipeline at all.** It creates a `Module`, replays the body
  into a `func.func`, then `module.operation.verify()` (`_compile.py:128-158`). Confirmed:
  `grep -rn "builtin.module(\|PassManager\|passmanager" python/air/api/` → **zero hits**.
  So `build()` needs **no device, no aircc, no XRT** — only the `air` Python package.
* `compile(target=, verbose=, output_format="xclbin", **kwargs)` (`_compile.py:245`) hands
  the module to `XRTBackend(target_device=self.target, ...)`, which shells out to `aircc`
  (`python/air/backend/xrt.py:474-478`).
* **The pass pipeline lives in C++, in `aircc`, not in Python.** Verbatim
  (`tools/aircc/aircc.cpp:905-995`, `buildOptimizationPipeline`):
  ```
  air-dependency,air-annotate-packet-ids{assign=true},air-hoist-dma-in-accum-pattern,
  [air-broadcast-detection,air-specialize-dma-broadcast,]
  air-dma-to-channel,canonicalize,cse,
  air-dependency-canonicalize,canonicalize,cse,
  air-isolate-async-dma-loop-nests{scope=launch},canonicalize,cse,
  air-fuse-channels,canonicalize,cse,
  func.func(air-split-l2-memref{max-launch-channels-mm2s=N max-launch-channels-s2mm=N}),
    canonicalize,cse,air-isolate-async-dma-loop-nests{scope=launch},canonicalize,cse,
  func.func(air-fuse-alloc-dealloc),func.func(air-shrink-memref-sizes-by-access),
  air-label-scf-for-to-ping-pong{device=D},air-ping-pong-transform,canonicalize,cse,
  [air-linalg-to-func{link-with=X} | func.func(convert-linalg-to-loops)],
  func.func(air-opt-memtile-dma-bds{device=D}),canonicalize,cse
  ```
  then placement + `air-enforce-channel-fifo-order` (`aircc.cpp:1155-1172`), then
  `air-verify-hierarchy-locality{strict=true}` (`aircc.cpp:1213-1217`), then
  `air-to-aie{...}` (`aircc.cpp:1230-1262`).

### D.10 — `_trace.py`: pure tracing builder, and Python control flow

Confirmed pure tracing. `_compile.py:8-10`:

```
The body is not executed when ``@launch.body`` runs -- it is recorded, and
replayed later by :meth:`LaunchContext.build` from inside an MLIR context.
```

Python control flow inside a body is handled by **refusal at the value level, not by
interpretation**: a comparison on a herd coordinate is a `Condition`, and `bool()` on one
raises (`_cond.py:41-45`), so `if tx == 0:` cannot silently pick one branch for all cores.
A Python `for` over a Python range *does* run at trace time and unrolls
(`_loop.py:14-19`). There is no `_interpret*`/`_sim*`/`_eval*` file — r3's B14 finding stands.

### D.11 — **THE ANSWER: can a herd body contain an arbitrary scalar loop nest over L1 buffers?**

**YES. Verified. This contradicts the design document's framing of `air.api`.**

A fully-integer subscript on an L1 buffer is a **rank-0 element access**, and it emits
`memref.load`/`memref.store` with no loop of the DSL's own. `_value.py:590-593`:

```python
        # A region of the buffer. numpy's rule decides the shape being written:
        # out[0:8, :] is a rank-2 block and out[i, j, k] is one element, so a
        # fully-integer subscript writes a scalar and emits no loop at all.
```

`_emit.py:647-650`:

```python
    # A rank-0 buffer is a single scalar -- the accumulator linalg.dot writes
    # into. There is no innermost dimension to vectorise, and the loop nest is
    # empty, so the scalar path handles it with no induction variables at all.
```

Wrap those in `air.sequential` loops and you have a scalar loop nest. The tree's own test
proves it — `python/test/api/partial_assign.py:85-92`:

```python
def a_shifted_window_reads_at_a_loop_variable():
    def body(h, src, dst):
        for i in air.sequential(4):
            for j in air.sequential(4):
                dst[i, j, 0] = src[i + 3, j, 1] + 1
```

with the expected IR, `partial_assign.py:79-84`:

```
// CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step
// CHECK: scf.for %[[J:.*]] = %{{.*}} to %{{.*}} step
// CHECK: %[[SH:.*]] = affine.apply #{{.*}}()[%[[I]]]
```

and the file's own statement of purpose, `partial_assign.py:11-15`:

```
That covered every kernel whose arithmetic is
tile-shaped, and none whose arithmetic is not: a convolution accumulates into
``out[oh, ow, co]`` from ``in[oh + kh, ow + kw, ci]``, and neither of those is a
tile. Refusing them pushed such a kernel out of the DSL entirely.
```

Accumulation into the same element works too — `partial_assign.py:120-127`
(`dst[i, 0, 0] = src[i, 0, 0] * 2 + src[i, 0, 0]`, one cached read).

**So `c[m,n] += a[m,t]*b[t,n]` under three `air.sequential` loops compiles for the AIE core.**
Compute does **not** have to be an extern kernel or a supported op. What an extern kernel
buys is **vectorisation**, not expressiveness — `_extern.py:11-14`:

```
Every vectorised matmul-family example in this tree computes through a ``.cc``
kernel written against the AIE intrinsics (``aie::load_v``, ``aie::mac``,
``aie::accum``), because ``ops.dot`` on its own lowers through
``convert-linalg-to-loops`` and comes out scalar.
```

**Three levels of compute, all reachable:** (1) scalar loop nest, native, slow;
(2) `ops.dot`/elementwise → `linalg` → either `convert-linalg-to-loops` (scalar) or
`transform.air.herd_vectorize` (direct codegen); (3) `air.extern` → `func.call` into a
hand-vectorised `.o`.

### D.12 — examples exercising each mechanism

**113 files** under `programming_examples/` + `test/` import `air.api`
(`grep -rln "from air import api\|air\.api\|from air.api"`). Only two channel examples and
two matmul variants still use the older raw `@module_builder` bindings.

| mechanism | example |
|---|---|
| tiled GEMM, `air.api`, L2 staging, extern kernel | `programming_examples/matrix_multiplication/bf16/run.py:57` (`from air import api as air`), `:234` (`air.ops.dot(l1_a, l1_b, acc=acc)`) |
| conv / stencil-shaped indexing | `programming_examples/conv2d/conv2d.py`, `conv2d_14x14/`, `conv1d_depthwise/` (all `air.api`) |
| herd dataflow (3 herds wired with channels) | `programming_examples/herd_dataflow/run.py:3` — *"A three-herd pipeline wired with channels, on air.api."* |
| vector add | `programming_examples/eltwise_add/eltwise_add.py`, `axpy/axpy.py` |
| **channels inside a herd body, put-before-get** | `programming_examples/channel_examples/worker_to_worker/worker_to_worker.py:116-131` |
| cascade channel + `ops.branch` per-core | `programming_examples/cascade_reduction/cascade_reduction.py` |
| packet-switched broadcast, `dest=` | `programming_examples/llms/llama32_1b_int4/multi_launch_builder/o_gemv_ffn_int4_fused.py` |
| extern kernel with scalars | `programming_examples/matrix_vector_multiplication/bf16/matvec.py` |

> **Design consequence — and it is the sharpest finding in this report.**
> §1.4's claim that the whole available delta is *(a) the program runs in CPython as its own
> spec* and *(b) spatial intent is declared* survives only in that exact narrow form.
> Everything else it lists as missing from `air.api` — an arbitrary loop nest, per-core
> branching, conditionals on coordinates, runtime parameters, explicit placement (`at=`),
> extern kernels, packet routing, strip-mined herds, view-based data layout — **is already
> there.** Our surface is a *notation* over `air.api`, not a capability extension. Any
> claim of the form "P<n> can express X and `air.api` cannot" must be re-checked against
> §D.5–D.11 before it goes in a pitch.

---

## E. Ping-pong passes (E4) and `air-broadcast-detection`

**Verdict: RESOLVED from source. The ping-pong passes run by default, and their applicability
conditions are precise and emittable.**

### E.1 — the passes run by default in `aircc`

`tools/aircc/aircc.cpp:966-980`:

```cpp
    std::string labelPass = "air-label-scf-for-to-ping-pong{" + labelOpts + "}";
    std::string ppPass = "air-ping-pong-transform" + ppOpts;
    os << "," << labelPass << "," << ppPass << ",canonicalize,cse";
```

guarded only by `--omit-ping-pong-transform` (`""`, `"L1"`, `"L2"` or `"all"`) — the exact
spelling, re-verified against `aircc --help` on 2026-09-12; an earlier revision of this line
said `--omit-pingpong`, which no build accepts. `device=` is passed through so
the labeller can decline on L1 budget.

### E.2 — what `air-label-scf-for-to-ping-pong` requires

Pattern `LabelScfForLoopForPingPongPattern` (`mlir/lib/Transform/AIRDependencyScheduleOpt.cpp:1604`),
predicate `isPingPongCandidate` (same file, ~`:1712`). A candidate `scf.for` must satisfy
**all** of:

1. **An `scf.for`** (not `affine.for`) not already carrying an `unroll` attribute.
2. **At least one `air.execute` wrapping a `memref.alloc` as a *direct child* of the loop
   body.** Summary, `mlir/include/air/Transform/Passes.td:970-971`:
   ```
    This pass labels all scf.for loops which contain air.execute event of memref.alloc,
    which is a direct child op of said scf.for, as candidate loop for ping-pong
   ```
   **This means the IR must already be asynchronous** — `air-dependency` must have run,
   which it has (it is first in the aircc pipeline).
3. **Each candidate alloc must be dead on entry**, i.e. its *first* access is a definite
   write. `AIRDependencyScheduleOpt.cpp:1620-1624`:
   ```cpp
    // (1) An alloc that would be duplicated must be dead on entry to the body.
    // If the first access reads it, the value flows from one iteration into the
    // next, and splitting it across the two instances gives each parity its own
   ```
   An `air.channel.get` filling the buffer counts as a definite write; an **opaque callee
   (an extern kernel `func.call`) does not** and disqualifies the loop.
4. **No opaque callee touching a herd block argument** (`:1626-1630`).
5. **At most one `air.channel.get` per candidate alloc per iteration**, and static trip
   counts on intervening loops.
6. **L1 budget**: `herd body total + duplicated allocs <= target local memory`
   (`exceedsL1Budget`, ~`:1690`).
7. **No `air.disable_ping_pong`** attribute on the candidate or any nested `scf.for`
   (the documented user-facing opt-out, `:1745-1757`).
8. `omit-memory-space` (`""`/`"L1"`/`"L2"`) not excluding the alloc's space.

`air-ping-pong-transform` then unrolls by 2 and `air-construct-ping-pong-dependency-pattern`
yields the loop-carried token edges.

### E.3 — the depth path closes in `air-to-aie`

`mlir/lib/Conversion/AIRToAIEPass.cpp:3873-3886`:

```cpp
    // Annotate channels with buffer_resource, i.e. object count
    for_op.walk([&](Operation *op) {
      if (auto get = dyn_cast_if_present<air::ChannelGetOp>(op)) {
```
```cpp
        chan_op->setAttr(
            "buffer_resources",
            IntegerAttr::get(IntegerType::get(chan_op->getContext(), 32),
                             unroll_factor));
```

So: ping-pong label sets `unroll=N` → `air-to-aie` writes `buffer_resources = N` on the
channel → `getBufferResources()` becomes the ObjectFIFO depth (§C.2) / the lock slot count
(§C.3). **B15 is closed: the depth realisation path is real and complete.**

### E.4 — `air-broadcast-detection`'s trigger pattern

`mlir/include/air/Transform/Passes.td:874-877`:

```
    This pass detects DMA broadcast opportunities by tracing the source indices'
    dependence to the induction variables of any parent spatial loop space. Upon
    successful detection, the DMA shall be annotated by an affine set attribute
```

Implementation `BroadcastDetection` (`AIRDependencyScheduleOpt.cpp:3325`). It fires only when
**all** of:

1. the op is an **`air.dma_memcpy_nd`** — *not* an `air.channel.put/get`
   (`:3328` — `f.walk([&](air::DmaMemcpyNdOp dma_op)`). It therefore runs **before**
   `air-dma-to-channel`, which it does in the aircc pipeline;
2. the DMA is **inside an `air.herd`** and one endpoint is **L1** (`:3338`);
3. the source indices are **invariant with respect to exactly one herd induction variable**
   (`isVariantWrtHerdRows` xor `isVariantWrtHerdCols`, `:3390-3402`), or invariant wrt both
   (broadcast to the whole herd, `:3414`);
4. the herd's `sizes` are `arith.constant` (`:3386-3390`), and the broadcast dimension has
   extent > 1.

The attribute it writes is `broadcast_pattern`, an `IntegerSetAttr`.

### E.5 — E4 RUN: the ping-pong passes **do** fire on an `air.api`-emitted herd loop

A W1-shaped GEMM was written against `air.api` and emitted with the installed wheel
(`air-probe/e4_api.py`, 20 lines, scratchpad only): `M=N=K=256`, `TM=TN=TK=64`, logical herd
`4x4` strip-mined onto `shape=(1,4)`, `acc` allocated in the herd body, and **`a` and `b`
allocated inside the `air.sequential` K loop** — exactly the shape §E.2 requires.

`launch.build(target="npu1")` succeeded with **no device and no XRT**, producing a 52-line
AIR module whose K loop is:

```mlir
        scf.for %arg11 = %c0_4 to %c256 step %c64_5 {
          %alloc_6 = memref.alloc() : memref<64x64xbf16, 2 : i32>
          %alloc_7 = memref.alloc() : memref<64x64xbf16, 2 : i32>
```

**Labelling — it fires.** After
`air-dependency,…,air-dma-to-channel,…,func.func(air-fuse-alloc-dealloc),air-label-scf-for-to-ping-pong{device=npu1}`:

```
--- unroll attrs found: ---   108:        } {unroll = 2 : i32}
--- hoist_alloc: ---          83:hoist_alloc = true    87:hoist_alloc = true
```

The K loop is labelled `{unroll = 2}` and both L1 tiles are marked for hoisting.

**Transform — it fires.** Adding `air-ping-pong-transform,canonicalize,cse`: the K loop's step
goes `64 → 128` (unrolled by two), the two L1 `bf16` allocs become **four**, and the loop
acquires **four loop-carried async tokens**:

```mlir
        %13:4 = scf.for %arg9 = %c0_2 to %c256_0 step %c128 iter_args(%arg10 = %12, %arg11 = %async_token_11, %arg12 = %11, %arg13 = %11) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
```

Those tokens are the *concurrency* edges §9's E4 asked for. **B15 and E4 are resolved
affirmatively.**

**Realisation — measured.** Through `air-to-aie{device=npu1}` the double buffering appears as
**eight locks with `init = 2`** against four with `init = 1` (§C.7): the ping-ponged A and B
channels get two slots, the single-buffered accumulator and output get one.

**One caveat worth recording.** `buffer_resources` did **not** appear on the channel in this
run. The pattern that writes it (`AIRToAIEPass.cpp:3862-3886`) requires the `unroll` attribute
to still be present when `air-to-aie` runs, and `air-ping-pong-transform` consumes it. On the
default lock path (`use-objectfifo=false`, §C.1) that costs nothing — the depth is realised as
the lock slot count instead, which is what the `init = 2` histogram shows. `buffer_resources`
is the **ObjectFIFO-path spelling** of the same number, and it is what a surface `depth=N`
should set directly when we want a depth the ping-pong heuristic would not choose.

**Bonus, unasked for:** `air-broadcast-detection` **also** fired on this module — the
`air.channel.get` for the A operand came out wrapped in `affine.if #set()[%arg3, %arg4]`,
the broadcast specialisation. A plain `ops.load` from L3 whose source index is invariant in
one herd axis gets multicast derived for free, with no `broadcast_shape` written.

> **Design consequence.** For double-buffering to fire, our emitter must produce, inside the
> herd body: an `scf.for` whose *direct* children include the L1 `memref.alloc`s (not hoisted
> above the loop, not nested one level deeper), each first touched by an `air.channel.get`
> or another definite write, at most one get per buffer per trip, static inner trip counts,
> and no extern `func.call` as a buffer's first toucher. **If our compute is an extern
> kernel and it is the first op to touch the tile, ping-pong is silently declined.**
> For multicast to be *derived*, our loads must reach the pipeline as `air.dma_memcpy_nd`
> (i.e. `ops.load`, not a hand-written `channel.put`) with coordinate-invariant source
> indices and a constant herd size. Declaring `broadcast_shape` on a channel is the
> alternative and bypasses the detector entirely.

---

## F. `air-to-aie` options, device values, and how `build(target=)` maps

**Verdict: TRUE and extended. §1.2's "default device is `xcvc1902`" is confirmed; the
`target=` → `device=` mapping is a four-hop chain, and it *never* produces a Versal value.**

### F.1 — every `air-to-aie` option (13), `mlir/include/air/Conversion/Passes.td:174-235`

| CLI name | Type | Default | Line |
|---|---|---|---|
| `row-offset` | unsigned | `1` | 178 |
| `col-offset` | unsigned | `1` | 180 |
| `emit-while-loop` | bool | `false` | 182 |
| `emit-herd-lock` | bool | `false` | 185 |
| `test-patterns` | string | `""` | 191 |
| `device` | string | `"xcvc1902"` | 194 |
| `use-objectfifo` | bool | `false` | 197 |
| `output-elf` | bool | `false` | 201 |
| `generate-shim-dma` | bool | `false` | 209 |
| `insert-trace-packet-flow` | bool | `false` | 213 |
| `use-lock-race-condition-fix` | bool | `false` | 216 |
| `use-lock-race-condition-fix-v2` | bool | `false` | 221 |
| `stack-size` | unsigned | `2048` | 231 |

`Passes.td:194-196`:

```tablegen
    Option<"clDevice", "device", "std::string",
          /*default=*/"\"xcvc1902\"",
           "AIE device to target.">,
```

**Correction:** `trace-size` / `trace-offset` are **not** `air-to-aie` options; they belong
to `airrt-to-npu` (`Conversion/Passes.td:434-439`). `air-to-aie`'s only trace knob is
`insert-trace-packet-flow`.

### F.2 — device validation

Not a StringSwitch in mlir-air; the string goes straight to mlir-aie's generated symbolizer.
`mlir/lib/Conversion/AIRToAIEPass.cpp:8095-8097`:

```cpp
    auto device = AIE::symbolizeAIEDevice(clDevice);
    if (!device) {
      module.emitOpError("Invalid aie.device option");
```

**The authoritative enum is in mlir-aie** (not present here). Read, at mlir-aie
`10767b50e5beec9b2ec97ce92298e3507a33490c`: `include/aie/Dialect/AIE/IR/AIEAttrs.td`
(the `AIEDevice` `I32EnumAttr`) and `include/aie/Dialect/AIE/IR/AIETargetModel.h`.

Enum constants actually referenced in mlir-air C++ (`computeSubDeviceType`,
`AIRToAIEPass.cpp:846-919`): `npu1, npu1_1col, npu1_2col, npu1_3col, npu2, npu2_1col,
npu2_2col, npu2_4col`. Device strings appearing in tests/examples: `xcvc1902` (65×),
`xcve2802` (68×), `npu1{,_1col,_2col,_3col,_4col}`, `npu2{,_1col,_2col,_4col,_8col}`.
**`xcve2302` appears zero times in mlir-air.**

### F.3 — `build(target=)` → `device=`

1. `_compile.py:125` — `self.target = resolve_target(target or self.target)`.
2. `_trace.py:164-183` — accepted: `None`, `"auto"`, `"npu1"`, `"npu2"`. Anything else
   raises `ValueError`. `"auto"` probes `xrt-smi`, default `"npu2"`.
3. `_compile.py:253-259` — `XRTBackend(target_device=self.target, ...)`; `xrt.py:420-427`
   may widen to `npu1_Ncol` / `npu2_Ncol`; `xrt.py:474-478` passes `--device <target>` to aircc.
4. `tools/aircc/aircc.cpp:1230-1240` builds the string:
   ```cpp
       os << "air-to-aie{";
       os << "emit-while-loop=" << (omitWhileTrueLoop ? "false" : "true");
   ```
   ```cpp
       os << " row-offset=" << resolvedRowOffset;
       os << " col-offset=" << resolvedColOffset;
       os << " device=" << deviceName.getValue();
   ```
   `aircc`'s own `device` default is `xcvc1902` (`aircc.cpp:164-166`); row/col offsets are
   device-derived (col-offset `0` for any name containing `npu`, else `7`; row-offset `2`).

> **Design consequence (E5).** `air.api` **cannot** reach a Versal device: `resolve_target`
> hard-rejects anything but `npu1`/`npu2`. A Versal attempt must go through `aircc --device
> xcvc1902` on a hand-built or `air.dialects`-built module, bypassing `LaunchContext.compile`.
> That makes E5 strictly more expensive than §9 assumes.

---

## G. Build / install path

**Verdict: a pip-only path exists, is the *documented recommended* path, and it WORKS. It was
run on this machine: ~6 minutes, ~2.3 GB, no NPU, no XRT, no Vitis, no source build.**

### G.1 — the repo says wheels first

`README.md:94`:

```
Prebuilt wheels are the recommended path. Each guide also covers a source build.
```

`docs/buildingRyzenLin.md:3-5`:

```
## Install Prebuilt Wheels (Recommended)
The fastest way to get MLIR-AIR is to install the prebuilt wheel — no source build, no CMake, no LLVM clone.
```

Prerequisites, `docs/buildingRyzenLin.md:9-11`: **Python 3.11–3.14**, **pip**, and
*"**XRT** (optional, required only for running on hardware)"*.

### G.2 — the wheel index URLs, verbatim

`docs/buildingRyzenLin.md:28-32`:

```
   pip install 'mlir_air[aie]' \
     -f https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels \
     -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-4 \
     -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
```

Nothing is on PyPI proper; resolution is entirely through those GitHub release pages, so the
`-f` flags are mandatory. A wheel-publishing CI job exists —
`.github/workflows/buildAIRWheels.yml:4` (`name: Build mlir-air Wheels`), running on
`ubuntu-latest` and `windows-2022`, releasing to the `latest-air-wheels` tag. Console scripts
are declared at `utils/mlir_air_wheels/pyproject.toml:23-27`: `air-opt`, `air-translate`,
`aircc`, `air-runner`.

### G.3 — the probe: run, and verified independently

Venv at `air-probe/venv`, Python 3.14.7, x86_64, **no `/dev/accel*`, no `amdxdna` module, no
`/opt/xilinx/xrt`**. Exit 0 in **5 min 43 s**. Installed versions, re-checked directly:

```
llvm-aie            22.0.0.2026091201+386ca5c6
mlir-aie            1.4.3.dev55+g10767b5
mlir-air            0.0.1.2026091204+ff95a9b
numpy               2.5.3
```

**The `mlir_air` wheel is built from the exact commit this document cites** (`+ff95a9b`), and
`mlir_aie` matches the pin in `utils/clone-mlir-aie.sh:17`.

Re-verified by hand, **with no environment variables set at all**:

* `air-opt --version` → `LLVM version 24.0.0` (AOMP-23.0-60).
* `python -c "import air.ir, air.passmanager; from air.dialects import air"` → OK.
* `aircc --help` → exit 0.
* `air-opt -air-to-std mlir/test/Conversion/AIRLowering/air_launch.mlir` → exit 0, correct IR.
* **And everything in §B.6, §C.7, §D.11 and §E.5 above was run on this install.**

A **core-only** install (`pip install mlir_air -f <air page>`, no `[aie]`) also works —
1 min 45 s, 972 MB — and gives `air-opt` + `import air` + `aircc --help`, but `aircc` cannot
actually compile because it needs `aiecc` from `mlir_aie`.

### G.4 — (a) does `air-opt` run on CPU with no NPU? **Yes.**

It is a plain `MlirOptMain` driver — `tools/air-opt/air-opt.cpp:50-51`:

```cpp
  return failed(
      MlirOptMain(argc, argv, "MLIR-AIR modular optimizer driver\n", registry));
```

`tools/air-opt/CMakeLists.txt:67-94` links only MLIR/AIR/AIE libraries — no XRT, no LibXAIE;
`ldd` on the shipped binary shows only `libm libz libstdc++ libgcc_s libc ld-linux`.
493 lit tests drive it, and CI runs them on stock GitHub runners:
`.github/workflows/buildAndTestNoRuntime.yml:29` (`runs-on: ubuntu-…`), `:109-111`
(`ninja check-air-cpp / check-air-mlir / check-air-python`). `CMakeLists.txt:280-288` skips
only the *hardware* test subdirectory when XRT is absent. **Confirmed by running it.**

### G.5 — (b) does `aircc` need XRT/a device for an xclbin? **No device. The `xclbin` container step needs XRT's `xclbinutil` binary.**

`aircc` is now a C++ binary; the Python driver is retired —
`python/air/compiler/aircc/main.py:10-11`:

```
The Python aircc driver has been retired in favor of the native C++ aircc
binary (tools/aircc/aircc.cpp).
```

`tools/aircc/CMakeLists.txt:16-48` links no XRT (confirmed by `ldd`). It shells out to
`aiecc`, `air-opt`, `mlir-opt`, `aie-translate`, `llc`, `clang`. `xchesscc` is **opt-in**
(`aircc.cpp:152-155`); Peano is the default. Output formats, `aircc.cpp:266`:

```cpp
enum OutputFormatKind { OF_xclbin, OF_txn, OF_elf, OF_pdi, OF_none };
```

The hardware-free mode is documented — `docs/buildingRyzenLin.md:188-203`:

```
### Example 1: Hardware-Free Compilation (No XRT Required)
- Does **not** generate xclbin (no `xclbinutil` needed)
- Does **not** require XRT or hardware
```

**Measured on this NPU-free machine:** `--output-format=none` → exit 0, all 22 aiecc stages
through Peano, core ELF produced. `--output-format=pdi` → exit 0, `aie.pdi` produced.
`--output-format=xclbin` → **exit 1 at step 40/41**, with:

```
aiecc: ShellCommand: tool 'xclbinutil' not found in search paths or PATH
```

So the *only* thing blocked without XRT is the final container packaging — not compilation,
not placement, not codegen.

### G.6 — (c) what does `import air` require at import time? **Nothing.**

There is **no `python/air/__init__.py`**; `air` is an implicit namespace package put on
`sys.path` by a `.pth` file (`utils/mlir_air_wheels/setup.py:275-277`). Dialect registration
is per-`Context`, not at import (`python/air/_mlir_libs/_site_initialize_0.py:16-18`). XRT is
imported lazily and defensively — `python/air/backend/xrt.py:717-719`:

```python
        # Try to import pyxrt - it's only needed for load(), not compile()
        try:
            import pyxrt as xrt
```

`ctypes.CDLL` appears only in `cpu_backend.py`, `linalg_on_tensors.py` and `txn_builder.py` —
none on the `import air` path. **Confirmed: `from air.backend.xrt import XRTBackend`
succeeds here with no XRT installed.**

### G.7 — the source build, for reference only (do not run it)

Three tiers in `docs/buildingRyzenLin.md`: wheels (`:3`), a source build of *MLIR-AIR only*
against prebuilt LLVM/MLIR-AIE wheels (`:88`, via `utils/build-mlir-air-using-wheels.sh`), and
the legacy full build (`:290-341`). Only one time estimate exists anywhere —
`docs/buildingRyzenWin.md:196`:

```
Build from source only to modify MLIR-AIR itself. LLVM/MLIR and MLIR-AIE come from prebuilt wheels, so this compiles MLIR-AIR alone — a few minutes on a modern laptop, not an LLVM-sized build.
```

Pins if it ever becomes necessary: LLVM is the **ROCm fork** at
`56bcc1871734e6c375a254dec0ec74eb18d04a2e` (`utils/clone-llvm.sh:21-23`); mlir-aie at
`10767b50e5beec9b2ec97ce92298e3507a33490c` (`utils/clone-mlir-aie.sh:17`); Peano
`llvm-aie==22.0.0.2026090201+a36c62b9` (`utils/build-mlir-air-using-wheels.sh:58`);
`cmake>=4.4.2,<5.0`, `ninja!=1.13.0` (`utils/requirements_dev.txt:2-6`). Vitis/`xchesscc` is
needed **only** for the legacy VCK5000 path — `docs/buildingVCK5000.md:36`: *"you do not need
the above additional Xilinx packages to make use of the AIR compiler passes."*

> **Design consequence.** Day 0 is **six minutes, not a day**. Every compile-time experiment
> in §9 — E3 and E4 included — is runnable by every team member on a laptop today. The only
> capability the team lacks without a Ryzen AI machine is `--output-format=xclbin` (needs
> `xclbinutil` from XRT) and actually executing a kernel.

---

## H. GEMM case-study pipeline

**Verdict: TRUE that `air.api` skips `air-par-to-herd`; the GEMM flow starts from `air.api`,
not from `linalg.generic`.**

### H.1 — where `matrix_multiplication` starts

`programming_examples/matrix_multiplication/bf16/run.py:1` — *"Tiled matrix multiplication on
air.api."*; `:57` — `from air import api as air`. The AIR hierarchy
(`air.launch`/`air.segment`/`air.herd`/`air.alloc`/channels) is authored **directly**. Only
the innermost kernel is linalg, emitted by `air.ops.dot` (`run.py:234`).

Six of eight subdirectories are `air.api` (`bf16`, `bf16_x_bfp16`, `bfp16`, `i16`, `i8`,
`int4_awq`); two (`bf16_in_bf16_out`, `bf16_in_fp32_out`) still use the older
`@module_builder` raw bindings. Neither group starts from `linalg.generic` at the top level.

### H.2 — the passes it actually uses

The Makefile contains **no pipeline** — it compiles the `.cc` micro-kernel and runs `run.py`
(`matrix_multiplication/bf16/Makefile:79-80`). `run.py` contains **no pipeline** either; it
calls `launch.build(target=...)` (`run.py:405`) and, only under `--direct-codegen`, applies a
`transform.with_named_sequence` script matching `linalg.generic` (`run.py:424`, `:533`).
**The real pipeline is the C++ one in `aircc`, quoted verbatim in §D.9.**

Two lowering strategies for the kernel body, selected by a flag:
`air-linalg-to-func{link-with=mm.o}` (default for this example) or
`func.func(convert-linalg-to-loops)` (`aircc.cpp:984-989`).

### H.3 — does `air.api` skip `air-par-to-herd`? **Yes, completely.**

`air-par-to-herd` / `air-par-to-launch` / `air-copy-to-dma` appear in Python only inside
`compile_from_linalg_on_tensors` — the legacy torch-mlir entry point, unreachable from
`air.api`. `python/air/backend/xrt.py:674-687`:

```python
            DEFAULT_PIPELINE = (
                "builtin.module("
                + ",".join(
                    [
                        "buffer-results-to-out-params",
                        "air-linalg-codegen",
                        "air-par-to-herd{depth=-1}",
```

`grep -n "air-par-to-herd\|air-copy-to-dma\|ConvertToAIR" tools/aircc/aircc.cpp` → **zero hits**.
The aircc pipeline begins at `air-dependency` and assumes the AIR hierarchy already exists.
`air-dma-to-channel` (`aircc.cpp:924`) still runs — it converts the `air.dma_memcpy_nd` that
`ops.load`/`ops.store` emit into `air.channel.put/get`.

> **Design consequence.** Our emitter should target `air.api` (or `air.dialects` directly) and
> inherit the whole aircc pipeline unchanged. We never need `linalg` → AIR promotion, which
> removes an entire class of "does the promotion pass find my loop?" risk that a Halide- or
> Triton-style front end carries. But it also means **`air-broadcast-detection` only sees our
> `ops.load`s, never our `channel.put`s** (§E.4).

---

## I. E2 semantic check (numpy, throwaway)

**Verdict: (a) PE-outermost MISMATCHES; (b) timestep-outermost MATCHES. Exactly as §9 predicted.**

Probe: `/tmp/claude-1000/.../scratchpad/air-probe/e2_jacobi.py` (20 lines, scratchpad only,
**not** in the repo). 8×8 grid, `PI = 2` strips of 4 rows, `T = 4`, fixed zero Dirichlet
boundary, `numpy.random.default_rng(0)` interior.

```
(a) p-outermost match: False maxerr 0.2034108083949373
(b) t-outermost match: True maxerr 0.0
```

**Refinement of §9's prediction.** §9 expected *"the first divergence should be at `t = 1` in
PE 0's last row"*. Running (a) at `T = 1, 2, 3` and diffing against the reference:

| T | cells differing | first differing cells |
|---|---|---|
| 1 | 6 | `[4,1] [4,2] [4,3] [4,4] [4,5] [4,6]` |
| 2 | 18 | `[3,1] …` |
| 3 | 30 | `[2,1] …` |

The first divergence at `T = 1` is **row 4 — PE 1's *first* row**, not PE 0's last. The
reason: under `p`-outermost, PE 0 completes all its timesteps first, so when PE 1 runs its
`t = 1` it reads PE 0's row 3 at time **T**, not at time 0. The direction of the error is
opposite to what §9 guessed. The *verdict* is unchanged.

> **Design consequence.** W2·P5's fallback needs a **lockstep (timestep-outermost)
> interpreter**; `spatial-dsl/04` §4's stated "loop over all `(pi,pj)`" grid-outermost
> fallback does **not** compute Jacobi. M4's P5 cell keeps its r3 value of 2 unless the
> fallback is redefined. B18 is resolved.

---

## E1–E5 status after source reading (and, for E2–E4, after running them)

| # | Status | How |
|---|---|---|
| **E1** — put-before-get on a depth-1 channel | **RESOLVED — safe.** *(no hardware needed)* | Four facts agree, one of them **measured**: (1) **measured** — every `aie.lock` emitted for the §E.5 GEMM is `init = 0` (consumer/full) or `init = 1`/`2` (producer/empty); **no producer lock is ever 0** (§C.7); (2) that is what the lock allocator does by construction (`AIRToAIESchedulingUtils.cpp:1230-1246`, `:502-503`); (3) the ObjectFIFO path emits `aie.objectfifo.acquire(Produce, 1)` — an ask for an empty buffer, not a rendezvous (`air_channel_to_objectfifo_buffer_resources.mlir:22-27`); (4) **`programming_examples/channel_examples/worker_to_worker/` is a hardware-CI-run ring of exactly this protocol** (`run_makefile_peano.lit:4` — `REQUIRES: ryzen_ai, peano`, `CHECK: PASS!`). The compute-model "contradiction" is a wrong inline comment on an example that is separately illegal (same-scope put/get). **A hardware run would only confirm this.** |
| **E2** — does the W2·P5 fallback compute Jacobi? | **RESOLVED** *(measured)* | §I. PE-outermost mismatches (maxerr 0.203), timestep-outermost matches exactly. First divergence is **PE 1's first row at t=1**, not PE 0's last — §9's prediction had the direction backwards. |
| **E3** — does `air-opt`/`aircc` reject unbalanced or cyclic channel programs? | **RESOLVED — no.** *(measured: all three violations run through the real `air-opt`)* | §B.6. **(i)** prints one error and **still exits 0**, from a crash-guard in `air-dependency-canonicalize` (`Dependency.cpp:2063-2066`) that only asks whether *any* counterpart exists. **(ii)** and **(iii)** are silent through `air-to-aie`, `air-enforce-channel-fifo-order` and `air-verify-hierarchy-locality`. `docs/AIRCorrectnessChecker.md` is an unimplemented plan (no `mlir/lib/Analysis/`, zero grep hits for `air-verify-program`). `AIRToAIEPass.cpp:4484-4553` *repairs* L2 imbalance with dummy ops instead of diagnosing it. **Nothing further to run.** |
| **E4** — do the ping-pong passes fire on an `air.api`-emitted herd loop? | **RESOLVED — yes.** *(measured on a real `air.api` GEMM)* | §E.5. The K loop is labelled `{unroll = 2 : i32}` with both L1 tiles `hoist_alloc = true`; `air-ping-pong-transform` then doubles the step (64→128), duplicates the two bf16 tiles to four, and yields a 4-token `iter_args` — the concurrency edges. Through `air-to-aie{device=npu1}` the result is **8 locks with `init = 2`** against 4 with `init = 1`. The passes are in the **default** aircc pipeline (`aircc.cpp:966-980`). `air-broadcast-detection` fired on the same module unasked. **B15 closed.** |
| **E5** — end-to-end Versal | **STILL NEEDS A RUN, and is cheaper than feared but differently shaped than §9 assumed.** | `air-to-aie` defaults to `device=xcvc1902` (`Conversion/Passes.td:194-196`; 65 occurrences repo-wide), so the pass path exists, and the toolchain is now installed (§G) so it is a 10-minute test. **But `air.api` cannot reach it**: `resolve_target` accepts only `npu1`/`npu2` (`_trace.py:172-177`). The test must drive `aircc --device xcvc1902 --output-format=none` on a module built by hand or through `air.dialects`, bypassing `LaunchContext.compile`. Note also that `xcve2302` — one of the three Versal parts §1.2 cites — appears **zero times** in mlir-air. |

**Net: four of the five experiments are closed, and none of the four needed hardware.**
Only E5 remains, and it is now a ten-minute `aircc` invocation rather than a day-6 task.

---

## Install path for the team (day 0)

**Six minutes. Verified end to end on a Linux laptop with no NPU, no XRT, no Vitis.**
Requires Python 3.11–3.14 on linux_x86_64 (or win_amd64).

```bash
python3 -m venv airenv
source airenv/bin/activate          # fish: source airenv/bin/activate.fish
pip install --upgrade pip

pip install 'mlir_air[aie]' \
  -f https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels \
  -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-4 \
  -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly

# verify — all three work with NO environment variables set
air-opt --version
aircc --help
python -c "import air.ir, air.passmanager; from air.dialects import air; print('ok')"
```

Only if you want to drive `aircc` end to end (it needs `aiecc` on `PATH`):

```bash
SP=$(python -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
export PATH=$SP/mlir_air/bin:$SP/mlir_aie/bin:$PATH
export PEANO_INSTALL_DIR=$SP/llvm-aie
aircc --device npu1 --output-format=none design.mlir    # or --output-format=pdi
```

**Notes.**

* `--output-format=xclbin` is the **default** and is the one thing that needs XRT installed
  (for `xclbinutil`) — not a device. Use `--output-format=none` or `pdi` off-hardware.
  `programming_examples/matrix_multiplication/i8/run.py:573` does exactly this:
  `"output_format": "none",  # Skip xclbin generation (no xrt dependencies)`.
* The env-var block in `docs/buildingRyzenLin.md:38-50` is **not** needed for
  `air-opt`/`import air`: `python/air/tools.py:47-49` resolves the bundled
  `mlir_air/bin/<tool>` first and falls back to `PATH`.
* `pip install mlir_air` **without** `[aie]` is 1 min 45 s / 972 MB and is enough for
  `air-opt` and `import air`, but not for `aircc` to compile anything.
* Wheels come from GitHub release pages, and
  `.github/workflows/pruneAIRReleaseAssets.yml:95-97` prunes assets — **pin a `v*.*.*` tag
  rather than `latest-air-wheels` if the team needs a reproducible day-7 environment.**
* `pip install mlir_air` gives `air-runner` too (the performance simulator, §S7), at no extra
  cost.

---

## Surprises — things that contradict `01-paradigm-comparison.md`

Each entry names the sentence it contradicts.

### S1. `air.api` can express an arbitrary scalar loop nest over L1 buffers

**Contradicts** §1.4 item 2 and the framing that *"the defensible delta of any surface in this
document over `air.api` is exactly two things"*, and every (a)/(c) cell that treats a plain
loop nest as something a column must add.

**Evidence:** `python/test/api/partial_assign.py:85-92` writes
`dst[i, j, 0] = src[i + 3, j, 1] + 1` under two `air.sequential` loops and the FileCheck
expects `scf.for` / `memref.load` / `memref.store`. `_value.py:590-593` and `_emit.py:647-650`
give the rule: a fully-integer subscript is rank 0 and emits no loop. `air.api`'s own
docstring states the motive (`partial_assign.py:11-15`): *"a convolution accumulates into
`out[oh, ow, co]` … Refusing them pushed such a kernel out of the DSL entirely."*

### S2. There **is** a channel depth knob, it **is** read by the lowering, and the ping-pong passes **drive it automatically**

**Contradicts** §1.1 fact 1: *"you must either drive the ping-pong passes or emit AIR text with
an attribute nothing currently reads"*, and B15 (*"`[UNVERIFIED]` the `depth` realisation path"*).

**Evidence:** `AIR.td:738-743` (`getBufferResources()`, default 1);
`AIRToAIEPass.cpp:3095-3097` (it becomes the ObjectFIFO depth);
`AIRToAIEPass.cpp:3873-3886` (**the ping-pong `unroll` factor is written onto the channel as
`buffer_resources`** — the two halves of §1.1 fact 1 are the *same* mechanism, not
alternatives); `python/air/dialects/_air_ops_ext.py:195-216` (settable from Python);
`mlir/test/Conversion/AIRToAIE/air_channel_to_objectfifo_buffer_resources.mlir` (round-trip test).
It is spelled `buffer_resources`, which is why `grep -i depth AIR.td` found nothing.
**And it was measured** (§E.5): the ping-pong passes fire on an `air.api`-emitted GEMM
unprompted, and the depth reaches the hardware as `aie.lock … {init = 2 : i32}`.

### S3. AMD has already written the specification for the missing checker, and has not built it

**Contradicts** §6.8's framing of the rival as an original contribution, and §1.1 fact 2's
*"names no pass and no verifier"* (true of the *code*; the repo does name the design).

**Evidence:** `docs/AIRCorrectnessChecker.md` (511 lines), `:15-20` listing P1 channel
balance / P2 deadlock freedom / P3 resource bounds / P4 token consistency, `:58-80` naming
`mlir/lib/Analysis/ChannelBalance.cpp` etc. and the umbrella pass `-air-verify-program`.
**None of those files exist** and `grep -rn 'air-verify-program\|AIRVerifyProgram\|ChannelBalance'`
over `mlir python test tools` returns zero. The *idea* is upstream and public; the
*implementation* is not. Both halves matter to the pitch.

### S4. The compiler silently **repairs** put/get imbalance rather than rejecting it

**Contradicts** the compute model's *"A violation of the balance condition is a compile-time
error"* (quoted in §1.1 fact 2) and the ten-or-so (e) cells that read "AIR catches it".

**Evidence:** `AIRToAIEPass.cpp:4484-4491` — *"Insert dummy air.channel.put or
air.channel.get operations for L2 memrefs to ensure that the number of puts and gets match"*;
`:4530` *"Balance puts and gets by inserting dummy ops"*; `:4781` *"Generate dummy air.channel
ops to balance the number of BDs"*. An unbalanced program is not diagnosed; it is rewritten.

### S5. `air.api` **cannot** target Versal, so E5 is not "one hour"

**Contradicts** §1.2's *"Versal device values are accepted by the pass"* being carried
forward into §9's E5 recipe (*"`aircc` W1 with `device=xcvc1902`"*, cost "an hour").

**Evidence:** `_trace.py:172-177` raises `ValueError` for any target but `npu1`/`npu2`.
The pass accepts `xcvc1902`; the Python front end does not let you ask for it. Also,
`xcve2302` — one of the three Versal parts §1.2 cites from mlir-aie's `Devices.md` — appears
**zero times** anywhere in mlir-air.

### S6. `dest=` is implemented on `Channel.put`, despite appearing in `_UNSUPPORTED`

**Contradicts** §1.4 item 3: *"`_UNSUPPORTED` in `_channel.py` names `buffer_resources` …,
`dest` ('the packet-demux destination index'), …"* listed as *"explicitly unimplemented"*.

**Evidence:** `_channel.py:511` declares `dest=None` as an explicit named parameter (so it
never reaches `_reject_unsupported`), `:522` forwards it, and `:459` materialises it:
`extra["dest"] = coerce_index(dest).materialize()`. The dict entry (`:563`) only fires for
`channel(dest=...)` or `get(dest=...)`. The genuinely unsupported set is `packet_ids`,
`buffer_resources`, `pad_before`, `pad_after`, and the `npu_mmio`/`gpu_symmetric_heap` types.

### S7. mlir-air ships a **performance simulator**, `air-runner`

**Relevant to** §1.4 item 1 and B14's "the only live caveat" (which named only
`cpu_backend.py`). A second, closer artefact exists.

**Evidence:** `tools/air-runner/air-runner.cpp`, `mlir/lib/Util/Runner/{RunnerNode,Resource,
ResourceHierarchy,CostExpr}.cpp`, `docs/AIRRunner.md:3` — *"`air-runner` is a performance
simulator which models the concurrent execution of an MLIR-AIR program."* It emits a Chrome
trace (`README.md:78`). It models channel stalls — `RunnerNode.cpp:1447`: *"satisfy this, so
an unbalanced put/get pair still stalls instead of"*. It is a **timing** model, not a
functional oracle, so §1.4's distinction survives — but a jury that finds `air-runner` will
ask, and "no oracle" is a weaker sentence than r3 believed.

### S8. `air-verify-hierarchy-locality` runs **by default and in strict mode**

**Refines** §1.1 fact 2's *"The only `air-verify-*` pass is `air-verify-hierarchy-locality`"*,
which left the impression it is opt-in.

**Evidence:** `aircc.cpp:264` — `cl::init(PIV_error)`; `aircc.cpp:1213-1217` runs
`air-verify-hierarchy-locality{strict=true}` on the placed IR. `air-enforce-channel-fifo-order`
is also unconditional (`aircc.cpp:1169`). So two of the three checks the document treats as
"exists but un-run" are in fact on by default in every `aircc` invocation.

### S9. `air-broadcast-detection` fires only on `air.dma_memcpy_nd`, never on `air.channel.put/get`

**Refines** §1.1 fact 4's *"the pass `air-broadcast-detection` already **derives** multicast …
This is the baseline any hackathon DSL must beat"*.

**Evidence:** `AIRDependencyScheduleOpt.cpp:3328` walks `air::DmaMemcpyNdOp` only, requires an
enclosing `air.herd`, an L1 endpoint, and constant herd sizes (`:3386-3390`). It runs before
`air-dma-to-channel` in the aircc pipeline (`aircc.cpp:919-923`). **A surface that emits
channels directly is invisible to it**; only a surface that emits `ops.load`/`ops.store`
benefits. The baseline is real but narrower than stated.

---

### S10. Day 0 is six minutes, from a pip wheel built at this very commit

**Contradicts** the whole framing of §9 ("*E3 is an afternoon*", "*E4 and E5 are optional*",
"*If a device exists…*") and §6.5's seven-day plan, which budgets for a toolchain the team
does not yet have.

**Evidence:** `README.md:94` — *"Prebuilt wheels are the recommended path."*;
`docs/buildingRyzenLin.md:28-32` gives the three `-f` URLs; **run here in 5 min 43 s**,
yielding `mlir-air 0.0.1.2026091204+ff95a9b` — the wheel is built from the exact commit under
review. `air-opt`, `aircc`, `air-runner` and `import air` all work with **no environment
variables and no XRT**. §B.6, §C.7, §D.11 and §E.5 of this document were all *run* on it.
Only `--output-format=xclbin` needs XRT, and only for `xclbinutil`
(`docs/buildingRyzenLin.md:188-203` documents the hardware-free mode explicitly).

### S11. `air-opt` prints an error and exits 0

**Contradicts** the implicit assumption behind E3's recipe (*"Record whether each errors"*)
and behind any checker design that shells out to `air-opt` and reads `$?`.

**Evidence:** §B.6. The unmatched-put module produces
`error: 'air.channel.put' op found channel op not in pairs` on stderr and **`exit=0`**, then
prints the module and compilation continues through `air-to-aie`. The emitting site
(`mlir/lib/Util/Dependency.cpp:2065`) calls `emitOpError` without `signalPassFailure()`, and
the test that covers it says why — `channel_unpaired.mlir:8`: *"Verify that
air-dependency-canonicalize does not crash"*. **Our checker must parse diagnostics, not exit
codes.**

---

*Produced by reading `/home/adi/Projects/Honours/mlir-air` @ `ff95a9b3` and by running
`mlir_air 0.0.1.2026091204+ff95a9b` from a scratchpad venv. Probe scripts live under
`/tmp/claude-1000/…/scratchpad/air-probe/` and are deliberately **not** in this repo. No file
in `SegFault_2k26/hackathon/` other than this one was created or modified. Nothing was
committed or pushed.*
