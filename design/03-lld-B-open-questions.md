# Person B — open questions, resolved

*Phase 1 LLD companion, 2026-09-12. Owner: **Person B** (M4 mapping, M5 emitter).
Reads on top of [`01-requirements.md`](01-requirements.md) §7, [`02-hld.md`](02-hld.md) §7 and
[`06-interfaces.md`](06-interfaces.md). Every verdict below is either a `path:line` citation into
the mlir-air clone at commit `ff95a9b` (`/home/adi/Projects/Honours/mlir-air`) or a probe
transcript. Probe sources and outputs live only in the scratchpad — `$PROBE =
/tmp/claude-1000/-home-adi-Projects-Honours-amd-npus/99be7b10-5bc0-4903-b647-2001f42971f1/scratchpad/air-probe/q`
— and are **upstream-API probes**, not our DSL. Nothing in them is committed.*

---

## 0. Probe environment

```
source $PROBE/../venv/bin/activate
SP=$(python -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
export PATH=$SP/mlir_air/bin:$SP/mlir_aie/bin:$PATH
export PEANO_INSTALL_DIR=$SP/llvm-aie
```

`importlib.metadata.version("mlir_air")` → `0.0.1.2026091204+ff95a9b`, which **matches the
FR-T6 pin**. Every `aircc` invocation below is
`aircc --device npu1 --output-format=none <module.mlir>`, and "passes" means **exit 0 *and* zero
lines matching `error:` on stderr** (FR-T5).

| Probe | File | What it is |
|---|---|---|
| P-A | `$PROBE/q7_a.py` | HLD §7.1's W1 shape, verbatim: segment-scope producer loop **before** the herd |
| P-B | `$PROBE/q2_w2.py`, `q2_w2v.py` | HLD §7.2's W2 halo, `size=[PI-1]`, `h.private()` strips |
| P-C | `$PROBE/q6_w2_3herd.py`, `q6_w2_3herd_same.py` | W2 split into load / compute / drain herds over `seg.per_core()` strips |
| P-D | `$PROBE/w3.py`, `w3b.py` | HLD §7.3's W3 wavefront, single `PJ+1` bundle vs three homogeneous channels |
| P-E | `$PROBE/flip.py`, `flip2.py` | W1-flip cascade, `size=[PK-1]`, `npu_cascade`, no `at=` |
| — | upstream | `programming_examples/cascade_reduction/cascade_reduction.py -p` → `aircc` |

---

## 1. Q-7 — may a segment-scope loop of puts precede the herd that consumes them?

> *Does a segment-scope loop of `K/TK` puts, issued before the `air.herd` op that consumes them
> one per iteration, make progress — or must the herd sit inside the producer loop as in
> `matrix_multiplication/bf16/run.py:161-165`?* — owner B, due D1.

### Verdict: **YES — the producer loop may precede the herd. HLD §7.1's shape stands unchanged. Q-7 is closed; its fallback (restructure to upstream's L2-staged shape) is not taken.**

**Evidence 1 — the idiom is upstream and hardware-CI-run.**
`programming_examples/channel_examples/broadcast_selective_capture/broadcast_selective_capture.py:64-67`
issues a Python loop of `NUM_TILES` puts at **launch** scope, *before* the `air.segment` that
contains the consuming herd; the herd body then takes `NUM_TILES` gets in its own loop
(`:84-88`). Its lit harness is `run_makefile_peano.lit:4` — `REQUIRES: ryzen_ai, peano`,
`CHECK: PASS!` — so it is numerically checked on a real device upstream.
`channel_examples/broadcast/single_herd/broadcast.py:55-59` is the one-put case of the same
shape (`chan_in.put(A)` then `air.herd(...)`), and `worker_to_worker.py:92-106` is the
bundle-indexed case (a `GRID_HEIGHT × GRID_WIDTH` Python loop of puts, then the herd).

**Evidence 2 — the scaled-up case builds, verifies and compiles.** P-A is HLD §7.1's W1 written
out in full (`M=N=K=64`, `TM=TN=32`, `TK=16`, `PI=PJ=2`, `f32`), with the A and B puts in a
**segment-scope `air.sequential(0,64,16)`** under a Python loop over the bundle index, and the
herd a *sibling after* them:

* `launch.build(target="npu1")` → a 127-line module, `module.operation.verify()` clean;
* `aircc --device npu1 --output-format=none` → **exit 0, zero `error:` lines** (22/22 steps).

**Evidence 3 — the token graph shows the herd is not ordered after the producers.** After
`air-opt --pass-pipeline="builtin.module(air-dependency)"` (`$PROBE/q7a_dep.mlir`) the four
producer loops each yield a token:

```
%3 = scf.for %arg13 = %c0 to %c64 step %c16 iter_args(%arg14 = %2) -> (!air.async.token) {
  %16 = air.channel.put async [%arg14]  @A2L1[%c0_19, %c0_20] (...) {id = 1 : i32}
```

and the herd takes **none of them**:

```
%10 = air.herd @gemm_herd async  tile (%arg13, %arg14) in (%arg15=%c1_10, %arg16=%c2) attributes {id = 1 : i32} {
```

There is no `[%3, %5, %7, %9]` dependency list on the `air.herd`. Producer and consumer are
concurrent async regions; the puts drain as the herd's gets consume them. Combined with the E1
verdict (VF §C.7 — no producer lock is ever initialised to 0, so a put into an empty channel
does not rendezvous) the producer loop cannot block waiting for the herd to be *entered*.

**Evidence 4 — the shape still fires ping-pong.** Running VF §E.5's labelling prefix on P-A
(`$PROBE/q7a_lab.mlir`) gives exactly one `unroll` attribute and two hoisted allocs:

```
111:            } {unroll = 2 : i32}
--- hoist_alloc occurrences: 2
```

so FR-E3/H-1 are satisfied by the same module.

### Consequence for the LLDs
`03-lld-M4-mapping.md` §6.1 and `03-lld-M5-emitter.md` §6.1 emit the W1 plan exactly as HLD §7.1
specifies. The L2 staging of upstream's bf16 GEMM is **not** adopted; our L1 tiles remain the
ping-pong candidates, and `double_buffer`'s message keeps its D-5 "the pass does the work"
reading for W1.

---

## 2. Q-2 — `size=[PI-1]` or `size=[PI]` for the halo bundles?

> *Does the W2 halo channel bundle need `size=[PI-1]` (one index per link) or `size=[PI]` with a
> guarded boundary?* — owner B, due D1.

### Verdict: **`size=[PI-1]` — one bundle index per physical link, no dead index. This is the recorded fallback, and it is adopted for a stronger reason than tidiness: it is the only convention under which no bundle-index expression is ever out of range.**

**The convention, stated once so the plan can refer to it.** For a 1-D herd of `PI` PEs with
coordinate `tx ∈ [0, PI)`:

| channel | `size` | meaning of index `k` | put site | get site |
|---|---|---|---|---|
| `ToNorth` | `[PI-1]` | the link PE `k+1` → PE `k` | PE `tx` puts at `[tx-1]`, guard `tx > 0` | PE `tx` gets at `[tx]`, guard `tx < PI-1` |
| `ToSouth` | `[PI-1]` | the link PE `k` → PE `k+1` | PE `tx` puts at `[tx]`, guard `tx < PI-1` | PE `tx` gets at `[tx-1]`, guard `tx > 0` |

Every index expression evaluates into `[0, PI-1)` **on the branch where it is emitted**:
`tx-1 ∈ [0, PI-2]` when `tx > 0`, `tx ∈ [0, PI-2]` when `tx < PI-1`.

**Why not `size=[PI]`.** Upstream's own chain uses the dead-index form —
`cascade_reduction.py:60-62` declares `size=[NUM_TILES]` and only ever touches indices
`0 … NUM_TILES-2`, leaving index `3` with zero puts and zero gets. It is legal (a key with
`0 == 0` is balanced), but for the halo it forces one of the two channels to name `tx+1` at the
last PE, i.e. an index expression whose value is `PI` — outside the bundle — on the branch where
it is *not* taken. `air-to-aie` folds that branch away once the coordinate is a literal
(`_cond.py:49-54`), so it would very likely survive; relying on a fold to keep an index in range
is the kind of thing that only fails on the one target where the fold does not happen.
`size=[PI-1]` removes the question.

**Probe.** P-B builds and verifies at `PI=4`
(`$PROBE/q2_w2.mlir`: `air.channel @ToNorth [3]`, `air.channel @ToSouth [3]`) and the guarded
sites come out exactly as planned — `scf.if` on `arith.cmpi sgt/slt`, `affine.apply ()[s0]->(s0-1)`
as the index:

```
scf.if %3 {
  %18 = affine.apply #map1()[%arg10]
  air.channel.put  @ToNorth[%18] (%alloc[1, 0] [1, 16] [16, 1]) : (memref<6x16xf32, 2 : i32>)
} else {
}
```

At `PI=2` the whole W2 module passes `aircc` (§5 below); at `PI=4` it hits an unrelated
**resource** limit, which is finding N-1 and is *not* a Q-2 issue — the same PI=4 module passes
`aircc` as soon as the per-core L3 staging is removed.

### Consequence for the LLDs
`02-hld.md` §7.2's table says `ToNorth`/`ToSouth` are `size=[4]` with "index `p` = the row PE `p`
sends to PE `p-1`". That is the `size=[PI]` convention. **The M4 LLD specifies `size=[PI-1]`
with the index table above**; HLD §7.2 needs the corresponding one-line edit. No frozen interface
changes (`ChannelPlan.size` is just a tuple), so this is an M4-internal decision under D-14, not
a `06-interfaces.md` change.

---

## 3. Q-6 — `h.private()` or `seg.per_core()` for the W2 strip?

> *Which of `seg.per_core()` vs `h.private()` does the W2 strip buffer need, given it must live
> across the `t` loop?* — owner B, due D1.

### Verdict: **`h.private()`, with the `t` loop inside a single herd body. The recorded fallback is correct, and the `seg.per_core()` alternative is actively wrong on this toolchain.**

**Why `h.private()` suffices.** `<herd>.private()`'s lifetime is *the herd body* (VF §D.2's
table). The `t` loop lives inside that body, so the `u`/`v` pair is allocated once per herd entry
and survives every timestep. `seg.per_core()` exists for a buffer that must outlive the herd body
— the case upstream's bf16 GEMM has, where the accumulator spans three separate herd entries
inside a segment-scope `k2` loop (`matrix_multiplication/bf16/run.py:138-149`, `scope=seg.shared()`).
W2 has one herd entry. P-B allocates both strips `h.private()` at herd-body top level and the
module builds, verifies, and (at `PI=2`) compiles.

**Why the `seg.per_core()` alternative fails.** The only reason to want segment-scope strips is
to move the L3 staging into separate herds and so free a per-core DMA channel (finding N-1).
P-C tests exactly that: `load_herd` / `jacobi_herd` / `drain_herd` over `seg.per_core()` strips.

* With **distinct herd names** it compiles (`aircc` exit 0) — but the ELF list shows the three
  herds were placed on **three different rows**:
  `elfs_jacobi_seg_core_{0..3}_2`, `…_3`, `…_4`. Twelve cores, not four. The `seg.per_core()`
  strip the load herd fills is not the memory the compute herd reads. It compiles and is
  semantically wrong.
* With **the same herd name** for all three (upstream's co-placement idiom — bf16 `run.py:151`,
  `:227`, `:233` all use `name="herd_0"`) the three herds do land on one row, and the module then
  fails with exactly the N-1 routing error. So co-placement and the DMA-channel relief are
  mutually exclusive.

### Consequence for the LLDs
`BufferPlan.scope == "herd.private"` for `u` and `v`, `loop_depth = 0` relative to the herd body,
`ping_pong_candidate = False` (they are outside the `t` loop — D-5's second meaning of
`double_buffer` applies, and the checker's message must say so). `seg.per_core()` is used
nowhere in W1/W2/W3/W1-flip. **Herd name is load-bearing**: two `air.herd` ops that must share
per-core state must carry the same `name=`; `HerdPlan.name` is therefore a plan field M5 must
emit verbatim, never derive.

---

## 4. Q-1 — does the cascade chain survive `aircc` on npu1?

> *Does the cascade chain of FR-M6 survive `aircc --output-format=none` on `npu1` with a 4-wide
> row herd and `at=` pinning?* — owner B, due D2.

### Verdict: **YES, and `at=` pinning is not required. One new constraint applies: the chain's direction is fixed by `aie.cascade_flow`'s verifier, and on npu1's 2-D physical herd that means a *descending* second coordinate.**

**Evidence 1 — upstream's own chain compiles.** `cascade_reduction.py -p --target npu1` emits

```
air.channel @chan_cascade [4] {channel_type = "npu_cascade"}
air.herd @herd_0  tile (%arg8, %arg9) in (%arg10=%c4, %arg11=%c1_1)
```

(note: a 1-D `[range(4)]` herd with `shape=(4,)` becomes a 4 × 1 tile space — the **first**
coordinate is the column). `aircc --device npu1 --output-format=none` → **exit 0, zero `error:`
lines**. The example declares **no `at=`**; `air-place-herds` picks a contiguous line on its own.

**Evidence 2 — our own shape compiles.** P-E is the W1-flip shape: a **2-D** herd
`[range(PI=2), range(PK=4)]` with `shape=(1,4)` (so `repeats=(2,1)`), a cascade bundle
`air.channel @CascadeK [3] {channel_type = "npu_cascade"}` — i.e. `size=[PK-1]`, no dead index,
no `broadcast_shape` — and head/middle/tail discrimination by nested `ops.branch`. First attempt
(`flip.py`, chain ascending in `ty`) **failed**:

```
error: 'aie.cascade_flow' op source tile must be to the North or West of the destination tile
```

Reversing the chain (`flip2.py`: head at `ty == PK-1`, tail at `ty == 0`, put at `[ty-1]`, get at
`[ty]`) → **exit 0, zero `error:` lines**, and the lowered design carries three cascade flows in
one column:

```
aie.cascade_flow(%tile_0_5, %tile_0_4)
aie.cascade_flow(%tile_0_4, %tile_0_3)
aie.cascade_flow(%tile_0_3, %tile_0_2)
```

**The rule, stated for M4.** `PHYSICAL_HERD["npu1"][2] == (1, 4)` (`_trace.py:88-91`), so a
**2-D** herd on npu1 is one column by four rows, and the cascade must run from the higher row to
the lower row — i.e. the chain index must **descend** in the second herd coordinate. A **1-D**
herd is four columns by one row and the chain runs **west to east**, ascending in the first
coordinate, which is the direction `cascade_reduction.py` uses. M4 must pick the orientation from
the herd rank, and the rule belongs in the plan (D-14), not in the emitter.

**`at=` is not planned.** Neither probe needed it, and `at=` removes `air-place-herds`' freedom
to avoid an occupied column. `HerdPlan.at` stays `None` for all four variants; the field remains
in the contract as the documented escape hatch if a device run ever shows a bad placement (R-03).

### One requirement that cannot be met as written
FR-M6 asks for "a chain of `PK-1` cascade channels" and FR-E8's acceptance test
`test_E8_cascade_text` expects `channel_type = "npu_cascade"` to appear **three times** for
`PK=4`. One bundle of `size=[PK-1]` prints the attribute **once**:
`air.channel @CascadeK [3] {channel_type = "npu_cascade"}`. Three separate `size=[1]` channels
would print it three times but would need `PK-1` guarded branches per PE instead of two, for no
gain. **Recommendation** (needs the architect, one line): keep one bundle, and restate the
acceptance as *"the plan contains `PK-1 == 3` cascade **links**, and the post-`air-to-aie` IR
contains three `aie.cascade_flow` ops"* — which is the fact that actually matters and which
`ir_facts.json`'s `cascade_channels` field (`06-interfaces.md` §8) already carries. The M4 LLD is
written to the one-bundle form.

---

## 5. N-1 (new) — W2 at `PI=4` exceeds the per-core inbound DMA budget

**This is the most consequential finding in this document and it changes the W2 fixture.**

P-B at `PI=4`, exactly as HLD §7.2 specifies (a `UIn` channel staging each PE's strip from L3, a
`UOut` channel draining it, and the two halo bundles), fails **late in `aircc`** — after AIR is
done, in mlir-aie's router:

```
error: 'aie.connect' op ; connecting (West: 3) to (DMA: 0) on TileID(1, 2) targets same dst as
another connect op; existing destinations: (East: 3), (South: 0), (West: 3), (DMA: 1), (DMA: 0)
```

An **interior** PE needs three inbound streams — its strip from L3, its north ghost from the east
neighbour, its south ghost from the west neighbour — and an AIE2 core tile has **two** S2MM DMA
channels. `TileID(1,2)` is column 1, row 2: the first interior PE.

**The budget, measured.** Removing only the `UIn` staging (`$PROBE/q2_w2v.py out_only`: strips
initialised on-core, `UOut` drain kept) **passes** — exit 0, zero errors — with these flows
(`air_project/aie.w2_out_only.mlir:553-562`):

```
aie.flow(%tile_1_2, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
aie.flow(%tile_1_2, DMA : 0, %tile_0_2,          DMA : 0)
aie.flow(%tile_1_2, DMA : 1, %tile_2_2,          DMA : 1)
aie.flow(%tile_0_2, DMA : 1, %tile_1_2,          DMA : 1)
aie.flow(%tile_2_2, DMA : 0, %tile_1_2,          DMA : 0)
```

So per core tile: **2 S2MM (inbound) and 2 MM2S (outbound)**, and the *third* outbound endpoint
was accommodated by **reusing MM2S:0 for two destinations** — the drain to the shim and the
`ToNorth` put to the west neighbour share one source channel. That is a silent behaviour, not an
error:

> **N-3.** An inbound overflow is a hard `aie.connect` error; an outbound overflow is silently
> merged onto one MM2S, which on a circuit-switched fabric is a multicast. Whether the merged
> stream is correctly demultiplexed at the two destinations is **not established** by a
> `--output-format=none` build and would need a device run. Treat >2 outbound endpoints per core
> as *unverified*, not as *fine*.

**What works.** `PI=2` — where every PE is a boundary PE and the counts are exactly 2 in / 2 out
with no sharing (`$PROBE/pi2/air_project/aie.w2_pi2.mlir:252-257`) — builds and passes `aircc`
with the full `UIn` + `UOut` + halo protocol, `h.private()` strips, `size=[PI-1]` bundles and the
unroll-by-two `t` loop. `PI=3` would put one interior PE back and fail again.

**Consequence for the LLDs and for the plan.** Two changes, both owned by B, neither touching a
frozen interface:

1. **The W2 fixture drops to `PI=2`** for every level that reaches `aircc` or a device
   (`04-test-plan.md` §4's `W2 H=W=16, PI=4, HS=4, T=4` becomes `PI=2, HS=8`; the L1 figure goes
   from `2 × 6 × 16 × 4 = 768` to `2 × 10 × 16 × 4 = 1280`, still trivial against 65 536). The
   `PI=4` schedule is kept as a **negative** fixture — see 2.
2. **M4 gains a P3 resource check** (`03-lld-M4-mapping.md` §3.8, code `DMA-CHANNELS`). This is
   the third of the four properties `docs/AIRCorrectnessChecker.md:15-20` names and does not
   implement, and it converts a 20-second `aircc` crash inside mlir-aie's router into one of our
   own four-part diagnostics naming the PE, the count and the clause to edit. It is the cheapest
   new differentiator in the whole design and it costs about half a day.
   **Amended after REVIEW-round1 P-R4**: the check counts **circuit-switched** core-to-core and
   non-packet L3 endpoints as *errors* and everything else as a *warning*. W3 with `q`/`r`
   staged has three logical inbound channels per core and compiles cleanly (`aircc` exit 0, zero
   `error:` lines), because its L3→L1 gets lower to `aie.packet_flow` and share one shim MM2S;
   `aie.w2pi4.mlir` has zero packet flows, which is why W2 at `PI=4` really does run out. A
   checker that counted logical channels would reject a program the toolchain accepts.

Two further alternatives were considered and are recorded so they are not re-litigated: staging
the strip through L2 does not help (the core still needs an inbound DMA), and a "seeding wave"
that hands strips east along `ToSouth` before the `t` loop does fit the budget but roughly triples
the protocol and its balance proof. Neither is taken inside the week.

---

## 6. N-2 (new) — a channel bundle may not mix L3 and core-to-core members

HLD §7.3's W3 uses **one** `West` bundle of extent `PJ+1`, where index `0` is fed from a
segment-scope L3 source, indices `1 … PJ-1` are core-to-core, and index `PJ` is drained to L3.
P-D (`$PROBE/w3.py`) builds and verifies, then fails in `aircc`:

```
error: 'airrt.dma_memcpy_nd' op failed to specialize channel bundle indices
error: failed to legalize operation 'func.func' that was explicitly marked illegal
```

from `mlir/lib/Conversion/AIRLoweringPass.cpp:798`. The surrounding code (`:770-799`) explains
it: an L3-attached channel op carries a `metadataArray` of shim allocations, and the bundle index
is used to select one (`air::getIndexToMetadataArrayFromChannelIndices`). The array only has
entries for members that *have* a shim allocation, so a bundle whose L3 members are not the
leading, contiguous ones cannot be specialised.

**The fix, and it is small.** Split the one bundle into three homogeneous channels and guard the
two ends:

| channel | `size` | endpoint kinds | sites |
|---|---|---|---|
| `WestIn` | `[1]` | L3 → L1 | segment puts `MQ` times at `[0]`; PE `0` gets, guard `tx == 0` |
| `West` | `[PJ-1]` | L1 → L1 | PE `tx` puts at `[tx]` (guard `tx < PJ-1`); PE `tx` gets at `[tx-1]` (guard `tx > 0`) |
| `EastOut` | `[1]` | L1 → L3 | PE `PJ-1` puts, guard `tx == PJ-1`; segment gets `MQ` times at `[0]` |

P-D's revised form (`$PROBE/w3b.py`) builds, verifies and **passes `aircc`** — exit 0, zero
`error:` lines — with

```
air.channel @WestIn [1]
air.channel @West [3]
air.channel @EastOut [1]
```

**Consequence for the LLDs.** FR-M5's *"a channel bundle of extent `PJ+1`, plus a constant source
at index 0 and a drain at index `PJ`"* is replaced by the three-channel form. Its acceptance test
`test_M5_wavefront_balance` survives unchanged in substance — there are still `1 + (PJ-1) + 1 =
PJ+1 = 5` channel indices, each with `put_count == get_count == MQ` — it just reads them across
three `ChannelPlan`s. `test_M5_uses_branch` gets *stronger*: the head and tail guards are now
genuinely required, where HLD §7.3 claimed "no per-core guard is needed". D-6's decision
(segment-scope source and drain rather than per-core boundary guards) survives in the part that
matters — there is still a constant source and a drain, and no PE has an unbalanced index — but
the two `ops.branch` guards are back.

**A per-core DMA recount for W3, so N-1 does not bite twice.** Per PE: inbound = one of
`WestIn`/`West` (they are on exclusive branches, so the router sees at most two: PE 0 has `WestIn`
only, PE `k>0` has `West[k-1]` only) → **1**. Outbound = one of `West`/`EastOut` → **1**, plus
the per-PE score drain `SOut` → **2**. Within budget at `PJ=4`, which the passing `aircc` run
confirms.

---

## 7. Smaller findings that change the plans or the goldens

| # | Finding | Evidence | Consequence |
|---|---|---|---|
| **N-4** | `air.herd`'s `name=` decides co-placement. Three herds with distinct names landed on rows 2, 3 and 4; renaming all three to one name put them on one row | P-C, ELF lists | `HerdPlan.name` is a plan decision, emitted verbatim (D-14). Upstream does the same: `bf16/run.py:151`, `:227`, `:233` are all `name="herd_0"` |
| **N-5** | `broadcast_shape` prints as `[2 : index, 2 : index]`, **not** `[2, 2]` | `$PROBE/q7_a.mlir:4` | FR-M2's acceptance string `air.channel @A2L1 [2, 1] {broadcast_shape = [2, 2]}` never matches. The golden must read `air.channel @A2L1 [2, 1] {broadcast_shape = [2 : index, 2 : index]}` |
| **N-6** | `ChannelPutOp::verify` rejects an index that **is** an `scf.for` IV (`scf::getForInductionVarOwner(idx)`, `AIRDialect.cpp:3586-3593`). An `affine.apply` *of* an IV passes — `air.api` itself emits one for the strip-mine repeat index | `AIRDialect.cpp:3586-3593`; `$PROBE/q7_a.mlir:74-76` | `06-interfaces.md` §5.2's invariant ("no `Expr` may reference a temporal loop IV") is **stricter than upstream**. Keep it: it is the conservative side, and M4's `BUNDLE-INDEX-IS-IV` check is then sound by construction |
| **N-7** | Strip-mining wraps the **whole herd body** in an `scf.for` over `repeats`, and the logical coordinate becomes `affine_map<()[s0,s1]->(s0*R + s1)>` | `$PROBE/q7_a.mlir:50-54, 74` | A buffer the plan calls "herd-body top level" is at IR depth 1 whenever `repeats > 1`. It is still not a ping-pong candidate (its first touch is a store, not a get), so H-1 is unaffected — but `BufferPlan.loop_depth` is a plan-space depth, and the M5 LLD says so explicitly |
| **N-8** | `air-broadcast-detection` finds **nothing** on our W1: `air-opt -air-dependency -air-broadcast-detection` → `broadcast_pattern` count `0` | `$PROBE/q7a_bc.mlir` | Confirms VF §S9 and D-4 by measurement. `w1.ir_facts.json`'s `broadcast_pattern_count` is **0**, and R-04's trigger is any change from 0 |
| **N-9** | A scalar `max` works at element level: `c[j] = ops.maximum(p[j-1] + 2, ops.maximum(p[j] - 1, c[j-1] - 1))` traces and lowers | P-D builds | FR-K4's 4-ary `max` needs no special handling; the plan carries it as `MaxMin("maximum", (a, b, c, d))` and M5 folds it right-to-left (`06-interfaces.md` §5.5) |
| **N-10** | A 1-D `air.herd` emits a 2-D tile space with the second extent 1 (`in (%c4, %c1)`) | `$PROBE/q2_w2.mlir:24`, `cascade.mlir:15` | The herd body still takes **one** positional coordinate; do not plan a second |

---

## 8. Still open, with fallbacks and due days

| # | Question | Owner | Due | Fallback |
|---|---|---|---|---|
| **B-O1** *(open; now also `01-requirements.md` §7 and risk R-19)* | Does the N-3 outbound MM2S merge deliver correct data to both destinations? Only a device run settles it | B (analysis), C (device) | D5 | Keep every core at ≤ 2 outbound endpoints. W1, W3 and W1-flip already are; W2 at `PI=2` is. M4's `DMA-CHANNELS` check counts outbound as well as inbound and warns at 3 |
| **B-O2** *(closed by REVIEW-round1 RULING 1: `U[T+1, H+2, W]`, `H = W = 16`, `PI = 2`, `HS = 8`, `PI·HS == H` — exactly the fallback this row proposed)* | W2's fixture arithmetic. `01-requirements.md` §3.2 gives the domain as `[1,H-1)×[1,W-1)`, i.e. `H-2` interior rows, but `04-test-plan.md` §4 sizes W2 as `H=16, PI=4, HS=4` — `PI·HS = 16 ≠ 14`. The strip staging in P-B reads `U[12:18]` off the end of a 16-row tensor as a result | A (checker), C (fixture) | D1 | Declare `U` as `[H+2, W]` with rows `0` and `H+1` the Dirichlet boundary and `PI·HS == H`; at `PI=2, HS=8, H=16` that is exact. B's plan assumes this shape |
| **B-O3** *(closed: all three are in the requirement text — HLD §7.2 `size=[PI-1]`, FR-M5's three channels, FR-M6/E8's `PK−1` links with `ir_facts.cascade_channels == 3`)* | Whether the architect accepts the three restatements this document forces: HLD §7.2's `size=[4]` → `[PI-1]`; FR-M5's single `PJ+1` bundle → three channels; FR-M6/E8's "three `npu_cascade` channels" → "three cascade links / three `aie.cascade_flow` ops" | architect | D1 | B implements the forms proved here regardless, since the alternatives do not compile; the FR text catches up |
| **B-O4** *(open; now also `01-requirements.md` §7 and risk R-21)* | Does everything above also hold for `--device npu2`? Only npu1 was probed | B | D2 | `npu2`'s caps are larger (`{1:(8,), 2:(2,4)}`, `_trace.py:88-91`) and its per-core DMA budget is not smaller, so npu1 passing is the binding case. Run the four modules through `aircc --device npu2 --output-format=none` at D2 and record |

*Q-3, Q-4 and Q-5 are Person C's and are untouched here.*
