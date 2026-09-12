# Test plan — Spatial DSL

*Phase 1, 2026-09-12. Owner: Person C (harness, fixtures, CI); each module owner writes their
own module's unit tests. Requirements are in [`01-requirements.md`](01-requirements.md);
contracts in [`06-interfaces.md`](06-interfaces.md).*

---

## 1. Levels

| Level | What it covers | Needs | Mark | Budget |
|---|---|---|---|---|
| **U — unit** | one module against hand-written inputs from the fixtures package | nothing | — | < 30 s total |
| **N — negative** | one failing schedule per legality/clause/grammar FR, asserting the `Diagnostic` | nothing | — | < 20 s total |
| **G — golden** | kernel + schedule → AIR text, mapping summary, plan JSON | `import air` | — | < 40 s total |
| **O — oracle** | the decorated kernel vs an independent numpy expression | numpy | — | < 20 s total |
| **I — IR inspection** | run `air-opt` with a named pass prefix and assert facts about the result | `air-opt` | `requires_air_opt` | < 40 s total |
| **S — toolchain smoke** | `aircc --device <t> --output-format=none` on each emitted module | `aircc`, `aiecc`, Peano | `slow`, `requires_aircc` | minutes; **excluded from the default run** |
| **D — device** | compile to `xclbin`, run on hardware, diff against the oracle | XRT + NPU | `requires_device` | skipped without hardware |

The default `pytest` invocation runs U, N, G, O and I, and must finish in **under 3 minutes**
(NFR-3). `pytest -m "slow or requires_device"` runs the rest.

---

## 2. Unit tests, per module

### M0 `model`
- every dataclass is frozen (mutating a field raises);
- every `__post_init__` invariant rejects a violating construction, one test per invariant in
  `06-interfaces.md` §2–§5;
- JSON round-trip of `ScheduleModel` and `MappingPlan` is stable and canonical.

### M1 frontend
- **Positive**: the three kernel sources produce the `KernelModel` stored in the fixtures — axes,
  access-map matrices, dependence vectors, reduction projection. Asserted field by field, not by
  repr.
- **Negative**: `test_grammar_rejects[...]`, one case per rejected construct, minimum set:
  `if`, `while`, `break`, `continue`, `return x`, a tuple assignment, a list comprehension, a
  lambda, `a.b`, a call to `abs`, a data-dependent subscript `A[B[i]]`, a non-affine subscript
  `A[i*j]`, `A[i**2]`, a `for` over a list, `range` with a non-affine bound, a missing
  annotation, an annotation that is not `sp.<dtype>[...]`. Each asserts `GrammarError`, the
  `code`, the construct name in `reason`, and a non-empty `fix`.
- **Oracle passthrough**: `test_kernel_call_is_oracle` (FR-S1) and `test_annotations_inert`
  (FR-S2).

### M2 schedule
- every clause records what it should and nothing else;
- `test_schedule_is_pure_data` (FR-S5) and `test_no_air_import_until_build` (FR-S20);
- `test_clause_errors[...]`, ≥ 20 cases spanning every `CLAUSE-*` code;
- `test_tile_handles` — `tile` returns usable outer/inner handles and a second `tile` on the
  inner handle also works.

### M3 legality
- one **positive** test per workload asserting the whole `LegalMapping` against a fixture:
  `sigma`, `pi`, `ker_pi`, `r_time`, `r_space`, `physical_herd`, `repeats`, `l1_bytes`;
- one **negative** test per `L*`/`STATIONARITY`/`HALO-*`/`CASCADE-*`/`PINGPONG-SHAPE` code
  (§5 lists them);
- `test_physical_herd_table` — a table test over `{npu1, npu2} × {1-D, 2-D} × several grid
  extents`, asserting the resolved physical shape is the largest divisor ≤ the cap
  (`_trace.py:88-91`, `_resolve_physical`);
- `test_l1_capacity_doubles` — the same buffer set with and without `double_buffer` gives byte
  counts differing by exactly the doubled buffers.

### M4 mapping
- `test_M1_trichotomy`, `test_M2_broadcast_shape`, `test_M4_halo_protocol`,
  `test_M5_wavefront_balance`, `test_M6_cascade_chain`, `test_M7_buffer_plan`,
  `test_M8_channel_plan_complete` (all defined in `01-requirements.md` §3.3);
- `test_M11_residency_line` — the summary's residency block (FR-M11, RULING 9): W1 prints
  `C: stationary (spatial), resident for the whole run` and `A: multicast along py,
  re-fetched per k0`; the W1-flip prints `B: stationary (spatial), resident for the whole run`
  and `A: stationary (spatial), re-fetched per i0`; a flip variant with `tile(ax.j, 32)`
  restored prints `B: … re-fetched per j0`, which is the regression the ruling exists to catch;
- **self-check negatives**, each on a hand-corrupted plan literal:
  - drop one `get` → `BALANCE`, with the per-key count table in `details`;
  - add a second `put` inside a loop body → `BALANCE` citing the per-iteration rule;
  - guard a put but not its matching get → `BALANCE` citing the per-branch rule;
  - reverse the halo order (get before put on both sides) → `CHANNEL-CYCLE` printing the cycle;
  - use the row-loop IV as a bundle index → `BUNDLE-INDEX-IS-IV`;
  - a broadcast channel whose fan-out set is under-consumed → `BALANCE` under the broadcast rule;
- `test_M12_plan_stable` — determinism across `PYTHONHASHSEED`.

### M5 emitter
- `test_E1_hierarchy`, `test_E2_native_body`, `test_E3_alloc_is_direct_child`,
  `test_E5_broadcast_emitted`, `test_E6_memory_spaces`, `test_E8_cascade_text`,
  `test_E9_verify_surfaced`, `test_E10_byte_identical`;
- `test_no_buffer_resources_arg` — a source-level assertion that the emitter never names
  `buffer_resources` (FR-E4; `air.api` raises on it, `_channel.py:563`);
- `test_emitter_makes_no_decisions` — a lint test asserting the emitter module contains no
  branch on `LegalMapping` or `ScheduleModel` fields; every decision lives in M4 (HLD §2).

### M6 toolchain
- `test_T1_targets`, `test_T3_xclbin_message`, `test_T5_error_without_exit_code`,
  `test_T6_versions`;
- `test_stderr_parser` — feeds the parser recorded stderr samples (one clean, one with
  `error: 'air.channel.put' op found channel op not in pairs`, one `aircc` failure) and asserts
  the verdict, **including the case where the exit code is 0 and the verdict is failure**
  (VF §B.6, §S11).

---

## 3. Integration tests

### 3.1 Golden pipeline (level G)
For each of the four variants — `W1`, `W1-flip`, `W2`, `W3` — and each target in
`{npu1, npu2}`:
1. build the schedule from the committed source;
2. `s.plan()` → compare against `*.plan.json`;
3. `s.summary()` → compare against `*.summary.txt`;
4. `s.mlir()` → compare against `*.<target>.air.mlir`, byte for byte.

Eight golden module files total (4 variants × 2 targets); the plan and summary goldens are
target-independent except for `physical_herd`/`repeats`, so those are stored per target too.

### 3.2 IR inspection (level I)
Each test runs `air-opt` on the emitted module with a named pass prefix and asserts one number,
recorded in `*.ir_facts.json`.

| Test | Pipeline | Assertion | Source |
|---|---|---|---|
| `test_I_pingpong_labels` | the measured two-pass form: `builtin.module(air-dependency,air-label-scf-for-to-ping-pong{device=npu1})` | the K loop carries `unroll = 2 : i32`; both L1 tiles carry `hoist_alloc = true` | `03-lld-M6-toolchain.md` §3.7, measured. The 16-pass VF §E.5 prefix is the **recorded fallback**: if the two-pass form ever stops labelling, the test switches to it and says so in the skip reason |
| `test_I_pingpong_transform` | the above plus `air-ping-pong-transform, canonicalize, cse` | the K loop's step doubles and the loop gains 4 `!air.token` `iter_args` | VF §E.5 |
| `test_I_broadcast_count` | `air-dependency, air-broadcast-detection` | the number of `broadcast_pattern` attributes equals the golden count | VF §E.4, §S9; R-04 |
| `test_I_unpaired_channel` | `air-dependency, air-dependency-canonicalize` on a **deliberately unbalanced** module | stderr contains `found channel op not in pairs` **and the exit code is 0** — this is the test that justifies FR-T5 | VF §B.6, §S11 |
| `test_I_lock_inits` *(optional, D6)* | `air-to-aie{device=npu1}` on W1 | histogram of `aie.lock … {init = N}` matches the golden; no producer lock has `init = 0` | VF §C.7 |

`test_I_unpaired_channel` is the one test that uses a hand-written broken module rather than our
emitter's output. It is a *characterisation* test of the toolchain, not of us: if upstream ever
starts failing on this, the test breaks loudly and FR-T5 can be relaxed.

### 3.3 Oracle tests (level O)
For each workload: run the decorated kernel on `inputs.npz`, compare against `expected.npz`
(computed by an independent numpy expression), exact for integer fixtures.
Plus `test_ignorability` (FR-S18) per workload, and `test_W2_oracle_is_timestep_outermost`
(FR-K3) — which asserts the kernel's own outermost loop is `t`, so the PE-outermost failure
measured in VF §I cannot arise for this surface.

### 3.4 Oracle vs emitted IR (semantics, where possible)
We cannot execute the emitted AIR on CPU as a functional oracle: `air.api` has no interpreter
(VF §D.10), `air-runner` is a **timing** model, not a functional one (VF §S7), and
`air/backend/cpu_backend.py` JITs a *lowered* module through torch-mlir's RefBackend, which
"tests the compiler's output, not the user's source" (PC §1.4 item 1). **So the semantic link
between oracle and emitted IR is established structurally, not by execution**, with three
checks:

1. `test_sem_access_regions` — every `ChannelSite.region` reconstructed back into an index set
   equals the index set the corresponding `AccessMap` produces for that PE's iteration subdomain.
   This is the check that catches a wrong slice expression, which is the class of bug the
   oracle alone cannot catch.
2. `test_sem_compute_nodes` — the `StoreNode` sequence in the plan, replayed as Python over
   numpy arrays with the same subscripts, reproduces the oracle's output for a one-PE grid
   (`PI=PJ=1`). This executes *our plan's arithmetic*, not the emitted MLIR, but it closes the
   gap between "the kernel is right" and "the plan says the same thing".
3. `test_sem_coverage` — two assertions, so that a partial drain fails **loudly** rather than
   by accident: (a) the union over PEs of the written index sets equals the write domain the
   plan **claims to drain**, with no overlap; and (b) the plan's claimed drain domain equals the
   kernel's full write domain. Both of our stencil workloads drain every plane (W2) and every
   row (W3) precisely so that (b) holds — see `03-lld-M4-mapping.md` §6.3 and §6.4. This catches
   a partition off-by-one, and it is the structural analogue of a data-race check.

**Only the device test (level D) establishes end-to-end semantics.** This is stated on the
honest-limits slide.

### 3.5 Toolchain smoke (level S, `slow`)
For each of the four variants and each target: write `s.mlir()` to a temp file, run
`aircc --device <target> --output-format=none <file>` and assert (a) exit 0 **and** (b) no
`error:` line on stderr. One variant additionally runs `--output-format=pdi` and asserts the
`.pdi` file exists. Both formats are measured to work with no XRT and no device (VF §G.5).

### 3.6 Device (level D, `requires_device`)
Compile with `output_format="xclbin"`, load through XRT, run on `inputs.npz`, diff against
`expected.npz` with the fixture's `tol`. Report `max_abs_err` and the mismatch count with its
denominator. Skipped with a clear reason when `/dev/accel*` is absent.

---

## 4. Fixtures

Sized so that the oracle runs in well under a second in CPython and the emitted module stays
small enough to read.

| Workload | Parameters | dtype | Oracle reference | Bytes in L1 (per core, doubled where ping-ponged) |
|---|---|---|---|---|
| **W1** GEMM OS | `M=N=K=64`, `TM=TN=32`, `TK=16`, `PI=PJ=2` | `f32`, integer-valued in `[-8, 8)` | `A @ B` in numpy, exact | `4096 + 2·2048 + 2·2048 = 12 288` of 65 536 |
| **W1-flip** WS | `M=N=K=64`, `PK=4`, grid `(4,)`, `TM=32`, `TK=16`, **`j` untiled** (`TN = N = 64`) | same | same | `8192 + 8192 + 2·2048 + 4096 = 24 576` (`acc [32,64]`, `recv [32,64]`, `a [32,16]` doubled by `double_buffer("A")`, `b [16,64]`) |
| **W2** Jacobi | `T=4`, `H=W=16`, `U: [T+1, H+2, W]`, `PI=2`, `HS=8` | `f32`, integer-valued | a two-loop numpy Jacobi over `T` steps | `2 × (8+2) × 16 × 4 = 1280` |
| **W3** SW | `MQ=NR=32`, `PJ=4`, `CW=8`, `GAP=1`, `MATCH=+2`, `MISMATCH=-1` | `i32` | a textbook two-loop DP in numpy | `128 (q) + 32 (r) + 2 × 9 × 4 (prev/cur) + 2 × 4 (edges) = 240` |

`j` is untiled on the W1-flip **so that the flip is genuinely weight-stationary** (RULING 9):
each PE's `B[kchunk, :]` tile is then constant for the whole run, while `A` is re-fetched on
each of the two `i0` trips. `double_buffer("A")` only — `double_buffer("B")` is a
`PINGPONG-SHAPE` negative. M3's own L1 figure for the flip is `16 384`; the `24 576` above is
the plan figure, which includes the cascade `recv` tile M4 synthesises (`02-hld.md` §7).

`PI·HS == H` is an invariant of the W2 fixture. `PI = 4` is a **`DMA-CHANNELS` negative**, not a
positive fixture: it is measured to fail with `'aie.connect' op … TileID(1, 2) targets same dst`
on the first interior PE (REVIEW-round1 P-R2), so the checker must reject it before `aircc` does.
A `T = 5` variant of the W2 fixture (`w2_odd`) exercises the odd trip count, and `T = 0`
(`w2_zero_t`) is a `SWAP-PARITY` negative.

Rules (repeated from `06-interfaces.md` §9 because they are the ones people break):
`expected.npz` is **never** produced by our own oracle; the seed is fixed and recorded; integer
values so `tol = 0.0`; and `oracle.npz` is committed so a CPython regression shows up as a file
diff rather than as a silently changed comparison.

A second, larger set of parameters is kept **only** for the toolchain-smoke and device levels
(`W1` at `M=N=K=256`, `TM=TN=TK=64`, grid `(4,4)` — the shape VF §E.5 actually measured, with
`A`/`B` in `bf16` and `C` accumulating in `f32`; in `f32` throughout the same shape is
`3 × 64 × 64 × 4 = 49 152` undoubled and `81 920` charged, which FR-L9 rejects), so
that the ping-pong and lock-init facts are asserted on a module of the same size upstream
exercised.

---

## 5. Negative-test corpus

One test per code in `06-interfaces.md` §6.3 — **43 codes, 43 tests minimum** — each asserting
`code`, a non-empty `reason`, a non-empty `fix`, and that `details` contains the numbers named
in the corresponding FR. Three of these are the demo's set pieces and get a dedicated,
hand-tuned message:

1. **`L2-CAUSALITY` on W3** — `skew(time=(ax.i,))`, the **dropped** `j0` term. (`skew` is a sum,
   so reversing two terms is a no-op; dropping one is not.) The message prints the violating
   dependence vector and the product that came out `≤ 0`. This is the rejection the pitch shows.
2. **`STATIONARITY` on W1** — `place(px=ax.i0, py=ax.k0)` with `stationary("C")`. The message
   prints `ker M_C` and `ker Sπ` and says which containment failed.
3. **`L1-CAPACITY`** — `M=N=K=192`, `TM=TN=96`, `TK=32`, `f32`, `double_buffer("A","B")`: a tile
   set that fits at `61 440` B until ping-pong doubles `A` and `B` to `86 016` B, `20 480` over
   the 65 536 budget. The message prints the per-buffer breakdown, the doubled total, and the
   budget, and names the clause to edit (`tile` or `double_buffer`).

`test_D3_catalogue_complete` asserts the corpus covers every code and that no code is raised
that the catalogue does not list.

---

## 6. CI

- **Runner**: a stock CPU machine, no NPU, no XRT — which is enough for everything except level
  D (VF §G.4, §G.5, measured).
- **Install**: the pinned recipe from `07-environment.md`, from the four cached wheels
  (R-13), not from the `latest-*` index, so a pruned release asset cannot break the build.
- **Steps**: install → `test_T6_versions` (fails fast on a pin mismatch) → `pytest` (default
  marks) with a wall-clock assertion of 3 minutes → `pytest -m "slow"` on a nightly schedule
  only.
- **Network**: disabled after the install step (NFR-6).
- **Determinism gate**: the default run is repeated once with a different `PYTHONHASHSEED`; any
  golden diff between the two runs fails the build (NFR-1).
- **Artifacts**: on failure, CI uploads the emitted `.air.mlir` and the captured stderr for
  every failing golden, so a golden diff can be read without re-running.

---

## 7. Traceability

Every FR in `01-requirements.md` names its acceptance test inline. The table below is the
inverse index, kept in `tests/README.md` and asserted by `test_traceability` — which parses the
FR ids out of `01-requirements.md` and asserts every one appears in **at least one** test's
`@pytest.mark.fr("FR-…")` marker, and that no marker names an FR that does not exist. (Three
tests legitimately carry two FR ids — `03-lld-M7-tests.md` §6.3 — so "exactly one" would fail on
a correct suite.)

| FR group | Tests | Level | Owner |
|---|---|---|---|
| FR-S1…S4 (kernel, grammar) | `test_kernel_call_is_oracle`, `test_annotations_inert`, `test_grammar_accepts`, `test_grammar_rejects[...]` | U, N, O | A |
| FR-S5…S17, S19, S20 (clauses) | `test_schedule_is_pure_data`, `test_clause_errors[...]`, `test_no_air_import_until_build`, per-clause unit tests | U, N | A |
| FR-S18 (ignorability) | `test_ignorability` × 3 | O | A |
| FR-L1…L14 | one positive + one negative per code | U, N | A |
| FR-M1…M12 | the M4 unit set of §2 | U, N | B |
| FR-E1…E10 | the M5 unit set of §2 + the golden pipeline | U, G | B |
| FR-T1…T6 | the M6 unit set + levels S and D | U, S, D | C |
| FR-D1…D3 | `test_D1_schema`, `test_D3_catalogue_complete` | N | A |
| FR-K1…K5 | `test_W1_end_to_end`, `test_W1_flip`, `test_W2_end_to_end`, `test_W3_end_to_end` | G, O, S | C |
| NFR-1…NFR-7 | `test_E10_byte_identical`, `test_M12_plan_stable`, `test_NFR2_deps`, the CI clock gate, `test_D1_schema`, `test_NFR5_docstrings`, the network gate, the `SpatialError` smoke test | U, CI | C |

---

## 8. Definition of "thoroughly tested"

The suite is done when **all** of the following hold. This is the checklist the D7 freeze uses.

1. Every FR in `01-requirements.md` has at least one test, and `test_traceability` passes.
2. Every error code in `06-interfaces.md` §6.3 is raised by at least one test, and no test
   raises a code the catalogue does not list.
3. All four variants (W1, W1-flip, W2, W3) produce byte-identical AIR text across two runs with
   different `PYTHONHASHSEED`, and match their goldens.
4. All four variants pass `aircc --device npu1 --output-format=none` **and** `--device npu2`,
   with no `error:` line on stderr.
5. W1, W1-flip and W3 diff exactly (`tol = 0.0`) on CPU; W2 diffs within `1e-5` absolute,
   because its kernel multiplies by `0.2` and `f32` rounding is not associative.
6. The ping-pong IR fact (`unroll = 2`) is asserted for W1, and the broadcast-pattern count is
   asserted for all four variants.
7. The three self-check negatives (`BALANCE` ×3, `CHANNEL-CYCLE`, `BUNDLE-INDEX-IS-IV`) all
   fire on corrupted plans.
8. The default run finishes in under 3 minutes on a laptop CPU.
9. Device tests are either **passing** or **skipped with a recorded reason** — never failing
   silently or xfailed without a note.
10. Every skipped test prints why it skipped, and the D7 rehearsal reads that list aloud once,
    because that list *is* the honest-limits slide.
