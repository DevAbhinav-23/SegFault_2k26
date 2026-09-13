# PROGRESS — Person B

*State file for Person B's work. Phase **P0a — repository setup**, 2026-09-13. Scaffold, pinned
toolchain, stubs. No module logic in this phase.*

---

## Done

| # | Item | Where |
|---|---|---|
| 1 | venv on the pinned interpreter (`uv venv --python /home/adi/.local/bin/python3.12 --seed`) | `.venv/` (git-ignored) |
| 2 | Wheel cache: **16** wheels, **644 MB**, resolved against the three `-f` release pages | `vendor/wheels/` (git-ignored), `vendor/wheels/SHA256SUMS` (committed) |
| 3 | Offline install from the cache — proves the cache is sufficient (R-13 mitigation) | — |
| 4 | `scripts/airenv.fish`, `scripts/airenv.sh` — 07-environment §4's env block, 7 lines each, with a "no mlir_air here" guard | `scripts/` |
| 5 | Repo skeleton: `spatial/` (8 modules), `kernels/`, `tests/` (helpers, unit, negative, integration, golden, fixtures{plans,mappings,corrupt,stderr,modules}), `demo/`, `vendor/`, `scripts/` | 07-environment §6 + M7 §2.1 |
| 6 | Eight `spatial/*.py` stubs: docstring naming owner + LLD, the §7.2 entry points with exact parameter names, each body `raise NotImplementedError`. `model.py` = docstring + `CONTRACT_VERSION = 3`. **No `import air` anywhere in `spatial/`** (FR-S20) | `spatial/` |
| 7 | `tests/conftest.py` — `--update-goldens` option + the five marks, M7 §2.2. Nothing else yet | `tests/conftest.py` |
| 8 | `pyproject.toml` — `spatial-dsl`, `requires-python = "==3.12.*"`, `numpy==2.5.3` only, dev extra `pytest`, `[tool.pytest.ini_options]` per M7 §2.2 | — |
| 9 | `.gitignore` extended: `vendor/wheels/*.whl`, `vendor/probes/`, `air_project/`, `*.pyc`, `.pytest_cache/`, `*.egg-info/` | — |
| 10 | `design/07-environment.md` §2's D0 block filled with 16 filenames + sha256s, and the "no `v*.*.*` tag" finding | — |
| 11 | Root `README.md` replaced with the quickstart | — |
| 12 | Probe sources copied for reference (venvs and build dirs excluded, 2.7 MB of the 3.2 GB) | `vendor/probes/` (git-ignored) |

### The wheel cache

The pin (07-environment §1) is four wheels; the cache is **16**, because `mlir_air[aie]` pulls
eight transitive dependencies (`ml_dtypes`, `filelock`, `aiofiles`, `cloudpickle`, `rich`,
`pygments`, `markdown-it-py`, `mdurl`) and `pytest` four more (`pluggy`, `iniconfig`,
`packaging`, and `pygments` again). Filenames and sha256s: `design/07-environment.md` §2 and
`vendor/wheels/SHA256SUMS`. Download: 4.5–5.8 MB/s, 55 s + 54 s + 29 s + 2 s for the four big
ones, ~3 min for the lot.

**No `v*.*.*` release tag exists on `Xilinx/mlir-air`** (only `latest-air-wheels` and
`latest-air-wheels-no-rtti`), verified 2026-09-13, so 07-environment §2 action 1 is impossible
and the local cache is the **only** mitigation for R-13 (asset pruning). Distribute
`vendor/wheels/` out of band; `SHA256SUMS` is the check.

---

## Verified (command → result)

Every line below was run. The five verify commands ran under `env -i` — **no environment
variables at all**, which is stronger than 07-environment §2's "with no environment variables
set".

| Command | Result |
|---|---|
| `env -i .venv/bin/air-opt --version` | `AOMP-23.0-60 (http://github.com/ROCm/aomp):` / ` Source ID:23.0-60-8b7c0c42edfe088700d312231739eff0dabd913c` / `  LLVM version 24.0.0` / `  Optimized build with assertions.` — exit **0**. (§2 predicted "LLVM version 24.0.0"; the banner carries three more lines) |
| `env -i .venv/bin/aircc --help` | exit **0**, 255 lines, first line `OVERVIEW: AIR Compiler Driver` |
| `env -i .venv/bin/python -c "import air.ir, air.passmanager; from air.dialects import air; print('ok')"` | `ok` — exit 0 |
| `env -i .venv/bin/python -c "from air import api as air; print('api ok')"` | `api ok` — exit 0 |
| `env -i .venv/bin/python -c "…m.version(…)"` | `0.0.1.2026091204+ff95a9b 1.4.3.dev55+g10767b5 22.0.0.2026091201+386ca5c6 2.5.3` — **the pin, exactly** |
| `pip install --no-index --find-links vendor/wheels 'mlir_air[aie]==…' pytest` | **succeeded, 11.5 s**, 16 packages, no network. The cache is sufficient |
| `source scripts/airenv.sh` (bash) and `scripts/airenv.fish` (fish) | both put `…/mlir_air/bin/aircc` and `…/mlir_aie/bin/aiecc` on `PATH` and set `PEANO_INSTALL_DIR=…/site-packages/llvm-aie` |
| `pip install -e . --no-deps` → `env -i .venv/bin/python -c "import spatial"` | `spatial ok 3` (`CONTRACT_VERSION`) |
| `.venv/bin/python -m pytest` | **0 tests collected, 0.14 s wall** (budget: < 2 s). Exit code **5** = pytest's `NO_TESTS_COLLECTED` — see *Open* below |
| `.venv/bin/python -m pytest --markers` | all five marks registered: `slow`, `requires_aircc`, `requires_air_opt`, `requires_device`, `fr` |

### Toolchain baseline, end to end (FR-T5 discipline)

`vendor/probes/q/q7_a.py` — the upstream-API W1 shape (`M=N=K=64, TM=TN=32, TK=16, PI=PJ=2`,
herd `shape=(1,2)`), **not** our DSL. No edit was needed to run it from its new location.

```
python q7_a.py > w1_base.mlir                 exit 0, 127 lines, empty stderr, 0.15 s
source scripts/airenv.sh
timeout 600 aircc --device npu1 --output-format=none w1_base.mlir
```

* **exit 0**
* **`error:` lines in stderr: 0** (`grep -c 'error:'` — the exit code alone is never trusted, FR-T5)
* **wall 1.017 s** (user 1.517 s — it parallelises)
* all **22/22** aiecc stages ran; two core ELFs produced,
  `elfs_gemm_seg_core_0_{2,3}/elfs_gemm_seg_core_0_{2,3}.elf`, 7 392 B each; `air_project/`
  holds `aie.w1_base.mlir` (30 325 B) and `npu.w1_base.mlir` (16 838 B)

**The toolchain compiles a W1-shaped module off-device, on this machine, from the cache.** Gate
G1's technical precondition is met on machine 1 of 3.

---

## Open / blockers

| # | Item | Detail |
|---|---|---|
| **B-P1** | `pytest` exits **5**, not 0, on an empty suite | M7 §9 step 1 asks for "collects zero tests and exits green in < 2 s". Zero collected is pytest's `NO_TESTS_COLLECTED` (exit 5) by design; `addopts` is set exactly as M7 §2.2 specifies and was not widened to mask it. It goes green the moment the first test lands. **No action proposed**; recorded so nobody re-diagnoses it. CI must not gate on `pytest` before D1 |
| **B-P2** | `tests/` is not a Python package | The brief's file list has no `tests/__init__.py`, but M7 §2.1 reaches fixtures with `importlib.resources.files("tests.fixtures")`. That resolves today only as an implicit namespace package from the repo root. **Person C decides at D1** whether to add `tests/__init__.py`; it is a one-line change either way |
| **B-P3** | `design/07-environment.md` §2 item 2 still reads "the four wheels" | The filled block immediately below it says 16 and lists them. Left as-is: the brief said change nothing else in that file. One-word edit whenever someone else touches §2 |
| **B-P4** | Toolchain installed on **1 of 3** machines | Machines 2 and 3 are Person C's D0 task (gate G1). `vendor/wheels/` must be moved by shared drive or USB — it is git-ignored and 644 MB |
| — | Nothing is blocked on another person. | |

Not verified: `aircc --output-format=pdi`, `air-opt` pass pipelines, `air-runner`, anything on a
device. None was in scope for this phase.

---

## Next phase

B's own critical path starts at M0/M4/M5. Before any of it:

1. **Sign `06-interfaces.md` v3** (`design/00-README.md` §4's signature block) — three signatures,
   still blank. M4 and M5 are written against it from D1.
2. **M0 `model` dataclasses** (Person A, D1) — `spatial/model.py` is a docstring and one constant
   until they land. B cannot type `m4.plan` without `LegalMapping` and `MappingPlan`.
3. **M5 skeleton at D1**, real at D2 (`design/03-lld-M5-emitter.md`); **M4 W1 at D2** — gate G2
   (W1 end to end) is D2.
4. `tests/fixtures/plans/w1_plan.py` — B's hand-written `MappingPlan` literal, M4's stand-in and
   later M5's input fixture (M7 §3.8). It is what makes `test_emitter_makes_no_decisions` mean
   anything.
5. Person C: machines 2 and 3, and the `.github/workflows/ci.yml` of M7 §3.9 — steps 2–4 restore
   from `vendor/wheels/SHA256SUMS`, so **CI never sees the `-f` release URLs**.

---

# Phase P0b — M0, the frozen shared contract

*2026-09-13. `spatial/model.py` at `CONTRACT_VERSION = 3`, its unit tests, and the test-harness
helpers. Nothing pushed.*

## Landed

| # | Item | Where |
|---|---|---|
| 1 | **M0** — every dataclass of `06-interfaces.md` §1–§6 (33 classes), frozen, tuple-valued, value-equal, hashable. `Dtype` enum with `.bits/.sizeof/.mlir/.numpy`; `Expr` canonical and structurally hashable; the `Literal` aliases plus a `frozenset` per alias; `Diagnostic`; `SpatialError` + the six leaf classes; `CATALOGUE` with all **43** codes | `spatial/model.py` |
| 2 | **73 enforced invariants** (`I01`–`I73`), each one line in the module docstring with its `06-interfaces.md` section, each with ≥ 1 negative test (**108** cases). Violations raise `TypeError`/`ValueError` naming class, field and value | `spatial/model.py` docstring |
| 3 | Canonical JSON: one generic encoder/decoder driven by dataclass fields and annotations, `__type__` tags for `Expr` and the `ExprNode`/`PlanNode` union members, `sort_keys=True, indent=2`, trailing `\n`. Round-trip exact and idempotent for all 33 classes | `to_json` / `from_json` |
| 4 | `check_pin()` + `PIN` (FR-T6, M6 §3.9). `importlib.metadata` imported inside the function; the other M6 stubs untouched | `spatial/m6_tools.py` |
| 5 | Harness helpers, spec-exact, marked "written by B at P0b to unblock; C owns": `assert_diagnostic` + `CATALOGUE` parsed from `06-interfaces.md` §6.3 (asserts 43 rows) + `RAISED_CODES`; `assert_golden`/`canonical_json`/`UPDATED`/terminal summary; `in_fresh_process` | `tests/helpers/{diagnostics,golden,determinism}.py` |
| 6 | `tests/conftest.py` — the five marks, `--update-goldens`, its **three guards** (CI env, pin mismatch, dirty `tests/golden`), and `tests.helpers.golden` registered as a plugin | `tests/conftest.py` |
| 7 | `tests/__init__.py` — **closes B-P2**: `tests` is now a package, so `importlib.resources.files("tests.fixtures")` resolves | `tests/__init__.py` |
| 8 | M0 unit tests: frozen-ness, hashability, no-list, one case per invariant, the Q-M2-3 *negative* rule, JSON, `Expr`, `Dtype`, the catalogue guard, rendering, the six leaf classes, `check_pin` | `tests/unit/test_m0_model.py` |

## Verified (command → result)

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest` | **328 passed, 0.33 s** reported (0.42 s wall including start-up). Budget NFR-3 is 180 s |
| `PYTHONHASHSEED=1 pytest -vv` vs `PYTHONHASHSEED=2 pytest -vv` | **identical**: `diff` of the 328 `PASSED` lines (ids, order, outcome) is empty |
| `env -i .venv/bin/python -W error -c "import spatial, spatial.model, spatial.m6_tools"` | `ok 3` — **no import-time warning**, no environment at all |
| `grep -rn 'import air\|from air' spatial/` | **no match** (FR-S20 holds) |
| `CI=1 pytest --update-goldens` | `UsageError: --update-goldens is refused in CI…` (guard layer 1) |
| `pytest --update-goldens` with `PIN["mlir_air"]` temporarily wrong | `UsageError: … is refused: the installed toolchain is not the pinned one…` + the `mismatched` pair (layer 2) |
| `pytest --update-goldens` with a stray file under `tests/golden` | `UsageError: tests/golden has uncommitted changes…` (layer 3) |
| `pytest --update-goldens` on a clean tree | passes, prints `--update-goldens rewrote 0 golden(s)` — the terminal-summary hook runs |
| `in_fresh_process(to_json, PLAN)` | equals the in-process text; a deliberately failing callable surfaces the child's stderr as an `AssertionError` |
| `check_pin()` in this venv | returns `None`; monkeypatching `metadata.version` raises `ToolchainError(TOOL-VERSION-PIN)` and passes `assert_diagnostic` |

Counts: **73** invariants / **108** negative cases / **33** classes with a minimal instance /
**43** catalogue codes, both in `model.CATALOGUE` and parsed from the document, asserted equal.

## Spec ambiguities, and the reading taken

1. **`SpatialError`'s base.** §6.2 reads `SpatialError(Diagnostic)`; HLD §4.1 and FR-D1 both say
   it *carries* a `Diagnostic`. Read as `SpatialError(Exception)` with a `.diagnostic`
   attribute — a `Diagnostic` is data, not an exception.
2. **`CATALOGUE`'s value type.** M7 §3.2 writes `CATALOGUE[code].stage`; the contract is
   `dict[str, Stage]`, so the helper indexes the stage directly.
3. **When a `clause` is required.** §6.1 requires it for `clause`/`legality`; M7 §3.2's
   `assert_diagnostic` requires it for *every* non-`grammar` stage. Both kept: M0 enforces the
   §6.1 rule, and `check_pin`'s diagnostic carries `clause="build()"` so the stricter helper
   passes. M6 §5's verbatim FR-T3 message also shows a toolchain diagnostic with a clause.
4. **No reference-existence checks at M0.** The brief excludes site→channel/buffer existence; the
   same reading is applied to every other *reference* (an `AccessMap.operand` naming a `Param`, a
   `Param.shape` entry naming a `shape_param`, `reductions` naming an axis). Those are M2/M3/M4.
5. **`ChannelPlan.sites` "≥ 1 put and ≥ 1 get" is not enforced.** `04-test-plan.md` §2's corrupt
   corpus reaches `BALANCE` by *dropping a get*; enforcing this at construction could make that
   code unreachable. Balance stays M4's self-check.
6. **`ChannelSite.id`'s `f"{channel}.{kind}.{order}@{scope}"` spelling is not enforced** — it is
   in the Meaning column, not the Invariant column, and the same `(channel, kind, order, scope)`
   can legitimately recur in two different loop bodies. M0 enforces non-emptiness, and that one
   id names one site across the plan (equal objects may appear twice, e.g. in `ChannelPlan.sites`
   and again in a body).
7. **Sorted means strictly increasing.** `stationary`, `double_buffer`, `sequential`,
   `shape_params`, `residency`, `stationary_ops`, `channels` are sorted *and* duplicate-free: a
   duplicate has no meaning and NFR-1 wants exactly one canonical form.
8. **Grid extents ≥ 1 and tile factors ≥ 1 are enforced** (stated in §3.2) even though
   `CLAUSE-BAD-VALUE` mentions them: M2 validates at clause-call time, before a `ScheduleModel`
   exists, so the code stays reachable. Grid *rank*, `len(place)==len(grid)`, tile-divides and
   `sequential ∩ place` are **not** enforced — `test_L11_rank`'s `grid=(2,2,2)` constructs, and
   there is a test asserting exactly that.
9. **`Statement.op` is set iff `kind == "accumulate"`** — from §2.4's Meaning column.
10. **`Diagnostic.details`** is typed `Mapping[str, object]` (§6.1) yet the object must hash. It
    is stored as an immutable `dict` subclass in sorted key order whose values are canonicalised
    (lists → tuples, nested mappings → the same type), so it stays `json.dumps`-able, hashable,
    and round-trips exactly. A `details` key literally named `__type__` is handled.
11. **`PIN` key spelling.** M6 §3.9 writes `llvm-aie`, the brief writes `llvm_aie`; both resolve
    through `importlib.metadata` (verified). Used `llvm_aie`, as the brief specifies.
12. **`check_pin(strict=True)`** — the pseudo-code never reads `strict`, so the parameter is not
    implemented.
13. **`Dtype.bf16.numpy`** returns `ml_dtypes.bfloat16`: numpy has no bf16. `ml_dtypes` is
    already a transitive dependency of `mlir_air[aie]`, so NFR-2's count is unchanged. The whole
    property is lazy, so `spatial.model` imports with no numpy present.

## Open / blockers

| # | Item | Detail |
|---|---|---|
| **B-P2** | **CLOSED** | `tests/__init__.py` added; `tests` is a package |
| **B-P1** | superseded | `pytest` exits 0 now that tests exist |
| **B-P5** | `assert_golden`'s *compare* path is unexercised | No golden file exists yet (M8/D2). Only the update path ran, on an empty set. The compare path is 10 lines and spec-shaped, but it is **not verified** |
| **B-P6** | `RAISED_CODES` is populated by exactly one code so far | `TOOL-VERSION-PIN`. `test_D3_catalogue_complete` (FR-D3) belongs to C's negative suite and is not written yet |

Not verified in this phase: anything on a device; `aircc`; `air-opt`; the network gate, the time
budget hook and the fixture fixtures of M7 §3.3/§3.6/§3.7 (Person C, later phase); the tensor
read-before-write ordering and every other M2/M3/M4 check, by design (Q-M2-3).

---

# Phase P0c — stub fixtures and the M6 off-device wrappers

*2026-09-13. The four `LegalMapping` literals, the W1 `MappingPlan` literal, `spatial/m6_tools.py`'s
off-device half, and the M6 fixtures and tests. Nothing pushed.*

## Landed

| # | Item | Where |
|---|---|---|
| 1 | **Four `LegalMapping` literals** — `kernel()`, `schedule(target)`, `legal(target)` each, both targets. Every value transcribed from M1 §6 / M2 §6 / M3 §6.1-§6.4 and HLD §7 | `tests/fixtures/mappings/{w1,w1flip,w2,w3}_legal.py` |
| 2 | Their shared helpers: `PHYSICAL_HERD`, `resolve_physical`, `tile_axis`, `const`/`lin`, `L1_BUDGET` | `tests/fixtures/mappings/__init__.py` |
| 3 | **The W1 `MappingPlan` literal** — segment body lines 0-9 and herd body lines 0-10 of M4 §6.1, node by node, with §3.9's summary | `tests/fixtures/plans/w1_plan.py` |
| 4 | **M6 off-device path**: `ToolRun`, `tool`, `tool_available`, `tool_env`, `invoke`, `ERROR_LINE`, `classify`/`FIX_FOR`, `verdict`, `artifact` (`none`/`pdi`/`xclbin` pre-check), `PIPELINES`, `EXTRACTORS`, `PIPELINE_OF`, `ir_facts`. `check_pin`/`PIN` unchanged. `has_device`/`run`/`diff`/`trace` stay `NotImplementedError` stubs marked "Person C, M6 §3.4-§3.6, §3.8" | `spatial/m6_tools.py` |
| 5 | `unpaired_channel.mlir` + a provenance header (see reading 1) | `tests/fixtures/modules/` |
| 6 | **Three stderr samples, recorded live**, plus a README carrying argv / exit code / date / the reproduction command for each | `tests/fixtures/stderr/` |
| 7 | Tests: 39 fixture-literal cases, 13 M6 unit cases, `test_I_unpaired_channel` (level I), `test_probe_w1_aircc_none` (level S, `slow`) | `tests/unit/`, `tests/integration/` |

`w1flip`, `w2` and `w3` **plans** are not written — only the W1 plan was in scope for P0c.

`w2_legal.legal(target="npu1", T=4, PI=2)` reproduces M3 §6.3 exactly; `T` moves the `t` extent,
the plane count and the recorded source line, and `PI` moves the grid, `HS = 16 // PI`, the
physical herd and `l1_bytes = 2·(HS+2)·W·4`. `T=5` (the FR-L14 peel), `T=0` (the `SWAP-PARITY`
negative) and `PI=4` (the `DMA-CHANNELS` negative, `l1_bytes = 768`) all construct.

## Verified (command → result)

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest` | **381 passed, 1 deselected, 0.55 s** (was 328; +53). Budget NFR-3 is 180 s |
| `.venv/bin/python -m pytest -m slow` | **1 passed, 381 deselected** — 2.40 s reported on the first (cold) run, 1.04 s on the second. The `aircc` smoke, run twice in all, one `aircc` at a time |
| `PYTHONHASHSEED=1` vs `=2` full run | both green; no field order differs |
| `grep -rn 'import air\|from air' spatial/` | exactly **two** hits, both `from air...` **inside** a function of `m6_tools.py` (FR-S20) |
| `python -c "import spatial.m6_tools, sys; print('air' in sys.modules)"` | `False` |
| `git status --porcelain` after the `slow` run | only the intended P0c files; `aircc` wrote nothing into the repo (invariant I-5) |
| all four `LegalMapping`s and the W1 `MappingPlan`, both targets | construct under M0's 73 invariants and round-trip through `to_json`/`from_json` exactly |

### The three recorded stderr samples (`tests/fixtures/stderr/README.md` has argv and dates)

| Sample | Tool | **Exit** | stderr |
|---|---|---|---|
| `clean.txt` | `air-opt <w1_base.mlir> -o /dev/null` | **0** | empty, 0 bytes — by measurement |
| `exit0_with_error.txt` | `air-opt tests/fixtures/modules/unpaired_channel.mlir -pass-pipeline='builtin.module(air-dependency,air-dependency-canonicalize)' -o /dev/null` | **0** | `error: 'air.channel.put' op found channel op not in pairs` + a `note:` line, 576 B |
| `exit1_with_error.txt` | `aircc --device npu1 --output-format=none --tmpdir air_project malformed.mlir` | **1** | `loc("malformed.mlir":15:39): error: use of undeclared SSA value name` + `Error parsing MLIR file: …`, 109 B |

A fourth measurement, not committed as a sample but recorded in the README: `aircc --device npu1
--output-format=none` on the **unmodified** `unpaired_channel.mlir` exits **0** and prints
`loc("-":9:7): error: 'air.channel.put' op found channel op not in pairs`. `aircc` reproduces the
FR-T5 trap too.

### Smoke result

`test_probe_w1_aircc_none`: imports `vendor/probes/q/q7_a.py` **in process** (no subprocess —
invariant I-1), writes its `launch.build(target="npu1")` text into `tmp_path`, and calls
`m6.artifact(..., "npu1", "none", workdir=tmp_path)`. **Passed**, returning
`<tmp_path>/air_project`, which exists as a directory. Skips with `"probe sources not present
(git-ignored)"` when `vendor/probes/` was not copied, and with the `[aie]`-extra reason when
`aircc` does not resolve.

## Spec readings taken, and the disagreements logged

1. **`unpaired_channel.mlir` is `vendor/probes/e3_i.mlir`, not `e3_i_unmatched_put.mlir`.**
   Measured: `e3_i_unmatched_put.mlir` uses `%c1` without defining it, so `air-opt` exits **1**
   with `error: use of undeclared SSA value name` — which is exactly what M6 §6.1's second
   paragraph says about that file. The module that produces `found channel op not in pairs` at
   **exit 0**, i.e. the one `04-test-plan.md` §3.2's `test_I_unpaired_channel` and M6 §6.1's
   reproduction need, is `e3_i.mlir`. The committed fixture is `e3_i.mlir` plus a six-line
   provenance header that says both things. **The brief's "copy `e3_i_unmatched_put.mlir` (352
   bytes)" cannot be satisfied and produce the required behaviour.**
2. **`ERROR_LINE` is widened by one alternative in its `loc` group** —
   `loc("<file>":<line>:<col>):` as well as `<file>:<line>:<col>:`. `aircc` uses the MLIR
   `loc(...)` spelling for **every** diagnostic, so M6 §3.2's pattern as written matches no
   `aircc` diagnostic at all and FR-T5 would have a hole exactly where it matters. Three live
   measurements are in `tests/fixtures/stderr/README.md`. Proposed as a one-alternative spec edit
   to M6 §3.2; the group names and the rest of the pattern are unchanged. **This is the one place
   P0c did not follow the letter of the spec, and it is the "never skimp on error handling" rule.**
3. **`clause` for the two `TOOL-MISSING-XCLBINUTIL` diagnostics is `build(output_format="xclbin")`,
   not `build()`.** The brief sets `build()` as the convention for toolchain diagnostics and that
   is what every other code uses, but M6 §5 prints the FR-T3 message **verbatim** with
   `in clause: build(output_format="xclbin")`, and `test_T3_xclbin_message` asserts "the message
   per M6 §5". The document's verbatim rendering wins for that one code.
4. **`m6.ir_facts` takes a fourth parameter `workdir=None`.** `06-interfaces.md` §7.2 lists three
   parameters; M6 §3.7's pseudo-code passes `workdir`. Defaulted, so both signatures call.
5. **`PIPELINE_OF["cascade_channels"] == "pingpong"`.** §3.7 fixes the pipeline for every other
   fact and not for this one. `channel_type = "npu_cascade"` sits on the module-level
   `air.channel` declarations and survives every pipeline, so it is grouped with `pingpong` to
   keep the common case at one `air-opt` run.
6. **A shape entry that is not a bare NAME is resolved to an `int`.** M1 §6.3 prints
   `S i32 ("MQ+1","NR+1")`, which M0 rejects two ways: **I06** requires a `str` shape entry to be
   a valid identifier (`"MQ+1".isidentifier()` is `False`), and `06-interfaces.md` §2.1's own
   invariant requires every `str` entry to appear in `shape_params` (which holds `MQ`, `NR`).
   M1 §3.1's EBNF agrees — `shape_entry ::= INT | NAME`. So: a bare NAME stays a `str` (W1 keeps
   `A ("M","K")` exactly as §6.1 prints), and anything else is resolved from the module-level
   integer bindings M1 §3.4 line 18 already reads. Hence `S (33, 33)` and `U (5, 18, "W")`.
   `shape_params` collects every NAME **appearing in** a shape entry, so W2 is `("H","T","W")` —
   otherwise M1 §3.4's `AFFINE(a, allowed = shape_params)` would reject `for i in range(1, H+1)`.
   **Person A's call to confirm**; recorded as B-P9.
7. **W2's dependence order is the sorted one.** M1 §6.2 prints
   `(1,0,0),(1,-1,0),(1,0,-1),(1,0,1),(1,1,0)` and calls it "sorted by vector"; under the tuple
   ordering M0's **I20** enforces the order is `(1,-1,0),(1,0,-1),(1,0,0),(1,0,1),(1,1,0)`. The
   literal uses the latter, or it would not construct. Recorded as B-P10.
8. **Tile-derived `Axis` fields M3 §3.3 does not fix**: `lo = 0`, `hi = extent`, `step = 1`, and
   `depth` = the axis's **index in the post-tiling coordinate order** (M2 §3.6). The index rule is
   applied to every entry of `LegalMapping.axes`, tiled or not, so the flip's untiled `j` is at
   `depth 2` there while `kernel.axes`'s `j` keeps its source depth of 1 — two frames, two depths.
9. **W1-flip's `ker_pi` is computed, not quoted.** M3 §6.2 prints `ker Sπ_u = span{e_i, e_j}` but
   no `ker Sπ` over `Coord`; the literal carries §3.2 `KERNEL_BASIS` applied to the printed
   `Sπ = [e_k0]` — one generator per free column, ascending: `i0, i1, j, k1`. Every other basis
   printed as `span{…}` is written out as the canonical row-echelon tuples it implies, with a
   comment at each site.
10. **W3's `reads` order** is the source order of the expression **after** `sub` is
    forward-substituted (M1 §3.5 line 15): `S[i-1,j-1], q[i-1], r[j-1], S[i-1,j], S[i,j-1]`.
    §6.3 gives the matrices and offsets but not the interleaving.
11. **`LoopPlan.depth` counts enclosing loops, 0 at the top of its own body.** M4 §6.1's herd body
    prints the two zeroing loops at `depth=1`/`depth=2` while its sibling `k0` loop is `depth=0`
    and the same section's buffer table puts `a`/`b` at `loop_depth 1` inside that loop; the
    segment body is consistent with the counting rule. The literal uses 0 and 1 for the zeroing
    loops.
12. **`ChannelSite.order` is the index in the enclosing body.** That is what §6.1's three given
    numbers pin: the `A2L1` put is the only child of its `k` loop (`order=0`), and the two herd
    gets are `2` and `3` after the two `BufferPlan` children of the `k` loop. Consequence:
    §5.2's "strictly increasing within a scope" holds **per enclosing body**, not per scope — all
    three segment-scope sites are `order = 0`, and `ChannelSite.id` stays plan-unique only because
    the channel name differs. Flagged.
13. **An `Expr` over a loop IV names the `LoopPlan.axis` that binds it.** §6.1 writes `pi`, `kk`,
    `pj`, `i`, `j` as short aliases for the loops it names `pi_bundle`, `k0`, `pj_bundle`,
    `i_drain`, `j_drain`.
14. **`plan.buffers` is in allocation order `(acc, a, b)`.** `06-interfaces.md` §5.6 says
    "non-L3 buffers, **in allocation order**"; M4 §3.3 line 23 says `sorted(out, key=name)`,
    which would give `(a, acc, b)`. The frozen contract wins, and it is also the order M8 §6.1's
    demo screen prints (`acc 4096 + a 2048x2 + b 2048x2`).
15. **The summary renders `{b.dtype}` as the MLIR spelling.** M4 §3.9 line 11 interpolates the
    `Dtype` enum, which Python renders `Dtype.f32`; the literal prints `f32`.
16. **W1's rendered reduction split is `R_time = span{e_k0, e_k1}`, `R_space = {}`.** §3.9 line 10
    always renders the **tiled** split and §6.1 does not print it for W1; neither `k0` nor `k1`
    is placed, so all of `R` is temporal. `LegalMapping.r_time`/`.r_space` stay untiled
    (`((0,0,1),)` / `()`), which is what M4's own classification reads.
17. **Region conventions** are M4 §3.4 unchanged: the L1 end of a whole-buffer transfer is the
    empty `Region((), (), ())`; the L3 end carries `offsets`/`sizes`/`strides` with row-major
    strides over the tensor shape. W1's `A2L1` put is `offsets=(pi_bundle*32, k0)`,
    `sizes=(32,16)`, `strides=(64,1)`, as §3.4 measured.

## Open / blockers

| # | Item | Detail |
|---|---|---|
| **B-P7** | `MappingPlan` cannot say **where the herd sits** inside `segment_body` | M5 §3.1 line 7 calls it "the `<HERD>` marker" and M4 §6.1 prints `6  <HERD>`, but `PlanNode` (`06-interfaces.md` §5.5) has no marker member and `MappingPlan` has no index field. The W1 literal keeps §6.1's node order — the two fill loops, then the drain loop — and M5 must place the herd between them by a rule nobody has written down. Needs either a contract addition (a marker node or a `herd_at` index) or a stated positional rule. **Architect's call; blocks nothing until M5 walks a plan.** |
| **B-P8** | `ERROR_LINE` in M6 §3.2 matches no `aircc` diagnostic | Reading 2. A one-alternative edit to the document; the code already carries the wider pattern and the evidence. |
| **B-P9** | M1 §6.3's `S i32 ("MQ+1","NR+1")` is not constructible under M0 | Reading 6. Person A's call: either the doc's notation is shorthand for the resolved `(33, 33)` (what the literal assumes) or M0 I06 / §2.1 must change, and M0 is frozen. |
| **B-P10** | M1 §6.2's dependence list is not in M0's sort order | Reading 7. A re-ordering of five lines in the document. |
| **B-P11** | `ChannelSite.order` is unique per enclosing body, not per scope | Reading 12. §5.2's invariant column overstates it for W1. |
| **B-P5** | still open | `assert_golden`'s compare path is still unexercised — no golden file exists yet |
| **B-P6** | narrowed | `RAISED_CODES` now carries `TOOL-VERSION-PIN`, `TOOL-DIAGNOSTIC`, `TOOL-AIRCC-FAILED`, `TOOL-TARGET`, `TOOL-MISSING-XCLBINUTIL` — five of the catalogue's 43 |

**Not verified in this phase**: anything on a device; `--output-format=pdi` and `xclbin` (the
`pdi`/`xclbin` branches of `artifact` are written and unit-tested only as far as the pre-check —
no `.pdi` has been produced); `ir_facts` against a real module (`PIPELINES`/`EXTRACTORS` are
tested against synthetic text and the `pairs` pipeline is exercised by
`test_I_unpaired_channel`, but no `pingpong_unroll == 2` has been measured on our own emitter's
output, because M5 does not exist yet); `air-runner`; `tool_env`'s Peano fallback branch on a
machine without `llvm-aie`; and every M3/M4 check — the four `LegalMapping`s are **literals**,
not checker output, so nothing here proves M3 will agree with them.

---

# Phase P0d — CONTRACT_VERSION 4 applied (architect ruling)

*Person B, 2026-09-13. The architect's ruling of 2026-09-13 on B-P7 and B-P9, plus the three LLD
corrections P0c's measurements forced. Two additions to the frozen interface, nothing else.
**Not pushed** — the architect verifies and pushes.*

## Landed

| # | What | Where |
|---|---|---|
| 1 | `CONTRACT_VERSION = 4`, with a *Version 4* paragraph naming both forcing requirements | `design/06-interfaces.md` header, `spatial/model.py` |
| 2 | **`HerdPlan` ∈ `PlanNode`.** `MappingPlan.segment_body` holds exactly one `HerdPlan`, at top level, `== MappingPlan.herd`; it marks where M5 opens `air.herd` and walks `herd_body`. Forced by FR-E1 + D-14 | `06-interfaces.md` §5.5, §5.6 (invariant **8**), `spatial/model.py` `I75`-`I77` |
| 3 | **`KernelModel.bindings`** — the integer value of every shape parameter at capture, one entry per `shape_params` name, sorted. Forced by FR-M7 (`TENSOR_PLAN` needs concrete L3 shapes) | `06-interfaces.md` §2.7, `spatial/model.py` `I74` |
| 4 | §2.1's `shape` meaning is now normative: a positive int, a shape-parameter NAME, or — for an expression that is not a bare NAME (`MQ + 1`) — the int it evaluates to under `bindings` at capture. That is P0c reading 6 | `06-interfaces.md` §2.1 |
| 5 | §5.2 `ChannelSite.order`: "strictly increasing within a **body**" | `06-interfaces.md` §5.2 |
| 6 | Signature row `06-interfaces.md v4` (unticked) and change-log row 4; §3's Interfaces row now reads `CONTRACT_VERSION = 4`, v4 pending signatures | `design/00-README.md` §3, §4 |
| 7 | `_TAGGED` gains `HerdPlan`, so a `segment_body` carrying the marker round-trips through §8's canonical JSON | `spatial/model.py` |
| 8 | Four `bindings` literals; the W1 plan's `segment_body` is now `(FILL_A, FILL_B, herd, DRAIN)` — the `HerdPlan` at **index 2**, which is §6.1 line 6's `<HERD>` | `tests/fixtures/mappings/*.py`, `tests/fixtures/plans/w1_plan.py` |
| 9 | Negatives for `I74`-`I77`; `test_w1_plan_marks_the_herd_position`; `bindings` asserted per literal and across the W2 parametrisation | `tests/unit/test_m0_model.py`, `tests/unit/test_fixture_literals.py` |
| 10 | M6 §3.2's `ERROR_LINE` replaced by the pattern in `spatial/m6_tools.py`, with the measured `exit1_with_error.txt` line cited | `design/03-lld-M6-toolchain.md` §3.2 |
| 11 | M4 §3.3 line 23 `sorted(out, key=name)` → `tuple(out)  # allocation order` — the frozen contract wins | `design/03-lld-M4-mapping.md` §3.3 |
| 12 | M4 §3.1 and M5 §3.1: the `<HERD>` marker **is** the `HerdPlan` node in `segment_body` | `design/03-lld-M4-mapping.md`, `design/03-lld-M5-emitter.md` |

Counts: **77** invariants (`I01`-`I77`) / **116** negative cases / 33 classes with a minimal
instance.

## Verified (command → result)

| # | Command | Result |
|---|---|---|
| 1 | `.venv/bin/python -m pytest` | **391 passed, 1 deselected** in 0.59 s (was 381 before this phase) |
| 2 | `PYTHONHASHSEED=1 .venv/bin/python -m pytest` | 391 passed, 1 deselected |
| 3 | `PYTHONHASHSEED=2 .venv/bin/python -m pytest` | 391 passed, 1 deselected |
| 4 | `.venv/bin/python -c "import spatial; print(spatial.CONTRACT_VERSION)"` | `4` |
| 5 | the M6 §3.2 regex compiled from the document, compared to `m6_tools.ERROR_LINE` and run over `tests/fixtures/stderr/exit1_with_error.txt` | patterns and flags identical; matches `loc("malformed.mlir":15:39): error: use of undeclared SSA value name` |

## Blockers closed

| # | Resolution |
|---|---|
| **B-P7** | **Closed by change 2.** The herd's position is a `HerdPlan` node in `segment_body`, not a positional rule: `PlanNode` gains the member, `06-interfaces.md` §5.6 invariant 8 states it, M0 enforces it (`I75`-`I77`) and the W1 literal carries it at index 2. M5 no longer has to infer anything (D-14). |
| **B-P9** | **Closed by changes 3 and 4.** The doc's `S i32 ("MQ+1","NR+1")` is shorthand: a shape entry that is not a bare NAME resolves to an int under `bindings` at capture, which §2.1 now says normatively. M0's `I06` and §2.1's "every `str` entry appears in `shape_params`" are unchanged, so the P0c literals stand as written. |
| **B-P8** | **Closed by change 10.** `03-lld-M6-toolchain.md` §3.2 now carries the two-alternative `loc` group already in `spatial/m6_tools.py`, and cites the measured line. Code and document agree; no code change was needed. |
| **B-P11** | **Closed by change 5.** §5.2's `order` invariant now reads "strictly increasing within a body", which is what the W1 literal does and what §6.1's three given numbers pin. |
| *(the `plan.buffers` note, P0c reading 14)* | **Closed by change 11.** M4 §3.3 line 23 no longer contradicts §5.6's "in allocation order". |

## One deviation from the ruling, for the architect to settle at signature time

The ruling's wording for `KernelModel.bindings` was, verbatim:

> `KernelModel` gains `bindings: tuple[tuple[str, int], ...]` — the integer value of every shape
> parameter at capture, sorted by name, **exactly one entry per `shape_params` name**, every
> value ≥ 1.

**The `every value ≥ 1` half is not enforced, and §2.7 records the exception.** Enforcing it at
M0 would make a §6.3 error code unreachable: `03-lld-M3-checker.md` §9 `test_L14_swap_parity`
says of W2's `T = 0` fixture that

> `T = 0` is the **only** reachable `SWAP-PARITY` condition after the D-4 override, so without
> this fixture `test_D3_catalogue_complete` fails

and `T` is in W2's `shape_params`, so `bindings` must be able to carry `("T", 0)`. The same
fixture is named in `04-test-plan.md` §4, `03-lld-M8-kernels-demo.md` §4, `REVIEW-round1.md`
G-11 and `01-requirements.md` Q-C16. `spatial/model.py`'s own Q-M2-3 rule — M0 enforces only
structural, object-local invariants, "so that the error codes of `06-interfaces.md` §6.3 stay
reachable" — points the same way. M0 therefore checks the names and the order; §2.7's invariant
column states the bound **and** the exception, and `I74`'s docstring line says why.
**If the architect prefers the bound enforced, the `w2_zero_t` negative needs another vehicle
first.**

Two smaller corrections, applied silently because they are citations, not decisions: the brief
cited `Q-M1-2` as `M3 §10`; it is **`03-lld-M1-frontend.md` §10**, and that is what §6.4's
Version 4 paragraph cites. And `w1flip_legal.py` needed no `bindings` edit — it imports
`kernel` from `w1_legal`, so FR-K2's "the flip is a schedule edit, not a source edit" gives it
the W1 bindings for free; `test_fixture_literals` asserts the value for both.

## Open / blockers

| # | Item | Detail |
|---|---|---|
| **v4 signatures** | A and C have not signed | `00-README.md` §4's process needs all three. **A** is affected: M1 must now emit `bindings` (it already reads the integer bindings at `03-lld-M1-frontend.md` §3.4 line 18) and M3 passes the `KernelModel` through unchanged. **C** is affected: nothing in M6/M7/M8 may construct a `HerdPlan` node anywhere but the top level of `segment_body`, and any hand-written plan fixture needs the marker. |
| **B-P10** | still open | M1 §6.2's dependence list is not in M0's sort order — a re-ordering of five lines in the document, Person A's. |
| **B-P5**, **B-P6** | unchanged from P0c | — |

**Not verified in this phase**: nothing new was measured. The one measurement quoted (the
`aircc` `loc(...)` spelling) is P0c's, re-checked against the committed fixture rather than
re-run against the tool. M3 and M4 still do not exist, so nothing here proves the checker will
produce these `bindings` or that M4 will place the `HerdPlan` where the W1 literal does.

---

# Phase P1 — M5, the AIR emitter

*2026-09-13. Branch `role-b`. `spatial/m5_emit.py` driven by the hand-written W1 `MappingPlan`
literal, through `air.api` to text, verified against the upstream-API probe, `air-opt` and
`aircc`. **Nothing here was pushed.***

## Landed

| # | What | Where |
|---|---|---|
| 1 | `emit(plan, target) -> EmitResult`: all thirteen rows of M5 §3.2, §3.4's binding, §3.6's compute walk, §3.7's slices, §3.8's build and text capture. 450 lines, one public name | `spatial/m5_emit.py` |
| 2 | `import air` happens **inside** `emit()` and nowhere else (FR-S20) | same |
| 3 | The five §5 precondition checks: bundle-index-is-IV, ping-pong direct child, row-major strides, allocation scope, `l1_peak` vs the plan's L1 total. Each is an `EmissionError` whose `fix` asks for a bug report and whose `details` carry `internal_consistency: true` | same |
| 4 | 21 unit cases, including the three lint tests that make D-14 mechanical | `tests/unit/test_m5_emit.py` |
| 5 | Goldens for W1 × {npu1, npu2}: module text, canonical plan JSON, the summary | `tests/golden/w1.base.*` |
| 6 | `ir_facts` tests and the facts golden | `tests/integration/test_ir_facts.py`, `tests/golden/w1.base.npu1.ir_facts.json` |
| 7 | `aircc` smoke on **our** module for both targets plus one `pdi` build | `tests/integration/test_smoke.py` |

## Verified (command → result)

| # | Command | Result |
|---|---|---|
| 1 | `.venv/bin/python -m pytest` | **420 passed, 4 deselected** in 1.17 s (was 391 passed, 1 deselected) |
| 2 | `PYTHONHASHSEED=1 … -m pytest` / `PYTHONHASHSEED=2 … -m pytest` | 420 passed, 4 deselected, 1.22 s / 1.19 s |
| 3 | `.venv/bin/python -m pytest -m slow` (`tests/integration/test_smoke.py`) | 4 passed in 3.3 s |
| 4 | `.venv/bin/python -c "import spatial.m5_emit, sys; print('air' in sys.modules)"` | `False` (FR-S20) |
| 5 | `aircc --device npu1 --output-format=none` on the emitted `w1.base.npu1.air.mlir` | **exit 0**, 0.88 s, zero `error:` lines, 22/22 stages, two core ELFs (`gemm_seg_core_0_2`, `_0_3`) |
| 6 | `aircc --device npu2 --output-format=none` on `w1.base.npu2.air.mlir` | **exit 0**, 0.30 s, zero `error:` lines, **four** core ELFs (`0_2`, `0_3`, `1_2`, `1_3`) — the 2×2 physical herd, so the compile is real and not a short circuit. **B-O4 answered for W1: npu2 needs no `xfail`** |
| 7 | `aircc --device npu1 --output-format=pdi` | exit 0, 0.90 s, `air.pdi` written |
| 8 | `air-opt <file> -o /dev/null` on the emitted text (`test_E7_text_roundtrip`) | exit 0, no `error:` line |
| 9 | `ir_facts` on the emitted npu1 W1 | `pingpong_unroll = 2`, `hoist_alloc_count = 2`, `broadcast_pattern_count = **0**`, `cascade_channels = 0`, `pingpong_iter_args = 4` |
| 10 | `air-opt -pass-pipeline=PIPELINES["transform"]` | the K loop becomes `scf.for … step %c32 iter_args(…4…) -> (4 × !air.async.token)` — step doubled from 16, exactly `04-test-plan.md` §3.2's expectation |
| 11 | `diff tests/golden/w1.base.npu1.air.mlir <(python vendor/probes/q/q7_a.py)` | see **the probe oracle** below |

## The probe oracle (D4)

**Byte-identical.** `cmp` of the golden against the probe's module text, with `print()`'s extra
trailing newline removed, reports no difference; the raw `diff` shows exactly one hunk,
`126a127 > ` — the probe file has 127 lines because `print(launch.build(...))` appends a newline
to a `str(module)` that already ends in one. Our emitter writes the 126-line text unchanged.

That is the strongest statement available for this phase: the plan literal, walked by M5's
translation table, reproduces a hand-written upstream-API program op for op, SSA name for SSA
name, `affine_map` for `affine_map` — including the strip-mine loop and
`#map1 = affine_map<()[s0, s1] -> (s0 * 2 + s1)>` that `air.api` inserts on npu1 because the
2×2 logical grid runs on a 1×2 physical herd (finding N-7).

## What `air.api` actually does, against what the LLD assumed

| # | LLD said | `air.api` does | Consequence |
|---|---|---|---|
| **1** | `ChannelSite.is_async` — "emitted in asynchronous form" (`06-interfaces.md` §5.2); W2 §6.3 sets it on the halo puts | **There is no asynchronous form to select.** `put`/`get` take `obj, indices, dependency` (+ `dest` on `put`) and nothing else (`_channel.py:511`, `:524`), and **every** call returns a `Token` on all three return paths (`_channel.py:471`, `:485`, `:507`). A `Token` "carries no SSA value … AIR's own asynchrony is built by the `air-dependency` pass from the program order this tracer emits" (`_value.py:26-33`) | M5 emits the same call whether `is_async` is true or false, and the field is read by nothing. **For the architect**: either §5.2 records `is_async` as documentation of intent (which is what it now is), or it is dropped at the next contract version. It cannot be honoured |
| **2** | `dependency=<tok or None>` (§3.2 rows 9-10) | `_check_dependency` (`ops.py:82-91`) accepts a `Token` or a list/tuple of them and **validates only** — the value is never used by `_emit`, because a v1 `Token` has no SSA value | Implemented as specified (a list of the tokens the named sites returned, `None` for `depends_on == ()`), and it is inert until upstream gives `Token` a value. W1 has `depends_on = ()` everywhere, so **untested** |
| **3** | FR-E9 / §5: "`build()` runs `module.operation.verify()`; failure surfaces as `EmissionError`" | The failure surfaces as a `RuntimeError` whose text begins `air.api emitted invalid IR -- this is a bug in the DSL, not in the kernel:` (`_compile.py:174-180`). That prefix is the **only** signal separating it from `_check_interface`'s `RuntimeError`, which must stay `EMIT-AIR-API` | M5 matches on that marker. `test_E9_verify_surfaced` raises the marker message in `build()`'s place and asserts the classification: **no plan we can construct reaches the real path**, because M5 checks its own preconditions first and every W1-shaped module verifies |
| **4** | §3.6 line 14: "fold `air.ops.maximum` … left to right"; the paragraph below it and §6.4's `ROW` show `maximum(a, maximum(b, maximum(c, d)))` | — | The two readings disagree. M5 implements the **shape the LLD draws** (right-nested, `maximum(op[0], maximum(op[1], …))`), because that is what the W3 golden will be compared against. Untested until W3 |
| **5** | §3.2 row 12 / §3.6: `MaxMin` and `Select` take the emitted operands | `ops.maximum`/`minimum` reject a bare `BufferSlice`: `_elementwise`'s guard admits `(Buffer, BufferExpr, int, float)` only (`ops.py:384-391`), even though `BufferExpr.coerce` handles `BufferSlice` two lines of the file later (`_value.py:1046`). `$PROBE/w3b.py` never hits it because every slice there goes through an operator first (`p[j] - 1`) | **A W3 finding, logged now**: `MaxMin(operands=(Load(p, j), …))` — an operand that is a bare load — will raise `TypeError` out of `air.api` and become `EMIT-AIR-API`. The fix when W3 lands is one coercion in M5 or one widened isinstance upstream; do not discover it on D4 |

## Spec readings taken

| # | Reading | Why |
|---|---|---|
| 1 | **Internal-consistency failures are raised as `EMIT-AIR-API`** with `details["internal_consistency"] = true`. §5 says what they mean but not which code, and `06-interfaces.md` §6.3 gives the emission stage exactly two codes, both of which `spatial/model.py` enforces (`I67`) | A third code (`EMIT-PLAN-INVARIANT`) would be the honest spelling. The contract is frozen; **for the architect to rule**. The `fix` line says "this is a defect in the compiler, not in your program … report it", so nothing tells the user to edit a clause |
| 2 | Every emission `Diagnostic` carries `clause="build()"` | `tests/helpers/diagnostics.py` asserts `clause is not None` for every non-grammar code, and `build()` is the surface call that funnels through `m5.emit` (§7.1). M6 already uses the same string |
| 3 | The `.summary.txt` golden is written for **npu1 only**, under §8's target-less name | §8's path is `<workload>.<variant>.summary.txt` while `04-test-plan.md` §3.1 says the summary is "stored per target too". W1's summary line 7 carries the *physical* herd and repeats, which differ between targets, so one file cannot hold both. `w1.base.npu2.plan.json` embeds the npu2 rendering of the same lines, so nothing is unwitnessed. **A naming conflict for the architect**: either §8 gains the target, or §3.1 drops the claim |
| 4 | The plan JSON goldens are per target (`w1.base.<target>.plan.json`) | Same reason; `MappingPlan.mapping` differs by `physical_herd`, `repeats` and `schedule.target` |
| 5 | `test_sequential_emits_scf_for` counts **emissions**, not `LoopPlan` nodes | A sequential loop's body is traced once whatever its trip count, an unrolled loop's body is traced once **per trip**, and `air.api` adds one `scf.for` of its own when `shape=` strip-mines the grid. W1/npu1: 4 (segment) + 6 (herd) + 1 (strip-mine) = **11**; npu2: 10, with no strip-mine. M5 §7's "count == count of `LoopPlan(kind='sequential')` in the plan" is true of neither |
| 6 | `air.tensor(..., name=<BufferPlan.name>)` | Left to itself `air.api` recovers the name by `inspect.stack()` on the caller's source line (`_trace.py:441-455`). It is cosmetic — the name never reaches the IR — but reading our own source at emission time is exactly what determinism rule 5 forbids in spirit, and the plan already carries the name |
| 7 | A herd body is a fresh index scope | `air.herd` is `IsolatedFromAbove`. A segment-scope loop axis referenced inside `herd_body` is the defect `_compile.py:162-172` was written for; M5 refuses it as an internal-consistency error naming the axis, rather than emitting IR that only `verify()` would catch |

## Not done, and why

| # | Item | Reason |
|---|---|---|
| 1 | Rows 5c (`channel_type=`) and 11 (`BranchNode`/`Guard`) are **implemented and untested** | W1 has no cascade channel and no guard. Both are single expressions — one keyword on `air.channel`, one `with air.ops.branch(...)` — and both wait on the W1-flip and W3 plan literals |
| 2 | `Select`, `MaxMin`, `Neg`, `BinOp` for `-`, `*`, `/` | W1's compute tree is `+` and `*` over `Load`/`Const` only. The rest are implemented from §3.6 and wait on W2/W3 |
| 3 | `depends_on` / `dependency=` | W1's sites all have `depends_on = ()` (finding 2 above) |
| 4 | M5 §7's `test_E1_herd_arity`, `test_E8_cascade_text`, `test_M5_uses_branch`, `test_E_peel_is_plan_driven`, `test_E_select_emitted`, `test_E_branch_node`, `test_E_tensor_order`, `test_pipeline_is_hint`, and the six other goldens | Every one needs a W1-flip, W2 or W3 `MappingPlan`, none of which exists. They are not skipped, they are **absent**; this row is the list |
| 5 | `lock_init_histogram` (`air-to-aie`) | M5 §3.8 lists it; `04-test-plan.md` §3.2 marks `test_I_lock_inits` optional at D6 |
| 6 | Person C's `has_device`/`run`/`diff`/`trace` | Not ours |

## Open / blockers

| # | Item | Detail |
|---|---|---|
| **B-P12** | `is_async` cannot be honoured | Finding 1 above. M4 may keep setting it; M5 will keep ignoring it. Needs a ruling before the W2 golden is frozen, because `06-interfaces.md` §5.2's wording implies emitted output that does not exist |
| **B-P13** | No error code for a broken plan precondition | Reading 1 above. `EMIT-AIR-API` with `internal_consistency: true` is the least-bad spelling under the frozen §6.3 |
| **B-P14** | `.summary.txt` naming | Reading 3 above |
| **B-P15** | M5 §7's `test_sequential_emits_scf_for` row is unimplementable as written | Reading 5 above; the test exists and is stricter, but the LLD row should be corrected |
| **B-P16** | `ops.maximum` rejects a bare `BufferSlice` | Finding 5 above. It will bite on W3's first `MaxMin(Load, …)` operand |
| **B-O8** | still open | Whether `str(module)` is stable across a `--upgrade` within the pin is Person C's Q-3; byte-for-byte goldens plus `check_pin` are the standing answer |
| **v4 signatures** | unchanged | A and C have still not signed `CONTRACT_VERSION = 4` |

**Not verified in this phase**: that M4 will produce this plan (M4 does not exist — the W1
literal is B's own hand transcription of `03-lld-M4-mapping.md` §6.1, and a D6 test is meant to
assert `m4.plan(...) == w1_plan.plan()`); that the emitted module computes GEMM (no device, and
`04-test-plan.md` §3.4 is explicit that the oracle link is structural, not executed); that
anything holds for W1-flip, W2 or W3.

---

# Phase P2 — M4, the W1 path from `LegalMapping` to `MappingPlan`

*2026-09-13. Branch `role-b`. `spatial/m4_mapping.py`'s ten passes, general machinery with one
protocol builder; `m4.plan(w1_legal.legal(t)) == w1_plan.plan(t)` for both targets, and M5 turns
that plan into the existing W1 goldens byte for byte. **Nothing here was pushed.***

## Landed

| # | What | Where |
|---|---|---|
| 1 | `m4.plan` — the ten passes of M4 §3.1 in their fixed order, each a module-level pure function: `preconditions`, `resolve_herd`, `tensor_plan`, `classify`, `buffer_plan`, `channel_plan` (via `l3_region`), `loop_plan`, `protocol`, `summary` (+ `residency`), `self_check`, then return | `spatial/m4_mapping.py`, 1 231 lines |
| 2 | General machinery, not W1 shapes: `tile_shape` from the `AccessMap` and the tile factors (§3.3 note 1); `classify` in `UCoord` over `pi_u`/`ker_pi_u` with `root()` via `Axis.parent`, including the CASCADE row and the declared override; `multicast_geometry`, `l3_region`, `loop_kind`, `tensor_plan` (reads before writes, shapes through `bindings`), `residency`/`temporal_axes`/`moved_axes` per §3.9 lines 1-8, `reduction_split` | same |
| 3 | Exact integer linear algebra in **pure Python** over `Fraction` — `_rref`, `_rank`, `_in_span`, `kernel_basis` (§3.2's `KERNEL_BASIS`). `numpy` is not imported: nothing here needs a float | same |
| 4 | One protocol builder: §6.1's fill / compute / drain. `exchange`, `forward` and a non-empty `r_space` raise `NotImplementedError` naming the phase; §3.1's `PLAN` wrapper re-raises every non-`SpatialError` as a `MappingError` with `details["internal_exception"]` (§5) | same |
| 5 | `reside(x="L2")` → a **real** `MappingError(PROTOCOL-UNSUPPORTED)` whose clause is `reside(U="L2")` (§3.3 note 5, FR-S12) | same |
| 6 | §4's preconditions asserted: `physical_herd` divides `grid` with those `repeats` and is within the target cap (`_trace.py:88-91`), `len(pi) == len(grid)` of rank 1-2, `l1_bytes ≤ 65536`, `rank(r_space) ≤ 1` with an A/C operator, declared halo ≥ derived footprint. Each names the two values and the section | same |
| 7 | Two internal-consistency checks that make a spec sentence true **by check**: every `ChannelSite.order` is its index in the body holding it (§5.2 at v4), and `MappingPlan.buffers` is the order the herd body allocates them (§5.6) | same |
| 8 | `m4.self_check` — the P3 stub: a pass-through returning `None`, at its final import path, called by `plan()` on its own result. **Replaced in P3** by §3.7-§3.8 (P1' balance, P2b acyclicity, bundle indices, ping-pong shape, the DMA-channel budget) | `spatial/m4_selfcheck.py` |
| 9 | 25 unit cases — the M4 §7 rows this phase can honour, plus the acceptance and the import lint | `tests/unit/test_m4_mapping.py` |
| 10 | `test_W1_legal_to_text[npu1\|npu2]` — gate **G2**'s B half: `LegalMapping` → `m4.plan` → `m5.emit` → the existing `w1.base.<target>.air.mlir` golden | `tests/integration/test_golden.py` |
| 11 | **B-P14 closed** (architect ruling): the summary golden is per target. `w1.base.summary.txt` → `w1.base.npu1.summary.txt` (`git mv`), `w1.base.npu2.summary.txt` generated, `06-interfaces.md` §8's row and `00-README.md` §4's change log carry the erratum | `tests/golden/`, `design/` |

## Verified (command → result)

| # | Command | Result |
|---|---|---|
| 1 | `.venv/bin/python -m pytest` | **448 passed, 4 deselected** in 1.55 s (was 420 passed, 4 deselected); 1.67 s wall. Budget NFR-3 is 180 s |
| 2 | `PYTHONHASHSEED=1 … -m pytest` / `PYTHONHASHSEED=2 … -m pytest` | 448 passed, 4 deselected both times (1.55 s / 1.58 s); `diff` of the two 448-line `-vv` id/outcome lists is empty |
| 3 | `.venv/bin/python -m pytest -m slow` | **4 passed**, 448 deselected, 3.24 s — unchanged by this phase |
| 4 | `.venv/bin/python -c "import spatial.m4_mapping, spatial.m4_selfcheck, sys; print('air' in sys.modules, 'spatial.m5_emit' in sys.modules)"` | `False False` (FR-S20, I-1) |
| 5 | `m4.plan(w1_legal.legal(t)) == w1_plan.plan(t)` for `t ∈ {npu1, npu2}` | **True**, and `to_json` of the two is equal — field for field, on the first run, with **no edit to the literal** |
| 6 | `m5.emit(m4.plan(...), t).mlir` vs `tests/golden/w1.base.<t>.air.mlir` | byte for byte for both targets |
| 7 | `git status --porcelain tests/golden` after `pytest --update-goldens` | only the intended rename and the one new npu2 summary; `w1.base.*.air.mlir`, `*.plan.json` and `*.ir_facts.json` unchanged |

## The RESIDENCY algorithm's own output for W1 (§3.9 lines 1-8, verbatim)

```
A: multicast along py, re-fetched per k0
B: multicast along px, re-fetched per k0
C: stationary (spatial), resident for the whole run
```

`T = (k0,)` — `i0` and `j0` are placed, `i1`/`j1`/`k1` run inside the tile — and
`M_A · e_k ≠ 0`, `M_B · e_k ≠ 0`, `M_C · e_k = 0`. The flip's half is checked at the function
level in the same test: `residency(flip, "B") == "resident for the whole run"` and
`residency(flip, "A") == "re-fetched per i0"`, which is FR-K2's claim.

## Reconciliation with the W1 literal

**None was needed.** The derived plan equalled `tests/fixtures/plans/w1_plan.py` field for field
at the first run, so no literal field was changed, no golden was regenerated beyond the B-P14
rename and the new per-target file, and `w1.base.<target>.plan.json` is byte-identical. Every
P0c reading the literal recorded — `LoopPlan.depth` counting enclosing loops, `ChannelSite.order`
as the index in its body, an `Expr` naming the `LoopPlan.axis` that binds it, `buffers` in
allocation order — is now a rule in the code, and two of them are *asserted* rather than
believed (landed row 7).

## Spec readings taken, where the documents disagree

| # | Reading | Why, and what it is worth |
|---|---|---|
| 1 | **A `stationary` clause marks the row `declared` unless the operand is the reduction target.** | §3.2's pseudocode marks only `stream`/`exchange` rows declared (lines 19-21), which makes the flip's `B` *derived* — but §7's `test_M1_trichotomy_flip` and §6.2's summary both require `("B", STATIONARY, None, **True**)` / `B: stationary (declared)`. Meanwhile §6.1, FR-M11's acceptance and the committed golden require W1's `C: stationary (**derived**)` although `stationary("C")` is equally declared. No rule in the documents produces both. The carve-out chosen is the only one with a reason behind it: the accumulator's PE-locality follows from `r_space == {}` (with `R_time = R` no partial sum ever leaves the PE), so the clause is redundant there, while on the flip the clause is the only thing that names `B`. **Affects the `declared` flag only, never the delivery.** Recorded as **B-P17** for the architect |
| 2 | **An `exchange` clause does not replace a delivery row.** | §3.2 line 20 lumps `exchanges` with `streams`, but `Delivery` has no halo member and §6.3 prints W2's delivery as two facts — `U: STATIONARY strip (derived); halo exchange along px, width 1 (declared)`. `classify` therefore ignores `exchanges`, and the halo is a *protocol* selected in `protocol()`. This is also load-bearing for reachability: if `classify` raised for W2, FR-S12's `test_M4_reside_l2` — a W2 schedule — could never reach §3.3 line 11a |
| 3 | **`DEPTH` drops its delivery half and keeps its residency half.** | §3.3 line 28 reads "`MULTICAST` or `FORWARD` **and** re-fetched per trip", which puts the flip's `a` at depth 0 — but §6.2's own table gives `a` `loop_depth 1` and `ping_pong_candidate = True`, and it is `STATIONARY`. Implemented as: a **read** operand is depth 1 iff a temporal outer tile axis moves its tile, and a written accumulator is always 0 (it is allocated once and drained). Reproduces §6.1's and §6.2's buffer tables both |
| 4 | **Bundle and drain loop names are `f"p{root(place[d])}_bundle"` and `f"{root(place[d])}_drain"`.** | `06-interfaces.md` §5.5's footnote says `<operand>_bundle` / `<operand>_drain`, which matches **neither** worked example: §6.1 prints `pi_bundle`, `pj_bundle`, `i_drain`, `j_drain` and §6.2 writes the flip's bundle index `pk`. The implemented rule reproduces both. Recorded as **B-P18** |
| 5 | **Compute-nest loop names are positional over `Statement.axes`** — `m`, `n`, `t`, and the zeroing nest's `m0`, `n0`. | §6.1 prints them for W1 and §6.2 prints the same three for the flip **including its untiled `j`** (a `0..64` loop named `n`), so the name cannot be the axis's own name. §6.3 and §6.4 print W2's and W3's nests with the kernel axis names (`i`, `j`) instead, which this rule does not reproduce — a P4/P5 decision, logged as **B-P18** |
| 6 | **`KernelModel` records no expression tree, so only the accumulate form can be rebuilt.** | §2.4's `Statement` carries `kind`, `target`, `reads`, `op` and nothing else. M4 can therefore build `acc[...] = acc[...] + <product of the reads>` — exactly what §6.1 shows — and **cannot** build W2's `0.2 * (five-term sum)` (§7's `test_M4_stencil_is_five_point`) or W3's `max`/`Select` recurrence (§6.4) from the model as frozen. This is a **contract gap**, not an implementation gap: the fix is an `ExprNode` on `Statement` (a v5 change) or M1 handing M4 the tree some other way, and it must be settled before P4. Recorded as **B-P19** |
| 7 | **Every mapping diagnostic this phase raises is `PROTOCOL-UNSUPPORTED`.** | §5 names no code for a §4 precondition failure, and none for the wrapper's re-raise; §6.3's mapping stage has exactly five codes and M0 enforces the set (I67). The same least-bad spelling as M5's P1 reading 1 (B-P13) is used: `PROTOCOL-UNSUPPORTED` with `details["internal_consistency"]` or `details["internal_exception"]`, `clause="plan()"`, and a `fix` that asks for a bug report rather than blaming a clause the user wrote. Recorded as **B-P20** |
| 8 | **The rendered reduction split unions `r_time` and `r_space` and re-splits over the post-tiling axes by placedness.** | §3.9 line 10's paragraph requires exactly that — printing `r_time` bare gives the flip `R_time = {}`, which is false. W1 renders `span{e_k0, e_k1}` / `{}`, which is what §6.1 and the golden carry |
| 9 | **A read-only operand is filled, a written one is drained, and the accumulator is never filled.** | §3.5 lines 7-10 say "if written: drain; if read: fill". W1's `C` is read *and* written in the source (`+=`), yet §6.1 has no `C2L1`: the accumulator is initialised by the zeroing `LoopPlan`. An operand that is genuinely both (W2's `U`) needs both and is not built |
| 10 | **`_PHYSICAL_CAP` duplicates `_trace.py:88-91` inside M4.** | §4's precondition is "within the target cap", and M4 may not import `air` (I-1) or reach into the test fixtures. Four lines, cited; skipped when `target == "auto"`, which M6 resolves |

## Not done, and why

| # | Item | Reason |
|---|---|---|
| 1 | §3.7-§3.8 — P1' balance, P2b acyclicity, the two structural checks, the DMA-channel budget | **P3.** `m4.self_check` is the pass-through stub above; it accepts every plan, including the six corrupted-plan negatives of §7, which are therefore **not** written yet |
| 2 | §3.6.1 halo (W2), §3.6.2 wavefront (W3), §3.6.3 cascade (the flip) | P4/P5/P6. Each raises, and `test_M4_unbuilt_protocols_fail_legibly` asserts the code, the clause and the phase for all three inputs |
| 3 | The M4 §7 rows those builders carry — `test_M4_halo_protocol`, `test_M4_balanced`, `test_M4_drains_every_plane`, `test_M4_stencil_is_five_point`, `test_M4_halo_indices`, `test_M4_odd_T_peel`, `test_M4_even_T_no_peel`, `test_M5_*`, `test_M6_cascade_*`, `test_M9_*`, `test_M10_*`, `test_M11_bundle_index_iv`, `test_M11_pingpong_shape`, `test_P3_*`, `test_M4_tensor_order_rejected` | They are **absent**, not skipped; this row is the list |
| 4 | The full-plan halves of `test_M1_trichotomy_flip` and `test_M3_stream_override` | Both are asserted at `classify()` level, which is the half that exists |

## Open / blockers

| # | Item | Detail |
|---|---|---|
| **B-P14** | **CLOSED** | The summary golden is `<workload>.<variant>.<target>.summary.txt`. Rename done, npu2 generated, §8 and the change log carry the erratum; no `CONTRACT_VERSION` bump |
| **B-P17** | `declared` for a `stationary` clause | Reading 1. Needs a one-line ruling in §3.2 before the flip's summary golden is frozen |
| **B-P18** | Two naming rules the documents spell three ways | Readings 4 and 5. §5.5's `<operand>_bundle` footnote, and the compute-nest names in §6.3/§6.4 |
| **B-P19** | `Statement` carries no expression tree | Reading 6. **Blocks P4/P5**: W2's stencil and W3's recurrence cannot be rebuilt from the frozen `KernelModel`. A contract question, and the one item here that is not B's to settle alone |
| **B-P20** | No mapping code for an internal-consistency failure | Reading 7, the twin of B-P13 |
| **B-P5** | **CLOSED** | `assert_golden`'s compare path has been exercised since P1 and is exercised again here by both summary goldens and the two `air.mlir` goldens |
| **B-P10**, **B-P12**, **B-P13**, **B-P15**, **B-P16**, **B-O8** | unchanged | — |
| **v4 signatures** | unchanged | A and C have still not signed `CONTRACT_VERSION = 4` |

**Not verified in this phase**: that any plan but W1's is right — the other three raise by
construction; that `self_check` catches anything at all (it is a pass-through); that the emitted
module computes GEMM (no device, `04-test-plan.md` §3.4 keeps the oracle link structural); and
every `NotImplementedError` path beyond the three inputs the tests drive through it.

---

# Phase P2b — CONTRACT_VERSION 5 applied (architect ruling)

*Person B, 2026-09-13. Branch `role-b`. The architect's three rulings of 2026-09-13 on B-P19
(`Statement.expr`), B-P17 (`declared`) and B-P18 (loop-axis naming), across the contract, M0, M4,
the literals, the goldens and the documents that quote them. **Not pushed** — the architect
verifies and pushes.*

## Landed

| # | What | Where |
|---|---|---|
| 1 | `CONTRACT_VERSION = 5`, with a *Version 5* paragraph naming FR-M8 + FR-E2 as the forcing requirements | `design/06-interfaces.md` header, `spatial/model.py` |
| 2 | **`Statement.expr: ExprNode`** — the complete right-hand side stored into `target`, after scalar forward substitution, over **kernel-level** operands; an `accumulate` carries the **desugared** tree; association follows Python's parser | `06-interfaces.md` §2.4, `spatial/model.py` |
| 3 | **`I78`**: every `Load` reachable in any statement's `expr` names a `Param` of that kernel — the one reference check M0 makes, because a `KernelModel` holds both sides. The module docstring's "M0 checks no reference" paragraph now states that exception | `spatial/model.py` (`_loads` + `KernelModel._validate`), `06-interfaces.md` §2.7 |
| 4 | **`declared` := a clause names this operand's delivery** — `stationary(a)`, `stream(a, …)`, `forward(a, …)`, `exchange(a, …)` — a fact about the schedule, not about whether the derivation agreed. W1's `C` becomes `("C", STATIONARY, None, True)` / `C: stationary (declared)`; `A`/`B` stay derived; the flip's `B` stays declared and its `A` derived | `06-interfaces.md` §5.6, `m4.declared_operands` + `m4.classify` |
| 5 | **The loop-axis naming rule**: `p<root>_bundle`, `<axis>_drain`, and a compute or zeroing nest named by the **post-tiling axis it realises** — W1 `i1`, `j1`, `k1` (zeroing `i1`, `j1`), never positional `m`/`n`/`t`/`m0`/`n0`. A `LoopPlan.axis` may recur in two bodies; M5 rebinds by name in order, which is what `_bound` already did | `06-interfaces.md` §5.5, `m4.compute_names`, `m4.bundle_name`, `m4.drain_name` |
| 6 | **M4 builds every compute `StoreNode` from `Statement.expr`**, generically: `l1_subscripts` rewrites each kernel-level `Load` into its L1 buffer, `_rewrite_loads` walks the tree, and the hard-coded `acc = acc + <product of the reads>` reconstruction is gone. The zeroing nest stays (no kernel statement corresponds to it) and now shares the compute store's subscripts | `spatial/m4_mapping.py` |
| 7 | `expr` on all three kernel literals, exactly per the ruling: W1's desugared accumulate; W2's `0.2 × (left-nested five-term sum in source order)`; W3's `max` with the `Select` and the resolved `MATCH`/`MISMATCH`/`GAP` tokens. `w1flip_legal` inherits W1's kernel unchanged | `tests/fixtures/mappings/w{1,2,3}_legal.py` |
| 8 | The W1 plan literal's loop names, store subscripts and `declared` flag | `tests/fixtures/plans/w1_plan.py` |
| 9 | Tests: `I78` negative and the minimal `Statement`'s `expr` (M0); `test_M4_loop_axis_names`, `test_M4_compute_node_is_the_kernel_expression`, `test_M4_l1_subscripts_rule` (M4); each literal's `expr` asserted structurally against the LLD text plus `test_statement_expr_loads_agree_with_the_access_maps` per workload | `tests/unit/test_m0_model.py`, `test_m4_mapping.py`, `test_fixture_literals.py` |
| 10 | Documents: `06-interfaces.md` §2.4/§2.7/§5.5/§5.6 + header; `00-README.md` §3, §4 (signature row and change-log row 5); `01-requirements.md` FR-M1, FR-M11; `03-lld-M4-mapping.md` §3.2, §3.9, §6.1, §6.2, §7; `05-work-breakdown.md` §5 step 2; `03-lld-M8-kernels-demo.md` §5 and §6.1 | `design/` |

Counts: **78** invariants (`I01`-`I78`) / **117** negative cases / 33 classes with a minimal
instance.

## The L1-subscript rewrite rule (landed row 6)

`l1_subscripts(subscripts, elements, origin, bindings)` is a pure function and has its own unit
test. Array dim `d` of a kernel-level access reads L3 at `subscripts[d]`, an `Expr` over kernel
axis names, shape-parameter names and constants. `elements[x]` is the **L3 element coordinate the
compute nest realises** for kernel axis `x` — the tile origin plus the inner loop variable where
`x` is tiled (`tx·TM + i1`, `k0 + k1`), the loop variable itself where it is not, since an
untiled axis's loop runs over the kernel's own range. `origin[d]` is where this PE's **staged
slab** starts in L3 along that dim, taken from `L3_REGION` over the same herd-scope variables, so
it carries whatever the staging region carries — a ghost row, a whole-width band. The buffer-local
index is then `substitute(subscripts[d], elements) − origin[d]`: the placed and outer-tile
contributions cancel exactly. `compute_frame` builds `elements` and the per-axis slab origins;
`_in_l1` turns one `Load` into `(buffer name, rewritten subscripts)`; a buffer that stages *fewer*
dims than the access indexes (a plane of a rank-3 tensor) raises `NotImplementedError` naming the
phase.

**W1, measured** (the derived plan equals the literal): `A[i,k] → a[i1,k1]`, `B[k,j] → b[k1,j1]`,
`C[i,j] → acc[i1,j1]`, so the store is `acc[i1,j1] = acc[i1,j1] + a[i1,k1]*b[k1,j1]` and the
zeroing store is `acc[i1,j1] = 0.0`.

**W2 and W3, derived by hand from §6.3/§6.4 — to be exercised in P4/P5, not asserted here:**

* **W2** (`cur`/`next` are `(HS+2, W)`, the row band ghost-padded above, all `W` columns staged
  from column 0): `U[t,i,j] → src[i1+1, j]`, `U[t,i-1,j] → src[i1, j]`, `U[t,i+1,j] →
  src[i1+2, j]`, `U[t,i,j-1] → src[i1+1, j-1]`, `U[t,i,j+1] → src[i1+1, j+1]`, and
  the write `U[t+1,i,j] → dst[i1+1, j]` — the nest being `i1`, `j` per §5.5's rule.
* **W3** (`prev`/`cur` are `(CW+1,)`, the PE's column band with one left-edge cell; `qb` the whole
  `q`; `rb` the PE's slice of `r`): `S[i-1,j-1] → p[j1]`, `S[i-1,j] → p[j1+1]`,
  `S[i,j-1] → c[j1]`, `q[i-1] → qb[i-1]`, `r[j-1] → rb[j1]`, and the write `S[i,j] → c[j1+1]`,
  the nest being `j1` inside the row loop `i`.
* **The one caveat both turn on**: `_origins` gives a placed tiled axis the origin `pe·factor`
  and ignores the axis's `lo`. W1's axes all have `lo = 0`, so this cut is unaffected, but W2's
  `i` starts at 1 and W3's `i`/`j` start at 1, and the results above hold only if the placed
  tile origin is `lo + pe·factor`. **P4 must add the `lo` term** (one `_add` in `_origins`), and
  the fixture regions of §6.3/§6.4 should be checked against it when they are written.

## Verified (command → result)

| # | Command | Result |
|---|---|---|
| 1 | `.venv/bin/python -m pytest` | **456 passed, 4 deselected** in 1.59 s (was 448 passed, 4 deselected) |
| 2 | `PYTHONHASHSEED=1` / `=2`, `-vv` | 456 passed both times; `diff` of the two 456-line id/outcome lists is **empty** |
| 3 | `.venv/bin/python -m pytest -m slow` | **4 passed**, 456 deselected, 3.22 s |
| 4 | `.venv/bin/python -c "import spatial; print(spatial.CONTRACT_VERSION)"` | `5` |
| 5 | `m4.plan(w1_legal.legal(t)) == w1_plan.plan(t)`, `to_json` equal, both targets | **True** — after the literal's loop names, store subscripts and `declared` flag were updated, and with no other change |
| 6 | `pytest --update-goldens` then `git diff --stat tests/golden` | `w1.base.npu{1,2}.plan.json` 120 changed lines each (the `expr` subtree, the loop names, `declared: true`) and `w1.base.npu{1,2}.summary.txt` 2 lines each (`C: stationary (declared)`). **`*.air.mlir` and `*.ir_facts.json` are untouched** — R3 and R2 do not reach the IR, which is the check the brief asked for |
| 7 | `import spatial.m4_mapping` in a fresh process | `air imported: False`, `m5 imported: False` (FR-S20, I-1) |

## Blockers closed

| # | Resolution |
|---|---|
| **B-P19** | **Closed.** `Statement.expr` is the M1→M4 route M1 §6.3 already described in prose; M4 rebuilds no arithmetic, so W2's five-point stencil and W3's `max`/`Select` reach M5 unchanged once their protocol builders land. The only nodes M4 still synthesises are the accumulator zeroing and (at P6) the cascade's `acc += recv`, and neither corresponds to a kernel statement. |
| **B-P17** | **Closed.** `declared` is a fact about the schedule: named by a clause or not. W1 now reads two derived rows and one declared, and the flip's `B` keeps the `(declared)` its §6.2 summary and §7 row require — one rule, both worked examples, no accumulator carve-out. |
| **B-P18** | **Closed.** `06-interfaces.md` §5.5 carries the naming rule; `m4.compute_names` returns the post-tiling axis; the `<operand>_bundle`/`<operand>_drain` footnote that matched neither worked example is gone. |

## Open / blockers

| # | Item | Detail |
|---|---|---|
| **v5 signatures** | A, B, C have not signed | **A is affected**: M1 must emit `Statement.expr` (the tree of M1 §6.1-§6.3, already printed there for W3), and no `KernelModel` constructs without it. **C is affected**: the demo screen of `03-lld-M8-kernels-demo.md` §6.1 and §5 step 2 now say `C: stationary (declared)`; any hand-written kernel fixture needs an `expr`. |
| **v4 signatures** | unchanged | A and C have still not signed `CONTRACT_VERSION = 4`. |
| **B-P21** | `02-hld.md` §7's W1 delivery-summary row still reads `C: stationary (derived)` | The brief's edit list is explicit — *"No other document"* — and `02-hld.md` is not on it, so the edit was **not** made. `design/02-hld.md:342`: `\| delivery summary \| `C: stationary (derived)`, `A: multicast along py (derived)`, `B: multicast along px (derived)` \|`. One word, architect's call. |
| **B-P22** | W2's and W3's delivery *sentences* predate the one-bit `declared` | Under R2 an `exchange("U", …)` clause names `U`'s delivery, so W2's row is `("U", STATIONARY, None, True)` — while `03-lld-M4-mapping.md` §6.3 prints `U: STATIONARY strip (derived); halo exchange along px, width 1 (declared)`, i.e. two facts in one sentence. W3 is the same shape (`S: column band stationary (derived); forward … (declared)`), and there `forward` also replaces the row, so the summary line becomes `S: forward along px (declared)`. Neither section was in the edit list and neither plan is built yet; **P4/P5 must reconcile the printed sentence with the row** or the architect must rule that the summary keeps a second, protocol sentence. |
| **B-P10**, **B-P12**, **B-P13**, **B-P15**, **B-P16**, **B-P20**, **B-O8** | unchanged | — |

**Not verified in this phase**: that M1 will produce these `expr` trees (M1 does not exist — the
three are B's transcription of M1 §6.1-§6.3, and `test_statement_expr_loads_agree_with_the_access_maps`
only proves each tree agrees with the `AccessMap`s beside it); that the W2/W3 rewrites above are
right (no plan is built for either — they are hand-derived and are recorded so P4/P5 can check
them, including the `lo` caveat); anything on a device.

---

# Phase P3 — the M4 self-check, the corrupted-plan corpus, the plan interpreter

*Person B, 2026-09-13. Branch `role-b`. P1' balance, P2b acyclicity, the structural invariants
and the P3 DMA-channel budget land as real checks; `spatial/m4_selfcheck.py` stops being a
pass-through. **Not pushed** — the architect verifies and pushes.*

## Landed

| # | What | Where |
|---|---|---|
| 1 | **P1' balance** (§3.7.1): concrete coordinate enumeration, `trips` through sequential loops, unrolled loops **enumerated**, guards and `BranchNode` arms evaluated per coordinate, fan-out by D-2. Failure carries the full per-key table (every index of every channel, `0 == 0` rows included), the channel, the index, the fan-out index, both site lists with their coordinates, and `details["rule"]` — one of `fan-out`, `per-branch`, `per-iteration`, `total` | `spatial/m4_selfcheck.py` |
| 2 | **P2b acyclicity** (§3.7.2): nodes are `(site id, coord, concrete index)`; channel edges put→get after fan-out expansion; program-order edges by rules 1-5 exactly, with the herd contracted to a single `HERD` node so **no** segment-site→herd-site edge exists; iterative Tarjan; `details["cycle"]` is the ordered edge list, each end naming site, channel, index and coordinate | same |
| 3 | **Structural** (§3.7.3, `06-interfaces.md` §5.6 invariants 3-8): bundle-index-is-IV (`BUNDLE-INDEX-IS-IV`, also reached through the balance walk so an unevaluable index never surfaces as an internal error), ping-pong shape, the L1 budget **and** the staged-subset equality with `LegalMapping.l1_bytes`, tensor order naming `_check_interface`, the names and the `HerdPlan` marker | same |
| 4 | **P3 DMA budget** (§3.8): per coordinate, live sites; `MAX_OVER_EXCLUSIVE_BRANCHES` falls out of enumeration; `DMA-CHANNELS` error naming the PE, the direction, the channel list, the budget, `circuit-switched` and `2 S2MM`/`2 MM2S`, with `clause = grid(...)`; a module-level `warnings(plan) -> tuple[str, ...]` for the packet-capable case, which raises nothing | same |
| 5 | **`may_packet(channel, plan)`** — the `CIRCUIT` predicate §3.8 leaves undefined, derived from evidence and then found to be the upstream pass's own arithmetic (below) | same |
| 6 | **The corrupted-plan corpus**: six `dataclasses.replace` corruptions of `w1_plan.plan("npu1")` — `dropped_get`, `extra_put_in_loop`, `broadcast_underconsumed`, `iv_bundle_index`, `pingpong_hoisted`, `tensor_order` — plus the synthetic 2-PE `halo.py` base and its `halo_guard_dropped`, `halo_reversed`, `dma_three_inbound`, `dma_packet_warn`, plus the legal `branch_gets.py` (a `BranchNode` plan, for the two rows that need one) | `tests/fixtures/corrupt/` |
| 7 | **The plan interpreter**: one coroutine per concurrent region, one FIFO per `(channel, concrete index)`, fan-out on every put, a get that yields while empty, round-robin with a deadlock report (a dynamic P2 check), `air.alloc` re-allocating per trip, regions as row-major slabs | `tests/helpers/plan_interp.py` (201 lines) |
| 8 | **The three semantic checks** of `04-test-plan.md` §3.4 for W1 on both targets: `test_sem_access_regions` (every region's index set equals the `AccessMap` image over that PE's subdomain), `test_sem_compute_nodes` (`C == A @ B` **exactly**, plus a mutation test proving it is not vacuous), `test_sem_coverage` (the drained regions partition the write domain) | `tests/integration/test_semantics.py` |
| 9 | 20 unit tests for the checks, every negative through `assert_diagnostic` with the numbers the LLD names | `tests/unit/test_m4_selfcheck.py` |
| 10 | **B-P21** and **B-P22** applied: `02-hld.md` line 342 `C: stationary (declared)`; the W2 and W3 delivery-summary rows of `02-hld.md` §7.2/§7.3 and of `03-lld-M4-mapping.md` §6.3/§6.4 replaced by the mechanical `f"{a}: {HOW} ({declared})"` lines, each with a one-line erratum quoting R2 | `design/02-hld.md`, `design/03-lld-M4-mapping.md` |

## The `may_packet` predicate, and the evidence it was derived from

§3.8 lines 4-9 state `CIRCUIT(site)` in prose and never define `may_packet`. It was derived from
measurement and then **found in the source**, which is a stronger result than the brief asked for:
it is `air-dma-to-channel`'s own auto-upgrade rule (`mlir/lib/Transform/AIRDmaToChannel.cpp:1598-1740`,
pass option `shim-dma-channels-per-col`, default 2, `mlir/include/air/Transform/Passes.td:1808-1812`;
wheel `0.0.1.2026091204+ff95a9b`, whose commit `ff95a9b` is the checked-out `mlir-air` tree).

> Per segment and per direction — **input** is a herd-side get with a launch-side L3 put,
> **output** a herd-side put with a launch-side L3 get — the per-column shim pressure is
> `#non-broadcast + Σ_span ceil(members_span / span)`, where a broadcast channel contributes
> `prod(size)` members at `span = broadcast_shape[0] // size[0]` (it is split into one channel per
> bundle index by `air-specialize-dma-broadcast`, which runs first). If that exceeds 2, **every**
> L3-attached channel of that direction becomes `channel_type = "npu_dma_packet"` and multiplexes;
> otherwise they stay circuit-switched `aie.flow`s. Core-to-core channels are never upgraded.

**Evidence, regenerated this session.** Five probes, each `python <probe> > x.mlir` then
`timeout 600 aircc --device npu1 --output-format=none --tmpdir <scratch>/x.tmp x.mlir`, **one at a
time**, tmpdirs outside the repo; then `aie.*.mlir` tabulated per channel.

| probe | L3 in-channels → pressure | predicted | measured lowering | `aircc` |
|---|---|---|---|---|
| `q/q7_a.py` (W1, PI=PJ=2) | `A2L1` bcast(2 members, span 1) + `B2L1` bcast(2, span 2) → **3** | packet | 3 `aie.packet_flow`; `air_A2L1_0`, `air_B2L1_0`, `air_B2L1_1` all on `shim_noc_tile_0_0, MM2S, 0`; `C2L3` (out, pressure 1) 4 circuit `aie.flow` | exit 0, 0 `error:` |
| `q/pi2/w2_pi2.py` (W2, PI=2) | `UIn` → **1** | circuit | **0** packet flows; `aie.flow(shim_x_0 → tile_x_2)`; halo links circuit | exit 0, 0 `error:` |
| `q/q2_w2.py` (W2, PI=4) | `UIn` → **1** | circuit | **0** packet flows; two circuit flows target `tile_1_2, DMA:0` | **exit 1**, `'aie.connect' op … targets same dst` |
| `q/w3b.py` (W3, no `q`/`r`) | `WestIn` → **1** | circuit | **0** packet flows | exit 0, 0 `error:` |
| `review/w3c.py` (W3, `q`/`r` staged) | `WestIn` + `QIn` + `RIn` → **3** | packet | 9 `aie.packet_flow`; all three channels carry `channel_type = "npu_dma_packet"` in `placed.w3c.mlir` | exit 0, 0 `error:` |

Every row is reproduced. The discriminating pair is `w3b` vs `w3c`: the **same** `WestIn` channel
is circuit alone and packet beside `QIn`/`RIn`, so no predicate over one channel's own shape can
work — which refutes all three candidates the brief listed (`broadcast_shape`: `RIn`/`WestIn` are
plain and packet; puts under a temporal segment-scope loop: `WestIn`'s are, and it is circuit in
`w3b`; distinct L3 endpoints per shim: `q7a`'s pressure is 3 from two channels). The pass printed
its own arithmetic when run directly:

```
$ air-opt --air-dma-to-channel w3c.mlir            # and with the real pipeline prefix for q7a
w3c.mlir:14:7: warning: auto-upgrading 3 input channels to dma_packet (per-column pressure 3
  exceeds shim DMA limit of 2)
q7a.mlir:11:7: warning: auto-upgrading 4 input channels to dma_packet (per-column pressure 3
  exceeds shim DMA limit of 2)
```

**Labelled measured, not contractual (R-19, R-21)**, because it is a pass option's default on a
pinned wheel rather than a documented guarantee: `--shim-dma-channels-per-col` or
`--force-shim-packet-flow` changes it, and a shim-column assignment other than the `same_column`
default the pass assumes would too. That is exactly why §3.8 splits the verdict: a circuit-switched
overflow is a `DMA-CHANNELS` **error**, and an overflow that only packet-capable channels cause is a
**warning**. W2 at `PI = 4` is an error (`UIn` alone → pressure 1 → all three inbound circuit) and
W3 is a warning (three L3 inputs → pressure 3 → packet), which is what §7 requires.
**Not measured**: the outbound half at pressure > 2 — no probe has three L1→L3 channels.

## Two readings the LLD's pseudocode leaves open

| # | Reading | Why |
|---|---|---|
| 1 | §3.7.1 line 24 multiplies `trips` through **every** `LoopPlan`; the implementation multiplies through a `"sequential"` loop and **enumerates** an `"unrolled"` one, binding its variable | An unrolled loop is a trace-time Python loop whose variable *is* a bundle index. Multiplying leaves `pi_bundle` unbound and `A2L1[pi_bundle, 0]` unevaluable; enumerating is what makes §6.1's own statement — "`A2L1` put key `[0,0]` count 4" — and §7's `test_M4_balanced` (per-index rows) come out |
| 2 | §3.7.2 rule 5 excludes the loop back edge; it is applied to unrolled loops too | Uniform, and conservative: fewer edges can hide a cycle, never invent one. A real cycle across two trips of an unrolled loop would be missed, which no plan we build can have (a bundle index is spatial, the trips are independent) |

## The interpreter's scope, and its limits

`run(plan, tensors)` executes **the plan, not the emitted IR** — D-9 and the honest-limits slide
stand unchanged. It models the four things that can make a plan wrong on paper: the region
arithmetic (a wrong slab shows as a wrong answer), the fan-out (a put reaches every destination
index), the blocking discipline (a get yields while its FIFO is empty, so a missing put is a
`RuntimeError("deadlock")` naming the blocked sites), and the allocation rule (`air.alloc` inside a
loop re-allocates per trip). It does **not** model: token dependencies and `is_async` (every actor
is sequential within itself), the physical herd and `repeats` (the logical grid is what runs),
`air.sequential`'s true concurrency, L2, DMA ordering beyond FIFO per `(channel, index)`, or
anything about the lowered IR. A W1 run is 4 PEs × 32×32×16 scalar stores ≈ 2 s of the suite's
4.2 s; W2 and W3 will need the same care when their plans exist.

## Verified (command → result)

| # | Command | Result |
|---|---|---|
| 1 | `.venv/bin/python -m pytest` | **483 passed, 4 deselected** in 4.16 s (was 456 passed) |
| 2 | `PYTHONHASHSEED=1` / `=2`, `-vv` | 483 passed both times; `diff` of the two 483-line id/outcome lists is **empty** |
| 3 | `.venv/bin/python -m pytest -m slow` | **4 passed**, 483 deselected, 3.23 s — unchanged |
| 4 | `import spatial.m4_selfcheck` in a fresh process | `air: False, m5: False, m4_mapping: False, numpy: False` (I-1; the checker is exact integer arithmetic, so numpy is not imported either) |
| 5 | the five `aircc` runs of the evidence table, one at a time, `--tmpdir` under the scratch dir | exits 0/0/**1**/0/0 with 0/0/**1**/0/0 `error:` lines, as tabulated |
| 5a | — | `aircc` also writes `elfs_<core>/` and `measured_stack_sizes.mlir` into its **cwd**, `--tmpdir` notwithstanding. The probe script ran from the repo root and left nine stray paths, which were removed; `m6_tools.aircc` already passes `cwd=work`, so the product is unaffected — but any future probe run must `cd` outside the repo first |
| 6 | `air-opt --air-dma-to-channel` on each probe | the two `auto-upgrading` warnings quoted above; silent on `w2pi2`, `w2pi4`, `w3b` |
| 7 | `m4.plan(w1_legal.legal(t))` with the real `self_check` wired in | accepted on both targets; no golden changed |

## Blockers closed

| # | Resolution |
|---|---|
| **B-P21** | **Closed.** `02-hld.md:342` now reads `C: stationary (declared)` — one word, the architect's ruling. |
| **B-P22** | **Closed.** The delivery block is the mechanical `f"{a}: {HOW} ({declared})"` line per operand and nothing else; protocol facts appear through the channel lines. W2 renders `U: stationary (declared)`, W3 renders `S: forward along px (declared)`, `q: multicast along px (derived)`, `r: stationary (derived)`. The prose sentences in `03-lld-M4-mapping.md` §6.3/§6.4 and `02-hld.md` §7.2/§7.3 are replaced, each with a one-line erratum quoting R2. |
| **B-P23** | **Applied as ruled.** A self-check failure that can only be M4's own construction bug raises `MappingError` with `PROTOCOL-UNSUPPORTED`, `details["internal_consistency"] = True`, `details["invariant"] = <§5.6 number>`, a `reason` beginning `internal:` and naming the workload and the invariant, and a `fix` asking for a bug report (HLD §4.3). The catalogue stays at 43. Recorded for a possible dedicated code after the freeze, with M5's B-P13. |

## Open / blockers

| # | Item | Detail |
|---|---|---|
| **B-P24** *(new)* | The `along` of an overridden delivery row is the **clause's** axis, not the PE axis | §3.2 line 21a says `clause.along`, so `forward("S", along=ax.j0)` gives `("S", FORWARD, "j0", True)` and the summary renders `S: forward along j0 (declared)` — while §6.4 and `02-hld.md` §7.3 print `along px`. The derived rows all use `PE_AXIS_NAME`. One of the two must move: either line 21a maps the clause axis to the PE dim carrying it, or the documents say `j0`. It bites at **P5**, when W3's summary golden is frozen. No code changed this phase. |
| **Substitutions in the corpus** | three, each recorded in the fixture's docstring | (a) `dropped_get` shortens the `j_drain` loop rather than deleting one `get`, because W1's four `C2L3` gets are four trips of one site; (b) `broadcast_underconsumed` guards the `A2L1` get to `ty == 0` for the same reason; (c) `dma_packet_warn` is three packet-capable inbound plus one circuit rather than "three of which two are packet-capable", because the upstream rule upgrades **every** L3 channel of a direction at once, so a two-of-three split is unreachable. `test_M4_l1_agrees_with_m3` widens `acc` to `(32,64)` rather than inflating `bytes`, which M0's `I39` forbids. |
| **M4 §7 rows still waiting** | `test_M4_balanced` (W2 at `PI=2`, both boundary PEs), `test_M4_drains_every_plane`, `test_M4_odd_T_peel`, `test_M4_even_T_no_peel`, `test_M10_w2_acyclic`, `test_M4_halo_indices`, `test_M4_stencil_is_five_point` — **P4**; `test_M5_wavefront_balance`, `test_M5_stages_q_and_r`, `test_M5_drains_every_row`, `test_M5_three_channels`, and `test_P3_dma_packet_warns` **on W3 itself** — **P5**; `test_M6_cascade_chain`, `test_M6_cascade_orientation`, `test_M11_residency_line`'s flip half — **P6**. Each has a hand-written stand-in of the same shape here, so the checker is exercised now and only the *plan* is missing. |
| **B-P10**, **B-P12**, **B-P13**, **B-P15**, **B-P16**, **B-P20**, **B-O8** | unchanged | — |

**Not verified in this phase**: that the packet/circuit rule holds on any wheel but the pin (it is a
pass option's default, R-19/R-21); the outbound half of `may_packet` at pressure > 2 (no probe
reaches it); that W2's and W3's **real** plans pass the checks (neither is built — the synthetic
halo and the `BranchNode` plan are hand-written stand-ins with the right shape, not the protocols
of §3.6.1/§3.6.2); that the interpreter agrees with the device (it interprets the plan, never the
emitted IR — D-9); anything on hardware.

---

# Phase P4 — W3, the anti-diagonal wavefront, end to end (gate G3)

*Person B, 2026-09-13. Branch `role-b`. M4's `FORWARD` protocol builder (§3.6.2), the `q`/`r`
staging, the segment-scope source and drain, M5's `BranchNode`/`Select`/`MaxMin` rows exercised
for real, goldens on both targets, `aircc`, and the interpreter against a textbook DP.
**Not pushed** — the architect verifies and pushes.*

## Landed

| # | What | Where |
|---|---|---|
| 1 | **`_origins` gained the axis's own `lo`** — the one edit P2b asked P4 to make. W1's axes all start at 0 so the term was invisible; W3's `i` and `j` start at 1, and without it `8·tx` does not cancel in the band rewrite | `spatial/m4_mapping.py` |
| 2 | **`band_geometry`** — the `Band` a wavefront needs, derived rather than tabulated: which array dim is the row and which the PE column band, the owned column count `CW`, the west overhang from the operand's own access offsets, the row span (which must be 2 — the row being computed and the one before it) and the row axis's bounds. Every shape it cannot build raises `NotImplementedError` naming the phase | same |
| 3 | **`wavefront_buffers`** — `qb`, `rb` (the staged read operands through the ordinary `TILE_SHAPE`), then `prev`/`cur` at `[CW + ghost]` carrying `operand` and `edge_in`/`edge_out` carrying none. `240 == 232 + 8` by construction | same |
| 4 | **`swap_loop(axis, lo, hi, kind, depth, trip, base)`** — the unroll-by-two-and-peel of §3.6.1 lines 1-11 as a **reusable function**, not a branch of the builder: `trip(index, phase, order)` builds one trip, an odd count is peeled at the loop's own depth, and the returned phase is what tells a caller which buffer is live (W2 reuses it at P5) | same |
| 5 | **`wavefront(...)`** — the six channels, the segment body (`QIn` put, the `pj_bundle` `RIn` puts, the `i_source` row loop, the herd marker, the `i_drain` `EastOut` loop, the `pj_bundle`×`i_drain` `SOut` drains) and the herd body (six allocs, the two staging gets, the `prev` zeroing, the row swap loop). The compute store is `Statement.expr` rewritten load by load: `_in_band` for the forwarded operand, `_in_l1` for everything else | same |
| 6 | **`classify` maps a declared `along` to its PE axis** and rejects an `along` no `place()` carries — **B-P24** closed | same |
| 7 | **A read operand named by `stream`/`forward` is `PROTOCOL-UNSUPPORTED`** naming the clause (ruling R-W3-4) | same |
| 8 | **`buffer_name` avoids colliding with a tensor name**: `q` stages into `qb`, `r` into `rb` (§5.6 forbids one name covering a tensor and its tile) | same |
| 9 | **The P3 warnings reach the summary**: `plan()` builds the summary, runs `self_check`, then `dataclasses.replace`s one `warning: …` line per warning after the channel lines | same |
| 10 | **M5's `Select` predicate** uses `ops.equal`/`ops.not_equal` for `==`/`!=`, and the `Load` arm coerces through `BufferExpr.coerce` — **B-P16** closed | `spatial/m5_emit.py` |
| 11 | **`PIPELINES["aie"]` places herds before `air-to-aie`**, so `lock_init_histogram` is readable at all | `spatial/m6_tools.py` |
| 12 | **`w3_legal.legal(target, MQ=…)`** — the fixture takes a row count, so the peel has a test | `tests/fixtures/mappings/w3_legal.py` |
| 13 | **32 new tests** and seven new goldens (below) | `tests/` |

## R-W3-1 — the verdict is **kept**, with both measurements

The brief allowed a fallback that drops `EastOut` and the tail PE's put. It is **not** taken:
the source puts `S[i, 0:1]` and the drain gets `S[i, NR:NR+1]` on the kernel's own `S`, and
`MappingPlan.tensors` is exactly `(q, r, S)`.

| # | Measurement | Result |
|---|---|---|
| (a) | `aircc --device npu1 --output-format=none` on the emitted module, `--tmpdir`/cwd in scratch | **exit 0**, **0** `error:` lines, **0.3 s**, all 22 stages, four `.elf`s written |
| (a) | the same with `--device npu2` | **exit 0**, **0** `error:` lines, **0.31 s** |
| (b) | `air-opt … -pass-pipeline='builtin.module(air-verify-hierarchy-locality)'` | exit 0, **stderr empty** |
| (b) | the same with `{strict=true}` | exit 0, **stderr empty** |

(a) is a real verdict rather than an exit code, because `aircc` runs
`air-verify-hierarchy-locality{strict=…}` on the **placed** IR by default —
`tools/aircc/aircc.cpp:1213-1218` builds the pipeline and `:264` is `cl::init(PIV_error)`, which
is `strict=true`. So the two ends of the same `S` row, one of them an address `SOut` also
writes, are not a race the strict verifier can see. FR-M5's "closed at both ends by a source and
a drain" is therefore true as written, and `test_M5_wavefront_balance` keeps its **five**
indices.

## The P2b table, reproduced on the real plan

`S[i-1,j-1] → prev[j1]`, `S[i-1,j] → prev[j1+1]`, `S[i,j-1] → cur[j1]`, `q[i-1] → qb[i-1]`,
`r[j-1] → rb[j1]`, and the write `S[i,j] → cur[j1+1]` — exactly what P2b derived by hand, now
asserted by `test_M4_wavefront_l1_subscripts`. The band is rank 1 where the access is rank 2:
the **row** offset does not index a buffer, it picks which of the two swapped buffers holds that
row, and the column is the access's column minus the band's L3 origin, so `8·tx` cancels the way
W1's tile origin does. The `i_off` of §6.4's `ROW` cancels out of the row offset by construction
(it is in both terms), which is why phase 1 needs no separate rule.

## The DMA warning line, verbatim (PE 0; PEs 1-3 differ only in the channel list)

```text
warning: PE [0] names 3 inbound channels (QIn, RIn, WestIn) against 2 S2MM, but QIn, RIn, WestIn
lower to packet flows and multiplex onto one shim DMA channel; the circuit-switched count is 0,
within the budget of 2 — measured, not contractual (design/03-lld-M4-mapping.md §3.8, risks
R-19/R-21)
```

(One physical line in the file; wrapped here.) PEs 1-3 name `(QIn, RIn, West)` with a
circuit-switched count of **1** — `West` is core-to-core and binds a real DMA channel, the two
west gets being on exclusive branches. Four warnings, one per PE, inbound only; the outbound
count is 2 on every PE and is within budget.

## `aircc`, `air-opt` and `ir_facts` results

| target | command | exit | `error:` lines | wall |
|---|---|---|---|---|
| npu1 | `aircc --device npu1 --output-format=none` | 0 | 0 | 0.30 s |
| npu2 | `aircc --device npu2 --output-format=none` | 0 | 0 | 0.31 s |
| npu1 | `air-opt -pass-pipeline='builtin.module(air-verify-hierarchy-locality)'` | 0 | 0 | < 0.1 s |
| npu1 | the same, `{strict=true}` | 0 | 0 | < 0.1 s |

`ir_facts` for W3/npu1: `broadcast_pattern_count = 0` (declaring `broadcast_shape` bypasses the
detector — R-04's tripwire, the same 0 W1 gives), `cascade_channels = 0`, and
**`lock_init_histogram = {"0": 20, "1": 16, "2": 4}`** — 40 locks over four cores.

**`PIPELINES["aie"]` had to be fixed to produce that number at all.** `air-to-aie` alone, on an
unplaced module, gives `'aie.tile' op column index (4) must be less than the number of columns in
the device (4)`: a 1-D `grid(4)` herd keeps its logical origin and asks for columns 1..4. `aircc`
places first (`aircc.cpp:1158-1162`), so §3.7's pipeline now prepends
`air-place-herds{num-rows=6 num-cols=<4|8> row-anchor=2 col-anchor=0}` with the geometry `aircc`
itself resolves (`aircc.cpp:1003-1022`). Cross-checked once: this pipeline and `aircc`'s own
`air_project/aie.w3.base.npu1.air.mlir` give the **identical** histogram.

## The probe comparison (`vendor/probes/review/w3c.py`)

| fact | ours | probe `w3c` | why |
|---|---|---|---|
| channels | `EastOut[1]`, `QIn[1] bcast[4]`, `RIn[4]`, `SOut[4]`, `West[3]`, `WestIn[1]` | same names and sizes except `QIn[4]` | ours is FR-M2's multicast geometry — one put, `broadcast_shape=[4]`; the probe replicates the put four times |
| `tensors` | 3 — `q`, `r`, `S` | 5 — `Zrow`, `Q`, `Rr`, `Sink`, `Out` | ruling R-W3-1: the source and the drain use `S`'s own boundary columns, so no synthetic tensor exists |
| `scf.if` | 4 | 4 | identical: head and tail guard, once per unrolled row |
| `arith.select` | **2** | **0** | the probe computes a literal `+2`; ours is the `Select` on `qb[i-1] == rb[j-1]` — G-8, and the difference between Smith-Waterman and something else |
| `arith.maxsi` | 6 | 4 | ours is the 4-ary `max(0, …)`; the probe's is 3-ary with no zero floor |
| `scf.for` | 10 | 7 | ours drains **every row** (four `i_drain` loops) and zeroes `prev` with a loop rather than `ops.fill` |
| `air.channel.put` / `get` | 12 / 11 | 14 / 11 | the `QIn` fan-out again: one put against four |
| `memref.alloc` | 6 | 6 | identical |
| `aie.flow` (circuit) | **8** | **8** | **byte-identical**: four `SOut` drains `tile_x_2 DMA:0 → shim_x_0 DMA:0`, one `EastOut` `tile_3_2 DMA:1 → shim_3_0 DMA:1`, three `West` links `tile_k_2 DMA:1 → tile_{k+1}_2 DMA:1`. The wavefront's physical realisation is exactly the shape the probe measured |
| `aie.packet_flow` | **6** | **9** | the probe's nine are 4 PEs × (`QIn`, `RIn`) + `WestIn`. Ours are six because `QIn`'s `broadcast_shape` makes one flow with **two** `packet_dest`s (`tile_0_2` and `tile_1_2`) where the probe needs four separate flows. The brief expected 9; 6 is the same design with the multicast declared |
| `aie.core` / `aie.lock` | 4 / 40 | 4 / 40 | identical |

## Readings taken, where the documents disagree

| # | Reading | Why |
|---|---|---|
| 1 | **The wavefront's staging channels are `QIn`/`RIn`/`SOut`, not §3.5's `{a}2L1`/`{a}2L3`** | §3.5's rule names W1's channels `A2L1`/`C2L3`; §6.4's table, `02-hld.md` §7.3, FR-M5's acceptance and §7's rows all name the wavefront's six `<Name>In`/`<Name>Out`, which is the family `WestIn`/`West`/`EastOut` already belongs to. `q2L1` would also read badly in the judge-facing summary. The fill/compute/drain protocol keeps §3.5's spelling; only the wavefront uses `wavefront_channel_name`. |
| 2 | **The summary's `warning:` lines are appended after `self_check`**, by `dataclasses.replace` on the finished plan | §3.8 says a warning "is recorded in the summary and printed", and `warnings(plan)` enumerates every herd coordinate of the **finished** plan, so it cannot run inside §3.1's pass 9. Restructuring §3.1 to compute the P3 report twice would be the alternative. `summary(...)` called on its own therefore carries no warning line; `plan()`'s result always does. |
| 3 | **A site inside a `BranchNode` arm carries the order of the `BranchNode`**, not 0 | §5.2 says `order` is "the index in its enclosing body", which for a one-site arm is 0 — and then W3's two unrolled rows give two `WestIn.get.0@herd` ids for two different sites, which collapses two distinct nodes into one in the P2b graph. Using the branch's own index keeps every id distinct (`WestIn.get.0@herd` and `WestIn.get.6@herd`) and keeps `_check_orders`' top-level rule exact. |
| 4 | **W3's npu1 and npu2 AIR texts are byte-identical** | `physical_herd` is `(4,)` on both (the cap is 4 and 8, and 4 divides both), and `build(target=)` stamps nothing into `str(module)`, so there is no strip-mine loop to differ over the way W1's does. Both goldens are kept: two identical files record the fact, and a future wheel that starts stamping the device shows up as a diff. The two `plan.json` goldens **do** differ, by `schedule.target`. |
| 5 | **`W3_FACTS` is `broadcast_pattern_count`, `cascade_channels`, `lock_init_histogram`** — not W1's five | `pingpong_unroll`, `hoist_alloc_count` and `pingpong_iter_args` are ping-pong facts and W3 has no ping-pong candidate (no operand is re-fetched per trip, so `DEPTH` is 0 everywhere and `double_buffer` is empty). Asserting 0 for them would freeze a number with no content. |

## Verified (command → result)

| # | Command | Result |
|---|---|---|
| 1 | `.venv/bin/python -m pytest` | **515 passed** (was 483), 7 deselected, 5.4 s |
| 2 | the same at `PYTHONHASHSEED=1` and `=12345` | 515 passed each; `test_M4_plan_stable[w1|w3]` and `test_E10_byte_identical_w3` also run a fresh process under another seed |
| 3 | `.venv/bin/python -m pytest -m slow` | **7 passed** (was 4: `test_W3_aircc_none[npu1]`, `[npu2]` and `test_W3_hierarchy_locality` are new) |
| 4 | `pytest --update-goldens` | 7 new goldens; `git diff --stat tests/golden` is **empty**, so every W1 golden is byte-identical before and after |
| 5 | the interpreter against a textbook two-loop numpy DP, `numpy.random.default_rng(0)`, both targets | **exactly equal**; row 0 and column 0 stay zero; `q` and `r` are untouched; `dtype == int32` |
| 6 | the same at `MQ = 31` (the peeled odd trip count) | exactly equal; every wavefront index shows 31 puts against 31 gets |
| 7 | `m4.plan(w3_legal.legal(t))` with the real `self_check` | accepted on both targets, four warnings, no error |

## Blockers closed

| # | Resolution |
|---|---|
| **B-P16** | **Closed.** `ops.maximum`/`minimum`'s `_elementwise` guard admits `(Buffer, BufferExpr, int, float)` and not `BufferSlice` (`python/air/api/ops.py:384-391`), so M5's `EMIT_EXPR` `Load` arm now returns `BufferExpr.coerce(buf[subs])` (`python/air/api/_value.py:1046-1049`) — the API's own entry point, not `load + 0`, which would put an `arith.addi` in the IR the plan never asked for. It is a no-op for every other consumer: `BufferSlice.__add__` and friends call `_as_leaf()` first (`_value.py:749-751`), which is what `coerce` calls, and **W1's goldens are byte-identical**. `BufferExpr` is not re-exported by `air.api.__init__`, so the import is `from air.api._value import BufferExpr`, lazy inside `emit()`. |
| **B-P24** | **Closed as R-W3-2.** `classify` maps a declared `along` through `mapping.schedule.place.index(...)` to `PE_AXIS_NAME`, so W3 renders `S: forward along px (declared)` and one column of `MappingPlan.delivery` means one thing. An `along` no `place()` carries raises `PROTOCOL-UNSUPPORTED` naming the clause. `test_M3_stream_override`'s classify half now expects `("A", "FORWARD", "py", True)`. |
| **A second finding, not previously logged** | `==`/`!=` are **not** operators on a buffer value: `BufferExpr` and `BufferSlice` deliberately leave `__eq__`/`__ne__` undefined, because defining `__eq__` sets `__hash__` to `None` (`_value.py:795-805`), so `x == y` is Python's identity comparison and `ops.select` rejects the `bool` it produces by name (`ops.py:809-816`). `ops.equal`/`ops.not_equal` (`ops.py:779-792`) are the spelling and build the same `arith.cmpi`. M5 §3.6 line 17's `<cmp>` therefore holds for `<`, `<=`, `>`, `>=` only; the erratum is in the LLD and the two names are in §3.2's closure test. |

## Naming rule, extended

`06-interfaces.md` §5.5 gained one bullet: a **source** loop is `<axis>_source`, the mirror of
`<axis>_drain`, naming the axis whose trips it enumerates — W3's `i_source`. §6.1's protocol has
no source loop, so the rule had no spelling for one. Both W3 row loops (`i_source`, `i_drain`)
are `air.sequential` per ruling **R-W3-3**: a row index is not a channel bundle index, so
`LOOP_KIND` takes its default, and only the `pj_bundle` loop around the `SOut` drains is
`unrolled`. `03-lld-M5-emitter.md` §6.4 lines 31-35, which draw both as Python loops, carry the
erratum.

## Documents edited

| file | edit |
|---|---|
| `design/03-lld-M4-mapping.md` §6.4 | the `EastOut` row is `S[i, NR:NR+1]` under a sequential `i_drain` loop, not `Sink[r:r+1]`, with R-W3-1's two measurements; the delivery line carries R-W3-2's erratum |
| `design/03-lld-M5-emitter.md` §6.4 | lines 17, 32 and 33-35 rewritten (`ZERO_COL`/`SINK` → `S`; the drain loops sequential, only `p` unrolled), with one erratum paragraph; §3.6's `Select` paragraph gains the `ops.equal` and `BufferExpr.coerce` errata |
| `design/02-hld.md` §7.3 | the `EastOut` row names `S[i, NR:NR+1]` and R-W3-1's measurements |
| `design/06-interfaces.md` §5.5 | the `<axis>_source` bullet |

## Open / blockers

| # | Item | Detail |
|---|---|---|
| **M4 §7 rows still waiting** | `test_M4_balanced`, `test_M4_drains_every_plane`, `test_M4_odd_T_peel`, `test_M4_even_T_no_peel`, `test_M10_w2_acyclic`, `test_M4_halo_indices`, `test_M4_stencil_is_five_point`, `test_M4_halo_protocol` — **P5** (W2's halo); `test_M6_cascade_chain`, `test_M6_cascade_orientation`, `test_M11_residency_line`'s flip half, `test_E8_cascade_text` — **P6**. W3's own rows are all live now. |
| **`PIPELINES["aie"]` is Person C's module** | The placement prefix was added by B because `lock_init_histogram` is unreadable without it. The geometry table `AIE_GEOMETRY` is two rows transcribed from `aircc.cpp:1003-1022`; a third target would need a third row. For C to confirm. |
| **`swap_loop`'s `phase` return is unused** | W3 drains inside `ROW`, so there is no "which buffer is live" question (§6.4 says so). W2's drain at P5 is the first caller that needs it, and it is returned now so the signature does not change then. |
| **B-P10**, **B-P12**, **B-P13**, **B-P15**, **B-P20**, **B-O8** | unchanged | — |

**Not verified in this phase**: that the emitted module **computes** Smith-Waterman on hardware —
the interpreter interprets the **plan**, never the emitted IR (D-9), and only a device run closes
that (the honest-limits slide is unchanged); that `aircc`'s 0.3 s holds on another machine or
another wheel; that the packet/circuit split holds off the pin (R-19/R-21); that a wavefront with
a herd rank of 2, more than one `FORWARD` row, a non-unit row step, an east-side overhang or a
row span other than 2 works — each raises `NotImplementedError` naming the phase rather than
guessing; W2's and W1-flip's real plans (neither builder exists yet).

---

# Phase P5 — W2, the Jacobi halo exchange, end to end (gate G4)

*Person B, 2026-09-13. Branch `role-b`. M4's halo protocol (§3.6.1), the `cur`/`next` pair with
unroll-by-two and the odd-`T` peel, every plane drained, the `PI = 4` `DMA-CHANNELS` negative on
the real plan, goldens on both targets, `aircc`, experiment **E1** on our own module, and the
interpreter against a two-loop numpy Jacobi. **Not pushed** — the architect verifies and pushes.*

## Landed

| # | What | Where |
|---|---|---|
| 1 | **`strip_geometry`** — the `Strip` a halo needs, derived rather than tabulated: which array dim is the timestep (the `sequential` axis, whose write offset `+1` against read offsets `0` is what makes the pair a pair), which is the exchanged dim (the placed axis the clause names), the owned extent, the per-dim halo from the `WindowClause`, and the L1 shape `[owned + 2·halo]` per kept dim. `(10, 16)` falls out of `HS = 8`, `halo = 1` and `W = 16`; **no W2 literal shape anywhere**. Every shape it cannot build raises `NotImplementedError` naming the phase, or `PROTOCOL-UNSUPPORTED` naming the clause | `spatial/m4_mapping.py` |
| 2 | **`halo_buffers`** — `cur`/`next`, both carrying `operand="U"`, so §5.6 invariant 5 charges them against `LegalMapping.l1_bytes`: `1280 == 1280`, no protocol buffer added. Neither is a ping-pong candidate (R-W2-4) | same |
| 3 | **`halo(...)`** — four channels, the segment body (`pi_bundle` `UIn` puts, the herd marker, `pi_bundle` × `t_drain` `UOut` gets) and the herd body (two allocs, the `UIn` get, the seed copy, the `swap_loop` over `t`). `swap_loop` is reused from P4 unchanged, which is what it was factored out for | same |
| 4 | **`_in_strip`** — the strip is rank 2 where the access is rank 3: the timestep offset picks the buffer (`0` → src, `+1` → dst) and every other dim is the access's index minus the staged slab's origin, so `tx·HS` cancels and the ghost row shifts the result by the halo. The mirror of P4's `_in_band` | same |
| 5 | **`_pairs`** — channel edges are FIFO-paired where the pairing is defined (below) | `spatial/m4_selfcheck.py` |
| 6 | **`_site` takes `guard` / `is_async` / `depends_on`** — the halo's boundary puts are `is_async=True` and its ghost gets keep `depends_on=()` | `spatial/m4_mapping.py` |
| 7 | **`wavefront_channel_name` → `staging_channel_name`** — W2's `UIn`/`UOut` follow the same `<Operand>In`/`<Operand>Out` rule as W3's `QIn`/`RIn`/`SOut`, so the name no longer says "wavefront" | same |
| 8 | **42 new tests** in the default run, two more `slow`, and nine new goldens (below) | `tests/` |

## The one substantive deviation: **the swap pair is seeded, not just `cur`**

§3.6.1 stages plane `lo` into `cur` alone and says so. §6.3's coverage paragraph then says of the
drain that *"the value written back is the one that was staged in, so the L3 image equals the
oracle everywhere"* — a claim about **every** drained plane. The planes alternate between `cur`
and `next`, and the drain puts the strip **whole** (all `W` columns, §6.3's own superset
sentence), so both strips must carry the staged read-only boundary. `next` is never a `get`
target. Three values are therefore undefined without a seed:

* `next[:, 0]` and `next[:, W-1]` — the Dirichlet **columns**, which the update never writes
  (`j` runs `1..W-1`) and no exchange touches;
* PE `0`'s `next[0, :]` and PE `PI-1`'s `next[HS+1, :]` — the domain-edge **ghost rows**, whose
  guards (`tx > 0`, `tx < PI-1`) are false exactly there.

With them undefined the plan computes the Dirichlet boundary as zero on every second plane and
as plane 0's value on the others, and the drain writes that alternation into L3 — so no oracle
of either reading matches, and §6.3's sentence is false as written. The plan therefore carries a
`StoreNode` copy `next[i1,j] = cur[i1,j]` over the whole strip, immediately after the `UIn` get:
the same device the accumulator zeroing (§6.1) and the cascade accumulate (§6.2) already use for
a node no kernel statement corresponds to. `UIn` stays **1 put / 1 get** per index, which is what
§7's `test_M4_balanced` row asks for, and the herd body gains one node. Errata are in
`03-lld-M4-mapping.md` §3.6.1 and §6.3 and in `03-lld-M5-emitter.md` §6.3;
`test_M4_halo_seeds_both_strips` is the test, and `test_sem_compute_nodes_w2` is what would fail
without it.

## The second deviation: **P2b channel edges are FIFO-paired**

§3.7.2 writes the channel edge as *"for every put node and every get node whose (channel,
concrete index) match … a directed edge put → get"*. On W2's real plan that rule reports a cycle:

```
put_n(phase 1, PE 1) --channel--> get_n(phase 0, PE 0) --program--> put_s(phase 1, PE 0)
                     --channel--> get_s(phase 0, PE 1) --program--> put_n(phase 1, PE 1)
```

Every channel edge in it pairs a **later** put with an **earlier** get, which a FIFO never does:
the k-th get receives the k-th put and waits on that one. The all-pairs rule is an
over-approximation that is harmless while one body holds one transfer per index and stops being
harmless once `swap_loop` puts two timesteps in one body — and §3.7.2's own argument that the
halo is acyclic is made *within a timestep*, which is the framing the unroll breaks.

`_pairs` therefore zips the k-th put to the k-th get **when the pairing is defined** — both sides
the same number of nodes, each side one coordinate's own program order — and falls back to
all-pairs otherwise (several producers into one index; a body shape that splits one side and not
the other, which is W3's guarded `WestIn`: one segment put node against two herd get nodes). It
is strictly fewer edges, so it cannot turn a cyclic plan acyclic by accident anywhere the pairing
does not apply, and where it does apply it is the exact semantics. The reversed halo still raises
`CHANNEL-CYCLE` on both the synthetic fixture and W2's real plan
(`test_M10_cycle_rejected`), and W1's, W3's and the corrupt corpus's verdicts are all unchanged.

## The `PI = 4` diagnostic, verbatim

```text
DMA-CHANNELS: PE [1] needs 3 circuit-switched inbound DMA channels (ToNorth, ToSouth, UIn) but
an AIE2 core tile has 2 S2MM — a budget of 2
  in clause: grid(4)
  fix:       shrink the herd so fewer neighbours meet at one PE, or stage fewer operands from
             L3: grid(4) puts 3 inbound flows on PE [1]
  details:   coord=(1,), direction='inbound', channels=('ToNorth','ToSouth','UIn'), count=3,
             budget=2, circuit_switched=('ToNorth','ToSouth','UIn'),
             all_channels=('ToNorth','ToSouth','UIn')
```

It is raised from inside `m4.plan`, so `w2_legal.legal(target, T=4, PI=4)` never produces a plan
at all. `aircc` is **measured** to fail on this shape (`'aie.connect' op … TileID(1, 2) targets
same dst`, REVIEW-round1 P-R2), so the failing `aircc` was **not** re-run this phase: the checker
rejects first and that is the point of the check. Per-column shim pressure for the inbound
direction is 1 (`UIn` alone), so `UIn` is not auto-upgraded to a packet flow and all three are
circuit-switched — which is why this is an error and W3's three inbound channels are a warning.

## `aircc`, `air-opt` and `ir_facts` results

| target | command | exit | `error:` lines | wall |
|---|---|---|---|---|
| npu1 | `aircc --device npu1 --output-format=none` | 0 | 0 | 0.23 s |
| npu2 | `aircc --device npu2 --output-format=none` | 0 | 0 | 0.22 s |
| npu1 | `air-opt -pass-pipeline='builtin.module(air-dependency)'` | 0 | 0 | < 0.1 s |

Both runs used a scratch `--tmpdir` and `cwd` outside the repository, one at a time. The lowered
design is **all circuit-switched on both targets**: 6 `aie.flow`, 0 `aie.packet_flow` —
`shim_0_0 → tile_0_2 DMA:0` and `shim_1_0 → tile_1_2 DMA:0` (`UIn`), the two reverse flows on
DMA:0 (`UOut`), and `tile_1_2 → tile_0_2 DMA:1` / `tile_0_2 → tile_1_2 DMA:1` (`ToNorth`,
`ToSouth`). Every core sits at exactly 2 S2MM and 2 MM2S, which is §3.8's prediction measured.

`ir_facts` for W2/npu1: `broadcast_pattern_count = 0` (no W2 channel declares a
`broadcast_shape`, so the detector has nothing to *find* rather than nothing *left* to find),
`pingpong_unroll = 0` (the `cur`/`next` pair sits outside the timestep loop, so
`isPingPongCandidate` has no candidate loop — D-5's second meaning), and
**`lock_init_histogram = {"0": 8, "1": 2, "2": 6}`** — 16 locks over two cores. Cross-checked:
the `PIPELINES["aie"]` figure and `aircc`'s own `air_project/aie.w2.base.npu1.air.mlir` are
identical.

## Experiment **E1**, run on our own module

`air-opt <w2 npu1 module> -pass-pipeline='builtin.module(air-dependency)'`, then a mechanical
walk of the token graph (`test_I_w2_no_put_get_token_edge`, `requires_air_opt`). **Verdict: no
token edge joins a PE's put to its own get.** Every one of the eight halo ops takes the same
dependency the pass gives the whole timestep:

```mlir
%14 = scf.for %arg11 = %c0_12 to %c4_13 step %c2_14 iter_args(%arg12 = %13) -> (!air.async.token) {
  %18 = scf.if %16 -> (!air.async.token) {
    %57 = air.channel.put async [%arg12]  @ToNorth[%56] (%results[1, 0] [1, 16] [16, 1]) ...
    %58 = air.wait_all async [%57]  {id = 6 : i32}
    scf.yield %58 : !air.async.token
  } else { ... }
  ...
  %26 = scf.if %24 -> (!air.async.token) {
    %57 = air.channel.get async [%arg12]  @ToNorth[%56] (%results[9, 0] [1, 16] [16, 1]) ...
```

Phase 0's four ops all take `%arg12`, the loop's iter-arg token — the state **before** the
timestep — and phase 1's all take `%32`, the token of phase 0's compute nest. The `scf.if`
results the puts yield (`%18`, `%22`, `%38`, `%42`) are **dead**: nothing consumes them. The
mechanical form of that: taint everything transitively derived from a halo put's token (through
dependency lists, and through the `scf.if` result its `scf.yield` feeds — `scf.for` yields are
deliberately not followed, that edge being the loop-carried one) and assert no halo get's
dependency list names a tainted value. The taint is 8 values; the gets' dependency lists are
`{%arg12, %32}`; the intersection is empty. MLIR prints SSA names **per region**, so the walk
resolves names lexically and gives every definition a unique id — a flat name table fuses the
`%57` of one `scf.if` with the `%57` of the next and answers a different question.

A *negative* cannot be constructed through the plan: `ChannelSite.depends_on` reaches `air.api`
as `dependency=`, which is type-validated and then unused by `_emit` (**B-P12**), so the puts and
the gets are emitted identically and E1's safety rests entirely on this lowering. That is
precisely what the test measures, and the non-vacuity guard is that each put's token is asserted
to reach at least its `air.wait_all` and its `scf.if` result.

## The probe comparison (`vendor/probes/q/pi2/w2_pi2.py`, regenerated and lowered this session)

| fact | ours | probe `w2_pi2` | why |
|---|---|---|---|
| `aie.flow` (circuit) | **6** | **6** | **byte-identical**, tile for tile and DMA channel for DMA channel |
| `aie.packet_flow` | **0** | **0** | identical: nothing multiplexes at `PI = 2` |
| `aie.core` / tiles | 2 / `tile_0_2`, `tile_1_2` | 2 / same | identical |
| `air.channel @` | 4 — `ToNorth[1]`, `ToSouth[1]`, `UIn[2]`, `UOut[2]` | 4, same names and sizes | the `size=[PI-1]` link convention is the probe's own |
| `scf.if` | **8** | **8** | identical: four guarded sites × two unrolled phases |
| `air.channel.get` | 7 | 7 | identical |
| `air.channel.put` | **8** | **7** | ours drains **every** timestep (G-10); the probe puts the strip once after the loop |
| `scf.for` | **9** | **5** | +2 for the `next` seed copy, +2 for the per-PE plane drain (`air.sequential(0, T)`); the probe drains with no loop at all |
| `arith.addf` | **8** | **6** | the kernel's **five**-point sum against the probe's four-point |
| `arith.mulf` | 2 | 2 | identical — `0.2` against the probe's `0.25` |
| `memref.alloc` | 2 | 2 | identical |
| `aie.lock` inits | `{0: 8, 1: 2, 2: 6}` | `{0: 8, 1: 4, 2: 4}` | 16 locks each; ours allocates more two-slot producers because it drains a plane per timestep |
| `tensors` | 1 — `U [5,18,16]` | 2 — `U [16,16]`, `Uout [16,16]` | RULING 1: one rank-3 parameter, read and written, is the kernel's L3 interface |

The physical realisation is the probe's exactly; every difference is a difference in the
**program** — a five-point stencil against a four-point one, a per-timestep drain against a
single one, a seeded pair against an unseeded one — and each is a fact the probe got wrong for
our purposes rather than a routing risk.

## Readings taken, where the documents disagree

| # | Reading | Why |
|---|---|---|
| 1 | **The update nest is `i1 ∈ [0, HS)` with `dst[i1+1, j]`**, not `i (1..HS+1)` with `dst[i,j]` | The brief gives both the bound `1..HS+1` and the subscripts `dst[i1+1, j]`, which are inconsistent; §6.3 gave the other pair. `_compute_nest`'s rule for a tiled axis is `lo = 0, hi = <tile extent>` and `l1_subscripts` then adds the halo back, so the machinery can only produce the first form. Same rows either way. Erratum in §6.3. |
| 2 | **`t_drain` is `air.sequential`, `pi_bundle` is unrolled** (ruling R-W2-1) | `LOOP_KIND`: `p` is a channel bundle index, `t` is not. M5 §6.3 lines 21-23 drew both as Python loops, the same slip §6.4 had for W3. Erratum in M5 §6.3. |
| 3 | **The `w2.odd` variant is a `plan.json` golden only** | The `T = 5` module differs from `base` by the peeled tail alone, which `test_E_peel_is_plan_driven` asserts on the text; a second module golden would freeze the same fact twice. `06-interfaces.md` §8 carries the `<variant>` erratum. |
| 4 | **W2's npu1 and npu2 AIR texts are byte-identical** | `physical_herd` is `(2,)` on both (2 divides both caps) and `build(target=)` stamps nothing into `str(module)`. Both goldens are kept, as for W3: two identical files record the fact, and a wheel that starts stamping the device shows up as a diff. The two `plan.json` goldens do differ, by `schedule.target`. |
| 5 | **The peeled `STEP`'s update nest keeps `LoopPlan.depth = 1`** although it now sits at herd-body top level | It is built by the same `trip` closure as an in-loop one, and W3's peel has carried the same since P4. `depth` is recorded and never emitted (`06-interfaces.md` §5.5), so this is cosmetic; changing it would churn W3's plan golden for no behaviour. Asserted explicitly in `test_M4_odd_T_peel` so it is a decision rather than an accident. |
| 6 | **The oracle carries the Dirichlet boundary forward** | `02-hld.md` §7.2 calls rows `0`/`H+1` and columns `0`/`W-1` the *read-only* Dirichlet boundary. The one-array kernel text never copies them into plane `t+1`, so a literal transcription would decay them to zero after plane 1 — and then §6.3's "the value written back is the one that was staged in" would be false. The constant-boundary reading is the only one consistent with both, and it is what the seeded pair computes. |

## Verified (command → result)

| # | Command | Result |
|---|---|---|
| 1 | `.venv/bin/python -m pytest` | **557 passed** (was 515), 9 deselected, 6.3 s |
| 2 | the same at `PYTHONHASHSEED=1` and `=12345` | 557 passed each |
| 3 | `.venv/bin/python -m pytest -m slow` | **9 passed** (was 7: `test_W2_aircc_none[npu1]`, `[npu2]` are new) |
| 4 | `pytest --update-goldens` | 9 new goldens; `git diff --stat tests/golden` lists **only** `w2.*` files, so every W1 and W3 golden is byte-identical before and after |
| 5 | the interpreter against the two-loop numpy Jacobi, `default_rng(0)`, both targets, `T = 4` and `T = 5` | **max abs error 0.0** in all four (`tol` is `1e-5`; the plan's left-nested association happens to match the oracle's, so the slack is unused). No deadlock: the round-robin scheduler makes put-before-get progress at every trip |
| 6 | the boundary assertions | columns `0` and `W-1` of rows `1..H` of every drained plane equal plane 0's; rows `0` and `H+1` of planes `1..T` stay zero (the drain covers rows `1..H` only); plane 0 is never written back |
| 7 | `m4.plan(w2_legal.legal(t, T=4, PI=4))` | `DMA-CHANNELS` on both targets, never a plan |

## Open / blockers

| # | Item | Detail |
|---|---|---|
| **M4 §7 rows still waiting** | `test_M6_cascade_chain`, `test_M6_cascade_orientation`, `test_M11_residency_line`'s flip half, `test_E8_cascade_text` — **P6**. W2's own rows are all live now, and `test_M4_unbuilt_protocols_fail_legibly` is down to the flip. |
| **`SWAP-PARITY` reachability** | §10's B-O6 note keeps `SWAP-PARITY` reachable through the `w2_zero_t` (`T = 0`) fixture, which is M3's and does not exist yet. M4 no longer raises it at all (the peel is built), so `test_D3_catalogue_complete` depends on M3 landing that negative. |
| **A rank-2 halo is not built** | `ToWest`/`ToEast` beside `ToNorth`/`ToSouth` — `strip_geometry` raises `NotImplementedError` naming both pairs rather than building half of it. B-O7 (`PI ≥ 3` at all) is unchanged and out of scope. |
| **B-P10**, **B-P12**, **B-P13**, **B-P15**, **B-P20**, **B-O8** | unchanged | — |

**Not verified in this phase**: that the emitted module **computes** Jacobi on hardware — the
interpreter interprets the **plan**, never the emitted IR (D-9), and only a device run closes that
(the honest-limits slide is unchanged); that `air-dependency`'s token graph is what the *lock*
lowering then does (E1's recipe checks the dependency pass, and VF §C.3's lock argument is the
other half — neither is a device run); that `aircc`'s 0.23 s holds on another machine or wheel;
that the seeded pair is what real hardware reads out of an uninitialised `memref.alloc` (the
interpreter zeroes it, the device does not — which is the whole reason the seed exists);
that a halo with `PI ≥ 3`, a rank-2 herd, a halo wider than the owned extent, a non-unit timestep
step or more than one exchanged operand works — each raises by name rather than guessing;
W1-flip's real plan (the cascade builder does not exist yet).
