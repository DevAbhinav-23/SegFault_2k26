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
