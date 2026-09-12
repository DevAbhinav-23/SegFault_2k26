# LLD M7 — Tests & CI harness

*Phase 1, 2026-09-12. Owner: **Person C**. Reads on top of [`04-test-plan.md`](04-test-plan.md)
(the levels, the corpus and the ten-point definition of done),
[`01-requirements.md`](01-requirements.md) (the FR ids and their acceptance tests),
[`06-interfaces.md`](06-interfaces.md) §6 (diagnostics), §8 (goldens) and §9 (fixtures), and
[`02-hld.md`](02-hld.md) §8 (build order and stubs). Citation convention as in
`01-requirements.md` §0.*

**M7 provides the harness; it does not write other people's unit tests.** Each module owner
writes their own (`02-hld.md` §2, M7's non-responsibility). What M7 owes them is: the layout,
the marks, the fixture package, the golden helper, the negative-test assertion helper, the
traceability gate, the time budget and CI.

---

## 1. Purpose and FR IDs satisfied

M7 satisfies no FR directly. It is the thing that makes every *other* FR's acceptance clause
executable, and it owns four requirements of its own shape:

| Requirement | What M7 owes it |
|---|---|
| **NFR-1** (determinism) | the `PYTHONHASHSEED` gate: the default run is repeated in a fresh process with a different seed and any golden diff between the two fails the build |
| **NFR-3** (< 3 min CPU) | fixture sizing, the per-test budget, and a session-end wall-clock assertion |
| **NFR-6** (no network) | an autouse fixture that makes `socket.socket` raise, plus the CI network gate |
| **NFR-7** (never crash) | the negative-corpus smoke test: every failure in the corpus is a `SpatialError` |
| **FR-D1, FR-D3** | the four-part assertion helper every negative test uses, and the catalogue-completeness gate |
| **04-test-plan §7** | `test_traceability` — the gate that makes "70 of 70" a fact rather than a claim |

**Non-responsibilities.** M7 does not decide what is legal, what a plan contains, or what the
AIR text should say. It does not own `conftest.py`'s *content* about legality — it owns the file.
It does not generate fixtures (M8 does); it consumes them.

---

## 2. Public entry points / CLI surface

### 2.1 Layout

```
tests/
  conftest.py                 marks, options, fixtures, the budget hook, the network gate
  README.md                   the FR → test inverse index (§7 below), asserted by test_traceability
  helpers/
    golden.py                 assert_golden(), the --update-goldens switch and its guards
    diagnostics.py            assert_diagnostic() — the four-part negative-test assertion
    determinism.py            in_fresh_process() — re-runs a callable under a new PYTHONHASHSEED
  unit/        test_m0_model.py  test_m1_frontend.py  test_m2_schedule.py
               test_m3_legality.py  test_m4_mapping.py  test_m4_selfcheck.py
               test_m5_emit.py   test_m6_tools.py   test_nfr.py
  negative/    test_grammar.py  test_clause.py  test_legality.py  test_mapping.py
               test_catalogue.py
  golden/      <workload>.<variant>.<target>.air.mlir
               <workload>.<variant>.summary.txt
               <workload>.<variant>.plan.json
               <workload>.<variant>.ir_facts.json
  fixtures/    __init__.py                       the fixtures package (importable, M8 fills it)
               w1/ w1_flip/ w2/ w2_odd/ w3/      meta.json inputs.npz expected.npz oracle.npz
               w1_large/                         smoke + IR-facts only (no oracle, no expected)
               plans/w1_plan.py w2_plan.py w3_plan.py     hand-written MappingPlan literals
               mappings/w1_legal.py …                     hand-written LegalMapping literals
               corrupt/                          the six hand-corrupted plans of 04-test-plan §2
               stderr/{clean,exit0_with_error,exit1_with_error}.txt
               modules/unpaired_channel.mlir     the hand-written broken module of §3.2
  integration/ test_golden.py  test_ir_facts.py  test_oracle.py  test_semantics.py
               test_smoke.py   test_device.py
```

`tests/fixtures/` is a **package**, not a directory of data, so that a stub `MappingPlan`
literal is a Python object other people's tests import rather than a JSON file they parse
(HLD §8: "Stubs are hand-written `MappingPlan` / `LegalMapping` literals in the fixtures
package"). The `.npz`/`.json`/`.mlir` files live inside it and are reached with
`importlib.resources.files("tests.fixtures")`, so no test ever computes a path from `__file__`.

### 2.2 Marks and options

```pseudo
# conftest.py — the whole CLI surface of the suite
FUNCTION pytest_addoption(parser):
    parser.addoption("--update-goldens", action="store_true", default=False,
        help="rewrite every golden from the current run. Refused in CI and "
             "refused when the toolchain pin does not match.")

FUNCTION pytest_configure(config):
    FOR name, doc IN [("slow",            "minutes, not seconds; excluded by default"),
                      ("requires_aircc",  "needs aircc + aiecc + Peano"),
                      ("requires_air_opt","needs air-opt"),
                      ("requires_device", "needs XRT and /dev/accel*"),
                      ("fr",              "fr('FR-…', …): the requirement(s) this test accepts")]:
        config.addinivalue_line("markers", f"{name}: {doc}")
```

`pyproject.toml`'s `[tool.pytest.ini_options]` sets
`addopts = "-m 'not slow and not requires_device' --strict-markers -q"`, so the **default
invocation is bare `pytest`** and it runs levels U, N, G, O, I (04-test-plan §1). No plugin is
added: `--strict-markers` and `addinivalue_line` are stock pytest, and NFR-2 caps the runtime
dependencies at four.

---

## 3. Internal design

### 3.1 Golden files and the single `--update-goldens` switch (Q-3)

```pseudo
# tests/helpers/golden.py
GOLDEN_DIR = files("tests") / "golden"

FUNCTION assert_golden(name, actual, *, kind):
    """kind ∈ {"text", "json"}; name is the file name from 06-interfaces §8."""
    path = GOLDEN_DIR / name
    IF UPDATE_GOLDENS:                        # set once, in pytest_configure
        payload = actual IF kind == "text" ELSE canonical_json(actual)
        old     = path.read_text() IF path.exists() ELSE None
        path.write_text(payload)
        IF old != payload: UPDATED.append((name, diff_summary(old, payload)))
        RETURN
    IF NOT path.exists():
        FAIL(f"golden {name} does not exist; run `pytest --update-goldens` and "
             f"say in the commit message why it is new")
    expected = path.read_text()
    IF kind == "text":
        ASSERT actual == expected, unified_diff(expected, actual, name)   # byte for byte
    ELSE:
        ASSERT json.loads(actual_json) == json.loads(expected)            # parsed objects

FUNCTION canonical_json(obj):
    RETURN json.dumps(obj, sort_keys=True, indent=2) + "\n"   # 06-interfaces §8

FUNCTION pytest_terminal_summary(...):
    IF UPDATE_GOLDENS: print(f"--update-goldens rewrote {len(UPDATED)} golden(s):", UPDATED)
```

**The CI guard, exactly.** Three layers, because one is not enough:

```pseudo
FUNCTION pytest_configure(config):
    global UPDATE_GOLDENS
    UPDATE_GOLDENS = config.getoption("--update-goldens")
    IF UPDATE_GOLDENS:
        # Layer 1 — never from CI. A rewritten golden must be a human's commit.
        IF os.environ.get("CI") OR os.environ.get("GITHUB_ACTIONS"):
            RAISE pytest.UsageError(
                "--update-goldens is refused in CI. Regenerate locally, inspect "
                "the diff, and commit it with a reason (06-interfaces.md §8 rule 1).")
        # Layer 2 — never from an unpinned wheel. A golden is only valid for the pin.
        TRY: m6.check_pin()
        CATCH ToolchainError AS e:
            RAISE pytest.UsageError(
                f"--update-goldens is refused: {e.diagnostic.reason}. "
                f"{e.diagnostic.details['mismatched']}")
        # Layer 3 — never with a dirty tree, so the golden diff is the only diff.
        IF git_dirty_under("tests/golden"):
            RAISE pytest.UsageError(
                "tests/golden has uncommitted changes; commit or stash them first "
                "so the regeneration diff is readable.")
```

and one CI step that catches a golden rewritten by any other route:

```pseudo
# CI, after the test step
run: git diff --exit-code -- tests/golden
     || fail("a golden changed during the test run; goldens are inputs, not outputs")
```

**Q-3 — RESOLVED: byte-for-byte for the pinned wheel.** The question asked whether the AIR text
is stable across a `--upgrade` within the pinned version. It is, **by construction of the version
string**: the wheel's version is `0.0.1.2026091204+ff95a9b` and the local part *is* the git
commit (VF §G.3 — "the `mlir_air` wheel is built from the exact commit this document cites"). The
AIR text comes from MLIR's printer for the ops our emitter builds; changing it requires changing
a dialect's assembly format or the printer, which requires a source change, which changes the
commit, which changes the version. So "the text changed but the version did not" is not a state
the release process can produce, and `test_T6_versions` is what turns that into an enforced
property rather than an assumption. **This is reasoning from a verified fact, not a measurement**
— only one build of this version exists, so the counterfactual cannot be measured.

The consequence, from 06-interfaces §8 rule 2: on a pin mismatch every golden test **skips** with
the reason, and only `test_T6_versions` fails. A wrong wheel then produces one red line that
explains itself, instead of eight red diffs that do not.

`ir_facts.json` (D-8) is the reason the golden set survives an upstream pass change at all: the
post-pass facts are five numbers in a JSON object rather than a second 400-line module, so
`air-broadcast-detection` firing once more (R-04, H-8) shows as one changed integer.

### 3.2 The negative-test convention

Every negative test asserts **the code and all four message parts**, never the prose.

```pseudo
# tests/helpers/diagnostics.py
FUNCTION assert_diagnostic(excinfo, *, code, clause=None, mentions=(), details_keys=()):
    d = excinfo.value.diagnostic
    ASSERT d.code == code
    ASSERT d.code IN CATALOGUE, f"{d.code} is not in 06-interfaces.md §6.3"
    ASSERT d.stage == CATALOGUE[d.code].stage
    ASSERT d.reason AND NOT d.reason.endswith(".")        # one line, no trailing period
    ASSERT d.fix                                          # one concrete edit
    IF clause IS NOT None: ASSERT d.clause == clause
    IF CATALOGUE[d.code].stage == "grammar": ASSERT d.location IS NOT None
    ELSE: ASSERT d.clause IS NOT None
    FOR token IN mentions:            # the *numbers*, not the wording
        ASSERT str(token) IN str(d.reason) + str(d.fix) + str(d.details)
    FOR key IN details_keys:          # the FR names what must be in details
        ASSERT key IN d.details
    ASSERT is_json_serialisable(d.details)
    RAISED_CODES.add(d.code)          # feeds test_D3_catalogue_complete
```

The `mentions=` list is how "the message prints the violating dependence vector" becomes an
assertion without freezing the sentence: the test asserts `(1, 0)` appears somewhere in
`reason + fix + details`, so Person A can rewrite the sentence at D6 without breaking 43 tests
(D-7). The three demo rejections additionally have a **golden message** file
(`tests/golden/reject.<code>.txt`, byte for byte) because their exact wording is the thing on
screen — those three, and only those three.

`RAISED_CODES` is a session-scoped set. `test_D3_catalogue_complete` runs last and asserts
`RAISED_CODES == set(CATALOGUE)` in both directions: every catalogue entry was raised, and no
test raised a code the catalogue does not list (FR-D3). `CATALOGUE` itself is parsed out of
`06-interfaces.md` §6.3's table at collection time, so the table is the single source and a code
added to the code but not the document fails the gate.

### 3.3 The fixture package (06-interfaces §9)

```pseudo
# tests/fixtures/__init__.py
FROZEN DATACLASS Fixture:
    name, params: dict, dtype: str, seed: int, tol: float, dir: Path
    def inputs(self)   -> dict[str, ndarray]     # np.load(dir/"inputs.npz")
    def expected(self) -> dict[str, ndarray]     # np.load(dir/"expected.npz")
    def oracle(self)   -> dict[str, ndarray]     # np.load(dir/"oracle.npz")

FUNCTION load(name) -> Fixture:
    meta = json.loads((DIR/name/"meta.json").read_text())
    ASSERT meta["workload"] AND meta["seed"] IS NOT None AND meta["tol"] IS NOT None
    RETURN Fixture(name, meta["params"], meta["dtype"], meta["seed"], meta["tol"], DIR/name)

# conftest.py exposes each as a fixture so a test says `def test_x(w1): ...`
@pytest.fixture(scope="session")  FUNCTION w1():      RETURN fixtures.load("w1")
… w1_flip, w2, w2_odd, w3, w1_large
```

Session scope matters for the budget: `inputs.npz` is decompressed once per session, not once
per test.

**M7 asserts the fixture *contract*; M8 owns the contents.** `test_fixture_contract`, one
parametrised case per fixture directory:

```pseudo
FOR each fixture dir:
    ASSERT meta.json has exactly the keys {workload, params, dtype, seed, tol, generator}
    ASSERT every array named in inputs.npz is a read parameter of the workload's kernel
    ASSERT every array named in expected.npz is a written parameter
    ASSERT expected.npz and oracle.npz have identical key sets, shapes and dtypes
    ASSERT allclose(expected, oracle, atol=meta.tol)    # the committed oracle still agrees
    ASSERT meta.tol == 0.0 OR workload == "W2"          # §3.6 of the M6 LLD
    ASSERT no array is all-zero unless it is a declared accumulator (C, S)
```

The last line catches the commonest fixture bug: a seed that silently produced zeros, against
which every implementation passes.

### 3.4 The ignorability property test (FR-S18)

FR-S18's acceptance is a two-run comparison. M7 strengthens it into a property, because the
property — *no clause, in any order, in any subset, changes the kernel's output* — is what the
column's whole claim rests on (PC §3.0), and it costs nothing to check.

```pseudo
# tests/integration/test_oracle.py
@pytest.mark.fr("FR-S18")
@parametrize("workload", ["W1", "W2", "W3"])
FUNCTION test_ignorability(workload, fixture):
    args_ref = fixture.inputs()                       # fresh arrays each time
    kernel(**deepcopy(args_ref))                      # (i) no schedule constructed at all
    ref = snapshot_bytes(args_ref)

    rng = random.Random(0)                            # fixed seed: NFR-1
    clauses = CLAUSES[workload]                       # the workload's clause call list
    FOR trial IN range(8):
        subset = clauses IF trial == 0 ELSE rng.sample(clauses, rng.randint(1, len(clauses)))
        order  = subset IF trial < 2 ELSE shuffled(subset, rng)
        args   = deepcopy(args_ref)
        s = sp.schedule(kernel, target="npu1")        # explicit target (D-13)
        FOR c IN order: apply(c, s)                   # built, never checked, never emitted
        kernel(**args)
        ASSERT snapshot_bytes(args) == ref, f"clause subset {order} changed the output"
    ASSERT "air" NOT IN sys.modules                   # shares FR-S20's property
```

`snapshot_bytes` is `{k: v.tobytes() for k, v in arrays.items()}` — bit-identical, as FR-S18
requires, not `allclose`. Some clause orders are *illegal* (e.g. `place` before `grid`); that is
fine and is the point — a `ClauseError` raised at clause time is caught and counted as "no
mutation happened", because the assertion is on the kernel's output, not on the schedule
succeeding. Eight trials with a fixed seed keeps this well inside the per-test budget.

### 3.5 The determinism gate (NFR-1)

```pseudo
# tests/helpers/determinism.py
FUNCTION in_fresh_process(dotted_callable, *args) -> bytes:
    """Run a callable in a subprocess with a different PYTHONHASHSEED, return its stdout."""
    env = {**os.environ, "PYTHONHASHSEED": str(int(os.environ.get("PYTHONHASHSEED", 0)) + 7)}
    RETURN subprocess.run([sys.executable, "-c",
        f"import importlib,sys; f=…; sys.stdout.buffer.write(f(*ARGS))"],
        env=env, capture_output=True, check=True).stdout
```

Used by `test_E10_byte_identical` (B), `test_M12_plan_stable` (B) and the CI gate. This is the
one place in the suite that spawns a Python subprocess; it is three tests, each well under a
second, and it keeps the alternative — running the whole suite twice inside pytest — out of the
3-minute budget. CI *additionally* runs the whole default suite twice with different seeds
(04-test-plan §6), which is the belt to this test's braces.

### 3.6 The time budget (NFR-3)

The budget is **180 s wall clock for the default run**. It is spent, not just asserted:

| Level | Tests (est.) | Budget | Per-test |
|---|---|---|---|
| U unit | ~90 | 30 s | 0.33 s |
| N negative | ~82 (43 codes + ≥20 clause + ≥15 grammar) | 20 s | 0.25 s |
| G golden | ~12 (4 variants × 2 targets, plus summary/plan) | 40 s | 3.3 s — each traces through `air.api` and calls `build()` |
| O oracle | ~9 (3 workloads × {oracle, ignorability, semantics}) | 20 s | 2.2 s |
| I IR inspection | 4–5 | 40 s | 8 s — each forks `air-opt` |
| slack | — | 30 s | — |

Enforced by two stdlib-only hooks, no `pytest-timeout`:

```pseudo
FUNCTION pytest_runtest_makereport(item, call):     # hookwrapper
    report = yield
    IF call.when == "call":
        DURATIONS[item.nodeid] = call.duration
        # A single unmarked test over 3 s is a design error, not a slow machine.
        IF call.duration > 3.0 AND NOT item.get_closest_marker("slow") \
           AND NOT item.get_closest_marker("requires_air_opt"):
            report.outcome = "failed"
            report.longrepr = (f"{item.nodeid} took {call.duration:.1f}s. Mark it "
                               f"`slow` or shrink its fixture (NFR-3).")

FUNCTION pytest_sessionfinish(session, exitstatus):
    total = sum(DURATIONS.values())
    write_json("tests/.durations.json", DURATIONS)   # gitignored; CI uploads it
    IF default_run(session) AND total > 180:
        session.exitstatus = 1
        print(f"NFR-3: default run took {total:.0f}s of a 180s budget. "
              f"Ten slowest:\n{top_n(DURATIONS, 10)}")
```

Printing the ten slowest is what turns "the suite got slow" into "this test got slow". The
`requires_air_opt` exemption exists because an `air-opt` fork is ~1 s of process start-up we
cannot shrink; those five tests have their own 40 s line in the table.

### 3.7 The network gate (NFR-6)

```pseudo
@pytest.fixture(autouse=True, scope="session")
FUNCTION _no_network():
    real = socket.socket
    CLASS Blocked(real):
        def connect(self, *a, **k):
            RAISE RuntimeError("NFR-6: the test suite must not reach the network. "
                               "The toolchain is installed once, at D0.")
    socket.socket = Blocked
    YIELD
    socket.socket = real
```

A unix-domain socket is left alone (`AF_UNIX` never "reaches the network" and XRT may use one),
so the block is on `connect` for `AF_INET`/`AF_INET6` only. CI additionally disables networking
after the install step, which is the real gate; this fixture is what makes a laptop run behave
the same way.

### 3.8 The stub strategy (HLD §8)

The rule from HLD §8 is *every module is testable against a stub from D1*. M7 owns where the
stubs live and when they die.

| Stub | Owner | Lives in | Stands in for | Deleted / reclassified |
|---|---|---|---|---|
| `m3.check` returns a `LegalMapping` for everything | **A** | `spatial/m3_legality.py`, guarded by nothing — it is the initial implementation | M3's checks | **replaced** at D2 as each of L1/L2/L3/L8/L9/L10 lands. Not a separate file, so there is nothing to forget to delete |
| `w1_plan.py`, `w2_plan.py`, `w3_plan.py` — hand-written `MappingPlan` literals | **B** | `tests/fixtures/plans/` | M4's output | **never deleted.** At D2 they stop being M4's stand-in and become M5's *input fixtures* — they are what makes `test_emitter_makes_no_decisions` meaningful (M5 must produce the same text from a literal plan as from M4's plan). A D6 test asserts `m4.plan(w1_legal) == plans.w1_plan` |
| `w1_legal.py`, … — hand-written `LegalMapping` literals | **A** | `tests/fixtures/mappings/` | M3's output | same: reclassified at D2 into M4's unit-test inputs |
| `corrupt/*.py` — six hand-corrupted plans | **B** writes the check, **A** writes the corpus | `tests/fixtures/corrupt/` | nothing — they are permanent negatives | never |
| `stderr/*.txt` — three recorded tool outputs | **C** | `tests/fixtures/stderr/` | a live `air-opt` run | never; they are what keeps `test_stderr_parser` in the default (fork-free) suite |
| `modules/unpaired_channel.mlir` | **C** | `tests/fixtures/modules/` | nothing | never; it is the characterisation test of 04-test-plan §3.2 |

**The rule that makes this work:** a stub is a *real implementation that is wrong yet*, living at
its final import path, never a `if STUB:` branch and never a second module. Then "delete the
stub" is "finish the function", and there is no dead branch to ship.

### 3.9 CI outline

One workflow, three jobs. Written here as specification; the file is created at D0.

```pseudo
# .github/workflows/ci.yml  (ubuntu-latest, python 3.11–3.14 matrix on 3.12 only for speed)
JOB default:
  1  checkout
  2  restore vendor/wheels from the runner cache, keyed on vendor/wheels/SHA256SUMS
  3  sha256sum -c vendor/wheels/SHA256SUMS
  4  python -m venv .venv && .venv/bin/pip install --no-index \
         --find-links vendor/wheels 'mlir_air[aie]' && pip install -e .
  5  pytest tests/unit/test_m6_tools.py::test_T6_versions          # fail fast on the pin
  6  NETWORK OFF                                                    # NFR-6
  7  PYTHONHASHSEED=0 pytest                                        # the default marks
  8  PYTHONHASHSEED=7 pytest -k "golden or plan or summary"         # NFR-1 determinism gate
  9  git diff --exit-code -- tests/golden                           # goldens are inputs
  10 fail if step 7's reported wall clock > 180 s                   # NFR-3
  ON FAILURE: upload tests/.durations.json, every emitted *.air.mlir, every captured stderr

JOB slow (nightly schedule only, never on a PR):
  steps 1–6, then  pytest -m "slow"                                 # levels S
  ON FAILURE: upload the aircc tmpdirs

JOB device (self-hosted, label `npu`, allowed to fail):
  steps 1–6, then  pytest -m "requires_device"
```

Steps 2–4 are the R-13 mitigation in operational form: **CI never sees the `-f` release URLs**,
so a pruned asset (`.github/workflows/pruneAIRReleaseAssets.yml:95-97`) cannot break the build.
Step 5 runs before step 6 so that a pin failure is diagnosed while the network could still have
fixed it. Step 9 is the third layer of the `--update-goldens` guard (§3.1).

---

## 4. Invariants

| # | Invariant | Enforced by |
|---|---|---|
| **I-1** | Every FR id in `01-requirements.md` §3 appears in at least one `@pytest.mark.fr(...)`, and every id inside a marker is a real FR | `test_traceability` |
| **I-2** | Every code in `06-interfaces.md` §6.3 is raised by at least one test, and no test raises a code the table lacks | `test_D3_catalogue_complete` |
| **I-3** | Every negative test asserts `code` + four non-empty message parts + the FR's named `details` keys | `assert_diagnostic` is the only way to assert a rejection; an AST lint over `tests/negative/` forbids bare `pytest.raises(...)` without it |
| **I-4** | Goldens are **inputs**. A test run never writes one unless `--update-goldens` was passed and all three guards passed | §3.1, CI step 9 |
| **I-5** | No test passes `target="auto"` | D-13; AST lint |
| **I-6** | No test reaches the network | §3.7 |
| **I-7** | No unmarked test exceeds 3 s; the default run stays under 180 s | §3.6 |
| **I-8** | A skipped test always prints why, in one sentence a non-expert can read | `pytest_runtest_setup`'s skip reasons; `test_skips_are_explained` asserts every `skip(` in `tests/` has a reason ≥ 20 characters |
| **I-9** | `expected.npz` is never produced by our own oracle | §3.3's contract test compares them but M8's generator computes them independently; `test_fixture_contract` asserts `meta["generator"]` names the numpy expression |
| **I-10** | The suite imports `air` only in tests marked `requires_air_opt`/`requires_aircc`/`requires_device`, or through M5 | `test_no_air_import_until_build` (A) plus a session check that `air` is absent from `sys.modules` after `tests/unit` and `tests/negative` |

---

## 5. Error paths

M7 raises no `SpatialError`; it raises pytest's own. Three cases are worth naming because they
are the ones that waste an afternoon:

| Situation | What the harness does | Why |
|---|---|---|
| the pin does not match | `test_T6_versions` **fails**; every golden and IR-fact test **skips** with `"toolchain is 0.0.1.x+abcdef, goldens are valid only for 0.0.1.2026091204+ff95a9b"` | one red line that explains itself, not eight red diffs that do not (06-interfaces §8 rule 2) |
| `--update-goldens` in CI, or on an unpinned wheel, or with a dirty `tests/golden` | `pytest.UsageError` before collection | a rewritten golden must be a human's reviewed commit (§3.1) |
| a golden text differs | the failure prints a **unified diff with 3 lines of context**, never the two full modules | a 400-line diff is unreadable; and `ir_facts.json` exists precisely so the *interesting* changes are integers (D-8) |
| a tool is missing | skip, with the install command | `requires_aircc` / `requires_air_opt` (§2.2) |
| no device | skip, with `/dev/accel*` named | R-09; the skip list **is** the honest-limits slide (04-test-plan §8 item 10) |

---

## 6. Worked examples

### 6.1 A negative test, end to end

```pseudo
@pytest.mark.fr("FR-L2")
FUNCTION test_L2_causality(w3):
    s = build_w3_schedule(target="npu1")
    s.skew(time=(ax.i,))                           # the DROPPED j0 term — the headline rejection
                                                   # (reversing the terms is a no-op: skew is a sum)
    WITH pytest.raises(LegalityError) AS e:
        s.check()
    assert_diagnostic(e, code="L2-CAUSALITY", clause="skew(time=(ax.i,))",
                      mentions=[(0, 1), (0, -7)],  # the violating vector and Sσ·d
                      details_keys=["dependence", "representative", "s_sigma", "product"])
    assert_golden("reject.L2-CAUSALITY.txt", str(e.value), kind="text")
```

The `assert_golden` line appears in exactly three negative tests — the three the pitch shows
(04-test-plan §5). Everything else keys on the code and the numbers, so Person A can improve
prose at D6 without a 43-file diff.

### 6.2 A golden test, end to end

```pseudo
@pytest.mark.fr("FR-E10", "FR-K1")
@parametrize("target", ["npu1", "npu2"])
FUNCTION test_W1_golden(target):
    s = w1_schedule(target=target)
    assert_golden(f"w1.base.{target}.air.mlir", s.mlir(),        kind="text")
    assert_golden(f"w1.base.{target}.summary.txt", s.summary(),  kind="text")
    assert_golden(f"w1.base.{target}.plan.json", s.plan().json(), kind="json")
```

Eight module goldens (4 variants × 2 targets) plus their summaries and plans, as 04-test-plan
§3.1 specifies. The plan and summary are stored per target too, because `physical_herd` and
`repeats` differ between `npu1` and `npu2` (`python/air/api/_trace.py:88-91`).

### 6.3 The traceability gate

```pseudo
FUNCTION test_traceability():
    declared = parse_fr_ids("design/01-requirements.md")       # regex ^\*\*FR-([A-Z]+\d+)
    covered  = defaultdict(list)
    FOR item IN collect_all_items(including_slow_and_device=True):
        FOR mark IN item.iter_markers("fr"):
            FOR fr IN mark.args:
                ASSERT fr IN declared, f"{item.nodeid} claims {fr}, which is not an FR"
                covered[fr].append(item.nodeid)
    missing = sorted(declared - covered.keys())
    ASSERT NOT missing, f"{len(missing)} FR(s) with no test: {missing}"
    write("tests/README.md", render_inverse_index(covered))     # only under --update-goldens
```

Note the correction to 04-test-plan §7, which says `test_traceability` asserts every FR "appears
in exactly one test's marker". **It must be "at least one".** Three tests legitimately accept
two FRs each — `test_no_buffer_resources_arg` (FR-S13 and FR-E4),
`test_stream_overrides_derivation` (FR-S11 and FR-M3), `test_M11_summary_golden` (FR-M11 and the
delegating FR-D2) — and forcing them apart would duplicate an assertion to satisfy a counter.
The property that matters is **coverage**, plus the reverse direction (no marker names a
non-existent FR), and both are asserted above. **Flagged for the architect** as a one-line edit
to `04-test-plan.md` §7.

---

## 7. Tests table — the full FR → test traceability index

`04-test-plan.md` §7 gives this by *group*; the gate needs it by *id*. Test names are the ones
`01-requirements.md` names in each FR's **Acceptance** clause. Owner is who writes the test.

### 7.1 Surface — FR-S1…S20 (A)

| FR | Test(s) | Level | Owner |
|---|---|---|---|
| FR-S1 | `test_kernel_call_is_oracle` | O | A |
| FR-S2 | `test_annotations_inert` | U | A |
| FR-S3 | `test_grammar_accepts`, `test_grammar_rejects[…]` (≥ 15 cases) | U, N | A |
| FR-S4 | `test_grammar_rejects[…]` (asserts all three parts) | N | A |
| FR-S5 | `test_schedule_is_pure_data` | U | A |
| FR-S6 | `test_grid_rank3_rejected` | N | A |
| FR-S7 | `test_tile_must_divide`, `test_tile_handles` | N, U | A |
| FR-S8 | `test_place_rank_matches_grid`, `test_place_duplicate_axis_rejected` | N | A |
| FR-S9 | `test_reduce_op_domain`, `test_reduce_axis_must_accumulate` | N | A |
| FR-S10 | `test_stationary_unknown_operand` | N | A |
| FR-S11 | `test_stream_overrides_derivation`, `test_stream_along_must_be_placed` | U, N | A |
| FR-S12 | `test_reside_l3_alloc_rejected`, `test_reside_maps_to_scope` | N, U | A |
| FR-S13 | `test_double_buffer_ir_shape`, `test_no_buffer_resources_arg` | I, U | C, B |
| FR-S14 | `test_sequential_emits_scf_for`, `test_pipeline_is_hint` | G | B |
| FR-S15 | `test_window_halo_derived` | U | A |
| FR-S16 | `test_exchange_protocol_shape` → the assertions of `test_M4_halo_protocol` | U | B |
| FR-S17 | `test_skew_sets_sigma` | U | A |
| FR-S18 | `test_ignorability[W1,W2,W3]` (the property form, §3.4) | O | C |
| FR-S19 | `test_clause_errors[…]` (≥ 20 cases) | N | A |
| FR-S20 | `test_no_air_import_until_build` | U | A |

### 7.2 Legality — FR-L1…L14 (A)

| FR | Test | Level | Owner |
|---|---|---|---|
| FR-L1 | `test_L1_conflict` | N | A |
| FR-L2 | `test_L2_causality` (+ golden message) | N | A |
| FR-L3 | `test_L3_stationarity` (+ golden message) | N | A |
| FR-L4 | `test_L4_split` | U | A |
| FR-L5 | `test_L5_ac_required` | N | A |
| FR-L6 | `test_L6_cascade_rank`, `test_L6_cascade_no_broadcast` | N | A |
| FR-L7 | `test_L7_halo_too_small` | N | A |
| FR-L8 | `test_L8_consistency[a,b,c,d]` | N | A |
| FR-L9 | `test_L9_capacity` (+ golden message), `test_l1_capacity_doubles` | N, U | A |
| FR-L10 | `test_L10_physical`, `test_physical_herd_table` | U | A |
| FR-L11 | `test_L11_rank` | N | A |
| FR-L12 | `test_legality_error_schema` | N | A |
| FR-L13 | `test_no_emission_on_illegal` | U | A |
| FR-L14 | `test_L14_swap_parity` | N | A |

### 7.3 Mapping — FR-M1…M12 (B)

| FR | Test | Level | Owner |
|---|---|---|---|
| FR-M1 | `test_M1_trichotomy` | U | B |
| FR-M2 | `test_M2_broadcast_shape` | U, G | B |
| FR-M3 | `test_stream_overrides_derivation` *(shared with FR-S11)* | U | B |
| FR-M4 | `test_M4_halo_protocol`, `test_M4_balanced` | U | B |
| FR-M5 | `test_M5_wavefront_balance`, `test_M5_uses_branch` | U, G | B |
| FR-M6 | `test_M6_cascade_chain` | U, G | B |
| FR-M7 | `test_M7_buffer_plan` | U | B |
| FR-M8 | `test_M8_channel_plan_complete` | U | B |
| FR-M9 | `test_M9_selfcheck_accepts`, `test_M9_selfcheck_rejects[drop_get, double_put, guarded_put, broadcast_underconsumed]` | U, N | B, A |
| FR-M10 | `test_M10_cycle_rejected`, `test_M10_w2_acyclic` | N, U | B, A |
| FR-M11 | `test_M11_summary_golden`, `test_M11_residency_line` | G, U | B |
| FR-M12 | `test_M12_plan_stable` | U | B |

### 7.4 Emission — FR-E1…E10 (B)

| FR | Test | Level | Owner |
|---|---|---|---|
| FR-E1 | `test_E1_hierarchy` | G | B |
| FR-E2 | `test_E2_native_body` | G | B |
| FR-E3 | `test_E3_alloc_is_direct_child`, `test_E3_pingpong_fires` | G, I | B, C |
| FR-E4 | `test_no_buffer_resources_arg` *(shared with FR-S13)* | U | B |
| FR-E5 | `test_E5_broadcast_emitted`, `test_E5_detector_count` | G, I | B, C |
| FR-E6 | `test_E6_memory_spaces` | G | B |
| FR-E7 | `test_E7_text_roundtrip` | I | C |
| FR-E8 | `test_E8_cascade_text` | G | B |
| FR-E9 | `test_E9_verify_surfaced` | U | B |
| FR-E10 | `test_E10_byte_identical` | U | B |

### 7.5 Toolchain — FR-T1…T6 (C)

As the M6 LLD §7 table: `test_T1_targets`, `test_T2_aircc_none`, `test_T3_xclbin_message`,
`test_T4_device_diff`, `test_T5_error_without_exit_code`, `test_T6_versions`.

### 7.6 Diagnostics — FR-D1…D3 (A)

| FR | Test | Level | Owner |
|---|---|---|---|
| FR-D1 | `test_D1_schema` (parametrised over every negative test's diagnostic) | N | A |
| FR-D2 | `test_M11_summary_golden` — FR-D2 delegates to FR-M11, so the marker carries both ids | G | B |
| FR-D3 | `test_D3_catalogue_complete` | N | A |

### 7.7 Kernels — FR-K1…K5 (C)

| FR | Test | Level | Owner |
|---|---|---|---|
| FR-K1 | `test_W1_end_to_end` | G, O, S | C |
| FR-K2 | `test_W1_flip` | G, S | C |
| FR-K3 | `test_W2_end_to_end`, `test_W2_oracle_is_timestep_outermost` | G, O, S | C |
| FR-K4 | `test_W3_end_to_end` | G, O, S | C |
| **FR-K5** | **`test_K5_scope_documented` — NEW, see below** | U | C |

### 7.8 Non-functional — NFR-1…NFR-7

| NFR | Test | Level | Owner |
|---|---|---|---|
| NFR-1 | `test_E10_byte_identical`, `test_M12_plan_stable`, + CI step 8 | U, CI | B, C |
| NFR-2 | `test_NFR2_deps` | U | C |
| NFR-3 | **`test_NFR3_budget`** (the `pytest_sessionfinish` assertion of §3.6, surfaced as a test id) + CI step 10 | U, CI | C |
| NFR-4 | `test_D1_schema` + **`test_NFR4_no_bare_raise`** (AST lint: no `raise Exception/AssertionError/RuntimeError` on a user-facing path) | N, U | A, C |
| NFR-5 | `test_NFR5_docstrings` | U | C |
| NFR-6 | **`test_NFR6_no_network`** (asserts the autouse gate is installed and that `connect` raises) + CI step 6 | U, CI | C |
| NFR-7 | **`test_NFR7_all_errors_are_spatial`** (runs the whole negative corpus, asserts every failure is a `SpatialError`) | N | C |

### 7.9 Traceability result

**70 functional requirements declared** (S: 20, L: 14, M: 12, E: 10, T: 6, D: 3, K: 5 —
`01-requirements.md` §8). **70 of 70 have a named acceptance test in `01-requirements.md`**, once `test_K5_scope_documented` lands (below); before it, 69.

**The one uncovered FR is FR-K5.** Its Acceptance clause says, verbatim: *"`test_W4_absent` is
not a test; the acceptance is a line in the honest-limits slide and in `00-README.md`'s status
table."* That is a real requirement with no executable gate, so `test_traceability` would fail on
it. Two options were considered and one is proposed:

* *Exempt FR-K5 from the gate* — rejected: an exemption list is the thing that grows.
* **Add a test (proposed).** `test_K5_scope_documented` asserts three things, all cheap and all
  stdlib: (a) `demo/honest_limits.md` contains the substring `W4` **and** the reason
  (`p ↦ p ⊕ 2^s`), (b) `design/00-README.md` §3's status table has a row reading
  `| **W4 FFT** | **out of scope** |`, (c) the `kernels/` package exports no name matching
  `w4|fft`. That turns a documentation obligation into a regression gate and makes the count
  **70 of 70**.

Four NFR acceptances were CI-only prose (NFR-3, NFR-4's lint, NFR-6, NFR-7) and are given pytest
ids above, so the same gate covers them.

---

## 8. Dependencies

### 8.1 On other modules

| From | What M7 needs | When |
|---|---|---|
| M0 (A) | the frozen dataclasses and `Diagnostic` | **first two hours of D1** — the harness cannot type its helpers without it |
| M5 (B) | `s.mlir()` producing text for the stub W1 plan | D1 end, for the first golden |
| M6 (C, self) | `check_pin`, `ir_facts`, `artifact`, `has_device` | D2 |
| M8 (C, self) | the fixture directories | D1 |
| `06-interfaces.md` §6.3 | the code catalogue, **parsed from the table at collection time** | D0 |
| `01-requirements.md` §3 | the FR ids, **parsed at collection time** | D0 |

Parsing both documents rather than duplicating them is what keeps the design set and the suite
from drifting (R-15). It costs ~30 lines of regex and buys the property that a code added to the
code but not the document fails CI.

### 8.2 Third-party

`pytest` (dev-only, not in NFR-2's runtime list) and `numpy`. **No pytest plugins**: `--strict-markers`,
`addinivalue_line`, `pytest_addoption`, `pytest_runtest_setup`, `pytest_runtest_makereport`,
`pytest_sessionfinish` and `pytest_terminal_summary` are all stock hooks, and the timing,
network and traceability gates are built on them rather than on `pytest-timeout`,
`pytest-socket` and `pytest-xdist`.

**`-p no:randomly` is set** in `addopts` if `pytest-randomly` happens to be installed in
someone's environment: test order must be deterministic for the `PYTHONHASHSEED` gate to mean
anything.

### 8.3 Files in the clone

| Concern | `path:line` |
|---|---|
| release assets are pruned — the reason CI installs from the cache | `.github/workflows/pruneAIRReleaseAssets.yml:95-97` |
| a plain `MlirOptMain` driver; CI runs `air-opt` on stock runners with no XRT | `tools/air-opt/air-opt.cpp:50-51`; `.github/workflows/buildAndTestNoRuntime.yml:29`, `:109-111` |
| the physical herd caps, which is why plan and summary goldens are stored per target | `python/air/api/_trace.py:88-91` |
| `air.api` builds under `Location.unknown()`, so goldens carry no absolute path | `python/air/api/_compile.py:127` |
| the exit-0-with-error behaviour the recorded stderr samples capture | `mlir/lib/Util/Dependency.cpp:2063-2066` |

---

## 9. Implementation order, effort, definition of done

Effort figures are **estimates**, consistent with `05-work-breakdown.md` §7's 1.5 pd for M7.

| Step | Day | Est. | Output | Done when |
|---|---|---|---|---|
| 1. layout, `pyproject` marks, `conftest` options, the budget and network hooks | D1 am | 0.3 pd | `tests/conftest.py` | bare `pytest` collects zero tests and exits green in < 2 s |
| 2. `helpers/golden.py` with all three `--update-goldens` guards | D1 am | 0.2 pd | — | `--update-goldens` raises `UsageError` under `CI=1` |
| 3. `helpers/diagnostics.py` + the catalogue parser | D1 pm | 0.2 pd | — | `assert_diagnostic` rejects a diagnostic with an empty `fix` |
| 4. the fixtures package + `test_fixture_contract` | D1 pm | 0.2 pd | `tests/fixtures/__init__.py` | the W1 fixture loads and its contract test passes |
| 5. `test_traceability` + the FR parser + `tests/README.md` renderer | D2 | 0.2 pd | — | it reports the FR-K5 gap on the first run, and 70/70 after step 6 |
| 6. `test_K5_scope_documented`, `test_NFR{2,3,4,5,6,7}` | D3 | 0.2 pd | — | 70/70 and 7/7 |
| 7. `helpers/determinism.py` + the golden pipeline test | D2–D3 | 0.1 pd | `tests/integration/test_golden.py` | W1's eight goldens exist and compare |
| 8. the ignorability property test | D3 | 0.05 pd | — | 8 trials × 3 workloads green in < 2 s |
| 9. CI workflow | D3 | 0.05 pd | `.github/workflows/ci.yml` | a green run from the wheel cache with the network off |
| 10. budget tuning, the ten-slowest report, `test_D3_catalogue_complete` | D6 | 0.2 pd | — | default run < 180 s with 43/43 codes covered |

**Definition of "thoroughly tested"** — 04-test-plan §8's ten points, each with the id that
asserts it, so the D7 freeze reads a checklist and not a mood:

| # | 04-test-plan §8 item | Asserted by |
|---|---|---|
| 1 | every FR has ≥ 1 test | `test_traceability` (70/70 after `test_K5_scope_documented`) |
| 2 | every code raised; no code outside the catalogue | `test_D3_catalogue_complete` |
| 3 | four variants byte-identical across two `PYTHONHASHSEED`s and matching goldens | `test_E10_byte_identical` + CI step 8 |
| 4 | four variants × both targets pass `aircc --output-format=none` with no `error:` line | `test_T2_aircc_none` |
| 5 | the three oracle diffs pass exactly | `test_W{1,2,3}_end_to_end` with `tol = 0.0` |
| 6 | `unroll = 2` on W1; broadcast count on all four | `test_I_pingpong_labels`, `test_I_broadcast_count` |
| 7 | the self-check negatives fire | `test_M9_selfcheck_rejects[…]`, `test_M10_cycle_rejected` |
| 8 | default run under 3 minutes | `test_NFR3_budget` + CI step 10 |
| 9 | device tests pass **or** skip with a recorded reason | `pytest_runtest_setup`'s skip text; `test_skips_are_explained` |
| 10 | every skip prints why, and D7 reads the list aloud | `pytest -ra` in CI; the D7 checklist in the M8 LLD §6.4 |

---

## 10. Open questions owned, and their resolutions

| # | Question | Verdict |
|---|---|---|
| **Q-3** | are goldens byte-for-byte, or FileCheck-style patterns? | **RESOLVED — byte-for-byte for the pinned wheel (§3.1).** The version string embeds the commit (`+ff95a9b`, VF §G.3), so a text change without a version change is not a state the release process can produce. `--update-goldens` is guarded three ways (CI, pin, dirty tree) and CI adds `git diff --exit-code -- tests/golden`. Reasoned from a verified fact; not measurable with one build in existence |
| **(new) Q-C5** *(closed)* | does `test_traceability` require "exactly one" test per FR, as 04-test-plan §7 said? | **RESOLVED — no, "at least one" (§6.3).** Three tests legitimately carry two FRs. `04-test-plan.md` §7 and `01-requirements.md` §8 now both say "at least one, and no marker names an FR that does not exist" (REVIEW-round1 EDIT-53, EDIT-60) |
| **(new) Q-C6** *(closed)* | FR-K5 has no acceptance test | **RESOLVED — `test_K5_scope_documented` (§7.7).** It asserts the honest-limits slide, `00-README.md`'s status row and the absence of a W4 kernel. `01-requirements.md` §8 now names it and records **70 of 70** (EDIT-60) |
| **(new) Q-C7** | four NFR acceptances were CI-only prose | **RESOLVED — §7.8** gives each a pytest id (`test_NFR3_budget`, `test_NFR4_no_bare_raise`, `test_NFR6_no_network`, `test_NFR7_all_errors_are_spatial`) so one gate covers FRs and NFRs alike |
| **(open) Q-C8** | is 180 s achievable once all eight golden modules exist, given each G-level test traces through `air.api` and calls `build()`? | **Owner C, due D3.** The measured input is that `build()` runs no pass pipeline (VF §D.9) so it is tracing cost only, and VF §E.5's 52-line module built instantly. If the eight goldens exceed the 40 s line, the fallback is to build each module **once** in a session-scoped fixture and have the three assertions (module, summary, plan) read it, which cuts the eight builds to four. Recorded so the fix is chosen in advance rather than by marking tests `slow` |
