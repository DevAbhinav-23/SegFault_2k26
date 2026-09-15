"""Level N — the grammar corpus (M1). Owner: Person A. Spec: `03-lld-M1-frontend.md` §7.

Every case goes through the **public surface**, `sp.kernel(...)`, on a kernel whose source
`inspect.getsource` can read: a nested `def` in a file is real source, which is why the
kernels are written inline rather than built with `exec` — except the one case that is
*about* unreadable source (`corpus_grammar_unsupported_stmt__no_source`).

The seven `raises_*` functions are the seven `GRAMMAR-*` codes of `06-interfaces.md` §6.3, one
each. The `corpus_*` functions are the rest of `03-lld-M1-frontend.md` §7's 28-row table —
the rows `tests/unit/test_a_smoke.py` does not carry (HANDOFF, Person A item 7).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import spatial as sp
from spatial.model import GrammarError
from tests.negative import _corpus

M = N = 8
"""A tiny square shape: these kernels are parsed, never run, so size buys nothing."""


# --------------------------------------------------------------------------------------------
# The seven codes, one canonical case each
# --------------------------------------------------------------------------------------------


def raises_grammar_bad_annotation():
    """A parameter annotated `int` rather than `sp.<dtype>[...]` (§7 row 25)."""
    def bad(A: int, C: sp.f32[M, N]):
        for i in range(M):
            C[i, 0] = 1.0

    sp.kernel(bad)


def raises_grammar_unsupported_stmt():
    """An `if` **statement** is control flow, which the subset does not have (§7 row 1)."""
    def bad(A: sp.f32[M, N], C: sp.f32[M, N]):
        for i in range(M):
            if i > 0:
                C[i, 0] = A[i, 0]

    sp.kernel(bad)


def raises_grammar_unsupported_expr():
    """`**` is not one of `+ - * /`, unary `-`, `max`/`min` or a select (§7 row 13)."""
    def bad(A: sp.f32[M, N], C: sp.f32[M, N]):
        for i in range(M):
            C[i, 0] = A[i, 0] ** 2

    sp.kernel(bad)


def raises_grammar_unsupported_call():
    """Only `max` and `min` are callable inside a kernel body (§7 row 14)."""
    def bad(A: sp.f32[M, N], C: sp.f32[M, N]):
        for i in range(M):
            C[i, 0] = abs(A[i, 0])

    sp.kernel(bad)


def raises_grammar_nonaffine_subscript():
    """`A[i*j, 0]` is a product of two loop variables (§7 row 16)."""
    def bad(A: sp.f32[M, N], C: sp.f32[M, N]):
        for i in range(M):
            for j in range(N):
                C[i, j] = A[i * j, 0]

    sp.kernel(bad)


def raises_grammar_bad_range():
    """A `for` over a list literal, not `range(...)` (§7 row 20)."""
    def bad(A: sp.f32[M, N], C: sp.f32[M, N]):
        for i in [1, 2]:
            C[i, 0] = A[i, 0]

    sp.kernel(bad)


def raises_grammar_nonuniform_dep():
    """`S[i,j] = S[j,i] + 1` reads a written array at a transposed access (§7 row 28).

    The dependence distance is not constant, so no uniform `Dependence` vector exists and the
    `(σ,π)` machinery has nothing to check causality against. This diagnostic used to be
    unreachable: it was built with `location=None`, which M0 invariant I71 rejects, so it
    raised `ValueError` out of `Diagnostic` instead (fixed 2026-09-15).
    """
    def bad(S: sp.i32[M, M]):
        for i in range(M):
            for j in range(M):
                S[i, j] = S[j, i] + 1

    sp.kernel(bad)


# --------------------------------------------------------------------------------------------
# The rest of §7's table
# --------------------------------------------------------------------------------------------


def corpus_grammar_unsupported_stmt__while():
    """§7 row 2."""
    def bad(A: sp.f32[M, N], C: sp.f32[M, N]):
        i = 0
        while i < M:
            C[i, 0] = A[i, 0]

    sp.kernel(bad)


def corpus_grammar_unsupported_stmt__break():
    """§7 row 3."""
    def bad(A: sp.f32[M, N], C: sp.f32[M, N]):
        for i in range(M):
            C[i, 0] = A[i, 0]
            break

    sp.kernel(bad)


def corpus_grammar_unsupported_stmt__continue():
    """§7 row 4."""
    def bad(A: sp.f32[M, N], C: sp.f32[M, N]):
        for i in range(M):
            continue

    sp.kernel(bad)


def corpus_grammar_unsupported_stmt__return():
    """§7 row 5."""
    def bad(A: sp.f32[M, N], C: sp.f32[M, N]):
        for i in range(M):
            C[i, 0] = A[i, 0]
        return C

    sp.kernel(bad)


def corpus_grammar_unsupported_stmt__tuple_assign():
    """§7 row 6."""
    def bad(A: sp.f32[M, N], C: sp.f32[M, N]):
        for i in range(M):
            a, b = 1, 2
            C[i, 0] = A[i, 0]

    sp.kernel(bad)


def corpus_grammar_unsupported_stmt__global():
    """§7 row 7."""
    def bad(A: sp.f32[M, N], C: sp.f32[M, N]):
        for i in range(M):
            global _unused
            C[i, 0] = A[i, 0]

    sp.kernel(bad)


def corpus_grammar_unsupported_stmt__binding_cycle():
    """§7 row 8's neighbour: a scalar binding cycle (§3.5 lines 15b-15f, FR-S3 item 6)."""
    def bad(A: sp.f32[M, N], C: sp.f32[M, N]):
        for i in range(M):
            x = y
            y = x
            C[i, 0] = x

    sp.kernel(bad)


def corpus_grammar_unsupported_stmt__no_loop():
    """A kernel with no `for` at all has no iteration domain to map."""
    def bad(C: sp.f32[M, N]):
        C[0, 0] = 1.0

    sp.kernel(bad)


def corpus_grammar_unsupported_stmt__no_store():
    """A kernel that binds a scalar and stores nothing writes no operand."""
    def bad(C: sp.f32[M, N]):
        for i in range(M):
            x = 1.0

    sp.kernel(bad)


def corpus_grammar_unsupported_stmt__no_source():
    """A kernel compiled from a string: `inspect.getsource` cannot read it.

    The A3 regression (HANDOFF, Person A item 5). The diagnostic now carries `fn.__code__`'s
    file and first line, so it is a `GrammarError` the user can read rather than a `ValueError`
    out of M0's I71.
    """
    namespace: dict = {"sp": sp}
    exec(compile("def k(C: sp.f32[8, 8]):\n"
                 "    for i in range(8):\n"
                 "        C[i, 0] = 1.0\n", "<corpus>", "exec"), namespace)
    sp.kernel(namespace["k"])


def corpus_grammar_unsupported_expr__lambda():
    """§7 row 10."""
    def bad(A: sp.f32[M, N], C: sp.f32[M, N]):
        for i in range(M):
            f = lambda x: x
            C[i, 0] = f

    sp.kernel(bad)


def corpus_grammar_unsupported_expr__attribute():
    """§7 row 11."""
    def bad(A: sp.f32[M, N], C: sp.f32[M, N]):
        for i in range(M):
            C[i, 0] = A.shape

    sp.kernel(bad)


def corpus_grammar_unsupported_expr__comprehension():
    """§7 row 9's shape: a comprehension is not a scalar expression."""
    def bad(A: sp.f32[M, N], C: sp.f32[M, N]):
        for i in range(M):
            C[i, 0] = [A[i, 0] for _ in range(2)]

    sp.kernel(bad)


def corpus_grammar_unsupported_expr__ifexp_without_compare():
    """§7 row 12: the select's condition must be one comparison (FR-S3 item 8)."""
    def bad(A: sp.f32[M, N], C: sp.f32[M, N]):
        for i in range(M):
            C[i, 0] = A[i, 0] if i else 0.0

    sp.kernel(bad)


def corpus_grammar_unsupported_expr__bare_compare():
    """A comparison outside `a if <cmp> else b` is still rejected (FR-S3, last sentence)."""
    def bad(A: sp.f32[M, N], C: sp.f32[M, N]):
        for i in range(M):
            C[i, 0] = A[i, 0] > 0

    sp.kernel(bad)


def corpus_grammar_unsupported_call__generator():
    """§7 row 9 as written (`sum(x for x in ...)`): the call is rejected before the generator.

    Recorded rather than silently re-pointed: the table predicts `GRAMMAR-UNSUPPORTED-EXPR`
    for `GeneratorExp`, and M1 reaches the call node first, so the code is
    `GRAMMAR-UNSUPPORTED-CALL` and names `sum`. The comprehension case above is the row's
    `GRAMMAR-UNSUPPORTED-EXPR` half.
    """
    def bad(A: sp.f32[M, N], C: sp.f32[M, N]):
        for i in range(M):
            C[i, 0] = sum(A[i, 0] for _ in range(2))

    sp.kernel(bad)


def corpus_grammar_nonaffine_subscript__data_dependent():
    """§7 row 15: `A[B[i], 0]` indexes with data."""
    def bad(A: sp.f32[M, N], B: sp.i32[M, N], C: sp.f32[M, N]):
        for i in range(M):
            C[i, 0] = A[B[i, 0], 0]

    sp.kernel(bad)


def corpus_grammar_nonaffine_subscript__power():
    """§7 row 17: `A[i**2, 0]`."""
    def bad(A: sp.f32[M, N], C: sp.f32[M, N]):
        for i in range(M):
            C[i, 0] = A[i ** 2, 0]

    sp.kernel(bad)


def corpus_grammar_nonaffine_subscript__axis_times_shape_param():
    """§7 row 18 (R-b): `A[i*P, 0]` with `P` a shape parameter is not affine either."""
    def bad(A: sp.f32[P, N], C: sp.f32[P, N]):
        for i in range(P):
            C[i, 0] = A[i * P, 0]

    sp.kernel(bad)


def corpus_grammar_nonaffine_subscript__slice():
    """§7 row 19: a slice is not an affine index."""
    def bad(A: sp.f32[M, N], C: sp.f32[M, N]):
        for i in range(M):
            C[i, 0] = A[0:4, 0]

    sp.kernel(bad)


def corpus_grammar_nonaffine_subscript__unknown_name():
    """A subscript naming neither an axis nor a shape parameter."""
    def bad(A: sp.f32[M, N], C: sp.f32[M, N]):
        for i in range(M):
            C[i, 0] = A[zz, 0]

    sp.kernel(bad)


def corpus_grammar_bad_range__non_rectangular():
    """§7 row 21 (R-a): `range(i, N)` makes the domain a triangle."""
    def bad(A: sp.f32[M, N], C: sp.f32[M, N]):
        for i in range(M):
            for j in range(i, N):
                C[i, j] = A[i, j]

    sp.kernel(bad)


def corpus_grammar_bad_range__nonaffine_bound():
    """§7 row 22: the bound names the enclosing loop variable in a product."""
    def bad(A: sp.f32[M, N], C: sp.f32[M, N]):
        for i in range(M):
            for j in range(i * N):
                C[i, 0] = A[i, 0]

    sp.kernel(bad)


def corpus_grammar_bad_range__zero_step():
    """§7 row 23: a step of 0 never terminates."""
    def bad(A: sp.f32[M, N], C: sp.f32[M, N]):
        for i in range(0, M, 0):
            C[i, 0] = A[i, 0]

    sp.kernel(bad)


def corpus_grammar_bad_range__duplicate_axis():
    """Two loops cannot share a variable name: the axis names are the `σ`/`π` frame."""
    def bad(A: sp.f32[M, N], C: sp.f32[M, N]):
        for i in range(M):
            for i in range(N):
                C[i, 0] = A[i, 0]

    sp.kernel(bad)


def corpus_grammar_bad_annotation__missing():
    """§7 row 24: no annotation at all."""
    def bad(A, C: sp.f32[M, N]):
        for i in range(M):
            C[i, 0] = 1.0

    sp.kernel(bad)


def corpus_grammar_bad_annotation__unknown_dtype():
    """§7 row 26: `sp.f64` is not one of the five dtypes."""
    def bad(A: sp.f64[M, N], C: sp.f32[M, N]):
        for i in range(M):
            C[i, 0] = 1.0

    sp.kernel(bad)


def corpus_grammar_bad_annotation__rank_mismatch():
    """§7 row 27: a rank-2 operand accessed with one index (§3.6 line 17)."""
    def bad(A: sp.f32[M, N], C: sp.f32[M, N]):
        for i in range(M):
            C[i, 0] = A[i]

    sp.kernel(bad)


def corpus_grammar_bad_annotation__shape_entry_not_a_name():
    """A shape entry that is neither an int nor a single identifier."""
    def bad(A: sp.f32[P + Q, N], C: sp.f32[M, N]):
        for i in range(M):
            C[i, 0] = 1.0

    sp.kernel(bad)


# --------------------------------------------------------------------------------------------
# The assertions
# --------------------------------------------------------------------------------------------


@pytest.mark.fr("FR-S3", "FR-D3")
def test_grammar_codes():
    """Every corpus function raises the `GrammarError` code its name spells."""
    _corpus.assert_codes(_corpus_module(), GrammarError)


@pytest.mark.fr("FR-S4")
def test_grammar_message_has_three_parts():
    """FR-S4: the construct by its Python name, the file and line, and one fix sentence.

    Asserted over the **whole** grammar corpus, not one case, because the requirement is about
    every rejection: a case that dropped the construct name would otherwise hide behind the
    twenty that carry it.
    """
    for name, exc in _corpus.collect(_corpus_module()):
        diagnostic = exc.diagnostic
        assert diagnostic.location is not None, f"{name} has no location"
        filename, line = diagnostic.location
        assert filename and line >= 1, f"{name} has location {diagnostic.location}"
        assert diagnostic.reason.strip(), f"{name} has no reason"
        # (c) "one sentence saying what the accepted subset allows instead" — the fix line
        assert len(diagnostic.fix.split()) >= 4, f"{name}'s fix is not a sentence"
    # (a) the construct by its own Python grammar name
    exc = dict(_corpus.collect(_corpus_module()))["raises_grammar_unsupported_stmt"]
    assert exc.diagnostic.details["construct"] == "If"
    assert "If" in exc.diagnostic.reason


@pytest.mark.fr("FR-S4")
def test_grammar_location_is_relative_to_the_working_directory():
    """A4: a kernel defined in a test module reports a path that is not machine-dependent.

    The suite runs from the repository root, so this file lies under the working directory and
    its diagnostics must name it relatively; a path outside the tree is left alone, which is
    what `corpus_grammar_unsupported_stmt__no_source`'s `<corpus>` exercises.

    The **line** is asserted against this file too: `ast.parse` numbers the extracted snippet,
    so before the 2026-09-15 erratum every grammar diagnostic counted from the `def`.
    """
    cases = dict(_corpus.collect(_corpus_module()))
    filename, line = cases["raises_grammar_unsupported_stmt"].diagnostic.location
    assert not filename.startswith("/"), f"{filename} is absolute"
    assert filename.endswith("tests/negative/test_grammar.py"), filename
    source = Path(__file__).read_text(encoding="utf-8").splitlines()
    assert source[line - 1].strip() == "if i > 0:", source[line - 1]


@pytest.mark.fr("FR-S3")
def test_grammar_unreadable_source_is_a_grammar_error():
    """A3: an `exec`-compiled kernel is rejected, not crashed on.

    `inspect.getsource` fails, `_fn_location` falls back to `fn.__code__`, and M0's I71 is
    satisfied — before the fix this raised `ValueError: Diagnostic.location: is required for a
    grammar diagnostic`.
    """
    with pytest.raises(GrammarError) as excinfo:
        corpus_grammar_unsupported_stmt__no_source()
    diagnostic = excinfo.value.diagnostic
    assert diagnostic.code == "GRAMMAR-UNSUPPORTED-STMT"
    assert diagnostic.location == ("<corpus>", 1)
    assert "could not be read" in diagnostic.reason


def _corpus_module():
    import sys
    return sys.modules[__name__]
