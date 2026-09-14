"""intlin -- integer linear algebra for M3 (and M1). Owner: Person A.

Spec: design/03-lld-M3-checker.md §3.2. Fraction-free (Bareiss) integer Gaussian elimination
over Python `int`. No `fractions.Fraction`, no `numpy.linalg` -- both are explicitly rejected by
the LLD because a rank or null-space that depends on a floating-point tolerance is not
deterministic across machines (NFR-1). Python `int` is unbounded, which matters: entries are
bounded by tile factors and n <= 8, so a determinant can reach the high teens of digits, past
int64.

Every basis returned here is **canonical**: row-echelon pivots left to right, one free-column
generator per row, back-substituted, cleared to primitive integers (gcd 1), sign-normalised (the
first nonzero entry is positive). Two calls on equal input, in different processes, with
different `PYTHONHASHSEED`, return equal output.
"""

from __future__ import annotations

from math import gcd
from typing import Sequence

Vector = tuple[int, ...]
Matrix = tuple[Vector, ...]


def _dims(m: Matrix) -> tuple[int, int]:
    if not m:
        return 0, 0
    return len(m), len(m[0])


def _row_gcd(row: Sequence[int]) -> int:
    g = 0
    for x in row:
        g = gcd(g, abs(x))
    return g


def _primitive_sign(v: list[int]) -> tuple[int, ...]:
    """Clear to primitive integers (gcd 1) and sign-normalise (first nonzero > 0)."""
    g = _row_gcd(v)
    if g > 1:
        v = [x // g for x in v]
    for x in v:
        if x != 0:
            if x < 0:
                v = [-y for y in v]
            break
    return tuple(v)


def _bareiss_row_echelon(m: Matrix) -> tuple[list[list[int]], list[int]]:
    """Fraction-free (Bareiss) row echelon form. Returns (H, pivot_columns).

    H is r x n with r = rank(m); pivot_columns is the column index of each row's pivot, in row
    order (so it is sorted ascending). H's entries are exact integers (minors of the input), not
    generally primitive -- callers that need primitive rows normalise themselves.
    """
    rows, cols = _dims(m)
    h = [list(row) for row in m]
    pivots: list[int] = []
    prev_pivot = 1
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        piv = next((i for i in range(r, rows) if h[i][c] != 0), None)
        if piv is None:
            continue
        h[r], h[piv] = h[piv], h[r]
        piv_val = h[r][c]
        for i in range(r + 1, rows):
            factor = h[i][c]
            if factor == 0:
                continue
            for j in range(cols):
                h[i][j] = (piv_val * h[i][j] - factor * h[r][j]) // prev_pivot
        prev_pivot = piv_val
        pivots.append(c)
        r += 1
    return h[:r], pivots


def rank(m: Matrix) -> int:
    """The rank of `m` over Q (equivalently over Z, up to torsion, which never arises here)."""
    if not m:
        return 0
    _, pivots = _bareiss_row_echelon(m)
    return len(pivots)


def kernel_basis(m: Matrix) -> Matrix:
    """A canonical basis of the integer null space of `m` (design/03-lld-M3-checker.md §3.2).

    One generator per free column, back-substituted over the pivot rows, primitive and
    sign-normalised, ordered by ascending free-column index.
    """
    rows, cols = _dims(m)
    if cols == 0:
        return ()
    if rows == 0:
        # No constraints at all: the whole space is free, one generator per column.
        basis = []
        for c in range(cols):
            v = [0] * cols
            v[c] = 1
            basis.append(_primitive_sign(v))
        return tuple(basis)
    h, pivots = _bareiss_row_echelon(m)
    pivot_set = set(pivots)
    free = [c for c in range(cols) if c not in pivot_set]
    basis = []
    for f in free:
        v = [0] * cols
        v[f] = 1
        # Back-substitute from the LAST pivot row to the first: row i has pivot at pivots[i]
        # and, being row-echelon, involves only columns >= pivots[i].
        for i in range(len(pivots) - 1, -1, -1):
            pc = pivots[i]
            row = h[i]
            # row[pc] * v[pc] + sum_{j > pc} row[j] * v[j] = 0  =>  solve for v[pc]
            acc = sum(row[j] * v[j] for j in range(pc + 1, cols))
            piv_val = row[pc]
            total = -acc
            if total % piv_val != 0:
                # Not integral over this pivot alone: rescale the whole vector by piv_val and
                # retry the accumulation (fraction-free clearing), then re-primitivise at the end.
                scale = piv_val
                v = [x * scale for x in v]
                acc = sum(row[j] * v[j] for j in range(pc + 1, cols))
                total = -acc
            v[pc] = total // piv_val
        basis.append(_primitive_sign(v))
    return tuple(basis)


def solve(m: Matrix, b: Sequence[int]) -> Vector | None:
    """An integer particular solution of `m @ x = b`, or `None` if none exists."""
    rows, cols = _dims(m)
    if rows == 0:
        return tuple(0 for _ in range(cols)) if all(v == 0 for v in b) else None
    # Augment and row-reduce the augmented matrix, but track pivots against `m` alone by
    # eliminating with the same multipliers on the augmented column.
    h = [list(row) + [bi] for row, bi in zip(m, b)]
    pivots: list[int] = []
    prev_pivot = 1
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        piv = next((i for i in range(r, rows) if h[i][c] != 0), None)
        if piv is None:
            continue
        h[r], h[piv] = h[piv], h[r]
        piv_val = h[r][c]
        for i in range(r + 1, rows):
            factor = h[i][c]
            if factor == 0:
                continue
            for j in range(cols + 1):
                h[i][j] = (piv_val * h[i][j] - factor * h[r][j]) // prev_pivot
        prev_pivot = piv_val
        pivots.append(c)
        r += 1
    # Consistency: every all-zero-in-m row of the reduced system must have b-entry 0.
    for i in range(r, rows):
        if h[i][cols] != 0:
            return None
    x = [0] * cols
    for i in range(len(pivots) - 1, -1, -1):
        pc = pivots[i]
        row = h[i]
        acc = sum(row[j] * x[j] for j in range(pc + 1, cols))
        num = row[cols] - acc
        if num % row[pc] != 0:
            return None
        x[pc] = num // row[pc]
    # Verify (free columns were left at 0; confirm m @ x == b exactly, catching any
    # inconsistency the triangular back-substitution alone would not surface).
    for row, bi in zip(m, b):
        if sum(a * xv for a, xv in zip(row, x)) != bi:
            return None
    return tuple(x)


def _row_space_canonical(vectors: Matrix) -> Matrix:
    """Canonicalise an arbitrary spanning set into the same normal form `kernel_basis` uses.

    Used to bring `intersect`/`complement` results (which are correct but not necessarily
    row-echelon) into the canonical form NFR-1 requires. A row space's canonical basis is
    computed as the kernel basis of *a* matrix whose null space is that row space -- i.e. via a
    dual computation using `kernel_basis` twice, which keeps every step in the same fraction-free
    machinery rather than introducing a second algorithm.
    """
    if not vectors:
        return ()
    n = len(vectors[0])
    # A dual (annihilator) of span(vectors): the kernel of the matrix whose rows are `vectors`
    # has row space equal to span(vectors)'s orthogonal complement in the *dual* sense used here
    # is not what we want directly; instead we just row-reduce `vectors` themselves with the
    # same Bareiss echelon used elsewhere, then back-substitute to reduced (not just echelon)
    # form so the basis is canonical.
    h, pivots = _bareiss_row_echelon(vectors)
    if not pivots:
        return ()
    # Reduce upward: clear entries above each pivot too (reduced row echelon), fraction-free,
    # then primitivise each row independently.
    h = [row[:] for row in h]
    for i in range(len(pivots)):
        pc = pivots[i]
        piv_val = h[i][pc]
        for above in range(i):
            if h[above][pc] == 0:
                continue
            factor_above, factor_i = h[above][pc], piv_val
            for j in range(n):
                h[above][j] = factor_i * h[above][j] - factor_above * h[i][j]
    return tuple(_primitive_sign(row) for row in h)


def intersect(a: Matrix, b: Matrix) -> Matrix:
    """A canonical basis of span(a) intersect span(b)."""
    if not a or not b:
        return ()
    n = len(a[0])
    m, k = len(a), len(b)
    # Solve alpha, beta with sum(alpha_i a_i) - sum(beta_j b_j) = 0: a system of n equations
    # in (m + k) unknowns, columns = a_1..a_m, -b_1..-b_k (each as a length-n column).
    constraint = []
    for row in range(n):
        constraint.append(tuple([a[i][row] for i in range(m)] + [-b[j][row] for j in range(k)]))
    ker = kernel_basis(tuple(constraint))
    vectors = []
    for v in ker:
        alpha = v[:m]
        vec = [0] * n
        for i, coeff in enumerate(alpha):
            if coeff == 0:
                continue
            for col in range(n):
                vec[col] += coeff * a[i][col]
        vectors.append(tuple(vec))
    return _row_space_canonical(tuple(vectors))


def contains(a: Matrix, b: Matrix) -> bool:
    """Is span(b) subseteq span(a)?"""
    if not b:
        return True
    if not a:
        return False
    return rank(a) == rank(tuple(a) + tuple(b))


def complement(space: Matrix, sub: Matrix) -> Matrix:
    """A basis of the rows of `space` extending `sub` to the same rank as `space + sub`.

    Greedy: walk `space`'s canonical basis in order, keep a row if it strictly increases the
    running rank starting from `sub`. Deterministic because both inputs are canonical and the
    walk is in index order.
    """
    kept: list[Vector] = []
    running = tuple(sub)
    for row in space:
        candidate = running + (row,)
        if rank(candidate) > rank(running):
            kept.append(row)
            running = candidate
    return tuple(kept)


def matvec(m: Matrix, v: Sequence[int]) -> Vector:
    return tuple(sum(a * b for a, b in zip(row, v)) for row in m)


def lexpos(v: Sequence[int]) -> bool:
    """Is `v` lexicographically positive: the first nonzero entry is >= 1?"""
    for x in v:
        if x != 0:
            return x >= 1
    return False


__all__ = ["Vector", "Matrix", "rank", "kernel_basis", "solve", "intersect", "contains",
           "complement", "matvec", "lexpos"]
