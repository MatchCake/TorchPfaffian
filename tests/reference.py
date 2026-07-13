from fractions import Fraction
from typing import Any, Sequence

import numpy as np


def _as_rows(matrix: Any) -> list[list[Any]]:
    if isinstance(matrix, np.ndarray):
        return [list(row) for row in matrix.tolist()] if matrix.dtype != object else [list(row) for row in matrix]
    return [list(row) for row in matrix]


def pfaffian_combinatorial(matrix: Any) -> Any:
    """
    Pfaffian by the perfect-matching (recursive Laplace) expansion.

    Exact for exact scalar types; O((m-1)!!) terms. ``matrix`` must be square and
    skew-symmetric; odd dimension returns 0.
    """
    rows = _as_rows(matrix)
    m = len(rows)
    if m == 0:
        return 1
    if m % 2 == 1:
        return 0

    def expand(indices: Sequence[int]) -> Any:
        if len(indices) == 2:
            return rows[indices[0]][indices[1]]
        first, rest = indices[0], indices[1:]
        total = None
        for k, j in enumerate(rest):
            remaining = tuple(idx for idx in rest if idx != j)
            term = rows[first][j] * expand(remaining)
            if k % 2 == 1:
                term = -term
            total = term if total is None else total + term
        return total

    return expand(tuple(range(m)))


def det_exact(matrix: Any) -> Fraction:
    """Exact determinant of a matrix with int/Fraction entries (fraction-free Bareiss)."""
    rows = [[Fraction(x) for x in row] for row in _as_rows(matrix)]
    m = len(rows)
    if m == 0:
        return Fraction(1)
    sign = 1
    prev = Fraction(1)
    for c in range(m - 1):
        if rows[c][c] == 0:
            for r in range(c + 1, m):
                if rows[r][c] != 0:
                    rows[c], rows[r] = rows[r], rows[c]
                    sign = -sign
                    break
            else:
                return Fraction(0)
        for r in range(c + 1, m):
            for k in range(c + 1, m):
                rows[r][k] = (rows[r][k] * rows[c][c] - rows[r][c] * rows[c][k]) / prev
            rows[r][c] = Fraction(0)
        prev = rows[c][c]
    return sign * rows[m - 1][m - 1]


def pfaffian_numpy(matrix: np.ndarray) -> complex | float:
    """
    Pfaffian of one skew-symmetric matrix by pivoted block Parlett-Reid elimination.

    Supports real and complex dtypes. Partial pivoting: at each step the row/column
    pair (c+1, r) with the largest |M[c, r]| is swapped into position c+1 (each swap
    flips the sign). A vanishing pivot column means Pf = 0 exactly.
    """
    a = np.array(matrix, copy=True)
    m = a.shape[-1]
    if a.shape != (m, m):
        raise ValueError(f"expected a single square matrix, got shape {a.shape}")
    if m % 2 == 1:
        return 0.0
    if m == 0:
        return 1.0

    result = a.dtype.type(1)
    for c in range(0, m - 2, 2):
        # Pivot: bring the largest |M[c, r]| (r > c) into position (c, c+1).
        r = int(np.argmax(np.abs(a[c, c + 1 :]))) + c + 1
        if a[c, r] == 0:
            return 0.0
        if r != c + 1:
            a[[c + 1, r], :] = a[[r, c + 1], :]
            a[:, [c + 1, r]] = a[:, [r, c + 1]]
            result = -result
        pivot = a[c, c + 1]
        result = result * pivot
        # Schur complement of the leading 2x2 block onto the trailing submatrix.
        b1 = a[c, c + 2 :]
        b2 = a[c + 1, c + 2 :]
        a[c + 2 :, c + 2 :] += (np.outer(b2, b1) - np.outer(b1, b2)) / pivot
    result = result * a[m - 2, m - 1]
    return complex(result) if np.iscomplexobj(a) else float(result)


def slog_pfaffian_numpy(matrix: np.ndarray) -> tuple[complex | float, float]:
    """(sign_or_phase, log|Pf|) variant of :func:`pfaffian_numpy`; (0, -inf) when Pf = 0."""
    a = np.array(matrix, copy=True)
    m = a.shape[-1]
    if m % 2 == 1:
        return 0.0, -np.inf
    if m == 0:
        return 1.0, 0.0

    phase = a.dtype.type(1)
    log_abs = 0.0
    for c in range(0, m - 2, 2):
        r = int(np.argmax(np.abs(a[c, c + 1 :]))) + c + 1
        if a[c, r] == 0:
            return 0.0, -np.inf
        if r != c + 1:
            a[[c + 1, r], :] = a[[r, c + 1], :]
            a[:, [c + 1, r]] = a[:, [r, c + 1]]
            phase = -phase
        pivot = a[c, c + 1]
        phase = phase * (pivot / abs(pivot))
        log_abs += float(np.log(abs(pivot)))
        b1 = a[c, c + 2 :]
        b2 = a[c + 1, c + 2 :]
        a[c + 2 :, c + 2 :] += (np.outer(b2, b1) - np.outer(b1, b2)) / pivot
    last = a[m - 2, m - 1]
    if last == 0:
        return 0.0, -np.inf
    phase = phase * (last / abs(last))
    log_abs += float(np.log(abs(last)))
    if np.iscomplexobj(a):
        return complex(phase), log_abs
    return float(np.real(phase)), log_abs


def random_skew(rng: np.random.Generator, m: int, dtype: type = np.float64, scale: float = 1.0) -> np.ndarray:
    """Random dense skew-symmetric matrix (complex if ``dtype`` is complex)."""
    x = rng.standard_normal((m, m)) * scale
    if np.issubdtype(dtype, np.complexfloating):
        x = x + 1j * rng.standard_normal((m, m)) * scale
    x = x.astype(dtype)
    return x - x.T


def random_skew_fractions(rng: np.random.Generator, m: int, denominator: int = 16, max_num: int = 32) -> np.ndarray:
    """Random skew-symmetric matrix of exact Fractions (object dtype), for exact-arithmetic tests."""
    a = np.zeros((m, m), dtype=object)
    for i in range(m):
        for j in range(i + 1, m):
            value = Fraction(int(rng.integers(-max_num, max_num + 1)), denominator)
            a[i, j] = value
            a[j, i] = -value
    for i in range(m):
        a[i, i] = Fraction(0)
    return a
