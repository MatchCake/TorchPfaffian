from functools import lru_cache

import torch

SMALL_N_MAX = 10


@lru_cache(maxsize=None)
def _matching_tables(
    dimension: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], tuple[float, ...]]:
    r"""
    Perfect-matching tables of ``{0, ..., dimension - 1}`` with their permutation signs.

    The ``(dimension - 1)!!`` perfect matchings are enumerated once per dimension (memoized) so the
    unrolled kernel can evaluate them as a single batched gather. Each matching contributes a term
    ``sign * prod_k a[rows[k], cols[k]]`` to the Pfaffian.

    :param dimension: Even matrix dimension.
    :return: A triple ``(rows, cols, signs)`` where ``rows`` and ``cols`` each hold one
        ``dimension // 2`` tuple of indices per matching and ``signs`` holds the matching's
        permutation sign.
    :rtype: tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], tuple[float, ...]]
    """
    rows: list[tuple[int, ...]] = []
    cols: list[tuple[int, ...]] = []
    signs: list[float] = []

    def expand(indices: tuple[int, ...], row_acc: tuple[int, ...], col_acc: tuple[int, ...], sign: float) -> None:
        if not indices:
            rows.append(row_acc)
            cols.append(col_acc)
            signs.append(sign)
            return
        first, rest = indices[0], indices[1:]
        for position, partner in enumerate(rest):
            remaining = tuple(index for index in rest if index != partner)
            expand(remaining, row_acc + (first,), col_acc + (partner,), sign * (-1.0) ** position)

    expand(tuple(range(dimension)), (), (), 1.0)
    return tuple(rows), tuple(cols), tuple(signs)


def small_pfaffian(matrix: torch.Tensor) -> torch.Tensor:
    r"""
    Signed/complex Pfaffian of a batch ``(..., m, m)`` for even ``m <= SMALL_N_MAX`` via unrolled matchings.

    This is the compile-time form of the recursive expansion: the perfect matchings are enumerated
    once per dimension and evaluated as one batched gather followed by unrolled elementwise products
    and a matvec with the sign vector, with no elimination loop or data-dependent control flow. The
    factor product is unrolled (rather than :func:`torch.prod`) so that zero entries still get correct
    product-rule gradients, making the kernel exact and autograd-friendly for the supported sizes.

    :param matrix: Skew-symmetric matrices of shape ``(..., m, m)`` with even ``m <= SMALL_N_MAX``.
    :return: The signed Pfaffian of shape ``(...,)``, sharing the input backend, dtype and device.
    :rtype: torch.Tensor
    """
    dimension = matrix.shape[-1]
    if matrix.ndim < 2 or matrix.shape[-2] != dimension:
        raise ValueError(f"expected (..., m, m) square matrices, got shape {tuple(matrix.shape)}")
    if dimension % 2 == 1:
        # Value 0 but carrying a grad connection to matrix (gradient 0), so autograd returns a zero
        # gradient instead of raising on a fresh constant with no grad_fn. Summing an empty slice
        # (rather than the whole matrix times 0) keeps this inf/nan-safe: the odd-dim Pfaffian is 0
        # regardless of the entries, so non-finite entries must not contaminate the result.
        return matrix.reshape(*matrix.shape[:-2], -1)[..., :0].sum(-1)
    if dimension == 0:
        return matrix.reshape(*matrix.shape[:-2], -1)[..., :0].sum(-1) + 1  # pf of a 0x0 matrix is 1
    if dimension == 2:
        return matrix[..., 0, 1]
    if dimension > SMALL_N_MAX:
        raise ValueError(f"small_pfaffian supports m <= {SMALL_N_MAX}, got {dimension}")

    rows, cols, signs = _matching_tables(dimension)
    row_index = torch.tensor(rows, dtype=torch.long, device=matrix.device)  # (n_terms, m // 2)
    col_index = torch.tensor(cols, dtype=torch.long, device=matrix.device)  # (n_terms, m // 2)
    sign_vector = torch.tensor(signs, dtype=matrix.dtype, device=matrix.device)  # (n_terms,)
    factors = matrix[..., row_index, col_index]  # (..., n_terms, m // 2)
    product = factors[..., 0]
    for factor_index in range(1, dimension // 2):
        product = product * factors[..., factor_index]
    return product @ sign_vector
