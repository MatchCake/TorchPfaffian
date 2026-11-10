from typing import cast

import torch

from .strategy import PfaffianStrategy


class PfaffianParlettReid(PfaffianStrategy):
    """
    Compute the signed Pfaffian of a skew-symmetric matrix with the Parlett-Reid algorithm.

    The forward performs a batched skew-tridiagonalization with partial pivoting: the batch is
    processed together and only the ``n / 2`` column-elimination steps are sequential. Each step swaps the
    pivot's row and column into place with one gather and one scatter each and updates the trailing block in
    place with two fused multiply-adds, without zeroing the eliminated entries (they are not read again), and the
    sign, the product and the zero-pivot test are formed once after the loop: about a dozen operators per step.
    On CUDA a step of a small input costs its kernel launches rather than its arithmetic, so that is the run
    time there (``pfaffian()`` sends small CUDA inputs to the Rust kernel on the host instead). The result
    is the signed Pfaffian, unlike the determinant-based strategies which return only the magnitude.
    The backward uses the closed form ``d pf(A) / d A = (1 / 2) pf(A) (A^{-1})^T`` via the Pfaffian
    adjugate, with no autograd graph over the elimination. For invertible inputs this is a single
    inverse; for singular inputs (``pf == 0``) the adjugate is computed exactly from minor
    Pfaffians, so the gradient is correct everywhere (see :meth:`PfaffianStrategy.pfaffian_grad_matrix`).

    The input is a skew-symmetric matrix of shape ``(..., 2n, 2n)``.
    """

    NAME = "PfaffianParlettReid"

    @staticmethod
    def forward(matrix: torch.Tensor) -> torch.Tensor:
        dimension = matrix.shape[-1]
        if dimension % 2 != 0:
            return torch.zeros(matrix.shape[:-2], dtype=matrix.dtype, device=matrix.device)
        if dimension == 0:
            return torch.ones(matrix.shape[:-2], dtype=matrix.dtype, device=matrix.device)

        working = matrix.reshape(-1, dimension, dimension).clone()  # (batch, 2n, 2n), eliminated in place
        batch = working.shape[0]
        rows = torch.arange(batch, device=matrix.device)[:, None].expand(batch, 2)  # (batch, 2)
        offsets = []
        for column in range(0, dimension - 2, 2):
            # Partial pivoting: the largest |A[p, column]| for p > column (the first one on ties, so column + 1 when
            # it is as large as any) is swapped into row and column column + 1. The two rows, then the two
            # columns, are read with one gather and written back swapped with one scatter; when p = column + 1
            # both writes carry the same values.
            offset = working[:, column + 1 :, column].abs().argmax(dim=1)  # (batch,), p - column - 1
            pair = torch.stack([offset.new_full((batch,), column + 1), offset + (column + 1)], dim=1)  # (batch, 2)
            working[rows, pair] = working[rows, pair].flip(1)  # rows p and column + 1, (batch, 2, 2n)
            working[rows, :, pair] = working[rows, :, pair].flip(1)  # their columns
            offsets.append(offset)
            # Schur complement of the pivot block onto the trailing block, in place. The entries left of and
            # above the trailing block are not read again, so they are not zeroed.
            tau = working[:, column + 2 :, column] / working[:, column + 1, column, None]  # (batch, 2n - column - 2)
            column_next = working[:, column + 2 :, column + 1]  # (batch, 2n - column - 2)
            trailing = working[:, column + 2 :, column + 2 :]
            trailing.addcmul_(tau[:, :, None], column_next[:, None, :])
            trailing.addcmul_(column_next[:, :, None], tau[:, None, :], value=-1)

        super_index = torch.arange(0, dimension, 2, device=matrix.device)
        super_entries = working[:, super_index, super_index + 1]  # (batch, n)
        pivots = working[:, super_index[:-1] + 1, super_index[:-1]]  # (batch, n - 1), A'[c + 1, c]
        raw_pfaffian = super_entries.prod(dim=1)  # (batch,)
        if not offsets:  # a 2 x 2 matrix: no elimination step
            return raw_pfaffian.reshape(matrix.shape[:-2])
        swaps = (torch.stack(offsets, dim=1) != 0).sum(dim=1)  # (batch,), each swap flips the sign
        raw_pfaffian = (1 - 2 * (swaps % 2)) * raw_pfaffian
        # A pivot below EPSILON is declared an exact zero, as in the other linear-domain strategies; the steps
        # after it divide by it, but each batch element is eliminated independently, so only it is affected.
        valid = pivots.abs().amin(dim=1) >= PfaffianStrategy.EPSILON
        pfaffian = torch.where(valid, raw_pfaffian, torch.zeros_like(raw_pfaffian))
        return pfaffian.reshape(matrix.shape[:-2])

    @staticmethod
    def backward(ctx: torch.autograd.function.BackwardCFunction, grad_output: torch.Tensor) -> torch.Tensor | None:
        r"""
        Gradient of the signed Pfaffian with respect to the input matrix.

        .. math::
            \frac{\partial \text{pf}(A)}{\partial A_{ij}} = \frac{\text{pf}(A)}{2} (A^{-1})_{ji}

        :param ctx: Context holding the saved input matrix and the forward Pfaffian.
        :param grad_output: Gradient of the output with respect to the loss.
        :return: Gradient of the input matrix, or ``None`` when the input does not require grad.
        :rtype: torch.Tensor | None
        """
        matrix, pfaffian = cast("tuple[torch.Tensor, torch.Tensor]", ctx.saved_tensors)
        if not ctx.needs_input_grad[0]:
            return None
        return PfaffianParlettReid.pfaffian_grad_matrix(matrix, pfaffian, grad_output)
