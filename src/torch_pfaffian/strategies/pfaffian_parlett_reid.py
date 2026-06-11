from typing import cast

import torch

from .strategy import PfaffianStrategy


class PfaffianParlettReid(PfaffianStrategy):
    """
    Compute the signed Pfaffian of a skew-symmetric matrix with the Parlett-Reid algorithm.

    The forward performs a batched skew-tridiagonalization with partial pivoting: the batch is
    processed together and only the ``n / 2`` column-elimination steps are sequential. The result
    is the signed Pfaffian, unlike the determinant-based strategies which return only the magnitude.
    The backward uses the closed form ``d pf(A) / d A = (1 / 2) pf(A) (A^{-1})^T`` via a single
    pseudo-inverse, with no autograd graph over the elimination. It equals the inverse for invertible
    inputs (the exact gradient) and returns a finite zero on singular inputs instead of raising. At an
    exactly-singular input that zero is the analytic continuation, not the true derivative, but such
    points are measure-zero.

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

        working = matrix.reshape(-1, dimension, dimension).clone()  # (batch, 2n, 2n)
        batch = working.shape[0]
        batch_index = torch.arange(batch, device=matrix.device)
        sign = torch.ones(batch, dtype=matrix.dtype, device=matrix.device)
        valid = torch.ones(batch, dtype=torch.bool, device=matrix.device)
        epsilon = PfaffianStrategy.EPSILON

        for column in range(0, dimension - 2, 2):
            sub_column = working[:, column + 2 :, column].abs()  # (batch, 2n - column - 2)
            pivot_relative = sub_column.argmax(dim=1)  # (batch,)
            pivot_row = pivot_relative + column + 2  # (batch,)
            pivot_magnitude = sub_column.gather(1, pivot_relative[:, None]).squeeze(1)  # (batch,)
            pivot_condition = pivot_magnitude > working[:, column + 1, column].abs()  # (batch,)
            mask = pivot_condition[:, None]  # (batch, 1)

            # Congruence swap of rows/columns column+1 <-> pivot_row where pivoting helps.
            row_fixed = working[:, column + 1, :].clone()
            row_pivot = working[batch_index, pivot_row, :].clone()
            working[:, column + 1, :] = torch.where(mask, row_pivot, row_fixed)
            working[batch_index, pivot_row, :] = torch.where(mask, row_fixed, row_pivot)
            col_fixed = working[:, :, column + 1].clone()
            col_pivot = working[batch_index, :, pivot_row].clone()
            working[:, :, column + 1] = torch.where(mask, col_pivot, col_fixed)
            working[batch_index, :, pivot_row] = torch.where(mask, col_fixed, col_pivot)
            sign = sign * torch.where(pivot_condition, -torch.ones_like(sign), torch.ones_like(sign))

            pivot_value = working[:, column + 1, column]  # (batch,)
            zero_pivot = pivot_value.abs() < epsilon
            valid = valid & ~zero_pivot
            safe_pivot = torch.where(zero_pivot, torch.ones_like(pivot_value), pivot_value)
            tau = working[:, column + 2 :, column] / safe_pivot[:, None]  # (batch, 2n - column - 2)
            column_next = working[:, column + 2 :, column + 1]  # (batch, 2n - column - 2)
            update = torch.einsum("bi,bj->bij", tau, column_next) - torch.einsum("bi,bj->bij", column_next, tau)
            working[:, column + 2 :, column + 2 :] = working[:, column + 2 :, column + 2 :] + update
            working[:, column + 2 :, column] = 0
            working[:, column, column + 2 :] = 0
            working[:, column + 2 :, column + 1] = 0
            working[:, column + 1, column + 2 :] = 0

        super_index = torch.arange(0, dimension, 2, device=matrix.device)
        super_entries = working[:, super_index, super_index + 1]  # (batch, n)
        raw_pfaffian = sign * super_entries.prod(dim=1)  # (batch,)
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
        grad_matrix = None
        if ctx.needs_input_grad[0]:
            # Use pinv, not inv: identical to inv for invertible A (exact gradient), but on a
            # singular A it returns a finite 0 (since pf = 0) instead of raising. That 0 is the
            # analytic continuation, not the true derivative at the exactly-singular point; such
            # points are measure-zero and the forward there is also 0, so this is acceptable.
            inverse = torch.linalg.pinv(matrix)
            grad_matrix = torch.einsum("...,...ij->...ji", 0.5 * grad_output * pfaffian, inverse)
        return grad_matrix
