from typing import cast

import torch

from .strategy import PfaffianStrategy


class PfaffianBlockDet(PfaffianStrategy):
    """
    This class implements the Pfaffian using the determinant of the matrix for the forward pass and the
    derivative of the Pfaffian with respect to the input matrix for the backward pass.

    The input matrix is considered to be a skew-symmetric matrix.
    """

    NAME = "PfaffianBlockDet"

    @staticmethod
    def forward(matrix: torch.Tensor):
        """
        Compute the Pfaffian of the input matrix using the determinant of the matrix.

        The matrix is a skew-symmetric matrix of shape (..., 2N, 2N).
        """
        # take the upper right block of shape (..., N, N) of the input matrix
        n = matrix.shape[-1] // 2
        sub_matrix = matrix[..., :n, n:]
        pf = (-1) ** (n * (n - 1) // 2) * torch.linalg.det(sub_matrix)
        return pf

    @staticmethod
    def backward(ctx: torch.autograd.function.BackwardCFunction, grad_output: torch.Tensor) -> torch.Tensor | None:
        r"""
        Compute the gradient of the Pfaffian with respect to the full input matrix.

        The forward depends only on the upper-right block ``block = matrix[..., :n, n:]`` through
        ``pf = c * det(block)``. By Jacobi's formula ``d det(block) / d block = det(block) (block^{-1})^T``,
        so ``d pf / d block = c * adj(block)^T`` where ``adj(block) = det(block) block^{-1}`` is the
        classical adjugate. The gradient is therefore zero everywhere except in the upper-right block.

        For an invertible block this is the single inverse ``pf * (block^{-1})^T``. For a singular block
        (``pf == 0``) that product would be ``0 * inf`` and :func:`torch.linalg.inv` raises, yet the true
        derivative ``c * adj(block)^T`` is finite and generally nonzero. The adjugate is then computed
        exactly from the cofactors via :meth:`_cofactor_matrix`, on the singular batch elements only so
        invertible inputs keep the single cheap inverse.

        :param ctx: Context holding the saved input matrix and the forward Pfaffian.
        :param grad_output: Gradient of the output with respect to the loss.
        :return: Gradient of the input matrix, or ``None`` when the input does not require grad.
        :rtype: torch.Tensor | None
        """
        matrix, pf = cast("tuple[torch.Tensor, torch.Tensor]", ctx.saved_tensors)
        if not ctx.needs_input_grad[0]:
            return None
        n = matrix.shape[-1] // 2
        block = matrix[..., :n, n:]  # (..., n, n) upper-right block
        constant = (-1) ** (n * (n - 1) // 2)
        singular = pf == 0
        if bool(singular.any()):
            identity = torch.eye(n, dtype=matrix.dtype, device=matrix.device).expand_as(block)
            safe_block = torch.where(singular[..., None, None], identity, block)
            adjugate_transpose = pf[..., None, None] * torch.linalg.inv(safe_block).transpose(-1, -2)
            flat_block = block.reshape(-1, n, n)
            flat_adjugate = adjugate_transpose.reshape(-1, n, n)
            singular_index = singular.reshape(-1).nonzero(as_tuple=True)[0]
            cofactor = constant * PfaffianBlockDet._cofactor_matrix(flat_block.index_select(0, singular_index))
            flat_adjugate = flat_adjugate.index_copy(0, singular_index, cofactor.to(flat_adjugate.dtype))
            adjugate_transpose = flat_adjugate.reshape_as(block)
        else:
            adjugate_transpose = pf[..., None, None] * torch.linalg.inv(block).transpose(-1, -2)
        grad_matrix = torch.zeros_like(matrix)
        grad_matrix[..., :n, n:] = grad_output[..., None, None] * adjugate_transpose
        return grad_matrix

    @staticmethod
    def _cofactor_matrix(blocks: torch.Tensor) -> torch.Tensor:
        r"""
        Cofactor matrix ``C`` of a batch of square blocks, with ``C_{ij} = (-1)^{i+j} det(B^{(ij)})``
        where ``B^{(ij)}`` is ``B`` with row ``i`` and column ``j`` removed.

        This equals the transpose of the classical adjugate ``adj(B) = det(B) B^{-1}``, so it stays
        finite when ``B`` is singular and an inverse would fail. The minor determinants are batched, so
        the cost is ``n^2`` determinant evaluations regardless of the batch size.

        :param blocks: Square blocks of shape ``(m, n, n)``.
        :return: The cofactor matrix of shape ``(m, n, n)``.
        :rtype: torch.Tensor
        """
        n = blocks.shape[-1]
        cofactor = torch.zeros_like(blocks)
        indices = torch.arange(n, device=blocks.device)
        for row in range(n):
            kept_rows = indices[indices != row]
            for column in range(n):
                kept_columns = indices[indices != column]
                minor = blocks.index_select(-2, kept_rows).index_select(-1, kept_columns)  # (m, n-1, n-1)
                sign = 1.0 if (row + column) % 2 == 0 else -1.0
                cofactor[..., row, column] = sign * torch.linalg.det(minor)
        return cofactor

    @classmethod
    def _pfaffian_adjugate(cls, matrices: torch.Tensor) -> torch.Tensor:
        # This strategy's forward only accepts block-antidiagonal matrices, but the Pfaffian minors
        # are general skew matrices, so the adjugate is computed with the general Parlett-Reid forward.
        from .pfaffian_parlett_reid import PfaffianParlettReid

        return PfaffianParlettReid._pfaffian_adjugate(matrices)
