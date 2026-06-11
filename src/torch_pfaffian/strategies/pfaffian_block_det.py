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
    def backward(ctx: torch.autograd.function.BackwardCFunction, grad_output):
        r"""
        Compute the gradient of the Pfaffian with respect to the full input matrix.

        The forward depends only on the upper-right block ``block = matrix[..., :n, n:]`` through
        ``pf = c * det(block)``. By Jacobi's formula ``d det(block) / d block = det(block) * (block^{-1})^T``,
        so ``d pf / d block = pf * (block^{-1})^T``. The gradient is therefore zero everywhere except in
        the upper-right block.

        :param ctx: Context holding the saved input matrix and the forward Pfaffian.
        :param grad_output: Gradient of the output with respect to the loss.
        :return: Gradient of the input matrix, or ``None`` when the input does not require grad.
        :rtype: torch.Tensor | None
        """
        matrix, pf = cast("tuple[torch.Tensor, torch.Tensor]", ctx.saved_tensors)
        grad_matrix = None
        if ctx.needs_input_grad[0]:
            n = matrix.shape[-1] // 2
            sub_matrix = matrix[..., :n, n:]  # (..., n, n) upper-right block
            inverse_transpose = torch.linalg.inv(sub_matrix).transpose(-1, -2)
            grad_block = grad_output[..., None, None] * pf[..., None, None] * inverse_transpose
            grad_matrix = torch.zeros_like(matrix)
            grad_matrix[..., :n, n:] = grad_block
        return grad_matrix

    @classmethod
    def _pfaffian_adjugate(cls, matrices: torch.Tensor) -> torch.Tensor:
        # This strategy's forward only accepts block-antidiagonal matrices, but the Pfaffian minors
        # are general skew matrices, so the adjugate is computed with the general Parlett-Reid forward.
        from .pfaffian_parlett_reid import PfaffianParlettReid

        return PfaffianParlettReid._pfaffian_adjugate(matrices)
