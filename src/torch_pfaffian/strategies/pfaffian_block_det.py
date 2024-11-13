from typing import Any
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
        pf = (-1)**(n*(n-1)//2) * torch.linalg.det(sub_matrix)
        return pf

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output):
        r"""

        ..math:
            \frac{\partial \text{pf}(A)}{\partial A_{ij}} = \frac{\text{pf}(A)}{2} A^{-1}_{ji}

        :param ctx: Context
        :param grad_output: Gradient of the output
        :return: Gradient of the input
        """
        matrix, pf = ctx.saved_tensors
        grad_matrix = None
        if ctx.needs_input_grad[0]:
            grad_matrix = torch.einsum('...,...ij->...ji', 0.5 * grad_output * pf, torch.linalg.pinv(matrix))
        return grad_matrix


