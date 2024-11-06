from typing import Any
import torch
from .strategy import PfaffianStrategy


class PfaffianFDBPf(PfaffianStrategy):
    """
    This class implements the Pfaffian using the determinant of the matrix for the forward pass and the
    derivative of the Pfaffian with respect to the input matrix for the backward pass.
    """
    NAME = "PfaffianFDBPf"

    @staticmethod
    def forward(matrix: torch.Tensor):
        _2n = matrix.shape[-1]
        if _2n % 2 != 0:
            return torch.zeros_like(matrix[..., 0, 0])
        det = torch.linalg.det(matrix)
        pf = torch.sqrt(torch.abs(det) + PfaffianFDBPf.EPSILON)
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


