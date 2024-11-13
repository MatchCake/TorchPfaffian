import itertools
from typing import Any
import torch
import numpy as np
from .strategy import PfaffianStrategy


class PfaffianPerm(PfaffianStrategy):
    """
    This class implements the Pfaffian using the determinant of the matrix for the forward pass and the
    derivative of the Pfaffian with respect to the input matrix for the backward pass.
    """
    NAME = "PfaffianPerm"

    @staticmethod
    def forward(matrix: torch.Tensor):
        _2n = matrix.shape[-1]
        if _2n % 2 != 0:
            return torch.zeros_like(matrix[..., 0, 0])
        _n = _2n // 2
        # Let P be the set of permutations, {i_1, i_2, ..., i_2n} with respect to {1, 2, ..., 2n}, such that
        # i_1 < j_1 < i_2 < j_2 < ... < i_2n < j_2n and i_1 < i_2 < ... < i_2n
        indexes = np.arange(_2n)
        # _i must be all the vector of length n from indexes. It should look like a sliding windows of size n
        _i_starting_idx = np.arange(_n - 1)
        _i_matrix = indexes[_i_starting_idx[:, None] + np.arange(_n)]
        _j_starting_idx = _i_starting_idx + 1
        _j_matrix = indexes[_j_starting_idx[:, None] + np.arange(_n)]

        # build a matrix of each permutation of i and j where j > i
        ij_matrix = np.concatenate(
            [
                np.concatenate([_i_matrix[i, :, None], _j_matrix[j, :, None]], axis=-1)[None, ...]
                for i in range(_i_matrix.shape[0])
                for j in range(i, _j_matrix.shape[0])
            ],
            axis=0
        )

        pf_matrix = torch.prod(matrix[..., ij_matrix[..., 0], ij_matrix[..., 1]], dim=-1)
        delta_ij = ij_matrix[..., 0, 1] - ij_matrix[..., 0, 0]
        signs = torch.tensor((-1) ** delta_ij).to(pf_matrix.device)
        pf = torch.sum(signs * pf_matrix, dim=-1)
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


