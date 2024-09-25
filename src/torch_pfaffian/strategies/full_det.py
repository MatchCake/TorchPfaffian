from typing import Any
import torch
from .strategy import PfaffianStrategy


class PfaffianFullDet(PfaffianStrategy):
    """
    This class implements the Pfaffian using the determinant of the matrix for the forward pass and the
    backward pass.
    """
    NAME = "PfaffianFullDet"

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, matrix: torch.Tensor):
        det = torch.det(matrix)
        pf = torch.sqrt(torch.abs(det) + PfaffianFullDet.EPSILON)
        ctx.save_for_backward(matrix, pf)
        return pf

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output):
        r"""
        :param ctx: Context
        :param grad_output: Gradient of the output
        :return: Gradient of the input
        """
        raise NotImplementedError("The backward pass for the PfaffianFullDet strategy is not implemented.")



