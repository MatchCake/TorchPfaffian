from abc import ABC, abstractmethod
import torch


class PfaffianStrategy(torch.autograd.Function):
    EPSILON = 1e-12
    NAME = "PfaffianStrategy"

    @staticmethod
    def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
        matrix, = inputs
        pf = output
        ctx.save_for_backward(matrix, pf)

    @staticmethod
    def forward(matrix: torch.Tensor):
        pass

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output):
        pass


