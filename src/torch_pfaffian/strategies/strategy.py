import torch


class PfaffianStrategy(torch.autograd.Function):
    EPSILON = 1e-30
    NAME = "PfaffianStrategy"

    @staticmethod
    def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
        (matrix,) = inputs
        pf = output
        ctx.save_for_backward(matrix, pf)

    @staticmethod
    def forward(matrix: torch.Tensor):
        pass

    @staticmethod
    def backward(ctx: torch.autograd.function.BackwardCFunction, grad_output):
        pass
