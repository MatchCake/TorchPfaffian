from abc import ABC, abstractmethod
import torch


class PfaffianStrategy(torch.autograd.Function):
    EPSILON = 1e-10
    NAME = "PfaffianStrategy"

    @abstractmethod
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, matrix: torch.Tensor):
        pass

    @abstractmethod
    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output):
        pass

