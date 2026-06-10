import torch

from .strategy import PfaffianStrategy


class PfaffianDet(PfaffianStrategy):
    """
    Compute the Pfaffian as ``pf = sqrt(|det(A)|)`` and rely on autograd for the backward pass.

    The other strategies are full ``torch.autograd.Function`` implementations with a custom analytic
    backward. This strategy instead differentiates straight through ``det`` and ``sqrt`` with PyTorch
    autograd, so it overrides :meth:`apply` to bypass the ``torch.autograd.Function`` machinery. The
    radicand is clamped to ``EPSILON`` for numerical stability, matching the other determinant-based
    strategy.
    """

    NAME = "PfaffianDet"

    @staticmethod
    def apply(matrix: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.clamp(torch.abs(torch.linalg.det(matrix)), min=PfaffianStrategy.EPSILON))
