from typing import Any

import torch

from ..kernels.parlett_reid_slog import parlett_reid_slog


class SlogPfaffianStrategy(torch.autograd.Function):
    r"""
    Log-domain signed Pfaffian returning ``(phase, log|Pf|)`` with an analytic backward for ``log|Pf|``.

    This is a standalone :class:`torch.autograd.Function` (deliberately not a
    :class:`~torch_pfaffian.strategies.strategy.PfaffianStrategy` subclass, so it does not enter the
    scalar-returning strategy registry). The forward is the host-synchronization-free
    :func:`~torch_pfaffian.kernels.parlett_reid_slog.parlett_reid_slog`; the returned ``phase`` is
    marked non-differentiable (for real inputs it is piecewise constant, and callers needing the full
    complex derivative use the linear-domain :class:`PfaffianParlettReid`). ``log|Pf|`` is
    differentiable wherever ``Pf != 0``, with gradient ``0.5 * (A^{-1})^T`` under PyTorch's Wirtinger
    convention (the ``.conj()`` is a no-op for real inputs); batch elements with ``Pf = 0`` receive an
    exactly zero gradient.

    The input is a skew-symmetric matrix of shape ``(..., m, m)``.
    """

    @staticmethod
    def forward(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return parlett_reid_slog(matrix)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[torch.Tensor, ...], output: tuple[torch.Tensor, torch.Tensor]) -> None:
        (matrix,) = inputs
        phase, _ = output
        ctx.save_for_backward(matrix, phase)
        ctx.mark_non_differentiable(phase)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> torch.Tensor:
        r"""
        Gradient of ``log|Pf(A)|`` with respect to the input matrix.

        .. math::
            \frac{\partial \log|\text{pf}(A)|}{\partial A_{ij}} = \frac{1}{2} (A^{-1})_{ji}

        The gradient of the non-differentiable ``phase`` output is ignored. Singular elements
        (``phase == 0``) are replaced by the identity before the batched inverse and then assigned an
        exactly zero gradient, so the backward never synchronizes with the host and never raises.

        :param ctx: Context holding the saved input matrix and the forward phase.
        :param grad_outputs: Gradients of ``(phase, log_abs)`` with respect to the loss; only the
            ``log_abs`` gradient is used.
        :return: Gradient of the input matrix, of shape ``(..., m, m)``.
        :rtype: torch.Tensor
        """
        _grad_phase, grad_log_abs = grad_outputs
        matrix, phase = ctx.saved_tensors
        singular = phase == 0
        identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
        safe_matrix = torch.where(singular[..., None, None], identity, matrix)
        inverse_transposed_conj = torch.linalg.inv(safe_matrix).conj().transpose(-1, -2)
        grad = 0.5 * grad_log_abs[..., None, None] * inverse_transposed_conj
        return torch.where(singular[..., None, None], torch.zeros_like(grad), grad)
