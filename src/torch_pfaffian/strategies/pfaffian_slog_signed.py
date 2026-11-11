from typing import Any

import torch

from ..kernels.checked_linalg import CheckedLinalg
from ..kernels.parlett_reid_slog import parlett_reid_slog


class SlogSignedPfaffian(torch.autograd.Function):
    r"""
    Signed Pfaffian reconstructed from the log domain, ``phase * exp(log|Pf|)``, with its full derivative.

    :func:`~torch_pfaffian.pfaffian` recovers with it the elements whose linear-domain result overflowed while
    their true Pfaffian is finite. Differentiating the reconstruction through
    :class:`~torch_pfaffian.strategies.pfaffian_slog.SlogPfaffianStrategy` would only follow ``log|Pf|``, its
    ``phase`` being non-differentiable: exact for a real input, whose phase is a piecewise-constant sign, but
    wrong for a complex one, whose phase depends on the matrix. The backward here is the holomorphic derivative
    of the Pfaffian, ``0.5 * Pf(A) (A^{-1})^T``, conjugated under PyTorch's Wirtinger convention, as for the
    linear-domain strategies. Elements with ``Pf = 0`` get a zero gradient, as in ``SlogPfaffianStrategy``; the
    linear-domain strategies give them the exact adjugate, so this function is meant for nonsingular inputs.

    It is a standalone :class:`torch.autograd.Function`, not a registered strategy. The input is a skew-symmetric
    matrix of shape ``(..., m, m)``, and the output has shape ``(...,)`` and the input dtype.
    """

    @staticmethod
    def forward(matrix: torch.Tensor) -> torch.Tensor:
        phase, log_abs = parlett_reid_slog(matrix)
        return phase * torch.exp(log_abs).to(phase.dtype)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        (matrix,) = inputs
        ctx.save_for_backward(matrix, output)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        r"""
        Gradient of the signed Pfaffian with respect to the input matrix.

        .. math::
            \frac{\partial \text{pf}(A)}{\partial A_{ij}} = \frac{1}{2} \text{pf}(A) (A^{-1})_{ji}

        Elements with ``Pf = 0`` are replaced by the identity before the batched inverse and then assigned a zero
        gradient, so the backward never synchronizes with the host and never raises. The operations are
        differentiable, so a double backward works.

        :param ctx: Context holding the saved input matrix and the forward Pfaffians.
        :param grad_output: Gradient of the output with respect to the loss.
        :return: Gradient of the input matrix, of shape ``(..., m, m)``.
        :rtype: torch.Tensor
        """
        matrix, pfaffian = ctx.saved_tensors
        singular = (pfaffian == 0)[..., None, None]
        identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
        inverse, _ = CheckedLinalg.inv_ex(torch.where(singular, identity, matrix))
        gradient = 0.5 * grad_output[..., None, None] * (pfaffian[..., None, None] * inverse.transpose(-1, -2)).conj()
        return torch.where(singular, torch.zeros_like(gradient), gradient)
