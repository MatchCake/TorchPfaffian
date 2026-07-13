from typing import cast

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
        pf = torch.sqrt(torch.clamp(torch.abs(det), min=PfaffianFDBPf.EPSILON))
        return pf

    @staticmethod
    def backward(ctx: torch.autograd.function.BackwardCFunction, grad_output: torch.Tensor) -> torch.Tensor | None:
        r"""
        Gradient of the Pfaffian magnitude with respect to the input matrix.

        .. math::
            \frac{\partial |\text{pf}(A)|}{\partial A_{ij}} = \frac{|\text{pf}(A)|}{2} (A^{-1})_{ji}

        The inverse uses :func:`torch.linalg.inv` (an LU factorization) rather than
        :func:`torch.linalg.pinv` (an SVD), which can fail to converge on ill-conditioned or
        near-repeated-singular-value inputs. The forward floors the magnitude at ``sqrt(EPSILON)``, so
        singular elements (``pf`` at that floor, and the always-singular odd-dimensional inputs whose
        ``pf`` is ``0``) are replaced by the identity before the batched inverse: ``inv`` raises on an
        exactly-singular matrix, and there the gradient is negligible anyway since it scales with ``pf``.

        :param ctx: Context holding the saved input matrix and the forward Pfaffian magnitude.
        :param grad_output: Gradient of the output with respect to the loss.
        :return: Gradient of the input matrix, or ``None`` when the input does not require grad.
        :rtype: torch.Tensor | None
        """
        matrix, pf = cast("tuple[torch.Tensor, torch.Tensor]", ctx.saved_tensors)
        if not ctx.needs_input_grad[0]:
            return None
        # The forward clamps the radicand at EPSILON, so a singular element has pf exactly at the floor
        # sqrt(EPSILON) (or 0 for odd dimensions). Computing the floor with the same dtype and sqrt as
        # the forward makes the comparison exact rather than dependent on a hand-written threshold.
        dimension = matrix.shape[-1]
        singular_floor = matrix.new_tensor(PfaffianFDBPf.EPSILON).sqrt()
        singular = pf <= singular_floor
        if not PfaffianFDBPf.EXACT_SINGULAR_GRAD:
            # Sync-free path: replace singular elements by the identity for the batched inverse and
            # assign them an exactly zero gradient, so no host branch runs.
            identity = torch.eye(dimension, dtype=matrix.dtype, device=matrix.device).expand_as(matrix)
            safe_matrix = torch.where(singular[..., None, None], identity, matrix)
            inverse, _ = torch.linalg.inv_ex(safe_matrix)
            grad = torch.einsum("...,...ij->...ji", 0.5 * grad_output * pf, inverse)
            return torch.where(singular[..., None, None], torch.zeros_like(grad), grad)
        if bool(singular.any()):
            identity = torch.eye(dimension, dtype=matrix.dtype, device=matrix.device).expand_as(matrix)
            safe_matrix = torch.where(singular[..., None, None], identity, matrix)
            inverse = torch.linalg.inv(safe_matrix)
        else:
            inverse = torch.linalg.inv(matrix)
        return torch.einsum("...,...ij->...ji", 0.5 * grad_output * pf, inverse)
