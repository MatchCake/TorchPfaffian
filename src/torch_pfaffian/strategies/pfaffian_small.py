import torch

from ..kernels.small_n import small_pfaffian
from .strategy import PfaffianStrategy


class PfaffianSmall(PfaffianStrategy):
    r"""
    Signed Pfaffian for small even dimensions via the unrolled perfect-matching kernel.

    Like :class:`PfaffianDet`, this strategy differentiates straight through plain tensor ops rather
    than a custom analytic backward, so it overrides :meth:`apply` to call
    :func:`~torch_pfaffian.kernels.small_n.small_pfaffian` directly and lets autograd handle the
    backward. The kernel is exact and autograd-friendly for even dimensions ``m <= SMALL_N_MAX``.

    For ``m >= 2`` the input is re-skew-symmetrized (``A = 0.5 * (A - A^T)``) before the kernel. The
    unrolled kernel only reads the upper triangle, so raw autograd would pile the whole gradient
    there, whereas the analytic vector-Jacobian product of the other strategies returns the
    skew-split convention ``0.5 * pf(A) (A^{-1})^T``. The projection is idempotent for skew inputs, so
    the forward is unchanged while the gradient matches the other strategies (including at ``m = 2``,
    where the gradient is ``[[0, 0.5], [-0.5, 0]]`` rather than the upper-triangle ``[[0, 1], [0, 0]]``).

    The input is a skew-symmetric matrix of shape ``(..., m, m)``.
    """

    NAME = "PfaffianSmall"

    @staticmethod
    def apply(matrix: torch.Tensor) -> torch.Tensor:
        if matrix.shape[-1] >= 2:
            matrix = 0.5 * (matrix - matrix.transpose(-1, -2))
        return small_pfaffian(matrix)
