from typing import cast

import torch

from ..kernels.checked_linalg import CheckedLinalg
from .strategy import PfaffianStrategy


class PfaffianMaskedMagnitude(PfaffianStrategy):
    r"""
    Pfaffian magnitude ``sqrt(max(|det(A)|, epsilon))`` whose gradient is zero where ``Pf(A)`` vanishes
    up to round-off.

    The forward is the fast determinant path of :class:`PfaffianDet`, floored at ``sqrt(epsilon)`` with
    a per-call ``epsilon``. The backward returns ``(|Pf(A)| / 2) conj(A^{-T})`` (PyTorch's Wirtinger
    convention for a real output of a complex input), except that it gives a zero gradient to:

    - elements whose determinant was clamped to ``epsilon`` in the forward (the mask is saved in the
      forward rather than recomputed from the floored value, which could be off by one ulp);
    - elements whose LU factorization fails, or whose 1-norm reciprocal condition number is not finite;
    - elements that vanish up to round-off, flagged when both
      ``rcond_1(A) <= ROUNDOFF_RCOND_SCALE * eps`` and
      ``sigma_min(A) / sigma_max(A) <= ROUNDOFF_SPECTRAL_SCALE * eps``, with ``eps`` the machine epsilon
      of the dtype.

    A Pfaffian that is zero in exact arithmetic is computed as round-off, and the unmasked backward then
    divides noise by noise and returns an ``O(1)`` gradient. No numerical test separates such a
    structural zero from a genuine value at round-off level (in float32 the rare outcomes of a state
    near a basis state are genuine yet already 10 to 35% off), so this strategy is opt-in: use it when
    exact zeros are expected, for example the parity-forbidden outcomes of a basis state.

    The determinant is computed with a backward error ``~ eps * sigma_max``, so the round-off of
    ``|Pf(A)| = prod sigma_pair`` is ``~ eps * sigma_max * |Pf(A)| / sigma_min``: the Pfaffian is
    unresolved when ``sigma_min / sigma_max`` is at round-off level, at every size. For a skew matrix
    ``rcond_1(A) <= sigma_min / sigma_max``, and the 1-norm ratio can be up to ``m`` times smaller, so
    the 1-norm test alone would mask small but resolved Pfaffians, while the spectral test alone would
    miss the structural zeros of long circuits. Both tests must agree:

    - a structural zero rounded once measures ``sigma_min / sigma_max <= 0.5 eps`` up to ``m = 256``;
      accumulated round-off (up to ``1e5`` Givens rotations applied in place in complex128) raises it to
      ``~10 eps`` while its ``rcond_1`` stays below ``4 eps``. Longer circuits drift past both tests;
    - a Pfaffian with ``sigma_min / sigma_max ~ 4 eps`` is still accurate to a few percent, so only the
      1-norm test separates it from a structural zero.

    The backward costs one inverse, two 1-norms and one set of singular values per element. When the
    SVD fails to converge, the whole batch falls back on the 1-norm test alone.

    The criterion has a known limit: it is not scale-free across blocks, so a resolved element whose
    exact block structure spans a very wide range (about ``0.25 / eps`` between block scales) can be
    masked; once rotated, such an element is no longer resolved in floating point.
    Elements whose determinant overflows are recovered by :func:`torch_pfaffian.pfaffian` in the log
    domain, which carries the unmasked gradient. Only dtypes with an LU factorization are supported
    (float32, float64, complex64, complex128).

    Odd-dimensional matrices are singular, so their magnitude is the floor ``sqrt(epsilon)`` with a zero
    gradient. :meth:`apply` returns only the magnitude.
    """

    NAME = "PfaffianMaskedMagnitude"
    # Scale c of the round-off test rcond_1(A) <= c * eps. Measured margins (MatchCake basis-state lookups):
    # structural zeros have rcond <= ~1.6 eps, genuine outcomes rcond >= 163 eps in float32 and 9e10 eps in
    # float64.
    ROUNDOFF_RCOND_SCALE = 4.0
    # Scale c of the round-off test sigma_min(A) / sigma_max(A) <= c * eps. Structural zeros of MatchCake
    # lookups measure up to ~10 eps after 1e5 in-place Givens rotations in float64.
    ROUNDOFF_SPECTRAL_SCALE = 16.0

    @staticmethod
    def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
        magnitude, clamped = output
        ctx.mark_non_differentiable(clamped)
        ctx.save_for_backward(inputs[0], magnitude, clamped)

    @staticmethod
    def forward(matrix: torch.Tensor, epsilon: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the floored magnitude and the mask of the elements clamped to the floor.

        :param matrix: Skew-symmetric matrices of shape ``(..., m, m)``, real or complex.
        :param epsilon: Floor of ``|det(A)|``. ``None`` (default) uses ``PfaffianStrategy.EPSILON``.
        :return: The magnitude ``sqrt(max(|det(A)|, epsilon))`` of shape ``(...,)`` in the real dtype of
            ``matrix``, and the boolean clamped mask of shape ``(...,)``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        if epsilon is None:
            epsilon = PfaffianStrategy.EPSILON
        if matrix.shape[-1] % 2 != 0:
            floor = torch.full_like(matrix[..., 0, 0].real, epsilon).sqrt()  # det of an odd skew matrix is 0
            return floor, torch.ones_like(floor, dtype=torch.bool)
        determinant_magnitude = torch.abs(CheckedLinalg.det(matrix))
        magnitude = torch.sqrt(torch.clamp(determinant_magnitude, min=epsilon))
        clamped = ~(determinant_magnitude > epsilon)  # also True where the determinant is NaN
        return magnitude, clamped

    @staticmethod
    def backward(
        ctx: torch.autograd.function.BackwardCFunction,
        grad_output: torch.Tensor,
        grad_clamped: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, None]:
        """
        Gradient ``grad_output * (|Pf(A)| / 2) conj(A^{-T})``, zero at clamped, singular and round-off
        elements.

        :param ctx: Context holding the input matrix, the forward magnitude and the clamped mask.
        :param grad_output: Gradient of the loss with respect to the magnitude, of shape ``(...,)``.
        :param grad_clamped: Unused; the clamped mask is not differentiable.
        :return: Gradient of the input matrix (or ``None`` when it does not require grad), and ``None``
            for ``epsilon``.
        :rtype: tuple[torch.Tensor | None, None]
        """
        matrix, magnitude, clamped = cast("tuple[torch.Tensor, torch.Tensor, torch.Tensor]", ctx.saved_tensors)
        if not ctx.needs_input_grad[0]:
            return None, None
        dimension = matrix.shape[-1]
        if dimension == 0 or dimension % 2 != 0:
            return torch.zeros_like(matrix), None
        epsilon = torch.finfo(matrix.dtype).eps
        threshold = PfaffianMaskedMagnitude.ROUNDOFF_RCOND_SCALE * epsilon
        spectral_threshold = PfaffianMaskedMagnitude.ROUNDOFF_SPECTRAL_SCALE * epsilon
        with torch.no_grad():  # the mask is not differentiable
            inverse, info = CheckedLinalg.inv_ex(matrix)
            rcond = 1.0 / (torch.linalg.matrix_norm(matrix, ord=1) * torch.linalg.matrix_norm(inverse, ord=1))
            # A non-finite magnitude (overflowed det, NaN input) is masked too, so 0 * inf never leaks a NaN.
            invalid = (info != 0) | ~torch.isfinite(rcond) | ~torch.isfinite(magnitude) | clamped
            # The SVD fails on non-finite entries: those elements are masked already, so use the identity.
            identity = torch.eye(dimension, dtype=matrix.dtype, device=matrix.device).expand_as(matrix)
            finite = torch.isfinite(matrix).all(dim=-1).all(dim=-1)
            try:
                singular_values = torch.linalg.svdvals(torch.where(finite[..., None, None], matrix, identity))
                spectral_round_off = ~(singular_values[..., -1] > spectral_threshold * singular_values[..., 0])
            except torch.linalg.LinAlgError:  # the SVD did not converge: fall back on the 1-norm test alone
                spectral_round_off = torch.ones_like(rcond, dtype=torch.bool)
            round_off = (rcond <= threshold) & spectral_round_off
            masked = invalid | round_off
        if torch.is_grad_enabled() and matrix.requires_grad:
            # Double backward: invert the identity in place of the masked elements so the derivative of
            # the inverse stays finite there (it is multiplied by zero, and 0 * inf would give NaN).
            inverse, _ = CheckedLinalg.inv_ex(torch.where(masked[..., None, None], identity, matrix))
        coefficient = (0.5 * grad_output * magnitude).to(matrix.dtype)  # (...,)
        grad = coefficient[..., None, None] * inverse.transpose(-1, -2).conj()  # (..., m, m)
        return torch.where(masked[..., None, None], torch.zeros_like(grad), grad), None

    @classmethod
    def apply(cls, matrix: torch.Tensor, epsilon: float | None = None) -> torch.Tensor:
        """
        Compute the masked-gradient Pfaffian magnitude.

        :param matrix: Skew-symmetric matrices of shape ``(..., m, m)``, real or complex.
        :param epsilon: Floor of ``|det(A)|``. ``None`` (default) uses ``PfaffianStrategy.EPSILON``.
        :return: The magnitude of shape ``(...,)`` in the real dtype of ``matrix``.
        :rtype: torch.Tensor
        """
        magnitude, _ = super().apply(matrix, epsilon)
        return magnitude
