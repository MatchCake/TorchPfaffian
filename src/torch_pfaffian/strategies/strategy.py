import math

import torch


class PfaffianStrategy(torch.autograd.Function):
    EPSILON = 1e-30
    NAME = "PfaffianStrategy"
    # A batch element is routed to the exact minor-based adjugate when |pf| <= eps^0.75 * scale^(n/2)
    # (relative to the entry scale, since pf is a degree-n/2 polynomial in the entries), or when the
    # LU inverse fails its residual check ||A A^{-1} - I||_max > eps^0.5. Exponents of the dtype eps
    # keep both criteria dtype-adaptive (float64: ~1e-12 and ~1.5e-8; float32: ~6e-6 and ~3e-4).
    SINGULARITY_RTOL_EXPONENT = 0.75
    INVERSE_RESIDUAL_RTOL_EXPONENT = 0.5

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

    @classmethod
    def pfaffian_grad_matrix(
        cls, matrix: torch.Tensor, pfaffian: torch.Tensor, grad_output: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Gradient of the signed Pfaffian with respect to the input matrix.

        Uses the closed form ``d pf(A) / d A = (1 / 2) pf(A) (A^{-1})^T`` via the Pfaffian adjugate
        ``pf(A) A^{-1}``. For well-conditioned inputs the adjugate is ``pf(A) * inv(A)`` (a single
        inverse); for (numerically) singular inputs, where that product would be inaccurate or
        garbage, the adjugate is recomputed exactly from minor Pfaffians via
        :meth:`_pfaffian_adjugate` (using ``cls``'s own forward). The minor-based path runs only on
        the flagged batch elements, so well-conditioned inputs keep the single cheap inverse.

        An element is flagged singular by two dtype-adaptive criteria (see the class constants):
        the relative magnitude test ``|pf| <= eps^0.75 * scale^(n/2)`` with ``scale`` the largest
        entry magnitude (an exactly-singular matrix has a forward Pfaffian of round-off size, never
        exactly ``0``, so an exact ``pf == 0`` test would route it to the LU inverse, which raises
        on real inputs and silently returns garbage on complex inputs), and a residual check
        ``||A A^{-1} - I||_max > eps^0.5`` that catches ill-conditioned elements whose Pfaffian is
        not small (for example one tiny and one huge singular-value pair). The magnitude test is
        evaluated in log space so ``scale^(n/2)`` cannot overflow. Both tests only ever move
        elements to the exact minor-based path, so flagging a well-conditioned element costs speed,
        never accuracy.

        The inverse uses :func:`torch.linalg.inv` (an LU factorization) rather than
        :func:`torch.linalg.pinv` (an SVD). A skew-symmetric ``A`` is invertible exactly when
        ``pf(A) != 0`` (since ``det(A) = pf(A)^2``), so the inverse is only ever relied upon on the
        invertible elements, where the LU factorization is the correct and robust tool. The SVD-based
        pseudo-inverse can fail to converge on ill-conditioned or near-repeated-singular-value inputs,
        which the LU factorization does not. Because ``inv`` raises on an exactly-singular matrix, the
        flagged elements (whose inverse is discarded anyway) are replaced by the identity before the
        batched inverse so the call stays well-posed.

        The Pfaffian is holomorphic in the entries of ``A``, so for complex inputs the backward returns
        the conjugate of the analytic derivative, ``conj(d pf / d A) * grad_output``, which is PyTorch's
        Wirtinger convention for complex autograd (``z.grad = d L / d conj(z)``). For real inputs the
        conjugation is a no-op, so real gradients are unchanged.

        :param matrix: The saved input matrix of shape ``(..., n, n)``.
        :param pfaffian: The saved forward Pfaffian of shape ``(...,)``.
        :param grad_output: Gradient of the output with respect to the loss, of shape ``(...,)``.
        :return: Gradient of the input matrix, of shape ``(..., n, n)``.
        :rtype: torch.Tensor
        """
        dimension = matrix.shape[-1]
        epsilon = torch.finfo(matrix.dtype).eps
        entry_scale = matrix.abs().amax(dim=(-2, -1))  # (...,)
        log_threshold = (dimension // 2) * torch.log(entry_scale) + cls.SINGULARITY_RTOL_EXPONENT * math.log(epsilon)
        singular = (entry_scale == 0) | (torch.log(pfaffian.abs()) <= log_threshold)  # log(0) = -inf is covered
        identity = torch.eye(dimension, dtype=matrix.dtype, device=matrix.device).expand_as(matrix)
        safe_matrix = torch.where(singular[..., None, None], identity, matrix)  # (..., n, n)
        inverse = torch.linalg.inv(safe_matrix)
        residual = (safe_matrix @ inverse - identity).abs().amax(dim=(-2, -1))  # (...,)
        singular = singular | (residual > epsilon**cls.INVERSE_RESIDUAL_RTOL_EXPONENT)
        adjugate = pfaffian[..., None, None] * inverse  # pf(A) A^{-1}; discarded where singular
        if bool(singular.any()):
            flat_matrix = matrix.reshape(-1, dimension, dimension)
            flat_adjugate = adjugate.reshape(-1, dimension, dimension)
            singular_index = singular.reshape(-1).nonzero(as_tuple=True)[0]
            with torch.no_grad():
                singular_adjugate = cls._pfaffian_adjugate(flat_matrix.index_select(0, singular_index))
            flat_adjugate = flat_adjugate.index_copy(0, singular_index, singular_adjugate.to(flat_adjugate.dtype))
            adjugate = flat_adjugate.reshape_as(matrix)
        return torch.einsum("...,...ij->...ji", 0.5 * grad_output, adjugate.conj())

    @classmethod
    def _pfaffian_adjugate(cls, matrices: torch.Tensor) -> torch.Tensor:
        r"""
        Pfaffian adjugate ``P = pf(A) A^{-1}`` of a batch of skew-symmetric matrices.

        The adjugate is a polynomial in the entries of ``A`` (it equals ``pf(A) A^{-1}`` for invertible
        ``A`` but stays finite when ``A`` is singular), so it is computed from minor Pfaffians rather
        than an inverse: ``P_{ij} = (-1)^{i+j} pf(A^{(ij)})`` for ``i < j``, where ``A^{(ij)}`` is ``A``
        with rows and columns ``i`` and ``j`` removed, and ``P`` is skew-symmetric. The minor Pfaffians
        are computed with this class's own :meth:`forward`; subclasses whose forward is not valid on the
        minors (e.g. :class:`PfaffianBlockDet`) override this to use a general strategy.

        :param matrices: Skew-symmetric matrices of shape ``(m, n, n)``.
        :return: The Pfaffian adjugate, of shape ``(m, n, n)``.
        :rtype: torch.Tensor
        """
        dimension = matrices.shape[-1]
        adjugate = torch.zeros_like(matrices)
        indices = torch.arange(dimension, device=matrices.device)
        for first in range(dimension):
            for second in range(first + 1, dimension):
                keep = indices[(indices != first) & (indices != second)]
                minor = matrices.index_select(-2, keep).index_select(-1, keep)  # (m, n-2, n-2)
                minor_pfaffian = cls.forward(minor)  # (m,)
                sign = 1.0 if (first + second) % 2 == 0 else -1.0
                adjugate[..., first, second] = sign * minor_pfaffian
                adjugate[..., second, first] = -sign * minor_pfaffian
        return adjugate
