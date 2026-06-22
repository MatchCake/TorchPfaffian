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

    @classmethod
    def pfaffian_grad_matrix(
        cls, matrix: torch.Tensor, pfaffian: torch.Tensor, grad_output: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Gradient of the signed Pfaffian with respect to the input matrix.

        Uses the closed form ``d pf(A) / d A = (1 / 2) pf(A) (A^{-1})^T`` via the Pfaffian adjugate
        ``pf(A) A^{-1}``. For invertible inputs the adjugate is ``pf(A) * inv(A)`` (a single inverse);
        for singular inputs (``pf == 0``), where that product would be ``0`` and miss the true
        derivative, the adjugate is recomputed exactly from minor Pfaffians via
        :meth:`_pfaffian_adjugate` (using ``cls``'s own forward). The minor-based path runs only on the
        singular batch elements, so invertible inputs keep the single cheap inverse.

        The inverse uses :func:`torch.linalg.inv` (an LU factorization) rather than
        :func:`torch.linalg.pinv` (an SVD). A skew-symmetric ``A`` is invertible exactly when
        ``pf(A) != 0`` (since ``det(A) = pf(A)^2``), so the inverse is only ever relied upon on the
        invertible elements, where the LU factorization is the correct and robust tool. The SVD-based
        pseudo-inverse can fail to converge on ill-conditioned or near-repeated-singular-value inputs,
        which the LU factorization does not. Because ``inv`` raises on an exactly-singular matrix, the
        ``pf == 0`` elements (whose inverse is discarded anyway) are replaced by the identity before the
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
        singular = pfaffian == 0
        dimension = matrix.shape[-1]
        any_singular = bool(singular.any())
        if any_singular:
            identity = torch.eye(dimension, dtype=matrix.dtype, device=matrix.device).expand_as(matrix)
            safe_matrix = torch.where(singular[..., None, None], identity, matrix)  # (..., n, n)
            inverse = torch.linalg.inv(safe_matrix)
        else:
            inverse = torch.linalg.inv(matrix)
        adjugate = pfaffian[..., None, None] * inverse  # pf(A) A^{-1}; 0 where pf == 0
        if any_singular:
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
