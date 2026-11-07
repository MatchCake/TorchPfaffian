import torch


class PfaffianStrategy(torch.autograd.Function):
    EPSILON = 1e-30
    NAME = "PfaffianStrategy"
    # When True (default) the singular-element gradient is the exact minor-based Pfaffian adjugate,
    # which preserves the historical behavior at the cost of one host synchronization per backward.
    # When False the backward is fully sync-free: singular elements (pf == 0) get an exactly zero
    # gradient via torch.where, so no .any()/.item() branch and no minor loop run.
    EXACT_SINGULAR_GRAD = True
    # A batch element is routed to the exact minor-based adjugate when its LU factorization has a pivot
    # ratio min|U_ii| / max|U_ii| <= eps^0.75 (scale-free, so it does not depend on how the entries are
    # spread), or when the LU inverse fails its residual check ||A A^{-1} - I||_max > eps^0.5. Exponents
    # of the dtype eps keep both criteria dtype-adaptive (float64: ~1e-12 and ~1.5e-8; float32: ~6e-6
    # and ~3e-4).
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
        the flagged batch elements, so well-conditioned inputs keep the single cheap inverse. Flagged
        elements whose ``grad_output`` is exactly zero (for example the elements of a magnitude whose
        gradient vanishes at ``pf = 0``) get a zero gradient without the minor-based path, since the
        gradient is linear in ``grad_output``; this skip is disabled when ``grad_output`` itself requires
        grad (double backward, forward-mode products such as ``torch.autograd.functional.jvp``).

        An element is flagged by :meth:`_flagged_inverse` (the LU pivot ratio
        ``min|U_ii| / max|U_ii| <= eps^0.75``, a nonzero LU ``info``, or an inverse residual
        ``||A A^{-1} - I||_max > eps^0.5``), or when its forward Pfaffian is exactly ``0`` (which also
        catches a Pfaffian that underflowed on a well-conditioned matrix with tiny entries). An
        exactly-singular matrix has a round-off-sized LU pivot (and a round-off forward Pfaffian, rarely
        exactly ``0``), so the pivot ratio catches it, while the residual catches ill-conditioned elements
        whose pivots look benign. The pivot ratio is
        scale-free: unlike a test of ``|pf|`` against ``max|A_ij|^(n/2)``, it does not flag
        well-conditioned matrices whose entries span several orders of magnitude (e.g. block-diagonal
        matrices). Flagging a well-conditioned element only costs speed, never accuracy.

        The inverse is an LU factorization (:func:`torch.linalg.lu_factor_ex` and
        :func:`torch.linalg.lu_solve`, or :func:`torch.linalg.inv_ex` in double backward) rather
        than :func:`torch.linalg.pinv` (an SVD), which can fail to converge on ill-conditioned or
        near-repeated-singular-value inputs. A skew-symmetric ``A`` is
        invertible exactly when ``pf(A) != 0`` (since ``det(A) = pf(A)^2``), so the inverse is only
        relied upon on the unflagged elements. Flagged elements are solved against the identity factors
        instead, so the batched solve never divides by a zero pivot.

        The Pfaffian is holomorphic in the entries of ``A``, so for complex inputs the backward returns
        the conjugate of the analytic derivative, ``conj(d pf / d A) * grad_output``, which is PyTorch's
        Wirtinger convention for complex autograd. For real inputs the conjugation is a no-op, so real
        gradients are unchanged.

        All of the above assumes :attr:`EXACT_SINGULAR_GRAD` is ``True`` (the default). When it is
        ``False`` the backward takes a fully host-synchronization-free path: singular elements
        (``pf == 0``) receive an exactly zero gradient (the true Pfaffian-adjugate derivative there is
        nonzero only at corank exactly 2), skipping the ``.any()`` branch and the minor loop entirely.

        :param matrix: The saved input matrix of shape ``(..., n, n)``.
        :param pfaffian: The saved forward Pfaffian of shape ``(...,)``.
        :param grad_output: Gradient of the output with respect to the loss, of shape ``(...,)``.
        :return: Gradient of the input matrix, of shape ``(..., n, n)``.
        :rtype: torch.Tensor
        """
        dimension = matrix.shape[-1]
        if dimension == 0:
            return torch.zeros_like(matrix)  # pf of a 0x0 matrix is the constant 1; the gradient is empty
        if not cls.EXACT_SINGULAR_GRAD:
            # Sync-free path: singular elements (pf == 0) get an exactly zero gradient (multiplying the
            # inverse by pf zeroes it there), so no host branch and no minor loop are needed.
            singular = pfaffian == 0
            identity = torch.eye(dimension, dtype=matrix.dtype, device=matrix.device).expand_as(matrix)
            safe_matrix = torch.where(singular[..., None, None], identity, matrix)  # (..., n, n)
            inverse, _ = torch.linalg.inv_ex(safe_matrix)  # non-raising; discarded where singular
            adjugate = pfaffian[..., None, None] * inverse  # pf(A) A^{-1}; exactly zero where pf == 0
            return torch.einsum("...,...ij->...ji", 0.5 * grad_output, adjugate.conj())
        inverse, rank_flag, residual_flag = cls._flagged_inverse(matrix)
        singular = rank_flag | residual_flag | (pfaffian == 0)  # pf == 0 also catches an underflowed pf
        grad_output = grad_output.expand_as(pfaffian)
        adjugate = pfaffian[..., None, None] * inverse  # pf(A) A^{-1}; replaced where singular
        adjugate = torch.where(singular[..., None, None], torch.zeros_like(adjugate), adjugate)
        exact = singular & cls._grad_output_is_used(grad_output)  # elements needing the minor adjugate
        if bool(exact.any()):
            flat_matrix = matrix.reshape(-1, dimension, dimension)
            flat_adjugate = adjugate.reshape(-1, dimension, dimension)
            exact_index = exact.reshape(-1).nonzero(as_tuple=True)[0]
            with torch.no_grad():
                exact_adjugate = cls._pfaffian_adjugate(flat_matrix.index_select(0, exact_index))
            flat_adjugate = flat_adjugate.index_copy(0, exact_index, exact_adjugate.to(flat_adjugate.dtype))
            adjugate = flat_adjugate.reshape_as(matrix)
        return torch.einsum("...,...ij->...ji", 0.5 * grad_output, adjugate.conj())

    @classmethod
    def _flagged_inverse(cls, matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batched LU inverse together with the masks of elements whose inverse cannot be trusted.

        The rank flag marks elements whose LU factorization reports a zero pivot (``info != 0``) or whose
        pivot ratio ``min|U_ii| / max|U_ii|`` is at most ``eps^SINGULARITY_RTOL_EXPONENT`` (this also
        covers the all-zero matrix): numerically rank-deficient elements. The residual flag marks the
        remaining elements whose residual ``||A A^{-1} - I||_max`` exceeds
        ``eps^INVERSE_RESIDUAL_RTOL_EXPONENT`` (for example LU pivot growth). The flags are computed
        without autograd. Flagged elements are inverted as the identity so the inverse stays finite;
        when a graph is being built (double backward) the inverse comes from
        :func:`torch.linalg.inv_ex` on the identity-substituted matrix, whose derivative is finite
        everywhere, instead of from the LU factors.

        :param matrix: Square matrices of shape ``(..., k, k)`` with ``k >= 1``.
        :return: The inverse, of shape ``(..., k, k)`` (the identity where flagged), the rank flag and
            the residual flag, both boolean of shape ``(...,)``.
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        size = matrix.shape[-1]
        epsilon = torch.finfo(matrix.dtype).eps
        identity = torch.eye(size, dtype=matrix.dtype, device=matrix.device).expand_as(matrix)
        with torch.no_grad():
            factors, pivots, info = torch.linalg.lu_factor_ex(matrix)
            pivot_magnitude = factors.diagonal(dim1=-2, dim2=-1).abs()  # (..., k) |U_ii|
            ratio_threshold = epsilon**cls.SINGULARITY_RTOL_EXPONENT * pivot_magnitude.amax(dim=-1)
            rank_flag = (info != 0) | ~(pivot_magnitude.amin(dim=-1) > ratio_threshold)  # nan-safe
            identity_pivots = torch.arange(1, size + 1, dtype=pivots.dtype, device=pivots.device)
            safe_factors = torch.where(rank_flag[..., None, None], identity, factors)
            safe_pivots = torch.where(rank_flag[..., None], identity_pivots.expand_as(pivots), pivots)
            inverse = torch.linalg.lu_solve(safe_factors, safe_pivots, identity)
            safe_matrix = torch.where(rank_flag[..., None, None], identity, matrix)
            residual = (safe_matrix @ inverse - identity).abs().amax(dim=(-2, -1))  # (...,)
            residual_flag = ~rank_flag & ~(residual <= epsilon**cls.INVERSE_RESIDUAL_RTOL_EXPONENT)
        if torch.is_grad_enabled() and matrix.requires_grad:
            flagged = rank_flag | residual_flag
            inverse, _ = torch.linalg.inv_ex(torch.where(flagged[..., None, None], identity, matrix))
        return inverse, rank_flag, residual_flag

    @staticmethod
    def _grad_output_is_used(grad_output: torch.Tensor) -> torch.Tensor:
        """
        Mask of the elements whose gradient depends on ``grad_output``.

        The gradient is linear in ``grad_output``, so an element with ``grad_output == 0`` has a zero
        gradient, unless ``grad_output`` requires grad (double backward, ``jvp``), where the derivative
        with respect to ``grad_output`` is needed too.

        :param grad_output: Gradient of the loss with respect to the Pfaffian, of shape ``(...,)``.
        :return: Boolean mask of shape ``(...,)``.
        :rtype: torch.Tensor
        """
        if grad_output.requires_grad:
            return torch.ones_like(grad_output, dtype=torch.bool)
        return grad_output != 0

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
