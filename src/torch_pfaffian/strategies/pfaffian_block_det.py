from typing import cast

import torch

from .strategy import PfaffianStrategy


class PfaffianBlockDet(PfaffianStrategy):
    r"""
    Pfaffian of a block-antidiagonal skew-symmetric matrix from the determinant of its off-diagonal block.

    For a matrix of the form

    .. math::
        A = \begin{pmatrix} 0 & B \\ -B^T & 0 \end{pmatrix}, \qquad
        \text{pf}(A) = (-1)^{n(n-1)/2} \det(B),

    with ``B`` the upper-right ``n x n`` block, a single ``n x n`` determinant gives the signed Pfaffian.
    This identity holds only for that block form: on a general skew-symmetric matrix the formula returns
    a wrong value. The forward therefore validates the input with :meth:`is_block_form` (both diagonal
    blocks exactly zero, even dimension) and raises ``ValueError`` otherwise. The check reads both
    diagonal blocks and costs up to three host synchronizations; set :attr:`CHECK_BLOCK_FORM` to ``False``,
    or pass ``False`` as the second positional argument of :meth:`apply` (``apply(matrix, False)``), to skip
    it for inputs known to have the block form.

    The backward returns the same gradient convention as the general strategies,
    ``(1 / 2) pf(A) (A^{-1})^T`` on the full matrix, split between the upper-right and lower-left blocks.
    """

    NAME = "PfaffianBlockDet"
    CHECK_BLOCK_FORM = True
    # Rank-deficient blocks of at least this size get their exact cofactor matrix from one SVD (O(n^3));
    # smaller blocks use the n^2 cofactor determinants (O(n^5), cheap there and componentwise accurate).
    SVD_COFACTOR_MIN_BLOCK_SIZE = 17

    @staticmethod
    def is_block_form(matrix: torch.Tensor) -> bool:
        """
        Whether every matrix of the batch has the form ``[[0, B], [-B^T, 0]]`` with square ``B``.

        Only the structure is checked (square, even dimension, both diagonal blocks exactly zero); the
        skew-symmetry relation between the two off-diagonal blocks is assumed, as for every strategy. The
        entry ``(0, 1)`` is tested first so a general skew-symmetric input is rejected after reading a
        single entry per batch element.

        :param matrix: Matrix of shape ``(..., m, m)``.
        :return: ``True`` when all batch elements have the block-antidiagonal form.
        :rtype: bool
        """
        if matrix.ndim < 2:
            return False
        dimension = matrix.shape[-1]
        if matrix.shape[-2] != dimension or dimension % 2 != 0:
            return False
        n = dimension // 2
        if n >= 2 and bool(matrix[..., 0, 1].any()):
            return False
        return not (bool(matrix[..., :n, :n].any()) or bool(matrix[..., n:, n:].any()))

    @staticmethod
    def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
        ctx.save_for_backward(inputs[0], output)

    @staticmethod
    def forward(matrix: torch.Tensor, check_block_form: bool | None = None) -> torch.Tensor:
        """
        Compute the Pfaffian of a block-antidiagonal skew-symmetric matrix from its upper-right block.

        :param matrix: Skew-symmetric matrix ``[[0, B], [-B^T, 0]]`` of shape ``(..., 2n, 2n)``.
        :param check_block_form: Whether to validate the block form. ``None`` (default) uses
            :attr:`CHECK_BLOCK_FORM`.
        :return: The signed Pfaffian, of shape ``(...,)``.
        :rtype: torch.Tensor
        :raises ValueError: If the check is enabled and ``matrix`` does not have the block form.
        """
        if check_block_form is None:
            check_block_form = PfaffianBlockDet.CHECK_BLOCK_FORM
        if check_block_form and not PfaffianBlockDet.is_block_form(matrix):
            raise ValueError(
                "PfaffianBlockDet requires a skew-symmetric matrix of the form [[0, B], [-B^T, 0]] (even "
                f"dimension, both diagonal blocks exactly zero), got shape {tuple(matrix.shape)} without that "
                "form. Use torch_pfaffian.pfaffian for general skew-symmetric matrices."
            )
        n = matrix.shape[-1] // 2
        block = matrix[..., :n, n:]  # (..., n, n) upper-right block
        return (-1) ** (n * (n - 1) // 2) * torch.linalg.det(block)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.BackwardCFunction, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, None]:
        r"""
        Compute the gradient of the Pfaffian with respect to the full input matrix.

        With ``pf = c * det(B)``, ``c = (-1)^{n(n-1)/2}`` and ``B`` the upper-right block, Jacobi's formula
        gives ``d pf / d B = c * adj(B)^T`` where ``adj(B) = det(B) B^{-1}`` is the classical adjugate. For
        the block form, ``A^{-1} = [[0, -B^{-T}], [B^{-1}, 0]]``, so the general-strategy gradient
        ``(1 / 2) pf(A) (A^{-1})^T`` is ``(1 / 2) c adj(B)^T`` in the upper-right block, its negated
        transpose in the lower-left block, and zero in the diagonal blocks. Returning that split (rather
        than the whole derivative in the upper-right block) makes ``A.grad`` independent of which strategy
        computed the Pfaffian; the gradient with respect to ``B`` of ``A = [[0, B], [-B^T, 0]]`` is the
        same either way.

        For an invertible block the adjugate is the single inverse ``pf * (B^{-1})^T``. For a (numerically)
        singular block that product is inaccurate or garbage, yet the true derivative ``c * adj(B)^T`` is
        finite and generally nonzero. The adjugate is then computed exactly by
        :meth:`_exact_cofactor_matrix` (from an SVD, with the cofactor determinants as fallback), on the
        flagged batch elements only so invertible inputs keep the single cheap inverse. A block is flagged
        by :meth:`PfaffianStrategy._flagged_inverse`, the same dtype-adaptive criteria as
        :meth:`PfaffianStrategy.pfaffian_grad_matrix` (LU pivot ratio, LU ``info`` and inverse residual;
        an exactly singular block can have a round-off nonzero forward determinant, so ``pf == 0`` alone
        misses it). Flagging a well-conditioned block only costs speed, never accuracy. Flagged blocks
        whose ``grad_output`` is exactly zero get a zero gradient without the exact path, since the
        gradient is linear in ``grad_output``. With :attr:`EXACT_SINGULAR_GRAD` set to ``False`` the
        backward is host-synchronization-free and blocks with ``pf == 0`` get a zero gradient.

        The Pfaffian is holomorphic in the entries of ``A``, so for complex inputs the backward returns the
        conjugate of the analytic derivative times ``grad_output``, which is PyTorch's Wirtinger convention.
        For real inputs the conjugation is a no-op.

        :param ctx: Context holding the saved input matrix and the forward Pfaffian.
        :param grad_output: Gradient of the output with respect to the loss.
        :return: Gradient of the input matrix (or ``None`` when the input does not require grad), and
            ``None`` for the ``check_block_form`` flag.
        :rtype: tuple[torch.Tensor | None, None]
        """
        matrix, pf = cast("tuple[torch.Tensor, torch.Tensor]", ctx.saved_tensors)
        if not ctx.needs_input_grad[0]:
            return None, None
        n = matrix.shape[-1] // 2
        if n == 0:
            return torch.zeros_like(matrix), None  # pf of a 0x0 matrix is the constant 1
        block = matrix[..., :n, n:]  # (..., n, n) upper-right block
        constant = (-1) ** (n * (n - 1) // 2)
        if not PfaffianBlockDet.EXACT_SINGULAR_GRAD:
            # Sync-free path: multiplying the inverse by pf zeroes the gradient at singular blocks
            # (pf == 0) exactly, so no host branch runs.
            identity = torch.eye(n, dtype=matrix.dtype, device=matrix.device).expand_as(block)
            singular = pf == 0
            safe_block = torch.where(singular[..., None, None], identity, block)
            inverse, _ = torch.linalg.inv_ex(safe_block)
            adjugate_transpose = pf[..., None, None] * inverse.transpose(-1, -2)  # zero where pf == 0
            return PfaffianBlockDet._skew_split_grad(matrix, grad_output, adjugate_transpose), None
        inverse, rank_flag, residual_flag = PfaffianBlockDet._flagged_inverse(block)
        singular = rank_flag | residual_flag | (pf == 0)  # pf == 0 also catches an underflowed det
        grad_output = grad_output.expand_as(pf)
        adjugate_transpose = pf[..., None, None] * inverse.transpose(-1, -2)  # replaced where singular
        adjugate_transpose = torch.where(
            singular[..., None, None], torch.zeros_like(adjugate_transpose), adjugate_transpose
        )
        exact = singular & PfaffianBlockDet._grad_output_is_used(grad_output)  # blocks needing the cofactor
        if bool(exact.any()):
            flat_block = block.reshape(-1, n, n)
            flat_adjugate = adjugate_transpose.reshape(-1, n, n)
            exact_index = exact.reshape(-1).nonzero(as_tuple=True)[0]
            # The SVD cofactor is used only for large rank-deficient blocks and only outside double
            # backward: its derivative is unstable at repeated singular values, and small blocks or blocks
            # flagged by the residual alone (LU pivot growth) are more accurate with the determinants.
            use_svd = n >= PfaffianBlockDet.SVD_COFACTOR_MIN_BLOCK_SIZE and not (
                torch.is_grad_enabled() and block.requires_grad
            )
            svd_mask = rank_flag.reshape(-1).index_select(0, exact_index) & use_svd
            cofactor = PfaffianBlockDet._exact_cofactor_matrix(flat_block.index_select(0, exact_index), svd_mask)
            flat_adjugate = flat_adjugate.index_copy(0, exact_index, (constant * cofactor).to(flat_adjugate.dtype))
            adjugate_transpose = flat_adjugate.reshape_as(block)
        return PfaffianBlockDet._skew_split_grad(matrix, grad_output, adjugate_transpose), None

    @staticmethod
    def _skew_split_grad(
        matrix: torch.Tensor, grad_output: torch.Tensor, adjugate_transpose: torch.Tensor
    ) -> torch.Tensor:
        """
        Assemble the full-matrix gradient from ``d pf / d B = c * adj(B)^T``.

        :param matrix: The saved input matrix of shape ``(..., 2n, 2n)``.
        :param grad_output: Gradient of the loss with respect to the Pfaffian, of shape ``(...,)``.
        :param adjugate_transpose: ``c * adj(B)^T`` of shape ``(..., n, n)``.
        :return: Gradient with ``(1 / 2) conj(c adj(B)^T) grad_output`` in the upper-right block, its
            negated transpose in the lower-left block and zeros elsewhere, of shape ``(..., 2n, 2n)``.
        :rtype: torch.Tensor
        """
        n = matrix.shape[-1] // 2
        block_grad = 0.5 * grad_output[..., None, None] * adjugate_transpose.conj()  # (..., n, n)
        grad_matrix = torch.zeros_like(matrix)
        grad_matrix[..., :n, n:] = block_grad
        grad_matrix[..., n:, :n] = -block_grad.transpose(-1, -2)
        return grad_matrix

    @staticmethod
    def _exact_cofactor_matrix(blocks: torch.Tensor, svd_mask: torch.Tensor) -> torch.Tensor:
        """
        Cofactor matrix ``C = adj(B)^T`` of a batch of square blocks, finite and exact at singular blocks.

        Blocks selected by ``svd_mask`` use :meth:`_svd_cofactor_matrix` (one SVD per block, ``O(n^3)``;
        its relative error grows with the condition number of the rank-``(n-1)`` part of the block),
        falling back to the cofactor determinants of
        :meth:`_cofactor_matrix` (``O(n^5)``) if the SVD fails to converge. The other blocks use the
        cofactor determinants directly, which are differentiable everywhere and componentwise accurate
        under LU pivot growth.

        :param blocks: Square blocks of shape ``(m, n, n)``.
        :param svd_mask: Boolean mask of shape ``(m,)`` selecting the blocks computed from the SVD.
        :return: The cofactor matrix of shape ``(m, n, n)``.
        :rtype: torch.Tensor
        """
        cofactor = torch.zeros_like(blocks)
        svd_index = svd_mask.nonzero(as_tuple=True)[0]
        determinant_index = (~svd_mask).nonzero(as_tuple=True)[0]
        if svd_index.numel() > 0:
            svd_blocks = blocks.index_select(0, svd_index)
            try:
                with torch.no_grad():
                    svd_cofactor = PfaffianBlockDet._svd_cofactor_matrix(svd_blocks)
            except torch.linalg.LinAlgError:
                svd_cofactor = PfaffianBlockDet._cofactor_matrix(svd_blocks)
            cofactor = cofactor.index_copy(0, svd_index, svd_cofactor)
        if determinant_index.numel() > 0:
            determinant_cofactor = PfaffianBlockDet._cofactor_matrix(blocks.index_select(0, determinant_index))
            cofactor = cofactor.index_copy(0, determinant_index, determinant_cofactor)
        return cofactor

    @staticmethod
    def _svd_cofactor_matrix(blocks: torch.Tensor) -> torch.Tensor:
        r"""
        Cofactor matrix ``C = adj(B)^T`` of a batch of square blocks from their SVD.

        With ``B = U \Sigma V^H``, ``adj(B) = \det(B) B^{-1} = \det(U) \det(V^H) V D U^H`` where
        ``D = diag(prod_{j != i} sigma_j)``. The products omit one singular value each, so the formula
        has no division and holds at singular ``B`` by continuity (``D`` has one nonzero entry at corank 1
        and vanishes at corank 2 or more). The products are built from prefix and suffix cumulative
        products.

        :param blocks: Square blocks of shape ``(m, n, n)`` with ``n >= 1``.
        :return: The cofactor matrix of shape ``(m, n, n)``.
        :rtype: torch.Tensor
        :raises torch.linalg.LinAlgError: If the SVD fails to converge.
        """
        left, singular_values, right_adjoint = torch.linalg.svd(blocks)
        ones = torch.ones_like(singular_values[..., :1])
        prefix = torch.cumprod(torch.cat([ones, singular_values[..., :-1]], dim=-1), dim=-1)  # prod_{j<i}
        reversed_values = singular_values.flip(-1)
        suffix = torch.cumprod(torch.cat([ones, reversed_values[..., :-1]], dim=-1), dim=-1).flip(-1)  # j>i
        products = (prefix * suffix).to(blocks.dtype)  # (m, n) prod_{j != i} sigma_j
        phase = torch.linalg.det(left) * torch.linalg.det(right_adjoint)  # (m,) unit modulus
        adjugate = phase[..., None, None] * (right_adjoint.mH * products[..., None, :]) @ left.mH
        return adjugate.transpose(-1, -2)

    @staticmethod
    def _cofactor_matrix(blocks: torch.Tensor) -> torch.Tensor:
        r"""
        Cofactor matrix ``C`` of a batch of square blocks, with ``C_{ij} = (-1)^{i+j} det(B^{(ij)})``
        where ``B^{(ij)}`` is ``B`` with row ``i`` and column ``j`` removed.

        This equals the transpose of the classical adjugate ``adj(B) = det(B) B^{-1}``, so it stays
        finite when ``B`` is singular and an inverse would fail. The minor determinants are batched, so
        the cost is ``n^2`` determinant evaluations regardless of the batch size.

        :param blocks: Square blocks of shape ``(m, n, n)``.
        :return: The cofactor matrix of shape ``(m, n, n)``.
        :rtype: torch.Tensor
        """
        n = blocks.shape[-1]
        cofactor = torch.zeros_like(blocks)
        indices = torch.arange(n, device=blocks.device)
        for row in range(n):
            kept_rows = indices[indices != row]
            for column in range(n):
                kept_columns = indices[indices != column]
                minor = blocks.index_select(-2, kept_rows).index_select(-1, kept_columns)  # (m, n-1, n-1)
                sign = 1.0 if (row + column) % 2 == 0 else -1.0
                cofactor[..., row, column] = sign * torch.linalg.det(minor)
        return cofactor

    @classmethod
    def _pfaffian_adjugate(cls, matrices: torch.Tensor) -> torch.Tensor:
        # This strategy's forward only accepts block-antidiagonal matrices, but the Pfaffian minors
        # are general skew matrices, so the adjugate is computed with the general Parlett-Reid forward.
        from .pfaffian_parlett_reid import PfaffianParlettReid

        return PfaffianParlettReid._pfaffian_adjugate(matrices)
