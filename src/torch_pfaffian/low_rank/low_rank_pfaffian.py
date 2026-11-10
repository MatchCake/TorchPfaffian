import functools

import torch

from .skew_gradient import SkewGradient


class LowRankPfaffian:
    r"""
    Ratios of Pfaffians and inverse updates under skew-symmetric low-rank updates.

    For a skew-symmetric invertible ``A`` and ``u`` of shape ``(n, 2k)``, the updated matrix is

    .. math::
        A' = A - u J u^T, \qquad J = \begin{pmatrix} 0 & I_k \\ -I_k & 0 \end{pmatrix}

    (the convention of lrux, arXiv:2602.05255), and with the ``2k x 2k`` skew-symmetric matrix
    ``R = J + u^T A^{-1} u``,

    .. math::
        \frac{\mathrm{Pf}(A')}{\mathrm{Pf}(A)} = \frac{\mathrm{Pf}(R)}{\mathrm{Pf}(J)}, \qquad
        A'^{-1} = A^{-1} + (A^{-1} u) R^{-1} (A^{-1} u)^T.

    The ratio costs ``O(n^2 k)`` instead of the ``O(n^3)`` of a new Pfaffian, and so does the updated inverse
    (Woodbury identity), which is skew-symmetric up to round-off. A rank-2 update ``u = (x, y)`` adds
    ``y x^T - x y^T`` to ``A``; replacing one row and column (:meth:`ratio_replace`) costs ``O(n)`` for the
    ratio.

    Only tensor operations are used, so every method is batched, runs on any device without host
    synchronization, and is differentiable. Complex inputs use the transpose, not the conjugate transpose, as
    the Pfaffian is bilinear. ``float16``, ``bfloat16`` and ``complex32`` inputs are computed in ``float32`` or
    ``complex64`` with a ``UserWarning``, as in :func:`~torch_pfaffian.pfaffian`. When a ratio is zero, the
    updated matrix and ``R`` are singular: the returned inverse is then meaningless (huge or non-finite) and
    should be recomputed from the updated matrix when it is needed.

    Gradients are exact for any skew-symmetric construction of the inputs (e.g. from circuit parameters). The
    gradient with respect to a raw ``inverse`` leaf is projected onto skew-symmetric matrices
    (:class:`~torch_pfaffian.low_rank.SkewGradient`), the convention of :func:`~torch_pfaffian.pfaffian`. The
    gradients with respect to ``update``, ``change``, ``border`` and ``corner`` sit on the entries that are
    read: when they are slices of a larger skew-symmetric leaf matrix, only the skew-symmetric part of that
    matrix's gradient is meaningful.

    :class:`~torch_pfaffian.low_rank.PfaffianUpdater` chains such updates, with delayed inverse updates.
    """

    @staticmethod
    def skew_identity(
        rank: int, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None
    ) -> torch.Tensor:
        """
        Skew-symmetric identity ``J = [[0, I_k], [-I_k, 0]]`` of the low-rank Pfaffian updates.

        Its Pfaffian is ``(-1)^(k (k - 1) / 2)`` (see :meth:`skew_identity_pfaffian`) and its inverse is ``-J``.

        :param rank: Half the size ``k`` of the matrix, at least 1.
        :param dtype: Dtype of the result (PyTorch's default dtype when ``None``).
        :param device: Device of the result (the default device when ``None``).
        :return: The matrix ``J``, of shape ``(2k, 2k)``.
        :rtype: torch.Tensor
        :raises ValueError: If ``rank`` is smaller than 1.
        """
        if rank < 1:
            raise ValueError(f"rank must be at least 1; got {rank}.")
        identity = torch.eye(rank, dtype=dtype, device=device)
        zeros = torch.zeros_like(identity)
        return torch.cat([torch.cat([zeros, identity], dim=-1), torch.cat([-identity, zeros], dim=-1)], dim=-2)

    @staticmethod
    def skew_identity_pfaffian(rank: int) -> int:
        """
        Pfaffian of :meth:`skew_identity`, ``(-1)^(k (k - 1) / 2)``.

        :param rank: Half the size ``k`` of the matrix.
        :return: ``1`` or ``-1``.
        :rtype: int
        """
        return -1 if (rank * (rank - 1) // 2) % 2 else 1

    @staticmethod
    def ratio(inverse: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        """
        Ratio ``Pf(A - u J u^T) / Pf(A)`` from the inverse ``A^{-1}``.

        :param inverse: Inverse ``A^{-1}`` of the original skew-symmetric matrix, of shape ``(..., n, n)``.
        :param update: Update vectors ``u``, of shape ``(..., n, 2k)``; the batch dimensions broadcast with those
            of ``inverse``.
        :return: The ratio, of shape ``(...)``.
        :rtype: torch.Tensor
        :raises ValueError: If the shapes do not match or ``update`` has an odd or zero number of columns.
        :raises TypeError: If the dtypes or devices of the inputs differ, or the dtype is not supported.
        """
        inverse, update, dtype = LowRankPfaffian._prepare_update("LowRankPfaffian.ratio", inverse, update)
        ratio, _ = LowRankPfaffian._ratio_and_r_matrix(update, LowRankPfaffian.project(inverse, update))
        return ratio.to(dtype)

    @staticmethod
    def ratio_and_inverse(inverse: torch.Tensor, update: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Ratio ``Pf(A - u J u^T) / Pf(A)`` of :meth:`ratio` and the inverse ``(A - u J u^T)^{-1}``.

        :param inverse: Inverse ``A^{-1}`` of the original skew-symmetric matrix, of shape ``(..., n, n)``.
        :param update: Update vectors ``u``, of shape ``(..., n, 2k)``; the batch dimensions broadcast with those
            of ``inverse``.
        :return: The ratio, of shape ``(...)``, and the updated inverse, of shape ``(..., n, n)``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        :raises ValueError: If the shapes do not match or ``update`` has an odd or zero number of columns.
        :raises TypeError: If the dtypes or devices of the inputs differ, or the dtype is not supported.
        """
        inverse, update, dtype = LowRankPfaffian._prepare_update("LowRankPfaffian.ratio_and_inverse", inverse, update)
        projected = LowRankPfaffian.project(inverse, update)  # (..., n, 2k)
        ratio, r_matrix = LowRankPfaffian._ratio_and_r_matrix(update, projected)
        solved = torch.linalg.solve_ex(r_matrix, projected.mT)[0]  # (..., 2k, n), R^{-1} (A^{-1} u)^T
        return ratio.to(dtype), LowRankPfaffian.add_product(inverse, projected, solved).to(dtype)

    @staticmethod
    def ratio_replace(inverse: torch.Tensor, index: torch.Tensor | int, change: torch.Tensor) -> torch.Tensor:
        r"""
        Ratio of Pfaffians when one row and the matching column of a skew-symmetric matrix change.

        Row ``i`` of ``A`` becomes ``A[i] + c`` and column ``i`` becomes ``A[:, i] - c``, so that ``A'`` stays
        skew-symmetric; ``c_i`` is ignored (the diagonal stays zero). This is the rank-2 update ``u = (c, e_i)``
        of :meth:`ratio`, with a one-hot ``e_i``, for which

        .. math::
            r = \frac{\mathrm{Pf}(A')}{\mathrm{Pf}(A)} = 1 + c^T A^{-1} e_i.

        It reads one column of the inverse and costs ``O(n)``. To replace row ``i`` by a new row ``x`` (with
        ``x_i = 0``), pass ``change = x - A[i]``. :meth:`ratio_replace_and_inverse` also updates the inverse.

        :param inverse: Inverse ``A^{-1}`` of the original skew-symmetric matrix, of shape ``(..., n, n)``.
        :param index: Index ``i`` of the row and column that change, an integer or an integer tensor of shape
            ``(...)``.
        :param change: Change ``c`` of row ``i``, of shape ``(..., n)``.
        :return: The ratio, whose shape is the broadcast of the batch shapes of ``inverse``, ``index`` and
            ``change``.
        :rtype: torch.Tensor
        :raises ValueError: If the shapes do not match.
        :raises TypeError: If the dtypes or devices of the inputs differ, ``index`` is not an integer tensor, or
            the dtype is not supported.
        """
        prepared = LowRankPfaffian._prepare_replace("LowRankPfaffian.ratio_replace", inverse, index, change)
        _, change, column, dtype = prepared
        return (1 + (change * column).sum(dim=-1)).to(dtype)

    @staticmethod
    def ratio_replace_and_inverse(
        inverse: torch.Tensor, index: torch.Tensor | int, change: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Ratio of :meth:`ratio_replace` and the inverse of the updated matrix, in ``O(n^2)``:

        .. math::
            A'^{-1} = A^{-1} + \frac{(A^{-1} e_i)(A^{-1} c)^T - (A^{-1} c)(A^{-1} e_i)^T}{r}.

        :param inverse: Inverse ``A^{-1}`` of the original skew-symmetric matrix, of shape ``(..., n, n)``.
        :param index: Index ``i`` of the row and column that change, an integer or an integer tensor of shape
            ``(...)``.
        :param change: Change ``c`` of row ``i``, of shape ``(..., n)``.
        :return: The ratio, of the broadcast batch shape ``(...)``, and the updated inverse, of shape
            ``(..., n, n)``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        :raises ValueError: If the shapes do not match.
        :raises TypeError: If the dtypes or devices of the inputs differ, ``index`` is not an integer tensor, or
            the dtype is not supported.
        """
        prepared = LowRankPfaffian._prepare_replace("LowRankPfaffian.ratio_replace_and_inverse", inverse, index, change)
        inverse, change, column, dtype = prepared
        ratio = 1 + (change * column).sum(dim=-1)
        projected_change = LowRankPfaffian.project(inverse, change[..., None])[..., 0]  # (..., n), A^{-1} c
        scaled = (1 / ratio)[..., None]
        left = torch.stack([column, projected_change], dim=-1)  # (..., n, 2)
        right = torch.stack([projected_change * scaled, -column * scaled], dim=-2)  # (..., 2, n)
        return ratio.to(dtype), LowRankPfaffian.add_product(inverse, left, right).to(dtype)

    @staticmethod
    def ratio_extend(inverse: torch.Tensor, border: torch.Tensor, corner: torch.Tensor) -> torch.Tensor:
        r"""
        Ratio of Pfaffians when a skew-symmetric matrix grows by ``2k`` rows and columns.

        For the bordered matrix

        .. math::
            M = \begin{pmatrix} A & B \\ -B^T & C \end{pmatrix}, \qquad
            \frac{\mathrm{Pf}(M)}{\mathrm{Pf}(A)} = \mathrm{Pf}\left(C + B^T A^{-1} B\right),

        the Pfaffian of the ``2k x 2k`` Schur complement, in ``O(m^2 k)`` from ``A^{-1}``. This is the step of
        sequential (qubit-by-qubit) sampling of matchgate circuits, where a qubit adds the ``2 x 2`` block of its
        two Majorana modes: the conditional probability of an outcome is ``|ratio| / 2``. ``A`` may be empty
        (``m = 0``), when the ratio is ``Pf(C)``.

        Only ``C`` depends on the outcome, so several candidate corners are evaluated at the cost of one: give
        them a leading dimension, e.g. ``corner`` of shape ``(2, ..., 2k, 2k)`` for the outcomes 0 and 1, and the
        ratios have shape ``(2, ...)`` (``A^{-1} B`` is computed once, at the batch shape of ``inverse`` and
        ``border``). :class:`~torch_pfaffian.low_rank.PfaffianChain` keeps ``A^{-1}`` from one step to the next.

        :param inverse: Inverse ``A^{-1}`` of the current skew-symmetric matrix, of shape ``(..., m, m)``.
        :param border: Coupling ``B`` of the current rows to the new ones, of shape ``(..., m, 2k)``.
        :param corner: Skew-symmetric block ``C`` of the new rows and columns, of shape ``(..., 2k, 2k)``.
        :return: The ratio ``Pf(M) / Pf(A)``, of the broadcast batch shape ``(...)``.
        :rtype: torch.Tensor
        :raises ValueError: If the shapes do not match.
        :raises TypeError: If the dtypes or devices of the inputs differ, or the dtype is not supported.
        """
        inverse, border, corner, dtype = LowRankPfaffian._prepare_extend(
            "LowRankPfaffian.ratio_extend", inverse, border, corner
        )
        schur, _ = LowRankPfaffian._schur_complement(inverse, border, corner)
        return LowRankPfaffian._small_pfaffian(schur).to(dtype)

    @staticmethod
    def ratio_extend_and_inverse(
        inverse: torch.Tensor, border: torch.Tensor, corner: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Ratio of :meth:`ratio_extend` and the inverse of the bordered matrix, in ``O(m^2 k)``.

        With ``P = A^{-1} B`` and the Schur complement ``S = C + B^T A^{-1} B``,

        .. math::
            M^{-1} = \begin{pmatrix} A^{-1} + P S^{-1} P^T & -P S^{-1} \\ -S^{-1} P^T & S^{-1} \end{pmatrix}.

        :param inverse: Inverse ``A^{-1}`` of the current skew-symmetric matrix, of shape ``(..., m, m)``.
        :param border: Coupling ``B`` of the current rows to the new ones, of shape ``(..., m, 2k)``.
        :param corner: Skew-symmetric block ``C`` of the new rows and columns, of shape ``(..., 2k, 2k)``.
        :return: The ratio, of the broadcast batch shape ``(...)``, and ``M^{-1}``, of shape
            ``(..., m + 2k, m + 2k)`` (huge or non-finite when the ratio is zero).
        :rtype: tuple[torch.Tensor, torch.Tensor]
        :raises ValueError: If the shapes do not match.
        :raises TypeError: If the dtypes or devices of the inputs differ, or the dtype is not supported.
        """
        inverse, border, corner, dtype = LowRankPfaffian._prepare_extend(
            "LowRankPfaffian.ratio_extend_and_inverse", inverse, border, corner
        )
        schur, projected = LowRankPfaffian._schur_complement(inverse, border, corner)
        ratio = LowRankPfaffian._small_pfaffian(schur)
        schur_inverse = LowRankPfaffian._small_inverse(schur, ratio)
        batch_shape = torch.broadcast_shapes(inverse.shape[:-2], projected.shape[:-2], schur.shape[:-2])
        size, width = inverse.shape[-1], schur.shape[-1]
        top_right = -(projected @ schur_inverse)  # (..., m, 2k)
        top_left = LowRankPfaffian.add_product(inverse, projected, top_right.mT)  # A^{-1} + P S^{-1} P^T
        top = torch.cat(
            [top_left.expand(*batch_shape, size, size), top_right.expand(*batch_shape, size, width)], dim=-1
        )
        bottom = torch.cat(
            [-top_right.mT.expand(*batch_shape, width, size), schur_inverse.expand(*batch_shape, width, width)],
            dim=-1,
        )
        return ratio.to(dtype), torch.cat([top, bottom], dim=-2).to(dtype)

    @staticmethod
    def ratio_and_r_inverse(update: torch.Tensor, projected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Ratio of :meth:`ratio` and the inverse of its matrix ``R``, from the projection ``A^{-1} u``.

        :param update: Update vectors ``u``, of shape ``(..., n, 2k)``.
        :param projected: Projection ``A^{-1} u``, of shape ``(..., n, 2k)``.
        :return: The ratio, of shape ``(...)``, and ``R^{-1}``, of shape ``(..., 2k, 2k)`` (huge or non-finite
            when the ratio is zero).
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        ratio, r_matrix = LowRankPfaffian._ratio_and_r_matrix(update, projected)
        return ratio, LowRankPfaffian._small_inverse(r_matrix, ratio)

    @staticmethod
    def project(inverse: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        """
        Product ``A^{-1} v`` with a skew-symmetric ``A^{-1}``, computed as ``-(v^T A^{-1})^T``.

        The transposed product reads ``A^{-1}`` row by row and is about 3x faster on CPU for the skinny ``v`` of
        low-rank updates; the two agree up to the round-off in the skew-symmetry of ``A^{-1}``.

        :param inverse: Skew-symmetric matrix ``A^{-1}``, of shape ``(..., n, n)``.
        :param vectors: Vectors ``v``, of shape ``(..., n, p)``.
        :return: ``A^{-1} v``, of shape ``(..., n, p)``.
        :rtype: torch.Tensor
        """
        return -(vectors.mT @ inverse).mT

    @staticmethod
    def add_product(matrix: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """
        ``matrix + left @ right`` in one fused pass over ``matrix`` when the batch shapes allow it.

        The low-rank corrections of the inverse are skew-symmetric by construction, so the result stays
        skew-symmetric up to round-off without an explicit (and, at large ``n``, costly) symmetrization.

        :param matrix: Matrix of shape ``(..., n, n)``.
        :param left: Left factor, of shape ``(..., n, p)``.
        :param right: Right factor, of shape ``(..., p, n)``.
        :return: ``matrix + left @ right``, of shape ``(..., n, n)``.
        :rtype: torch.Tensor
        """
        batch_shape = matrix.shape[:-2]
        if left.shape[:-2] != batch_shape or right.shape[:-2] != batch_shape or matrix.numel() == 0:
            return matrix + left @ right
        if not matrix.is_contiguous() and matrix.mT.is_contiguous():
            # Column-major input, as torch.linalg.inv returns: (M + L R)^T = M^T + R^T L^T on the contiguous
            # transposed view avoids copying the n x n matrix, and the result keeps the input's layout.
            return LowRankPfaffian.add_product(matrix.mT, right.mT, left.mT).mT
        dimension = matrix.shape[-1]
        flat = torch.baddbmm(
            matrix.reshape(-1, dimension, dimension),
            left.reshape(-1, dimension, left.shape[-1]),
            right.reshape(-1, right.shape[-2], dimension),
        )
        return flat.reshape(matrix.shape)

    @staticmethod
    def as_index(index: torch.Tensor | int, device: torch.device) -> torch.Tensor:
        """
        Row index of :meth:`ratio_replace` as an ``int64`` tensor on ``device``.

        :param index: An integer or an integer tensor.
        :param device: Device of the matrices.
        :return: The index tensor.
        :rtype: torch.Tensor
        :raises TypeError: If ``index`` is not an integer.
        """
        index = torch.as_tensor(index, device=device)
        if index.dtype.is_floating_point or index.dtype.is_complex or index.dtype == torch.bool:
            raise TypeError(f"index must be an integer tensor; got {index.dtype}.")
        return index.long()

    @staticmethod
    @functools.lru_cache(maxsize=64)
    def _cached_skew_identity(rank: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        # J is a constant: building it costs five small kernels, more than the rest of a rank-2 ratio on CPU.
        return LowRankPfaffian.skew_identity(rank, dtype=dtype, device=device)

    @staticmethod
    def _ratio_and_r_matrix(update: torch.Tensor, projected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rank = update.shape[-1] // 2
        inner = update.mT @ projected  # (..., 2k, 2k), skew-symmetric up to round-off
        identity = LowRankPfaffian._cached_skew_identity(rank, update.dtype, update.device)
        r_matrix = identity + 0.5 * (inner - inner.mT)
        ratio = LowRankPfaffian._small_pfaffian(r_matrix)
        if LowRankPfaffian.skew_identity_pfaffian(rank) < 0:
            ratio = -ratio
        return ratio, r_matrix

    @staticmethod
    def _schur_complement(
        inverse: torch.Tensor, border: torch.Tensor, corner: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # S = C + B^T A^{-1} B and P = A^{-1} B; S is computed at the batch shape of A and B before adding C, so
        # that candidate corners in leading dimensions share it.
        projected = LowRankPfaffian.project(inverse, border)  # (..., m, 2k)
        inner = border.mT @ projected  # (..., 2k, 2k)
        schur = corner + inner
        return 0.5 * (schur - schur.mT), projected

    @staticmethod
    def _small_pfaffian(matrix: torch.Tensor) -> torch.Tensor:
        from .. import pfaffian

        if matrix.shape[-1] == 2:
            return matrix[..., 0, 1]
        # check_finite would synchronize with the host; the matrix is small and its Pfaffian cannot overflow.
        return pfaffian(matrix, check_finite=False)

    @staticmethod
    def _small_inverse(matrix: torch.Tensor, pfaffian_value: torch.Tensor) -> torch.Tensor:
        if matrix.shape[-1] == 2:
            # [[0, r], [-r, 0]]^{-1} = -[[0, r], [-r, 0]] / r^2
            return -matrix / (pfaffian_value * pfaffian_value)[..., None, None]
        return torch.linalg.inv_ex(matrix)[0]

    @staticmethod
    def _prepare_update(
        function_name: str, inverse: torch.Tensor, update: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.dtype]:
        LowRankPfaffian._check_inputs(function_name, inverse, update)
        if update.shape[-2] != inverse.shape[-1]:
            raise ValueError(f"update must have shape (..., n, 2k) with n = {inverse.shape[-1]}; got {update.shape}.")
        if update.shape[-1] == 0 or update.shape[-1] % 2:
            raise ValueError(f"update must have an even, nonzero number of columns 2k; got {update.shape[-1]}.")
        dtype = inverse.dtype
        compute_dtype = LowRankPfaffian._compute_dtype(function_name, dtype)
        inverse = SkewGradient.apply_if_needed(inverse)
        return inverse.to(compute_dtype), update.to(compute_dtype), dtype

    @staticmethod
    def _prepare_replace(
        function_name: str, inverse: torch.Tensor, index: torch.Tensor | int, change: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.dtype]:
        # Validated, upcast and broadcast inputs, the change with c_i = 0, and the column A^{-1} e_i.
        LowRankPfaffian._check_inputs(function_name, inverse, change)
        dimension = inverse.shape[-1]
        if change.shape[-1] != dimension:
            raise ValueError(f"change must have shape (..., n) with n = {dimension}; got {change.shape}.")
        index = LowRankPfaffian.as_index(index, inverse.device)
        dtype = inverse.dtype
        compute_dtype = LowRankPfaffian._compute_dtype(function_name, dtype)
        inverse, change = SkewGradient.apply_if_needed(inverse).to(compute_dtype), change.to(compute_dtype)
        batch_shape = torch.broadcast_shapes(inverse.shape[:-2], index.shape, change.shape[:-1])
        if change.shape[:-1] != batch_shape:
            change = change.expand(*batch_shape, dimension)
        if index.shape != batch_shape:
            index = index.expand(batch_shape)
        index = index[..., None]  # (..., 1)
        change = change.scatter(-1, index, 0)  # c_i does not change A (its diagonal stays zero)
        if inverse.shape[:-2] != batch_shape:
            inverse = inverse.expand(*batch_shape, dimension, dimension)
        column = torch.take_along_dim(inverse, index[..., None, :], dim=-1)[..., 0]  # (..., n), A^{-1} e_i
        return inverse, change, column, dtype

    @staticmethod
    def _prepare_extend(
        function_name: str, inverse: torch.Tensor, border: torch.Tensor, corner: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.dtype]:
        LowRankPfaffian._check_inputs(function_name, inverse, border)
        LowRankPfaffian._check_inputs(function_name, inverse, corner)
        size = inverse.shape[-1]
        if border.dim() < 2 or border.shape[-2] != size:
            raise ValueError(f"border must have shape (..., m, 2k) with m = {size}; got {tuple(border.shape)}.")
        width = border.shape[-1]
        if width == 0 or width % 2:
            raise ValueError(f"border must have an even, nonzero number of columns 2k; got {width}.")
        if corner.shape[-2:] != (width, width):
            raise ValueError(f"corner must have shape (..., {width}, {width}); got {tuple(corner.shape)}.")
        dtype = inverse.dtype
        compute_dtype = LowRankPfaffian._compute_dtype(function_name, dtype)
        inverse = SkewGradient.apply_if_needed(inverse)
        return inverse.to(compute_dtype), border.to(compute_dtype), corner.to(compute_dtype), dtype

    @staticmethod
    def _check_inputs(function_name: str, inverse: torch.Tensor, other: torch.Tensor) -> None:
        from .. import _check_dtype

        if inverse.dim() < 2 or inverse.shape[-1] != inverse.shape[-2]:
            raise ValueError(f"{function_name}: inverse must have shape (..., n, n); got {tuple(inverse.shape)}.")
        if other.dtype != inverse.dtype or other.device != inverse.device:
            raise TypeError(
                f"{function_name}: the inputs must share a dtype and a device; got {inverse.dtype} on "
                f"{inverse.device} and {other.dtype} on {other.device}."
            )
        _check_dtype(function_name, inverse.dtype)

    @staticmethod
    def _compute_dtype(function_name: str, dtype: torch.dtype) -> torch.dtype:
        from .. import LOW_PRECISION_DTYPES, _upcast_dtype, _warn_upcast

        if dtype not in LOW_PRECISION_DTYPES:
            return dtype
        _warn_upcast(function_name, dtype, stacklevel=4)
        return _upcast_dtype(dtype)
