import torch

from .low_rank_pfaffian import LowRankPfaffian
from .skew_gradient import SkewGradient


class PfaffianUpdater:
    r"""
    Successive low-rank updates of a skew-symmetric matrix, with Pfaffian ratios and delayed inverse updates.

    The state is the inverse ``A_0^{-1}`` of the last merged matrix and, for each of the ``tau`` updates
    accepted since, the projection ``a_t = A_{t-1}^{-1} u_t`` and ``R_t^{-1}`` (see
    :class:`~torch_pfaffian.low_rank.LowRankPfaffian` for the convention ``A_t = A_{t-1} - u_t J u_t^T``). The
    current inverse is never formed between merges:

    .. math::
        A_\tau^{-1} u = A_0^{-1} u + \sum_{t=1}^{\tau} a_t R_t^{-1} (a_t^T u),

    so an update costs ``O(n^2 k + \tau n k^2)`` and does not rewrite the ``n x n`` inverse. After
    ``max_delay`` accepted updates the corrections are merged into the inverse. lrux (arXiv:2602.05255)
    suggests ``max_delay`` of about ``n / (20 k)``.

    Every operation is a tensor operation on the batch, without host synchronization, so a batch of Monte
    Carlo walkers is updated at once: :meth:`accept` takes a mask of the walkers whose move is accepted, and
    leaves the others unchanged. Gradients flow through the state like through any tensor computation.

    Round-off accumulates over many updates, mostly in single precision. :meth:`reset` recomputes the
    inverse from the current matrix (which the caller keeps); the running :attr:`pfaffian` is the Pfaffian
    of the initial matrix times the accepted ratios, and can over- or underflow over long runs.

    The input matrix is a skew-symmetric matrix of shape ``(..., n, n)``, in float32, float64, complex64 or
    complex128.
    """

    _max_delay: int
    _inverse: torch.Tensor  # A_0^{-1}, (..., n, n)
    _pfaffian: torch.Tensor  # (...)
    _projections: list[torch.Tensor]  # a_t, each (..., n, 2k_t)
    _r_inverses: list[torch.Tensor]  # R_t^{-1}, each (..., 2k_t, 2k_t)

    @staticmethod
    def _check_matrix(matrix: torch.Tensor) -> None:
        from .. import LU_DTYPES

        if matrix.dim() < 2 or matrix.shape[-1] != matrix.shape[-2]:
            raise ValueError(f"matrix must have shape (..., n, n); got {tuple(matrix.shape)}.")
        if matrix.dtype not in LU_DTYPES:
            names = ", ".join(str(dtype).removeprefix("torch.") for dtype in LU_DTYPES)
            raise TypeError(f"PfaffianUpdater supports {names}; got {matrix.dtype}.")

    @classmethod
    def from_inverse(
        cls, inverse: torch.Tensor, *, pfaffian: torch.Tensor | None = None, max_delay: int = 8
    ) -> "PfaffianUpdater":
        """
        Start from an inverse that is already known, without inverting a matrix or computing its Pfaffian.

        :param inverse: Inverse of the initial skew-symmetric matrix, of shape ``(..., n, n)``.
        :param pfaffian: Pfaffian of the initial matrix, of shape ``(...)``. When ``None``, :attr:`pfaffian`
            starts at 1 and is the product of the accepted ratios, i.e. the Pfaffian relative to the start.
        :param max_delay: Number of accepted updates after which they are merged into the inverse, at least 1.
        :return: The updater.
        :rtype: PfaffianUpdater
        :raises ValueError: If ``max_delay`` is smaller than 1 or ``inverse`` is not square.
        :raises TypeError: If the dtype of ``inverse`` is not supported.
        """
        updater = cls.__new__(cls)
        updater._setup(max_delay)
        cls._check_matrix(inverse)
        updater._inverse = SkewGradient.apply_if_needed(inverse)
        updater._pfaffian = torch.ones(inverse.shape[:-2], dtype=inverse.dtype, device=inverse.device)
        if pfaffian is not None:
            updater._pfaffian = updater._pfaffian * pfaffian
        return updater

    def __init__(self, matrix: torch.Tensor, *, max_delay: int = 8) -> None:
        """
        Start from ``matrix``, computing its inverse and its Pfaffian.

        :param matrix: Invertible skew-symmetric matrix of shape ``(..., n, n)``.
        :param max_delay: Number of accepted updates after which they are merged into the inverse, at least 1.
        :raises ValueError: If ``max_delay`` is smaller than 1 or ``matrix`` is not square.
        :raises TypeError: If the dtype of ``matrix`` is not supported.
        """
        self._setup(max_delay)
        self.reset(matrix)

    def ratio(self, update: torch.Tensor) -> torch.Tensor:
        """
        Ratio ``Pf(A - u J u^T) / Pf(A)`` for the current matrix ``A``, without changing the state.

        :param update: Update vectors ``u``, of shape ``(..., n, 2k)``.
        :return: The ratio, of shape ``(...)``.
        :rtype: torch.Tensor
        """
        self._check_update(update)
        return LowRankPfaffian.ratio_and_r_inverse(update, self._project(update))[0]

    def accept(self, update: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply the update ``A <- A - u J u^T`` to the batch elements selected by ``mask``.

        :param update: Update vectors ``u``, of shape ``(..., n, 2k)``.
        :param mask: Boolean tensor of shape ``(...)``, ``True`` where the update is applied; every element
            when ``None``.
        :return: The ratio of the applied update, of shape ``(...)``, equal to 1 where ``mask`` is ``False``.
        :rtype: torch.Tensor
        """
        self._check_update(update)
        if mask is not None:
            update = torch.where(mask[..., None, None], update, torch.zeros_like(update))
        projected = self._project(update)
        ratio, r_inverse = LowRankPfaffian.ratio_and_r_inverse(update, projected)
        self._projections.append(projected)
        self._r_inverses.append(r_inverse)
        self._pfaffian = self._pfaffian * ratio
        if len(self._projections) >= self._max_delay:
            self.merge()
        return ratio

    def ratio_replace(self, index: torch.Tensor | int, change: torch.Tensor) -> torch.Tensor:
        """
        Ratio for adding ``change`` to row ``index`` (and subtracting it from the column), without changing the
        state. See :meth:`~torch_pfaffian.low_rank.LowRankPfaffian.ratio_replace`.

        :param index: Index of the row and column, an integer or an integer tensor of shape ``(...)``.
        :param change: Change of the row, of shape ``(..., n)``.
        :return: The ratio, of shape ``(...)``.
        :rtype: torch.Tensor
        """
        return self.ratio(self._replace_update(index, change))

    def accept_replace(
        self, index: torch.Tensor | int, change: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Add ``change`` to row ``index`` and subtract it from column ``index``, where ``mask`` is ``True``.

        :param index: Index of the row and column, an integer or an integer tensor of shape ``(...)``.
        :param change: Change of the row, of shape ``(..., n)``.
        :param mask: Boolean tensor of shape ``(...)``, ``True`` where the change is applied; every element
            when ``None``.
        :return: The ratio of the applied change, of shape ``(...)``, equal to 1 where ``mask`` is ``False``.
        :rtype: torch.Tensor
        """
        return self.accept(self._replace_update(index, change), mask)

    def merge(self) -> None:
        """
        Fold the delayed updates into the stored inverse, ``A_0^{-1} <- A_tau^{-1}``.
        """
        self._inverse = self.inverse
        self._projections.clear()
        self._r_inverses.clear()

    def reset(self, matrix: torch.Tensor) -> None:
        """
        Restart from ``matrix``: recompute its inverse and Pfaffian and drop the delayed updates.

        Useful to bound the round-off accumulated over many updates, with the current matrix kept by the caller.

        :param matrix: Invertible skew-symmetric matrix of shape ``(..., n, n)``.
        :raises ValueError: If ``matrix`` is not square.
        :raises TypeError: If the dtype of ``matrix`` is not supported.
        """
        from .. import pfaffian

        self._check_matrix(matrix)
        inverse = torch.linalg.inv_ex(matrix)[0]  # no host synchronization; non-finite if matrix is singular
        self._inverse = 0.5 * (inverse - inverse.mT)
        self._pfaffian = pfaffian(matrix)
        self._projections.clear()
        self._r_inverses.clear()

    def _setup(self, max_delay: int) -> None:
        if max_delay < 1:
            raise ValueError(f"max_delay must be at least 1; got {max_delay}.")
        self._max_delay = max_delay
        self._projections = []
        self._r_inverses = []

    def _check_update(self, update: torch.Tensor) -> None:
        dimension = self._inverse.shape[-1]
        if update.shape[-2:-1] != (dimension,) or update.shape[-1] == 0 or update.shape[-1] % 2:
            raise ValueError(f"update must have shape (..., {dimension}, 2k) with k >= 1; got {tuple(update.shape)}.")
        if update.dtype != self._inverse.dtype or update.device != self._inverse.device:
            raise TypeError(
                f"update must have the dtype and device of the matrix, {self._inverse.dtype} on "
                f"{self._inverse.device}; got {update.dtype} on {update.device}."
            )

    def _project(self, update: torch.Tensor) -> torch.Tensor:
        projected = LowRankPfaffian.project(self._inverse, update)  # (..., n, 2k)
        for projection, r_inverse in zip(self._projections, self._r_inverses):
            projected = projected + projection @ (r_inverse @ (projection.mT @ update))  # O(n k^2)
        return projected

    def _replace_update(self, index: torch.Tensor | int, change: torch.Tensor) -> torch.Tensor:
        dimension = self._inverse.shape[-1]
        if change.shape[-1] != dimension:
            raise ValueError(f"change must have shape (..., {dimension}); got {tuple(change.shape)}.")
        index = LowRankPfaffian.as_index(index, change.device)
        batch_shape = torch.broadcast_shapes(change.shape[:-1], index.shape)
        change = change.expand(*batch_shape, dimension)
        index = index.expand(batch_shape)[..., None]  # (..., 1)
        change = change.scatter(-1, index, 0)  # the diagonal stays zero
        one_hot = torch.zeros_like(change).scatter(-1, index, 1)
        return torch.stack([change, one_hot], dim=-1)  # (..., n, 2), u = (c, e_i)

    @property
    def delay(self) -> int:
        """
        Number of accepted updates not yet merged into the stored inverse.

        :return: The number of delayed updates, between 0 and ``max_delay - 1``.
        :rtype: int
        """
        return len(self._projections)

    @property
    def inverse(self) -> torch.Tensor:
        """
        Inverse of the current matrix, with the delayed updates applied (the state is not changed).

        :return: The inverse, of shape ``(..., n, n)``.
        :rtype: torch.Tensor
        """
        if not self._projections:
            return self._inverse
        # One fused pass over the n x n inverse for all the delayed corrections: A_0^{-1} + a R^{-1} a^T, with
        # a = (a_1, ..., a_tau) and R^{-1} block diagonal.
        projections = torch.cat(self._projections, dim=-1)  # (..., n, sum 2k_t)
        right = torch.cat(
            [r_inverse @ projection.mT for projection, r_inverse in zip(self._projections, self._r_inverses)], dim=-2
        )  # (..., sum 2k_t, n)
        return LowRankPfaffian.add_product(self._inverse, projections, right)

    @property
    def max_delay(self) -> int:
        """
        Number of accepted updates after which they are merged into the stored inverse.

        :return: The maximum delay.
        :rtype: int
        """
        return self._max_delay

    @property
    def pfaffian(self) -> torch.Tensor:
        """
        Pfaffian of the current matrix: that of the matrix given to :meth:`reset` times the accepted ratios.

        :return: The Pfaffian, of shape ``(...)``.
        :rtype: torch.Tensor
        """
        return self._pfaffian
