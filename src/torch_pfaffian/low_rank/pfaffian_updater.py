from collections.abc import Callable
from typing import Any

import torch

from .compiled_low_rank_pfaffian import CompiledLowRankPfaffian
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
        A_\tau^{-1} u = A_0^{-1} u + \sum_{t=1}^{\tau} a_t R_t^{-1} (a_t^T u) = A_0^{-1} u + P D (P^T u),

    with the projections side by side, ``P = (a_1, ..., a_tau)``, and the block-diagonal ``D`` of the
    ``R_t^{-1}``, so an update costs ``O(n^2 k + \tau n k^2)`` and does not rewrite the ``n x n`` inverse. After
    ``max_delay`` accepted updates the corrections are merged into the inverse. lrux (arXiv:2602.05255)
    suggests ``max_delay`` of about ``n / (20 k)``.

    Every operation is a tensor operation on the batch, without host synchronization, so a batch of Monte
    Carlo walkers is updated at once: :meth:`accept` takes a mask of the walkers whose move is accepted, and
    leaves the others unchanged. Gradients flow through the state like through any tensor computation.

    With ``compiled=True``, each step runs through ``torch.compile`` (see
    :class:`~torch_pfaffian.low_rank.CompiledLowRankPfaffian`): a few seconds of compilation on the first steps,
    then less per-call overhead on small matrices.

    On CPU without a graph to build, accepted moves are applied to the inverse at once, in place, by the Rust
    kernels of :class:`~torch_pfaffian.low_rank.LowRankPfaffian` instead of being delayed, which is faster;
    :attr:`delay` then stays 0. The results are the same up to round-off. The updater copies an inverse it did not
    compute (:meth:`from_inverse`) or that it returned (:attr:`inverse`) before updating it, so tensors held by
    the caller are never changed.

    Round-off accumulates over many updates, mostly in single precision. :meth:`reset` recomputes the
    inverse from the current matrix (which the caller keeps); the running :attr:`pfaffian` is the Pfaffian
    of the initial matrix times the accepted ratios, and can over- or underflow over long runs.

    The input matrix is a skew-symmetric matrix of shape ``(..., n, n)``, in float32, float64, complex64 or
    complex128.
    """

    _max_delay: int
    _backend: type[LowRankPfaffian]
    _inverse: torch.Tensor  # A_0^{-1}, (..., n, n)
    _pfaffian: torch.Tensor  # (...)
    _projections: torch.Tensor  # P = (a_1, ..., a_tau), (..., n, W) with W = sum 2k_t
    _r_inverses: torch.Tensor  # D = diag(R_1^{-1}, ..., R_tau^{-1}), (..., W, W)
    _delay: int
    _owns_inverse: bool
    # Inverses with at most this many entries (batch * n^2) take one native immediate update per accepted move
    # on CPU without a graph to build, instead of delaying it (see _accept_in_place). The in-place kernels beat
    # the delayed updates at every measured size, so the bound is only a switch (the tests set it to -1).
    IMMEDIATE_MAX_NUMEL = 2**62

    @staticmethod
    def _check_matrix(matrix: torch.Tensor) -> None:
        from .. import LU_DTYPES

        if matrix.dim() < 2 or matrix.shape[-1] != matrix.shape[-2]:
            raise ValueError(f"matrix must have shape (..., n, n); got {tuple(matrix.shape)}.")
        if matrix.dtype not in LU_DTYPES:
            names = ", ".join(str(dtype).removeprefix("torch.") for dtype in LU_DTYPES)
            raise TypeError(f"PfaffianUpdater supports {names}; got {matrix.dtype}.")

    @staticmethod
    def _step_core(
        inverse: torch.Tensor, projections: torch.Tensor, r_inverses: torch.Tensor, update: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # A_tau^{-1} u = A_0^{-1} u + P D (P^T u), then the ratio and R^{-1} of the step.
        projected = LowRankPfaffian.project(inverse, update)
        if projections.shape[-1]:  # no delayed update right after a merge
            matmul = LowRankPfaffian._matmul
            projected = projected + matmul(projections, matmul(r_inverses, matmul(projections.mT, update)))
        ratio, r_inverse = LowRankPfaffian._ratio_and_r_inverse_core(update, projected)
        return ratio, projected, r_inverse

    @staticmethod
    def _ratio_core(
        inverse: torch.Tensor, projections: torch.Tensor, r_inverses: torch.Tensor, update: torch.Tensor
    ) -> torch.Tensor:
        return PfaffianUpdater._step_core(inverse, projections, r_inverses, update)[0]

    @staticmethod
    def _accept_core(
        inverse: torch.Tensor,
        projections: torch.Tensor,
        r_inverses: torch.Tensor,
        pfaffian_value: torch.Tensor,
        update: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # One accepted step: its ratio, P with a_tau appended, diag(D, R_tau^{-1}) and the new Pfaffian.
        ratio, projected, r_inverse = PfaffianUpdater._step_core(inverse, projections, r_inverses, update)
        batch_shape = torch.broadcast_shapes(projections.shape[:-2], projected.shape[:-2])
        width, size = r_inverses.shape[-1], r_inverse.shape[-1]
        projections = torch.cat(
            [
                projections.expand(*batch_shape, *projections.shape[-2:]),
                projected.expand(*batch_shape, *projected.shape[-2:]),
            ],
            dim=-1,
        )
        r_inverses = r_inverses.expand(*batch_shape, width, width)
        r_inverse = r_inverse.expand(*batch_shape, size, size)
        top = torch.cat([r_inverses, r_inverses.new_zeros(*batch_shape, width, size)], dim=-1)
        bottom = torch.cat([r_inverse.new_zeros(*batch_shape, size, width), r_inverse], dim=-1)
        return ratio, projections, torch.cat([top, bottom], dim=-2), pfaffian_value * ratio

    @staticmethod
    def _row_update(index: torch.Tensor, change: torch.Tensor) -> torch.Tensor:
        # u = (c, e_i) with c_i = 0, of shape (..., n, 2); index and change share their batch shape.
        index = index[..., None]  # (..., 1)
        change = change.scatter(-1, index, 0)  # the diagonal stays zero
        return torch.stack([change, torch.zeros_like(change).scatter(-1, index, 1)], dim=-1)

    @staticmethod
    def _ratio_replace_core(
        inverse: torch.Tensor,
        projections: torch.Tensor,
        r_inverses: torch.Tensor,
        index: torch.Tensor,
        change: torch.Tensor,
    ) -> torch.Tensor:
        update = PfaffianUpdater._row_update(index, change)
        return PfaffianUpdater._step_core(inverse, projections, r_inverses, update)[0]

    @staticmethod
    def _accept_replace_core(
        inverse: torch.Tensor,
        projections: torch.Tensor,
        r_inverses: torch.Tensor,
        pfaffian_value: torch.Tensor,
        index: torch.Tensor,
        change: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        update = PfaffianUpdater._row_update(index, change)
        return PfaffianUpdater._accept_core(inverse, projections, r_inverses, pfaffian_value, update)

    @classmethod
    def from_inverse(
        cls,
        inverse: torch.Tensor,
        *,
        pfaffian: torch.Tensor | None = None,
        max_delay: int = 8,
        compiled: bool = False,
    ) -> "PfaffianUpdater":
        """
        Start from an inverse that is already known, without inverting a matrix or computing its Pfaffian.

        :param inverse: Inverse of the initial skew-symmetric matrix, of shape ``(..., n, n)``.
        :param pfaffian: Pfaffian of the initial matrix, of shape ``(...)``. When ``None``, :attr:`pfaffian`
            starts at 1 and is the product of the accepted ratios, i.e. the Pfaffian relative to the start.
        :param max_delay: Number of accepted updates after which they are merged into the inverse, at least 1.
        :param compiled: When ``True``, run the steps through ``torch.compile``.
        :return: The updater.
        :rtype: PfaffianUpdater
        :raises ValueError: If ``max_delay`` is smaller than 1 or ``inverse`` is not square.
        :raises TypeError: If the dtype of ``inverse`` is not supported.
        """
        updater = cls.__new__(cls)
        updater._setup(max_delay, compiled)
        cls._check_matrix(inverse)
        updater._inverse = SkewGradient.apply_if_needed(inverse)
        updater._owns_inverse = False  # the caller's tensor: copied before an in-place update
        updater._pfaffian = torch.ones(inverse.shape[:-2], dtype=inverse.dtype, device=inverse.device)
        if pfaffian is not None:
            updater._pfaffian = updater._pfaffian * pfaffian
        updater._clear_delays()
        return updater

    def __init__(self, matrix: torch.Tensor, *, max_delay: int = 8, compiled: bool = False) -> None:
        """
        Start from ``matrix``, computing its inverse and its Pfaffian.

        :param matrix: Invertible skew-symmetric matrix of shape ``(..., n, n)``.
        :param max_delay: Number of accepted updates after which they are merged into the inverse, at least 1.
        :param compiled: When ``True``, run the steps through ``torch.compile``.
        :raises ValueError: If ``max_delay`` is smaller than 1 or ``matrix`` is not square.
        :raises TypeError: If the dtype of ``matrix`` is not supported.
        """
        self._setup(max_delay, compiled)
        self.reset(matrix)

    def ratio(self, update: torch.Tensor) -> torch.Tensor:
        """
        Ratio ``Pf(A - u J u^T) / Pf(A)`` for the current matrix ``A``, without changing the state.

        :param update: Update vectors ``u``, of shape ``(..., n, 2k)``.
        :return: The ratio, of shape ``(...)``.
        :rtype: torch.Tensor
        """
        self._check_update(update)
        native = self._native_immediate(LowRankPfaffian._ratio_core, update)
        if native is not None:
            return native
        return self._backend._call(
            PfaffianUpdater._ratio_core, self._inverse, self._projections, self._r_inverses, update
        )

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
        native = self._accept_in_place(LowRankPfaffian._ratio_and_inverse_core, update)
        if native is not None:
            return native
        state = (self._inverse, self._projections, self._r_inverses, self._pfaffian)
        ratio, self._projections, self._r_inverses, self._pfaffian = self._backend._call(
            PfaffianUpdater._accept_core, *state, update
        )
        return self._after_accept(ratio)

    def ratio_replace(self, index: torch.Tensor | int, change: torch.Tensor) -> torch.Tensor:
        """
        Ratio for adding ``change`` to row ``index`` (and subtracting it from the column), without changing the
        state. See :meth:`~torch_pfaffian.low_rank.LowRankPfaffian.ratio_replace`.

        :param index: Index of the row and column, an integer or an integer tensor of shape ``(...)``.
        :param change: Change of the row, of shape ``(..., n)``.
        :return: The ratio, of shape ``(...)``.
        :rtype: torch.Tensor
        """
        index, change = self._prepare_replace(index, change)
        native = self._native_immediate(LowRankPfaffian._ratio_replace_core, index, change)
        if native is not None:
            return native
        state = (self._inverse, self._projections, self._r_inverses)
        return self._backend._call(PfaffianUpdater._ratio_replace_core, *state, index, change)

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
        index, change = self._prepare_replace(index, change)
        if mask is not None:
            # c = 0 gives u = (0, e_i), so that -u J u^T = 0 and the ratio is exactly 1.
            change = torch.where(mask[..., None], change, torch.zeros_like(change))
        native = self._accept_in_place(LowRankPfaffian._ratio_replace_and_inverse_core, index, change)
        if native is not None:
            return native
        state = (self._inverse, self._projections, self._r_inverses, self._pfaffian)
        ratio, self._projections, self._r_inverses, self._pfaffian = self._backend._call(
            PfaffianUpdater._accept_replace_core, *state, index, change
        )
        return self._after_accept(ratio)

    def merge(self) -> None:
        """
        Fold the delayed updates into the stored inverse, ``A_0^{-1} <- A_tau^{-1}``.
        """
        if self._delay:
            self._inverse = self._current_inverse()  # a new tensor, which nobody else holds
            self._owns_inverse = True
        self._clear_delays()

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
        self._owns_inverse = True
        self._pfaffian = pfaffian(matrix)
        self._clear_delays()

    def _setup(self, max_delay: int, compiled: bool) -> None:
        if max_delay < 1:
            raise ValueError(f"max_delay must be at least 1; got {max_delay}.")
        self._max_delay = max_delay
        self._backend = CompiledLowRankPfaffian if compiled else LowRankPfaffian

    def _clear_delays(self) -> None:
        inverse = self._inverse
        self._projections = inverse.new_zeros(*inverse.shape[:-1], 0)
        self._r_inverses = inverse.new_zeros(*inverse.shape[:-2], 0, 0)
        self._delay = 0

    def _after_accept(self, ratio: torch.Tensor) -> torch.Tensor:
        self._delay += 1
        if self._delay >= self._max_delay:
            self.merge()
        return ratio

    def _native_immediate(self, core: Callable, *tensors: torch.Tensor) -> Any:
        # A ratio from the native kernel when nothing is delayed (see _accept_in_place); None otherwise.
        if self._delay or self._inverse.numel() > PfaffianUpdater.IMMEDIATE_MAX_NUMEL:
            return None
        return LowRankPfaffian._native_call(core, (self._inverse, *tensors))

    def _accept_in_place(self, core: Callable, *tensors: torch.Tensor) -> torch.Tensor | None:
        # With nothing delayed, an accepted move is applied to the inverse at once, in place, by the native kernel:
        # one read and one write of the inverse, faster than the delayed step's PyTorch operators at every size
        # measured (one thread; n = 32: 0.8 us against 4.9 us per row move on a batch of 64; n = 512 with a batch
        # of 64: 0.18 ms against 0.53 ms). An inverse the caller holds is copied first. None when the native path
        # does not apply (see LowRankPfaffian._native_plan).
        if self._delay or self._inverse.numel() > PfaffianUpdater.IMMEDIATE_MAX_NUMEL:
            return None
        if not self._owns_inverse:
            if LowRankPfaffian._native_plan(core, (self._inverse, *tensors), in_place=True) is None:
                return None
            self._inverse = self._inverse.clone()  # keeps the layout, row- or column-major
            self._owns_inverse = True
        ratio = LowRankPfaffian._native_call_in_place(core, (self._inverse, *tensors))
        if ratio is not None:
            self._pfaffian = self._pfaffian * ratio
        return ratio

    def _current_inverse(self) -> torch.Tensor:
        # The inverse with the delayed corrections applied, A_0^{-1} + P D P^T in one fused pass.
        if not self._delay:
            return self._inverse
        right = LowRankPfaffian._matmul(self._r_inverses, self._projections.mT)
        return LowRankPfaffian.add_product(self._inverse, self._projections, right)

    def _check_update(self, update: torch.Tensor) -> None:
        dimension = self._inverse.shape[-1]
        if update.shape[-2:-1] != (dimension,) or update.shape[-1] == 0 or update.shape[-1] % 2:
            raise ValueError(f"update must have shape (..., {dimension}, 2k) with k >= 1; got {tuple(update.shape)}.")
        if update.dtype != self._inverse.dtype or update.device != self._inverse.device:
            raise TypeError(
                f"update must have the dtype and device of the matrix, {self._inverse.dtype} on "
                f"{self._inverse.device}; got {update.dtype} on {update.device}."
            )

    def _prepare_replace(self, index: torch.Tensor | int, change: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dimension = self._inverse.shape[-1]
        if change.shape[-1] != dimension:
            raise ValueError(f"change must have shape (..., {dimension}); got {tuple(change.shape)}.")
        if change.dtype != self._inverse.dtype or change.device != self._inverse.device:
            raise TypeError(
                f"change must have the dtype and device of the matrix, {self._inverse.dtype} on "
                f"{self._inverse.device}; got {change.dtype} on {change.device}."
            )
        index = LowRankPfaffian.as_index(index, change.device)
        if index.shape == change.shape[:-1]:  # broadcast_shapes costs about 10 us
            return index, change
        batch_shape = torch.broadcast_shapes(change.shape[:-1], index.shape)
        if change.shape[:-1] != batch_shape:
            change = change.expand(*batch_shape, dimension)
        return index.expand(batch_shape), change

    @property
    def delay(self) -> int:
        """
        Number of accepted updates not yet merged into the stored inverse.

        :return: The number of delayed updates, between 0 and ``max_delay - 1``.
        :rtype: int
        """
        return self._delay

    @property
    def inverse(self) -> torch.Tensor:
        """
        Inverse of the current matrix, with the delayed updates applied (the state is not changed).

        :return: The inverse, of shape ``(..., n, n)``.
        :rtype: torch.Tensor
        """
        if not self._delay:
            self._owns_inverse = False  # the caller holds it now: the next in-place update works on a copy
        return self._current_inverse()

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
