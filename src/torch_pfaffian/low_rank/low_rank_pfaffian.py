import functools
import importlib
import types
from collections.abc import Callable
from typing import Any, TypeVar

import torch

from ..kernels.small_n import SMALL_N_MAX, small_pfaffian
from .skew_gradient import SkewGradient

T = TypeVar("T")


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

    On CPU, without a graph to build (``torch.no_grad()``, or inputs that do not require grad), ``ratio``,
    ``ratio_and_inverse``, ``ratio_replace`` and ``ratio_replace_and_inverse`` run in one call of the compiled
    Rust extension instead of a handful of PyTorch operators, which dominate small inputs (a rank-2 ratio and
    inverse of one ``n = 32`` matrix: 17 us instead of 77 us). Updates wider than rank 2 on large batches keep
    the PyTorch path, whose matrix products are faster there.
    """

    # Updates wider than rank 2 take the Rust kernel only below this many inverse entries (batch * n^2): above,
    # PyTorch's BLAS products win (measured on one thread: n = 128 with a batch of 64, rank 4, 13 us against 9 us
    # per matrix; a batch of 64 with n = 32, 1.8 us against 1.8 us).
    NATIVE_WIDE_MAX_NUMEL = 2**16

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
        return -LowRankPfaffian._matmul(vectors.mT, inverse).mT

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
    def _ratio_core(inverse: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        rows = LowRankPfaffian._matmul(update.mT, inverse)  # (..., 2k, n), u^T A^{-1}
        if update.shape[-1] == 2:
            return (
                1 + LowRankPfaffian._matmul(rows[..., :1, :], update[..., 1:])[..., 0, 0]
            )  # R_01 = 1 + u_0^T A^{-1} u_1
        return LowRankPfaffian._r_pfaffian(LowRankPfaffian._r_matrix(LowRankPfaffian._matmul(rows, update)))

    @staticmethod
    def _ratio_and_inverse_core(inverse: torch.Tensor, update: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # With the rows u^T A^{-1} = -(A^{-1} u)^T, the correction (A^{-1} u) R^{-1} (A^{-1} u)^T is
        # rows^T R^{-1} rows.
        rows = LowRankPfaffian._matmul(update.mT, inverse)  # (..., 2k, n)
        inner = LowRankPfaffian._matmul(rows, update)  # (..., 2k, 2k), u^T A^{-1} u
        if update.shape[-1] == 2:
            ratio = 1 + inner[..., 0, 1]
            # R^{-1} = [[0, -1/r], [1/r, 0]]: rows^T R^{-1} rows = (rows_1^T rows_0 - rows_0^T rows_1) / r.
            right = torch.stack([rows[..., 0, :], -rows[..., 1, :]], dim=-2) / ratio[..., None, None]
            return ratio, LowRankPfaffian.add_product(inverse, rows.mT.flip(-1), right)
        r_matrix = LowRankPfaffian._r_matrix(LowRankPfaffian._skew(inner))
        ratio = LowRankPfaffian._r_pfaffian(r_matrix)
        solved = torch.linalg.solve_ex(r_matrix, rows)[0]  # (..., 2k, n), R^{-1} rows
        return ratio, LowRankPfaffian.add_product(inverse, rows.mT, solved)

    @staticmethod
    def _ratio_replace_core(inverse: torch.Tensor, index: torch.Tensor, change: torch.Tensor) -> torch.Tensor:
        column = LowRankPfaffian._column(inverse, index)  # (..., n), A^{-1} e_i
        return 1 + (change * column).sum(dim=-1)

    @staticmethod
    def _ratio_replace_and_inverse_core(
        inverse: torch.Tensor, index: torch.Tensor, change: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        column = LowRankPfaffian._column(inverse, index)  # (..., n), A^{-1} e_i
        ratio = 1 + (change * column).sum(dim=-1)
        row = LowRankPfaffian._matmul(change[..., None, :], inverse)  # (..., 1, n), c^T A^{-1} = -(A^{-1} c)^T
        # (A^{-1} e_i)(A^{-1} c)^T - (A^{-1} c)(A^{-1} e_i)^T = [col, row^T] @ [-row; col^T]
        left = torch.cat([column[..., :, None], row.mT], dim=-1)  # (..., n, 2)
        right = torch.cat([-row, column[..., None, :]], dim=-2) / ratio[..., None, None]  # (..., 2, n)
        return ratio, LowRankPfaffian.add_product(inverse, left, right)

    @staticmethod
    def _column(inverse: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        # Column i of the skew-symmetric A^{-1} as minus its row i: the same values, and a row gather, which
        # torch.compile's C++ backend fuses correctly (torch 2.10 generated invalid code for the column gather
        # fused with the concatenation of _ratio_replace_and_inverse_core).
        return -torch.take_along_dim(inverse, index[..., None, None], dim=-2)[..., 0, :]  # (..., n)

    @staticmethod
    def _ratio_extend_core(inverse: torch.Tensor, border: torch.Tensor, corner: torch.Tensor) -> torch.Tensor:
        schur = corner + LowRankPfaffian._matmul(
            LowRankPfaffian._matmul(border.mT, inverse), border
        )  # (..., 2k, 2k), C + B^T A^{-1} B
        return LowRankPfaffian._small_pfaffian(schur)

    @staticmethod
    def _ratio_extend_and_inverse_core(
        inverse: torch.Tensor, border: torch.Tensor, corner: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # With rows = B^T A^{-1} = -P^T and T = S^{-1} rows: A^{-1} + P S^{-1} P^T = A^{-1} + rows^T T,
        # -P S^{-1} = -T^T and -S^{-1} P^T = T.
        rows = LowRankPfaffian._matmul(border.mT, inverse)  # (..., 2k, m)
        schur = LowRankPfaffian._skew(
            corner + LowRankPfaffian._matmul(rows, border)
        )  # (..., 2k, 2k), kept skew for its inverse
        ratio = LowRankPfaffian._small_pfaffian(schur)
        schur_inverse = LowRankPfaffian._small_inverse(schur, ratio)
        solved = LowRankPfaffian._matmul(schur_inverse, rows)  # (..., 2k, m), T
        batch_shape = torch.broadcast_shapes(inverse.shape[:-2], solved.shape[:-2])
        size, width = inverse.shape[-1], schur.shape[-1]
        top_left = LowRankPfaffian.add_product(inverse, rows.mT, solved)
        top = torch.cat(
            [top_left.expand(*batch_shape, size, size), -solved.mT.expand(*batch_shape, size, width)], dim=-1
        )
        bottom = torch.cat(
            [solved.expand(*batch_shape, width, size), schur_inverse.expand(*batch_shape, width, width)], dim=-1
        )
        return ratio, torch.cat([top, bottom], dim=-2)

    @staticmethod
    def _ratio_and_r_inverse_core(update: torch.Tensor, projected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # R is re-skew-symmetrized: its inverse enters the stored inverse, which must stay skew-symmetric (the
        # delayed projections and project() rely on it), and near-singular steps amplify any asymmetry.
        r_matrix = LowRankPfaffian._r_matrix(LowRankPfaffian._skew(LowRankPfaffian._matmul(update.mT, projected)))
        ratio = LowRankPfaffian._r_pfaffian(r_matrix)
        if r_matrix.shape[-1] == 2:
            return ratio, -r_matrix / (ratio * ratio)[..., None, None]  # [[0, r], [-r, 0]]^{-1} = -R / r^2
        return ratio, torch.linalg.inv_ex(r_matrix)[0]

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _package() -> types.ModuleType:
        # The package's dtype helpers and pfaffian(); torch_pfaffian imports this module before defining them, so
        # they are looked up on first use, once.
        return importlib.import_module("torch_pfaffian")

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _native_kernels() -> dict[Callable, dict[torch.dtype, Callable]]:
        # The Rust kernels of the cores, per dtype; empty without the compiled extension.
        try:
            from .. import _rust
        except ImportError:  # pragma: no cover - without the compiled extension
            return {}
        suffixes = {torch.float32: "f32", torch.float64: "f64", torch.complex64: "c64", torch.complex128: "c128"}
        names: dict[Callable, str] = {
            LowRankPfaffian._ratio_core: "low_rank_ratio",
            LowRankPfaffian._ratio_and_inverse_core: "low_rank_ratio_and_inverse",
            LowRankPfaffian._ratio_replace_core: "replace_ratio",
            LowRankPfaffian._ratio_replace_and_inverse_core: "replace_ratio_and_inverse",
        }
        return {
            core: {dtype: getattr(_rust, f"{name}_{suffix}") for dtype, suffix in suffixes.items()}
            for core, name in names.items()
        }

    @staticmethod
    def _native_call(core: Callable[..., T], tensors: tuple[torch.Tensor, ...]) -> T | None:
        """
        Run a core with its Rust kernel, for CPU inputs without a graph to build.

        A core is a handful of small PyTorch operators, whose per-operator cost (a few microseconds each on CPU)
        dominates single small matrices; the kernel does the same arithmetic in one native call. It applies when
        the core has a kernel for the dtype, every input is on CPU, no input needs a gradient, the inputs share
        their batch shape without broadcasting, and the inverse is stored row- or column-major; otherwise it
        returns ``None``, for the PyTorch core.

        :param core: One of the cores with a kernel (see :meth:`_native_kernels`).
        :param tensors: The core's validated inputs: ``(inverse, update)`` or ``(inverse, index, change)``.
        :return: The core's result, or ``None``.
        :rtype: T | None
        """
        kernel = LowRankPfaffian._native_kernels().get(core, {}).get(tensors[0].dtype)
        if kernel is None or any(tensor.device.type != "cpu" for tensor in tensors):
            return None
        if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
            return None
        inverse, others = tensors[0], tensors[1:]
        size = inverse.shape[-1]
        batch_shape = inverse.shape[:-2]
        offset = 2 if len(others) == 1 else 1  # update (..., n, 2k), or index (...) and change (..., n)
        if others[-1].shape[: others[-1].dim() - offset] != batch_shape or (
            len(others) == 2 and others[0].shape != batch_shape
        ):
            return None
        if len(others) == 1 and others[0].shape[-1] > 2 and inverse.numel() >= LowRankPfaffian.NATIVE_WIDE_MAX_NUMEL:
            return None  # wide updates of large batches: BLAS's matrix products beat the kernel's dot products
        # A column-major inverse (as torch.linalg.inv returns) is passed as its row-major transpose.
        if inverse.is_contiguous():
            buffer, transposed = inverse, False
        elif inverse.mT.is_contiguous():
            buffer, transposed = inverse.mT, True
        else:
            return None  # broadcast or strided: copying it would cost more than the PyTorch core
        array = LowRankPfaffian._numpy
        if len(others) == 1:
            result = kernel(array(buffer, 3), transposed, array(others[0], 3))
        else:
            result = kernel(array(buffer, 3), transposed, array(others[0], 1), array(others[1], 2))
        flat = len(batch_shape) == 1
        if isinstance(result, tuple):
            ratio, new_inverse = torch.from_numpy(result[0]), torch.from_numpy(result[1])
            if transposed:  # the kernel updated the transpose: the result keeps the input's column-major layout
                new_inverse = new_inverse.mT
            if not flat:
                ratio, new_inverse = ratio.reshape(batch_shape), new_inverse.reshape(*batch_shape, size, size)
            return ratio, new_inverse  # type: ignore[return-value]
        ratio = torch.from_numpy(result)
        return ratio if flat else ratio.reshape(batch_shape)  # type: ignore[return-value]

    @staticmethod
    def _numpy(tensor: torch.Tensor, dimensions: int) -> Any:
        # A C-contiguous NumPy view with the batch dimensions flattened (dimensions - 1 trailing ones kept), with
        # only the conversions the tensor needs: each costs about a microsecond, a fair share of a small call.
        if tensor.dim() != dimensions:
            tensor = tensor.reshape(-1, *tensor.shape[tensor.dim() - dimensions + 1 :])
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        if tensor.is_conj():
            tensor = tensor.resolve_conj()
        return tensor.numpy()

    @staticmethod
    @functools.lru_cache(maxsize=64)
    def _cached_skew_identity(rank: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        # J is a constant: building it costs five small kernels, more than the rest of a rank-2 ratio on CPU.
        with torch.inference_mode(False):
            return LowRankPfaffian.skew_identity(rank, dtype=dtype, device=device)

    @staticmethod
    def _matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        # torch.matmul's broadcasting logic (expand, reshape) costs about as much as the bmm kernel itself on the
        # small operands of these updates; when the batch shapes already agree, call mm or bmm directly.
        if left.dim() == right.dim() == 3 and left.shape[0] == right.shape[0]:
            return torch.bmm(left, right)
        if left.dim() == right.dim() == 2:
            return torch.mm(left, right)
        return left @ right

    @staticmethod
    def _skew(matrix: torch.Tensor) -> torch.Tensor:
        return 0.5 * (matrix - matrix.mT)

    @staticmethod
    def _r_matrix(inner: torch.Tensor) -> torch.Tensor:
        # R = J + u^T A^{-1} u. The ratio-only paths do not re-skew-symmetrize it, as the Pfaffian kernels only read
        # its upper triangle; the paths that invert it do (see ratio_and_r_inverse).
        rank = inner.shape[-1] // 2
        return LowRankPfaffian._cached_skew_identity(rank, inner.dtype, inner.device) + inner

    @staticmethod
    def _r_pfaffian(r_matrix: torch.Tensor) -> torch.Tensor:
        ratio = LowRankPfaffian._small_pfaffian(r_matrix)
        return -ratio if LowRankPfaffian.skew_identity_pfaffian(r_matrix.shape[-1] // 2) < 0 else ratio

    @staticmethod
    def _small_pfaffian(matrix: torch.Tensor) -> torch.Tensor:
        # The unrolled kernel reads the upper triangle and needs no routing, finiteness check or host
        # synchronization; larger blocks (k > 5, rare) go through pfaffian().
        if matrix.shape[-1] == 2:
            return matrix[..., 0, 1]
        if matrix.shape[-1] <= SMALL_N_MAX:
            return small_pfaffian(matrix)
        return LowRankPfaffian._package().pfaffian(matrix, check_finite=False)

    @staticmethod
    def _small_inverse(matrix: torch.Tensor, pfaffian_value: torch.Tensor) -> torch.Tensor:
        if matrix.shape[-1] == 2:
            # [[0, r], [-r, 0]]^{-1} = -[[0, r], [-r, 0]] / r^2
            return -matrix / (pfaffian_value * pfaffian_value)[..., None, None]
        return torch.linalg.inv_ex(matrix)[0]

    @staticmethod
    def _restore(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return tensor if tensor.dtype == dtype else tensor.to(dtype)

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
        return LowRankPfaffian._restore(inverse, compute_dtype), LowRankPfaffian._restore(update, compute_dtype), dtype

    @staticmethod
    def _prepare_replace(
        function_name: str, inverse: torch.Tensor, index: torch.Tensor | int, change: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.dtype]:
        # Validated, upcast inputs, broadcast to a common batch shape.
        LowRankPfaffian._check_inputs(function_name, inverse, change)
        dimension = inverse.shape[-1]
        if change.shape[-1] != dimension:
            raise ValueError(f"change must have shape (..., n) with n = {dimension}; got {change.shape}.")
        index = LowRankPfaffian.as_index(index, inverse.device)
        dtype = inverse.dtype
        compute_dtype = LowRankPfaffian._compute_dtype(function_name, dtype)
        inverse = LowRankPfaffian._restore(SkewGradient.apply_if_needed(inverse), compute_dtype)
        change = LowRankPfaffian._restore(change, compute_dtype)
        batch_shape = inverse.shape[:-2]
        if index.shape != batch_shape or change.shape[:-1] != batch_shape:  # broadcast_shapes costs about 10 us
            batch_shape = torch.broadcast_shapes(batch_shape, index.shape, change.shape[:-1])
        if change.shape[:-1] != batch_shape:
            change = change.expand(*batch_shape, dimension)
        if index.shape != batch_shape:
            index = index.expand(batch_shape)
        if inverse.shape[:-2] != batch_shape:
            inverse = inverse.expand(*batch_shape, dimension, dimension)
        return inverse, index, change, dtype

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
        inverse = LowRankPfaffian._restore(SkewGradient.apply_if_needed(inverse), compute_dtype)
        border = LowRankPfaffian._restore(border, compute_dtype)
        return inverse, border, LowRankPfaffian._restore(corner, compute_dtype), dtype

    @staticmethod
    def _check_inputs(function_name: str, inverse: torch.Tensor, other: torch.Tensor) -> None:
        if inverse.dim() < 2 or inverse.shape[-1] != inverse.shape[-2]:
            raise ValueError(f"{function_name}: inverse must have shape (..., n, n); got {tuple(inverse.shape)}.")
        if other.dtype != inverse.dtype or other.device != inverse.device:
            raise TypeError(
                f"{function_name}: the inputs must share a dtype and a device; got {inverse.dtype} on "
                f"{inverse.device} and {other.dtype} on {other.device}."
            )
        LowRankPfaffian._package()._check_dtype(function_name, inverse.dtype)

    @staticmethod
    def _compute_dtype(function_name: str, dtype: torch.dtype) -> torch.dtype:
        package = LowRankPfaffian._package()
        if dtype not in package.LOW_PRECISION_DTYPES:
            return dtype
        package._warn_upcast(function_name, dtype, stacklevel=4)
        return package._upcast_dtype(dtype)

    @classmethod
    def ratio(cls, inverse: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
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
        inverse, update, dtype = cls._prepare_update("LowRankPfaffian.ratio", inverse, update)
        return cls._restore(cls._call(LowRankPfaffian._ratio_core, inverse, update), dtype)

    @classmethod
    def ratio_and_inverse(cls, inverse: torch.Tensor, update: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        inverse, update, dtype = cls._prepare_update("LowRankPfaffian.ratio_and_inverse", inverse, update)
        ratio, new_inverse = cls._call(LowRankPfaffian._ratio_and_inverse_core, inverse, update)
        return cls._restore(ratio, dtype), cls._restore(new_inverse, dtype)

    @classmethod
    def ratio_replace(cls, inverse: torch.Tensor, index: torch.Tensor | int, change: torch.Tensor) -> torch.Tensor:
        r"""
        Ratio of Pfaffians when one row and the matching column of a skew-symmetric matrix change.

        Row ``i`` of ``A`` becomes ``A[i] + c`` and column ``i`` becomes ``A[:, i] - c``, so that ``A'`` stays
        skew-symmetric; ``c_i`` has no effect (the diagonal stays zero, and it only meets the zero diagonal of
        ``A^{-1}``, up to round-off). This is the rank-2 update ``u = (c, e_i)``
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
        inverse, index, change, dtype = cls._prepare_replace("LowRankPfaffian.ratio_replace", inverse, index, change)
        return cls._restore(cls._call(LowRankPfaffian._ratio_replace_core, inverse, index, change), dtype)

    @classmethod
    def ratio_replace_and_inverse(
        cls, inverse: torch.Tensor, index: torch.Tensor | int, change: torch.Tensor
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
        inverse, index, change, dtype = cls._prepare_replace(
            "LowRankPfaffian.ratio_replace_and_inverse", inverse, index, change
        )
        ratio, new_inverse = cls._call(LowRankPfaffian._ratio_replace_and_inverse_core, inverse, index, change)
        return cls._restore(ratio, dtype), cls._restore(new_inverse, dtype)

    @classmethod
    def ratio_extend(cls, inverse: torch.Tensor, border: torch.Tensor, corner: torch.Tensor) -> torch.Tensor:
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
        inverse, border, corner, dtype = cls._prepare_extend("LowRankPfaffian.ratio_extend", inverse, border, corner)
        return cls._restore(cls._call(LowRankPfaffian._ratio_extend_core, inverse, border, corner), dtype)

    @classmethod
    def ratio_extend_and_inverse(
        cls, inverse: torch.Tensor, border: torch.Tensor, corner: torch.Tensor
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
        inverse, border, corner, dtype = cls._prepare_extend(
            "LowRankPfaffian.ratio_extend_and_inverse", inverse, border, corner
        )
        ratio, new_inverse = cls._call(LowRankPfaffian._ratio_extend_and_inverse_core, inverse, border, corner)
        return cls._restore(ratio, dtype), cls._restore(new_inverse, dtype)

    @classmethod
    def ratio_and_r_inverse(cls, update: torch.Tensor, projected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Ratio of :meth:`ratio` and the inverse of its matrix ``R``, from the projection ``A^{-1} u``.

        :param update: Update vectors ``u``, of shape ``(..., n, 2k)``.
        :param projected: Projection ``A^{-1} u``, of shape ``(..., n, 2k)``.
        :return: The ratio, of shape ``(...)``, and ``R^{-1}``, of shape ``(..., 2k, 2k)`` (huge or non-finite
            when the ratio is zero).
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        return cls._call(LowRankPfaffian._ratio_and_r_inverse_core, update, projected)

    @classmethod
    def _call(cls, core: Callable[..., T], *tensors: torch.Tensor) -> T:
        # The hook the compiled variant overrides: run a core on validated tensors, natively where possible.
        native = LowRankPfaffian._native_call(core, tensors)
        return core(*tensors) if native is None else native
