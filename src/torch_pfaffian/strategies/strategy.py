from collections.abc import Callable
from typing import Any

import numpy as np
import torch

# The Rust inverse of _flagged_inverse, per dtype; empty when the extension is not built.
_NATIVE_INVERSE: dict[torch.dtype, Callable[[Any], tuple[Any, Any, Any]]]
try:
    from .. import _rust

    _NATIVE_INVERSE = {
        torch.float32: _rust.flagged_inverse_f32,
        torch.float64: _rust.flagged_inverse_f64,
        torch.complex64: _rust.flagged_inverse_c64,
        torch.complex128: _rust.flagged_inverse_c128,
    }
except (ImportError, AttributeError):  # pragma: no cover - without the compiled extension
    _NATIVE_INVERSE = {}


class PfaffianStrategy(torch.autograd.Function):
    EPSILON = 1e-30
    NAME = "PfaffianStrategy"
    # When True (default) the singular-element gradient is the exact minor-based Pfaffian adjugate,
    # which preserves the historical behavior at the cost of one host synchronization per backward.
    # When False the backward is fully sync-free: elements singular at round-off level (LU pivot ratio
    # <= n * eps, an LU failure, or pf == 0) get an exactly zero gradient via torch.where, so no
    # .any()/.item() branch and no adjugate computation run.
    EXACT_SINGULAR_GRAD = True
    # A batch element is routed to the exact minor-based adjugate when its LU factorization has a pivot
    # ratio min|U_ii| / max|U_ii| <= eps^0.75 (scale-free, so it does not depend on how the entries are
    # spread), or when the LU inverse fails its residual check ||A A^{-1} - I||_max > eps^0.5. Exponents
    # of the dtype eps keep both criteria dtype-adaptive (float64: ~1e-12 and ~1.5e-8; float32: ~6e-6
    # and ~3e-4).
    SINGULARITY_RTOL_EXPONENT = 0.75
    INVERSE_RESIDUAL_RTOL_EXPONENT = 0.5
    # A flagged element's Pfaffian adjugate comes from its singular values sigma_max >= ... >= sigma_3 >=
    # sigma_min when they show a clear corank. Corank 2: the null vectors are accurate when sigma_3 >=
    # eps^0.25 * sigma_max (their error is ~eps * sigma_max / sigma_3) and the rank-2 truncation when
    # sigma_min <= eps^0.75 * sigma_3, so the adjugate is accurate to ~eps^0.75. When sigma_3 <= n * eps *
    # sigma_max (a second pair at round-off level, corank 4 or more) the rank-2 formula is used as well: it
    # is exact for structured inputs whose smallest pair is isolated, and at the round-off level otherwise.
    # Everything else takes the exact minor Pfaffians.
    NULL_SPACE_SEPARATION_EXPONENT = 0.25
    # Elements per chunk (k * pairs * (n - 2)^2) of the batched minor Pfaffians: bounds the memory to 4-16 MiB
    # per chunk, which also measured as fast as larger chunks (cache-friendly).
    MINOR_CHUNK_ELEMENTS = 1 << 20
    # The Rust inverse of _flagged_inverse (Gauss-Jordan, one thread per matrix) beats PyTorch's LAPACK path up to
    # k = NATIVE_INVERSE_MAX_SIZE on one matrix (k = 32: 18 us against 173 us; k = 96: 321 us against 341 us), and
    # up to NATIVE_INVERSE_PARALLEL_MAX_SIZE when the batch of at least NATIVE_INVERSE_PARALLEL_MIN_BATCH matrices
    # runs in parallel (k = 128, batch 8: 1.5 ms against 3.0 ms); LAPACK wins beyond (measured on 12 cores).
    NATIVE_INVERSE_MAX_SIZE = 96
    # Device tensors up to this size are copied to the host with tensor.cpu() (see _host_array).
    HOST_COPY_SMALL_BYTES = 2**16
    NATIVE_INVERSE_PARALLEL_MAX_SIZE = 128
    NATIVE_INVERSE_PARALLEL_MIN_BATCH = 8

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
        garbage, the adjugate is recomputed exactly by :meth:`_singular_pfaffian_adjugate` (from the
        singular values and at most one minor Pfaffian when the corank is clear, else from all the minor
        Pfaffians of :meth:`_pfaffian_adjugate`, using ``cls``'s own forward). The minor-based path runs only on
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
        ``False`` the backward takes a fully host-synchronization-free path, skipping the ``.any()``
        branch and the adjugate entirely: elements singular at round-off level (an LU pivot ratio at most
        ``n * eps``, an LU failure, or ``pf == 0``; finite inputs only, so NaN still propagates) receive an
        exactly zero gradient (the true derivative there is nonzero only at corank exactly 2), and every
        other element, however ill-conditioned, the single inverse ``pf(A) A^{-1}`` (accurate to about
        ``cond(A) * eps``). In this mode flagging trades accuracy for speed, unlike the exact mode.

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
            # Sync-free path: elements singular at round-off level get a zero gradient, the others the single
            # inverse (even when ill-conditioned), so no host branch and no adjugate computation are needed.
            inverse, round_off = cls._round_off_inverse(matrix, pfaffian)
            adjugate = pfaffian[..., None, None] * inverse  # pf(A) A^{-1}; replaced where singular
            adjugate = torch.where(round_off[..., None, None], torch.zeros_like(adjugate), adjugate)
            return torch.einsum("...,...ij->...ji", 0.5 * grad_output, adjugate.conj())
        native = cls._native_grad_matrix(matrix, pfaffian, grad_output)
        if native is not None:
            return native
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
                exact_adjugate = cls._singular_pfaffian_adjugate(flat_matrix.index_select(0, exact_index))
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
        native = cls._native_flagged_inverse(matrix)
        if native is not None:
            inverse, rank_flag, residual_flag = native
            batch_shape = matrix.shape[:-2]
            return (
                torch.from_numpy(inverse).reshape(matrix.shape),
                torch.from_numpy(rank_flag).reshape(batch_shape),
                torch.from_numpy(residual_flag).reshape(batch_shape),
            )
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
    def _native_flagged_inverse(matrix: torch.Tensor) -> tuple[Any, Any, Any] | None:
        """
        :meth:`_flagged_inverse` from the Rust kernel, as NumPy arrays, for CPU inputs without a graph to build.

        One native call replaces the LU factorization, the solve, the residual product and the dozen small
        operators around them, which dominate the backward of small inputs. The flags follow the same
        criteria (the kernel's LU pivots on the modulus, LAPACK's complex LU on ``|re| + |im|``, so a complex
        element right at a threshold can be flagged differently). It is used up to ``k =
        NATIVE_INVERSE_MAX_SIZE``, and up to ``NATIVE_INVERSE_PARALLEL_MAX_SIZE`` for batches of at least
        ``NATIVE_INVERSE_PARALLEL_MIN_BATCH`` matrices, where it is faster.

        :param matrix: Square matrices of shape ``(..., k, k)`` with ``k >= 1``.
        :return: The inverse, of shape ``(batch, k, k)``, and the two flags, of shape ``(batch,)``, with the
            batch dimensions flattened; or ``None`` when the input is not on CPU, has another dtype or size,
            needs a differentiable inverse (double backward), or the extension is not built.
        :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray] | None
        """
        kernel = _NATIVE_INVERSE.get(matrix.dtype)
        if kernel is None or matrix.device.type != "cpu" or (torch.is_grad_enabled() and matrix.requires_grad):
            return None
        size = matrix.shape[-1]
        batch = matrix.numel() // max(1, size * size)
        if size > PfaffianStrategy.NATIVE_INVERSE_MAX_SIZE and not (
            batch >= PfaffianStrategy.NATIVE_INVERSE_PARALLEL_MIN_BATCH
            and size <= PfaffianStrategy.NATIVE_INVERSE_PARALLEL_MAX_SIZE
        ):
            return None
        return kernel(matrix.detach().reshape(-1, size, size).contiguous().resolve_conj().numpy())

    @classmethod
    def _native_grad_matrix(
        cls, matrix: torch.Tensor, pfaffian: torch.Tensor, grad_output: torch.Tensor
    ) -> torch.Tensor | None:
        """
        :meth:`pfaffian_grad_matrix` in NumPy around the Rust inverse, where :meth:`_native_flagged_inverse` applies.

        The same computation as the PyTorch path (the adjugate ``pf(A) A^{-1}``, zero where flagged, then the
        exact adjugate on the flagged elements whose ``grad_output`` is nonzero), with NumPy's microsecond
        operators instead of PyTorch's, which dominate the backward of small inputs.

        :param matrix: The saved input matrix of shape ``(..., n, n)``.
        :param pfaffian: The saved forward Pfaffian of shape ``(...,)``.
        :param grad_output: Gradient of the output with respect to the loss, broadcastable to ``(...,)``.
        :return: The gradient, of shape ``(..., n, n)``, or ``None`` when the native path does not apply
            (including a ``grad_output`` that requires grad, or a Pfaffian or ``grad_output`` off the CPU).
        :rtype: torch.Tensor | None
        """
        if grad_output.requires_grad or pfaffian.device.type != "cpu" or grad_output.device.type != "cpu":
            return None
        native = cls._native_flagged_inverse(matrix)
        if native is None:
            return None
        inverse, rank_flag, residual_flag = native
        values = pfaffian.detach().resolve_conj().reshape(-1).numpy()
        return cls._gradient_from_inverse(matrix, values, grad_output, inverse, rank_flag | residual_flag)

    @classmethod
    def _gradient_from_inverse(
        cls, matrix: torch.Tensor, values: Any, grad_output: torch.Tensor, inverse: Any, flagged: Any
    ) -> torch.Tensor:
        """
        The gradient of :meth:`pfaffian_grad_matrix` from a native inverse and its flags, in NumPy.

        :param matrix: The input matrix of shape ``(..., n, n)``, on CPU.
        :param values: The forward Pfaffians, as a NumPy array of shape ``(batch,)``.
        :param grad_output: Gradient of the output with respect to the loss, broadcastable to ``(...,)``.
        :param inverse: The inverses, as a NumPy array of shape ``(batch, n, n)`` (the identity where flagged).
        :param flagged: The singularity flags (rank or residual), as a NumPy boolean array of shape ``(batch,)``.
        :return: The gradient, of shape ``(..., n, n)``.
        :rtype: torch.Tensor
        """
        weight_tensor = grad_output.detach()
        if weight_tensor.shape != matrix.shape[:-2]:
            weight_tensor = weight_tensor.expand(matrix.shape[:-2])
        weights = weight_tensor.resolve_conj().reshape(-1).numpy()
        complex_input = inverse.dtype.kind == "c"
        # 0.5 * w * conj(pf(A) A^{-1})^T, as one scaled transpose of the inverse (the common, regular case).
        scale = 0.5 * weights * (values.conj() if complex_input else values)
        transposed = inverse.transpose(0, 2, 1)
        if complex_input:
            transposed = transposed.conj()
        singular = flagged | (values == 0)  # pf == 0 also catches an underflowed pf
        if not singular.any():
            gradient = (scale.astype(inverse.dtype, copy=False)[:, None, None] * transposed).astype(
                inverse.dtype, copy=False
            )
            return torch.from_numpy(gradient).reshape(matrix.shape)
        gradient = (np.where(singular, 0, scale).astype(inverse.dtype)[:, None, None] * transposed).astype(
            inverse.dtype, copy=False
        )
        exact = singular & (weights != 0)  # elements needing the minor adjugate
        if exact.any():
            exact_index = torch.from_numpy(exact.nonzero()[0])
            dimension = matrix.shape[-1]
            with torch.no_grad():
                matrices = matrix.detach().reshape(-1, dimension, dimension).index_select(0, exact_index)
                adjugate = cls._singular_pfaffian_adjugate(matrices).to(matrix.dtype).numpy()
            gradient[exact] = 0.5 * weights[exact][:, None, None] * adjugate.conj().transpose(0, 2, 1)
        return torch.from_numpy(gradient).reshape(matrix.shape)

    @staticmethod
    def _host_array(tensor: torch.Tensor) -> Any:
        """
        A C-contiguous NumPy array of ``tensor`` on the host, copied into NumPy's memory when it is on a device.

        ``tensor.cpu()`` lands in fresh torch memory that the GPU driver may write by direct memory access without
        mapping its pages for the CPU; the first CPU read then takes a page fault per 4 KiB (2048 for 8 MiB, about
        8 ms with the copy). NumPy's allocation of a large array asks for transparent huge pages, mapped up front.

        :param tensor: Any tensor; it is detached, made C-contiguous and its conjugation resolved.
        :return: The array, sharing the memory of a contiguous CPU ``tensor``.
        :rtype: numpy.ndarray
        """
        tensor = tensor.detach()
        if tensor.device.type == "cpu":
            return tensor.contiguous().resolve_conj().numpy()
        if (
            tensor.numel() * tensor.element_size() <= PfaffianStrategy.HOST_COPY_SMALL_BYTES
        ):  # pragma: no cover - needs a GPU
            return tensor.resolve_conj().cpu().numpy()  # a few pages: cheaper than allocating through NumPy
        host = torch.from_numpy(  # pragma: no cover - needs a GPU
            np.empty(tuple(tensor.shape), dtype=torch.empty(0, dtype=tensor.dtype).numpy().dtype)
        )
        host.copy_(tensor.resolve_conj())  # pragma: no cover - needs a GPU
        return host.numpy()  # pragma: no cover - needs a GPU

    @classmethod
    def _round_off_inverse(cls, matrix: torch.Tensor, pfaffian: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Batched inverse together with the mask of elements singular at round-off level, without host sync.

        An element is singular at round-off level when its LU factorization reports a zero pivot, its
        pivot ratio ``min|U_ii| / max|U_ii|`` is at most ``n * eps``, or ``pfaffian == 0``; elements with
        non-finite entries are never masked, so NaN and inf propagate. Masked elements are inverted as the
        identity. When a graph is being built (double backward) the inverse comes from
        :func:`torch.linalg.inv_ex` on the identity-substituted matrix.

        :param matrix: Square matrices of shape ``(..., k, k)`` with ``k >= 1``.
        :param pfaffian: Forward Pfaffians of shape ``(...,)``.
        :return: The inverse, of shape ``(..., k, k)``, and the boolean mask, of shape ``(...,)``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        size = matrix.shape[-1]
        identity = torch.eye(size, dtype=matrix.dtype, device=matrix.device).expand_as(matrix)
        with torch.no_grad():
            factors, pivots, info = torch.linalg.lu_factor_ex(matrix)
            pivot_magnitude = factors.diagonal(dim1=-2, dim2=-1).abs()  # (..., k) |U_ii|
            threshold = size * torch.finfo(matrix.dtype).eps * pivot_magnitude.amax(dim=-1)
            singular = (info != 0) | ~(pivot_magnitude.amin(dim=-1) > threshold) | (pfaffian == 0)
            singular = singular & torch.isfinite(matrix).flatten(-2).all(dim=-1)  # NaN and inf propagate
            identity_pivots = torch.arange(1, size + 1, dtype=pivots.dtype, device=pivots.device)
            safe_factors = torch.where(singular[..., None, None], identity, factors)
            safe_pivots = torch.where(singular[..., None], identity_pivots.expand_as(pivots), pivots)
            inverse = torch.linalg.lu_solve(safe_factors, safe_pivots, identity)
        if torch.is_grad_enabled() and matrix.requires_grad:
            inverse, _ = torch.linalg.inv_ex(torch.where(singular[..., None, None], identity, matrix))
        return inverse, singular

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
        count, dimension = matrices.shape[0], matrices.shape[-1]
        adjugate = torch.zeros_like(matrices)
        if count == 0 or dimension < 2:
            return adjugate
        first, second = torch.triu_indices(dimension, dimension, offset=1, device=matrices.device)  # (P,)
        indices = torch.arange(dimension, device=matrices.device)
        keep_mask = (indices[None, :] != first[:, None]) & (indices[None, :] != second[:, None])  # (P, n)
        keep = indices.expand(len(first), dimension)[keep_mask].reshape(len(first), dimension - 2)  # (P, n-2)
        signs = 1 - 2 * ((first + second) % 2)  # (P,) the cofactor sign (-1)^(i+j)
        pairs_per_chunk = max(1, cls.MINOR_CHUNK_ELEMENTS // max(1, count * (dimension - 2) ** 2))
        for start in range(0, len(first), pairs_per_chunk):
            chunk = slice(start, start + pairs_per_chunk)
            chunk_keep = keep[chunk]  # (p, n-2)
            minors = matrices[:, chunk_keep[:, :, None], chunk_keep[:, None, :]]  # (m, p, n-2, n-2)
            pair_count = chunk_keep.shape[0]
            flat_minors = minors.reshape(count * pair_count, dimension - 2, dimension - 2)  # explicit: n - 2 may be 0
            minor_pfaffians = cls.forward(flat_minors).reshape(count, pair_count)
            values = signs[chunk].to(minor_pfaffians.dtype) * minor_pfaffians  # (m, p)
            adjugate[:, first[chunk], second[chunk]] = values.to(adjugate.dtype)
            adjugate[:, second[chunk], first[chunk]] = -values.to(adjugate.dtype)
        return adjugate

    @classmethod
    def _singular_pfaffian_adjugate(cls, matrices: torch.Tensor) -> torch.Tensor:
        r"""
        Pfaffian adjugate of (numerically) singular skew-symmetric matrices, in ``O(n^3)`` where possible.

        The singular values of a skew-symmetric matrix come in pairs, and its corank is even. With
        ``sigma_max`` the largest singular value, ``sigma_min`` the smallest and ``sigma_3`` the first of
        the next pair:

        - corank 2, or a second pair at round-off level (``sigma_3 <= n * eps * sigma_max``, corank 4 or
          more), when ``sigma_3 >= eps^0.25 * sigma_max`` (accurate null vectors) and
          ``sigma_min <= eps^0.75 * sigma_3`` (negligible rank-2 truncation): the adjugate is
          skew-symmetric with its range in ``null(A) = span(u, v)``, so ``P = c (u v^T - v u^T)`` with
          ``u, v`` the right singular vectors of the two smallest singular values (transposes, not adjoints,
          also for complex input), and ``c`` is fixed by one minor Pfaffian at the pair ``(i, j)``
          maximizing ``|u_i v_j - v_i u_j|``; the result is accurate to about ``eps^0.75`` relative at a
          clear corank 2, exact for structured inputs whose smallest pair is isolated (e.g. a
          block-diagonal matrix with one zero block and tiny others), and at the round-off level of the
          matrix otherwise;
        - otherwise (ill-conditioned without a clear corank, for example a block-diagonal matrix whose
          blocks span many orders of magnitude, non-finite entries, or a failed SVD): the exact minor
          Pfaffians of :meth:`_pfaffian_adjugate`.

        Odd-dimensional matrices have an exactly zero adjugate (every minor is odd-dimensional).

        :param matrices: Skew-symmetric matrices of shape ``(m, n, n)``.
        :return: The Pfaffian adjugate, of shape ``(m, n, n)``.
        :rtype: torch.Tensor
        """
        count, dimension = matrices.shape[0], matrices.shape[-1]
        if count == 0 or dimension % 2 == 1:
            return torch.zeros_like(matrices)
        if dimension <= 4:
            return cls._pfaffian_adjugate(matrices)  # at most 6 tiny minors: the exact path is cheapest
        adjugate = torch.zeros_like(matrices)
        finite = torch.isfinite(matrices).flatten(-2).all(dim=-1)  # (m,)
        identity = torch.eye(dimension, dtype=matrices.dtype, device=matrices.device).expand_as(matrices)
        try:
            _, singular_values, right_adjoint = torch.linalg.svd(
                torch.where(finite[..., None, None], matrices, identity)
            )
        except torch.linalg.LinAlgError:
            return cls._pfaffian_adjugate(matrices)
        epsilon = torch.finfo(matrices.dtype).eps
        largest = singular_values[:, 0]  # (m,)
        smallest, third = singular_values[:, -1], singular_values[:, -3]  # the smallest and the next pair
        round_off_pair = third <= dimension * epsilon * largest  # a second pair at round-off level
        separated = third >= epsilon**cls.NULL_SPACE_SEPARATION_EXPONENT * largest
        truncated = smallest <= epsilon**cls.SINGULARITY_RTOL_EXPONENT * third
        corank_two = finite & (round_off_pair | (separated & truncated))
        fallback = ~corank_two
        if bool(corank_two.any()):
            index = corank_two.nonzero(as_tuple=True)[0]
            null = right_adjoint.index_select(0, index).mH[..., -2:]  # (r, n, 2) basis of null(A)
            adjugate[index] = cls._rank_two_adjugate(matrices.index_select(0, index), null[..., 0], null[..., 1])
        if bool(fallback.any()):
            index = fallback.nonzero(as_tuple=True)[0]
            adjugate[index] = cls._pfaffian_adjugate(matrices.index_select(0, index))
        return adjugate

    @classmethod
    def _rank_two_adjugate(
        cls, matrices: torch.Tensor, first_null: torch.Tensor, second_null: torch.Tensor
    ) -> torch.Tensor:
        """
        Pfaffian adjugate ``c (u v^T - v u^T)`` of corank-2 skew-symmetric matrices with null space ``(u, v)``.

        :param matrices: Corank-2 skew-symmetric matrices of shape ``(r, n, n)`` with ``n >= 4``.
        :param first_null: First null vector ``u``, of shape ``(r, n)``.
        :param second_null: Second null vector ``v``, of shape ``(r, n)``.
        :return: The Pfaffian adjugate, of shape ``(r, n, n)``.
        :rtype: torch.Tensor
        """
        count, dimension = matrices.shape[0], matrices.shape[-1]
        wedge = first_null[:, :, None] * second_null[:, None, :] - second_null[:, :, None] * first_null[:, None, :]
        upper = torch.triu(torch.ones(dimension, dimension, dtype=torch.bool, device=matrices.device), diagonal=1)
        magnitude = torch.where(upper, wedge.abs(), torch.zeros_like(wedge.abs())).reshape(count, -1)
        best = magnitude.argmax(dim=-1)  # (r,) flat index of the best pair, first < second
        first, second = best // dimension, best % dimension
        indices = torch.arange(dimension, device=matrices.device)
        keep_mask = (indices[None, :] != first[:, None]) & (indices[None, :] != second[:, None])
        keep = indices.expand(count, dimension)[keep_mask].reshape(count, dimension - 2)  # (r, n-2)
        rows = torch.gather(matrices, 1, keep[:, :, None].expand(count, dimension - 2, dimension))
        minors = torch.gather(rows, 2, keep[:, None, :].expand(count, dimension - 2, dimension - 2))
        minor_pfaffians = cls.forward(minors)  # (r,)
        sign = 1 - 2 * ((first + second) % 2)
        entry = sign.to(minor_pfaffians.dtype) * minor_pfaffians  # P[first, second]
        best_wedge = wedge[torch.arange(count, device=matrices.device), first, second]  # |best| <= 1
        # Normalize the wedge before scaling by the entry, so a huge minor Pfaffian cannot overflow.
        return (wedge / best_wedge[:, None, None] * entry.to(wedge.dtype)[:, None, None]).to(matrices.dtype)
