import math

import torch


class LogMagnitudePfaffian(torch.autograd.Function):
    r"""
    Differentiable ``log|Pf(A)| = 0.5 * log|det(A)|`` using a single factorization in each direction.

    The forward computes one :func:`torch.linalg.slogdet`; the backward supplies the analytic gradient
    ``0.5 * (A^{-1})^T`` (Wirtinger-conjugated for complex inputs) through a single non-raising
    :func:`torch.linalg.inv_ex`, instead of differentiating through ``slogdet``. Differentiating through
    ``slogdet`` forms ``grad * A^{-T}``, which is infinite at an exactly-singular input, so even the
    correct upstream multiplier of ``0`` would give ``0 * inf = NaN``. Here exactly-singular elements
    (``det(A) == 0``, so ``slogdet`` returns sign ``0``) keep the correct ``-inf`` value and receive a
    zero gradient: the identity is substituted before the inverse so the factorization stays finite, and
    the gradient is masked to zero there. Numerically singular elements (a round-off ``det`` with nonzero
    sign) are already finite and left unchanged. No host synchronization happens in either direction.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, matrix: torch.Tensor) -> torch.Tensor:
        sign, log_abs_det = torch.linalg.slogdet(matrix)
        ctx.save_for_backward(matrix, sign == 0)  # sign == 0 marks exactly-singular (det == 0) elements
        return 0.5 * log_abs_det

    @staticmethod
    def backward(ctx: torch.autograd.function.BackwardCFunction, grad_output: torch.Tensor) -> torch.Tensor | None:
        matrix, singular = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None
        identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device).expand_as(matrix)
        safe_matrix = torch.where(singular[..., None, None], identity, matrix)  # invertible at singular elements
        inverse, _ = torch.linalg.inv_ex(safe_matrix)  # non-raising; safe_matrix is never singular
        grad = 0.5 * grad_output[..., None, None] * inverse.conj().transpose(-1, -2)
        return torch.where(singular[..., None, None], torch.zeros_like(grad), grad)


def log_magnitude_pfaffian(matrix: torch.Tensor) -> torch.Tensor:
    r"""
    Log-magnitude ``log|Pf(A)| = 0.5 * log|det(A)|`` of a batch of skew-symmetric matrices.

    The value is computed through :func:`torch.linalg.slogdet`, so it works in the log domain and stays
    finite (and keeps its gradient alive) even for tiny Pfaffians such as probabilities of order
    ``2^{-k}`` that would underflow the linear-domain ``sqrt(|det|)``. The gradient uses the analytic
    :class:`LogMagnitudePfaffian`, a single factorization each direction that is ``NaN``-free at
    exactly-singular inputs (see that class). Odd-dimensional inputs are singular, so the log-magnitude
    is ``-inf`` (returned with an autograd connection so a backward through it yields a zero gradient
    rather than raising, and without reading the entries so ``inf`` / ``nan`` inputs stay ``-inf``).

    :param matrix: Skew-symmetric matrices of shape ``(..., m, m)``, real or complex floating point.
    :return: The log-magnitude of shape ``(...,)`` in the input's real floating dtype (real for
        complex inputs, since a magnitude is real-valued).
    :rtype: torch.Tensor
    """
    if matrix.shape[-1] % 2 == 1:
        batch_shape = matrix.shape[:-2]
        real_dtype = matrix.real.dtype if matrix.dtype.is_complex else matrix.dtype
        connected_zero = matrix.reshape(*batch_shape, -1)[..., :0].sum(-1)  # value 0, grad_fn, entries unread
        if matrix.dtype.is_complex:
            connected_zero = connected_zero.real
        return connected_zero + torch.full(batch_shape, -torch.inf, dtype=real_dtype, device=matrix.device)
    return LogMagnitudePfaffian.apply(matrix)


def magnitude_pfaffian(matrix: torch.Tensor, *, epsilon: float = 0.0) -> torch.Tensor:
    r"""
    Magnitude ``|Pf(A)|`` of a batch of skew-symmetric matrices, real-valued and differentiable.

    The magnitude is obtained by exponentiating :func:`log_magnitude_pfaffian`. When ``epsilon`` is
    positive the log-magnitude is floored at ``0.5 * log(epsilon)`` before exponentiation, reproducing
    the historical ``sqrt(clamp(|det|, epsilon))`` floor as a function argument rather than a mutable
    global. Prefer the default ``0.0`` together with :func:`log_magnitude_pfaffian` for gradient-safe
    small probabilities.

    :param matrix: Skew-symmetric matrices of shape ``(..., m, m)``, real or complex floating point.
    :param epsilon: When positive, floors the result at ``sqrt(epsilon)``. The default ``0.0`` applies
        no floor.
    :return: The magnitude ``|Pf(A)|`` of shape ``(...,)`` in the input's real floating dtype.
    :rtype: torch.Tensor
    """
    log_pf = log_magnitude_pfaffian(matrix)
    if epsilon > 0.0:
        log_pf = torch.clamp(log_pf, min=0.5 * math.log(epsilon))
    return torch.exp(log_pf)
