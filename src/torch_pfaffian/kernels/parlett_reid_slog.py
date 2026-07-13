import torch


def _real_dtype_of(dtype: torch.dtype) -> torch.dtype:
    """
    Real floating dtype associated with ``dtype``.

    :param dtype: A real or complex floating dtype.
    :return: ``dtype`` itself when real, otherwise its real component dtype
        (``complex64`` maps to ``float32`` and ``complex128`` to ``float64``).
    :rtype: torch.dtype
    """
    if dtype.is_complex:
        return torch.float32 if dtype == torch.complex64 else torch.float64
    return dtype


@torch.no_grad()
def parlett_reid_slog(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Log-domain signed Pfaffian of a batch of skew-symmetric matrices via Parlett-Reid elimination.

    The batched skew-tridiagonalization with partial pivoting is run device-side with no host
    synchronization (no ``.item()``, ``.any()`` or ``isfinite().all()``), and the result is
    accumulated as ``(phase, log|Pf|)`` rather than a raw product. The log domain removes both the
    overflow at large dimensions and the underflow for tiny Pfaffians (probabilities of order
    ``2^{-k}``). Real and complex inputs share the single code path: ``phase`` is ``+-1`` (or ``0``)
    for real inputs and a unit-modulus complex number for complex inputs. An exact zero pivot column
    terminates that batch element cleanly with ``(0, -inf)``, since its remaining rank-2 updates
    vanish, so no epsilon threshold is needed.

    :param matrix: Skew-symmetric matrices of shape ``(..., m, m)``, real or complex floating point.
    :return: A pair ``(phase, log_abs)`` where ``phase`` has shape ``(...,)`` and matches the input
        dtype (with ``Pf = phase * exp(log_abs)``, and ``phase = 0`` when ``Pf = 0``), and ``log_abs``
        has shape ``(...,)`` in the matching real dtype (``-inf`` when ``Pf = 0``). Odd ``m`` gives
        ``(0, -inf)`` and ``m = 0`` gives ``(1, 0)``.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(f"expected (..., m, m) square matrices, got shape {tuple(matrix.shape)}")
    batch_shape = matrix.shape[:-2]
    dimension = matrix.shape[-1]
    device = matrix.device
    dtype = matrix.dtype
    real_dtype = _real_dtype_of(dtype)

    if dimension % 2 == 1:
        return (
            torch.zeros(batch_shape, dtype=dtype, device=device),
            torch.full(batch_shape, -torch.inf, dtype=real_dtype, device=device),
        )
    if dimension == 0:
        return (
            torch.ones(batch_shape, dtype=dtype, device=device),
            torch.zeros(batch_shape, dtype=real_dtype, device=device),
        )

    working = matrix.reshape(-1, dimension, dimension).clone()  # (batch, m, m)
    batch = working.shape[0]
    batch_index = torch.arange(batch, device=device)
    phase = torch.ones(batch, dtype=dtype, device=device)
    log_abs = torch.zeros(batch, dtype=real_dtype, device=device)
    alive = torch.ones(batch, dtype=torch.bool, device=device)
    zero_log = torch.zeros((), dtype=real_dtype, device=device)
    one_phase = torch.ones((), dtype=dtype, device=device)

    for column in range(0, dimension - 2, 2):
        # Partial pivoting: bring the largest |a[column, row]| (row > column) into position (column, column + 1).
        pivot_row = torch.argmax(working[:, column, column + 1 :].abs(), dim=1) + column + 1
        need_swap = pivot_row != (column + 1)
        row_next = working[:, column + 1, :].clone()
        row_pivot = working[batch_index, pivot_row, :]
        working[:, column + 1, :] = row_pivot
        working[batch_index, pivot_row, :] = torch.where(need_swap[:, None], row_next, row_pivot)
        col_next = working[:, :, column + 1].clone()
        col_pivot = working[batch_index, :, pivot_row]
        working[:, :, column + 1] = col_pivot
        working[batch_index, :, pivot_row] = torch.where(need_swap[:, None], col_next, col_pivot)
        phase = torch.where(need_swap, -phase, phase)

        pivot = working[:, column, column + 1]
        abs_pivot = pivot.abs()
        is_zero = abs_pivot == 0
        alive = alive & ~is_zero
        safe_abs = torch.where(is_zero, torch.ones_like(abs_pivot), abs_pivot)
        safe_pivot = torch.where(is_zero, one_phase, pivot)
        phase = phase * (safe_pivot / safe_abs)
        log_abs = log_abs + torch.where(is_zero, zero_log, safe_abs.log())

        # Schur complement of the leading 2x2 block onto the trailing submatrix.
        first_row_tail = working[:, column, column + 2 :]  # (batch, m - column - 2)
        second_row_tail = working[:, column + 1, column + 2 :]  # (batch, m - column - 2)
        update = second_row_tail.unsqueeze(-1) * first_row_tail.unsqueeze(-2)
        update = update - update.transpose(-1, -2)
        working[:, column + 2 :, column + 2 :] += update / safe_pivot[:, None, None]

    last = working[:, dimension - 2, dimension - 1]
    abs_last = last.abs()
    is_zero = abs_last == 0
    alive = alive & ~is_zero
    safe_abs = torch.where(is_zero, torch.ones_like(abs_last), abs_last)
    safe_last = torch.where(is_zero, one_phase, last)
    phase = phase * (safe_last / safe_abs)
    log_abs = log_abs + torch.where(is_zero, zero_log, safe_abs.log())

    phase = torch.where(alive, phase, torch.zeros_like(phase))
    log_abs = torch.where(alive, log_abs, torch.full_like(log_abs, -torch.inf))
    return phase.reshape(batch_shape), log_abs.reshape(batch_shape)
