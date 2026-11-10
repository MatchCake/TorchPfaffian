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

    working = matrix.reshape(-1, dimension, dimension).clone()  # (batch, m, m), eliminated in place
    batch = working.shape[0]
    rows = torch.arange(batch, device=device)[:, None].expand(batch, 2)  # (batch, 2)
    offsets = []
    for column in range(0, dimension - 2, 2):
        # Partial pivoting: the largest |a[column, p]| (p > column) is swapped into position (column, column + 1).
        # The two rows, then the two columns, are read with one gather and written back swapped with one scatter;
        # when p = column + 1 both writes carry the same values.
        offset = torch.argmax(working[:, column, column + 1 :].abs(), dim=1)  # (batch,), p - column - 1
        pair = torch.stack([offset.new_full((batch,), column + 1), offset + (column + 1)], dim=1)  # (batch, 2)
        working[rows, pair] = working[rows, pair].flip(1)
        working[rows, :, pair] = working[rows, :, pair].flip(1)
        offsets.append(offset)
        # Schur complement of the leading 2x2 block onto the trailing block, in place. An exact zero pivot makes
        # the steps after it non-finite for that batch element only; its result is replaced below.
        tau = working[:, column, column + 2 :] / working[:, column, column + 1, None]  # (batch, m - column - 2)
        second_row_tail = working[:, column + 1, column + 2 :]  # (batch, m - column - 2)
        trailing = working[:, column + 2 :, column + 2 :]
        trailing.addcmul_(second_row_tail[:, :, None], tau[:, None, :])
        trailing.addcmul_(tau[:, :, None], second_row_tail[:, None, :], value=-1)

    super_index = torch.arange(0, dimension, 2, device=device)
    pivots = working[:, super_index, super_index + 1]  # (batch, m / 2), the last one included
    magnitudes = pivots.abs()
    alive = (magnitudes != 0).all(dim=1)
    log_abs = magnitudes.log().sum(dim=1)
    phase = (pivots / magnitudes).prod(dim=1)  # unit-modulus factors (+-1 for real inputs)
    if offsets:
        swaps = (torch.stack(offsets, dim=1) != 0).sum(dim=1)  # (batch,), each swap flips the sign
        phase = (1 - 2 * (swaps % 2)) * phase

    phase = torch.where(alive, phase, torch.zeros_like(phase))
    log_abs = torch.where(alive, log_abs, torch.full_like(log_abs, -torch.inf))
    return phase.reshape(batch_shape), log_abs.reshape(batch_shape)
