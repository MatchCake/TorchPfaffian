"""
TorchPfaffian is a Python package for efficiently computing the Pfaffian of skew-symmetric matrices using PyTorch.
"""

import importlib_metadata

__author__ = "Jérémie Gince"
__email__ = "gincejeremie@gmail.com"
__copyright__ = "Copyright 2024, Jérémie Gince"
__license__ = "Apache 2.0"
__url__ = "https://github.com/MatchCake/TorchPfaffian"
__package__ = "torch_pfaffian"
try:
    __version__ = importlib_metadata.version(__package__)
except importlib_metadata.PackageNotFoundError:
    __version__ = importlib_metadata.version("torchpfaffian")

import warnings
from collections.abc import Callable

import torch

from .kernels import log_magnitude_pfaffian, magnitude_pfaffian
from .strategies import *
from .utils import get_all_subclasses

warnings.filterwarnings("ignore", category=Warning, module="docutils")
warnings.filterwarnings("ignore", category=Warning, module="sphinx")

# Matrices with last dimension at most AUTO_SMALL_MAX are routed to the exact unrolled small-m kernel,
# which is the measured CPU winner in that range; larger inputs use the Parlett-Reid / determinant paths.
AUTO_SMALL_MAX = 6

pfaffian_strategy_map = {_cls.NAME.lower().strip(): _cls for _cls in get_all_subclasses(PfaffianStrategy)}


def get_pfaffian_function(name: str = PfaffianFDBPf.NAME) -> Callable[[torch.Tensor], torch.Tensor]:
    name = name.lower().strip()
    if name not in pfaffian_strategy_map:
        raise ValueError(f"Unknown strategy name: {name}. Available strategies: {list(pfaffian_strategy_map.keys())}")
    return pfaffian_strategy_map[name].apply


def slog_pfaffian(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Log-domain signed Pfaffian of a batch of skew-symmetric matrices as ``(phase, log|Pf|)``.

    Thin wrapper over :class:`~torch_pfaffian.strategies.pfaffian_slog.SlogPfaffianStrategy`. The
    reconstruction is ``Pf = phase * exp(log_abs)``. The log domain neither overflows for large
    matrices nor underflows for tiny Pfaffians, and the kernel never synchronizes with the host.
    ``log_abs`` is differentiable wherever ``Pf != 0``; ``phase`` is non-differentiable (use
    :func:`pfaffian` with ``sign=True`` for the full derivative).

    :param matrix: Skew-symmetric matrix of shape ``(..., 2n, 2n)``.
    :return: A pair ``(phase, log_abs)``, each of shape ``(...,)``. ``phase`` shares the input dtype
        (``0`` when ``Pf = 0``); ``log_abs`` is in the matching real dtype (``-inf`` when ``Pf = 0``).
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    return SlogPfaffianStrategy.apply(matrix)


def pfaffian(
    matrix: torch.Tensor,
    *,
    sign: bool = True,
    check_input: bool = False,
    check_finite: bool = True,
    epsilon: float | None = None,
) -> torch.Tensor:
    """
    Compute the Pfaffian of a skew-symmetric matrix, choosing the strategy from the input.

    The matrix has shape ``(..., 2n, 2n)`` and the result has shape ``(...,)``, sharing the
    backend, dtype, and device of ``matrix``.

    Strategy selection:

    ==========================================  ===========================  ==================================
    Condition                                   Strategy                     Reason
    ==========================================  ===========================  ==================================
    ``m <= AUTO_SMALL_MAX``                     ``PfaffianSmall``            exact unrolled kernel (all dtypes)
    ``sign=True``, Rust built, CPU input         ``RustPfaffianParlettReid``  fastest signed path (native Rust)
    ``sign=True``, otherwise                     ``PfaffianParlettReid``      GPU-native and pure-Python fallback
    ``sign=False``, ``epsilon`` given            ``magnitude_pfaffian``       log-domain magnitude with floor
    ``sign=False``, grad needed                  ``PfaffianFDBPf``            magnitude only; robust backward
    ``sign=False``, no grad                      ``PfaffianDet``              cheapest: ``sqrt(|det|)`` only
    ==========================================  ===========================  ==================================

    Matrices with last dimension ``m <= AUTO_SMALL_MAX`` take the exact unrolled ``PfaffianSmall`` path
    first, regardless of ``sign`` and device (``|Pf|`` for ``sign=False``); it is exact for every dtype
    and differentiable end-to-end. For larger matrices the Rust kernel runs on CPU (in real and complex
    precisions), so a non-CPU (e.g. CUDA) input is routed to ``PfaffianParlettReid``, which runs
    natively on the input device and avoids a host round-trip. All signed strategies compute the correct
    complex signed Pfaffian without discarding the imaginary part and are differentiable end-to-end for
    real and complex inputs (the complex backward follows PyTorch's Wirtinger convention).

    The Pfaffian is only defined for skew-symmetric matrices; the strategies assume this and do not
    check it. Pass ``check_input=True`` to validate the assumption. The chosen strategy is the fastest
    correct one for the input, so calling ``pfaffian`` is all most callers need. For inputs with a wide
    dynamic range the fast linear-domain path can overflow to ``inf`` even when the true Pfaffian is
    finite; with ``check_finite=True`` (default) such results are transparently recomputed in the log
    domain and recovered, so the caller gets the correct value without reaching for :func:`slog_pfaffian`.
    Only a magnitude that genuinely exceeds the input dtype's floating range remains non-finite, and only
    then is a ``RuntimeWarning`` emitted. One extreme case is not auto-recovered: for ``m > AUTO_SMALL_MAX``
    the linear signed path declares a pivot magnitude below ``PfaffianStrategy.EPSILON`` (``1e-30``) an
    exact zero, so a matrix whose Pfaffian hinges on such a sub-``1e-30`` pivot is reported as ``0``.
    :func:`slog_pfaffian`, which uses an exact-zero pivot test, computes these extreme inputs correctly
    (as does the unrolled ``m <= AUTO_SMALL_MAX`` path, which has no pivot floor).

    :param matrix: Skew-symmetric matrix of shape ``(..., 2n, 2n)``.
    :param sign: When ``True`` (default) return the signed Pfaffian, otherwise its magnitude.
    :param check_input: When ``True``, validate that ``matrix`` is square in its last two dimensions
        and skew-symmetric (``A == -A^T``) before computing, raising ``ValueError`` otherwise. Off by
        default (``False``) so trusted inputs pay nothing; the check is an O(n^2) comparison, cheap
        relative to the O(n^3) Pfaffian.
    :param check_finite: When ``True`` (default), verify the result is finite; if the fast path
        overflowed, recompute in the log domain and recover the value, warning only if it still exceeds
        the dtype's range. This costs one device-to-host synchronization; pass ``False`` to skip both the
        check and the recovery in hot loops (the raw fast-path result, possibly ``inf``, is returned).
    :param epsilon: Only affects the magnitude path (``sign=False``). When a float is given, the
        magnitude is computed in the log domain and floored at ``sqrt(epsilon)`` (keeping gradients
        alive below the floor). When ``None`` (default), the routing is unchanged and the magnitude is
        computed via ``PfaffianFDBPf`` / ``PfaffianDet`` using ``PfaffianStrategy.EPSILON``.
    :return: The Pfaffian of the input, of shape ``(...,)``.
    :rtype: torch.Tensor
    """
    if check_input:
        if matrix.shape[-1] != matrix.shape[-2]:
            raise ValueError(f"Expected a square matrix in the last two dimensions, got shape {tuple(matrix.shape)}.")
        if not torch.allclose(matrix, -matrix.transpose(-1, -2)):
            raise ValueError("Input matrix is not skew-symmetric (A != -A^T).")

    if matrix.shape[-1] <= AUTO_SMALL_MAX:
        # The exact unrolled kernel wins in this range and is device- and dtype-agnostic.
        if sign:
            result = PfaffianSmall.apply(matrix)
        else:
            result = PfaffianSmall.apply(matrix).abs()
            if epsilon is not None:
                result = torch.clamp(result, min=epsilon**0.5)
    elif sign:
        # The Rust kernel is CPU-only, so non-CPU inputs use the device-native PyTorch strategy.
        if RustPfaffianParlettReid is not None and matrix.device.type == "cpu":
            result = RustPfaffianParlettReid.apply(matrix)
        else:
            result = PfaffianParlettReid.apply(matrix)
    elif epsilon is not None:
        result = magnitude_pfaffian(matrix, epsilon=epsilon)
    elif matrix.requires_grad and torch.is_grad_enabled():
        result = PfaffianFDBPf.apply(matrix)
    else:
        result = PfaffianDet.apply(matrix)

    if check_finite and not torch.isfinite(result).all():
        # The linear-domain fast paths can overflow to inf on wide-dynamic-range inputs whose true
        # Pfaffian is finite. Transparently recover those elements in the log domain (Pf = phase *
        # exp(log|Pf|) for the signed path, exp(log|Pf|) for the magnitude path), so the caller gets
        # the correct value without reaching for slog_pfaffian. Only a magnitude that genuinely exceeds
        # the dtype's float range stays non-finite, and only then is a warning emitted.
        # Recompute the whole batch in the log domain and use it: its reconstruction equals the linear
        # result on the finite elements (to log/exp rounding) and stays finite on the overflowed ones,
        # and its gradient (``0.5 * Pf * (A^{-1})^T`` for the signed path) is finite everywhere. The
        # linear graph is discarded, avoiding the ``0 * inf = NaN`` its backward would form from the
        # saved infinite Pfaffian.
        if sign:
            phase, log_abs = slog_pfaffian(matrix)
            result = phase * torch.exp(log_abs).to(phase.dtype)
        else:
            result = torch.exp(log_magnitude_pfaffian(matrix))
            if epsilon is not None:
                result = torch.clamp(result, min=epsilon**0.5)
        if not torch.isfinite(result).all():
            warnings.warn(
                "Pfaffian is not finite after log-domain recovery: its true magnitude genuinely exceeds "
                "the floating range of the input dtype at this matrix dimension. Use slog_pfaffian for the "
                "log-magnitude, or a higher-precision dtype.",
                RuntimeWarning,
                stacklevel=2,
            )
    return result
