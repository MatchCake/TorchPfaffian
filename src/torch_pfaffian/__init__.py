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
    __version__ = importlib_metadata.version("TorchPfaffian")

import warnings
from collections.abc import Callable

import torch

from .strategies import *
from .utils import get_all_subclasses

warnings.filterwarnings("ignore", category=Warning, module="docutils")
warnings.filterwarnings("ignore", category=Warning, module="sphinx")

pfaffian_strategy_map = {_cls.NAME.lower().strip(): _cls for _cls in get_all_subclasses(PfaffianStrategy)}


def get_pfaffian_function(name: str = PfaffianFDBPf.NAME) -> Callable[[torch.Tensor], torch.Tensor]:
    name = name.lower().strip()
    if name not in pfaffian_strategy_map:
        raise ValueError(f"Unknown strategy name: {name}. Available strategies: {list(pfaffian_strategy_map.keys())}")
    return pfaffian_strategy_map[name].apply


def pfaffian(matrix: torch.Tensor, *, sign: bool = True, check_input: bool = False) -> torch.Tensor:
    """
    Compute the Pfaffian of a skew-symmetric matrix, choosing the strategy from the input.

    The matrix has shape ``(..., 2n, 2n)`` and the result has shape ``(...,)``, sharing the
    backend, dtype, and device of ``matrix``.

    Strategy selection:

    ===================================  ===========================  =========================================
    Condition                            Strategy                     Reason
    ===================================  ===========================  =========================================
    ``sign=True``, Rust built, CPU input  ``RustPfaffianParlettReid``  fastest signed path (native Rust kernel)
    ``sign=True``, otherwise              ``PfaffianParlettReid``      GPU-native and pure-Python fallback
    ``sign=False``, grad needed           ``PfaffianFDBPf``            magnitude only; robust analytic backward
    ``sign=False``, no grad               ``PfaffianDet``              cheapest: ``sqrt(|det|)`` only
    ===================================  ===========================  =========================================

    The Rust kernel runs on CPU, so a non-CPU (e.g. CUDA) input is routed to ``PfaffianParlettReid``,
    which runs natively on the input device and avoids a host round-trip.

    The Pfaffian is only defined for skew-symmetric matrices; the strategies assume this and do not
    check it. Pass ``check_input=True`` to validate the assumption. For large matrices the Pfaffian
    can exceed the floating range and overflow to ``inf``; a ``RuntimeWarning`` is emitted when the
    result is not finite.

    :param matrix: Skew-symmetric matrix of shape ``(..., 2n, 2n)``.
    :param sign: When ``True`` (default) return the signed Pfaffian, otherwise its magnitude.
    :param check_input: When ``True``, validate that ``matrix`` is square in its last two dimensions
        and skew-symmetric (``A == -A^T``) before computing, raising ``ValueError`` otherwise. Off by
        default (``False``) so trusted inputs pay nothing; the check is an O(n^2) comparison, cheap
        relative to the O(n^3) Pfaffian.
    :return: The Pfaffian of the input, of shape ``(...,)``.
    :rtype: torch.Tensor
    """
    if check_input:
        if matrix.shape[-1] != matrix.shape[-2]:
            raise ValueError(f"Expected a square matrix in the last two dimensions, got shape {tuple(matrix.shape)}.")
        if not torch.allclose(matrix, -matrix.transpose(-1, -2)):
            raise ValueError("Input matrix is not skew-symmetric (A != -A^T).")

    if sign:
        # The Rust kernel is CPU-only, so non-CPU inputs use the device-native PyTorch strategy.
        if RustPfaffianParlettReid is not None and matrix.device.type == "cpu":
            result = RustPfaffianParlettReid.apply(matrix)
        else:
            result = PfaffianParlettReid.apply(matrix)
    elif matrix.requires_grad and torch.is_grad_enabled():
        result = PfaffianFDBPf.apply(matrix)
    else:
        result = PfaffianDet.apply(matrix)

    if not torch.isfinite(result).all():
        warnings.warn(
            "Pfaffian is not finite (overflow to inf/nan): its magnitude exceeds the floating range "
            "of the input dtype at this matrix dimension. Consider a higher-precision dtype.",
            RuntimeWarning,
            stacklevel=2,
        )
    return result
