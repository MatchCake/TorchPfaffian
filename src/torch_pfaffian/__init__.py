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
__version__ = importlib_metadata.version(__package__)

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


_OPTIMIZE_CHOICES = ("auto", "time", "memory")


def pfaffian(matrix: torch.Tensor, *, sign: bool = True, optimize: str = "auto") -> torch.Tensor:
    """
    Compute the Pfaffian of a skew-symmetric matrix, choosing a strategy from the input.

    The matrix has shape ``(..., 2n, 2n)`` and the result has shape ``(...,)``, sharing the
    backend, dtype, and device of ``matrix``.

    Strategy selection (``optimize="auto"``):

    ===========================  ======================  =================================================
    Condition                    Strategy                Reason
    ===========================  ======================  =================================================
    ``sign=True`` (default)      ``PfaffianParlettReid`` only general strategy that returns the sign
    ``sign=False``, grad needed  ``PfaffianFDBPf``       magnitude only; robust analytic backward (pinv)
    ``sign=False``, no grad      ``PfaffianDet``         cheapest: ``sqrt(|det|)`` only
    ===========================  ======================  =================================================

    :param matrix: Skew-symmetric matrix of shape ``(..., 2n, 2n)``.
    :param sign: When ``True`` (default) return the signed Pfaffian, otherwise its magnitude.
    :param optimize: Optimization target, one of ``"auto"``, ``"time"``, ``"memory"``. Currently a
        tie-breaker reserved for when more strategies are available; ``"auto"`` follows the table above.
    :return: The Pfaffian of the input, of shape ``(...,)``.
    :rtype: torch.Tensor
    """
    if optimize not in _OPTIMIZE_CHOICES:
        raise ValueError(f"Unknown optimize value: {optimize!r}. Choose from {_OPTIMIZE_CHOICES}.")
    if sign:
        return PfaffianParlettReid.apply(matrix)
    grad_needed = matrix.requires_grad and torch.is_grad_enabled()
    if grad_needed:
        return PfaffianFDBPf.apply(matrix)
    return PfaffianDet.apply(matrix)
