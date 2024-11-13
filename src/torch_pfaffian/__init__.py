"""
Project description.
"""

__author__ = "Jérémie Gince"
__email__ = "gincejeremie@gmail.com"
__copyright__ = "Copyright 2024, Jérémie Gince"
__license__ = "Apache 2.0"
__url__ = "https://github.com/MatchCake/TorchPfaffian"
__version__ = "0.0.1-beta0"

import warnings
from collections.abc import Callable

import torch

warnings.filterwarnings("ignore", category=Warning, module="docutils")
warnings.filterwarnings("ignore", category=Warning, module="sphinx")

from .strategies import *
from .utils import get_all_subclasses

pfaffian_strategy_map = {
    _cls.NAME.lower().strip(): _cls
    for _cls in get_all_subclasses(PfaffianStrategy)
}


def get_pfaffian_function(name: str = PfaffianFDBPf.NAME) -> Callable[[torch.Tensor], torch.Tensor]:
    name = name.lower().strip()
    if name not in pfaffian_strategy_map:
        raise ValueError(f"Unknown strategy name: {name}. Available strategies: {list(pfaffian_strategy_map.keys())}")
    return pfaffian_strategy_map[name].apply

