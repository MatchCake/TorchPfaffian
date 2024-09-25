"""
Project description.
"""

__author__ = "Jérémie Gince"
__email__ = "gincejeremie@gmail.com"
__copyright__ = "Copyright 2024, Jérémie Gince"
__license__ = "Apache 2.0"
__url__ = "https://github.com/MatchCake/torch_pfaffian"
__version__ = "0.0.1-beta0"

import warnings
from collections.abc import Callable

warnings.filterwarnings("ignore", category=Warning, module="docutils")
warnings.filterwarnings("ignore", category=Warning, module="sphinx")


from .strategies.strategy import PfaffianStrategy
from .utils import get_all_subclasses

strategy_map = {
    _cls.NAME.lower().strip(): _cls
    for _cls in get_all_subclasses(PfaffianStrategy)
}


def get_pfaffian_function(name: str) -> Callable:
    name = name.lower().strip()
    if name not in strategy_map:
        raise ValueError(f"Unknown ansatz name: {name}")
    return strategy_map[name].apply

