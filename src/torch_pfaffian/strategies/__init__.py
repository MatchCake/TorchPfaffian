from typing import Callable

import torch

from .pfaffian_block_det import PfaffianBlockDet
from .pfaffian_det import PfaffianDet
from .pfaffian_fdbpf import PfaffianFDBPf
from .pfaffian_parlett_reid import PfaffianParlettReid
from .strategy import PfaffianStrategy

try:
    from .pfaffian_rust_parlett_reid import RustPfaffianParlettReid
except ImportError:
    RustPfaffianParlettReid = None
