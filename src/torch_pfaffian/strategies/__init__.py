from typing import Any, Callable

import torch

from .pfaffian_block_det import PfaffianBlockDet
from .pfaffian_det import PfaffianDet
from .pfaffian_fdbpf import PfaffianFDBPf
from .pfaffian_masked_magnitude import PfaffianMaskedMagnitude
from .pfaffian_parlett_reid import PfaffianParlettReid
from .pfaffian_slog import SlogPfaffianStrategy
from .pfaffian_small import PfaffianSmall
from .strategy import PfaffianStrategy

# None when the package was installed without its compiled Rust extension.
_RustStrategy: type[PfaffianStrategy] | None
try:
    from .pfaffian_rust_parlett_reid import RustPfaffianParlettReid as _RustStrategy
except ImportError:  # pragma: no cover
    _RustStrategy = None  # pragma: no cover
RustPfaffianParlettReid = _RustStrategy

# The fused Pfaffian-and-inverse function of pfaffian()'s graphs on CPU; None without the compiled extension.
_RustWithInverse: Any
try:
    from .pfaffian_rust_parlett_reid import RustPfaffianWithInverse as _RustWithInverse
except ImportError:  # pragma: no cover
    _RustWithInverse = None  # pragma: no cover
RustPfaffianWithInverse = _RustWithInverse
