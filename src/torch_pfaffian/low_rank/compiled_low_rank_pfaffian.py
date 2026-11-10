import functools
from collections.abc import Callable
from typing import TypeVar

import torch

from .low_rank_pfaffian import LowRankPfaffian

T = TypeVar("T")


class CompiledLowRankPfaffian(LowRankPfaffian):
    """
    :class:`LowRankPfaffian` whose tensor computations run through ``torch.compile``.

    Every method has the signature, results and gradients of :class:`LowRankPfaffian`; the inputs are validated
    in eager Python as there, and only the tensor computation behind it (one core per method) is compiled, which
    fuses its small operators into a few kernels. That removes most of the per-operator overhead that dominates
    small inputs: on CPU, the inverse updates of a single ``n = 32`` matrix take about half the eager time, and
    on CUDA the per-call launches drop by about 2x. The ratios alone gain little on CPU, where the fixed cost of
    a compiled call (about 20 µs) is close to that of the eager operators.

    Each core is compiled on its first call (a few seconds) with ``dynamic=None``: a second shape makes the
    sizes dynamic, so a matrix that grows step by step (:class:`~torch_pfaffian.low_rank.PfaffianChain`) is
    compiled twice, not once per size. Compilation needs what ``torch.compile`` needs on the platform (a C++
    compiler on CPU, Triton on CUDA). The compiled cores are shared by all callers in the process and are
    reset by ``torch._dynamo.reset()``. Use it for long runs (Monte Carlo chains, many samples), where the
    compilation is amortized; :class:`LowRankPfaffian` stays the default.
    """

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _compiled(core: Callable[..., T]) -> Callable[..., T]:
        return torch.compile(core, dynamic=None)

    @classmethod
    def _call(cls, core: Callable[..., T], *tensors: torch.Tensor) -> T:
        return CompiledLowRankPfaffian._compiled(core)(*tensors)
