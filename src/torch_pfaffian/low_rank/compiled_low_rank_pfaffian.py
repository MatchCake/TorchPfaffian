import functools
import warnings
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
    compiler on CPU, Triton on CUDA). A core whose compiled call fails while its eager computation succeeds (no
    compiler, or a ``torch.compile`` bug of the installed PyTorch) runs eagerly from then on, with a
    ``RuntimeWarning`` the first time; an error that the eager computation raises too is raised as is. The
    compiled cores are shared by all callers in the process and are reset by ``torch._dynamo.reset()``. Use it
    for long runs (Monte Carlo chains, many samples), where the compilation is amortized;
    :class:`LowRankPfaffian` stays the default.
    """

    # Cores whose compiled call failed, run eagerly from then on.
    _EAGER_CORES: set[Callable] = set()

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _compiled(core: Callable[..., T]) -> Callable[..., T]:
        return torch.compile(core, dynamic=None)

    @classmethod
    def _call(cls, core: Callable[..., T], *tensors: torch.Tensor) -> T:
        if core in CompiledLowRankPfaffian._EAGER_CORES:
            return core(*tensors)
        try:
            return CompiledLowRankPfaffian._compiled(core)(*tensors)
        except Exception as error:
            result = core(*tensors)  # an error of the computation itself is raised here, by the eager core
            CompiledLowRankPfaffian._EAGER_CORES.add(core)
            message = (str(error).strip().splitlines() or [""])[0]
            warnings.warn(
                f"torch.compile failed on {core.__name__} ({type(error).__name__}: {message}); it runs eagerly "
                f"from now on, with the same results.",
                RuntimeWarning,
                stacklevel=3,
            )
            return result
