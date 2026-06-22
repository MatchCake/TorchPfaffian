from typing import cast

import torch

from .. import _rust
from .strategy import PfaffianStrategy


class RustPfaffianParlettReid(PfaffianStrategy):
    """
    Compute the signed Pfaffian with the Rust Parlett-Reid kernel.

    The forward moves the input to a contiguous CPU array and calls the compiled
    ``torch_pfaffian._rust`` kernel for the input dtype, each running natively at that precision:
    ``float16``, ``float32``/``complex64`` and ``float64``/``complex128`` map to the half-, single-
    and double-precision kernels. Half precision is the caller's explicit choice and carries its risk:
    the elimination runs entirely in ``float16``, which is the least accurate kernel and can overflow
    the narrow ``float16`` range to ``inf`` for larger or larger-scaled matrices (use ``float32`` for a
    measurably more accurate result). Complex inputs are computed natively over the complex field (no
    imaginary part is discarded), giving the correct complex signed Pfaffian. Any other dtype raises
    :class:`TypeError`.
    The result is cast back to the input dtype and device. The backward is the same as
    :class:`PfaffianParlettReid` (the Pfaffian adjugate
    ``d pf(A) / d A = (1 / 2) pf(A) (A^{-1})^T``, exact for invertible and singular inputs and
    conjugated for complex autograd), computed in PyTorch. CUDA inputs are evaluated on CPU for the
    forward; the backward runs on the input device.

    The input is a skew-symmetric matrix of shape ``(..., n, n)``.
    """

    NAME = "RustPfaffianParlettReid"

    _KERNEL_BY_DTYPE = {
        torch.float16: ("signed_pfaffian_f16", torch.float16),
        torch.float32: ("signed_pfaffian_f32", torch.float32),
        torch.float64: ("signed_pfaffian_f64", torch.float64),
        torch.complex64: ("signed_pfaffian_c64", torch.complex64),
        torch.complex128: ("signed_pfaffian_c128", torch.complex128),
    }

    @staticmethod
    def forward(matrix: torch.Tensor) -> torch.Tensor:
        dimension = matrix.shape[-1]
        if dimension % 2 != 0:
            return torch.zeros(matrix.shape[:-2], dtype=matrix.dtype, device=matrix.device)
        if dimension == 0:
            return torch.ones(matrix.shape[:-2], dtype=matrix.dtype, device=matrix.device)
        if matrix.dtype not in RustPfaffianParlettReid._KERNEL_BY_DTYPE:
            supported = ", ".join(str(dtype) for dtype in RustPfaffianParlettReid._KERNEL_BY_DTYPE)
            raise TypeError(
                f"RustPfaffianParlettReid has no Rust kernel for dtype {matrix.dtype}; supported dtypes "
                f"are {supported}. Cast the matrix to a supported dtype, or use the pure-PyTorch "
                "PfaffianParlettReid strategy which handles any dtype. To request a Rust kernel for this "
                "dtype, please open an issue at https://github.com/MatchCake/TorchPfaffian/issues."
            )
        flat = matrix.reshape(-1, dimension, dimension)  # (batch, n, n)
        kernel_name, working_dtype = RustPfaffianParlettReid._KERNEL_BY_DTYPE[matrix.dtype]
        kernel = getattr(_rust, kernel_name)
        array = flat.detach().to(working_dtype).cpu().contiguous().numpy()
        result = kernel(array)  # (batch,) in working_dtype
        pfaffian = torch.from_numpy(result).to(dtype=matrix.dtype, device=matrix.device)
        return pfaffian.reshape(matrix.shape[:-2])

    @staticmethod
    def backward(ctx: torch.autograd.function.BackwardCFunction, grad_output: torch.Tensor) -> torch.Tensor | None:
        r"""
        Gradient of the signed Pfaffian with respect to the input matrix.

        .. math::
            \frac{\partial \text{pf}(A)}{\partial A_{ij}} = \frac{\text{pf}(A)}{2} (A^{-1})_{ji}

        :param ctx: Context holding the saved input matrix and the forward Pfaffian.
        :param grad_output: Gradient of the output with respect to the loss.
        :return: Gradient of the input matrix, or ``None`` when the input does not require grad.
        :rtype: torch.Tensor | None
        """
        matrix, pfaffian = cast("tuple[torch.Tensor, torch.Tensor]", ctx.saved_tensors)
        if not ctx.needs_input_grad[0]:
            return None
        return RustPfaffianParlettReid.pfaffian_grad_matrix(matrix, pfaffian, grad_output)
