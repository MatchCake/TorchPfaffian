from typing import cast

import torch

from .. import _rust
from .strategy import PfaffianStrategy


class RustPfaffianParlettReid(PfaffianStrategy):
    """
    Compute the signed Pfaffian with the Rust Parlett-Reid kernel.

    The forward moves the input to a contiguous CPU array and calls the compiled
    ``torch_pfaffian._rust`` kernel at a precision chosen from the input dtype: ``float32`` inputs
    use the single-precision kernel and every other floating dtype uses the double-precision kernel.
    The result is cast back to the input dtype and device. The backward is the same as
    :class:`PfaffianParlettReid` (the Pfaffian adjugate ``d pf(A) / d A = (1 / 2) pf(A) (A^{-1})^T``,
    exact for invertible and singular inputs), computed in PyTorch. CUDA inputs are evaluated on CPU
    for the forward; the backward runs on the input device.

    The input is a skew-symmetric matrix of shape ``(..., n, n)``.
    """

    NAME = "RustPfaffianParlettReid"

    @staticmethod
    def forward(matrix: torch.Tensor) -> torch.Tensor:
        dimension = matrix.shape[-1]
        if dimension % 2 != 0:
            return torch.zeros(matrix.shape[:-2], dtype=matrix.dtype, device=matrix.device)
        if dimension == 0:
            return torch.ones(matrix.shape[:-2], dtype=matrix.dtype, device=matrix.device)
        flat = matrix.reshape(-1, dimension, dimension)  # (batch, n, n)
        if matrix.dtype == torch.float32:
            working_dtype = torch.float32
            kernel = _rust.signed_pfaffian_f32
        else:
            working_dtype = torch.float64
            kernel = _rust.signed_pfaffian_f64
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
