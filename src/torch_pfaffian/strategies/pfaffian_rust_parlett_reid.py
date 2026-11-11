from collections.abc import Callable
from typing import Any, cast

import numpy as np
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
    forward; the backward runs on the host too for small inputs (at most ``HOST_BACKWARD_MAX_NUMEL`` entries),
    except a double backward, and on the input device otherwise.

    The input is a skew-symmetric matrix of shape ``(..., n, n)``.
    """

    NAME = "RustPfaffianParlettReid"
    # Attribute holding the host values of a device result computed through the host, so that pfaffian()'s
    # finiteness check reads them instead of copying the result back (a transfer and a synchronization).
    HOST_VALUES_ATTRIBUTE = "_torch_pfaffian_host_values"
    # Non-CPU inputs with at most this many entries also run their backward on the host (measured on an RTX 3070:
    # m = 32 and 64, batch 1, 30% faster; m = 8 with a batch of 256 (2^14 entries) already slower).
    HOST_BACKWARD_MAX_NUMEL = 2**13

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
        array = RustPfaffianParlettReid._host_array(flat.to(working_dtype))
        result = kernel(array)  # (batch,) in working_dtype
        pfaffian = torch.from_numpy(result).to(dtype=matrix.dtype, device=matrix.device).reshape(matrix.shape[:-2])
        if pfaffian.device.type != "cpu":  # pragma: no cover - needs a GPU
            setattr(pfaffian, RustPfaffianParlettReid.HOST_VALUES_ATTRIBUTE, result)
        return pfaffian

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
        if (
            matrix.device.type != "cpu"
            and matrix.numel() <= RustPfaffianParlettReid.HOST_BACKWARD_MAX_NUMEL
            and not torch.is_grad_enabled()
        ):  # pragma: no cover - needs a GPU
            # On a device the LU, solve and checks of a small input are a few dozen kernels bound by their launches,
            # so the backward runs on the host like the forward. A double backward (create_graph) stays on the
            # device, as do larger inputs, whose host backward grows with the batch while the device one does not.
            gradient = RustPfaffianParlettReid.pfaffian_grad_matrix(matrix.cpu(), pfaffian.cpu(), grad_output.cpu())
            return gradient.to(matrix.device)
        return RustPfaffianParlettReid.pfaffian_grad_matrix(matrix, pfaffian, grad_output)


class RustPfaffianWithInverse(torch.autograd.Function):
    """
    Signed Pfaffian with the Rust kernel, computing the inverse of its backward in the same native call.

    :func:`~torch_pfaffian.pfaffian` takes it when a graph is built on CPU (and for the small CUDA inputs it
    computes on the host, through the host), within the sizes where the Rust
    inverse is used (see :meth:`PfaffianStrategy._native_flagged_inverse`): one native call returns the Pfaffian,
    the inverse and its singularity flags, and the backward only forms ``0.5 * grad * pf(A) (A^{-1})^T`` in NumPy,
    with the exact adjugate of :class:`RustPfaffianParlettReid` on the flagged elements. The forward and the
    backward of one small matrix then cost a few native microseconds on top of autograd's own overhead. A double
    backward (``create_graph=True``) takes the differentiable PyTorch backward of :class:`RustPfaffianParlettReid`.
    It is a standalone :class:`torch.autograd.Function`, not a registered strategy: the values and the gradients
    are those of :class:`RustPfaffianParlettReid`.

    The input is a skew-symmetric matrix of shape ``(..., n, n)``.
    """

    _KERNEL_BY_DTYPE: dict[torch.dtype, Callable[[Any], tuple[Any, Any, Any, Any]]] = {
        torch.float32: _rust.signed_pfaffian_and_inverse_f32,
        torch.float64: _rust.signed_pfaffian_and_inverse_f64,
        torch.complex64: _rust.signed_pfaffian_and_inverse_c64,
        torch.complex128: _rust.signed_pfaffian_and_inverse_c128,
    }

    @staticmethod
    def supports(matrix: torch.Tensor) -> bool:
        """
        Whether the input takes this function.

        :param matrix: Skew-symmetric matrix of shape ``(..., n, n)``.
        :return: ``True`` for an input of a dtype with a kernel, of even ``n >= 2``, at the sizes where the Rust
            inverse is used, when the exact singular gradient is on (:attr:`PfaffianStrategy.EXACT_SINGULAR_GRAD`);
            on another device than the CPU, only up to ``RustPfaffianParlettReid.HOST_BACKWARD_MAX_NUMEL``
            entries, the inputs whose backward runs on the host (``pfaffian()`` sends only small device inputs to
            the host).
        :rtype: bool
        """
        size = matrix.shape[-1]
        if matrix.dtype not in RustPfaffianWithInverse._KERNEL_BY_DTYPE:
            return False
        if matrix.device.type != "cpu" and matrix.numel() > RustPfaffianParlettReid.HOST_BACKWARD_MAX_NUMEL:
            return False  # larger device inputs keep their backward on the device
        if size < 2 or size % 2 or not RustPfaffianParlettReid.EXACT_SINGULAR_GRAD:
            return False
        batch = matrix.numel() // (size * size)
        return size <= PfaffianStrategy.NATIVE_INVERSE_MAX_SIZE or (
            batch >= PfaffianStrategy.NATIVE_INVERSE_PARALLEL_MIN_BATCH
            and size <= PfaffianStrategy.NATIVE_INVERSE_PARALLEL_MAX_SIZE
        )

    @staticmethod
    def forward(ctx: Any, matrix: torch.Tensor) -> torch.Tensor:
        size = matrix.shape[-1]
        kernel = RustPfaffianWithInverse._KERNEL_BY_DTYPE[matrix.dtype]
        host = torch.from_numpy(RustPfaffianParlettReid._host_array(matrix.reshape(-1, size, size)))
        values, inverse, rank_flag, residual_flag = kernel(host.numpy())
        flagged = rank_flag | residual_flag
        ctx.native = (host.reshape(matrix.shape), values, inverse, flagged)
        ctx.template = None
        batch_shape = matrix.shape[:-2]
        if matrix.device.type != "cpu" and not (flagged | (values == 0)).any():  # pragma: no cover - needs a GPU
            # A device input: the gradient is grad * 0.5 * conj(pf(A) A^{-1})^T, so that template goes to the device
            # with the Pfaffians, in one transfer, and the backward is one multiplication there instead of a round
            # trip through the host (two transfers and a synchronization).
            template = 0.5 * values.conj()[:, None, None] * inverse.conj().transpose(0, 2, 1)
            packed = torch.from_numpy(np.concatenate([values, template.astype(values.dtype, copy=False).ravel()]))
            output, template_on_device = packed.to(matrix.device).split((values.size, template.size))
            output = output.view(batch_shape)
            ctx.template = template_on_device.view(matrix.shape)
        else:
            output = torch.from_numpy(values).reshape(batch_shape).to(matrix.device)
        # The output is saved too: a double backward differentiates the backward through it.
        ctx.save_for_backward(matrix, output)
        setattr(output, RustPfaffianParlettReid.HOST_VALUES_ATTRIBUTE, values)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Gradient of the signed Pfaffian, ``0.5 * grad * pf(A) (A^{-1})^T``, from the inverse of the forward.

        :param ctx: Context holding the saved input matrix and output, and the forward's inverses and flags.
        :param grad_output: Gradient of the output with respect to the loss.
        :return: Gradient of the input matrix (the only input, so it requires grad when this runs).
        :rtype: torch.Tensor
        """
        matrix, output = ctx.saved_tensors
        host, values, inverse, flagged = ctx.native
        if torch.is_grad_enabled() or grad_output.requires_grad:  # double backward: the differentiable path
            return RustPfaffianParlettReid.pfaffian_grad_matrix(matrix, output, grad_output)
        if ctx.template is not None:  # pragma: no cover - needs a GPU
            return grad_output.view(*grad_output.shape, 1, 1) * ctx.template  # a view: cheaper than indexing
        gradient = RustPfaffianParlettReid._gradient_from_inverse(host, values, grad_output.cpu(), inverse, flagged)
        return gradient.to(matrix.device)
