import torch


class SkewGradient(torch.autograd.Function):
    r"""
    Identity whose backward pass projects the gradient onto skew-symmetric matrices, ``(g - g^T) / 2``.

    A function of a skew-symmetric matrix only depends on it through its skew-symmetric directions, but how an
    implementation reads the matrix (an upper triangle, a transposed product) decides where autograd puts the
    gradient of a raw leaf matrix. :func:`~torch_pfaffian.pfaffian` returns the skew-split convention
    ``Pf(A) A^{-T} / 2``; wrapping the skew-symmetric inputs of the low-rank updates in this identity gives them
    the same convention. The gradient of any parameter the matrix is built from (keeping it skew-symmetric) is
    unchanged, since only the skew-symmetric part of the gradient enters it.

    The forward pass returns a view of the input and costs nothing; the backward pass costs one pass over the
    ``n x n`` gradient. The input is a matrix of shape ``(..., n, n)``; complex matrices use the transpose, not
    the conjugate transpose.
    """

    @staticmethod
    def forward(matrix: torch.Tensor) -> torch.Tensor:
        return matrix.view_as(matrix)

    @staticmethod
    def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs: tuple, output: torch.Tensor) -> None:
        pass

    @staticmethod
    def backward(ctx: torch.autograd.function.BackwardCFunction, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Skew-symmetric part of the incoming gradient.

        :param ctx: Unused context.
        :param grad_output: Gradient with respect to the output, of shape ``(..., n, n)``.
        :return: ``(grad_output - grad_output^T) / 2``.
        :rtype: torch.Tensor
        """
        return 0.5 * (grad_output - grad_output.mT)

    @staticmethod
    def apply_if_needed(matrix: torch.Tensor) -> torch.Tensor:
        """
        Wrap ``matrix`` in the identity only when a gradient will flow through it, so that inference pays nothing.

        :param matrix: Skew-symmetric matrix of shape ``(..., n, n)``.
        :return: ``matrix``, wrapped when it requires a gradient and gradients are enabled.
        :rtype: torch.Tensor
        """
        if matrix.requires_grad and torch.is_grad_enabled():
            return SkewGradient.apply(matrix)
        return matrix
