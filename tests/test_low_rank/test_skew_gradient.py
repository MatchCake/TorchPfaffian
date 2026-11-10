import torch

from tests.configs import ATOL_MATRIX_COMPARISON, RTOL_MATRIX_COMPARISON, TEST_SEED
from torch_pfaffian.low_rank import SkewGradient


class TestSkewGradient:
    def test_forward_is_the_identity_and_backward_the_skew_part(self):
        generator = torch.Generator().manual_seed(TEST_SEED)
        matrix = torch.randn(3, 5, 5, generator=generator, dtype=torch.float64, requires_grad=True)
        weights = torch.randn(3, 5, 5, generator=generator, dtype=torch.float64)
        output = SkewGradient.apply(matrix)
        torch.testing.assert_close(output, matrix, atol=0, rtol=0)
        (output * weights).sum().backward()
        torch.testing.assert_close(
            matrix.grad, 0.5 * (weights - weights.mT), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_complex_uses_the_transpose(self):
        generator = torch.Generator().manual_seed(TEST_SEED)
        real, imaginary = torch.randn(2, 4, 4, generator=generator, dtype=torch.float64)
        matrix = torch.complex(real, imaginary).requires_grad_(True)
        weights = torch.complex(imaginary, real)
        (SkewGradient.apply(matrix) * weights).real.sum().backward()
        expected = matrix.detach().clone().requires_grad_(True)
        (expected * weights).real.sum().backward()
        torch.testing.assert_close(
            matrix.grad, 0.5 * (expected.grad - expected.grad.mT), atol=ATOL_MATRIX_COMPARISON, rtol=0
        )

    def test_apply_if_needed_only_wraps_tensors_that_need_a_gradient(self):
        matrix = torch.zeros(4, 4, dtype=torch.float64)
        assert SkewGradient.apply_if_needed(matrix) is matrix
        leaf = matrix.clone().requires_grad_(True)
        with torch.no_grad():
            assert SkewGradient.apply_if_needed(leaf) is leaf
        wrapped = SkewGradient.apply_if_needed(leaf)
        assert wrapped is not leaf and wrapped.grad_fn is not None
