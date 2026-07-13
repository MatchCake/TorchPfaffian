import numpy as np
import torch

from tests.configs import (
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_MATRIX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    TEST_SEED,
)
from tests.reference import random_skew
from torch_pfaffian import pfaffian
from torch_pfaffian.kernels.det_magnitude import LogMagnitudePfaffian, log_magnitude_pfaffian, magnitude_pfaffian


def _skew_batch(
    shape: tuple[int, ...], dimension: int, rng: np.random.Generator, dtype: type = np.float64
) -> torch.Tensor:
    count = int(np.prod(shape))
    matrices = np.stack([random_skew(rng, dimension, dtype=dtype) for _ in range(count)])
    return torch.from_numpy(matrices.reshape(*shape, dimension, dimension))


def _block_antidiagonal(block: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(block)
    top = torch.cat([zero, block], dim=-1)
    bottom = torch.cat([-block.transpose(-1, -2), zero], dim=-1)
    return torch.cat([top, bottom], dim=-2)


class TestDetMagnitude:
    def test_magnitude_matches_sqrt_abs_det(self):
        for dimension in (2, 4, 6, 8):
            rng = np.random.default_rng(TEST_SEED + dimension)
            batch = _skew_batch((N_RANDOM_TESTS_PER_CASE,), dimension, rng)
            result = magnitude_pfaffian(batch)
            expected = torch.sqrt(torch.abs(torch.linalg.det(batch)))
            torch.testing.assert_close(result, expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_log_magnitude_matches_half_logabsdet(self):
        rng = np.random.default_rng(TEST_SEED)
        batch = _skew_batch((N_RANDOM_TESTS_PER_CASE,), 6, rng)
        result = log_magnitude_pfaffian(batch)
        expected = 0.5 * torch.linalg.slogdet(batch).logabsdet
        torch.testing.assert_close(result, expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_odd_dimension_real_is_neg_inf_and_zero(self):
        matrix = torch.zeros(3, 5, 5, dtype=torch.float64)
        assert torch.all(log_magnitude_pfaffian(matrix) == -torch.inf)
        assert torch.all(magnitude_pfaffian(matrix) == 0.0)

    def test_odd_dimension_complex_returns_real_neg_inf(self):
        matrix = torch.zeros(2, 5, 5, dtype=torch.complex128)
        result = log_magnitude_pfaffian(matrix)
        assert result.dtype == torch.float64
        assert torch.all(result == -torch.inf)

    def test_epsilon_floors_tiny_magnitude(self):
        block = 1e-3 * torch.eye(4, dtype=torch.float64)
        matrix = _block_antidiagonal(block)  # |Pf| = (1e-3)^4 = 1e-12
        floored = magnitude_pfaffian(matrix, epsilon=1e-10)
        torch.testing.assert_close(
            floored, torch.tensor(1e-5, dtype=torch.float64), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )
        unfloored = magnitude_pfaffian(matrix)
        assert unfloored.item() < floored.item()

    def test_gradcheck(self):
        generator = torch.Generator().manual_seed(TEST_SEED)
        matrix = torch.randn(2, 4, 4, dtype=torch.float64, generator=generator, requires_grad=True)
        assert torch.autograd.gradcheck(lambda tensor: magnitude_pfaffian(tensor - tensor.transpose(-1, -2)), (matrix,))
        assert torch.autograd.gradcheck(
            lambda tensor: log_magnitude_pfaffian(tensor - tensor.transpose(-1, -2)), (matrix,)
        )

    def test_log_magnitude_gradient_alive_for_tiny_pfaffians(self):
        # Regression against the sqrt(clamp(|det|, eps)) dead-gradient issue: the log-domain magnitude
        # stays finite and differentiable for a Pfaffian around 1e-60.
        rng = np.random.default_rng(TEST_SEED)
        matrix = torch.from_numpy(random_skew(rng, 8, scale=1e-60)).requires_grad_(True)
        log_pf = log_magnitude_pfaffian(matrix)
        assert torch.isfinite(log_pf).all()
        log_pf.backward()
        assert matrix.grad is not None
        assert torch.any(matrix.grad != 0)
        assert not torch.isnan(matrix.grad).any()

    def test_preserves_device_and_batch_shape(self):
        rng = np.random.default_rng(TEST_SEED)
        batch = _skew_batch((2, 3), 4, rng)
        result = magnitude_pfaffian(batch)
        assert result.shape == (2, 3)
        assert result.device == batch.device

    def test_singular_gradient_is_finite_and_zero_without_nan(self):
        # An exactly-singular element (det == 0) previously made slogdet's backward 0 * inf = NaN. It
        # must now give a finite, exactly-zero gradient, while the invertible element is unchanged.
        rng = np.random.default_rng(TEST_SEED)
        invertible = torch.from_numpy(random_skew(rng, 4))
        corank_two = torch.zeros(4, 4, dtype=torch.float64)
        corank_two[0, 1] = 1.0
        corank_two[1, 0] = -1.0  # rank 2, det == 0 exactly
        zero = torch.zeros(4, 4, dtype=torch.float64)
        batch = torch.stack([invertible, corank_two, zero]).requires_grad_(True)
        magnitude_pfaffian(batch).sum().backward()
        assert not torch.isnan(batch.grad).any()
        assert torch.isfinite(batch.grad).all()
        assert torch.all(batch.grad[1] == 0)
        assert torch.all(batch.grad[2] == 0)
        magnitude = magnitude_pfaffian(invertible)
        expected = magnitude * 0.5 * torch.linalg.inv(invertible).transpose(-1, -2)
        torch.testing.assert_close(batch.grad[0], expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_singular_gradient_complex_is_finite_without_nan(self):
        rng = np.random.default_rng(TEST_SEED)
        invertible = torch.from_numpy(random_skew(rng, 4, dtype=np.complex128))
        zero = torch.zeros(4, 4, dtype=torch.complex128)
        batch = torch.stack([invertible, zero]).requires_grad_(True)
        magnitude_pfaffian(batch).sum().backward()
        assert not torch.isnan(batch.grad.real).any()
        assert not torch.isnan(batch.grad.imag).any()
        assert torch.isfinite(batch.grad.real).all()
        assert torch.isfinite(batch.grad.imag).all()
        assert torch.all(batch.grad[1] == 0)

    def test_log_magnitude_backward_returns_none_without_input_grad(self):
        matrix = torch.zeros(4, 4, dtype=torch.float64)

        class _Context:
            saved_tensors = (matrix, torch.zeros((), dtype=torch.bool))
            needs_input_grad = (False,)

        assert LogMagnitudePfaffian.backward(_Context(), torch.ones(())) is None

    def test_singular_gradient_via_public_epsilon_path(self):
        # m = 8 > AUTO_SMALL_MAX with sign=False and epsilon routes to magnitude_pfaffian; the
        # exactly-singular element must get a finite zero gradient there too.
        block = 1e-3 * torch.eye(4, dtype=torch.float64)
        good = _block_antidiagonal(block)  # 8x8, |Pf| = 1e-12, invertible
        singular = torch.zeros(8, 8, dtype=torch.float64)
        batch = torch.stack([good, singular]).requires_grad_(True)
        pfaffian(batch, sign=False, epsilon=1e-120).sum().backward()
        assert not torch.isnan(batch.grad).any()
        assert torch.isfinite(batch.grad).all()
        assert torch.all(batch.grad[1] == 0)
