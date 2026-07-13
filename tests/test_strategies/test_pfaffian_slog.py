import numpy as np
import torch

from tests.configs import (
    ATOL_SCALAR_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_SCALAR_COMPARISON,
    TEST_SEED,
)
from tests.reference import random_skew
from torch_pfaffian.strategies.pfaffian_parlett_reid import PfaffianParlettReid
from torch_pfaffian.strategies.pfaffian_slog import SlogPfaffianStrategy


def _skew_batch(
    shape: tuple[int, ...], dimension: int, rng: np.random.Generator, dtype: type = np.float64
) -> torch.Tensor:
    count = int(np.prod(shape))
    matrices = np.stack([random_skew(rng, dimension, dtype=dtype) for _ in range(count)])
    return torch.from_numpy(matrices.reshape(*shape, dimension, dimension))


class TestSlogPfaffianStrategy:
    def test_reconstructs_signed_pfaffian(self):
        for dimension in (2, 4, 6, 8):
            rng = np.random.default_rng(TEST_SEED + dimension)
            batch = _skew_batch((N_RANDOM_TESTS_PER_CASE,), dimension, rng)
            phase, log_abs = SlogPfaffianStrategy.apply(batch)
            reconstructed = phase * torch.exp(log_abs)
            expected = PfaffianParlettReid.apply(batch)
            torch.testing.assert_close(
                reconstructed, expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
            )

    def test_reconstructs_signed_pfaffian_complex(self):
        rng = np.random.default_rng(TEST_SEED)
        batch = _skew_batch((N_RANDOM_TESTS_PER_CASE,), 6, rng, dtype=np.complex128)
        phase, log_abs = SlogPfaffianStrategy.apply(batch)
        reconstructed = phase * torch.exp(log_abs).to(phase.dtype)
        expected = PfaffianParlettReid.apply(batch)
        torch.testing.assert_close(reconstructed, expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_phase_is_non_differentiable(self):
        matrix = _skew_batch((2,), 4, np.random.default_rng(TEST_SEED)).requires_grad_(True)
        phase, log_abs = SlogPfaffianStrategy.apply(matrix)
        assert not phase.requires_grad
        assert log_abs.requires_grad

    def test_gradcheck_log_abs_real(self):
        for dimension in (4, 6):
            generator = torch.Generator().manual_seed(TEST_SEED)
            matrix = torch.randn(2, dimension, dimension, dtype=torch.float64, generator=generator, requires_grad=True)
            assert torch.autograd.gradcheck(
                lambda tensor: SlogPfaffianStrategy.apply(tensor - tensor.transpose(-1, -2))[1], (matrix,)
            )

    def test_complex_backward_is_finite(self):
        generator = torch.Generator().manual_seed(TEST_SEED)
        matrix = torch.randn(2, 4, 4, dtype=torch.complex128, generator=generator, requires_grad=True)
        _, log_abs = SlogPfaffianStrategy.apply(matrix - matrix.transpose(-1, -2))
        log_abs.sum().backward()
        assert torch.isfinite(matrix.grad.real).all()
        assert torch.isfinite(matrix.grad.imag).all()

    def test_odd_empty_and_zero(self):
        rng = np.random.default_rng(TEST_SEED)
        odd = _skew_batch((3,), 5, rng)
        phase, log_abs = SlogPfaffianStrategy.apply(odd)
        assert torch.all(phase == 0)
        assert torch.all(log_abs == -torch.inf)
        empty = torch.zeros(2, 0, 0, dtype=torch.float64)
        phase, log_abs = SlogPfaffianStrategy.apply(empty)
        torch.testing.assert_close(phase, torch.ones(2, dtype=torch.float64))
        torch.testing.assert_close(log_abs, torch.zeros(2, dtype=torch.float64))
        zero = torch.zeros(2, 4, 4, dtype=torch.float64)
        phase, log_abs = SlogPfaffianStrategy.apply(zero)
        assert torch.all(phase == 0)
        assert torch.all(log_abs == -torch.inf)

    def test_singular_elements_get_zero_gradient(self):
        rng = np.random.default_rng(TEST_SEED)
        batch = torch.stack(
            [torch.from_numpy(random_skew(rng, 4)), torch.zeros(4, 4, dtype=torch.float64)]
        ).requires_grad_(True)
        _, log_abs = SlogPfaffianStrategy.apply(batch)
        finite = torch.isfinite(log_abs)
        log_abs[finite].sum().backward()
        assert batch.grad is not None
        assert not torch.isnan(batch.grad).any()
        assert torch.all(batch.grad[1] == 0)
        assert torch.any(batch.grad[0] != 0)

    def test_preserves_dtype_and_device(self):
        rng = np.random.default_rng(TEST_SEED)
        for dtype in (np.float32, np.float64, np.complex64, np.complex128):
            batch = _skew_batch((2,), 4, rng, dtype=dtype)
            phase, _ = SlogPfaffianStrategy.apply(batch)
            assert phase.dtype == batch.dtype
            assert phase.device == batch.device
