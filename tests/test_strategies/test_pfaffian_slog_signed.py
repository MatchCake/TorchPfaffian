import numpy as np
import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from tests.configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_APPROX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    TEST_SEED,
)
from tests.reference import random_skew
from torch_pfaffian.strategies.pfaffian_parlett_reid import PfaffianParlettReid
from torch_pfaffian.strategies.pfaffian_slog_signed import SlogSignedPfaffian


def _skew_batch(count: int, dimension: int, dtype: type, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.from_numpy(np.stack([random_skew(rng, dimension, dtype=dtype) for _ in range(count)]))


def _skew_from_parameters(parameters: torch.Tensor, dimension: int) -> torch.Tensor:
    # A skew-symmetric A(theta) from its strictly-upper-triangular entries, so that gradcheck differentiates on
    # the manifold where the Pfaffian is defined.
    upper = torch.zeros(dimension, dimension, dtype=parameters.dtype)
    indices = torch.triu_indices(dimension, dimension, offset=1)
    upper = upper.index_put((indices[0], indices[1]), parameters)
    return upper - upper.transpose(-1, -2)


class TestSlogSignedPfaffian:
    @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
    def test_forward_matches_linear_domain(self, dtype):
        batch = _skew_batch(N_RANDOM_TESTS_PER_CASE, 8, dtype, TEST_SEED)
        torch.testing.assert_close(
            SlogSignedPfaffian.apply(batch),
            PfaffianParlettReid.apply(batch),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
    def test_backward_matches_linear_domain(self, dtype):
        # The complex phase's derivative is kept, unlike a reconstruction through SlogPfaffianStrategy.
        batch = _skew_batch(N_RANDOM_TESTS_PER_CASE, 8, dtype, TEST_SEED + 1)
        weights = torch.from_numpy(np.random.default_rng(TEST_SEED + 2).standard_normal(N_RANDOM_TESTS_PER_CASE))
        weights = weights.to(batch.dtype)
        slog_input = batch.clone().requires_grad_(True)
        (SlogSignedPfaffian.apply(slog_input) * weights).real.sum().backward()
        linear_input = batch.clone().requires_grad_(True)
        (PfaffianParlettReid.apply(linear_input) * weights).real.sum().backward()
        torch.testing.assert_close(
            slog_input.grad, linear_input.grad, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    def test_backward_passes_gradcheck_and_gradgradcheck(self, dtype):
        dimension = 6
        count = dimension * (dimension - 1) // 2
        generator = torch.Generator().manual_seed(TEST_SEED)
        parameters = (torch.randn(count, dtype=dtype, generator=generator) + 0.5).requires_grad_(True)

        def function(values: torch.Tensor) -> torch.Tensor:
            return SlogSignedPfaffian.apply(_skew_from_parameters(values, dimension))

        assert gradcheck(function, (parameters,), eps=1e-6, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)
        assert gradgradcheck(
            function, (parameters,), eps=1e-6, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON
        )

    def test_backward_is_zero_on_singular_elements(self):
        batch = _skew_batch(2, 6, np.complex128, TEST_SEED + 3)
        batch[1] = 0.0
        batch.requires_grad_(True)
        SlogSignedPfaffian.apply(batch).real.sum().backward()
        assert batch.grad is not None
        torch.testing.assert_close(batch.grad[1], torch.zeros_like(batch.grad[1]), atol=0.0, rtol=0.0)
        assert torch.isfinite(batch.grad[0]).all()

    def test_supports_leading_batch_dimensions(self):
        batch = _skew_batch(6, 4, np.float64, TEST_SEED + 4).reshape(2, 3, 4, 4).requires_grad_(True)
        result = SlogSignedPfaffian.apply(batch)
        assert result.shape == (2, 3)
        result.sum().backward()
        assert batch.grad is not None and batch.grad.shape == batch.shape
