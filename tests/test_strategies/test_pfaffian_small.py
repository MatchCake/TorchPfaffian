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
from tests.reference import pfaffian_combinatorial, random_skew
from torch_pfaffian import get_pfaffian_function, pfaffian_strategy_map
from torch_pfaffian.strategies.pfaffian_parlett_reid import PfaffianParlettReid
from torch_pfaffian.strategies.pfaffian_small import PfaffianSmall


def _skew_batch(
    shape: tuple[int, ...], dimension: int, rng: np.random.Generator, dtype: type = np.float64
) -> torch.Tensor:
    count = int(np.prod(shape))
    matrices = np.stack([random_skew(rng, dimension, dtype=dtype) for _ in range(count)])
    return torch.from_numpy(matrices.reshape(*shape, dimension, dimension))


class TestPfaffianSmall:
    def test_matches_combinatorial_oracle_real(self):
        for dimension in (2, 4, 6):
            rng = np.random.default_rng(TEST_SEED + dimension)
            batch = _skew_batch((N_RANDOM_TESTS_PER_CASE,), dimension, rng)
            expected = torch.tensor(
                [pfaffian_combinatorial(batch[index].numpy()) for index in range(batch.shape[0])], dtype=batch.dtype
            )
            torch.testing.assert_close(
                PfaffianSmall.apply(batch), expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
            )

    def test_matches_combinatorial_oracle_complex(self):
        rng = np.random.default_rng(TEST_SEED)
        batch = _skew_batch((N_RANDOM_TESTS_PER_CASE,), 6, rng, dtype=np.complex128)
        expected = torch.tensor(
            [pfaffian_combinatorial(batch[index].numpy()) for index in range(batch.shape[0])], dtype=batch.dtype
        )
        torch.testing.assert_close(
            PfaffianSmall.apply(batch), expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_name_and_registration(self):
        assert PfaffianSmall.NAME == "PfaffianSmall"
        assert PfaffianSmall.NAME.lower().strip() in pfaffian_strategy_map
        assert get_pfaffian_function(PfaffianSmall.NAME) == PfaffianSmall.apply

    def test_gradcheck_real(self):
        # gradcheck on full (non-skew) inputs exercises the internal re-skew-symmetrization.
        for dimension in (2, 4, 6):
            generator = torch.Generator().manual_seed(TEST_SEED)
            matrix = torch.randn(2, dimension, dimension, dtype=torch.float64, generator=generator, requires_grad=True)
            assert torch.autograd.gradcheck(PfaffianSmall.apply, (matrix,))

    def test_gradcheck_complex(self):
        generator = torch.Generator().manual_seed(TEST_SEED)
        matrix = torch.randn(2, 4, 4, dtype=torch.complex128, generator=generator, requires_grad=True)
        assert torch.autograd.gradcheck(PfaffianSmall.apply, (matrix,))

    def test_signed_gradient_matches_analytic_formula(self):
        # The re-skew-symmetrization makes the gradient of a full-matrix skew input equal the analytic
        # VJP 0.5 * pf(A) * (A^{-1})^T (skew-split), rather than piling in the upper triangle.
        rng = np.random.default_rng(TEST_SEED)
        matrix = torch.from_numpy(random_skew(rng, 6)).requires_grad_(True)
        pfaffian = PfaffianSmall.apply(matrix)
        pfaffian.backward()
        expected = 0.5 * pfaffian.detach() * torch.linalg.inv(matrix.detach()).transpose(-1, -2)
        torch.testing.assert_close(matrix.grad, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_odd_and_empty(self):
        rng = np.random.default_rng(TEST_SEED)
        odd = _skew_batch((2,), 5, rng)
        torch.testing.assert_close(PfaffianSmall.apply(odd), torch.zeros(2, dtype=torch.float64))
        empty = torch.zeros(2, 0, 0, dtype=torch.float64)
        torch.testing.assert_close(PfaffianSmall.apply(empty), torch.ones(2, dtype=torch.float64))

    def test_two_by_two_is_signed(self):
        matrix = torch.tensor([[0.0, -3.0], [3.0, 0.0]], dtype=torch.float64)
        torch.testing.assert_close(PfaffianSmall.apply(matrix), torch.tensor(-3.0, dtype=torch.float64))

    def test_two_by_two_gradient_matches_parlett_reid(self):
        # Re-skewing at m == 2 gives the skew-split gradient [[0, 0.5], [-0.5, 0]] (matching every other
        # strategy) rather than the upper-triangle [[0, 1], [0, 0]] of the raw unrolled kernel.
        for dtype in (torch.float64, torch.complex128):
            small_input = torch.tensor([[0.0, -3.0], [3.0, 0.0]], dtype=dtype, requires_grad=True)
            reference_input = small_input.detach().clone().requires_grad_(True)
            small_output = PfaffianSmall.apply(small_input)
            small_output.backward(torch.ones_like(small_output))
            reference_output = PfaffianParlettReid.apply(reference_input)
            reference_output.backward(torch.ones_like(reference_output))
            torch.testing.assert_close(
                small_input.grad, reference_input.grad, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
            )

    def test_preserves_dtype_and_device(self):
        rng = np.random.default_rng(TEST_SEED)
        for dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128):
            batch = _skew_batch((2,), 6, rng).to(dtype)
            result = PfaffianSmall.apply(batch)
            assert result.dtype == dtype
            assert result.device == batch.device
