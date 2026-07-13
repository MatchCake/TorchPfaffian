import numpy as np
import pytest
import torch

from tests.configs import (
    ATOL_SCALAR_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_SCALAR_COMPARISON,
    TEST_SEED,
)
from tests.reference import pfaffian_combinatorial, random_skew
from torch_pfaffian.kernels.small_n import SMALL_N_MAX, _matching_tables, small_pfaffian


def _skew_batch(
    shape: tuple[int, ...], dimension: int, rng: np.random.Generator, dtype: type = np.float64
) -> torch.Tensor:
    count = int(np.prod(shape))
    matrices = np.stack([random_skew(rng, dimension, dtype=dtype) for _ in range(count)])
    return torch.from_numpy(matrices.reshape(*shape, dimension, dimension))


class TestSmallN:
    def test_matches_combinatorial_real(self):
        for dimension in (2, 4, 6, 8, 10):
            rng = np.random.default_rng(TEST_SEED + dimension)
            batch = _skew_batch((N_RANDOM_TESTS_PER_CASE,), dimension, rng)
            expected = torch.tensor(
                [pfaffian_combinatorial(batch[index].numpy()) for index in range(batch.shape[0])], dtype=batch.dtype
            )
            torch.testing.assert_close(
                small_pfaffian(batch), expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
            )

    def test_matches_combinatorial_complex(self):
        for dimension in (2, 4, 6, 8):
            rng = np.random.default_rng(TEST_SEED + dimension)
            batch = _skew_batch((N_RANDOM_TESTS_PER_CASE,), dimension, rng, dtype=np.complex128)
            expected = torch.tensor(
                [pfaffian_combinatorial(batch[index].numpy()) for index in range(batch.shape[0])], dtype=batch.dtype
            )
            torch.testing.assert_close(
                small_pfaffian(batch), expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
            )

    def test_two_by_two_returns_upper_entry(self):
        matrix = torch.tensor([[0.0, -3.0], [3.0, 0.0]], dtype=torch.float64)
        torch.testing.assert_close(small_pfaffian(matrix), torch.tensor(-3.0, dtype=torch.float64))

    def test_edge_cases(self):
        assert small_pfaffian(torch.zeros(0, 0, dtype=torch.float64)).item() == 1.0
        rng = np.random.default_rng(TEST_SEED)
        odd = _skew_batch((2,), 5, rng)
        torch.testing.assert_close(small_pfaffian(odd), torch.zeros(2, dtype=torch.float64))
        with pytest.raises(ValueError, match="m <= 10"):
            small_pfaffian(torch.zeros(12, 12, dtype=torch.float64))
        with pytest.raises(ValueError, match="square"):
            small_pfaffian(torch.zeros(4, 5, dtype=torch.float64))

    def test_gradcheck_real(self):
        for dimension in (4, 6):
            generator = torch.Generator().manual_seed(TEST_SEED)
            matrix = torch.randn(2, dimension, dimension, dtype=torch.float64, generator=generator, requires_grad=True)
            assert torch.autograd.gradcheck(lambda tensor: small_pfaffian(tensor - tensor.transpose(-1, -2)), (matrix,))

    def test_gradcheck_complex(self):
        generator = torch.Generator().manual_seed(TEST_SEED)
        matrix = torch.randn(2, 4, 4, dtype=torch.complex128, generator=generator, requires_grad=True)
        assert torch.autograd.gradcheck(lambda tensor: small_pfaffian(tensor - tensor.transpose(-1, -2)), (matrix,))

    def test_matching_table_term_counts(self):
        for dimension, count in ((2, 1), (4, 3), (6, 15), (8, 105), (10, 945)):
            rows, cols, signs = _matching_tables(dimension)
            assert len(rows) == count
            assert len(cols) == count
            assert len(signs) == count

    def test_small_n_max_is_ten(self):
        assert SMALL_N_MAX == 10

    def test_preserves_dtype_and_device(self):
        rng = np.random.default_rng(TEST_SEED)
        for dtype in (np.float32, np.float64, np.complex64, np.complex128):
            batch = _skew_batch((2,), 6, rng, dtype=dtype)
            result = small_pfaffian(batch)
            assert result.dtype == batch.dtype
            assert result.device == batch.device

    def test_odd_dimension_backward_returns_zero_grad(self):
        # The odd short-circuit must carry a grad connection so backward returns a zero gradient
        # instead of raising on a constant with no grad_fn.
        generator = torch.Generator().manual_seed(TEST_SEED)
        matrix = torch.randn(2, 5, 5, dtype=torch.float64, generator=generator, requires_grad=True)
        small_pfaffian(matrix).sum().backward()
        assert matrix.grad is not None
        assert torch.isfinite(matrix.grad).all()
        assert torch.all(matrix.grad == 0)

    def test_empty_dimension_backward_returns_zero_grad(self):
        matrix = torch.zeros(2, 0, 0, dtype=torch.float64, requires_grad=True)
        result = small_pfaffian(matrix)
        torch.testing.assert_close(result, torch.ones(2, dtype=torch.float64))
        result.sum().backward()  # must not raise
        assert matrix.grad is not None
        assert matrix.grad.shape == matrix.shape

    def test_odd_dimension_with_non_finite_entries_is_zero(self):
        # The odd-dim Pfaffian is 0 regardless of the entries; inf/nan entries must not contaminate the
        # result (the grad-carrying short-circuit must not compute inf * 0 = nan, including cross-batch).
        single = torch.full((5, 5), float("inf"), dtype=torch.float64)
        result = small_pfaffian(single)
        assert result.item() == 0.0
        assert torch.isfinite(result).all()
        generator = torch.Generator().manual_seed(TEST_SEED)
        finite_odd = torch.randn(5, 5, dtype=torch.float64, generator=generator)
        finite_odd = finite_odd - finite_odd.transpose(-1, -2)
        non_finite_odd = torch.full((5, 5), float("nan"), dtype=torch.float64)
        batched = small_pfaffian(torch.stack([finite_odd, non_finite_odd]))
        torch.testing.assert_close(batched, torch.zeros(2, dtype=torch.float64))
        assert not torch.isnan(batched).any()

    def test_odd_dimension_non_finite_backward_returns_finite_zero_grad(self):
        matrix = torch.full((5, 5), float("inf"), dtype=torch.float64, requires_grad=True)
        small_pfaffian(matrix).sum().backward()
        assert matrix.grad is not None
        assert torch.isfinite(matrix.grad).all()
        assert torch.all(matrix.grad == 0)
