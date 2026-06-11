import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

pytest.importorskip("torch_pfaffian._rust")

from tests.configs import (  # noqa: E402
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_APPROX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    TEST_SEED,
)
from torch_pfaffian.strategies.pfaffian_parlett_reid import PfaffianParlettReid  # noqa: E402
from torch_pfaffian.strategies.pfaffian_rust_parlett_reid import RustPfaffianParlettReid  # noqa: E402

_RNG = np.random.default_rng(TEST_SEED)
_HALF_SIZES = [1, 2, 3, 4]
_RANDOM_BLOCKS = [_RNG.random((size, size)) for size in _HALF_SIZES for _ in range(N_RANDOM_TESTS_PER_CASE)]
_GRADCHECK_DIMENSIONS = [2, 4, 6]


def _block_antidiagonal(block: np.ndarray) -> torch.Tensor:
    zero = np.zeros_like(block)
    top = np.concatenate([zero, block], axis=-1)
    bottom = np.concatenate([-np.einsum("...ij->...ji", block), zero], axis=-1)
    return torch.tensor(np.concatenate([top, bottom], axis=-2))


def _random_skew(dimension: int, rng: np.random.Generator) -> torch.Tensor:
    upper = rng.random((dimension, dimension))
    skew = np.triu(upper, k=1)
    return torch.tensor(skew - skew.T)


def _skew_from_parameters(parameters: torch.Tensor, dimension: int) -> torch.Tensor:
    upper = torch.zeros(dimension, dimension, dtype=parameters.dtype)
    indices = torch.triu_indices(dimension, dimension, offset=1)
    upper = upper.index_put((indices[0], indices[1]), parameters)
    return upper - upper.transpose(-1, -2)


class TestRustPfaffianParlettReid:
    @pytest.mark.parametrize("block", _RANDOM_BLOCKS)
    def test_forward_matches_python_parlett_reid(self, block):
        matrix = _block_antidiagonal(block)
        torch.testing.assert_close(
            RustPfaffianParlettReid.apply(matrix),
            PfaffianParlettReid.apply(matrix),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("block", _RANDOM_BLOCKS)
    def test_forward_square_matches_determinant(self, block):
        matrix = _block_antidiagonal(block)
        torch.testing.assert_close(
            RustPfaffianParlettReid.apply(matrix) ** 2,
            torch.linalg.det(matrix),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_forward_two_by_two_is_signed(self):
        matrix = torch.tensor([[0.0, -3.0], [3.0, 0.0]], dtype=torch.float64)
        torch.testing.assert_close(
            RustPfaffianParlettReid.apply(matrix),
            torch.tensor(-3.0, dtype=torch.float64),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_forward_supports_leading_batch_dims(self):
        matrix = _block_antidiagonal(_RNG.random((2, 3, 4, 4)))
        pfaffian = RustPfaffianParlettReid.apply(matrix)
        assert pfaffian.shape == matrix.shape[:-2]
        torch.testing.assert_close(
            pfaffian**2, torch.linalg.det(matrix), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_forward_preserves_dtype_and_device(self, dtype):
        matrix = _random_skew(6, _RNG).to(dtype)
        pfaffian = RustPfaffianParlettReid.apply(matrix)
        assert pfaffian.dtype == dtype
        assert pfaffian.device == matrix.device

    def test_forward_float32_kernel_matches_float64(self):
        # The float32 input must route to the single-precision Rust kernel and agree with the
        # double-precision result within single-precision tolerance.
        matrix_double = _block_antidiagonal(_RNG.random((3, 4, 4)))
        matrix_single = matrix_double.to(torch.float32)
        result_single = RustPfaffianParlettReid.apply(matrix_single)
        result_double = RustPfaffianParlettReid.apply(matrix_double)
        assert result_single.dtype == torch.float32
        torch.testing.assert_close(
            result_single,
            result_double.to(torch.float32),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    def test_forward_odd_dimension_is_zero(self):
        matrix = _random_skew(5, _RNG)
        torch.testing.assert_close(
            RustPfaffianParlettReid.apply(matrix),
            torch.zeros((), dtype=matrix.dtype),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_forward_empty_matrix_is_one(self):
        matrix = torch.zeros((0, 0), dtype=torch.float64)
        torch.testing.assert_close(
            RustPfaffianParlettReid.apply(matrix),
            torch.ones((), dtype=torch.float64),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_forward_singular_matrix_is_zero(self):
        matrix = torch.zeros((4, 4), dtype=torch.float64)
        matrix[2, 3] = 1.0
        matrix[3, 2] = -1.0
        torch.testing.assert_close(
            RustPfaffianParlettReid.apply(matrix),
            torch.zeros((), dtype=torch.float64),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    @pytest.mark.parametrize("dimension", _GRADCHECK_DIMENSIONS)
    def test_backward_passes_gradcheck_on_skew_parameterization(self, dimension):
        count = dimension * (dimension - 1) // 2
        parameters = torch.tensor(_RNG.random(count) + 0.5, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda values: RustPfaffianParlettReid.apply(_skew_from_parameters(values, dimension)),
            (parameters,),
            eps=1e-6,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    def test_backward_exact_on_singular_matrix(self):
        # At an exactly-singular skew input (pf=0) the gradient must be the TRUE derivative, via the
        # Pfaffian adjugate. For the rank-2 matrix with only a_{23}=1, pf = a01*a23 - a02*a13 + a03*a12,
        # so d pf / d a01 = a23 = 1 and the rest are 0.
        parameters = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float64, requires_grad=True)
        RustPfaffianParlettReid.apply(_skew_from_parameters(parameters, 4)).backward()
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        torch.testing.assert_close(parameters.grad, expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_backward_gradcheck_at_singular_point(self):
        # gradcheck centered at an exactly-singular skew point (would fail with the inverse-only form).
        parameters = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda values: RustPfaffianParlettReid.apply(_skew_from_parameters(values, 4)),
            (parameters,),
            eps=1e-6,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    def test_backward_returns_none_when_input_does_not_require_grad(self):
        matrix = _random_skew(4, _RNG)
        pfaffian = RustPfaffianParlettReid.apply(matrix)

        class _Context:
            saved_tensors = (matrix, pfaffian)
            needs_input_grad = (False,)

        assert RustPfaffianParlettReid.backward(_Context(), torch.ones_like(pfaffian)) is None
