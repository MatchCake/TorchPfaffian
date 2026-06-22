import warnings

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


def _rand_skew_complex(dimension: int, rng: np.random.Generator) -> np.ndarray:
    entries = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
    return entries - entries.T


def _cofactor_pfaffian(matrix: np.ndarray) -> complex:
    # Exact recursive cofactor expansion; ground-truth oracle for complex skew-symmetric inputs.
    dimension = matrix.shape[0]
    if dimension == 0:
        return 1 + 0j
    if dimension % 2:
        return 0j
    if dimension == 2:
        return matrix[0, 1]
    total = 0j
    rest = list(range(1, dimension))
    for position, column in enumerate(rest):
        sub = [index for index in rest if index != column]
        total += (-1) ** position * matrix[0, column] * _cofactor_pfaffian(matrix[np.ix_(sub, sub)])
    return total


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

    @pytest.mark.parametrize("dimension", [2, 4, 6, 8])
    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_forward_complex_matches_cofactor_oracle(self, dimension, dtype):
        rng = np.random.default_rng(TEST_SEED + dimension)
        skew = _rand_skew_complex(dimension, rng)
        matrix = torch.tensor(skew, dtype=dtype)
        result = RustPfaffianParlettReid.apply(matrix)
        assert result.dtype == dtype
        atol = ATOL_SCALAR_COMPARISON if dtype == torch.complex128 else 1e-4
        torch.testing.assert_close(result, torch.tensor(_cofactor_pfaffian(skew), dtype=dtype), atol=atol, rtol=atol)

    def test_forward_complex_matches_python_parlett_reid(self):
        rng = np.random.default_rng(TEST_SEED)
        matrix = torch.tensor(_rand_skew_complex(6, rng), dtype=torch.complex128)
        torch.testing.assert_close(
            RustPfaffianParlettReid.apply(matrix),
            PfaffianParlettReid.apply(matrix),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_forward_complex_emits_no_imaginary_discard_warning(self):
        rng = np.random.default_rng(TEST_SEED)
        matrix = torch.tensor(_rand_skew_complex(6, rng), dtype=torch.complex128)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            RustPfaffianParlettReid.apply(matrix)
        assert not any("imaginary part" in str(warning.message) for warning in caught)

    @pytest.mark.parametrize("dimension", _GRADCHECK_DIMENSIONS)
    def test_backward_complex_passes_gradcheck_on_skew_parameterization(self, dimension):
        count = dimension * (dimension - 1) // 2
        rng = np.random.default_rng(TEST_SEED + dimension)
        values = rng.normal(size=count) + 1j * rng.normal(size=count)
        parameters = torch.tensor(values, dtype=torch.complex128, requires_grad=True)
        assert gradcheck(
            lambda free: RustPfaffianParlettReid.apply(_skew_from_parameters(free, dimension)),
            (parameters,),
            eps=1e-6,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    def test_forward_float16_runs_natively_in_half_precision(self):
        # float16 is computed by the native half-precision kernel (the caller opts into half-precision
        # risk); the result agrees with the double-precision Pfaffian of the same f16 values at f16
        # tolerance and preserves the float16 dtype.
        half = _random_skew(4, _RNG).to(torch.float16)
        result = RustPfaffianParlettReid.apply(half)
        assert result.dtype == torch.float16
        reference = RustPfaffianParlettReid.apply(half.to(torch.float64))
        torch.testing.assert_close(
            result.to(torch.float64), reference, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON
        )

    def test_forward_unsupported_dtype_raises_with_guidance(self):
        # Unsupported dtypes must raise rather than be silently downcast; the message points at the
        # pure-PyTorch fallback and the issue tracker.
        matrix = torch.tensor([[0.0, 1.0], [-1.0, 0.0]]).to(torch.bfloat16)
        with pytest.raises(TypeError, match="no Rust kernel for dtype.*PfaffianParlettReid.*issues"):
            RustPfaffianParlettReid.apply(matrix)

    def test_backward_returns_none_when_input_does_not_require_grad(self):
        matrix = _random_skew(4, _RNG)
        pfaffian = RustPfaffianParlettReid.apply(matrix)

        class _Context:
            saved_tensors = (matrix, pfaffian)
            needs_input_grad = (False,)

        assert RustPfaffianParlettReid.backward(_Context(), torch.ones_like(pfaffian)) is None
