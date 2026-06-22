import warnings
from unittest import mock

import numpy as np
import pytest
import torch

import torch_pfaffian
from tests.configs import (
    ATOL_SCALAR_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_SCALAR_COMPARISON,
    TEST_SEED,
)
from torch_pfaffian import get_pfaffian_function, pfaffian, pfaffian_strategy_map
from torch_pfaffian.strategies import PfaffianDet, PfaffianFDBPf, PfaffianParlettReid


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


def _rand_skew_complex(dimension: int, rng: np.random.Generator) -> np.ndarray:
    entries = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
    return entries - entries.T


class TestTorchPfaffian:
    def test_rust_parlett_reid_registered_when_available(self):
        pytest.importorskip("torch_pfaffian._rust")
        from torch_pfaffian.strategies import RustPfaffianParlettReid

        assert RustPfaffianParlettReid is not None
        assert RustPfaffianParlettReid.NAME.lower().strip() in pfaffian_strategy_map
        assert get_pfaffian_function(RustPfaffianParlettReid.NAME) == RustPfaffianParlettReid.apply

    def test_pfaffian_strategy_map_contains_registered_strategies(self):
        assert PfaffianFDBPf.NAME.lower().strip() in pfaffian_strategy_map
        assert PfaffianDet.NAME.lower().strip() in pfaffian_strategy_map
        assert all(issubclass(cls, PfaffianFDBPf.__bases__[0]) for cls in pfaffian_strategy_map.values())

    def test_get_pfaffian_function_returns_apply_of_registered_strategy(self):
        function = get_pfaffian_function(PfaffianFDBPf.NAME)
        assert function == PfaffianFDBPf.apply

    def test_get_pfaffian_function_dispatches_to_pfaffian_det(self):
        assert get_pfaffian_function(PfaffianDet.NAME) == PfaffianDet.apply

    def test_get_pfaffian_function_default_name_is_callable(self):
        function = get_pfaffian_function()
        matrix = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64)
        torch.testing.assert_close(function(matrix), matrix[..., 0, 1])

    def test_get_pfaffian_function_is_case_and_whitespace_insensitive(self):
        function = get_pfaffian_function(f"  {PfaffianFDBPf.NAME.upper()}  ")
        assert function == PfaffianFDBPf.apply

    def test_get_pfaffian_function_unknown_name_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown strategy name"):
            get_pfaffian_function("not_a_real_strategy")

    def test_parlett_reid_is_registered(self):
        assert PfaffianParlettReid.NAME.lower().strip() in pfaffian_strategy_map
        assert get_pfaffian_function(PfaffianParlettReid.NAME) == PfaffianParlettReid.apply

    @staticmethod
    def _skew_matrix() -> torch.Tensor:
        return torch.tensor([[0.0, -3.0], [3.0, 0.0]], dtype=torch.float64)

    def test_pfaffian_signed_by_default_matches_parlett_reid(self):
        matrix = self._skew_matrix()
        torch.testing.assert_close(pfaffian(matrix), PfaffianParlettReid.apply(matrix))
        assert pfaffian(matrix).item() < 0  # signed, not magnitude

    def test_pfaffian_sign_false_returns_magnitude(self):
        matrix = self._skew_matrix()
        torch.testing.assert_close(pfaffian(matrix, sign=False), PfaffianDet.apply(matrix))
        assert pfaffian(matrix, sign=False).item() > 0

    def test_pfaffian_routes_sign_true_to_rust_when_available(self):
        pytest.importorskip("torch_pfaffian._rust")
        with mock.patch.object(torch_pfaffian, "RustPfaffianParlettReid") as fake:
            fake.apply.return_value = torch.zeros(())
            pfaffian(self._skew_matrix(), sign=True)
            fake.apply.assert_called_once()

    def test_pfaffian_routes_sign_true_to_python_when_rust_unavailable(self):
        with (
            mock.patch.object(torch_pfaffian, "RustPfaffianParlettReid", None),
            mock.patch.object(torch_pfaffian, "PfaffianParlettReid") as fake,
        ):
            fake.apply.return_value = torch.zeros(())
            pfaffian(self._skew_matrix(), sign=True)
            fake.apply.assert_called_once()

    def test_pfaffian_sign_true_routes_to_python_on_non_cpu_device(self):
        # The Rust kernel is CPU-only, so a non-CPU input must use the device-native PyTorch strategy.
        pytest.importorskip("torch_pfaffian._rust")
        non_cpu_matrix = mock.MagicMock()
        non_cpu_matrix.device.type = "cuda"
        with (
            mock.patch.object(torch_pfaffian, "RustPfaffianParlettReid") as fake_rust,
            mock.patch.object(torch_pfaffian, "PfaffianParlettReid") as fake_python,
        ):
            fake_python.apply.return_value = torch.zeros(())
            pfaffian(non_cpu_matrix, sign=True)
            fake_python.apply.assert_called_once()
            fake_rust.apply.assert_not_called()

    def test_pfaffian_routes_magnitude_no_grad_to_det(self):
        with mock.patch.object(torch_pfaffian, "PfaffianDet") as fake:
            fake.apply.return_value = torch.zeros(())
            pfaffian(self._skew_matrix(), sign=False)
            fake.apply.assert_called_once()

    def test_pfaffian_routes_magnitude_with_grad_to_fdbpf(self):
        matrix = self._skew_matrix().requires_grad_(True)
        with mock.patch.object(torch_pfaffian, "PfaffianFDBPf") as fake:
            fake.apply.return_value = torch.zeros((), requires_grad=True)
            pfaffian(matrix, sign=False)
            fake.apply.assert_called_once()

    def test_pfaffian_signed_backward_flows(self):
        matrix = self._skew_matrix().requires_grad_(True)
        pfaffian(matrix).backward()
        assert matrix.grad is not None
        assert matrix.grad.shape == matrix.shape

    def test_pfaffian_check_input_rejects_non_skew(self):
        non_skew = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        with pytest.raises(ValueError, match="skew-symmetric"):
            pfaffian(non_skew, check_input=True)

    def test_pfaffian_check_input_rejects_non_square(self):
        non_square = torch.zeros((2, 4), dtype=torch.float64)
        with pytest.raises(ValueError, match="square"):
            pfaffian(non_square, check_input=True)

    def test_pfaffian_check_input_accepts_skew_matrix(self):
        matrix = self._skew_matrix()
        torch.testing.assert_close(pfaffian(matrix, check_input=True), pfaffian(matrix))

    def test_pfaffian_check_input_off_by_default_allows_non_skew(self):
        non_skew = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        pfaffian(non_skew)  # no validation by default, so no error is raised

    @pytest.mark.parametrize("dimension", [2, 4, 6, 8, 10, 12])
    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_pfaffian_sign_true_matches_complex_oracle(self, dimension, dtype):
        rng = np.random.default_rng(TEST_SEED + dimension)
        for _ in range(N_RANDOM_TESTS_PER_CASE):
            skew = _rand_skew_complex(dimension, rng)
            expected = _cofactor_pfaffian(skew)
            matrix = torch.tensor(skew, dtype=dtype)
            result = pfaffian(matrix, sign=True)
            assert result.dtype == dtype
            atol = ATOL_SCALAR_COMPARISON if dtype == torch.complex128 else 1e-4
            torch.testing.assert_close(
                result,
                torch.tensor(expected, dtype=dtype),
                atol=atol,
                rtol=atol,
            )
            assert abs(expected.imag) > 1e-6  # the oracle carries a genuine imaginary part

    def test_pfaffian_sign_true_complex_squares_to_det(self):
        rng = np.random.default_rng(TEST_SEED)
        matrix = torch.tensor(_rand_skew_complex(6, rng), dtype=torch.complex128)
        result = pfaffian(matrix, sign=True)
        torch.testing.assert_close(
            result**2, torch.linalg.det(matrix), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_pfaffian_sign_true_complex_supports_batched_shapes(self):
        rng = np.random.default_rng(TEST_SEED)
        batch = np.stack([_rand_skew_complex(4, rng) for _ in range(6)]).reshape(2, 3, 4, 4)
        matrix = torch.tensor(batch, dtype=torch.complex128)
        result = pfaffian(matrix, sign=True)
        assert result.shape == (2, 3)
        flat = matrix.reshape(-1, 4, 4)
        expected = torch.stack([pfaffian(flat[index], sign=True) for index in range(flat.shape[0])]).reshape(2, 3)
        torch.testing.assert_close(result, expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_pfaffian_sign_true_complex_routes_to_rust_on_cpu(self):
        # The Rust kernel now handles complex natively, so complex CPU inputs take the fast Rust path.
        pytest.importorskip("torch_pfaffian._rust")
        rng = np.random.default_rng(TEST_SEED)
        matrix = torch.tensor(_rand_skew_complex(4, rng), dtype=torch.complex128)
        with mock.patch.object(torch_pfaffian, "RustPfaffianParlettReid") as fake_rust:
            fake_rust.apply.return_value = torch.zeros((), dtype=torch.complex128)
            pfaffian(matrix, sign=True)
            fake_rust.apply.assert_called_once()

    def test_pfaffian_sign_true_complex_emits_no_imaginary_discard_warning(self):
        rng = np.random.default_rng(TEST_SEED)
        matrix = torch.tensor(_rand_skew_complex(6, rng), dtype=torch.complex128)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pfaffian(matrix, sign=True)
        assert not any("imaginary part" in str(warning.message) for warning in caught)

    def test_pfaffian_sign_true_complex_backward_passes_gradcheck(self):
        dimension = 4
        count = dimension * (dimension - 1) // 2
        rng = np.random.default_rng(TEST_SEED)
        values = rng.normal(size=count) + 1j * rng.normal(size=count)
        parameters = torch.tensor(values, dtype=torch.complex128, requires_grad=True)

        def from_parameters(free):
            upper = torch.zeros(dimension, dimension, dtype=free.dtype)
            indices = torch.triu_indices(dimension, dimension, offset=1)
            upper = upper.index_put((indices[0], indices[1]), free)
            return upper - upper.transpose(-1, -2)

        assert torch.autograd.gradcheck(
            lambda free: pfaffian(from_parameters(free), sign=True), (parameters,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_pfaffian_sign_true_complex_odd_dimension_is_zero(self):
        rng = np.random.default_rng(TEST_SEED)
        matrix = torch.tensor(_rand_skew_complex(5, rng), dtype=torch.complex128)
        torch.testing.assert_close(
            pfaffian(matrix, sign=True),
            torch.zeros((), dtype=torch.complex128),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_pfaffian_sign_true_complex_empty_is_one(self):
        matrix = torch.zeros((0, 0), dtype=torch.complex128)
        torch.testing.assert_close(
            pfaffian(matrix, sign=True),
            torch.ones((), dtype=torch.complex128),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_pfaffian_sign_true_complex_singular_is_zero_without_nan(self):
        matrix = torch.zeros((4, 4), dtype=torch.complex128)
        matrix[2, 3] = 1.0 + 1.0j
        matrix[3, 2] = -(1.0 + 1.0j)
        result = pfaffian(matrix, sign=True)
        assert torch.isfinite(result.real) and torch.isfinite(result.imag)
        torch.testing.assert_close(
            result, torch.zeros((), dtype=torch.complex128), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_pfaffian_warns_when_result_overflows(self):
        # A 4x4 block-antidiagonal with huge entries makes the Pfaffian overflow to inf.
        block = torch.tensor([[1e200, 0.0], [0.0, 1e200]], dtype=torch.float64)
        zero = torch.zeros_like(block)
        top = torch.cat([zero, block], dim=-1)
        bottom = torch.cat([-block.transpose(-1, -2), zero], dim=-1)
        matrix = torch.cat([top, bottom], dim=-2)
        with pytest.warns(RuntimeWarning, match="not finite"):
            result = pfaffian(matrix)
        assert not torch.isfinite(result).all()
