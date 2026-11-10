import warnings
from unittest import mock

import numpy as np
import pytest
import torch

import torch_pfaffian
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
from torch_pfaffian import get_pfaffian_function, pfaffian, pfaffian_strategy_map
from torch_pfaffian.strategies import (
    PfaffianBlockDet,
    PfaffianDet,
    PfaffianFDBPf,
    PfaffianMaskedMagnitude,
    PfaffianParlettReid,
    PfaffianSmall,
)

# pfaffian() keyword sets covering each route a low-precision input is upcast into.
_LOW_PRECISION_CASES = [
    {"sign": True},
    {"sign": False},
    {"sign": False, "epsilon": 1e-6},
    {"sign": False, "mask_roundoff_gradient": True},
]


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


def _block_antidiagonal(block: torch.Tensor) -> torch.Tensor:
    # Skew matrix [[0, block], [-block^T, 0]] whose Pfaffian magnitude is |det(block)|.
    zero = torch.zeros_like(block)
    top = torch.cat([zero, block], dim=-1)
    bottom = torch.cat([-block.transpose(-1, -2), zero], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def _block_diagonal_skew(coefficients: list[float]) -> torch.Tensor:
    # Block-diagonal skew matrix of 2x2 blocks [[0, c], [-c, 0]]; its Pfaffian is the product of the c.
    dimension = 2 * len(coefficients)
    matrix = torch.zeros(dimension, dimension, dtype=torch.float64)
    for index, coefficient in enumerate(coefficients):
        matrix[2 * index, 2 * index + 1] = coefficient
        matrix[2 * index + 1, 2 * index] = -coefficient
    return matrix


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

    @staticmethod
    def _large_skew_matrix() -> torch.Tensor:
        # A skew matrix larger than AUTO_SMALL_MAX so dispatch bypasses the small-m kernel and the
        # Rust/Parlett-Reid/Det/FDBPf routing under test is exercised.
        generator = torch.Generator().manual_seed(0)
        full = torch.randn(8, 8, dtype=torch.float64, generator=generator)
        return full - full.transpose(-1, -2)

    @staticmethod
    def _non_cpu_matrix() -> mock.MagicMock:
        matrix = mock.MagicMock()
        matrix.device.type = "cuda"
        matrix.shape = (8, 8)  # larger than AUTO_SMALL_MAX so the small-m kernel is bypassed
        matrix.numel.return_value = 64  # below AUTO_BLOCK_DET_MIN_NUMEL, so no block-form check
        matrix.dtype = torch.float64
        return matrix

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
            pfaffian(self._large_skew_matrix(), sign=True)
            fake.apply.assert_called_once()

    def test_pfaffian_routes_sign_true_to_python_when_rust_unavailable(self):
        with (
            mock.patch.object(torch_pfaffian, "RustPfaffianParlettReid", None),
            mock.patch.object(torch_pfaffian, "PfaffianParlettReid") as fake,
        ):
            fake.apply.return_value = torch.zeros(())
            pfaffian(self._large_skew_matrix(), sign=True)
            fake.apply.assert_called_once()

    @pytest.mark.parametrize(
        ("host_max", "capturing", "on_host"),
        [(torch_pfaffian.AUTO_HOST_MAX_BATCH_SQUARE, False, True), (63, False, False), (64, True, False)],
    )
    def test_pfaffian_sign_true_on_non_cpu_device_takes_the_host_only_when_small(self, host_max, capturing, on_host):
        # Small inputs take the Rust kernel through the host (the device elimination is launch-bound); large
        # ones, and inputs during a CUDA graph capture (which cannot copy to the host), stay on the device.
        pytest.importorskip("torch_pfaffian._rust")
        with (
            mock.patch.object(torch_pfaffian, "AUTO_HOST_MAX_BATCH_SQUARE", host_max),
            mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=capturing),
            mock.patch.object(torch_pfaffian, "RustPfaffianParlettReid") as fake_rust,
            mock.patch.object(torch_pfaffian, "PfaffianParlettReid") as fake_python,
        ):
            fake_rust.apply.return_value = torch.zeros(())
            fake_python.apply.return_value = torch.zeros(())
            pfaffian(self._non_cpu_matrix(), sign=True)
            assert fake_rust.apply.call_count == int(on_host)
            assert fake_python.apply.call_count == int(not on_host)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    def test_pfaffian_on_cuda_matches_cpu_on_both_routes(self, dtype):  # pragma: no cover - needs a GPU
        generator = torch.Generator().manual_seed(TEST_SEED)
        entries = torch.randn(4, 12, 12, generator=generator, dtype=torch.float64).to(dtype)
        matrix = (entries - entries.mT).requires_grad_(True)
        expected = pfaffian(matrix)
        (expected_gradient,) = torch.autograd.grad(expected.real.sum(), matrix)
        for host_max in (torch_pfaffian.AUTO_HOST_MAX_BATCH_SQUARE, 0):  # through the host, then on the device
            with mock.patch.object(torch_pfaffian, "AUTO_HOST_MAX_BATCH_SQUARE", host_max):
                on_cuda = matrix.detach().cuda().requires_grad_(True)
                result = pfaffian(on_cuda)
                (gradient,) = torch.autograd.grad(result.real.sum(), on_cuda)
            assert result.device.type == "cuda" and gradient.device.type == "cuda"
            torch.testing.assert_close(
                result.cpu(), expected.detach(), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
            )
            torch.testing.assert_close(
                gradient.cpu(), expected_gradient, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
            )

    def test_pfaffian_routes_magnitude_no_grad_to_det(self):
        with mock.patch.object(torch_pfaffian, "PfaffianDet") as fake:
            fake.apply.return_value = torch.zeros(())
            pfaffian(self._large_skew_matrix(), sign=False)
            fake.apply.assert_called_once()

    def test_pfaffian_routes_magnitude_with_grad_to_fdbpf(self):
        matrix = self._large_skew_matrix().requires_grad_(True)
        with mock.patch.object(torch_pfaffian, "PfaffianFDBPf") as fake:
            fake.apply.return_value = torch.zeros((), requires_grad=True)
            pfaffian(matrix, sign=False)
            fake.apply.assert_called_once()

    def test_pfaffian_routes_small_dimension_to_pfaffian_small(self):
        with mock.patch.object(torch_pfaffian, "PfaffianSmall") as fake:
            fake.apply.return_value = torch.zeros(())
            pfaffian(self._skew_matrix(), sign=True)
            fake.apply.assert_called_once()

    def test_pfaffian_routes_small_dimension_magnitude_to_pfaffian_small(self):
        with mock.patch.object(torch_pfaffian, "PfaffianSmall") as fake:
            fake.apply.return_value = torch.zeros(())
            pfaffian(self._skew_matrix(), sign=False)
            fake.apply.assert_called_once()

    def test_pfaffian_signed_backward_flows(self):
        matrix = self._skew_matrix().requires_grad_(True)
        pfaffian(matrix).backward()
        assert matrix.grad is not None
        assert matrix.grad.shape == matrix.shape

    def test_pfaffian_small_odd_dimension_backward_returns_zero_grad(self):
        # Odd m <= AUTO_SMALL_MAX with grad must not raise (the small path once returned a constant with
        # no grad_fn); backward returns a finite zero gradient, matching the large-m Parlett-Reid path.
        generator = torch.Generator().manual_seed(1)
        full = torch.randn(5, 5, dtype=torch.float64, generator=generator)
        matrix = (full - full.transpose(-1, -2)).requires_grad_(True)
        pfaffian(matrix, sign=True).sum().backward()
        assert matrix.grad is not None
        assert torch.isfinite(matrix.grad).all()
        assert torch.all(matrix.grad == 0)

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
        # The Rust kernel now handles complex natively, so complex CPU inputs above AUTO_SMALL_MAX take
        # the fast Rust path (smaller ones take the exact unrolled kernel).
        pytest.importorskip("torch_pfaffian._rust")
        rng = np.random.default_rng(TEST_SEED)
        matrix = torch.tensor(_rand_skew_complex(8, rng), dtype=torch.complex128)
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

    @pytest.mark.parametrize("epsilon", [None, 1e-32])
    def test_pfaffian_sign_false_complex_backward_passes_gradcheck(self, epsilon):
        # Regression for issue #17: m = 8 > AUTO_SMALL_MAX with grad routes the default (epsilon=None)
        # magnitude through PfaffianFDBPf, whose backward raised on complex input.
        dimension = 8
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
            lambda free: pfaffian(from_parameters(free), sign=False, epsilon=epsilon),
            (parameters,),
            eps=1e-6,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
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
        matrix = _block_antidiagonal(block)
        with pytest.warns(RuntimeWarning, match="not finite"):
            result = pfaffian(matrix)
        assert not torch.isfinite(result).all()

    def test_pfaffian_check_finite_false_skips_overflow_warning(self):
        # check_finite=False skips the mandatory isfinite host sync (and its warning).
        block = torch.tensor([[1e200, 0.0], [0.0, 1e200]], dtype=torch.float64)
        matrix = _block_antidiagonal(block)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = pfaffian(matrix, check_finite=False)
        assert not any("not finite" in str(warning.message) for warning in caught)
        assert not torch.isfinite(result).all()  # the overflow still happens; only the check is skipped

    def test_pfaffian_auto_recovers_signed_transient_overflow(self):
        # A wide-dynamic-range input whose linear product overflows mid-way while the true Pfaffian is
        # finite is transparently recovered in the log domain: the default path returns the finite value
        # with no warning, while check_finite=False exposes the raw inf. No user action (no slog) needed.
        matrix = _block_diagonal_skew([1e200, 1e200, 1e-29, 1e-29, 1e-29, 1e-29, 1e-29, 1e-29])
        raw = pfaffian(matrix, sign=True, check_finite=False)
        assert not torch.isfinite(raw)  # the raw fast path overflows to inf
        phase, log_abs = torch_pfaffian.slog_pfaffian(matrix)
        expected = phase * torch.exp(log_abs).to(phase.dtype)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            recovered = pfaffian(matrix, sign=True)
        assert torch.isfinite(recovered)
        assert not any(issubclass(warning.category, RuntimeWarning) for warning in caught)
        torch.testing.assert_close(recovered, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_pfaffian_auto_recovers_signed_transient_overflow_gradient_is_finite(self):
        # The recovered value stays differentiable: the linear graph (whose saved inf Pfaffian would
        # give 0 * inf = NaN in its backward) is discarded in favor of the log-domain reconstruction.
        matrix = _block_diagonal_skew([1e200, 1e200, 1e-29, 1e-29, 1e-29, 1e-29, 1e-29, 1e-29])
        matrix.requires_grad_(True)
        pfaffian(matrix, sign=True).backward()
        assert matrix.grad is not None
        assert torch.isfinite(matrix.grad).all()
        assert torch.any(matrix.grad != 0)

    def test_pfaffian_auto_recovers_magnitude_transient_overflow(self):
        # The magnitude path (sign=False) recovers overflow the same way, via the log-domain magnitude.
        matrix = _block_diagonal_skew([1e200, 1e200, 1e-29, 1e-29, 1e-29, 1e-29, 1e-29, 1e-29])
        raw = pfaffian(matrix, sign=False, check_finite=False)
        assert not torch.isfinite(raw)
        expected = torch.exp(torch_pfaffian.log_magnitude_pfaffian(matrix))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            recovered = pfaffian(matrix, sign=False)
        assert torch.isfinite(recovered)
        assert not any(issubclass(warning.category, RuntimeWarning) for warning in caught)
        torch.testing.assert_close(recovered, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_pfaffian_magnitude_epsilon_genuinely_huge_still_warns(self):
        # sign=False with epsilon on a genuinely out-of-range magnitude: the log-domain recovery
        # re-applies the sqrt(epsilon) floor and, since the true value exceeds the dtype range, warns.
        matrix = _block_antidiagonal(torch.tensor([[1e200, 0.0], [0.0, 1e200]], dtype=torch.float64))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = pfaffian(matrix, sign=False, epsilon=1e-10)
        assert not torch.isfinite(result)
        assert any(issubclass(warning.category, RuntimeWarning) for warning in caught)

    def test_auto_small_max_constant(self):
        assert torch_pfaffian.AUTO_SMALL_MAX == 6

    def test_auto_block_det_min_numel_constant(self):
        assert torch_pfaffian.AUTO_BLOCK_DET_MIN_NUMEL == 2**14

    @staticmethod
    def _large_block_form_batch(dtype: torch.dtype = torch.float64) -> torch.Tensor:
        # 256 matrices of size 8x8 = 2**14 entries, the smallest input routed to PfaffianBlockDet.
        generator = torch.Generator().manual_seed(TEST_SEED)
        block = torch.randn(256, 4, 4, dtype=dtype, generator=generator)
        return _block_antidiagonal(block)

    @pytest.mark.parametrize("sign", [True, False])
    def test_pfaffian_routes_large_block_form_to_block_det(self, sign):
        with mock.patch.object(torch_pfaffian, "PfaffianBlockDet", wraps=PfaffianBlockDet) as fake:
            pfaffian(self._large_block_form_batch(), sign=sign)
            fake.apply.assert_called_once()

    def test_pfaffian_block_form_with_epsilon_keeps_magnitude_path(self):
        # The log-domain magnitude keeps gradients alive below the floor, so it is not replaced.
        with mock.patch.object(torch_pfaffian, "PfaffianBlockDet", wraps=PfaffianBlockDet) as fake:
            pfaffian(self._large_block_form_batch(), sign=False, epsilon=1e-20)
            fake.apply.assert_not_called()

    def test_pfaffian_small_block_form_input_is_not_routed_to_block_det(self):
        with mock.patch.object(torch_pfaffian, "PfaffianBlockDet", wraps=PfaffianBlockDet) as fake:
            pfaffian(self._large_block_form_batch()[:255], sign=True)
            fake.is_block_form.assert_not_called()
            fake.apply.assert_not_called()

    def test_pfaffian_large_general_input_is_not_routed_to_block_det(self):
        general = self._large_block_form_batch().clone()
        general[-1] = self._large_skew_matrix()
        with mock.patch.object(torch_pfaffian, "PfaffianBlockDet", wraps=PfaffianBlockDet) as fake:
            result = pfaffian(general, sign=True)
            fake.apply.assert_not_called()
        torch.testing.assert_close(
            result, PfaffianParlettReid.apply(general), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_pfaffian_low_precision_block_form_is_upcast_and_routed_to_block_det(self):
        # float16 has no LU determinant: the input is upcast to float32, which the block route accepts.
        dtype = torch.float16
        matrix = self._large_block_form_batch().to(dtype)
        with mock.patch.object(torch_pfaffian, "PfaffianBlockDet", wraps=PfaffianBlockDet) as fake:
            with pytest.warns(UserWarning, match="computes torch.float16 inputs in torch.float32"):
                result = pfaffian(matrix, sign=True)
            fake.apply.assert_called_once()
        assert result.dtype == dtype

    def test_pfaffian_block_form_magnitude_keeps_determinant_floor(self):
        # The magnitude route keeps the sqrt(PfaffianStrategy.EPSILON) floor of PfaffianDet/PfaffianFDBPf.
        matrix = self._large_block_form_batch()
        matrix[0] = 0.0
        result = pfaffian(matrix, sign=False)
        floor = torch.tensor(PfaffianDet.EPSILON**0.5, dtype=torch.float64)
        torch.testing.assert_close(result[0], floor, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)
        torch.testing.assert_close(
            result[1:], PfaffianDet.apply(matrix[1:]), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    @pytest.mark.parametrize("sign", [True, False])
    def test_pfaffian_block_form_route_matches_general_value_and_gradient(self, dtype, sign):
        matrix = self._large_block_form_batch(dtype)
        routed_input = matrix.clone().requires_grad_(True)
        general_input = matrix.clone().requires_grad_(True)
        routed = pfaffian(routed_input, sign=sign)
        general = PfaffianParlettReid.apply(general_input)
        if not sign:
            general = general.abs()
        torch.testing.assert_close(routed, general, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)
        routed.real.sum().backward()
        general.real.sum().backward()
        torch.testing.assert_close(
            routed_input.grad, general_input.grad, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_slog_pfaffian_reconstructs_signed_pfaffian(self):
        # The public slog wrapper returns (phase, log|Pf|) reconstructing the signed Pfaffian.
        rng = np.random.default_rng(TEST_SEED)
        matrix = torch.tensor(_rand_skew_complex(8, rng), dtype=torch.complex128)
        phase, log_abs = torch_pfaffian.slog_pfaffian(matrix)
        reconstructed = phase * torch.exp(log_abs).to(phase.dtype)
        torch.testing.assert_close(
            reconstructed, pfaffian(matrix, sign=True), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_pfaffian_epsilon_floors_magnitude_on_determinant_path(self):
        # m = 8 > AUTO_SMALL_MAX, sign=False, epsilon given: the log-domain magnitude path floors at
        # sqrt(epsilon), whereas the default (epsilon=None) returns the true tiny magnitude.
        block = 1e-3 * torch.eye(4, dtype=torch.float64)
        matrix = _block_antidiagonal(block)  # 8x8 skew, |Pf| = (1e-3)^4 = 1e-12
        floored = pfaffian(matrix, sign=False, epsilon=1e-10)
        torch.testing.assert_close(
            floored, torch.tensor(1e-5, dtype=torch.float64), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )
        unfloored = pfaffian(matrix, sign=False)
        assert unfloored.item() < floored.item()

    @staticmethod
    def _round_off_zero(dimension: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        # Rotated block-diagonal skew matrix with entries of size 10 and one block at 0.25 * eps * 10: |Pf|
        # vanishes up to round-off yet sits above the default sqrt(PfaffianStrategy.EPSILON) floor.
        coefficients = [2.5 * torch.finfo(dtype).eps] + [10.0] * (dimension // 2 - 1)
        block = _block_diagonal_skew(coefficients)
        generator = torch.Generator().manual_seed(TEST_SEED)
        orthogonal, _ = torch.linalg.qr(torch.randn(dimension, dimension, dtype=torch.float64, generator=generator))
        return (orthogonal @ block @ orthogonal.T).to(dtype)

    @pytest.mark.parametrize("dimension", [4, 6, 8, 12, 16, 24])
    def test_pfaffian_mask_roundoff_gradient_zeroes_round_off_gradient_at_every_size(self, dimension):
        # The unfloored magnitude sqrt(|det A|) gives the round-off zero a garbage gradient: O(1) noise where
        # the platform's LAPACK computes a round-off determinant, NaN where it computes exactly 0 (Windows).
        # The masked path gives zero, for small and large m.
        unmasked_input = self._round_off_zero(dimension).requires_grad_(True)
        torch.sqrt(torch.abs(torch.linalg.det(unmasked_input))).backward()
        assert (~(unmasked_input.grad.abs() <= ATOL_APPROX_COMPARISON)).any()  # some entry is large or NaN
        masked_input = self._round_off_zero(dimension).requires_grad_(True)
        pfaffian(masked_input, sign=False, mask_roundoff_gradient=True).backward()
        assert torch.all(masked_input.grad == 0)

    @pytest.mark.parametrize("dimension", [4, 8])
    def test_pfaffian_mask_roundoff_gradient_routes_to_masked_magnitude(self, dimension):
        matrix = self._round_off_zero(dimension)
        with mock.patch.object(torch_pfaffian, "PfaffianMaskedMagnitude", wraps=PfaffianMaskedMagnitude) as fake:
            pfaffian(matrix, sign=False, mask_roundoff_gradient=True, epsilon=1e-20)
            fake.apply.assert_called_once_with(matrix, 1e-20)

    def test_pfaffian_mask_roundoff_gradient_routes_block_form_to_masked_magnitude(self):
        generator = torch.Generator().manual_seed(TEST_SEED)
        matrix = _block_antidiagonal(torch.randn(256, 4, 4, dtype=torch.float64, generator=generator))
        with mock.patch.object(torch_pfaffian, "PfaffianBlockDet", wraps=PfaffianBlockDet) as fake:
            pfaffian(matrix, sign=False, mask_roundoff_gradient=True)
            fake.apply.assert_not_called()

    def test_pfaffian_mask_roundoff_gradient_with_sign_raises(self):
        with pytest.raises(ValueError, match="sign=False"):
            pfaffian(self._round_off_zero(4), sign=True, mask_roundoff_gradient=True)

    def test_pfaffian_mask_roundoff_gradient_uses_epsilon_floor(self):
        matrix = torch.zeros(8, 8, dtype=torch.float64)
        result = pfaffian(matrix, sign=False, mask_roundoff_gradient=True, epsilon=1e-32)
        torch.testing.assert_close(
            result, torch.tensor(1e-16, dtype=torch.float64), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_pfaffian_mask_roundoff_gradient_keeps_mask_when_another_element_overflows(self):
        # Regression: the check_finite recovery replaced the whole batch's graph by the unmasked log-domain
        # one when any element overflowed, so the round-off zero next to it got its O(1) gradient back.
        overflow = _block_diagonal_skew([1e200, 1e200, 1e-200, 1e-200])  # true |Pf| = 1, det overflows
        matrix = torch.stack([self._round_off_zero(8), overflow]).requires_grad_(True)
        result = pfaffian(matrix, sign=False, mask_roundoff_gradient=True)
        torch.testing.assert_close(
            result[1], torch.tensor(1.0, dtype=torch.float64), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )
        result.sum().backward()
        assert torch.isfinite(matrix.grad).all()
        assert torch.all(matrix.grad[0] == 0)

    def test_pfaffian_mask_roundoff_gradient_nan_element_keeps_neighbour_gradient(self):
        neighbour = _block_diagonal_skew([2.0, 3.0, 4.0, 5.0])
        nan_matrix = neighbour.clone()
        nan_matrix[0, 1] = float("nan")
        matrix = torch.stack([nan_matrix, neighbour, self._round_off_zero(8)]).requires_grad_(True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pfaffian(matrix, sign=False, mask_roundoff_gradient=True).sum().backward()
        assert torch.isfinite(matrix.grad).all()
        expected = 0.5 * 120.0 * torch.linalg.inv(neighbour).transpose(-1, -2)
        torch.testing.assert_close(matrix.grad[1], expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)
        assert torch.all(matrix.grad[2] == 0)

    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
    def test_pfaffian_mask_roundoff_gradient_non_finite_input_gets_finite_gradient(self, dtype, bad_value):
        # Regression: the log-domain recovery ran its backward on the discarded non-finite elements too,
        # so 0 * NaN reached their gradient even when their output was unused.
        regular = _block_diagonal_skew([2.0, 3.0, 4.0, 5.0]).to(dtype)
        bad = regular.clone()
        bad[0, 1] = bad_value
        matrix = torch.stack([regular, bad]).requires_grad_(True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = pfaffian(matrix, sign=False, mask_roundoff_gradient=True)
        (gradient,) = torch.autograd.grad(result[0], matrix)
        assert torch.isfinite(gradient).all()
        assert torch.all(gradient[1] == 0)

    def test_pfaffian_mask_roundoff_gradient_recovered_element_below_floor_is_floored(self):
        # An overflowing determinant whose true |Pf| (1e-40) is below the default floor: the recovery must
        # apply the same sqrt(PfaffianStrategy.EPSILON) floor, with a zero gradient.
        overflow = _block_diagonal_skew([1e200, 1e-120, 1e-120])
        matrix = torch.stack([overflow, _block_diagonal_skew([1.0, 2.0, 3.0])]).requires_grad_(True)
        result = pfaffian(matrix, sign=False, mask_roundoff_gradient=True)
        result.sum().backward()
        floor = torch.tensor(PfaffianDet.EPSILON**0.5, dtype=torch.float64)
        torch.testing.assert_close(result[0], floor, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)
        assert torch.all(matrix.grad[0] == 0)
        assert matrix.grad[1].abs().amax() > 0

    @pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.float8_e4m3fn])
    @pytest.mark.parametrize("kwargs", [{}, {"sign": False}, {"sign": False, "mask_roundoff_gradient": True}])
    def test_pfaffian_rejects_unsupported_dtype(self, dtype, kwargs):
        with pytest.raises(TypeError, match=f"pfaffian supports .*; got {dtype}"):
            pfaffian(torch.zeros(4, 4, dtype=dtype), **kwargs)

    def test_pfaffian_checks_dtype_before_check_input(self):
        # A non-skew input would fail check_input with a ValueError; the dtype check must come first.
        matrix = torch.ones(4, 4, dtype=torch.int64)
        with pytest.raises(TypeError, match="got torch.int64"):
            pfaffian(matrix, check_input=True)

    @pytest.mark.parametrize("sign", [True, False])
    def test_pfaffian_complex32_result_beyond_range_warns(self, sign):
        # Pf = 300^4 = 8.1e9 is finite in complex64 but exceeds the float16 range of complex32 and its magnitude.
        matrix = _block_diagonal_skew([300.0] * 4).to(torch.complex32)
        expected_dtype = torch.complex32 if sign else torch.float16
        with pytest.warns(RuntimeWarning, match=f"exceeds the floating range of {expected_dtype}"):
            with pytest.warns(UserWarning, match="computes torch.complex32"):
                result = pfaffian(matrix, sign=sign)
        assert result.dtype == expected_dtype
        assert not torch.isfinite(result.to(torch.complex64)).all()

    @pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.float8_e4m3fn])
    def test_slog_pfaffian_rejects_unsupported_dtype(self, dtype):
        with pytest.raises(TypeError, match=f"slog_pfaffian supports .*; got {dtype}"):
            torch_pfaffian.slog_pfaffian(torch.zeros(4, 4, dtype=dtype))

    def test_supported_dtypes_constant(self):
        assert set(torch_pfaffian.SUPPORTED_DTYPES) == {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.complex32,
            torch.complex64,
            torch.complex128,
        }

    @pytest.mark.parametrize("dimension", [2, 4, 8])
    @pytest.mark.parametrize("kwargs", _LOW_PRECISION_CASES)
    def test_pfaffian_complex32_is_computed_in_complex64(self, dimension, kwargs):
        generator = torch.Generator().manual_seed(TEST_SEED)
        full = torch.randn(3, dimension, dimension, dtype=torch.complex64, generator=generator)
        matrix = (full - full.mT).to(torch.complex32).requires_grad_(True)
        with pytest.warns(UserWarning, match="computes torch.complex32 inputs in torch.complex64"):
            result = pfaffian(matrix, **kwargs)
        reference_input = matrix.detach().to(torch.complex64).requires_grad_(True)
        reference = pfaffian(reference_input, **kwargs)
        expected_dtype = torch.complex32 if kwargs["sign"] else torch.float16
        assert result.dtype == expected_dtype
        torch.testing.assert_close(
            result.to(reference.dtype), reference.to(expected_dtype).to(reference.dtype), atol=0.0, rtol=0.0
        )
        (result.real if result.dtype.is_complex else result).sum().backward()
        (reference.real if reference.dtype.is_complex else reference).sum().backward()
        assert matrix.grad is not None and matrix.grad.dtype == torch.complex32
        assert reference_input.grad is not None
        torch.testing.assert_close(
            matrix.grad.to(torch.complex64),
            reference_input.grad.to(torch.complex32).to(torch.complex64),
            atol=0.0,
            rtol=0.0,
        )

    def test_slog_pfaffian_complex32_is_computed_in_complex64(self):
        generator = torch.Generator().manual_seed(TEST_SEED)
        full = torch.randn(2, 6, 6, dtype=torch.complex64, generator=generator)
        matrix = (full - full.mT).to(torch.complex32)
        with pytest.warns(UserWarning, match="slog_pfaffian computes torch.complex32"):
            phase, log_abs = torch_pfaffian.slog_pfaffian(matrix)
        reference_phase, reference_log_abs = torch_pfaffian.slog_pfaffian(matrix.to(torch.complex64))
        assert phase.dtype == torch.complex32 and log_abs.dtype == torch.float16
        torch.testing.assert_close(
            phase.to(torch.complex64), reference_phase.to(torch.complex32).to(torch.complex64), atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(log_abs, reference_log_abs.to(torch.float16), atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("batch_shape", [(0,), (2, 0)])
    @pytest.mark.parametrize("dimension", [0, 1, 2, 3, 5, 7, 8])
    @pytest.mark.parametrize("kwargs", _LOW_PRECISION_CASES)
    def test_pfaffian_empty_batch_returns_empty_result(self, batch_shape, dimension, kwargs):
        matrix = torch.zeros(*batch_shape, dimension, dimension, dtype=torch.float64, requires_grad=True)
        result = pfaffian(matrix, **kwargs)
        assert result.shape == batch_shape
        assert result.dtype == torch.float64
        result.sum().backward()
        assert matrix.grad is not None and matrix.grad.shape == matrix.shape

    @pytest.mark.parametrize("dimension", [0, 1, 4, 7])
    def test_slog_pfaffian_empty_batch_returns_empty_result(self, dimension):
        phase, log_abs = torch_pfaffian.slog_pfaffian(torch.zeros(0, dimension, dimension))
        assert phase.shape == (0,) and log_abs.shape == (0,)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("kwargs", _LOW_PRECISION_CASES)
    def test_pfaffian_low_precision_large_input_is_computed_in_float32(self, dtype, kwargs):
        matrix = self._large_skew_matrix().to(dtype).requires_grad_(True)
        with pytest.warns(UserWarning, match=f"computes {dtype} inputs in torch.float32"):
            result = pfaffian(matrix, **kwargs)
        result.sum().backward()
        reference_input = matrix.detach().to(torch.float32).requires_grad_(True)
        reference = pfaffian(reference_input, **kwargs)
        reference.sum().backward()
        assert result.dtype == dtype
        assert matrix.grad is not None and matrix.grad.dtype == dtype
        torch.testing.assert_close(result, reference.to(dtype), atol=0.0, rtol=0.0)
        torch.testing.assert_close(matrix.grad, reference_input.grad.to(dtype), atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("dimension", [2, 6])
    def test_pfaffian_low_precision_small_input_is_not_upcast(self, dtype, dimension):
        # The unrolled small kernel supports every dtype, so it runs natively and does not warn.
        matrix = self._large_skew_matrix()[:dimension, :dimension].to(dtype).requires_grad_(True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = pfaffian(matrix)
        result.sum().backward()
        torch.testing.assert_close(result, PfaffianSmall.apply(matrix.detach()), atol=0.0, rtol=0.0)
        assert matrix.grad is not None and matrix.grad.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_pfaffian_low_precision_small_input_with_mask_is_upcast(self, dtype):
        matrix = self._skew_matrix().to(dtype)
        with pytest.warns(UserWarning, match="computes"):
            result = pfaffian(matrix, sign=False, mask_roundoff_gradient=True)
        expected = pfaffian(matrix.to(torch.float32), sign=False, mask_roundoff_gradient=True).to(dtype)
        torch.testing.assert_close(result, expected, atol=0.0, rtol=0.0)

    def test_pfaffian_low_precision_result_beyond_dtype_range_warns(self):
        # Pf = 300^4 = 8.1e9 is finite in float32 but exceeds the float16 maximum (65504).
        matrix = _block_diagonal_skew([300.0] * 4).to(torch.float16)
        with pytest.warns(RuntimeWarning, match="exceeds the floating range of torch.float16"):
            with pytest.warns(UserWarning, match="computes"):
                result = pfaffian(matrix)
        assert torch.isinf(result)

    def test_pfaffian_low_precision_check_finite_false_skips_range_warning(self):
        matrix = _block_diagonal_skew([300.0] * 4).to(torch.float16)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = pfaffian(matrix, check_finite=False)
        assert not any(issubclass(warning.category, RuntimeWarning) for warning in caught)
        assert torch.isinf(result)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_pfaffian_low_precision_forwards_epsilon(self, dtype):
        # A zero matrix sits on the epsilon floor; without forwarding, the default floor would apply.
        matrix = torch.zeros(8, 8, dtype=dtype)
        with pytest.warns(UserWarning, match="computes"):
            result = pfaffian(matrix, sign=False, epsilon=1e-6)
        # The expected floor is sqrt(1e-6) = 1e-3; the default floor sqrt(1e-30) would round to 0 in float16.
        expected = torch.tensor(1e-3, dtype=dtype)
        torch.testing.assert_close(result, expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_APPROX_COMPARISON)

    def test_pfaffian_low_precision_forwards_mask_roundoff_gradient(self):
        # B C B^T with small integer B (8x6) and C (6x6) is exact in float16 and has rank 6, so Pf = 0 exactly;
        # its float32 determinant is round-off, which gives an O(1) unmasked gradient and a zero masked one.
        generator = torch.Generator().manual_seed(0)
        factor = torch.randint(-2, 3, (8, 6), generator=generator).to(torch.float32)
        core = torch.randint(-2, 3, (6, 6), generator=generator).to(torch.float32)
        matrix = (factor @ (core - core.mT) @ factor.mT).to(torch.float16)
        gradients = []
        for mask in (True, False):
            leaf = matrix.clone().requires_grad_(True)
            with pytest.warns(UserWarning, match="computes"):
                pfaffian(leaf, sign=False, mask_roundoff_gradient=mask).backward()
            assert leaf.grad is not None
            gradients.append(leaf.grad)
        assert torch.all(gradients[0] == 0)
        assert gradients[1].abs().amax() > 1

    def test_pfaffian_low_precision_nan_input_warns_not_finite_only(self):
        matrix = torch.zeros(8, 8, dtype=torch.float16)
        matrix[0, 1], matrix[1, 0] = float("nan"), float("nan")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = pfaffian(matrix)
        messages = [str(warning.message) for warning in caught if issubclass(warning.category, RuntimeWarning)]
        assert torch.isnan(result)
        assert len(messages) == 1 and "not finite after log-domain recovery" in messages[0]

    def test_pfaffian_low_precision_forwards_check_finite_false(self):
        matrix = torch.zeros(8, 8, dtype=torch.float16)
        matrix[0, 1], matrix[1, 0] = float("nan"), float("nan")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pfaffian(matrix, check_finite=False)
        assert not any(issubclass(warning.category, RuntimeWarning) for warning in caught)

    @pytest.mark.parametrize("sign", [True, False])
    def test_pfaffian_low_precision_small_overflow_is_recovered_in_float32(self, sign):
        # The unrolled kernel forms 0.5 * (A - A^T), which overflows float16 for entries above 32752.
        dtype = torch.float16
        matrix = torch.tensor([[0.0, 4e4], [-4e4, 0.0]]).to(dtype)
        with pytest.warns(UserWarning, match="pfaffian computes"):
            result = pfaffian(matrix, sign=sign)
        torch.testing.assert_close(result, torch.tensor(4e4).to(dtype), atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("sign", [True, False])
    def test_pfaffian_low_precision_small_nan_input_returns_nan(self, sign):
        matrix = torch.zeros(4, 4, dtype=torch.float16)
        matrix[0, 1], matrix[1, 0], matrix[2, 3], matrix[3, 2] = float("nan"), float("nan"), 1.0, -1.0
        with pytest.warns(RuntimeWarning, match="not finite"):
            with pytest.warns(UserWarning, match="pfaffian computes"):
                result = pfaffian(matrix, sign=sign)
        assert torch.isnan(result)

    def test_pfaffian_low_precision_warnings_point_at_the_caller(self):
        nan_matrix = torch.zeros(8, 8, dtype=torch.float16)
        nan_matrix[0, 1], nan_matrix[1, 0] = float("nan"), float("nan")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pfaffian(nan_matrix)
            pfaffian(_block_diagonal_skew([300.0] * 4).to(torch.float16))
            torch_pfaffian.slog_pfaffian(torch.zeros(4, 4, dtype=torch.float16))
        categories = {warning.category for warning in caught}
        assert categories == {UserWarning, RuntimeWarning}
        assert len(caught) == 5
        assert all(warning.filename == __file__ for warning in caught)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("dimension", [4, 8])
    def test_slog_pfaffian_low_precision_is_computed_in_float32(self, dtype, dimension):
        matrix = self._large_skew_matrix()[:dimension, :dimension].to(dtype).requires_grad_(True)
        with pytest.warns(UserWarning, match=f"slog_pfaffian computes {dtype} inputs in torch.float32"):
            phase, log_abs = torch_pfaffian.slog_pfaffian(matrix)
        log_abs.backward()
        reference_input = matrix.detach().to(torch.float32).requires_grad_(True)
        reference_phase, reference_log_abs = torch_pfaffian.slog_pfaffian(reference_input)
        reference_log_abs.backward()
        assert phase.dtype == dtype and log_abs.dtype == dtype
        torch.testing.assert_close(phase, reference_phase.to(dtype), atol=0.0, rtol=0.0)
        torch.testing.assert_close(log_abs, reference_log_abs.to(dtype), atol=0.0, rtol=0.0)
        assert matrix.grad is not None
        torch.testing.assert_close(matrix.grad, reference_input.grad.to(dtype), atol=0.0, rtol=0.0)

    def test_pfaffian_epsilon_none_matches_default_magnitude(self):
        rng = np.random.default_rng(TEST_SEED)
        matrix = torch.tensor(_rand_skew_complex(8, rng).real, dtype=torch.float64)
        matrix = 0.5 * (matrix - matrix.transpose(-1, -2))
        torch.testing.assert_close(
            pfaffian(matrix, sign=False, epsilon=None),
            PfaffianDet.apply(matrix),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_pfaffian_epsilon_floors_magnitude_on_small_path(self):
        # m = 4 <= AUTO_SMALL_MAX, sign=False, epsilon given: the small path clamps |Pf| at sqrt(epsilon).
        block = 1e-3 * torch.eye(2, dtype=torch.float64)
        matrix = _block_antidiagonal(block)  # 4x4 skew, |Pf| = (1e-3)^2 = 1e-6
        floored = pfaffian(matrix, sign=False, epsilon=1e-8)
        torch.testing.assert_close(
            floored, torch.tensor(1e-4, dtype=torch.float64), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )
