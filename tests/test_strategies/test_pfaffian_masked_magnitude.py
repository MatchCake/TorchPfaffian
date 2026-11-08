import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

from tests.configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    ATOL_SHAPE_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_APPROX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
)
from torch_pfaffian.strategies.pfaffian_masked_magnitude import PfaffianMaskedMagnitude
from torch_pfaffian.strategies.strategy import PfaffianStrategy

_RNG = np.random.default_rng(TEST_SEED)
_DTYPES = [torch.float32, torch.float64, torch.complex64, torch.complex128]


def _random_skew(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    entries = _RNG.normal(size=shape)
    if dtype.is_complex:
        entries = entries + 1j * _RNG.normal(size=shape)
    return torch.tensor(entries - np.swapaxes(entries, -1, -2), dtype=dtype)


def _well_conditioned_skew(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    # A random skew matrix plus a dominant block-diagonal part keeps the condition number O(1).
    dimension = shape[-1]
    dominant = torch.zeros(dimension, dimension, dtype=torch.float64)
    even = torch.arange(0, dimension, 2)
    dominant[even, even + 1] = 4.0
    dominant[even + 1, even] = -4.0
    return _random_skew(shape, dtype) + dominant.to(dtype)


def _rotated_block_diagonal(coefficients: list[float], dtype: torch.dtype) -> torch.Tensor:
    # Q diag([[0, c_k], [-c_k, 0]]) Q^T with a random orthogonal Q: |Pf| = prod |c_k| and the entries are
    # mixed, so a coefficient at round-off level produces a round-off (not exactly zero) determinant.
    dimension = 2 * len(coefficients)
    block = torch.zeros(dimension, dimension, dtype=torch.float64)
    for index, coefficient in enumerate(coefficients):
        block[2 * index, 2 * index + 1] = coefficient
        block[2 * index + 1, 2 * index] = -coefficient
    orthogonal, _ = torch.linalg.qr(torch.tensor(_RNG.normal(size=(dimension, dimension))))
    return (orthogonal @ block @ orthogonal.T).to(dtype)


def _unmasked_reference(matrix: torch.Tensor, epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    leaf = matrix.detach().clone().requires_grad_(True)
    value = torch.sqrt(torch.clamp(torch.abs(torch.linalg.det(leaf)), min=epsilon))
    value.sum().backward()
    return value.detach(), leaf.grad


_GRADCHECK_SHAPES = [(4, 4), (8, 8), (3, 6, 6)]


class TestPfaffianMaskedMagnitude:
    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize("shape", [(6, 6), (8, 8), (5, 10, 10)])
    def test_well_conditioned_value_and_gradient_match_unmasked(self, shape, dtype):
        matrix = _well_conditioned_skew(shape, dtype).requires_grad_(True)
        epsilon = PfaffianStrategy.EPSILON
        magnitude = PfaffianMaskedMagnitude.apply(matrix, epsilon)
        magnitude.sum().backward()
        expected_value, expected_grad = _unmasked_reference(matrix, epsilon)
        tolerance = ATOL_MATRIX_COMPARISON if dtype in (torch.float64, torch.complex128) else ATOL_APPROX_COMPARISON
        assert magnitude.dtype == expected_value.dtype
        assert matrix.grad.dtype == dtype
        torch.testing.assert_close(magnitude, expected_value, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(matrix.grad, expected_grad, atol=tolerance, rtol=tolerance)

    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    @pytest.mark.parametrize("shape", _GRADCHECK_SHAPES)
    def test_passes_gradcheck(self, shape, dtype):
        for _ in range(N_RANDOM_TESTS_PER_CASE // 5):
            matrix = _well_conditioned_skew(shape, dtype).requires_grad_(True)
            assert gradcheck(
                lambda value: PfaffianMaskedMagnitude.apply(value, PfaffianStrategy.EPSILON),
                (matrix,),
                eps=1e-6,
                atol=ATOL_APPROX_COMPARISON,
                rtol=RTOL_APPROX_COMPARISON,
            )

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_round_off_zero_gets_zero_gradient_and_keeps_value(self, dtype):
        # One block at 0.25 * eps makes |Pf| vanish up to round-off: the unmasked backward returns an
        # O(1) gradient there, the masked one returns zero. A well-conditioned neighbour is untouched.
        epsilon = torch.finfo(dtype).eps
        round_off = _rotated_block_diagonal([0.25 * epsilon, 1.0, 1.0], dtype)
        neighbour = _well_conditioned_skew((6, 6), dtype)
        matrix = torch.stack([round_off, neighbour]).requires_grad_(True)
        floor = 1e-60 if dtype in (torch.float64, torch.complex128) else 1e-40
        magnitude = PfaffianMaskedMagnitude.apply(matrix, floor)
        magnitude.sum().backward()
        expected_value, expected_grad = _unmasked_reference(matrix, floor)
        torch.testing.assert_close(magnitude, expected_value)  # the forward is unchanged
        # Without the mask (and without a floor) the gradient is garbage: O(1) noise, or NaN where the
        # platform's LAPACK computes the round-off determinant as exactly 0.
        leaf = round_off.detach().clone().requires_grad_(True)
        torch.sqrt(torch.abs(torch.linalg.det(leaf))).backward()
        assert (~(leaf.grad.abs() <= ATOL_APPROX_COMPARISON)).any()
        assert torch.all(matrix.grad[0] == 0)
        torch.testing.assert_close(matrix.grad[1], expected_grad[1])

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_clamped_element_gets_zero_gradient(self, dtype):
        epsilon = 1e-3
        small = _rotated_block_diagonal([1e-2, 1e-2, 1.0], dtype)  # |det| = 1e-8 < epsilon
        matrix = torch.stack([small, _well_conditioned_skew((6, 6), dtype)]).requires_grad_(True)
        magnitude = PfaffianMaskedMagnitude.apply(matrix, epsilon)
        magnitude.sum().backward()
        expected_floor = torch.sqrt(torch.tensor(epsilon, dtype=magnitude.dtype))
        torch.testing.assert_close(magnitude[0], expected_floor)
        assert torch.all(matrix.grad[0] == 0)
        assert matrix.grad[1].abs().amax() > 0

    def test_small_resolved_pfaffian_of_ill_conditioned_matrix_keeps_gradient(self):
        # cond ~ 1e12 is far above round-off in float64: the gradient must not be masked.
        matrix = _rotated_block_diagonal([1e-12, 1.0, 1.0], torch.float64).requires_grad_(True)
        magnitude = PfaffianMaskedMagnitude.apply(matrix, 1e-60)
        magnitude.backward()
        expected = 0.5 * magnitude.detach() * torch.linalg.inv(matrix.detach()).transpose(-1, -2)
        torch.testing.assert_close(matrix.grad, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)
        assert matrix.grad.abs().amax() > ATOL_APPROX_COMPARISON

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_nan_input_gives_finite_gradient(self, dtype):
        matrix = _well_conditioned_skew((2, 6, 6), dtype)
        matrix[0, 0, 1] = float("nan")
        matrix.requires_grad_(True)
        PfaffianMaskedMagnitude.apply(matrix, PfaffianStrategy.EPSILON).sum().backward()
        assert torch.isfinite(matrix.grad).all()
        assert torch.all(matrix.grad[0] == 0)

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_gradient_is_conjugate_inverse_transpose(self, dtype):
        matrix = _well_conditioned_skew((6, 6), dtype).requires_grad_(True)
        magnitude = PfaffianMaskedMagnitude.apply(matrix, PfaffianStrategy.EPSILON)
        magnitude.backward()
        expected = 0.5 * magnitude.detach() * torch.linalg.inv(matrix.detach()).transpose(-1, -2).conj()
        torch.testing.assert_close(matrix.grad, expected)

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_odd_dimension_is_floor_with_zero_gradient(self, dtype):
        matrix = _random_skew((2, 5, 5), dtype).requires_grad_(True)
        epsilon = 1e-20
        magnitude = PfaffianMaskedMagnitude.apply(matrix, epsilon)
        magnitude.sum().backward()
        torch.testing.assert_close(magnitude, torch.full_like(magnitude, epsilon**0.5))
        assert not magnitude.is_complex()
        torch.testing.assert_close(matrix.grad, torch.zeros_like(matrix))

    def test_empty_matrix_has_unit_magnitude_and_empty_gradient(self):
        matrix = torch.zeros(3, 0, 0, dtype=torch.float64, requires_grad=True)
        magnitude = PfaffianMaskedMagnitude.apply(matrix, PfaffianStrategy.EPSILON)
        magnitude.sum().backward()
        torch.testing.assert_close(magnitude, torch.ones(3, dtype=torch.float64))
        assert matrix.grad.shape == matrix.shape

    def test_default_epsilon_is_strategy_epsilon(self):
        matrix = torch.zeros(4, 4, dtype=torch.float64)
        expected = torch.tensor(PfaffianStrategy.EPSILON**0.5, dtype=torch.float64)
        torch.testing.assert_close(PfaffianMaskedMagnitude.apply(matrix), expected)

    def test_apply_returns_single_tensor(self):
        result = PfaffianMaskedMagnitude.apply(_well_conditioned_skew((4, 4), torch.float64))
        assert isinstance(result, torch.Tensor)
        assert result.shape == ()

    def test_backward_returns_none_when_input_does_not_require_grad(self):
        matrix = _well_conditioned_skew((4, 4), torch.float64)
        magnitude = PfaffianMaskedMagnitude.apply(matrix)

        class _Context:
            saved_tensors = (matrix, magnitude, torch.zeros((), dtype=torch.bool))
            needs_input_grad = (False, False)

        assert PfaffianMaskedMagnitude.backward(_Context(), torch.ones_like(magnitude), None) == (None, None)

    def test_rcond_scale_constant(self):
        assert PfaffianMaskedMagnitude.ROUNDOFF_RCOND_SCALE == 4.0
        assert PfaffianMaskedMagnitude.ROUNDOFF_SPECTRAL_SCALE == 16.0

    def test_resolved_pfaffian_with_round_off_rcond_keeps_gradient(self):
        # rcond_1 is at round-off level here but sigma_min / sigma_max is not (the 1-norm ratio is 19x
        # smaller): |Pf| is resolved, so this pins the second (spectral) criterion of the AND.
        matrix = self._resolved_with_round_off_rcond()
        threshold = PfaffianMaskedMagnitude.ROUNDOFF_RCOND_SCALE * torch.finfo(torch.float64).eps
        spectral_threshold = PfaffianMaskedMagnitude.ROUNDOFF_SPECTRAL_SCALE * torch.finfo(torch.float64).eps
        inverse = torch.linalg.inv(matrix)
        rcond = 1.0 / (torch.linalg.matrix_norm(matrix, ord=1) * torch.linalg.matrix_norm(inverse, ord=1))
        singular_values = torch.linalg.svdvals(matrix)
        spectral_rcond = singular_values[-1] / singular_values[0]
        assert 2 * rcond <= threshold and spectral_threshold <= spectral_rcond / 2  # the two tests disagree
        leaf = matrix.clone().requires_grad_(True)
        magnitude = PfaffianMaskedMagnitude.apply(leaf, 1e-60)
        magnitude.backward()
        expected = 0.5 * magnitude.detach() * inverse.transpose(-1, -2)  # the unmasked formula
        assert leaf.grad.abs().amax() > 0
        torch.testing.assert_close(leaf.grad, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_nan_element_does_not_disable_spectral_test_of_neighbours(self):
        # Regression guard: an SVD over the raw batch raises on the NaN element, and the fallback on the
        # 1-norm test alone would then mask the resolved neighbour.
        nan_matrix = _well_conditioned_skew((64, 64), torch.float64)
        nan_matrix[0, 1] = float("nan")
        matrix = torch.stack([nan_matrix, self._resolved_with_round_off_rcond()]).requires_grad_(True)
        PfaffianMaskedMagnitude.apply(matrix, 1e-60).sum().backward()
        assert torch.all(matrix.grad[0] == 0)
        assert matrix.grad[1].abs().amax() > 0

    @staticmethod
    def _resolved_with_round_off_rcond() -> torch.Tensor:
        # Corank-2 skew matrix resolved by a 6.2e-12 entry: rcond_1 ~ 1.8 eps, sigma_min / sigma_max ~ 35 eps.
        dimension = 64
        generator = torch.Generator().manual_seed(19)
        entries = torch.randn(dimension, dimension, dtype=torch.float64, generator=generator)
        matrix = entries - entries.T
        matrix[0], matrix[:, 0] = matrix[1].clone(), matrix[:, 1].clone()  # corank 2
        matrix[0, 0], matrix[0, 1], matrix[1, 0] = 0.0, 0.0, 0.0
        matrix[0, 2] += 6.2e-12  # resolve the Pfaffian slightly away from zero
        matrix[2, 0] -= 6.2e-12
        assert torch.equal(matrix, -matrix.T)
        return matrix

    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize("dimension", [12, 16, 24, 32])
    def test_large_round_off_zero_gets_zero_gradient(self, dtype, dimension):
        # Regression: the magnitude test compared |Pf| with max|A_ij|^(m/2), which falls behind the
        # round-off of |Pf| as m grows (by 4e2 at m = 24), so large structural zeros kept their noise.
        epsilon = torch.finfo(dtype).eps
        coefficients = [0.25 * epsilon] + [10.0] * (dimension // 2 - 1)
        matrix = torch.stack([_rotated_block_diagonal(coefficients, dtype) for _ in range(N_RANDOM_TESTS_PER_CASE)])
        matrix.requires_grad_(True)
        PfaffianMaskedMagnitude.apply(matrix, 1e-300 if epsilon < 1e-10 else 1e-40).sum().backward()
        assert torch.all(matrix.grad == 0)

    @pytest.mark.parametrize("dimension", [16, 24])
    def test_round_off_zero_with_accumulated_drift_gets_zero_gradient(self, dimension):
        # A structural zero computed through a long circuit drifts to sigma_min / sigma_max ~ 8 eps (MatchCake
        # lookups after 1e5 in-place Givens rotations), with rcond_1 still at round-off level: an 8 eps block
        # models it. A spectral threshold of 4 eps would keep its gradient.
        epsilon = torch.finfo(torch.float64).eps
        coefficients = [8.0 * epsilon] + [1.0] * (dimension // 2 - 1)
        matrix = torch.stack(
            [_rotated_block_diagonal(coefficients, torch.float64) for _ in range(N_RANDOM_TESTS_PER_CASE)]
        ).requires_grad_(True)
        PfaffianMaskedMagnitude.apply(matrix, 1e-300).sum().backward()
        assert torch.all(matrix.grad == 0)

    def test_svd_failure_falls_back_on_one_norm_test(self, monkeypatch):
        epsilon = torch.finfo(torch.float64).eps
        round_off = _rotated_block_diagonal([0.25 * epsilon, 1.0, 1.0], torch.float64)
        resolved = _well_conditioned_skew((6, 6), torch.float64)
        matrix = torch.stack([round_off, resolved]).requires_grad_(True)

        def failing_svdvals(*args, **kwargs):
            raise torch.linalg.LinAlgError("did not converge")

        monkeypatch.setattr(torch.linalg, "svdvals", failing_svdvals)
        PfaffianMaskedMagnitude.apply(matrix, 1e-60).sum().backward()
        assert torch.all(matrix.grad[0] == 0)
        assert matrix.grad[1].abs().amax() > 0

    def test_magnitude_test_alone_does_not_mask_resolved_ill_conditioned_matrix(self):
        # rcond ~ 1e-12 is far above round-off in both norms: the 1-norm half of the AND keeps this
        # element's gradient, although |Pf| = 1e-18 is tiny.
        matrix = _rotated_block_diagonal([1e-12, 1.0, 1e-6], torch.float64).requires_grad_(True)
        magnitude = PfaffianMaskedMagnitude.apply(matrix, 1e-60)
        magnitude.backward()
        expected = 0.5 * magnitude.detach() * torch.linalg.inv(matrix.detach()).transpose(-1, -2)
        assert matrix.grad.abs().amax() > 0
        torch.testing.assert_close(matrix.grad, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)

    @pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
    def test_mask_decision_is_invariant_to_matrix_scale(self, scale):
        # Both tests are ratios of norms, unchanged under A -> s A.
        epsilon = torch.finfo(torch.float64).eps
        round_off = (scale * _rotated_block_diagonal([0.25 * epsilon, 1.0, 1.0], torch.float64)).requires_grad_(True)
        resolved = (scale * _rotated_block_diagonal([1e-12, 1.0, 1e-6], torch.float64)).requires_grad_(True)
        PfaffianMaskedMagnitude.apply(round_off, 1e-300).backward()
        PfaffianMaskedMagnitude.apply(resolved, 1e-300).backward()
        assert torch.all(round_off.grad == 0)
        assert resolved.grad.abs().amax() > 0

    @pytest.mark.parametrize("epsilon", [1e-20, 1e-30])
    def test_odd_dimension_floor_is_exactly_sqrt_epsilon(self, epsilon):
        magnitude = PfaffianMaskedMagnitude.apply(torch.zeros(5, 5, dtype=torch.float64), epsilon)
        expected = torch.tensor(epsilon**0.5, dtype=torch.float64)
        torch.testing.assert_close(magnitude, expected, atol=ATOL_SHAPE_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    def test_double_backward_is_finite_at_exactly_singular_elements(self, dtype):
        # Regression: the masked gradient was a where() over coefficient * inverse, whose derivative is NaN
        # where the LU inverse fails; the identity is inverted there instead.
        zero_row = _well_conditioned_skew((6, 6), dtype)
        zero_row[0], zero_row[:, 0] = 0.0, 0.0
        matrix = torch.stack([torch.zeros(6, 6, dtype=dtype), zero_row, _well_conditioned_skew((6, 6), dtype)])
        matrix.requires_grad_(True)
        magnitude = PfaffianMaskedMagnitude.apply(matrix, 1e-60)
        (gradient,) = torch.autograd.grad(magnitude.sum(), matrix, create_graph=True)
        (second,) = torch.autograd.grad(gradient.abs().pow(2).sum(), matrix)
        assert torch.isfinite(second).all()

    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    def test_double_backward_matches_autograd_of_unmasked_magnitude(self, dtype):
        matrix = _well_conditioned_skew((6, 6), dtype).requires_grad_(True)
        reference = matrix.detach().clone().requires_grad_(True)
        for leaf, function in (
            (matrix, lambda value: PfaffianMaskedMagnitude.apply(value, 1e-60)),
            (reference, lambda value: torch.sqrt(torch.abs(torch.linalg.det(value)))),
        ):
            (gradient,) = torch.autograd.grad(function(leaf), leaf, create_graph=True)
            (leaf.grad,) = torch.autograd.grad(gradient.abs().pow(2).sum(), leaf)
        torch.testing.assert_close(
            matrix.grad, reference.grad, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_APPROX_COMPARISON
        )
