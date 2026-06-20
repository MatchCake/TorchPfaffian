import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

from tests.configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_APPROX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
)
from torch_pfaffian.strategies.pfaffian_fdbpf import PfaffianFDBPf

_RNG = np.random.default_rng(TEST_SEED)


def _skew(matrix: np.ndarray) -> np.ndarray:
    return matrix - np.einsum("...ij->...ji", matrix)


def _random_skew_matrices(shapes: list[tuple[int, ...]], count: int) -> list[np.ndarray]:
    return [_skew(_RNG.random(shape)) for shape in shapes for _ in range(count)]


def _well_conditioned_skew_matrices(shapes: list[tuple[int, ...]], count: int, min_abs_det: float = 0.1):
    # The gradient relies on the (pseudo)inverse, which is unstable for near-singular matrices, so the
    # gradcheck inputs are restricted to skew matrices whose determinant is comfortably away from zero.
    matrices = []
    for shape in shapes:
        kept = 0
        while kept < count:
            candidate = _skew(_RNG.random(shape))
            if np.all(np.abs(np.linalg.det(candidate)) > min_abs_det):
                matrices.append(candidate)
                kept += 1
    return matrices


_FORWARD_MATRICES = _random_skew_matrices([(8, 8), (16, 6, 6), (18, 10, 10)], N_RANDOM_TESTS_PER_CASE)
_GRADCHECK_MATRICES = _well_conditioned_skew_matrices([(4, 4), (6, 6), (8, 4, 4)], N_RANDOM_TESTS_PER_CASE)


class TestPfaffianFDBPf:
    @pytest.mark.parametrize("skew_matrix", _FORWARD_MATRICES)
    def test_forward_square_of_pfaffian_matches_determinant(self, skew_matrix):
        skew_tensor = torch.tensor(skew_matrix)
        pfaffian = PfaffianFDBPf.apply(skew_tensor)
        determinant = torch.linalg.det(skew_tensor)
        torch.testing.assert_close(pfaffian**2, determinant, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("skew_matrix", _GRADCHECK_MATRICES)
    def test_backward_passes_gradcheck(self, skew_matrix):
        skew_tensor = torch.tensor(skew_matrix, requires_grad=True)
        assert gradcheck(
            PfaffianFDBPf.apply, (skew_tensor,), eps=1e-6, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON
        )

    def test_forward_matches_known_pfaffian_magnitude(self):
        # For the 2x2 skew matrix [[0, a], [-a, 0]] the Pfaffian is a, so its magnitude is |a|.
        value = 0.7
        matrix = torch.tensor([[0.0, value], [-value, 0.0]], dtype=torch.float64)
        pfaffian = PfaffianFDBPf.apply(matrix)
        expected = torch.tensor(abs(value), dtype=torch.float64)
        torch.testing.assert_close(pfaffian, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_forward_returns_absolute_value_so_sign_is_not_recovered(self):
        # The strategy computes |pf| via sqrt(|det|); the true Pfaffian of [[0, -a], [a, 0]] is -a,
        # but the strategy returns +|a|.
        value = 0.7
        matrix = torch.tensor([[0.0, -value], [value, 0.0]], dtype=torch.float64)
        pfaffian = PfaffianFDBPf.apply(matrix)
        expected = torch.tensor(abs(value), dtype=torch.float64)
        torch.testing.assert_close(pfaffian, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_forward_preserves_dtype(self, dtype):
        matrix = torch.tensor(_skew(_RNG.random((6, 6))), dtype=dtype)
        assert PfaffianFDBPf.apply(matrix).dtype == dtype

    def test_forward_supports_batched_input(self):
        batched_matrix = torch.tensor(_skew(_RNG.random((5, 6, 6))))
        pfaffian = PfaffianFDBPf.apply(batched_matrix)
        assert pfaffian.shape == batched_matrix.shape[:-2]
        torch.testing.assert_close(
            pfaffian**2, torch.linalg.det(batched_matrix), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_forward_returns_zero_for_odd_dimension(self):
        odd_matrix = torch.tensor(_skew(_RNG.random((3, 3))))
        pfaffian = PfaffianFDBPf.apply(odd_matrix)
        torch.testing.assert_close(pfaffian, torch.zeros_like(odd_matrix[..., 0, 0]))

    def test_backward_returns_none_when_input_does_not_require_grad(self):
        matrix = torch.tensor(_skew(_RNG.random((4, 4))))
        pfaffian = PfaffianFDBPf.apply(matrix)

        class _Context:
            saved_tensors = (matrix, pfaffian)
            needs_input_grad = (False,)

        assert PfaffianFDBPf.backward(_Context(), torch.ones_like(pfaffian)) is None

    def test_backward_does_not_use_svd(self, monkeypatch):
        # Regression: the magnitude gradient must not route through torch.linalg.pinv (SVD), whose
        # iterative solver can fail to converge on ill-conditioned inputs. Force pinv to fail.
        def _fail(*args, **kwargs):
            raise AssertionError("torch.linalg.pinv (SVD) must not be used by PfaffianFDBPf.backward")

        monkeypatch.setattr(torch.linalg, "pinv", _fail)
        matrix = torch.tensor(_skew(_RNG.random((6, 6))), requires_grad=True)
        PfaffianFDBPf.apply(matrix).backward()
        assert torch.isfinite(matrix.grad).all()

    def test_backward_odd_dimension_gradient_is_finite_zero(self):
        # Odd-dimensional skew matrices are always singular (inv would raise / return garbage); the
        # magnitude is 0 there, so the gradient must be exactly zero and finite, never NaN.
        matrix = torch.tensor(_skew(_RNG.random((5, 5))), requires_grad=True)
        PfaffianFDBPf.apply(matrix).backward()
        assert torch.isfinite(matrix.grad).all()
        torch.testing.assert_close(matrix.grad, torch.zeros_like(matrix))

    def test_backward_mixed_singular_invertible_batch_matches_inverse(self):
        # A batch mixing an exactly-singular element (pf at the floor) with an invertible one must not
        # raise, and the invertible element's gradient must match the inverse-based closed form.
        singular = torch.zeros(4, 4, dtype=torch.float64)
        singular[2, 3] = 1.0
        singular[3, 2] = -1.0
        invertible = torch.tensor(_skew(_RNG.random((4, 4))))
        matrix = torch.stack([singular, invertible]).requires_grad_(True)
        magnitude = PfaffianFDBPf.apply(matrix)
        magnitude.sum().backward()
        assert torch.isfinite(matrix.grad).all()
        expected_invertible = torch.einsum(
            "ij->ji", 0.5 * magnitude[1].detach() * torch.linalg.inv(invertible)
        )
        torch.testing.assert_close(
            matrix.grad[1], expected_invertible, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_backward_ill_conditioned_is_finite_and_matches_inverse(self):
        # Ill-conditioned invertible input (the regime where the SVD pseudo-inverse fails to converge):
        # the LU inverse stays finite and the gradient matches the inverse closed form.
        scales = torch.tensor([1e8, 1e-8, 1e6, 1e-6], dtype=torch.float64)
        blocks = [torch.tensor([[0.0, scale], [-scale, 0.0]], dtype=torch.float64) for scale in scales]
        matrix = torch.block_diag(*blocks).requires_grad_(True)
        magnitude = PfaffianFDBPf.apply(matrix)
        magnitude.backward()
        expected = torch.einsum("ij->ji", 0.5 * magnitude.detach() * torch.linalg.inv(matrix.detach()))
        assert torch.isfinite(matrix.grad).all()
        torch.testing.assert_close(matrix.grad, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)
