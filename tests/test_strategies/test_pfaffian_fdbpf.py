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


def _well_conditioned_complex_skew_matrices(shapes: list[tuple[int, ...]], count: int, min_abs_det: float = 0.1):
    matrices = []
    for shape in shapes:
        kept = 0
        while kept < count:
            candidate = _skew(_RNG.normal(size=shape) + 1j * _RNG.normal(size=shape))
            if np.all(np.abs(np.linalg.det(candidate)) > min_abs_det):
                matrices.append(candidate)
                kept += 1
    return matrices


_COMPLEX_GRADCHECK_MATRICES = _well_conditioned_complex_skew_matrices(
    [(4, 4), (8, 8), (3, 6, 6)], N_RANDOM_TESTS_PER_CASE
)


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
        expected_invertible = torch.einsum("ij->ji", 0.5 * magnitude[1].detach() * torch.linalg.inv(invertible))
        torch.testing.assert_close(
            matrix.grad[1], expected_invertible, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_exact_singular_grad_false_gives_zero_grad_at_singular(self, monkeypatch):
        # With EXACT_SINGULAR_GRAD disabled the magnitude backward is host-sync-free: a singular
        # element (pf at the floor) gets an exactly zero gradient, while an invertible one matches the
        # inverse-based closed form.
        monkeypatch.setattr(PfaffianFDBPf, "EXACT_SINGULAR_GRAD", False)
        singular = torch.zeros(4, 4, dtype=torch.float64)
        invertible = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0], [0.0, 0.0, -2.0, 0.0]],
            dtype=torch.float64,
        )
        matrix = torch.stack([singular, invertible]).requires_grad_(True)
        magnitude = PfaffianFDBPf.apply(matrix)
        magnitude.sum().backward()
        assert torch.isfinite(matrix.grad).all()
        assert torch.all(matrix.grad[0] == 0)  # sync-free: singular element gets exactly zero gradient
        expected_invertible = torch.einsum("ij->ji", 0.5 * magnitude[1].detach() * torch.linalg.inv(invertible))
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

    @pytest.mark.parametrize("skew_matrix", _COMPLEX_GRADCHECK_MATRICES)
    def test_backward_complex_passes_gradcheck(self, skew_matrix):
        # Regression for issue #17: the complex backward raised on a complex comparison, and the
        # gradient of |pf| needs conj(A^{-T}) under PyTorch's Wirtinger convention.
        skew_tensor = torch.tensor(skew_matrix, dtype=torch.complex128, requires_grad=True)
        assert gradcheck(
            PfaffianFDBPf.apply, (skew_tensor,), eps=1e-6, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON
        )

    @pytest.mark.parametrize("skew_matrix", _COMPLEX_GRADCHECK_MATRICES[:N_RANDOM_TESTS_PER_CASE])
    def test_backward_complex_sync_free_passes_gradcheck(self, skew_matrix, monkeypatch):
        monkeypatch.setattr(PfaffianFDBPf, "EXACT_SINGULAR_GRAD", False)
        skew_tensor = torch.tensor(skew_matrix, dtype=torch.complex128, requires_grad=True)
        assert gradcheck(
            PfaffianFDBPf.apply, (skew_tensor,), eps=1e-6, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON
        )

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_backward_complex_gradient_matches_conjugate_inverse_closed_form(self, dtype):
        # d|pf|/d conj(A) in PyTorch's convention is (|pf| / 2) conj(A^{-T}); the gradient keeps the
        # complex dtype of the input.
        matrix = torch.tensor(_COMPLEX_GRADCHECK_MATRICES[0], dtype=dtype, requires_grad=True)
        magnitude = PfaffianFDBPf.apply(matrix)
        magnitude.backward()
        expected = 0.5 * magnitude.detach() * torch.linalg.inv(matrix.detach()).transpose(-1, -2).conj()
        assert matrix.grad.dtype == dtype
        torch.testing.assert_close(matrix.grad, expected)

    @pytest.mark.parametrize("exact_singular_grad", [True, False])
    def test_backward_complex_mixed_singular_invertible_batch_is_finite(self, exact_singular_grad, monkeypatch):
        monkeypatch.setattr(PfaffianFDBPf, "EXACT_SINGULAR_GRAD", exact_singular_grad)
        singular = torch.zeros(4, 4, dtype=torch.complex128)
        invertible = torch.tensor(_COMPLEX_GRADCHECK_MATRICES[0], dtype=torch.complex128)
        matrix = torch.stack([singular, invertible]).requires_grad_(True)
        magnitude = PfaffianFDBPf.apply(matrix)
        magnitude.sum().backward()
        assert torch.isfinite(matrix.grad).all()
        expected_invertible = 0.5 * magnitude[1].detach() * torch.linalg.inv(invertible).transpose(-1, -2).conj()
        torch.testing.assert_close(
            matrix.grad[1], expected_invertible, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("exact_singular_grad", [True, False])
    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    @pytest.mark.parametrize("shape", [(3, 3), (7, 7), (2, 5, 5)])
    def test_backward_complex_odd_dimension_gradient_is_zero(self, shape, dtype, exact_singular_grad, monkeypatch):
        # Odd-dimensional skew matrices are singular, so |pf| = 0 and the gradient is exactly zero. The
        # forward returns a real magnitude so the backward's floor comparison is defined for complex input.
        monkeypatch.setattr(PfaffianFDBPf, "EXACT_SINGULAR_GRAD", exact_singular_grad)
        entries = _RNG.normal(size=shape) + 1j * _RNG.normal(size=shape)
        matrix = torch.tensor(_skew(entries), dtype=dtype, requires_grad=True)
        magnitude = PfaffianFDBPf.apply(matrix)
        assert not magnitude.is_complex()
        magnitude.sum().backward()
        assert matrix.grad.dtype == dtype
        torch.testing.assert_close(matrix.grad, torch.zeros_like(matrix))
