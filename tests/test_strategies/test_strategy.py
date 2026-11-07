import pytest
import torch

from tests.configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    TEST_SEED,
)
from torch_pfaffian.strategies.pfaffian_parlett_reid import PfaffianParlettReid
from torch_pfaffian.strategies.strategy import PfaffianStrategy


def _random_skew(dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    full = torch.randn(dimension, dimension, dtype=torch.float64, generator=generator)
    return full - full.transpose(-1, -2)


class _RecordingContext:
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


class TestPfaffianStrategy:
    def test_class_attributes(self):
        assert PfaffianStrategy.NAME == "PfaffianStrategy"
        assert PfaffianStrategy.EPSILON == 1e-30

    def test_setup_context_saves_inputs_and_output(self):
        matrix = torch.eye(2, dtype=torch.float64)
        pfaffian = torch.tensor(1.0, dtype=torch.float64)
        context = _RecordingContext()
        PfaffianStrategy.setup_context(context, (matrix,), pfaffian)
        assert context.saved_tensors[0] is matrix
        assert context.saved_tensors[1] is pfaffian

    def test_base_forward_is_not_implemented(self):
        assert PfaffianStrategy.forward(torch.eye(2, dtype=torch.float64)) is None

    def test_base_backward_is_not_implemented(self):
        assert PfaffianStrategy.backward(None, None) is None

    def test_grad_matrix_matches_inverse_form_on_invertible(self):
        # For invertible inputs the adjugate-based gradient equals the inverse-based closed form.
        matrix = _random_skew(6, seed=0)
        pfaffian = PfaffianParlettReid.forward(matrix)
        grad_output = torch.tensor(1.7, dtype=torch.float64)
        expected = torch.einsum("...,...ij->...ji", 0.5 * grad_output * pfaffian, torch.linalg.inv(matrix))
        result = PfaffianParlettReid.pfaffian_grad_matrix(matrix, pfaffian, grad_output)
        torch.testing.assert_close(result, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_grad_matrix_uses_adjugate_on_singular(self):
        # Singular batch element (pf=0): the gradient is the exact Pfaffian-adjugate form, not zero.
        singular = torch.zeros(4, 4, dtype=torch.float64)
        singular[2, 3] = 1.0
        singular[3, 2] = -1.0
        invertible = _random_skew(4, seed=1)
        matrix = torch.stack([singular, invertible])
        pfaffian = PfaffianParlettReid.forward(matrix)
        assert pfaffian[0] == 0.0  # the singular element really has Pfaffian 0
        grad_output = torch.ones(2, dtype=torch.float64)
        result = PfaffianParlettReid.pfaffian_grad_matrix(matrix, pfaffian, grad_output)
        # d pf / d A_{01} = a_{23} = 1, so the gradient of the singular element is nonzero.
        assert torch.isfinite(result).all()
        assert result[0].abs().sum() > 0

    def test_grad_matrix_mixed_batch_matches_closed_forms(self):
        # A batch mixing a singular (pf=0) and an invertible element must not raise (a plain
        # torch.linalg.inv would, on the singular pivot) and each element must match its exact
        # closed form: the invertible one via the inverse, the singular one via the minor adjugate.
        singular = torch.zeros(4, 4, dtype=torch.float64)
        singular[2, 3] = 1.0
        singular[3, 2] = -1.0
        invertible = _random_skew(4, seed=2)
        matrix = torch.stack([singular, invertible])
        pfaffian = PfaffianParlettReid.forward(matrix)
        grad_output = torch.tensor([1.3, -0.7], dtype=torch.float64)
        result = PfaffianParlettReid.pfaffian_grad_matrix(matrix, pfaffian, grad_output)

        expected_invertible = torch.einsum("ij->ji", 0.5 * grad_output[1] * pfaffian[1] * torch.linalg.inv(invertible))
        minor_adjugate = PfaffianParlettReid._pfaffian_adjugate(singular[None])[0]
        expected_singular = torch.einsum("ij->ji", 0.5 * grad_output[0] * minor_adjugate)
        torch.testing.assert_close(
            result[1], expected_invertible, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        torch.testing.assert_close(
            result[0], expected_singular, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_grad_matrix_all_singular_batch_is_finite_and_exact(self):
        # Every element singular: the inverse path is entirely bypassed by the minor adjugate.
        singular = torch.zeros(4, 4, dtype=torch.float64)
        singular[2, 3] = 1.0
        singular[3, 2] = -1.0
        matrix = torch.stack([singular, singular.clone()])
        pfaffian = PfaffianParlettReid.forward(matrix)
        assert bool((pfaffian == 0).all())
        grad_output = torch.tensor([1.0, 2.0], dtype=torch.float64)
        result = PfaffianParlettReid.pfaffian_grad_matrix(matrix, pfaffian, grad_output)
        assert torch.isfinite(result).all()
        minor_adjugate = PfaffianParlettReid._pfaffian_adjugate(matrix)
        expected = torch.einsum("...,...ij->...ji", 0.5 * grad_output, minor_adjugate)
        torch.testing.assert_close(result, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_grad_matrix_does_not_use_svd(self, monkeypatch):
        # Regression for the SVD-non-convergence crash: the gradient must not route through
        # torch.linalg.pinv (SVD) anymore. Force pinv to fail and confirm the gradient still computes.
        def _fail(*args, **kwargs):
            raise AssertionError("torch.linalg.pinv (SVD) must not be used by pfaffian_grad_matrix")

        monkeypatch.setattr(torch.linalg, "pinv", _fail)
        singular = torch.zeros(4, 4, dtype=torch.float64)
        singular[2, 3] = 1.0
        singular[3, 2] = -1.0
        matrix = torch.stack([singular, _random_skew(4, seed=3)])
        pfaffian = PfaffianParlettReid.forward(matrix)
        grad_output = torch.ones(2, dtype=torch.float64)
        result = PfaffianParlettReid.pfaffian_grad_matrix(matrix, pfaffian, grad_output)
        assert torch.isfinite(result).all()

    def test_grad_matrix_ill_conditioned_invertible_is_finite_and_exact(self):
        # The bug surfaced on ill-conditioned skew inputs where the SVD pseudo-inverse fails to
        # converge. The LU inverse stays finite; the gradient must match the inverse closed form.
        scales = torch.tensor([1e8, 1e-8, 1e6, 1e-6], dtype=torch.float64)
        blocks = [torch.tensor([[0.0, scale], [-scale, 0.0]], dtype=torch.float64) for scale in scales]
        matrix = torch.block_diag(*blocks)  # invertible, condition number ~1e16
        pfaffian = PfaffianParlettReid.forward(matrix)
        assert pfaffian != 0.0
        grad_output = torch.tensor(1.0, dtype=torch.float64)
        result = PfaffianParlettReid.pfaffian_grad_matrix(matrix, pfaffian, grad_output)
        expected = torch.einsum("ij->ji", 0.5 * grad_output * pfaffian * torch.linalg.inv(matrix))
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_grad_matrix_numerically_singular_real_does_not_raise(self):
        # A rank-deficient skew matrix has a forward Pfaffian of
        # round-off size, not exactly 0, so an exact ``pf == 0`` test routed it to the LU inverse,
        # which raised on real inputs. The relative magnitude test must route it to the exact
        # minor adjugate instead.
        generator = torch.Generator().manual_seed(4)
        first = torch.randn(6, dtype=torch.float64, generator=generator)
        second = torch.randn(6, dtype=torch.float64, generator=generator)
        matrix = torch.outer(first, second) - torch.outer(second, first)  # rank 2, pf = 0 exactly in math
        pfaffian = PfaffianParlettReid.forward(matrix)
        assert pfaffian != 0.0  # round-off, the regression trigger
        grad_output = torch.tensor(1.0, dtype=torch.float64)
        result = PfaffianParlettReid.pfaffian_grad_matrix(matrix, pfaffian, grad_output)
        minor_adjugate = PfaffianParlettReid._pfaffian_adjugate(matrix[None])[0]
        expected = torch.einsum("ij->ji", 0.5 * grad_output * minor_adjugate)
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_grad_matrix_numerically_singular_complex_is_exact(self):
        # On complex inputs the LU inverse of a
        # numerically singular matrix returns garbage without raising, so the old exact
        # ``pf == 0`` test produced an O(1)-wrong gradient silently.
        generator = torch.Generator().manual_seed(5)
        first = torch.randn(6, dtype=torch.float64, generator=generator) + 1j * torch.randn(
            6, dtype=torch.float64, generator=generator
        )
        second = torch.randn(6, dtype=torch.float64, generator=generator) + 1j * torch.randn(
            6, dtype=torch.float64, generator=generator
        )
        matrix = torch.outer(first, second) - torch.outer(second, first)  # rank 2, pf = 0 exactly in math
        pfaffian = PfaffianParlettReid.forward(matrix)
        grad_output = torch.tensor(1.0 + 0.0j, dtype=torch.complex128)
        result = PfaffianParlettReid.pfaffian_grad_matrix(matrix, pfaffian, grad_output)
        minor_adjugate = PfaffianParlettReid._pfaffian_adjugate(matrix[None])[0]
        expected = torch.einsum("ij->ji", 0.5 * grad_output * minor_adjugate.conj())
        assert torch.isfinite(result.real).all() and torch.isfinite(result.imag).all()
        torch.testing.assert_close(result, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_grad_matrix_residual_check_catches_moderate_pfaffian_ill_conditioning(self):
        # One tiny and several unit singular-value pairs: the Pfaffian magnitude (1e-10) passes the
        # relative magnitude test, but the LU inverse is too inaccurate (condition number ~1e10);
        # the residual check must route the element to the exact minor adjugate.
        scales = torch.tensor([1e-10, 1.0, 1.0, 1.0], dtype=torch.float64)
        blocks = [torch.tensor([[0.0, scale], [-scale, 0.0]], dtype=torch.float64) for scale in scales]
        generator = torch.Generator().manual_seed(6)
        random_full = torch.randn(8, 8, dtype=torch.float64, generator=generator)
        rotation, _ = torch.linalg.qr(random_full)
        matrix = rotation.transpose(-1, -2) @ torch.block_diag(*blocks) @ rotation
        matrix = 0.5 * (matrix - matrix.transpose(-1, -2))
        pfaffian = PfaffianParlettReid.forward(matrix)
        grad_output = torch.tensor(1.0, dtype=torch.float64)
        result = PfaffianParlettReid.pfaffian_grad_matrix(matrix, pfaffian, grad_output)
        minor_adjugate = PfaffianParlettReid._pfaffian_adjugate(matrix[None])[0]
        expected = torch.einsum("ij->ji", 0.5 * grad_output * minor_adjugate)
        torch.testing.assert_close(result, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_grad_matrix_lu_singular_above_magnitude_threshold_does_not_raise(self):
        # Regression for the backward core dump: a matrix that is exactly singular to the LU
        # factorization (a zero pivot) can carry a forward Pfaffian whose magnitude sits *above* the
        # relative magnitude threshold (the forward's round-off for a singular matrix can exceed
        # eps^0.75 * scale^(n/2)), so the magnitude test never flags it. It then reached the raising
        # torch.linalg.inv and, inside autograd backward, aborted the process. The LU factorization
        # (its info and pivot ratio) must flag it and route it to the exact minor adjugate.
        singular = torch.zeros(4, 4, dtype=torch.float64)
        singular[2, 3] = 1.0
        singular[3, 2] = -1.0  # rank 2, exactly singular, true pf = 0
        entry_scale = singular.abs().amax()
        threshold = (
            entry_scale ** (4 // 2) * torch.finfo(torch.float64).eps ** PfaffianStrategy.SINGULARITY_RTOL_EXPONENT
        )
        above_threshold_pfaffian = torch.tensor(0.5, dtype=torch.float64)
        assert above_threshold_pfaffian.abs() > threshold  # not caught by the magnitude test
        grad_output = torch.tensor(1.0, dtype=torch.float64)
        result = PfaffianParlettReid.pfaffian_grad_matrix(singular, above_threshold_pfaffian, grad_output)
        minor_adjugate = PfaffianParlettReid._pfaffian_adjugate(singular[None])[0]
        expected = torch.einsum("ij->ji", 0.5 * grad_output * minor_adjugate)
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_grad_matrix_lu_singular_complex_above_threshold_is_finite_and_exact(self):
        # The complex counterpart: torch.linalg.inv returns garbage (no raise) for a numerically
        # singular complex matrix above the magnitude threshold, silently corrupting the gradient.
        # The LU info flag must route it to the exact minor adjugate instead.
        singular = torch.zeros(4, 4, dtype=torch.complex128)
        singular[2, 3] = 1.0 + 0.0j
        singular[3, 2] = -1.0 + 0.0j  # rank 2, exactly singular, true pf = 0
        above_threshold_pfaffian = torch.tensor(0.5 + 0.0j, dtype=torch.complex128)
        grad_output = torch.tensor(1.0 + 0.0j, dtype=torch.complex128)
        result = PfaffianParlettReid.pfaffian_grad_matrix(singular, above_threshold_pfaffian, grad_output)
        minor_adjugate = PfaffianParlettReid._pfaffian_adjugate(singular[None])[0]
        expected = torch.einsum("ij->ji", 0.5 * grad_output * minor_adjugate.conj())
        assert torch.isfinite(result.real).all() and torch.isfinite(result.imag).all()
        torch.testing.assert_close(result, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_grad_matrix_mixed_lu_singular_batch_matches_closed_forms(self):
        # A batch mixing an LU-singular element (above the magnitude threshold) with an invertible
        # element: the batched torch.linalg.inv_ex must not raise, and each element must match its
        # exact closed form. Guards against inf/nan from the singular element's discarded inverse
        # leaking into the invertible element's gradient.
        singular = torch.zeros(4, 4, dtype=torch.float64)
        singular[2, 3] = 1.0
        singular[3, 2] = -1.0
        invertible = _random_skew(4, seed=7)
        matrix = torch.stack([singular, invertible])
        invertible_pfaffian = PfaffianParlettReid.forward(invertible)
        pfaffian = torch.stack([torch.tensor(0.5, dtype=torch.float64), invertible_pfaffian])
        grad_output = torch.tensor([1.3, -0.7], dtype=torch.float64)
        result = PfaffianParlettReid.pfaffian_grad_matrix(matrix, pfaffian, grad_output)
        expected_invertible = torch.einsum(
            "ij->ji", 0.5 * grad_output[1] * invertible_pfaffian * torch.linalg.inv(invertible)
        )
        minor_adjugate = PfaffianParlettReid._pfaffian_adjugate(singular[None])[0]
        expected_singular = torch.einsum("ij->ji", 0.5 * grad_output[0] * minor_adjugate)
        assert torch.isfinite(result).all()
        torch.testing.assert_close(
            result[1], expected_invertible, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        torch.testing.assert_close(
            result[0], expected_singular, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_grad_matrix_empty_matrix_returns_empty_gradient(self):
        # A 0x0 matrix has the constant Pfaffian 1, so the forward supports dimension == 0. The
        # backward must return the empty (..., 0, 0) gradient instead of raising on the zero-size
        # amax reduction used by the singularity criteria.
        matrix = torch.zeros(1, 0, 0, dtype=torch.float64)
        pfaffian = torch.ones(1, dtype=torch.float64)
        grad_output = torch.ones(1, dtype=torch.float64)
        result = PfaffianParlettReid.pfaffian_grad_matrix(matrix, pfaffian, grad_output)
        assert result.shape == matrix.shape
        assert result.dtype == matrix.dtype

    def test_class_singularity_constants(self):
        assert PfaffianStrategy.SINGULARITY_RTOL_EXPONENT == 0.75
        assert PfaffianStrategy.INVERSE_RESIDUAL_RTOL_EXPONENT == 0.5

    def test_exact_singular_grad_default_is_true(self):
        # The default keeps the historical exact minor-based adjugate at singular elements.
        assert PfaffianStrategy.EXACT_SINGULAR_GRAD is True

    def test_grad_matrix_sync_free_gives_zero_grad_at_singular(self, monkeypatch):
        # With EXACT_SINGULAR_GRAD disabled the backward is host-sync-free: exactly-singular elements
        # (pf == 0) receive an exactly zero gradient, while invertible elements match the inverse form.
        monkeypatch.setattr(PfaffianParlettReid, "EXACT_SINGULAR_GRAD", False)
        singular = torch.zeros(4, 4, dtype=torch.float64)
        singular[2, 3] = 1.0
        singular[3, 2] = -1.0
        invertible = _random_skew(4, seed=1)
        matrix = torch.stack([singular, invertible])
        pfaffian = PfaffianParlettReid.forward(matrix)
        assert pfaffian[0] == 0.0  # the improved forward returns exactly 0 for the singular element
        grad_output = torch.ones(2, dtype=torch.float64)
        result = PfaffianParlettReid.pfaffian_grad_matrix(matrix, pfaffian, grad_output)
        assert torch.isfinite(result).all()
        assert torch.all(result[0] == 0)  # sync-free: singular element gets exactly zero gradient
        expected_invertible = torch.einsum("ij->ji", 0.5 * grad_output[1] * pfaffian[1] * torch.linalg.inv(invertible))
        torch.testing.assert_close(
            result[1], expected_invertible, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    def test_flagged_inverse_does_not_flag_well_conditioned_block_diagonal(self, dtype):
        # Regression: |pf| <= eps^0.75 * max|A_ij|^(n/2) flagged block-diagonal matrices whose entries span
        # a few orders of magnitude (|pf| = prod |c_k| is far below max|c|^(n/2)), sending them to the
        # minor-based adjugate. The scale-free pivot ratio keeps them on the single inverse.
        dimension = 64
        generator = torch.Generator().manual_seed(TEST_SEED)
        coefficients = torch.randn(8, dimension // 2, dtype=dtype, generator=generator)
        matrix = torch.zeros(8, dimension, dimension, dtype=dtype)
        even = torch.arange(0, dimension, 2)
        matrix[:, even, even + 1] = coefficients
        matrix[:, even + 1, even] = -coefficients
        inverse, rank_flag, residual_flag = PfaffianStrategy._flagged_inverse(matrix)
        assert not (rank_flag | residual_flag).any()
        torch.testing.assert_close(
            inverse, torch.linalg.inv(matrix), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
    def test_flagged_inverse_flags_exactly_singular_and_zero_matrices(self, dtype):
        rank_deficient = _random_skew(6, seed=3).to(dtype)
        rank_deficient[0], rank_deficient[:, 0] = rank_deficient[1], rank_deficient[:, 1]  # duplicate row/col
        zero = torch.zeros(6, 6, dtype=dtype)
        invertible = _random_skew(6, seed=4).to(dtype)
        inverse, rank_flag, residual_flag = PfaffianStrategy._flagged_inverse(
            torch.stack([rank_deficient, zero, invertible])
        )
        assert rank_flag.tolist() == [True, True, False]
        assert not residual_flag.any()  # the residual flag only marks elements the rank flag missed
        torch.testing.assert_close(inverse[0], torch.eye(6, dtype=dtype))  # flagged: solved against identity
        assert torch.isfinite(inverse).all()

    def test_flagged_inverse_flags_nan_input(self):
        matrix = _random_skew(4, seed=5)
        matrix[0, 1] = float("nan")
        _, rank_flag, residual_flag = PfaffianStrategy._flagged_inverse(matrix[None])
        assert (rank_flag | residual_flag).tolist() == [True]

    def test_grad_matrix_skips_minor_adjugate_where_grad_output_is_zero(self, monkeypatch):
        # The gradient is linear in grad_output, so a flagged element with zero grad_output gets a zero
        # gradient without the (expensive) minor-based adjugate.
        def _fail(matrices):
            raise AssertionError("the minor-based adjugate must not run for zero grad_output")

        monkeypatch.setattr(PfaffianParlettReid, "_pfaffian_adjugate", classmethod(lambda cls, m: _fail(m)))
        singular = torch.zeros(4, 4, dtype=torch.float64)
        singular[2, 3], singular[3, 2] = 1.0, -1.0
        invertible = _random_skew(4, seed=6)
        matrix = torch.stack([singular, invertible])
        pfaffian = PfaffianParlettReid.forward(matrix)
        grad_output = torch.tensor([0.0, 2.0], dtype=torch.float64)
        result = PfaffianParlettReid.pfaffian_grad_matrix(matrix, pfaffian, grad_output)
        assert torch.all(result[0] == 0)
        expected = torch.einsum("ij->ji", 0.5 * grad_output[1] * pfaffian[1] * torch.linalg.inv(invertible))
        torch.testing.assert_close(result[1], expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_grad_matrix_accepts_broadcast_grad_output(self):
        # grad_output from .sum() arrives as an expanded scalar; it must broadcast over the batch.
        matrix = torch.stack([_random_skew(4, seed=7), _random_skew(4, seed=8)])
        pfaffian = PfaffianParlettReid.forward(matrix)
        broadcast = torch.ones((), dtype=torch.float64).expand(2)
        result = PfaffianParlettReid.pfaffian_grad_matrix(matrix, pfaffian, broadcast)
        expected = torch.einsum("...,...ij->...ji", 0.5 * pfaffian, torch.linalg.inv(matrix))
        torch.testing.assert_close(result, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_double_backward_is_finite_at_exact_zero_pivot(self):
        # Regression: differentiating through the LU factors gave a NaN Hessian at an exact zero pivot.
        matrix = torch.zeros(2, 8, 8, dtype=torch.float64)
        matrix[1] = _random_skew(8, seed=9)
        matrix.requires_grad_(True)
        pfaffian = PfaffianParlettReid.apply(matrix)
        (gradient,) = torch.autograd.grad(pfaffian.sum(), matrix, create_graph=True)
        (hessian_vector,) = torch.autograd.grad((gradient**2).sum(), matrix)
        assert torch.isfinite(hessian_vector).all()

    def test_jvp_at_corank_two_matches_finite_difference(self):
        # Regression: jvp passes a zero grad_output that requires grad; skipping the exact adjugate for it
        # zeroed the forward-mode derivative at a flagged element.
        matrix = _random_skew(6, seed=10)
        matrix[0], matrix[:, 0] = matrix[1].clone(), matrix[:, 1].clone()  # corank 2, pf = 0
        matrix[0, 0] = 0.0
        direction = _random_skew(6, seed=11)
        _, derivative = torch.autograd.functional.jvp(PfaffianParlettReid.apply, (matrix,), (direction,))
        step = 1e-6
        finite_difference = (
            PfaffianParlettReid.apply(matrix + step * direction) - PfaffianParlettReid.apply(matrix - step * direction)
        ) / (2 * step)
        torch.testing.assert_close(
            derivative, finite_difference, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON
        )
        assert derivative.abs() > ATOL_APPROX_COMPARISON  # the true derivative is not zero

    def test_grad_matrix_pure_imaginary_grad_output_uses_exact_adjugate(self):
        # A purely imaginary grad_output is nonzero: the flagged element must still get its gradient.
        singular = torch.zeros(4, 4, dtype=torch.complex128)
        singular[2, 3], singular[3, 2] = 1.0, -1.0
        pfaffian = PfaffianParlettReid.forward(singular[None])
        grad_output = torch.tensor([1j], dtype=torch.complex128)
        result = PfaffianParlettReid.pfaffian_grad_matrix(singular[None], pfaffian, grad_output)
        assert result.abs().amax() > 0

    def test_grad_matrix_underflowed_pfaffian_keeps_gradient(self):
        # Regression: a well-conditioned float32 matrix with tiny entries has a Pfaffian that underflows
        # to 0; the pivot ratio cannot see it, so pf == 0 must flag it for the exact adjugate.
        matrix = (1e-12 * _random_skew(8, seed=12)).to(torch.float32)
        pfaffian = PfaffianParlettReid.forward(matrix[None])
        assert pfaffian.item() == 0.0
        grad_output = torch.ones(1, dtype=torch.float32)
        result = PfaffianParlettReid.pfaffian_grad_matrix(matrix[None], pfaffian, grad_output)[0]
        reference_matrix = matrix.to(torch.float64)
        reference_pfaffian = PfaffianParlettReid.forward(reference_matrix)
        expected = 0.5 * reference_pfaffian * torch.linalg.inv(reference_matrix).transpose(-1, -2)
        torch.testing.assert_close(
            result.to(torch.float64) / expected.abs().amax(),
            expected / expected.abs().amax(),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    def test_grad_matrix_float32_rank_deficient_block_needs_pivot_ratio(self):
        # The residual check is blind on this float32 rank-1 off-diagonal block; only the pivot ratio
        # routes it to the exact adjugate (without it the gradient is ~18% off).
        block = torch.tensor([[-0.04963864, -0.0144493], [-0.24733615, -0.07199704]], dtype=torch.float32)
        zero = torch.zeros(2, 2, dtype=torch.float32)
        matrix = torch.cat([torch.cat([zero, block], -1), torch.cat([-block.T, zero], -1)], -2)
        pfaffian = PfaffianParlettReid.forward(matrix[None])
        result = PfaffianParlettReid.pfaffian_grad_matrix(matrix[None], pfaffian, torch.ones(1))[0]
        with torch.no_grad():
            exact = PfaffianParlettReid._pfaffian_adjugate(matrix.to(torch.float64)[None])[0]
        expected = 0.5 * exact.transpose(-1, -2)
        torch.testing.assert_close(
            result.to(torch.float64), expected, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON
        )
