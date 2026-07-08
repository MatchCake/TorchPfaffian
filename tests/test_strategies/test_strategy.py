import torch

from tests.configs import ATOL_MATRIX_COMPARISON, RTOL_MATRIX_COMPARISON
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
        # torch.linalg.inv and, inside autograd backward, aborted the process. The LU info from
        # torch.linalg.inv_ex must flag it and route it to the exact minor adjugate.
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
