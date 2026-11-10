import pytest
import torch

from tests.configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    ATOL_SINGLE_PRECISION_COMPARISON,
    RTOL_APPROX_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    RTOL_SINGLE_PRECISION_COMPARISON,
    TEST_SEED,
)
from torch_pfaffian.strategies import strategy as strategy_module
from torch_pfaffian.strategies.pfaffian_parlett_reid import PfaffianParlettReid
from torch_pfaffian.strategies.strategy import PfaffianStrategy


def _random_skew(dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    full = torch.randn(dimension, dimension, dtype=torch.float64, generator=generator)
    return full - full.transpose(-1, -2)


def _rotated_corank(count: int, dimension: int, corank: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    # Q^T B Q with B block-diagonal skew whose first corank / 2 blocks are zero: numerically singular, with a
    # round-off Pfaffian. A random complex Q (not unitary) keeps the skew-symmetry and the corank.
    generator = torch.Generator().manual_seed(seed)
    coefficients = torch.rand(count, dimension // 2, dtype=torch.float64, generator=generator) + 0.5
    coefficients[:, : corank // 2] = 0.0
    block = torch.zeros(count, dimension, dimension, dtype=torch.float64)
    even = torch.arange(0, dimension, 2)
    block[:, even, even + 1] = coefficients
    block[:, even + 1, even] = -coefficients
    if dtype.is_complex:
        rotation = torch.complex(
            torch.randn(count, dimension, dimension, dtype=torch.float64, generator=generator),
            torch.randn(count, dimension, dimension, dtype=torch.float64, generator=generator),
        )
    else:
        rotation, _ = torch.linalg.qr(
            torch.randn(count, dimension, dimension, dtype=torch.float64, generator=generator)
        )
    return (rotation.mT @ block.to(rotation.dtype) @ rotation).to(dtype)


def _loop_adjugate(matrices: torch.Tensor) -> torch.Tensor:
    # Reference definition P_ij = (-1)^(i+j) pf(A^(ij)) for i < j, one minor at a time.
    dimension = matrices.shape[-1]
    adjugate = torch.zeros_like(matrices)
    indices = torch.arange(dimension)
    for first in range(dimension):
        for second in range(first + 1, dimension):
            keep = indices[(indices != first) & (indices != second)]
            minor = matrices.index_select(-2, keep).index_select(-1, keep)
            value = (-1) ** (first + second) * PfaffianParlettReid.forward(minor)
            adjugate[..., first, second] = value
            adjugate[..., second, first] = -value
    return adjugate


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
        assert PfaffianStrategy.NULL_SPACE_SEPARATION_EXPONENT == 0.25
        assert PfaffianStrategy.MINOR_CHUNK_ELEMENTS == 1 << 20

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

    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    @pytest.mark.parametrize("dimension", [2, 5, 6, 9])
    def test_pfaffian_adjugate_matches_minor_definition(self, dimension, dtype, monkeypatch):
        matrices = _rotated_corank(3, dimension, 0, dtype, seed=20) if dimension % 2 == 0 else None
        if matrices is None:
            entries = torch.randn(
                3, dimension, dimension, dtype=torch.float64, generator=torch.Generator().manual_seed(21)
            )
            matrices = (entries - entries.mT).to(dtype)
        expected = _loop_adjugate(matrices)
        torch.testing.assert_close(PfaffianParlettReid._pfaffian_adjugate(matrices), expected)
        monkeypatch.setattr(PfaffianParlettReid, "MINOR_CHUNK_ELEMENTS", 1)  # one pair per chunk
        torch.testing.assert_close(PfaffianParlettReid._pfaffian_adjugate(matrices), expected)

    def test_pfaffian_adjugate_of_empty_batch(self):
        assert PfaffianParlettReid._pfaffian_adjugate(torch.zeros(0, 6, 6)).shape == (0, 6, 6)

    @staticmethod
    def _forbid_minors(monkeypatch):
        def _fail(matrices):
            raise AssertionError("the minor Pfaffians must not be used here")

        monkeypatch.setattr(PfaffianParlettReid, "_pfaffian_adjugate", staticmethod(_fail))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
    @pytest.mark.parametrize("dimension", [6, 8, 16])
    def test_singular_adjugate_corank_two_uses_rank_two_formula(self, dimension, dtype, monkeypatch):
        matrices = _rotated_corank(4, dimension, 2, dtype, seed=dimension)
        expected = _loop_adjugate(matrices.to(torch.complex128 if dtype.is_complex else torch.float64))
        self._forbid_minors(monkeypatch)
        result = PfaffianParlettReid._singular_pfaffian_adjugate(matrices)
        assert result.dtype == dtype
        scale = expected.abs().amax()
        double = dtype in (torch.float64, torch.complex128)
        atol = ATOL_MATRIX_COMPARISON if double else ATOL_SINGLE_PRECISION_COMPARISON
        rtol = RTOL_MATRIX_COMPARISON if double else RTOL_SINGLE_PRECISION_COMPARISON
        torch.testing.assert_close(result.to(expected.dtype) / scale, expected / scale, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    @pytest.mark.parametrize("corank", [4, 6])
    def test_singular_adjugate_corank_four_or_more_is_at_round_off_level(self, corank, dtype, monkeypatch):
        # The true adjugate vanishes; the rank-2 formula returns round-off relative to the natural scale
        # sigma_max^(n/2 - 1) of the adjugate, without the minors.
        matrices = _rotated_corank(4, 8, corank, dtype, seed=corank)
        natural = torch.linalg.svdvals(matrices)[:, 0] ** 3
        self._forbid_minors(monkeypatch)
        result = PfaffianParlettReid._singular_pfaffian_adjugate(matrices)
        relative = result.abs().amax(dim=(-2, -1)) / natural
        assert torch.all(relative <= 8 * torch.finfo(dtype).eps)

    def test_singular_adjugate_keeps_exact_minors_for_wide_dynamic_range(self):
        # Invertible block-diagonal matrix whose blocks span 16 orders of magnitude: its small singular values
        # are genuine, so neither shortcut applies and the exact minors give the true adjugate pf A^{-1}.
        scales = [1e8, 1e-8, 1e6, 1e-6]
        matrix = torch.zeros(8, 8, dtype=torch.float64)
        for index, scale in enumerate(scales):
            matrix[2 * index, 2 * index + 1], matrix[2 * index + 1, 2 * index] = scale, -scale
        result = PfaffianParlettReid._singular_pfaffian_adjugate(matrix[None])[0]
        expected = PfaffianParlettReid.forward(matrix) * torch.linalg.inv(matrix)
        torch.testing.assert_close(result, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_singular_adjugate_mixed_batch_routes_each_element(self, monkeypatch):
        corank_two = _rotated_corank(2, 8, 2, torch.float64, seed=30)
        corank_four = _rotated_corank(1, 8, 4, torch.float64, seed=31)
        with_nan = _rotated_corank(1, 8, 2, torch.float64, seed=32)
        with_nan[0, 0, 1] = float("nan")
        matrices = torch.cat([corank_two, corank_four, with_nan])
        calls = []
        original = PfaffianParlettReid._pfaffian_adjugate

        def _recording(matrices_subset):
            calls.append(matrices_subset.shape[0])
            return original(matrices_subset)

        monkeypatch.setattr(PfaffianParlettReid, "_pfaffian_adjugate", staticmethod(_recording))
        result = PfaffianParlettReid._singular_pfaffian_adjugate(matrices)
        assert calls == [1]  # only the non-finite element takes the minors
        torch.testing.assert_close(
            result[:2], _loop_adjugate(corank_two), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_APPROX_COMPARISON
        )
        natural = torch.linalg.svdvals(corank_four[0])[0] ** 3
        assert result[2].abs().amax() <= 8 * torch.finfo(torch.float64).eps * natural
        torch.testing.assert_close(result[3], _loop_adjugate(with_nan)[0], equal_nan=True)

    def test_singular_adjugate_falls_back_when_svd_fails(self, monkeypatch):
        def _no_convergence(*args, **kwargs):
            raise torch.linalg.LinAlgError("SVD did not converge")

        matrices = _rotated_corank(2, 8, 2, torch.float64, seed=33)
        monkeypatch.setattr(torch.linalg, "svd", _no_convergence)
        torch.testing.assert_close(PfaffianParlettReid._singular_pfaffian_adjugate(matrices), _loop_adjugate(matrices))

    @pytest.mark.parametrize(("dimension", "expected_zero"), [(5, True), (4, False), (2, False)])
    def test_singular_adjugate_small_and_odd_dimensions(self, dimension, expected_zero):
        entries = torch.randn(2, dimension, dimension, dtype=torch.float64, generator=torch.Generator().manual_seed(34))
        matrices = entries - entries.mT
        result = PfaffianParlettReid._singular_pfaffian_adjugate(matrices)
        if expected_zero:
            assert torch.all(result == 0)
        else:
            torch.testing.assert_close(result, _loop_adjugate(matrices))
        assert PfaffianParlettReid._singular_pfaffian_adjugate(matrices[:0]).shape == (0, dimension, dimension)

    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    def test_grad_matrix_sync_free_zeroes_round_off_singular_elements(self, dtype, monkeypatch):
        # Regression for issue #20: the sync-free mode only tested pf == 0, so a round-off Pfaffian got
        # pf * A^{-1} (noise times a huge inverse, up to 3e2 or NaN). It now uses the exact mode's flags.
        monkeypatch.setattr(PfaffianParlettReid, "EXACT_SINGULAR_GRAD", False)
        singular = _rotated_corank(3, 8, 2, dtype, seed=35)
        regular = _rotated_corank(1, 8, 0, dtype, seed=36)
        matrix = torch.cat([singular, regular])
        pfaffian = PfaffianParlettReid.forward(matrix)
        assert torch.any(pfaffian[:3] != 0)  # at least one round-off Pfaffian, not an exact zero
        grad_output = torch.ones(4, dtype=dtype)
        result = PfaffianParlettReid.pfaffian_grad_matrix(matrix, pfaffian, grad_output)
        assert torch.all(result[:3] == 0)
        expected = 0.5 * (pfaffian[3] * torch.linalg.inv(matrix[3])).transpose(-1, -2).conj()
        torch.testing.assert_close(result[3], expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    def test_signed_backward_on_singular_batch_matches_minor_definition(self, dtype):
        # End to end: the gradient of the signed Pfaffian on corank-2 and corank-4 elements equals the one from
        # the minor-by-minor adjugate.
        matrix = torch.cat([_rotated_corank(2, 10, 2, dtype, seed=37), _rotated_corank(1, 10, 4, dtype, seed=38)])
        leaf = matrix.clone().requires_grad_(True)
        PfaffianParlettReid.apply(leaf).real.sum().backward()
        expected = 0.5 * _loop_adjugate(matrix).transpose(-1, -2).conj()
        scale = expected.abs().amax()
        torch.testing.assert_close(
            leaf.grad / scale, expected / scale, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("coefficients", [[0.0, 1e10, 1e10, 1e-8], [0.0, 1.0, 1.0, 1e-15]])
    def test_singular_adjugate_is_exact_on_structured_inputs(self, coefficients):
        # Regression: setting the adjugate to zero at corank >= 4 lost the exact gradient of block-diagonal
        # inputs with one zero block and one block below the round-off level (5e11 for the first one).
        matrix = torch.zeros(8, 8, dtype=torch.float64)
        for index, coefficient in enumerate(coefficients):
            matrix[2 * index, 2 * index + 1], matrix[2 * index + 1, 2 * index] = coefficient, -coefficient
        result = PfaffianParlettReid._singular_pfaffian_adjugate(matrix[None])[0]
        expected = _loop_adjugate(matrix[None])[0]
        scale = expected.abs().amax()
        assert scale > 0
        torch.testing.assert_close(
            result / scale, expected / scale, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_rank_two_adjugate_does_not_overflow_near_the_float_range(self):
        # Regression: entry / wedge_best (|wedge_best| < 1) overflowed although every adjugate entry is at most
        # |entry|; here the true entries reach 1.6e308 and |wedge_best| = 0.52, so entry / wedge_best overflows.
        matrices = _rotated_corank(1, 6, 2, torch.float64, seed=40) * 10.0**154.19
        expected = _loop_adjugate(matrices)
        result = PfaffianParlettReid._singular_pfaffian_adjugate(matrices)
        assert torch.isfinite(expected).all() and torch.isfinite(result).all()
        scale = expected.abs().amax()
        torch.testing.assert_close(
            result / scale, expected / scale, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_backward_uses_the_singular_adjugate(self, monkeypatch):
        calls = []
        original = PfaffianParlettReid._singular_pfaffian_adjugate

        def _recording(matrices):
            calls.append(matrices.shape[0])
            return original(matrices)

        monkeypatch.setattr(PfaffianParlettReid, "_singular_pfaffian_adjugate", staticmethod(_recording))
        singular = _rotated_corank(2, 8, 2, torch.float64, seed=41)
        regular = _rotated_corank(3, 8, 0, torch.float64, seed=42)
        leaf = torch.cat([singular, regular]).requires_grad_(True)
        PfaffianParlettReid.apply(leaf).sum().backward()
        assert calls == [2]  # only the two singular elements

    def test_singular_adjugate_without_clear_truncation_takes_the_minors(self, monkeypatch):
        # sigma_min / sigma_3 = 1e-10 lies between eps^0.75 (2e-12) and eps^0.5 (1.5e-8): the rank-2 truncation
        # would be off by ~1e-10, so the exact minors must be used.
        block = torch.zeros(8, 8, dtype=torch.float64)
        even = torch.arange(0, 8, 2)
        coefficients = torch.tensor([1e-10, 1.0, 1.0, 1.0], dtype=torch.float64)
        block[even, even + 1], block[even + 1, even] = coefficients, -coefficients
        generator = torch.Generator().manual_seed(44)
        rotation, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64, generator=generator))
        matrices = (rotation.T @ block @ rotation)[None]
        calls = []
        original = PfaffianParlettReid._pfaffian_adjugate

        def _recording(subset):
            calls.append(subset.shape[0])
            return original(subset)

        monkeypatch.setattr(PfaffianParlettReid, "_pfaffian_adjugate", staticmethod(_recording))
        result = PfaffianParlettReid._singular_pfaffian_adjugate(matrices)
        assert calls == [1]
        torch.testing.assert_close(
            result, _loop_adjugate(matrices), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("smallest", [1e-9, 1e-13])
    def test_grad_matrix_sync_free_keeps_invertible_ill_conditioned_gradient(self, smallest, monkeypatch):
        # Regression: the sync-free mode reused the exact mode's flags (residual, eps^0.75 pivot ratio) and
        # zeroed genuinely invertible elements from cond ~1e9; only round-off level singularity is zeroed now.
        monkeypatch.setattr(PfaffianParlettReid, "EXACT_SINGULAR_GRAD", False)
        matrix = torch.zeros(1, 8, 8, dtype=torch.float64)
        for index, coefficient in enumerate([smallest, 1.0, 1.0, 1.0]):
            matrix[0, 2 * index, 2 * index + 1], matrix[0, 2 * index + 1, 2 * index] = coefficient, -coefficient
        pfaffian = PfaffianParlettReid.forward(matrix)
        result = PfaffianParlettReid.pfaffian_grad_matrix(matrix, pfaffian, torch.ones(1, dtype=torch.float64))
        torch.testing.assert_close(
            result[0, 0, 1],
            torch.tensor(0.5, dtype=torch.float64),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_grad_matrix_sync_free_propagates_nan(self, monkeypatch):
        monkeypatch.setattr(PfaffianParlettReid, "EXACT_SINGULAR_GRAD", False)
        matrix = _rotated_corank(1, 8, 0, torch.float64, seed=45)
        matrix[0, 0, 1] = float("nan")
        pfaffian = PfaffianParlettReid.forward(matrix)
        result = PfaffianParlettReid.pfaffian_grad_matrix(matrix, pfaffian, torch.ones(1, dtype=torch.float64))
        assert torch.isnan(result).any()

    def test_sync_free_supports_double_backward(self, monkeypatch):
        monkeypatch.setattr(PfaffianParlettReid, "EXACT_SINGULAR_GRAD", False)
        singular = _rotated_corank(1, 6, 2, torch.float64, seed=46)
        regular = _rotated_corank(1, 6, 0, torch.float64, seed=47)
        leaf = torch.cat([singular, regular]).requires_grad_(True)
        (gradient,) = torch.autograd.grad(PfaffianParlettReid.apply(leaf).sum(), leaf, create_graph=True)
        (second,) = torch.autograd.grad(gradient.pow(2).sum(), leaf)
        assert torch.isfinite(second).all()

    @staticmethod
    def _native_cases(dtype: torch.dtype, size: int, batch: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(TEST_SEED + size)
        entries = torch.randn(batch, size, size, generator=generator, dtype=torch.float64)
        if dtype.is_complex:
            entries = torch.complex(entries, torch.randn(entries.shape, generator=generator, dtype=torch.float64))
        matrices = (entries - entries.mT).to(dtype)
        matrices[1] = 0  # rank-flagged
        matrices[2, :, 0] = 0  # rank-flagged: an exactly zero column and row
        matrices[2, 0, :] = 0
        matrices[3, 0, 1] = float("nan")  # NaN: flagged
        return matrices

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
    @pytest.mark.parametrize(("size", "batch"), [(8, 6), (64, 9)])  # one thread, and a parallel batch
    def test_native_flagged_inverse_matches_pytorch(self, dtype, size, batch, monkeypatch):
        pytest.importorskip("torch_pfaffian._rust")
        matrices = self._native_cases(dtype, size, batch)
        assert PfaffianStrategy._native_flagged_inverse(matrices) is not None
        native = PfaffianStrategy._flagged_inverse(matrices)
        monkeypatch.setattr(strategy_module, "_NATIVE_INVERSE", {})
        reference = PfaffianStrategy._flagged_inverse(matrices)
        torch.testing.assert_close(native[1], reference[1])
        torch.testing.assert_close(native[2], reference[2])
        kept = ~(reference[1] | reference[2])
        if dtype in (
            torch.float32,
            torch.complex64,
        ):  # random 64 x 64 matrices can be ill-conditioned in single precision
            atol, rtol = ATOL_APPROX_COMPARISON, RTOL_APPROX_COMPARISON
        else:
            atol, rtol = ATOL_MATRIX_COMPARISON, RTOL_MATRIX_COMPARISON
        torch.testing.assert_close(native[0][kept], reference[0][kept], atol=atol, rtol=rtol)

    def test_native_paths_apply_only_where_faster_and_without_a_graph(self):
        pytest.importorskip("torch_pfaffian._rust")
        small, large = torch.zeros(1, 8, 8, dtype=torch.float64), torch.zeros(1, 64, 64, dtype=torch.float64)
        assert PfaffianStrategy._native_flagged_inverse(small) is not None
        assert PfaffianStrategy._native_flagged_inverse(large) is None  # one large matrix: LAPACK is faster
        assert PfaffianStrategy._native_flagged_inverse(large.expand(8, 64, 64)) is not None  # parallel batch
        assert PfaffianStrategy._native_flagged_inverse(small.half()) is None  # no kernel for float16
        assert PfaffianStrategy._native_flagged_inverse(small.requires_grad_(True)) is None  # double backward
        pfaffian_value, weight = torch.ones(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64)
        assert PfaffianStrategy._native_grad_matrix(small.detach(), pfaffian_value, weight.requires_grad_(True)) is None
        assert PfaffianStrategy._native_grad_matrix(large, pfaffian_value, torch.ones(1, dtype=torch.float64)) is None

    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    def test_native_gradient_matches_pytorch_including_singular_elements(self, dtype, monkeypatch):
        pytest.importorskip("torch_pfaffian._rust")
        matrices = self._native_cases(dtype, 8, 6)
        matrices[3] = matrices[4]  # no NaN here: the singular elements take the exact adjugate
        matrices[2, 1, 2], matrices[2, 2, 1] = 1.0, -1.0
        pfaffians = PfaffianParlettReid.forward(matrices)
        grad_output = torch.arange(1.0, 7.0, dtype=torch.float64).to(dtype)
        grad_output[5] = 0  # skipped, though singular or not
        regular = [0, 4]  # no singular element: no exact adjugate on either path
        native = PfaffianParlettReid.pfaffian_grad_matrix(matrices, pfaffians, grad_output)
        native_regular = PfaffianParlettReid.pfaffian_grad_matrix(
            matrices[regular], pfaffians[regular], grad_output[regular]
        )
        monkeypatch.setattr(strategy_module, "_NATIVE_INVERSE", {})
        reference = PfaffianParlettReid.pfaffian_grad_matrix(matrices, pfaffians, grad_output)
        reference_regular = PfaffianParlettReid.pfaffian_grad_matrix(
            matrices[regular], pfaffians[regular], grad_output[regular]
        )
        torch.testing.assert_close(native, reference, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)
        torch.testing.assert_close(
            native_regular, reference_regular, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
