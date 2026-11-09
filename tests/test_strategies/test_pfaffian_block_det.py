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
from torch_pfaffian.strategies.pfaffian_block_det import PfaffianBlockDet

_RNG = np.random.default_rng(TEST_SEED)
_BLOCK_SIZES = [1, 2, 3, 4]
_RANDOM_BLOCKS = [_RNG.random((size, size)) for size in _BLOCK_SIZES for _ in range(N_RANDOM_TESTS_PER_CASE)]


def _skew_general(dimension: int) -> np.ndarray:
    entries = _RNG.normal(size=(dimension, dimension))
    return entries - entries.T


def _block_antidiagonal(block: np.ndarray) -> torch.Tensor:
    # Build the skew-symmetric matrix [[0, block], [-block^T, 0]] that PfaffianBlockDet targets.
    zero = np.zeros_like(block)
    top = np.concatenate([zero, block], axis=-1)
    bottom = np.concatenate([-np.einsum("...ij->...ji", block), zero], axis=-1)
    return torch.tensor(np.concatenate([top, bottom], axis=-2))


def _well_conditioned_blocks(sizes: list[int], count: int, min_abs_det: float = 0.1) -> list[np.ndarray]:
    # The gradient relies on the inverse of the upper-right block, which is unstable for near-singular
    # blocks, so the gradcheck inputs are restricted to blocks whose determinant is away from zero.
    blocks = []
    for size in sizes:
        kept = 0
        while kept < count:
            candidate = _RNG.random((size, size))
            if np.abs(np.linalg.det(candidate)) > min_abs_det:
                blocks.append(candidate)
                kept += 1
    return blocks


_GRADCHECK_BLOCKS = _well_conditioned_blocks(_BLOCK_SIZES, N_RANDOM_TESTS_PER_CASE)


def _block_antidiagonal_torch(block: torch.Tensor) -> torch.Tensor:
    # Differentiable [[0, block], [-block^T, 0]] so gradcheck perturbs only the free block entries.
    zero = torch.zeros_like(block)
    top = torch.cat([zero, block], dim=-1)
    bottom = torch.cat([-block.transpose(-1, -2), zero], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def _random_complex_skew(dimension: int) -> torch.Tensor:
    entries = _RNG.normal(size=(dimension, dimension)) + 1j * _RNG.normal(size=(dimension, dimension))
    return torch.tensor(entries - entries.T)


class TestPfaffianBlockDet:
    @pytest.mark.parametrize("block", _RANDOM_BLOCKS)
    def test_forward_square_of_pfaffian_matches_determinant(self, block):
        matrix = _block_antidiagonal(block)
        pfaffian = PfaffianBlockDet.apply(matrix)
        determinant = torch.linalg.det(matrix)
        torch.testing.assert_close(pfaffian**2, determinant, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_forward_supports_batched_input(self):
        matrix = _block_antidiagonal(_RNG.random((5, 3, 3)))
        pfaffian = PfaffianBlockDet.apply(matrix)
        assert pfaffian.shape == matrix.shape[:-2]
        torch.testing.assert_close(
            pfaffian**2, torch.linalg.det(matrix), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_forward_preserves_dtype(self, dtype):
        matrix = _block_antidiagonal(_RNG.random((3, 3))).to(dtype)
        assert PfaffianBlockDet.apply(matrix).dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    @pytest.mark.parametrize("block", _GRADCHECK_BLOCKS)
    def test_backward_passes_gradcheck(self, block, dtype):
        # The forward only accepts the block form, so gradcheck perturbs the free block B of
        # A = [[0, B], [-B^T, 0]]; the complex case covers the conjugate of the Wirtinger convention.
        free_block = torch.tensor(block, dtype=dtype)
        if dtype.is_complex:
            free_block = free_block + 1j * torch.tensor(_RNG.random(block.shape), dtype=dtype)
        free_block.requires_grad_(True)
        assert gradcheck(
            lambda free: PfaffianBlockDet.apply(_block_antidiagonal_torch(free)),
            (free_block,),
            eps=1e-6,
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )

    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    @pytest.mark.parametrize("exact_singular_grad", [True, False])
    def test_backward_matches_general_strategy_gradient(self, dtype, exact_singular_grad, monkeypatch):
        # The full-matrix gradient follows the general convention (1/2) pf(A) A^{-T}, so A.grad is the
        # same whichever strategy computed the Pfaffian.
        from torch_pfaffian.strategies.pfaffian_parlett_reid import PfaffianParlettReid

        monkeypatch.setattr(PfaffianBlockDet, "EXACT_SINGULAR_GRAD", exact_singular_grad)
        block = torch.tensor(_RNG.normal(size=(3, 4, 4)), dtype=dtype)
        if dtype.is_complex:
            block = block + 1j * torch.tensor(_RNG.normal(size=(3, 4, 4)), dtype=dtype)
        block_det_input = _block_antidiagonal_torch(block).requires_grad_(True)
        general_input = block_det_input.detach().clone().requires_grad_(True)
        PfaffianBlockDet.apply(block_det_input).real.sum().backward()
        PfaffianParlettReid.apply(general_input).real.sum().backward()
        torch.testing.assert_close(
            block_det_input.grad, general_input.grad, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("dimension", [6, 8, 12])
    def test_forward_rejects_general_skew_matrix(self, dimension):
        # Regression for issue #18: a general skew matrix used to return a wrong value silently.
        matrix = torch.tensor(_skew_general(dimension))
        with pytest.raises(ValueError, match="block"):
            PfaffianBlockDet.apply(matrix)

    def test_forward_rejects_odd_dimension(self):
        with pytest.raises(ValueError, match="block"):
            PfaffianBlockDet.apply(torch.zeros(5, 5, dtype=torch.float64))

    def test_forward_rejects_complex_general_skew_matrix(self):
        with pytest.raises(ValueError, match="block"):
            PfaffianBlockDet.apply(_random_complex_skew(8))

    def test_forward_check_disabled_by_class_flag(self, monkeypatch):
        # With the check disabled the forward trusts the caller and evaluates the block formula.
        monkeypatch.setattr(PfaffianBlockDet, "CHECK_BLOCK_FORM", False)
        matrix = torch.tensor(_skew_general(4))
        expected = -torch.linalg.det(matrix[:2, 2:])  # (-1)^(2*1/2) det(B)
        torch.testing.assert_close(PfaffianBlockDet.apply(matrix), expected)

    def test_forward_check_disabled_by_argument(self):
        matrix = torch.tensor(_skew_general(4))
        expected = -torch.linalg.det(matrix[:2, 2:])
        torch.testing.assert_close(PfaffianBlockDet.apply(matrix, False), expected)

    def test_forward_check_argument_overrides_disabled_class_flag(self, monkeypatch):
        monkeypatch.setattr(PfaffianBlockDet, "CHECK_BLOCK_FORM", False)
        with pytest.raises(ValueError, match="block"):
            PfaffianBlockDet.apply(torch.tensor(_skew_general(4)), True)

    @pytest.mark.parametrize("dimension", [4, 8, 10])
    def test_forward_matches_parlett_reid_on_complex_block_form(self, dimension):
        from torch_pfaffian.strategies.pfaffian_parlett_reid import PfaffianParlettReid

        size = dimension // 2
        block = torch.tensor(_RNG.normal(size=(size, size)) + 1j * _RNG.normal(size=(size, size)))
        matrix = _block_antidiagonal_torch(block)
        torch.testing.assert_close(
            PfaffianBlockDet.apply(matrix),
            PfaffianParlettReid.apply(matrix),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_is_block_form_accepts_block_form(self):
        assert PfaffianBlockDet.is_block_form(_block_antidiagonal(_RNG.random((5, 3, 3))))

    @pytest.mark.parametrize("dimension", [2, 4, 8])
    def test_is_block_form_rejects_general_skew_matrix(self, dimension):
        # A 2x2 skew matrix always has the block form; larger general ones do not.
        assert PfaffianBlockDet.is_block_form(torch.tensor(_skew_general(dimension))) == (dimension == 2)

    def test_is_block_form_checks_full_blocks_when_first_entry_is_zero(self):
        # The early exit only reads entry (0, 1); a nonzero elsewhere in a diagonal block must still fail.
        matrix = _block_antidiagonal(_RNG.random((3, 3))).clone()
        matrix[1, 2], matrix[2, 1] = 1.0, -1.0
        assert not PfaffianBlockDet.is_block_form(matrix)
        lower = _block_antidiagonal(_RNG.random((3, 3))).clone()
        lower[4, 5], lower[5, 4] = 1.0, -1.0
        assert not PfaffianBlockDet.is_block_form(lower)

    def test_is_block_form_rejects_batch_with_one_general_element(self):
        block_form = _block_antidiagonal(_RNG.random((3, 3)))
        general = torch.tensor(_skew_general(6))
        assert not PfaffianBlockDet.is_block_form(torch.stack([block_form, general]))

    @pytest.mark.parametrize("shape", [(5, 5), (4, 6), (6,), (0, 3, 3)])
    def test_is_block_form_rejects_bad_shapes(self, shape):
        assert not PfaffianBlockDet.is_block_form(torch.zeros(shape))

    @pytest.mark.parametrize("shape", [(0, 0), (0, 4, 4)])
    def test_is_block_form_accepts_empty_inputs(self, shape):
        assert PfaffianBlockDet.is_block_form(torch.zeros(shape))

    def test_backward_returns_none_when_input_does_not_require_grad(self):
        matrix = _block_antidiagonal(_RNG.random((4, 4)))
        pfaffian = PfaffianBlockDet.apply(matrix)

        class _Context:
            saved_tensors = (matrix, pfaffian)
            needs_input_grad = (False,)

        assert PfaffianBlockDet.backward(_Context(), torch.ones_like(pfaffian)) == (None, None)

    @pytest.mark.parametrize("size", _BLOCK_SIZES)
    def test_cofactor_matrix_matches_det_inverse_on_invertible(self, size):
        # On invertible blocks the cofactor matrix must equal the classical adjugate transpose
        # det(B) * (B^{-1})^T, the relation the singular gradient relies on.
        blocks = torch.tensor(_RNG.random((4, size, size))) + torch.eye(size)
        cofactor = PfaffianBlockDet._cofactor_matrix(blocks)
        expected = torch.linalg.det(blocks)[..., None, None] * torch.linalg.inv(blocks).transpose(-1, -2)
        torch.testing.assert_close(cofactor, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_backward_singular_block_is_finite_and_exact(self):
        # A singular upper-right block (pf=0) used to crash the inverse-based backward. The gradient is
        # c * adj(B)^T (finite, generally nonzero); verify it against an independent numpy cofactor.
        block = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 5.0]])  # rows 0,1 dependent -> det 0
        matrix = _block_antidiagonal(block).requires_grad_(True)
        pfaffian = PfaffianBlockDet.apply(matrix)
        assert pfaffian.item() == 0.0
        pfaffian.backward()
        assert torch.isfinite(matrix.grad).all()

        size = block.shape[0]
        constant = (-1) ** (size * (size - 1) // 2)
        cofactor = np.array(
            [
                [(-1) ** (i + j) * np.linalg.det(np.delete(np.delete(block, i, 0), j, 1)) for j in range(size)]
                for i in range(size)
            ]
        )
        expected_block = torch.tensor(0.5 * constant * cofactor)  # split between the off-diagonal blocks
        torch.testing.assert_close(
            matrix.grad[:size, size:], expected_block, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        torch.testing.assert_close(
            matrix.grad[size:, :size], -expected_block.T, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        assert matrix.grad[:size, :size].abs().max() == 0  # the diagonal blocks get no gradient
        assert matrix.grad[size:, size:].abs().max() == 0

    def test_backward_mixed_singular_invertible_batch_matches_inverse(self):
        # A batch mixing a singular block with an invertible one must not raise, and the invertible
        # element's gradient must match the inverse-based closed form.
        singular_block = torch.zeros(3, 3, dtype=torch.float64)
        invertible_block = torch.tensor(_RNG.random((3, 3))) + torch.eye(3)
        matrix = torch.stack(
            [_block_antidiagonal(singular_block), _block_antidiagonal(invertible_block)]
        ).requires_grad_(True)
        PfaffianBlockDet.apply(matrix).sum().backward()
        assert torch.isfinite(matrix.grad).all()
        constant = (-1) ** (3 * (3 - 1) // 2)
        expected = (
            0.5 * constant * torch.linalg.det(invertible_block) * torch.linalg.inv(invertible_block).transpose(-1, -2)
        )
        torch.testing.assert_close(
            matrix.grad[1, :3, 3:], expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        torch.testing.assert_close(
            matrix.grad[1, 3:, :3], -expected.T, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_exact_singular_grad_false_gives_zero_grad_at_singular(self, monkeypatch):
        # With EXACT_SINGULAR_GRAD disabled the backward is host-sync-free: a singular block (pf == 0)
        # gets an exactly zero gradient, while an invertible one matches the inverse-based closed form.
        monkeypatch.setattr(PfaffianBlockDet, "EXACT_SINGULAR_GRAD", False)
        rng = np.random.default_rng(TEST_SEED)
        singular_block = torch.zeros(3, 3, dtype=torch.float64)
        invertible_block = torch.tensor(rng.random((3, 3))) + torch.eye(3)
        matrix = torch.stack(
            [_block_antidiagonal(singular_block), _block_antidiagonal(invertible_block)]
        ).requires_grad_(True)
        PfaffianBlockDet.apply(matrix).sum().backward()
        assert torch.isfinite(matrix.grad).all()
        assert torch.all(matrix.grad[0] == 0)  # sync-free: singular block gets exactly zero gradient
        constant = (-1) ** (3 * (3 - 1) // 2)
        expected = (
            0.5 * constant * torch.linalg.det(invertible_block) * torch.linalg.inv(invertible_block).transpose(-1, -2)
        )
        torch.testing.assert_close(
            matrix.grad[1, :3, 3:], expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        torch.testing.assert_close(
            matrix.grad[1, 3:, :3], -expected.T, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    def test_exact_singular_grad_false_zeroes_round_off_singular_block(self, dtype, monkeypatch):
        # A rank-deficient block built by rotations has a round-off det (not 0), so testing pf == 0 alone gave
        # det * B^{-1}, round-off times a huge inverse. The sync-free mode now uses the LU pivot ratio too.
        monkeypatch.setattr(PfaffianBlockDet, "EXACT_SINGULAR_GRAD", False)
        generator = torch.Generator().manual_seed(TEST_SEED)
        size = 8
        left, _ = torch.linalg.qr(torch.randn(size, size, dtype=dtype, generator=generator))
        right, _ = torch.linalg.qr(torch.randn(size, size, dtype=dtype, generator=generator))
        singular_values = torch.linspace(0.5, 1.5, size, dtype=torch.float64)
        singular_values[0] = 0.0
        singular_block = (left * singular_values.to(dtype)) @ right.mH
        invertible_block = torch.eye(size, dtype=dtype) + 0.1 * torch.randn(
            size, size, dtype=dtype, generator=generator
        )
        matrix = torch.stack(
            [_block_antidiagonal_torch(singular_block), _block_antidiagonal_torch(invertible_block)]
        ).requires_grad_(True)
        pf = PfaffianBlockDet.apply(matrix)
        assert pf[0] != 0  # a round-off determinant, not an exact zero
        pf.real.sum().backward()
        assert matrix.grad is not None
        assert torch.all(matrix.grad[0] == 0)
        constant = (-1) ** (size * (size - 1) // 2)
        expected = (
            0.5
            * constant
            * (torch.linalg.det(invertible_block) * torch.linalg.inv(invertible_block).transpose(-1, -2)).conj()
        )
        torch.testing.assert_close(
            matrix.grad[1, :size, size:], expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_adjugate_override_delegates_to_parlett_reid(self):
        # PfaffianBlockDet.forward is block-only, so its adjugate must defer to the general strategy.
        from torch_pfaffian.strategies.pfaffian_parlett_reid import PfaffianParlettReid

        matrices = _block_antidiagonal(_RNG.random((3, 3, 3)))
        torch.testing.assert_close(
            PfaffianBlockDet._pfaffian_adjugate(matrices),
            PfaffianParlettReid._pfaffian_adjugate(matrices),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_backward_round_off_singular_block_next_to_zero_block_is_exact(self, dtype):
        # Regression: an exactly singular block whose forward determinant is round-off nonzero, batched
        # with a zero block, used to get a wrong gradient (or an inverse error) from the LU inverse.
        block = torch.tensor(
            [[2.0, 0.0, 12.0, -4.0], [4.0, -2.0, 16.0, -3.0], [-4.0, 8.0, -8.0, -4.0], [2.0, 2.0, 10.0, -4.0]],
            dtype=torch.float64,
        )
        blocks = torch.stack([torch.zeros(4, 4, dtype=torch.float64), block, block]).to(dtype)
        matrix = _block_antidiagonal_torch(blocks).requires_grad_(True)
        PfaffianBlockDet.apply(matrix).sum().backward()
        expected = 0.5 * PfaffianBlockDet._cofactor_matrix(block[None])[0]  # constant (-1)^(4*3/2) = 1
        tolerance = ATOL_MATRIX_COMPARISON if dtype == torch.float64 else ATOL_APPROX_COMPARISON
        for index in (1, 2):
            torch.testing.assert_close(
                matrix.grad[index, :4, 4:].to(torch.float64), expected, atol=tolerance, rtol=tolerance
            )

    def test_backward_lu_singular_block_next_to_zero_block_is_finite(self):
        # Regression: a singular 2x2 sub-block whose LU fails only on the contiguous copy used to raise.
        blocks = torch.eye(4, dtype=torch.float64).repeat(3, 1, 1)
        blocks[0, :2, :2] = 0.0
        blocks[1, :2, :2] = torch.tensor([[-3.0, 6.0], [-2.0, 4.0]], dtype=torch.float64)
        matrix = _block_antidiagonal_torch(blocks).requires_grad_(True)
        PfaffianBlockDet.apply(matrix).sum().backward()
        expected = 0.5 * PfaffianBlockDet._cofactor_matrix(blocks[1][None])[0]
        torch.testing.assert_close(
            matrix.grad[1, :4, 4:], expected, atol=ATOL_MATRIX_COMPARISON, rtol=ATOL_MATRIX_COMPARISON
        )

    def test_backward_empty_matrix_gradient_is_empty(self):
        matrix = torch.zeros(3, 0, 0, dtype=torch.float64, requires_grad=True)
        PfaffianBlockDet.apply(matrix).sum().backward()
        assert matrix.grad.shape == matrix.shape

    def test_is_block_form_rejects_zero_dimensional_tensor(self):
        assert not PfaffianBlockDet.is_block_form(torch.tensor(1.0))

    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    @pytest.mark.parametrize(
        ("size", "corank"), [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (5, 0), (5, 1), (5, 2), (8, 2)]
    )
    def test_svd_cofactor_matrix_matches_cofactor_determinants(self, size, corank, dtype):
        blocks = torch.tensor(_RNG.normal(size=(3, size, size)), dtype=dtype)
        if dtype.is_complex:
            blocks = blocks + 1j * torch.tensor(_RNG.normal(size=(3, size, size)), dtype=dtype)
        for row in range(corank):
            blocks[:, row] = 0.0  # zero rows give corank exactly `corank`
        torch.testing.assert_close(
            PfaffianBlockDet._svd_cofactor_matrix(blocks),
            PfaffianBlockDet._cofactor_matrix(blocks),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_exact_cofactor_matrix_falls_back_when_svd_fails(self, monkeypatch):
        def _no_convergence(*args, **kwargs):
            raise torch.linalg.LinAlgError("SVD did not converge")

        monkeypatch.setattr(torch.linalg, "svd", _no_convergence)
        blocks = torch.tensor(_RNG.normal(size=(2, 3, 3)))
        torch.testing.assert_close(
            PfaffianBlockDet._exact_cofactor_matrix(blocks, torch.ones(2, dtype=torch.bool)),
            PfaffianBlockDet._cofactor_matrix(blocks),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_backward_skips_exact_path_where_grad_output_is_zero(self, monkeypatch):
        def _fail(blocks, svd_mask):
            raise AssertionError("the exact cofactor path must not run for zero grad_output")

        monkeypatch.setattr(PfaffianBlockDet, "_exact_cofactor_matrix", staticmethod(_fail))
        blocks = torch.stack([torch.zeros(3, 3, dtype=torch.float64), torch.tensor(_RNG.random((3, 3))) + torch.eye(3)])
        matrix = _block_antidiagonal_torch(blocks).requires_grad_(True)
        (PfaffianBlockDet.apply(matrix) * torch.tensor([0.0, 1.0], dtype=torch.float64)).sum().backward()
        assert torch.all(matrix.grad[0] == 0)
        assert torch.isfinite(matrix.grad).all()

    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    def test_backward_large_singular_block_matches_cofactor_determinants(self, dtype):
        # A 20x20 rank-deficient block goes through the SVD cofactor; compare to the determinant oracle.
        size = 20
        block = torch.tensor(_RNG.normal(size=(size, size)), dtype=dtype)
        if dtype.is_complex:
            block = block + 1j * torch.tensor(_RNG.normal(size=(size, size)), dtype=dtype)
        block[0] = block[1]
        matrix = _block_antidiagonal_torch(block).requires_grad_(True)
        PfaffianBlockDet.apply(matrix).real.backward()
        constant = (-1) ** (size * (size - 1) // 2)
        expected = 0.5 * constant * PfaffianBlockDet._cofactor_matrix(block[None])[0].conj()
        scale = expected.abs().amax()  # cofactors of a 20x20 block are ~1e8; compare relative to that scale
        torch.testing.assert_close(
            matrix.grad[:size, size:] / scale,
            expected / scale,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_double_backward_at_repeated_singular_values_is_finite(self):
        # Regression: double backward through the SVD cofactor is unstable at repeated singular values,
        # so when a graph is built the exact path uses the (differentiable) cofactor determinants.
        generator = torch.Generator().manual_seed(TEST_SEED)
        left, _ = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64, generator=generator))
        right, _ = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64, generator=generator))
        block = left @ torch.diag(torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float64)) @ right.T
        matrix = _block_antidiagonal_torch(block).requires_grad_(True)
        pfaffian = PfaffianBlockDet.apply(matrix)
        (gradient,) = torch.autograd.grad(pfaffian, matrix, create_graph=True)
        (hessian_vector,) = torch.autograd.grad((gradient**2).sum(), matrix)
        assert torch.isfinite(hessian_vector).all()

    @pytest.mark.parametrize("size", [12, 16])
    def test_backward_small_ill_conditioned_singular_block_uses_cofactor_determinants(self, size, monkeypatch):
        # Below SVD_COFACTOR_MIN_BLOCK_SIZE the exact path uses the cofactor determinants, which stay exact on
        # a singular Wilkinson-type block (LU growth 2^(n-1)) where the SVD loses accuracy.
        def _no_svd(blocks):
            raise AssertionError("small blocks must not use the SVD cofactor")

        monkeypatch.setattr(PfaffianBlockDet, "_svd_cofactor_matrix", staticmethod(_no_svd))
        block = torch.eye(size, dtype=torch.float64) - torch.tril(torch.ones(size, size, dtype=torch.float64), -1)
        block[:, -1] = 1.0
        block[-1, -1] = 1.0 - 2.0 ** (size - 1)  # singular in exact arithmetic
        matrix = _block_antidiagonal_torch(block).requires_grad_(True)
        PfaffianBlockDet.apply(matrix, False).backward()
        constant = (-1) ** (size * (size - 1) // 2)
        expected = 0.5 * constant * PfaffianBlockDet._cofactor_matrix(block[None])[0]
        scale = expected.abs().amax()
        torch.testing.assert_close(
            matrix.grad[:size, size:] / scale,
            expected / scale,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_backward_large_singular_block_uses_svd_cofactor(self, monkeypatch):
        # From SVD_COFACTOR_MIN_BLOCK_SIZE on, rank-deficient blocks take the O(n^3) SVD cofactor.
        size = PfaffianBlockDet.SVD_COFACTOR_MIN_BLOCK_SIZE
        calls = []
        original = PfaffianBlockDet._svd_cofactor_matrix

        def _recording(blocks):
            calls.append(blocks.shape)
            return original(blocks)

        monkeypatch.setattr(PfaffianBlockDet, "_svd_cofactor_matrix", staticmethod(_recording))
        block = torch.tensor(_RNG.normal(size=(size, size)))
        block[0] = block[1]
        matrix = _block_antidiagonal_torch(block).requires_grad_(True)
        PfaffianBlockDet.apply(matrix).backward()
        assert calls == [(1, size, size)]
        constant = (-1) ** (size * (size - 1) // 2)
        expected = 0.5 * constant * PfaffianBlockDet._cofactor_matrix(block[None])[0]
        torch.testing.assert_close(
            matrix.grad[:size, size:], expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_svd_cofactor_min_block_size_constant(self):
        assert PfaffianBlockDet.SVD_COFACTOR_MIN_BLOCK_SIZE == 17
