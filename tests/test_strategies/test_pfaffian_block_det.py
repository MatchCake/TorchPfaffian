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

    @pytest.mark.parametrize("block", _GRADCHECK_BLOCKS)
    def test_backward_passes_gradcheck(self, block):
        matrix = _block_antidiagonal(block).to(torch.float64).requires_grad_(True)
        assert gradcheck(
            PfaffianBlockDet.apply, (matrix,), eps=1e-6, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON
        )

    def test_backward_returns_none_when_input_does_not_require_grad(self):
        matrix = _block_antidiagonal(_RNG.random((4, 4)))
        pfaffian = PfaffianBlockDet.apply(matrix)

        class _Context:
            saved_tensors = (matrix, pfaffian)
            needs_input_grad = (False,)

        assert PfaffianBlockDet.backward(_Context(), torch.ones_like(pfaffian)) is None
