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
from torch_pfaffian.strategies.pfaffian_det import PfaffianDet
from torch_pfaffian.strategies.pfaffian_fdbpf import PfaffianFDBPf

_RNG = np.random.default_rng(TEST_SEED)


def _skew(matrix: np.ndarray) -> np.ndarray:
    return matrix - np.einsum("...ij->...ji", matrix)


def _well_conditioned_skew_matrices(shapes: list[tuple[int, ...]], count: int, min_abs_det: float = 0.1):
    matrices = []
    for shape in shapes:
        kept = 0
        while kept < count:
            candidate = _skew(_RNG.random(shape))
            if np.all(np.abs(np.linalg.det(candidate)) > min_abs_det):
                matrices.append(candidate)
                kept += 1
    return matrices


_FORWARD_MATRICES = [
    _skew(_RNG.random(shape)) for shape in [(8, 8), (16, 6, 6)] for _ in range(N_RANDOM_TESTS_PER_CASE)
]
_GRADCHECK_MATRICES = _well_conditioned_skew_matrices([(4, 4), (6, 6), (8, 4, 4)], N_RANDOM_TESTS_PER_CASE)


class TestPfaffianDet:
    @pytest.mark.parametrize("skew_matrix", _FORWARD_MATRICES)
    def test_forward_square_of_pfaffian_matches_determinant(self, skew_matrix):
        skew_tensor = torch.tensor(skew_matrix)
        pfaffian = PfaffianDet.apply(skew_tensor)
        determinant = torch.linalg.det(skew_tensor)
        torch.testing.assert_close(pfaffian**2, determinant, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize("skew_matrix", _FORWARD_MATRICES)
    def test_forward_matches_fdbpf(self, skew_matrix):
        skew_tensor = torch.tensor(skew_matrix)
        torch.testing.assert_close(
            PfaffianDet.apply(skew_tensor),
            PfaffianFDBPf.apply(skew_tensor),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("skew_matrix", _GRADCHECK_MATRICES)
    def test_backward_passes_gradcheck(self, skew_matrix):
        skew_tensor = torch.tensor(skew_matrix, requires_grad=True)
        assert gradcheck(
            PfaffianDet.apply, (skew_tensor,), eps=1e-6, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_forward_preserves_dtype(self, dtype):
        matrix = torch.tensor(_skew(_RNG.random((6, 6))), dtype=dtype)
        assert PfaffianDet.apply(matrix).dtype == dtype

    def test_forward_supports_batched_input(self):
        batched_matrix = torch.tensor(_skew(_RNG.random((5, 6, 6))))
        pfaffian = PfaffianDet.apply(batched_matrix)
        assert pfaffian.shape == batched_matrix.shape[:-2]
        torch.testing.assert_close(
            pfaffian**2, torch.linalg.det(batched_matrix), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
