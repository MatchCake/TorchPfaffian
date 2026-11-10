import warnings

import pytest
import torch

import torch_pfaffian
from tests.configs import (
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    ATOL_SINGLE_PRECISION_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    RTOL_SINGLE_PRECISION_COMPARISON,
    TEST_SEED,
)
from tests.reference import pfaffian_combinatorial
from torch_pfaffian import pfaffian
from torch_pfaffian.low_rank import LowRankPfaffian

DIMENSION = 10
BATCH = 4


class TestLowRankPfaffian:
    @staticmethod
    def _skew(*shape: int, dtype: torch.dtype = torch.float64, seed: int = TEST_SEED) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        real_dtype = torch.float64 if dtype in (torch.float64, torch.complex128) else torch.float32
        entries = torch.randn(*shape, generator=generator, dtype=real_dtype)
        if dtype.is_complex:
            entries = torch.complex(entries, torch.randn(*shape, generator=generator, dtype=real_dtype))
        entries = entries.to(dtype)
        return entries - entries.mT

    @staticmethod
    def _random(*shape: int, dtype: torch.dtype = torch.float64, seed: int = TEST_SEED + 1) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        real_dtype = torch.float64 if dtype in (torch.float64, torch.complex128) else torch.float32
        entries = torch.randn(*shape, generator=generator, dtype=real_dtype)
        if dtype.is_complex:
            entries = torch.complex(entries, torch.randn(*shape, generator=generator, dtype=real_dtype))
        return entries.to(dtype)

    @staticmethod
    def _apply_update(matrix: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        identity = LowRankPfaffian.skew_identity(update.shape[-1] // 2, dtype=update.dtype)
        return matrix - update @ identity @ update.mT

    @staticmethod
    def _apply_replace(matrix: torch.Tensor, index: torch.Tensor, change: torch.Tensor) -> torch.Tensor:
        batch = torch.arange(matrix.shape[0])
        change = change.clone()
        change[batch, index] = 0
        updated = matrix.clone()
        updated[batch, index, :] += change
        updated[batch, :, index] -= change
        return updated

    @staticmethod
    def _replace_update(index: torch.Tensor, change: torch.Tensor) -> torch.Tensor:
        change = change.clone()
        change[torch.arange(change.shape[0]), index] = 0
        one_hot = torch.nn.functional.one_hot(index, change.shape[-1]).to(change.dtype)
        return torch.stack([change, one_hot], dim=-1)

    @pytest.mark.parametrize("rank", [1, 2, 3, 4])
    def test_skew_identity_pfaffian_matches_combinatorial(self, rank):
        identity = LowRankPfaffian.skew_identity(rank, dtype=torch.float64)
        assert LowRankPfaffian.skew_identity_pfaffian(rank) == pfaffian_combinatorial(identity.numpy())
        torch.testing.assert_close(identity.mT, -identity, atol=0, rtol=0)
        torch.testing.assert_close(identity @ identity, -torch.eye(2 * rank, dtype=torch.float64), atol=0, rtol=0)

    def test_skew_identity_rejects_zero_rank(self):
        with pytest.raises(ValueError, match="rank"):
            LowRankPfaffian.skew_identity(0)

    @pytest.mark.parametrize("rank", [1, 2, 3, 4, 6])
    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    def test_ratio_and_inverse_match_direct_computation(self, rank, dtype):
        matrix = self._skew(BATCH, DIMENSION, DIMENSION, dtype=dtype)
        update = self._random(BATCH, DIMENSION, 2 * rank, dtype=dtype)
        updated = self._apply_update(matrix, update)
        ratio, new_inverse = LowRankPfaffian.ratio_and_inverse(torch.linalg.inv(matrix), update)
        torch.testing.assert_close(
            ratio, pfaffian(updated) / pfaffian(matrix), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )
        torch.testing.assert_close(
            new_inverse, torch.linalg.inv(updated), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        assert ratio.dtype == dtype and new_inverse.dtype == dtype

    def test_ratio_without_inverse_matches_ratio_with_inverse(self):
        matrix = self._skew(BATCH, DIMENSION, DIMENSION)
        update = self._random(BATCH, DIMENSION, 2)
        inverse = torch.linalg.inv(matrix)
        ratio = LowRankPfaffian.ratio(inverse, update)
        torch.testing.assert_close(ratio, LowRankPfaffian.ratio_and_inverse(inverse, update)[0], atol=0, rtol=0)

    def test_unbatched_inputs(self):
        matrix = self._skew(DIMENSION, DIMENSION)
        update = self._random(DIMENSION, 2)
        ratio, new_inverse = LowRankPfaffian.ratio_and_inverse(torch.linalg.inv(matrix), update)
        assert ratio.shape == () and new_inverse.shape == (DIMENSION, DIMENSION)
        updated = self._apply_update(matrix, update)
        torch.testing.assert_close(
            ratio, pfaffian(updated) / pfaffian(matrix), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )
        torch.testing.assert_close(
            new_inverse, torch.linalg.inv(updated), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_single_matrix_and_broadcast_batch(self):
        matrix = self._skew(DIMENSION, DIMENSION)
        updates = self._random(3, 2, DIMENSION, 2)  # several updates of one matrix
        ratio = LowRankPfaffian.ratio(torch.linalg.inv(matrix), updates)
        assert ratio.shape == (3, 2)
        expected = pfaffian(self._apply_update(matrix.expand(3, 2, DIMENSION, DIMENSION), updates)) / pfaffian(matrix)
        torch.testing.assert_close(ratio, expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    def test_replace_matches_direct_computation(self, dtype):
        matrix = self._skew(BATCH, DIMENSION, DIMENSION, dtype=dtype)
        index = torch.tensor([0, 3, 9, 3])
        change = self._random(BATCH, DIMENSION, dtype=dtype)
        updated = self._apply_replace(matrix, index, change)
        ratio, new_inverse = LowRankPfaffian.ratio_replace_and_inverse(torch.linalg.inv(matrix), index, change)
        torch.testing.assert_close(
            ratio, pfaffian(updated) / pfaffian(matrix), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )
        torch.testing.assert_close(
            new_inverse, torch.linalg.inv(updated), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_replace_ignores_the_diagonal_entry_of_the_change(self):
        matrix = self._skew(BATCH, DIMENSION, DIMENSION)
        inverse = torch.linalg.inv(matrix)
        index = torch.tensor([1, 2, 3, 4])
        change = self._random(BATCH, DIMENSION)
        shifted = change.clone()
        shifted[torch.arange(BATCH), index] += 100.0
        torch.testing.assert_close(
            LowRankPfaffian.ratio_replace(inverse, index, shifted),
            LowRankPfaffian.ratio_replace(inverse, index, change),
        )

    def test_replace_agrees_with_the_rank_two_update(self):
        matrix = self._skew(BATCH, DIMENSION, DIMENSION)
        inverse = torch.linalg.inv(matrix)
        index = torch.tensor([0, 1, 2, 3])
        change = self._random(BATCH, DIMENSION)
        change[torch.arange(BATCH), index] = 0
        one_hot = torch.nn.functional.one_hot(index, DIMENSION).to(change.dtype)
        update = torch.stack([change, one_hot], dim=-1)
        expected_ratio, expected_inverse = LowRankPfaffian.ratio_and_inverse(inverse, update)
        ratio, new_inverse = LowRankPfaffian.ratio_replace_and_inverse(inverse, index, change)
        torch.testing.assert_close(ratio, expected_ratio, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)
        torch.testing.assert_close(
            new_inverse, expected_inverse, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_replace_broadcasts_one_matrix_over_several_moves(self):
        matrix = self._skew(DIMENSION, DIMENSION)
        index = torch.tensor([0, 4, 7])
        change = self._random(DIMENSION)  # one change, three rows, one matrix
        ratio, new_inverse = LowRankPfaffian.ratio_replace_and_inverse(torch.linalg.inv(matrix), index, change)
        assert ratio.shape == (3,) and new_inverse.shape == (3, DIMENSION, DIMENSION)
        updated = self._apply_replace(matrix.expand(3, DIMENSION, DIMENSION), index, change.expand(3, DIMENSION))
        torch.testing.assert_close(
            ratio, pfaffian(updated) / pfaffian(matrix), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )
        torch.testing.assert_close(
            new_inverse, torch.linalg.inv(updated), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_replace_with_integer_index_broadcasts(self):
        matrix = self._skew(BATCH, DIMENSION, DIMENSION)
        change = self._random(BATCH, DIMENSION)
        ratio = LowRankPfaffian.ratio_replace(torch.linalg.inv(matrix), 2, change)
        expected = pfaffian(self._apply_replace(matrix, torch.full((BATCH,), 2), change)) / pfaffian(matrix)
        torch.testing.assert_close(ratio, expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_replacing_a_row_by_zero_gives_a_zero_ratio(self):
        matrix = self._skew(BATCH, DIMENSION, DIMENSION)
        index = torch.tensor([0, 1, 2, 3])
        change = -matrix[torch.arange(BATCH), index, :]  # the new row is zero, so the new matrix is singular
        ratio = LowRankPfaffian.ratio_replace(torch.linalg.inv(matrix), index, change)
        torch.testing.assert_close(ratio, torch.zeros_like(ratio), atol=ATOL_SCALAR_COMPARISON, rtol=0)
        dense = LowRankPfaffian.ratio(torch.linalg.inv(matrix), self._replace_update(index, change))
        torch.testing.assert_close(dense, torch.zeros_like(dense), atol=ATOL_SCALAR_COMPARISON, rtol=0)

    @staticmethod
    def _skew_part(matrix: torch.Tensor) -> torch.Tensor:
        # gradcheck perturbs every entry independently; the inverse of a skew-symmetric matrix only moves along
        # skew-symmetric directions, so the checks go through this parametrization.
        return 0.5 * (matrix - matrix.mT)

    def test_gradcheck(self):
        matrix = self._skew(2, 6, 6)
        inverse = torch.linalg.inv(matrix).requires_grad_(True)
        update = self._random(2, 6, 4).requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda a, u: LowRankPfaffian.ratio_and_inverse(self._skew_part(a), u), (inverse, update)
        )
        change = self._random(2, 6).requires_grad_(True)
        index = torch.tensor([1, 4])
        assert torch.autograd.gradcheck(
            lambda a, c: LowRankPfaffian.ratio_replace_and_inverse(self._skew_part(a), index, c), (inverse, change)
        )

    def test_gradcheck_complex(self):
        matrix = self._skew(2, 6, 6, dtype=torch.complex128)
        inverse = torch.linalg.inv(matrix).requires_grad_(True)
        update = self._random(2, 6, 2, dtype=torch.complex128).requires_grad_(True)
        assert torch.autograd.gradcheck(lambda a, u: LowRankPfaffian.ratio(self._skew_part(a), u), (inverse, update))

    def test_gradient_of_a_leaf_inverse_is_skew_symmetric(self):
        # Option B of the gradient convention: same as pfaffian(), the skew part of the plain gradient.
        matrix = self._skew(BATCH, DIMENSION, DIMENSION)
        update = self._random(BATCH, DIMENSION, 2)
        index = torch.tensor([0, 2, 4, 6])
        change = self._random(BATCH, DIMENSION)
        border, corner = self._random(BATCH, DIMENSION, 2), self._skew(BATCH, 2, 2)
        cases = [
            lambda a: LowRankPfaffian.ratio(a, update),
            lambda a: LowRankPfaffian.ratio_and_inverse(a, update)[1].sum(),
            lambda a: LowRankPfaffian.ratio_replace(a, index, change),
            lambda a: LowRankPfaffian.ratio_replace_and_inverse(a, index, change)[1].sum(),
            lambda a: LowRankPfaffian.ratio_extend(a, border, corner),
            lambda a: LowRankPfaffian.ratio_extend_and_inverse(a, border, corner)[1].sum(),
        ]
        for function in cases:
            leaf = torch.linalg.inv(matrix).requires_grad_(True)
            function(leaf).sum().backward()
            torch.testing.assert_close(leaf.grad, -leaf.grad.mT, atol=0, rtol=0)
            reference = torch.linalg.inv(matrix).requires_grad_(True)
            function(self._skew_part(reference)).sum().backward()
            torch.testing.assert_close(
                leaf.grad, reference.grad, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
            )

    def test_gradient_of_the_ratio_matches_that_of_the_pfaffian_quotient(self):
        matrix = self._skew(BATCH, DIMENSION, DIMENSION)
        update = self._random(BATCH, DIMENSION, 2).requires_grad_(True)
        LowRankPfaffian.ratio(torch.linalg.inv(matrix), update).sum().backward()
        reference = update.detach().clone().requires_grad_(True)
        (pfaffian(self._apply_update(matrix, reference)) / pfaffian(matrix)).sum().backward()
        torch.testing.assert_close(
            update.grad, reference.grad, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_ratio_and_r_inverse_agree_with_ratio(self):
        matrix = self._skew(BATCH, DIMENSION, DIMENSION)
        inverse = torch.linalg.inv(matrix)
        update = self._random(BATCH, DIMENSION, 4)
        projected = LowRankPfaffian.project(inverse, update)
        expected_ratio = LowRankPfaffian.ratio(inverse, update)
        ratio, r_inverse = LowRankPfaffian.ratio_and_r_inverse(update, projected)
        torch.testing.assert_close(ratio, expected_ratio, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)
        identity = LowRankPfaffian.skew_identity(2, dtype=torch.float64)
        r_matrix = identity + update.mT @ projected
        torch.testing.assert_close(
            r_inverse @ r_matrix,
            torch.eye(4, dtype=torch.float64).expand(BATCH, 4, 4),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=0,
        )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_is_upcast_with_a_warning(self, dtype):
        matrix = self._skew(BATCH, 6, 6, dtype=torch.float32) / 4
        update = self._random(BATCH, 6, 2, dtype=torch.float32) / 4
        inverse = torch.linalg.inv(matrix)
        expected = LowRankPfaffian.ratio(inverse.to(dtype).float(), update.to(dtype).float())
        with pytest.warns(UserWarning, match="computes"):
            ratio, new_inverse = LowRankPfaffian.ratio_and_inverse(inverse.to(dtype), update.to(dtype))
        assert ratio.dtype == dtype and new_inverse.dtype == dtype
        torch.testing.assert_close(ratio.float(), expected.to(dtype).float())
        with pytest.warns(UserWarning, match="computes"):
            replaced = LowRankPfaffian.ratio_replace(inverse.to(dtype), 1, update[..., 0].to(dtype))
        assert replaced.dtype == dtype
        with pytest.warns(UserWarning, match="computes"):
            extended = LowRankPfaffian.ratio_extend(inverse.to(dtype), update.to(dtype), matrix[..., :2, :2].to(dtype))
        assert extended.dtype == dtype

    def test_float32(self):
        matrix = self._skew(BATCH, DIMENSION, DIMENSION, dtype=torch.float32)
        update = self._random(BATCH, DIMENSION, 2, dtype=torch.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ratio = LowRankPfaffian.ratio(torch.linalg.inv(matrix), update)
        expected = pfaffian(self._apply_update(matrix.double(), update.double())) / pfaffian(matrix.double())
        torch.testing.assert_close(
            ratio.double(), expected, atol=ATOL_SINGLE_PRECISION_COMPARISON, rtol=RTOL_SINGLE_PRECISION_COMPARISON
        )

    def test_invalid_inputs_raise(self):
        inverse = torch.linalg.inv(self._skew(DIMENSION, DIMENSION))
        with pytest.raises(ValueError, match="inverse"):
            LowRankPfaffian.ratio(inverse[:, :4], self._random(DIMENSION, 2))
        with pytest.raises(ValueError, match="update"):
            LowRankPfaffian.ratio(inverse, self._random(DIMENSION + 1, 2))
        with pytest.raises(ValueError, match="even"):
            LowRankPfaffian.ratio(inverse, self._random(DIMENSION, 3))
        with pytest.raises(TypeError, match="dtype"):
            LowRankPfaffian.ratio(inverse, self._random(DIMENSION, 2, dtype=torch.float32))
        with pytest.raises(TypeError, match="supports"):
            LowRankPfaffian.ratio(inverse.long(), self._random(DIMENSION, 2).long())
        with pytest.raises(ValueError, match="change"):
            LowRankPfaffian.ratio_replace(inverse, 0, self._random(DIMENSION + 1))
        with pytest.raises(TypeError, match="integer"):
            LowRankPfaffian.ratio_replace(inverse, torch.tensor(0.0), self._random(DIMENSION))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    def test_cuda_matches_cpu(self):  # pragma: no cover - needs a GPU
        matrix = self._skew(BATCH, DIMENSION, DIMENSION)
        update = self._random(BATCH, DIMENSION, 4)
        inverse = torch.linalg.inv(matrix)
        expected = LowRankPfaffian.ratio_and_inverse(inverse, update)
        result = LowRankPfaffian.ratio_and_inverse(inverse.cuda(), update.cuda())
        assert result[0].device.type == "cuda"
        for got, want in zip(result, expected):
            torch.testing.assert_close(got.cpu(), want, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)
        index = torch.tensor([0, 1, 2, 3])
        replaced = LowRankPfaffian.ratio_replace(inverse.cuda(), index.cuda(), update[..., 0].cuda())
        torch.testing.assert_close(
            replaced.cpu(),
            LowRankPfaffian.ratio_replace(inverse, index, update[..., 0]),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=0,
        )

    def test_package_shortcuts(self):
        assert torch_pfaffian.pfaffian_ratio == LowRankPfaffian.ratio
        assert torch_pfaffian.pfaffian_ratio_and_inverse == LowRankPfaffian.ratio_and_inverse
        assert torch_pfaffian.pfaffian_ratio_replace == LowRankPfaffian.ratio_replace
        assert torch_pfaffian.pfaffian_ratio_replace_and_inverse == LowRankPfaffian.ratio_replace_and_inverse
        assert torch_pfaffian.pfaffian_extension_ratio == LowRankPfaffian.ratio_extend
        assert torch_pfaffian.pfaffian_extension_ratio_and_inverse == LowRankPfaffian.ratio_extend_and_inverse
        assert torch_pfaffian.skew_identity == LowRankPfaffian.skew_identity

    @pytest.mark.parametrize("width", [2, 4])
    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    def test_extend_matches_direct_computation(self, width, dtype):
        full = self._skew(BATCH, DIMENSION + width, DIMENSION + width, dtype=dtype)
        leading = full[..., :DIMENSION, :DIMENSION]
        border, corner = full[..., :DIMENSION, DIMENSION:], full[..., DIMENSION:, DIMENSION:]
        inverse = torch.linalg.inv(leading)
        expected = pfaffian(full) / pfaffian(leading)
        ratio = LowRankPfaffian.ratio_extend(inverse, border, corner)
        torch.testing.assert_close(ratio, expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)
        ratio, new_inverse = LowRankPfaffian.ratio_extend_and_inverse(inverse, border, corner)
        torch.testing.assert_close(ratio, expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)
        torch.testing.assert_close(
            new_inverse, torch.linalg.inv(full), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_extend_from_the_empty_matrix_is_the_pfaffian_of_the_corner(self):
        corner = self._skew(BATCH, 4, 4)
        empty = torch.zeros(BATCH, 0, 0, dtype=torch.float64)
        border = torch.zeros(BATCH, 0, 4, dtype=torch.float64)
        ratio, new_inverse = LowRankPfaffian.ratio_extend_and_inverse(empty, border, corner)
        torch.testing.assert_close(ratio, pfaffian(corner), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)
        torch.testing.assert_close(
            new_inverse, torch.linalg.inv(corner), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        torch.testing.assert_close(LowRankPfaffian.ratio_extend(empty, border, corner), ratio, atol=0, rtol=0)

    def test_extend_with_candidate_corners_in_a_leading_dimension(self):
        full = self._skew(BATCH, DIMENSION + 2, DIMENSION + 2)
        inverse = torch.linalg.inv(full[..., :DIMENSION, :DIMENSION])
        border = full[..., :DIMENSION, DIMENSION:]
        corners = torch.stack([full[..., DIMENSION:, DIMENSION:], 2 * full[..., DIMENSION:, DIMENSION:]])
        ratios = LowRankPfaffian.ratio_extend(inverse, border, corners)  # (2, BATCH)
        assert ratios.shape == (2, BATCH)
        for candidate in range(2):
            torch.testing.assert_close(
                ratios[candidate], LowRankPfaffian.ratio_extend(inverse, border, corners[candidate]), atol=0, rtol=0
            )
        ratio, new_inverse = LowRankPfaffian.ratio_extend_and_inverse(inverse, border, corners)
        assert new_inverse.shape == (2, BATCH, DIMENSION + 2, DIMENSION + 2)

    def test_extend_gradcheck(self):
        full = self._skew(2, 8, 8)
        inverse = torch.linalg.inv(full[..., :6, :6]).requires_grad_(True)
        border = full[..., :6, 6:].clone().requires_grad_(True)
        corner = full[..., 6:, 6:].clone().requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda a, b, c: LowRankPfaffian.ratio_extend_and_inverse(self._skew_part(a), b, c),
            (inverse, border, corner),
        )
        wide = self._skew(2, 10, 10)
        assert torch.autograd.gradcheck(
            lambda a, b, c: LowRankPfaffian.ratio_extend(self._skew_part(a), b, c),
            (
                torch.linalg.inv(wide[..., :6, :6]).requires_grad_(True),
                wide[..., :6, 6:].clone().requires_grad_(True),
                wide[..., 6:, 6:].clone().requires_grad_(True),
            ),
        )

    def test_extend_invalid_inputs_raise(self):
        inverse = torch.linalg.inv(self._skew(DIMENSION, DIMENSION))
        corner = self._skew(2, 2)
        with pytest.raises(ValueError, match="border"):
            LowRankPfaffian.ratio_extend(inverse, self._random(DIMENSION + 1, 2), corner)
        with pytest.raises(ValueError, match="even"):
            LowRankPfaffian.ratio_extend(inverse, self._random(DIMENSION, 3), self._skew(3, 3))
        with pytest.raises(ValueError, match="corner"):
            LowRankPfaffian.ratio_extend(inverse, self._random(DIMENSION, 2), self._skew(4, 4))
        with pytest.raises(TypeError, match="dtype"):
            LowRankPfaffian.ratio_extend(inverse, self._random(DIMENSION, 2), corner.float())

    def test_project_and_add_product(self):
        inverse = torch.linalg.inv(self._skew(BATCH, DIMENSION, DIMENSION))
        vectors = self._random(BATCH, DIMENSION, 3)
        torch.testing.assert_close(
            LowRankPfaffian.project(inverse, vectors), inverse @ vectors, atol=ATOL_MATRIX_COMPARISON, rtol=0
        )
        left, right = self._random(BATCH, DIMENSION, 2), self._random(BATCH, 2, DIMENSION)
        torch.testing.assert_close(LowRankPfaffian.add_product(inverse, left, right), inverse + left @ right)
        column_major = inverse.mT.contiguous().mT  # the layout torch.linalg.inv returns
        result = LowRankPfaffian.add_product(column_major, left, right)
        torch.testing.assert_close(result, inverse + left @ right)
        assert result.mT.is_contiguous()
        broadcast_left = self._random(2, BATCH, DIMENSION, 2)  # batch shapes differ: no fused path
        torch.testing.assert_close(
            LowRankPfaffian.add_product(inverse, broadcast_left, right), inverse + broadcast_left @ right
        )

    def test_rank_one_r_inverse_matches_the_general_inverse(self):
        matrix = self._skew(BATCH, DIMENSION, DIMENSION)
        update = self._random(BATCH, DIMENSION, 2)
        projected = torch.linalg.inv(matrix) @ update
        _, r_inverse = LowRankPfaffian.ratio_and_r_inverse(update, projected)
        r_matrix = LowRankPfaffian.skew_identity(1, dtype=torch.float64) + update.mT @ projected
        torch.testing.assert_close(
            r_inverse,
            torch.linalg.inv(0.5 * (r_matrix - r_matrix.mT)),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )
