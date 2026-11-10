import pytest
import torch

from tests.configs import (
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    ATOL_SINGLE_PRECISION_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    RTOL_SINGLE_PRECISION_COMPARISON,
    TEST_SEED,
)
from torch_pfaffian import pfaffian
from torch_pfaffian.low_rank import LowRankPfaffian, PfaffianUpdater

DIMENSION = 12
BATCH = 3
STEPS = 11


class TestPfaffianUpdater:
    @staticmethod
    def _skew(dtype: torch.dtype = torch.float64, seed: int = TEST_SEED) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        entries = torch.randn(BATCH, DIMENSION, DIMENSION, generator=generator, dtype=torch.float64)
        if dtype.is_complex:
            entries = torch.complex(entries, torch.randn(entries.shape, generator=generator, dtype=torch.float64))
        entries = entries.to(dtype)
        return entries - entries.mT

    @staticmethod
    def _updates(rank: int, dtype: torch.dtype = torch.float64, seed: int = TEST_SEED + 1) -> list[torch.Tensor]:
        # Small updates keep the matrices well conditioned over many steps.
        generator = torch.Generator().manual_seed(seed)
        updates = []
        for _ in range(STEPS):
            update = 0.3 * torch.randn(BATCH, DIMENSION, 2 * rank, generator=generator, dtype=torch.float64)
            if dtype.is_complex:
                imaginary = torch.randn(update.shape, generator=generator, dtype=torch.float64)
                update = torch.complex(update, 0.3 * imaginary)
            updates.append(update.to(dtype))
        return updates

    @staticmethod
    def _apply(matrix: torch.Tensor, update: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        identity = LowRankPfaffian.skew_identity(update.shape[-1] // 2, dtype=update.dtype)
        updated = matrix - update @ identity @ update.mT
        return updated if mask is None else torch.where(mask[..., None, None], updated, matrix)

    @pytest.fixture(autouse=True)
    def _delayed_updates_only(self, monkeypatch):
        # These tests are about the delayed updates; small CPU inverses otherwise take native immediate updates.
        monkeypatch.setattr(PfaffianUpdater, "IMMEDIATE_MAX_NUMEL", -1)

    @pytest.mark.parametrize("max_delay", [1, 3, 100])
    @pytest.mark.parametrize("rank", [1, 2])
    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    def test_delayed_updates_match_direct_computation(self, max_delay, rank, dtype):
        matrix = self._skew(dtype)
        updater = PfaffianUpdater(matrix, max_delay=max_delay)
        for step, update in enumerate(self._updates(rank, dtype)):
            proposed = updater.ratio(update)
            accepted = updater.accept(update)
            torch.testing.assert_close(accepted, proposed, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)
            updated = self._apply(matrix, update)
            torch.testing.assert_close(
                accepted, pfaffian(updated) / pfaffian(matrix), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
            )
            matrix = updated
            assert updater.delay == (step + 1) % max_delay
        torch.testing.assert_close(
            updater.inverse, torch.linalg.inv(matrix), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        torch.testing.assert_close(
            updater.pfaffian, pfaffian(matrix), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_delayed_and_immediate_updates_agree(self):
        matrix = self._skew()
        delayed = PfaffianUpdater(matrix, max_delay=4)
        inverse = torch.linalg.inv(matrix)
        for update in self._updates(2):
            immediate_ratio, inverse = LowRankPfaffian.ratio_and_inverse(inverse, update)
            torch.testing.assert_close(
                delayed.accept(update), immediate_ratio, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
            )
        torch.testing.assert_close(delayed.inverse, inverse, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_mask_leaves_rejected_elements_unchanged(self):
        matrix = self._skew()
        updater = PfaffianUpdater(matrix, max_delay=5)
        for step, update in enumerate(self._updates(1)):
            mask = torch.tensor([True, step % 2 == 0, False])
            ratio = updater.accept(update, mask)
            torch.testing.assert_close(ratio[2], torch.ones((), dtype=ratio.dtype), atol=0, rtol=0)
            matrix = self._apply(matrix, update, mask)
        torch.testing.assert_close(
            updater.inverse, torch.linalg.inv(matrix), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        torch.testing.assert_close(
            updater.pfaffian, pfaffian(matrix), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_row_replacement(self):
        matrix = self._skew()
        updater = PfaffianUpdater(matrix, max_delay=3)
        generator = torch.Generator().manual_seed(TEST_SEED + 2)
        batch = torch.arange(BATCH)
        for step in range(STEPS):
            index = torch.randint(0, DIMENSION, (BATCH,), generator=generator)
            change = 0.3 * torch.randn(BATCH, DIMENSION, generator=generator, dtype=torch.float64)
            mask = torch.tensor([True, True, step % 3 != 0])
            proposed = updater.ratio_replace(index, change)
            ratio = updater.accept_replace(index, change, mask)
            torch.testing.assert_close(
                ratio[:2], proposed[:2], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
            )
            change[batch, index] = 0
            updated = matrix.clone()
            updated[batch, index, :] += change
            updated[batch, :, index] -= change
            matrix = torch.where(mask[:, None, None], updated, matrix)
        torch.testing.assert_close(
            updater.inverse, torch.linalg.inv(matrix), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        torch.testing.assert_close(
            updater.pfaffian, pfaffian(matrix), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )

    def test_row_replacement_without_mask_and_with_a_shared_change(self):
        matrix = self._skew()
        updater = PfaffianUpdater(matrix, max_delay=2)
        index = torch.tensor([1, 5, 9])
        change = 0.3 * self._updates(1)[0][0, :, 0]  # one change of shape (n,), broadcast over the 3 rows
        ratio = updater.accept_replace(index, change)
        rows = torch.arange(BATCH)
        expanded = change.expand(BATCH, DIMENSION).clone()
        expanded[rows, index] = 0
        updated = matrix.clone()
        updated[rows, index, :] += expanded
        updated[rows, :, index] -= expanded
        torch.testing.assert_close(
            ratio, pfaffian(updated) / pfaffian(matrix), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )
        torch.testing.assert_close(
            updater.inverse, torch.linalg.inv(updated), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_merge_and_reset(self):
        matrix = self._skew()
        updater = PfaffianUpdater(matrix, max_delay=10)
        update = self._updates(1)[0]
        updater.accept(update)
        before = updater.inverse
        assert updater.delay == 1
        updater.merge()
        assert updater.delay == 0
        torch.testing.assert_close(updater.inverse, before, atol=0, rtol=0)
        updated = self._apply(matrix, update)
        updater.reset(updated)
        assert updater.delay == 0 and updater.max_delay == 10
        torch.testing.assert_close(
            updater.inverse, torch.linalg.inv(updated), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_from_inverse_matches_the_constructor(self):
        matrix = self._skew()
        inverse = torch.linalg.inv(matrix)
        from_matrix = PfaffianUpdater(matrix, max_delay=3)
        relative = PfaffianUpdater.from_inverse(inverse, max_delay=3)
        absolute = PfaffianUpdater.from_inverse(inverse, pfaffian=pfaffian(matrix), max_delay=3)
        for update in self._updates(1):
            ratio = from_matrix.accept(update)
            torch.testing.assert_close(relative.accept(update), ratio, atol=ATOL_SCALAR_COMPARISON, rtol=0)
            absolute.accept(update)
        torch.testing.assert_close(
            relative.pfaffian * pfaffian(matrix),
            from_matrix.pfaffian,
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )
        torch.testing.assert_close(
            absolute.pfaffian, from_matrix.pfaffian, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )
        torch.testing.assert_close(relative.inverse, from_matrix.inverse, atol=ATOL_MATRIX_COMPARISON, rtol=0)
        with pytest.raises(ValueError, match="max_delay"):
            PfaffianUpdater.from_inverse(inverse, max_delay=0)
        with pytest.raises(TypeError, match="supports"):
            PfaffianUpdater.from_inverse(inverse.long())

    def test_gradient_of_a_leaf_inverse_is_skew_symmetric(self):
        matrix = self._skew()
        leaf = torch.linalg.inv(matrix).requires_grad_(True)
        updater = PfaffianUpdater.from_inverse(leaf, max_delay=3)
        for update in self._updates(1)[:4]:
            updater.accept(update)
        updater.pfaffian.sum().backward()
        torch.testing.assert_close(leaf.grad, -leaf.grad.mT, atol=0, rtol=0)

    def test_gradient_flows_through_the_updates(self):
        matrix = self._skew()
        update = self._updates(1)[0].requires_grad_(True)
        updater = PfaffianUpdater(matrix, max_delay=4)
        updater.accept(update)
        updater.ratio(self._updates(1, seed=TEST_SEED + 5)[0]).sum().backward()
        reference = update.detach().clone().requires_grad_(True)
        second = self._updates(1, seed=TEST_SEED + 5)[0]
        updated = self._apply(matrix, reference)
        (pfaffian(self._apply(updated, second)) / pfaffian(updated)).sum().backward()
        torch.testing.assert_close(
            update.grad, reference.grad, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_float32_drift_stays_small(self):
        matrix = self._skew()
        updater = PfaffianUpdater(matrix.float(), max_delay=4)
        for update in self._updates(1):
            updater.accept(update.float())
            matrix = self._apply(matrix, update)
        torch.testing.assert_close(
            updater.inverse.double(),
            torch.linalg.inv(matrix),
            atol=ATOL_SINGLE_PRECISION_COMPARISON,
            rtol=RTOL_SINGLE_PRECISION_COMPARISON,
        )

    def test_invalid_inputs_raise(self):
        matrix = self._skew()
        with pytest.raises(ValueError, match="max_delay"):
            PfaffianUpdater(matrix, max_delay=0)
        with pytest.raises(ValueError, match="matrix"):
            PfaffianUpdater(matrix[..., :4])
        with pytest.raises(TypeError, match="supports"):
            PfaffianUpdater(matrix.half())
        updater = PfaffianUpdater(matrix)
        with pytest.raises(ValueError, match="update"):
            updater.ratio(torch.zeros(BATCH, DIMENSION, 3, dtype=torch.float64))
        with pytest.raises(TypeError, match="dtype"):
            updater.accept(torch.zeros(BATCH, DIMENSION, 2))
        with pytest.raises(ValueError, match="change"):
            updater.ratio_replace(0, torch.zeros(BATCH, DIMENSION + 1, dtype=torch.float64))
        with pytest.raises(TypeError, match="dtype"):
            updater.accept_replace(0, torch.zeros(BATCH, DIMENSION))
        with pytest.raises(TypeError, match="integer"):
            updater.ratio_replace(torch.tensor(0.5), torch.zeros(BATCH, DIMENSION, dtype=torch.float64))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    def test_cuda_matches_cpu(self):  # pragma: no cover - needs a GPU
        matrix = self._skew()
        on_cpu = PfaffianUpdater(matrix, max_delay=3)
        on_cuda = PfaffianUpdater(matrix.cuda(), max_delay=3)
        for update in self._updates(1):
            torch.testing.assert_close(
                on_cuda.accept(update.cuda()).cpu(), on_cpu.accept(update), atol=ATOL_SCALAR_COMPARISON, rtol=0
            )
        torch.testing.assert_close(on_cuda.inverse.cpu(), on_cpu.inverse, atol=ATOL_MATRIX_COMPARISON, rtol=0)

    def test_row_replacement_with_one_index_for_the_batch(self):
        matrix = self._skew()
        updater = PfaffianUpdater(matrix, max_delay=2)
        change = 0.3 * self._updates(1)[0][..., 0]  # (batch, n), one index broadcast over the batch
        ratio = updater.ratio_replace(4, change)
        expected = updater.ratio_replace(torch.full((BATCH,), 4), change)
        torch.testing.assert_close(ratio, expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)

    def test_small_cpu_inverses_take_native_immediate_updates(self, monkeypatch):
        pytest.importorskip("torch_pfaffian._rust")
        matrix = self._skew()
        delayed = PfaffianUpdater(matrix, max_delay=3)
        monkeypatch.setattr(PfaffianUpdater, "IMMEDIATE_MAX_NUMEL", 2**18)
        immediate = PfaffianUpdater(matrix, max_delay=3)
        for step, update in enumerate(self._updates(4)):
            mask = torch.tensor([True, step % 2 == 0, True])
            for updater in (immediate, delayed):
                updater.ratio(update)
            torch.testing.assert_close(
                immediate.ratio(update), delayed.ratio(update), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
            )
            torch.testing.assert_close(
                immediate.accept(update, mask),
                delayed.accept(update, mask),
                atol=ATOL_SCALAR_COMPARISON,
                rtol=RTOL_SCALAR_COMPARISON,
            )
            index, change = torch.tensor([1, 4, 7]), 0.3 * update[..., 0]
            torch.testing.assert_close(
                immediate.ratio_replace(index, change),
                delayed.ratio_replace(index, change),
                atol=ATOL_SCALAR_COMPARISON,
                rtol=RTOL_SCALAR_COMPARISON,
            )
            torch.testing.assert_close(
                immediate.accept_replace(index, change, mask),
                delayed.accept_replace(index, change, mask),
                atol=ATOL_SCALAR_COMPARISON,
                rtol=RTOL_SCALAR_COMPARISON,
            )
            assert immediate.delay == 0
        torch.testing.assert_close(
            immediate.inverse, delayed.inverse, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )
        torch.testing.assert_close(
            immediate.pfaffian, delayed.pfaffian, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )
