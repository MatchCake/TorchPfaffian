import functools

import pytest
import torch

from tests.configs import (
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    TEST_SEED,
)
from torch_pfaffian import pfaffian
from torch_pfaffian.low_rank import CompiledLowRankPfaffian, LowRankPfaffian, PfaffianChain, PfaffianUpdater

DIMENSION = 8
BATCH = 3


class TestCompiledLowRankPfaffian:
    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _compile_works() -> bool:
        # torch.compile needs a working toolchain (a C++ compiler on CPU), which some CI platforms lack.
        try:
            return bool(torch.compile(lambda tensor: tensor * 2 + 1)(torch.ones(2)).sum() == 6)
        except Exception:  # pragma: no cover - depends on the platform
            return False

    @staticmethod
    def _skew(*shape: int, seed: int = TEST_SEED) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        entries = torch.randn(*shape, generator=generator, dtype=torch.float64)
        return entries - entries.mT

    @staticmethod
    def _random(*shape: int, seed: int = TEST_SEED + 1) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    @pytest.fixture(autouse=True)
    def _require_compile(self):
        if not self._compile_works():  # pragma: no cover - depends on the platform
            pytest.skip("torch.compile is not available on this platform")

    def test_is_a_low_rank_pfaffian_whose_cores_are_compiled(self):
        assert issubclass(CompiledLowRankPfaffian, LowRankPfaffian)
        core = LowRankPfaffian._ratio_core
        assert CompiledLowRankPfaffian._compiled(core) is CompiledLowRankPfaffian._compiled(core)
        assert CompiledLowRankPfaffian._compiled(core) is not core

    @pytest.mark.parametrize("width", [2, 4])
    def test_every_method_matches_the_eager_one(self, width):
        matrix = self._skew(BATCH, DIMENSION, DIMENSION)
        inverse = torch.linalg.inv(matrix)
        update = self._random(BATCH, DIMENSION, width)
        index, change = torch.tensor([0, 3, 7]), self._random(BATCH, DIMENSION, seed=TEST_SEED + 2)
        corner = self._skew(BATCH, width, width, seed=TEST_SEED + 3)
        calls = [
            ("ratio", (inverse, update)),
            ("ratio_and_inverse", (inverse, update)),
            ("ratio_replace", (inverse, index, change)),
            ("ratio_replace_and_inverse", (inverse, index, change)),
            ("ratio_extend", (inverse, update, corner)),
            ("ratio_extend_and_inverse", (inverse, update, corner)),
            ("ratio_and_r_inverse", (update, LowRankPfaffian.project(inverse, update))),
        ]
        for name, arguments in calls:
            expected = getattr(LowRankPfaffian, name)(*arguments)
            result = getattr(CompiledLowRankPfaffian, name)(*arguments)
            torch.testing.assert_close(result, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_gradients_match_the_eager_ones(self):
        matrix = self._skew(BATCH, DIMENSION, DIMENSION)
        gradients = []
        for backend in (LowRankPfaffian, CompiledLowRankPfaffian):
            inverse = torch.linalg.inv(matrix).requires_grad_(True)
            update = self._random(BATCH, DIMENSION, 2).requires_grad_(True)
            ratio, new_inverse = backend.ratio_and_inverse(inverse, update)
            (ratio.sum() + new_inverse.square().sum()).backward()
            gradients.append((inverse.grad, update.grad))
        for compiled, eager in zip(gradients[1], gradients[0]):
            torch.testing.assert_close(compiled, eager, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_compiled_chain_matches_the_pfaffians_of_the_leading_blocks(self):
        matrix = self._skew(BATCH, 2 * DIMENSION, 2 * DIMENSION)
        chain = PfaffianChain((BATCH,), compiled=True)
        for size in range(2, 2 * DIMENSION + 1, 2):  # a new size at every step: dynamic shapes, not recompiles
            border = matrix[..., : size - 2, size - 2 : size]
            corner = matrix[..., size - 2 : size, size - 2 : size]
            corners = torch.stack([corner, 2 * corner], dim=-3)
            ratios = chain.ratios(border, corners)
            torch.testing.assert_close(
                ratios[..., 0], chain.ratio(border, corner), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
            )
            chain.extend(border, corner)
        torch.testing.assert_close(
            chain.pfaffian, pfaffian(matrix), atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
        )
        assert PfaffianChain.from_matrix(matrix, compiled=True)._backend is CompiledLowRankPfaffian

    def test_compiled_updater_matches_the_eager_one(self):
        matrix = self._skew(BATCH, DIMENSION, DIMENSION)
        eager = PfaffianUpdater(matrix, max_delay=3)
        compiled = PfaffianUpdater(matrix, max_delay=3, compiled=True)
        relative = PfaffianUpdater.from_inverse(torch.linalg.inv(matrix), max_delay=3, compiled=True)
        for step in range(7):
            update = 0.3 * self._random(BATCH, DIMENSION, 2, seed=TEST_SEED + 10 + step)
            mask = torch.tensor([True, step % 2 == 0, True])
            expected = eager.accept(update, mask)
            torch.testing.assert_close(
                compiled.accept(update, mask), expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
            )
            relative.accept(update, mask)
        torch.testing.assert_close(compiled.inverse, eager.inverse, atol=ATOL_MATRIX_COMPARISON, rtol=0)
        torch.testing.assert_close(compiled.pfaffian, eager.pfaffian, atol=ATOL_SCALAR_COMPARISON, rtol=0)
        torch.testing.assert_close(
            relative.pfaffian * pfaffian(matrix),
            eager.pfaffian,
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )
