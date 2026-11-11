import pytest
import torch

from tests.configs import (
    ATOL_SCALAR_COMPARISON,
    ATOL_SINGLE_PRECISION_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    RTOL_SINGLE_PRECISION_COMPARISON,
    TEST_SEED,
)
from torch_pfaffian.kernels.parlett_reid_slog import parlett_reid_slog
from torch_pfaffian.kernels.parlett_reid_triton import ParlettReidTriton
from torch_pfaffian.strategies import PfaffianParlettReid

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


class TestParlettReidTriton:
    @staticmethod
    def _skew(*shape: int, dtype: torch.dtype = torch.float64, seed: int = TEST_SEED) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        entries = torch.randn(*shape, generator=generator, dtype=torch.float64)
        return ((entries - entries.mT) / shape[-1] ** 0.5).to(dtype)

    def test_supports_only_real_cuda_inputs_of_even_size_in_range(self):
        assert not ParlettReidTriton.supports(self._skew(2, 8, 8))  # on CPU
        assert not ParlettReidTriton.supports(torch.empty(2, 8, 8, dtype=torch.complex128, device="meta"))
        assert not ParlettReidTriton.supports(torch.empty(2, 7, 7, device="meta"))  # odd
        assert not ParlettReidTriton.supports(torch.empty(2, 2, 2, device="meta"))  # no elimination step
        assert not ParlettReidTriton.supports(torch.empty(1, 1026, 1026, device="meta"))  # beyond MAX_DIMENSION

    @CUDA
    def test_supports_on_cuda(self):  # pragma: no cover - needs a GPU
        pytest.importorskip("triton")
        matrix = self._skew(2, 8, 8).cuda()
        assert ParlettReidTriton.supports(matrix)
        assert ParlettReidTriton.supports(matrix.float())
        assert not ParlettReidTriton.supports(matrix.to(torch.complex128))  # no complex arithmetic in Triton
        assert not ParlettReidTriton.supports(matrix[..., :7, :7])  # odd
        assert not ParlettReidTriton.supports(matrix[..., :2, :2])  # no elimination step
        assert not ParlettReidTriton.supports(torch.zeros(1, 1026, 1026, device="cuda"))  # beyond MAX_DIMENSION
        assert ParlettReidTriton.supports(torch.empty(4, 1024, 1024, dtype=torch.float64, device="cuda"))

    @CUDA
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize(
        ("size", "batch"), [(4, 3), (10, 5), (40, 3), (64, 7), (130, 2), (290, 2)]
    )  # panels and tiles
    def test_matches_the_pytorch_loop(self, dtype, size, batch, monkeypatch):  # pragma: no cover - needs a GPU
        pytest.importorskip("triton")
        matrix = self._skew(batch, size, size, dtype=dtype).cuda()
        matrix[0, :, 1] = 0  # a zero pivot column: Pf = 0
        matrix[0, 1, :] = 0
        assert ParlettReidTriton.supports(matrix)
        result = PfaffianParlettReid.forward(matrix)
        working, swaps = ParlettReidTriton.eliminate(matrix)
        assert working.shape == (batch, size, size) and swaps.shape == (batch,)
        monkeypatch.setattr(ParlettReidTriton, "supports", staticmethod(lambda tensor: False))
        expected = PfaffianParlettReid.forward(matrix)
        assert result[0] == 0
        if dtype == torch.float32:
            atol, rtol = ATOL_SINGLE_PRECISION_COMPARISON, RTOL_SINGLE_PRECISION_COMPARISON
        else:
            atol, rtol = ATOL_SCALAR_COMPARISON, RTOL_SCALAR_COMPARISON
        torch.testing.assert_close(result, expected, atol=atol, rtol=rtol)

    @CUDA
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_log_domain_pfaffian_matches_the_pytorch_loop(self, dtype, monkeypatch):  # pragma: no cover - needs a GPU
        pytest.importorskip("triton")
        matrix = self._skew(5, 40, 40, dtype=dtype).cuda()
        matrix[1] = 0  # Pf = 0: phase 0, log -inf
        phase, log_abs = parlett_reid_slog(matrix)
        monkeypatch.setattr(ParlettReidTriton, "supports", staticmethod(lambda tensor: False))
        expected_phase, expected_log = parlett_reid_slog(matrix)
        if dtype == torch.float32:
            atol, rtol = ATOL_SINGLE_PRECISION_COMPARISON, RTOL_SINGLE_PRECISION_COMPARISON
        else:
            atol, rtol = ATOL_SCALAR_COMPARISON, RTOL_SCALAR_COMPARISON
        torch.testing.assert_close(phase, expected_phase)
        torch.testing.assert_close(log_abs, expected_log, atol=atol, rtol=rtol)
        assert phase[1] == 0 and log_abs[1] == -torch.inf
