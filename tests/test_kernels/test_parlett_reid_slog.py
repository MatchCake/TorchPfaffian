import numpy as np
import pytest
import torch

from tests.configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_APPROX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    TEST_SEED,
)
from tests.reference import pfaffian_numpy, random_skew, slog_pfaffian_numpy
from torch_pfaffian.kernels.parlett_reid_slog import parlett_reid_slog


def _skew_batch(
    shape: tuple[int, ...], dimension: int, rng: np.random.Generator, dtype: type = np.float64, scale: float = 1.0
) -> torch.Tensor:
    count = int(np.prod(shape))
    matrices = np.stack([random_skew(rng, dimension, dtype=dtype, scale=scale) for _ in range(count)])
    return torch.from_numpy(matrices.reshape(*shape, dimension, dimension))


class TestParlettReidSlog:
    def test_reconstructs_pfaffian_numpy_real(self):
        for dimension in (2, 4, 6, 8, 10):
            rng = np.random.default_rng(TEST_SEED + dimension)
            batch = _skew_batch((N_RANDOM_TESTS_PER_CASE,), dimension, rng)
            phase, log_abs = parlett_reid_slog(batch)
            reconstructed = phase * torch.exp(log_abs)
            expected = torch.tensor(
                [pfaffian_numpy(batch[index].numpy()) for index in range(batch.shape[0])], dtype=batch.dtype
            )
            torch.testing.assert_close(
                reconstructed, expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
            )

    def test_reconstructs_pfaffian_numpy_complex(self):
        for dimension in (2, 4, 6, 8):
            rng = np.random.default_rng(TEST_SEED + dimension)
            batch = _skew_batch((N_RANDOM_TESTS_PER_CASE,), dimension, rng, dtype=np.complex128)
            phase, log_abs = parlett_reid_slog(batch)
            reconstructed = phase * torch.exp(log_abs).to(phase.dtype)
            expected = torch.tensor(
                [pfaffian_numpy(batch[index].numpy()) for index in range(batch.shape[0])], dtype=batch.dtype
            )
            torch.testing.assert_close(
                reconstructed, expected, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
            )

    def test_matches_slog_numpy_phase_and_log(self):
        rng = np.random.default_rng(TEST_SEED)
        batch = _skew_batch((N_RANDOM_TESTS_PER_CASE,), 8, rng)
        phase, log_abs = parlett_reid_slog(batch)
        for index in range(batch.shape[0]):
            expected_phase, expected_log = slog_pfaffian_numpy(batch[index].numpy())
            torch.testing.assert_close(
                phase[index], torch.tensor(expected_phase, dtype=batch.dtype), atol=ATOL_SCALAR_COMPARISON, rtol=0.0
            )
            torch.testing.assert_close(
                log_abs[index],
                torch.tensor(expected_log, dtype=batch.dtype),
                atol=ATOL_SCALAR_COMPARISON,
                rtol=RTOL_SCALAR_COMPARISON,
            )

    def test_nested_batch_shapes(self):
        rng = np.random.default_rng(TEST_SEED)
        batch = _skew_batch((2, 3), 6, rng)
        phase, log_abs = parlett_reid_slog(batch)
        assert phase.shape == (2, 3)
        assert log_abs.shape == (2, 3)
        flat_phase, flat_log = parlett_reid_slog(batch.reshape(6, 6, 6))
        torch.testing.assert_close(phase.reshape(6), flat_phase)
        torch.testing.assert_close(log_abs.reshape(6), flat_log)

    def test_zero_matrix_gives_zero_phase_and_neg_inf_log(self):
        matrix = torch.zeros(2, 4, 4, dtype=torch.float64)
        phase, log_abs = parlett_reid_slog(matrix)
        assert torch.all(phase == 0)
        assert torch.all(log_abs == -torch.inf)

    def test_odd_dimension(self):
        rng = np.random.default_rng(TEST_SEED)
        batch = _skew_batch((3,), 5, rng)
        phase, log_abs = parlett_reid_slog(batch)
        assert torch.all(phase == 0)
        assert torch.all(log_abs == -torch.inf)

    def test_empty_matrix(self):
        matrix = torch.zeros(3, 0, 0, dtype=torch.float64)
        phase, log_abs = parlett_reid_slog(matrix)
        torch.testing.assert_close(phase, torch.ones(3, dtype=torch.float64))
        torch.testing.assert_close(log_abs, torch.zeros(3, dtype=torch.float64))

    def test_no_overflow_or_underflow_at_extreme_scale(self):
        rng = np.random.default_rng(TEST_SEED)
        for scale in (1e120, 1e-120):
            batch = _skew_batch((2,), 32, rng, scale=scale)
            phase, log_abs = parlett_reid_slog(batch)
            assert torch.isfinite(log_abs).all()
            torch.testing.assert_close(phase.abs(), torch.ones_like(phase.abs()))

    def test_float32_reasonable_accuracy(self):
        rng = np.random.default_rng(TEST_SEED)
        batch = _skew_batch((4,), 8, rng, dtype=np.float32)
        phase, log_abs = parlett_reid_slog(batch)
        reconstructed = (phase * torch.exp(log_abs)).numpy()
        expected = np.array(
            [pfaffian_numpy(batch[index].numpy().astype(np.float64)) for index in range(batch.shape[0])]
        )
        np.testing.assert_allclose(reconstructed, expected, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)

    def test_complex64_supported(self):
        rng = np.random.default_rng(TEST_SEED)
        batch = _skew_batch((3,), 6, rng, dtype=np.complex64)
        phase, log_abs = parlett_reid_slog(batch)
        reconstructed = (phase * torch.exp(log_abs).to(phase.dtype)).numpy()
        expected = np.array(
            [pfaffian_numpy(batch[index].numpy().astype(np.complex128)) for index in range(batch.shape[0])]
        )
        np.testing.assert_allclose(reconstructed, expected, atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            parlett_reid_slog(torch.zeros(3, 4, dtype=torch.float64))

    def test_preserves_dtype_and_device(self):
        rng = np.random.default_rng(TEST_SEED)
        for dtype in (np.float32, np.float64, np.complex64, np.complex128):
            batch = _skew_batch((2,), 4, rng, dtype=dtype)
            phase, _ = parlett_reid_slog(batch)
            assert phase.dtype == batch.dtype
            assert phase.device == batch.device

    def test_log_abs_real_dtype_matches_complex_precision(self):
        rng = np.random.default_rng(TEST_SEED)
        complex64_batch = _skew_batch((2,), 4, rng, dtype=np.complex64)
        complex128_batch = _skew_batch((2,), 4, rng, dtype=np.complex128)
        assert parlett_reid_slog(complex64_batch)[1].dtype == torch.float32
        assert parlett_reid_slog(complex128_batch)[1].dtype == torch.float64
