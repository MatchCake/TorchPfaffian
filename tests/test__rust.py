import numpy as np
import pytest

from tests.configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    N_RANDOM_TESTS_PER_CASE,
    RTOL_APPROX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    TEST_SEED,
)
from tests.reference import random_skew, slog_pfaffian_numpy

_rust = pytest.importorskip("torch_pfaffian._rust")

# (signed kernel, slog kernel, numpy dtype, tight?) for each precision that exposes a slog entry point.
_KERNELS = [
    ("signed_pfaffian_f64", "signed_slog_pfaffian_f64", np.float64, True),
    ("signed_pfaffian_f32", "signed_slog_pfaffian_f32", np.float32, False),
    ("signed_pfaffian_c128", "signed_slog_pfaffian_c128", np.complex128, True),
    ("signed_pfaffian_c64", "signed_slog_pfaffian_c64", np.complex64, False),
]


def _skew_batch(count: int, dimension: int, rng: np.random.Generator, dtype: type, scale: float = 1.0) -> np.ndarray:
    return np.stack([random_skew(rng, dimension, dtype=dtype, scale=scale) for _ in range(count)])


def _reconstruct(phase: np.ndarray, log_abs: np.ndarray) -> np.ndarray:
    # phase * exp(log_abs); exponentiate in float64 so a moderate float32 log does not overflow.
    return phase * np.exp(log_abs.astype(np.float64)).astype(phase.dtype)


class TestRustSlog:
    @pytest.mark.parametrize("signed_name, slog_name, dtype, tight", _KERNELS)
    def test_reconstructs_signed_kernel(self, signed_name, slog_name, dtype, tight):
        rng = np.random.default_rng(TEST_SEED)
        batch = _skew_batch(N_RANDOM_TESTS_PER_CASE, 8, rng, dtype)
        signed = getattr(_rust, signed_name)(batch)
        phase, log_abs = getattr(_rust, slog_name)(batch)
        assert phase.dtype == np.dtype(dtype)
        assert log_abs.dtype == np.empty(0, dtype).real.dtype  # f64->f64, f32->f32, c128->f64, c64->f32
        atol = ATOL_SCALAR_COMPARISON if tight else ATOL_APPROX_COMPARISON
        rtol = RTOL_SCALAR_COMPARISON if tight else RTOL_APPROX_COMPARISON
        np.testing.assert_allclose(_reconstruct(phase, log_abs), signed, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("signed_name, slog_name, dtype, tight", _KERNELS)
    def test_matches_slog_numpy_oracle(self, signed_name, slog_name, dtype, tight):
        rng = np.random.default_rng(TEST_SEED + 1)
        batch = _skew_batch(N_RANDOM_TESTS_PER_CASE, 6, rng, dtype)
        phase, log_abs = getattr(_rust, slog_name)(batch)
        atol = ATOL_SCALAR_COMPARISON if tight else ATOL_APPROX_COMPARISON
        rtol = RTOL_SCALAR_COMPARISON if tight else RTOL_APPROX_COMPARISON
        for index in range(batch.shape[0]):
            reference = (
                batch[index].astype(np.complex128) if np.iscomplexobj(batch) else batch[index].astype(np.float64)
            )
            expected_phase, expected_log = slog_pfaffian_numpy(reference)
            np.testing.assert_allclose(phase[index], expected_phase, atol=atol, rtol=rtol)
            np.testing.assert_allclose(float(log_abs[index]), expected_log, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        "slog_name, dtype", [("signed_slog_pfaffian_f64", np.float64), ("signed_slog_pfaffian_c128", np.complex128)]
    )
    def test_exact_zero_pivot_gives_zero_phase_and_neg_inf_log(self, slog_name, dtype):
        matrix = np.zeros((1, 4, 4), dtype=dtype)
        matrix[0, 2, 3] = 1.0
        matrix[0, 3, 2] = -1.0
        phase, log_abs = getattr(_rust, slog_name)(matrix)
        assert phase[0] == 0
        assert np.isneginf(log_abs[0])

    @pytest.mark.parametrize(
        "signed_name, slog_name, dtype",
        [
            ("signed_pfaffian_f64", "signed_slog_pfaffian_f64", np.float64),
            ("signed_pfaffian_c128", "signed_slog_pfaffian_c128", np.complex128),
        ],
    )
    def test_no_overflow_at_large_scale(self, signed_name, slog_name, dtype):
        # At scale 1e120 the linear signed kernel overflows to inf, but the log-domain kernel stays finite.
        rng = np.random.default_rng(TEST_SEED)
        batch = _skew_batch(2, 8, rng, dtype, scale=1e120)
        signed = getattr(_rust, signed_name)(batch)
        _, log_abs = getattr(_rust, slog_name)(batch)
        assert not np.isfinite(signed).all()
        assert np.isfinite(log_abs).all()

    def test_odd_dimension_and_empty(self):
        rng = np.random.default_rng(TEST_SEED)
        odd = _skew_batch(3, 5, rng, np.float64)
        phase, log_abs = _rust.signed_slog_pfaffian_f64(odd)
        assert np.all(phase == 0)
        assert np.all(np.isneginf(log_abs))
        empty = np.zeros((2, 0, 0), dtype=np.float64)
        phase, log_abs = _rust.signed_slog_pfaffian_f64(empty)
        np.testing.assert_array_equal(phase, np.ones(2))
        np.testing.assert_array_equal(log_abs, np.zeros(2))

    def test_batched_matches_serial_below_and_above_parallel_threshold(self):
        # The rayon split at PARALLEL_BATCH_THRESHOLD must not change results: a large batch agrees
        # element-wise with the same matrices evaluated one at a time.
        rng = np.random.default_rng(TEST_SEED)
        batch = _skew_batch(16, 6, rng, np.float64)
        phase, log_abs = _rust.signed_slog_pfaffian_f64(batch)
        for index in range(batch.shape[0]):
            one_phase, one_log = _rust.signed_slog_pfaffian_f64(batch[index : index + 1])
            np.testing.assert_allclose(
                phase[index], one_phase[0], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
            )
            np.testing.assert_allclose(
                log_abs[index], one_log[0], atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON
            )
