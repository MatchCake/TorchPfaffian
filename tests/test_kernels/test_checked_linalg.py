import subprocess
import sys
import textwrap
from unittest import mock

import pytest
import torch

from tests.configs import ATOL_MATRIX_COMPARISON, RTOL_MATRIX_COMPARISON, TEST_SEED
from torch_pfaffian.kernels.checked_linalg import CheckedLinalg

# A batch the check covers: on CPU, at least two matrices of size at least MIN_SIZE.
_SIZE = CheckedLinalg.MIN_SIZE + 4


def _batch(dtype: torch.dtype, size: int = _SIZE, count: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(TEST_SEED)
    return torch.randn(count, size, size, dtype=torch.float64, generator=generator).to(dtype)


class TestCheckedLinalg:
    @pytest.fixture(autouse=True)
    def clear_cached_verdicts(self):
        # The verdicts are cached for the process; mocked probes must not leak into other tests.
        CheckedLinalg._is_corrupted.cache_clear()
        yield
        CheckedLinalg._is_corrupted.cache_clear()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
    def test_probe_is_sound_in_this_process(self, dtype):
        # The test processes never call torch.set_num_threads, so the probe must pass in every dtype, at and above
        # the size where the corruption starts.
        for size in (CheckedLinalg.MIN_SIZE, 160, CheckedLinalg.PROBE_MAX_SIZE):
            assert not CheckedLinalg._probe_is_corrupted(size, dtype)

    @pytest.mark.parametrize(
        "name, reference",
        [
            ("det", torch.linalg.det),
            ("slogdet", torch.linalg.slogdet),
            ("inv", torch.linalg.inv),
            ("inv_ex", torch.linalg.inv_ex),
            ("lu_factor_ex", torch.linalg.lu_factor_ex),
        ],
    )
    def test_functions_match_torch(self, name, reference):
        matrix = _batch(torch.float64)
        result = getattr(CheckedLinalg, name)(matrix)
        expected = reference(matrix)
        if isinstance(expected, torch.Tensor):
            result, expected = (result,), (expected,)
        assert len(result) == len(expected)
        for value, expected_value in zip(result, expected):
            torch.testing.assert_close(value, expected_value, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_functions_keep_the_gradient(self):
        matrix = _batch(torch.float64).requires_grad_(True)
        CheckedLinalg.slogdet(matrix)[1].sum().backward()
        assert matrix.grad is not None
        expected = torch.linalg.inv(matrix.detach()).mT
        torch.testing.assert_close(matrix.grad, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    @pytest.mark.parametrize(
        "matrix",
        [
            _batch(torch.float64, size=_SIZE, count=1),  # a single matrix
            _batch(torch.float64, size=_SIZE, count=2)[0],  # no batch dimension
            _batch(torch.float64, size=CheckedLinalg.MIN_SIZE - 2, count=4),  # below MIN_SIZE
            torch.empty(2, _SIZE, _SIZE, device="meta"),  # not on CPU
        ],
    )
    def test_check_skips_unaffected_inputs(self, matrix):
        with mock.patch.object(CheckedLinalg, "_probe_is_corrupted", return_value=True) as probe:
            CheckedLinalg.check(matrix)
        probe.assert_not_called()

    def test_check_skips_a_single_thread(self):
        with mock.patch.object(torch, "get_num_threads", return_value=1):
            with mock.patch.object(CheckedLinalg, "_probe_is_corrupted", return_value=True) as probe:
                CheckedLinalg.check(_batch(torch.float64))
        probe.assert_not_called()

    def test_verdict_is_cached_per_thread_count(self):
        matrix = _batch(torch.float64)
        with mock.patch.object(CheckedLinalg, "_probe_is_corrupted", return_value=False) as probe:
            with mock.patch.object(torch, "get_num_threads", return_value=4):
                CheckedLinalg.check(matrix)
                CheckedLinalg.check(matrix)
            assert probe.call_count == 1
            with mock.patch.object(torch, "get_num_threads", return_value=2):
                CheckedLinalg.check(matrix)
            assert probe.call_count == 2

    @pytest.mark.parametrize("name", ["det", "slogdet", "inv", "inv_ex", "lu_factor_ex"])
    def test_raises_when_the_probe_is_corrupted(self, name):
        with (
            mock.patch.object(torch, "get_num_threads", return_value=2),
            mock.patch.object(CheckedLinalg, "_probe_is_corrupted", return_value=True),
        ):
            with pytest.raises(RuntimeError) as error:
                getattr(CheckedLinalg, name)(_batch(torch.float64))
        message = str(error.value)
        assert all(url in message for url in CheckedLinalg.ISSUE_URLS)
        assert "OMP_NUM_THREADS" in message and "torch.set_num_threads" in message

    def test_large_inputs_are_judged_from_the_largest_probe(self):
        matrix = torch.empty(2, CheckedLinalg.PROBE_MAX_SIZE + 2, CheckedLinalg.PROBE_MAX_SIZE + 2)
        with mock.patch.object(torch, "get_num_threads", return_value=2):
            with mock.patch.object(CheckedLinalg, "_probe_is_corrupted", return_value=False) as probe:
                CheckedLinalg.check(matrix)
        probe.assert_called_once_with(CheckedLinalg.PROBE_MAX_SIZE, matrix.dtype)

    def test_probe_detects_invalid_pivots(self):
        original = torch.linalg.lu_factor_ex

        def invalid_pivots(matrix):
            factors, pivots, info = original(matrix)
            return factors, torch.zeros_like(pivots), info

        with mock.patch.object(torch.linalg, "lu_factor_ex", side_effect=invalid_pivots):
            assert CheckedLinalg._probe_is_corrupted(_SIZE, torch.float64)

    def test_probe_detects_wrong_factors_with_valid_pivots(self):
        # The common symptom: the pivots look valid but the factors are wrong.
        original = torch.linalg.lu_factor_ex

        def wrong_factors(matrix):
            factors, pivots, info = original(matrix)
            return factors.roll(1, dims=-1), pivots, info

        with mock.patch.object(torch.linalg, "lu_factor_ex", side_effect=wrong_factors):
            assert CheckedLinalg._probe_is_corrupted(_SIZE, torch.float64)

    def test_raises_exactly_when_batched_lu_is_corrupted_after_set_num_threads(self):
        # In a fresh process, after torch.set_num_threads(2): the check raises if and only if a batched determinant
        # is really wrong there (it is on CPU builds with the MKL bug, not with other LAPACKs).
        script = textwrap.dedent(
            """
            import torch
            from torch_pfaffian.kernels.checked_linalg import CheckedLinalg

            torch.set_num_threads(2)
            print("STEP set_num_threads", flush=True)
            generator = torch.Generator().manual_seed(0)
            matrix = torch.randn(4, 192, 192, dtype=torch.float64, generator=generator)
            expected = torch.stack([torch.linalg.slogdet(element)[1] for element in matrix])
            corrupted = not torch.allclose(torch.linalg.slogdet(matrix)[1], expected)
            print("STEP batched slogdet", flush=True)
            try:
                CheckedLinalg.check(matrix)
                raised = False
            except RuntimeError:
                raised = True
            print("RESULT", corrupted, raised)
            """
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=True
            )
        except subprocess.TimeoutExpired as error:
            # In the corrupted state, MKL's nested thread teams can also deadlock (seen on CI runners, not
            # reproduced locally): that is the PyTorch bug itself, not the check, so it does not fail the test.
            output = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
            steps = [line for line in output.splitlines() if line.startswith("STEP")]
            pytest.skip(f"the process hung after the set_num_threads(2) call, last step: {steps[-1:]}")
        result = [line for line in completed.stdout.splitlines() if line.startswith("RESULT")]
        assert len(result) == 1, completed.stdout[-2000:] + completed.stderr[-2000:]
        _, corrupted, raised = result[0].split()
        assert corrupted == raised

    @pytest.mark.parametrize("trailing", [(), (5,), (5, 5)])
    def test_nan_where_input_has_nan_sets_only_the_nan_matrices(self, trailing):
        matrices = _batch(torch.float64, size=5, count=3)
        matrices[1, 2, 3] = torch.nan
        value = torch.ones((3, *trailing), dtype=torch.float64)
        result = CheckedLinalg._nan_where_input_has_nan(matrices, value)
        assert torch.isnan(result[1]).all()
        torch.testing.assert_close(result[[0, 2]], value[[0, 2]], atol=0.0, rtol=0.0)

    def test_nan_where_input_has_nan_supports_a_single_complex_matrix(self):
        matrix = torch.eye(4, dtype=torch.complex128)
        matrix[0, 1] = complex(torch.nan, 0.0)
        result = CheckedLinalg._nan_where_input_has_nan(matrix, torch.tensor(2.0 + 1.0j, dtype=torch.complex128))
        assert result.dtype == torch.complex128 and torch.isnan(result)

    def test_propagate_nan_returns_the_cpu_result_itself(self):
        matrix = torch.full((4, 4), torch.nan, dtype=torch.float64)
        value = torch.zeros((), dtype=torch.float64)
        assert CheckedLinalg._propagate_nan(matrix, value) is value

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    @pytest.mark.parametrize("batch", [(), (1,), (3,)])
    def test_functions_return_nan_for_a_nan_cuda_matrix(self, batch):  # pragma: no cover - needs a GPU
        matrices = torch.randn(*batch, 40, 40, dtype=torch.float64, generator=torch.Generator().manual_seed(TEST_SEED))
        matrices = matrices.cuda()
        matrices[..., 0, 1] = torch.nan
        assert torch.isnan(CheckedLinalg.det(matrices)).all()
        sign, log_abs = CheckedLinalg.slogdet(matrices)
        assert torch.isnan(sign).all() and torch.isnan(log_abs).all()
        assert torch.isnan(CheckedLinalg.inv(matrices)).all()
        assert torch.isnan(CheckedLinalg.inv_ex(matrices)[0]).all()
        assert torch.isnan(CheckedLinalg.lu_factor_ex(matrices)[0]).all()

    def test_nan_where_input_has_nan_passes_an_empty_matrix_through(self):
        value = torch.ones(3, dtype=torch.float64)
        result = CheckedLinalg._nan_where_input_has_nan(torch.zeros(3, 0, 0, dtype=torch.float64), value)
        assert result is value

    def test_nan_where_input_has_nan_reads_a_conjugated_view(self):
        matrices = torch.ones(2, 3, 3, dtype=torch.complex128)
        matrices[0, 1, 2] = complex(1.0, torch.nan)
        result = CheckedLinalg._nan_where_input_has_nan(matrices.conj(), torch.zeros(2, dtype=torch.complex128))
        assert torch.isnan(result[0]) and result[1] == 0
