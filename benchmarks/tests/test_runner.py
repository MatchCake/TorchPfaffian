import signal

import pytest
import runner
import torch
from cases import BenchmarkCase

import torch_pfaffian


def _case(operation: str = "signed+grad", sign: bool = True, backward: bool = True, masked: bool = False):
    return BenchmarkCase("general", 8, 2, torch.float64, "cpu", operation, sign, backward, masked)


class TestRunner:
    @pytest.mark.parametrize(
        ("operation", "sign", "backward", "masked"),
        [("signed", True, False, False), ("signed+grad", True, True, False), ("masked+grad", False, True, True)],
    )
    def test_run_case_records_time_memory_and_summary(self, operation, sign, backward, masked):
        record = runner.run_case(_case(operation, sign, backward, masked), "small")
        assert record["status"] == "ok"
        assert 0 < record["seconds"] <= record["median_seconds"]
        assert record["repeats"] >= runner.MIN_REPEATS
        assert record["peak_bytes"] is None or record["peak_bytes"] >= 0
        expected_length = runner.GRADIENT_PROJECTIONS + 1 if backward else 2
        assert len(record["summary"]["real"]) == expected_length

    def test_run_case_marks_missing_keyword_as_unsupported(self, monkeypatch):
        def _old_pfaffian(matrix, *, sign=True):
            return matrix[..., 0, 1]

        monkeypatch.setattr(torch_pfaffian, "pfaffian", _old_pfaffian)
        record = runner.run_case(_case("masked+grad", False, True, True), "small")
        assert record["status"] == "unsupported"

    def test_run_case_records_errors(self, monkeypatch):
        def _broken(matrix, **keyword_arguments):
            raise RuntimeError("boom")

        monkeypatch.setattr(torch_pfaffian, "pfaffian", _broken)
        record = runner.run_case(_case(), "small")
        assert record["status"] == "error"
        assert "boom" in record["message"]

    def test_run_case_records_timeouts(self, monkeypatch):
        def _too_slow(matrix, **keyword_arguments):
            raise runner.CaseTimeout

        monkeypatch.setattr(torch_pfaffian, "pfaffian", _too_slow)
        assert runner.run_case(_case(), "small")["status"] == "timeout"

    def test_slow_case_is_timed_once(self, monkeypatch):
        monkeypatch.setitem(runner.TIMING_BUDGET, "small", (0.05, 0.0, 30))
        record = runner.run_case(_case(), "small")
        assert record["single_run"] is True
        assert record["repeats"] == 1

    def test_run_case_measures_the_memory_the_case_allocates(self, monkeypatch):
        def _allocating(matrix, **keyword_arguments):
            block = torch.ones(64 << 20, dtype=torch.uint8)  # 64 MiB, touched
            return matrix[..., 0, 1] + block[:1].to(matrix.dtype)

        monkeypatch.setattr(torch_pfaffian, "pfaffian", _allocating)
        record = runner.run_case(_case("signed", True, False, False), "small")
        assert record["status"] == "ok"
        assert 60 << 20 <= record["peak_bytes"] <= 80 << 20

    def test_run_case_other_type_errors_are_errors(self, monkeypatch):
        def _wrong_type(matrix, **keyword_arguments):
            raise TypeError("unsupported operand type(s)")

        monkeypatch.setattr(torch_pfaffian, "pfaffian", _wrong_type)
        assert runner.run_case(_case(), "small")["status"] == "error"

    def test_run_case_leaves_no_pending_alarm(self):
        runner.run_case(_case("signed", True, False, False), "small")
        assert signal.alarm(0) == 0

    def test_memory_timeout_keeps_the_timing(self, monkeypatch):
        def _slow_memory_run(run, device):
            raise runner.CaseTimeout

        monkeypatch.setattr(runner, "_peak_memory", _slow_memory_run)
        record = runner.run_case(_case("signed", True, False, False), "small")
        assert record["status"] == "ok"
        assert record["peak_bytes"] is None
