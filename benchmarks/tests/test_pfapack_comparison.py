import numpy as np
import pfapack_comparison
import pytest
from pfapack_comparison import (
    THREAD_VARIABLES,
    Measurement,
    best_time,
    max_relative_difference,
    measure,
    render_table,
    set_threads,
)


def _measurement(method: str, seconds: float, difference: float = 0.0) -> Measurement:
    return Measurement(8, 4, "float64", method, seconds, 3, difference)


class TestPfapackComparison:
    def test_set_threads_sets_every_variable(self, monkeypatch):
        for variable in THREAD_VARIABLES:
            monkeypatch.delenv(variable, raising=False)
        set_threads(3)
        assert all(pfapack_comparison.os.environ[variable] == "3" for variable in THREAD_VARIABLES)

    def test_set_threads_none_keeps_environment(self, monkeypatch):
        for variable in THREAD_VARIABLES:
            monkeypatch.setenv(variable, "5")
        set_threads(None)
        assert all(pfapack_comparison.os.environ[variable] == "5" for variable in THREAD_VARIABLES)

    def test_best_time_warms_up_and_repeats_within_bounds(self):
        calls = []
        seconds, repeats = best_time(lambda: calls.append(1), budget=0.0)
        assert repeats == 1  # the first repeat always runs, then the budget is spent
        assert len(calls) == 2  # warm-up plus the timed repeat
        assert seconds >= 0.0

    def test_best_time_stops_at_max_repeats(self):
        _, repeats = best_time(lambda: None, budget=60.0)
        assert repeats == pfapack_comparison.MAX_REPEATS

    def test_max_relative_difference(self):
        assert max_relative_difference([1.0, -2.2], [1.0, -2.0]) == pytest.approx(0.1)
        assert max_relative_difference([1 + 1j], [1 + 1j]) == 0.0

    def test_max_relative_difference_handles_zero_reference(self):
        assert max_relative_difference([1.0], [0.0]) > 1e300  # divided by the smallest float, not by 0
        assert max_relative_difference([0.0], [0.0]) == 0.0

    def test_render_table_reports_speedup_over_pfapack_c(self):
        table = render_table([_measurement("pfapack (C)", 4e-3), _measurement("torchpfaffian", 1e-3, 2e-15)])
        assert "| 8 | 4 | float64 | torchpfaffian | 250.0 µs | 4.00x | 2.0e-15 |" in table
        assert "| 8 | 4 | float64 | pfapack (C) | 1.00 ms | 1.00x | 0.0e+00 |" in table

    def test_render_table_without_pfapack(self):
        table = render_table([_measurement("torchpfaffian", 2.0, float("nan"))])
        assert "| torchpfaffian | 500.00 ms | n/a | n/a |" in table

    @pytest.mark.parametrize("dtype_name", ["float64", "complex128"])
    def test_measure_without_pfapack_times_torchpfaffian_only(self, dtype_name, monkeypatch):
        monkeypatch.setattr(pfapack_comparison, "_import_pfapack", lambda: (None, None))
        monkeypatch.setattr(pfapack_comparison, "TIME_BUDGET_SECONDS", 0.0)
        measurements = measure(8, 3, dtype_name)
        methods = {item.method for item in measurements}
        assert {"torchpfaffian", "torchpfaffian + gradient"} <= methods
        assert "pfapack (C)" not in methods
        assert all(np.isnan(item.max_relative_difference) for item in measurements)

    def test_measure_compares_with_a_reference_implementation(self, monkeypatch):
        # A stand-in for pfapack's C module, computing the Pfaffian with TorchPfaffian itself.
        import torch

        import torch_pfaffian

        class FakePfapack:
            @staticmethod
            def pfaffian(array: np.ndarray) -> float:
                return float(torch_pfaffian.pfaffian(torch.from_numpy(array)))

        monkeypatch.setattr(pfapack_comparison, "_import_pfapack", lambda: (FakePfapack, FakePfapack))
        monkeypatch.setattr(pfapack_comparison, "TIME_BUDGET_SECONDS", 0.0)
        measurements = {item.method: item for item in measure(8, 2, "float64")}
        assert {"pfapack (C)", "pfapack (Python)", "torchpfaffian"} <= set(measurements)
        assert measurements["torchpfaffian"].max_relative_difference < 1e-12
