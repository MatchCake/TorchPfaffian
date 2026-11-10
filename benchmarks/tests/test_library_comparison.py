import json
import sys

import library_comparison
import numpy as np
import pytest
from library_comparison import (
    THREAD_VARIABLES,
    Measurement,
    PeakMemory,
    best_time,
    load_results,
    max_relative_difference,
    measure,
    measure_ratios,
    render_table,
    render_terminal_table,
    save_results,
    set_threads,
)


def _measurement(method: str, seconds: float, difference: float = 0.0) -> Measurement:
    return Measurement(8, 4, "float64", method, seconds, 3, difference)


class TestLibraryComparison:
    def test_set_threads_sets_every_variable(self, monkeypatch):
        for variable in THREAD_VARIABLES:
            monkeypatch.delenv(variable, raising=False)
        set_threads(3)
        assert all(library_comparison.os.environ[variable] == "3" for variable in THREAD_VARIABLES)

    def test_set_threads_none_keeps_environment(self, monkeypatch):
        for variable in THREAD_VARIABLES:
            monkeypatch.setenv(variable, "5")
        set_threads(None)
        assert all(library_comparison.os.environ[variable] == "5" for variable in THREAD_VARIABLES)

    def test_best_time_warms_up_and_repeats_within_bounds(self):
        calls = []
        seconds, repeats = best_time(lambda: calls.append(1), budget=0.0)
        assert repeats == 1  # the first repeat always runs, then the budget is spent
        assert len(calls) == 2  # warm-up plus the timed repeat
        assert seconds >= 0.0

    def test_best_time_stops_at_max_repeats(self):
        _, repeats = best_time(lambda: None, budget=60.0)
        assert repeats == library_comparison.MAX_REPEATS

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
        monkeypatch.setattr(library_comparison, "_import_pfapack", lambda: (None, None))
        monkeypatch.setattr(library_comparison, "_import_torchpfaff", lambda: None)
        monkeypatch.setattr(library_comparison, "_import_lrux", lambda: None)
        monkeypatch.setattr(library_comparison, "TIME_BUDGET_SECONDS", 0.0)
        measurements = measure(8, 3, dtype_name)
        methods = {item.method for item in measurements}
        assert {"torchpfaffian", "torchpfaffian + gradient"} <= methods
        assert "pfapack (C)" not in methods
        assert all(np.isnan(item.max_relative_difference) for item in measurements)

    def test_render_terminal_table_marks_fastest_method_per_input(self):
        measurements = [
            _measurement("pfapack (C)", 4e-3),
            _measurement("torchpfaffian", 1e-3, 2e-15),
            Measurement(16, 4, "float64", "pfapack (C)", 1e-3, 3, 0.0),
            Measurement(16, 4, "float64", "torchpfaffian", 2e-3, 3, 1e-15),
        ]
        lines = render_terminal_table(measurements).splitlines()
        assert lines[0].startswith("┌") and lines[-1].startswith("└")
        assert len({len(line) for line in lines}) == 1  # every line has the same width
        rows = [line for line in lines if line.startswith("│")][1:]
        assert "torchpfaffian ★" in rows[1] and "★" not in rows[0]
        assert "pfapack (C) ★" in rows[2] and "★" not in rows[3]
        assert sum(line.startswith("├") for line in lines) == 1  # one separator between the two inputs
        speedup_width = len("speed-up over pfapack (C)")
        assert f"│ {'4.00x':>{speedup_width}} │" in rows[1]  # numbers are right-aligned

    def test_measure_times_torchpfaff_and_skips_unsupported_inputs(self, monkeypatch, capsys):
        # A stand-in for torchpfaff that, like the real one, fails on complex matrices.
        import torch

        import torch_pfaffian

        class FakeTorchpfaff:
            @staticmethod
            def log_pfaffian(matrices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                if matrices.is_complex():
                    raise NotImplementedError("complex matrices are not supported")
                return torch_pfaffian.slog_pfaffian(matrices)

        monkeypatch.setattr(library_comparison, "_import_pfapack", lambda: (None, None))
        monkeypatch.setattr(library_comparison, "_import_torchpfaff", lambda: FakeTorchpfaff)
        monkeypatch.setattr(library_comparison, "_import_lrux", lambda: None)
        monkeypatch.setattr(library_comparison, "TIME_BUDGET_SECONDS", 0.0)
        monkeypatch.setattr(library_comparison, "REPORTED_FAILURES", set())
        real = {item.method for item in measure(8, 2, "float64")}
        assert {"torchpfaff", "torchpfaff + gradient"} <= real
        complex_methods = {item.method for item in measure(8, 2, "complex128")}
        complex_methods |= {item.method for item in measure(10, 2, "complex128")}
        assert "torchpfaff" not in complex_methods and "torchpfaffian" in complex_methods
        errors = capsys.readouterr().err
        assert errors.count("skipped torchpfaff on complex128: NotImplementedError") == 1  # reported once

    def test_measure_compares_with_a_reference_implementation(self, monkeypatch):
        # A stand-in for pfapack's C module, computing the Pfaffian with TorchPfaffian itself.
        import torch

        import torch_pfaffian

        class FakePfapack:
            @staticmethod
            def pfaffian(array: np.ndarray) -> float:
                return float(torch_pfaffian.pfaffian(torch.from_numpy(array)))

        monkeypatch.setattr(library_comparison, "_import_pfapack", lambda: (FakePfapack, FakePfapack))
        monkeypatch.setattr(library_comparison, "_import_lrux", lambda: None)
        monkeypatch.setattr(library_comparison, "TIME_BUDGET_SECONDS", 0.0)
        measurements = {item.method: item for item in measure(8, 2, "float64")}
        assert {"pfapack (C)", "pfapack (Python)", "torchpfaffian"} <= set(measurements)
        assert measurements["torchpfaffian"].max_relative_difference < 1e-12

    @pytest.mark.parametrize("dtype_name", ["float64", "complex128"])
    def test_measure_times_lrux_against_the_reference(self, dtype_name, monkeypatch):
        pytest.importorskip("lrux")
        import torch

        import torch_pfaffian

        class FakePfapack:
            @staticmethod
            def pfaffian(array: np.ndarray) -> complex:
                return complex(torch_pfaffian.pfaffian(torch.from_numpy(array)))

        monkeypatch.setattr(library_comparison, "_import_pfapack", lambda: (FakePfapack, None))
        monkeypatch.setattr(library_comparison, "_import_torchpfaff", lambda: None)
        monkeypatch.setattr(library_comparison, "TIME_BUDGET_SECONDS", 0.0)
        measurements = {item.method: item for item in measure(10, 2, dtype_name)}
        assert {"lrux", "lrux + gradient"} <= set(measurements)
        assert measurements["lrux"].max_relative_difference < 1e-12
        assert measurements["lrux + gradient"].max_relative_difference < 1e-12

    def test_save_and_load_results_round_trip(self, tmp_path):
        measurements = [_measurement("pfapack (C)", 4e-3), _measurement("lrux", 1e-3, float("nan"))]
        path = tmp_path / "results.json"
        save_results(path, "header", measurements)
        header, loaded = load_results(path)
        assert header == "header"
        assert render_table(loaded) == render_table(measurements)

    def test_measure_skips_slow_methods_on_larger_inputs(self, monkeypatch, capsys):
        monkeypatch.setattr(library_comparison, "_import_pfapack", lambda: (None, None))
        monkeypatch.setattr(library_comparison, "_import_torchpfaff", lambda: None)
        monkeypatch.setattr(library_comparison, "_import_lrux", lambda: None)
        monkeypatch.setattr(library_comparison, "TIME_BUDGET_SECONDS", 0.0)
        monkeypatch.setattr(library_comparison, "MAX_CALL_SECONDS", 0.0)  # every call is "slow"
        monkeypatch.setattr(library_comparison, "SLOW_METHODS", {})
        monkeypatch.setattr(library_comparison, "REPORTED_FAILURES", set())
        first = measure(8, 2, "float64")
        assert first and all(item.repeats == 1 for item in first)  # timed from the single call
        assert measure(8, 4, "float64") == []  # more work: every method is skipped
        assert measure(8, 1, "float64")  # less work: timed again
        assert "skipped torchpfaffian: skipped from m^3 * batch = 1024" in capsys.readouterr().err

    @pytest.mark.parametrize("update", list(library_comparison.UPDATES))
    def test_measure_ratios_agrees_with_a_reference(self, update, monkeypatch):
        import torch

        import torch_pfaffian

        class FakePfapack:
            @staticmethod
            def pfaffian(array: np.ndarray) -> float:
                return float(torch_pfaffian.pfaffian(torch.from_numpy(array)))

        monkeypatch.setattr(library_comparison, "_import_pfapack", lambda: (FakePfapack, None))
        monkeypatch.setattr(library_comparison, "TIME_BUDGET_SECONDS", 0.0)
        measurements = {item.method: item for item in measure_ratios(8, 2, "float64", update)}
        assert "pfapack (C)" in measurements and "torchpfaffian, new Pfaffian" in measurements
        assert all(item.update == update for item in measurements.values())
        assert all(item.max_relative_difference < 1e-10 for item in measurements.values())
        if update == f"row x{library_comparison.STEPS}":
            assert f"torchpfaffian, delayed updates (T={library_comparison.DELAY})" in measurements
        else:
            assert {"torchpfaffian ratio", "torchpfaffian ratio + inverse"} <= set(measurements)

    def test_ratio_tables_have_an_update_column(self):
        measurements = [
            Measurement(8, 4, "float64", "pfapack (C)", 4e-3, 3, 0.0, "row"),
            Measurement(8, 4, "float64", "torchpfaffian ratio", 1e-3, 3, 1e-15, "row"),
            Measurement(8, 4, "float64", "pfapack (C)", 2e-3, 3, 0.0, "rank 2"),
        ]
        assert "| m | batch | dtype | update | method |" in render_table(measurements)
        assert "| 8 | 4 | float64 | row | torchpfaffian ratio | 250.0 µs | 4.00x | 1.0e-15 |" in render_table(
            measurements
        )
        lines = render_terminal_table(measurements).splitlines()
        assert len({len(line) for line in lines}) == 1
        assert sum(line.startswith("├") for line in lines) == 1  # "row" and "rank 2" are separate blocks
        assert any("torchpfaffian ratio ★" in line for line in lines)

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/self")
    def test_peak_memory_sees_a_large_temporary_allocation(self):
        size = 64 << 20

        def allocate():
            array = np.ones(size // 8)  # touched, so resident
            return float(array[-1])

        allocate()
        assert 0.9 * size <= PeakMemory.measure("numpy", allocate) < 2 * size
        assert PeakMemory.measure("numpy", lambda: None) < size // 16

    def test_peak_memory_is_nan_without_proc(self, monkeypatch, tmp_path):
        monkeypatch.setattr(PeakMemory, "CLEAR_REFS_FILE", tmp_path / "missing" / "clear_refs")
        assert PeakMemory.measure("numpy", lambda: None) != PeakMemory.measure("numpy", lambda: None)  # nan
        monkeypatch.setattr(PeakMemory, "CLEAR_REFS_FILE", tmp_path / "clear_refs")
        (tmp_path / "status").write_text("Name: python\n")
        monkeypatch.setattr(PeakMemory, "STATUS_FILE", tmp_path / "status")
        assert PeakMemory.measure("numpy", lambda: None) != PeakMemory.measure("numpy", lambda: None)

    def test_measure_records_the_peak_memory(self, monkeypatch):
        monkeypatch.setattr(library_comparison, "_import_pfapack", lambda: (None, None))
        monkeypatch.setattr(library_comparison, "_import_torchpfaff", lambda: None)
        monkeypatch.setattr(library_comparison, "_import_lrux", lambda: None)
        monkeypatch.setattr(library_comparison, "TIME_BUDGET_SECONDS", 0.0)
        monkeypatch.setattr(PeakMemory, "measure", staticmethod(lambda name, function: 123.0))
        measurements = [item for item in measure(8, 2, "float64") if not item.method.endswith("(CUDA)")]
        assert measurements and all(item.peak_memory_bytes == 123.0 for item in measurements)

    def test_results_saved_before_the_memory_was_recorded_load(self, tmp_path):
        item = {"dimension": 8, "batch": 4, "dtype": "float64", "method": "lrux", "seconds": 1e-3, "repeats": 3}
        path = tmp_path / "old.json"
        path.write_text(json.dumps({"header": "old", "measurements": [{**item, "max_relative_difference": 0.0}]}))
        _, (loaded,) = load_results(path)
        assert loaded.update == "" and loaded.peak_memory_bytes != loaded.peak_memory_bytes  # nan
