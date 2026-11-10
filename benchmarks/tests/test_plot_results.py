import pytest
from library_comparison import Measurement, save_results
from plot_results import ResultsPlot

pytest.importorskip("seaborn")


class TestResultsPlot:
    @staticmethod
    def _measurements(
        methods: list[str], update: str = "", dtypes: tuple[str, ...] = ("float64",)
    ) -> list[Measurement]:
        return [
            Measurement(dimension, batch, dtype, method, 1e-4 * dimension, 3, 0.0, update, 1e3 * dimension)
            for dtype in dtypes
            for batch in (1, 4)
            for dimension in (8, 16)
            for method in methods
        ]

    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            ("pfapack (C)", ("pfapack (C)", "Pfaffian", "CPU")),
            ("torchpfaffian + gradient (CUDA)", ("torchpfaffian", "Pfaffian + gradient", "CUDA")),
            ("torchpfaffian, new Pfaffian (CUDA)", ("torchpfaffian, new Pfaffian", "new Pfaffian", "CUDA")),
            ("lrux ratio", ("lrux", "ratio", "CPU")),
            ("torchpfaffian ratio + inverse, compiled (CUDA)", ("torchpfaffian, compiled", "ratio + inverse", "CUDA")),
            (
                "torchpfaffian, delayed updates (T=4), compiled",
                ("torchpfaffian, compiled", "delayed updates (T=4)", "CPU"),
            ),
            ("lrux, immediate updates (CUDA)", ("lrux", "immediate updates", "CUDA")),
        ],
    )
    def test_describe_splits_library_operation_and_device(self, method, expected):
        assert ResultsPlot.describe(method) == expected

    def test_panels_and_figures_of_a_full_pfaffian_run(self):
        measurements = self._measurements(
            ["pfapack (C)", "torchpfaffian", "torchpfaffian + gradient"], dtypes=("float64", "complex128")
        )
        assert set(ResultsPlot.figure_groups(measurements)) == {(1, ""), (4, "")}
        assert ResultsPlot.panels(measurements) == [
            ("float64", "Pfaffian"),
            ("float64", "Pfaffian + gradient"),
            ("complex128", "Pfaffian"),
            ("complex128", "Pfaffian + gradient"),
        ]

    def test_ratio_baselines_are_not_panels(self):
        methods = ["pfapack (C)", "torchpfaffian, new Pfaffian", "torchpfaffian ratio", "lrux ratio + inverse (CUDA)"]
        measurements = self._measurements(methods, update="row", dtypes=("float64", "complex128"))
        assert set(ResultsPlot.figure_groups(measurements)) == {
            (1, "float64"),
            (4, "float64"),
            (1, "complex128"),
            (4, "complex128"),
        }
        assert ResultsPlot.panels(measurements) == [("row", "ratio"), ("row", "ratio + inverse")]

    def test_tick_labels(self):
        assert ResultsPlot.format_seconds(2.5e-5) == "25 µs"
        assert ResultsPlot.format_seconds(3e-9) == "3 ns"
        assert ResultsPlot.format_seconds(2.0) == "2 s"
        assert ResultsPlot.format_bytes(4096) == "4.096 kB"
        assert ResultsPlot.format_bytes(2e9) == "2 GB"
        assert ResultsPlot.format_bytes(10) == "10 B"

    @pytest.mark.parametrize("ratios", [False, True])
    def test_main_saves_one_figure_per_group(self, ratios, tmp_path, capsys):
        if ratios:
            methods = ["pfapack (C)", "torchpfaffian ratio", "torchpfaffian ratio (CUDA)", "lrux ratio"]
            measurements = self._measurements(methods, update="rank 2") + self._measurements(methods[:2], update="row")
        else:
            measurements = self._measurements(["pfapack (C)", "torchpfaffian", "lrux + gradient (CUDA)", "unknown"])
        measurements[0].peak_memory_bytes = float("nan")  # not recorded
        measurements[1].peak_memory_bytes = 0.0  # within resident pages: shown at the floor
        results = tmp_path / "results.json"
        save_results(results, "header", measurements)
        ResultsPlot.main([str(results), "--output-directory", str(tmp_path / "plots")])
        saved = capsys.readouterr().out.split()
        suffix = "-float64" if ratios else ""
        assert saved == [str(tmp_path / "plots" / f"results-batch{batch}{suffix}.png") for batch in (1, 4)]
        assert all(
            (tmp_path / "plots" / name).stat().st_size > 0 for name in [path.rsplit("/", 1)[1] for path in saved]
        )

    def test_memory_that_was_not_recorded_is_marked(self, tmp_path):
        measurements = self._measurements(["torchpfaffian"])
        for item in measurements:
            item.peak_memory_bytes = float("nan")
        figure = ResultsPlot.figure(measurements[:2], "header", "title")
        texts = [text.get_text() for axis in figure.axes for text in axis.texts]
        assert texts == ["not recorded"]

    def test_package_merges_the_variants(self):
        assert ResultsPlot.package("torchpfaffian, compiled") == "torchpfaffian"
        assert ResultsPlot.package("torchpfaffian, new Pfaffian") == "torchpfaffian"
        assert ResultsPlot.package("pfapack (C)") == "pfapack (C)"

    def test_overview_puts_every_run_in_one_figure(self, tmp_path, capsys):
        full = self._measurements(["pfapack (C)", "torchpfaffian (CUDA)", "unknown"], dtypes=("float64", "complex128"))
        ratios = self._measurements(["torchpfaffian ratio, compiled", "lrux ratio (CUDA)"], update="row")
        ratios[0].peak_memory_bytes = float("nan")
        paths = [tmp_path / "libraries.json", tmp_path / "ratios.json"]
        save_results(paths[0], "header", full)
        save_results(paths[1], "header", ratios)
        ResultsPlot.main([str(path) for path in paths] + ["--overview"])
        assert capsys.readouterr().out.split()[-1] == str(tmp_path / "overview.png")
        assert (tmp_path / "overview.png").stat().st_size > 0
        figure = ResultsPlot.overview(full + ratios, "header")
        time_axis, memory_axis = figure.axes
        points = len(ResultsPlot.relative_to_best(full + ratios))
        assert sum(len(collection.get_offsets()) for collection in time_axis.collections) == points
        labels = [text.get_text() for text in time_axis.get_legend().get_texts()]
        assert labels[0].startswith("torchpfaffian: fastest on 100% of") and "1.0x on average" in labels[0]
        assert memory_axis.get_legend().get_texts()[0].get_text().startswith("torchpfaffian: leanest on")

    def test_relative_to_best_compares_each_package_at_its_best(self):
        def measured(method: str, seconds: float, memory: float, update: str = "") -> Measurement:
            return Measurement(8, 1, "float64", method, seconds, 3, 0.0, update, 1e3 * memory)  # above the floor

        measurements = [
            measured("pfapack (C)", 2e-3, 1e3),
            measured("torchpfaffian", 1e-3, 4e3),
            measured("torchpfaffian (CUDA)", 5e-4, float("nan")),
            measured("lrux", 4e-3, 2e3),
            measured("unknown + gradient", 1e-3, 1e3),  # the only package on its input: dropped
            measured("pfapack (C)", 3e-3, 1e3, "row"),
            measured("torchpfaffian, new Pfaffian", 2e-3, 1e3, "row"),
            measured("torchpfaffian ratio", 1e-3, 2e3, "row"),
            measured("torchpfaffian ratio + inverse", 6e-3, 2e3, "row"),
            measured("lrux ratio", 5e-4, 4e3, "row"),
        ]
        frame = ResultsPlot.relative_to_best(measurements).set_index(["update", "operation", "package"])
        assert set(frame.index) == {
            ("", "Pfaffian", "pfapack (C)"),
            ("", "Pfaffian", "torchpfaffian"),
            ("", "Pfaffian", "lrux"),
            ("row", "ratio", "pfapack (C)"),
            ("row", "ratio", "torchpfaffian"),
            ("row", "ratio", "lrux"),
            ("row", "ratio + inverse", "pfapack (C)"),
            ("row", "ratio + inverse", "torchpfaffian"),
        }
        torchpfaffian = frame.loc[("", "Pfaffian", "torchpfaffian")]
        assert torchpfaffian["device"] == "CUDA" and torchpfaffian["relative_seconds"] == 1.0
        assert frame.loc[("", "Pfaffian", "lrux"), "relative_seconds"] == 8.0
        assert torchpfaffian["relative_memory"] == 4.0  # its leanest method, the CPU one
        assert frame.loc[("row", "ratio", "torchpfaffian"), "relative_seconds"] == 2.0
        # Without an inverse from lrux, the new Pfaffian (2 ms) is TorchPfaffian's best on "ratio + inverse".
        assert frame.loc[("row", "ratio + inverse", "torchpfaffian"), "relative_seconds"] == 1.0
