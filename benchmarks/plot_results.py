import argparse
import textwrap
from collections.abc import Iterable, Sequence
from pathlib import Path

from library_comparison import Measurement, load_results

CUDA_SUFFIX = " (CUDA)"
COMPILED_SUFFIX = ", compiled"
NEW_PFAFFIAN = "new Pfaffian"
# One fixed color per library, whether or not a run has it, so that a library keeps its color across figures.
LIBRARY_COLORS = {
    "torchpfaffian": "#2a78d6",
    "pfapack (C)": "#eb6834",
    "lrux": "#1baf7a",
    "torchpfaff": "#eda100",
    "pfapack (Python)": "#e87ba4",
    "torchpfaffian, new Pfaffian": "#008300",
    "torchpfaffian, compiled": "#6250d6",
}
DEVICE_DASHES = {"CPU": "", "CUDA": (4, 2)}
DEVICE_MARKERS = {"CPU": "o", "CUDA": "s"}
PANELS_PER_ROW = 4
PANEL_SIZE = (3.6, 2.8)  # inches
# The CPU memory is read in pages: a call that stays within the pages already resident reads as 0 and is shown at
# one page, the resolution, so that it stays on the log scale.
MEMORY_FLOOR_BYTES = 4096
METRICS = (("seconds", "time per matrix"), ("peak_memory_bytes", "peak memory per call"))
# The overview shows packages, not variants; each has a marker shape too, since its colors are not all pairwise
# distinct enough for a scatter plot (color is never the only cue).
PACKAGE_MARKERS = {"torchpfaffian": "o", "pfapack (C)": "s", "lrux": "^", "torchpfaff": "D", "pfapack (Python)": "v"}
ITEM_SIZES = {"float32": 4, "float64": 8, "complex64": 8, "complex128": 16}
OVERVIEW_METRICS = (("call_seconds", "time per call"), ("peak_memory_bytes", "peak memory per call"))


class ResultsPlot:
    """
    Plot the time and the peak memory of a ``library_comparison.py`` run against the matrix size.

    Each results file gives one figure per batch size (and per dtype, for ``--ratios`` runs), saved as a PNG next
    to it. A figure has one panel per comparison: the Pfaffian and the Pfaffian with its gradient, per dtype, for
    a full-Pfaffian run; the ratio, and the ratio with the updated inverse, per low-rank update, for a ``--ratios``
    run, with the Pfaffian of the updated matrix (``new Pfaffian``) as the baseline of every panel. Each panel has
    the time per matrix above and the peak memory of one call of the whole batch below, on log scales. The color
    is the library, and the line style the device (solid on CPU, dashed on CUDA). A method without a memory
    record (lrux on CUDA, results saved before memory was recorded) has no memory line.

    ``--overview`` adds one figure with every measurement of every file given: the time and the peak memory of
    one call against the size of its input (the batch of matrices, or of inverses for ``--ratios``, in bytes),
    which is defined the same way for every operation, dtype and batch size. The color and the marker shape are the
    package (TorchPfaffian's compiled variant and its ``new Pfaffian`` baseline count as TorchPfaffian), filled on
    CPU and hollow on CUDA. A dotted line marks the memory equal to the input size.
    """

    @staticmethod
    def package(library: str) -> str:
        """
        Package of a library returned by :meth:`describe`.

        :param library: A library, such as ``"torchpfaffian, compiled"``.
        :return: Its package, such as ``"torchpfaffian"``.
        :rtype: str
        """
        return library.removesuffix(COMPILED_SUFFIX).removesuffix(f", {NEW_PFAFFIAN}")

    @staticmethod
    def describe(method: str) -> tuple[str, str, str]:
        """
        Split a method name of ``library_comparison.py`` into its library, operation and device.

        :param method: A method name, such as ``"torchpfaffian ratio + inverse, compiled (CUDA)"``.
        :return: The library (``"torchpfaffian, compiled"``), the operation (``"ratio + inverse"``) and the device
            (``"CUDA"`` or ``"CPU"``).
        :rtype: tuple[str, str, str]
        """
        device = "CUDA" if method.endswith(CUDA_SUFFIX) else "CPU"
        name = method.removesuffix(CUDA_SUFFIX)
        compiled = name.endswith(COMPILED_SUFFIX)
        name = name.removesuffix(COMPILED_SUFFIX)
        library, operation = name, "Pfaffian"
        if name.endswith(f", {NEW_PFAFFIAN}"):
            library, operation = name, NEW_PFAFFIAN
        elif ", " in name:  # "torchpfaffian, delayed updates (T=4)"
            library, operation = name.split(", ", 1)
        elif name.endswith(" + gradient"):
            library, operation = name.removesuffix(" + gradient"), "Pfaffian + gradient"
        elif " ratio" in name:  # "lrux ratio", "torchpfaffian ratio + inverse"
            library, operation = name.split(" ratio", 1)
            operation = "ratio" + operation
        return library + (COMPILED_SUFFIX if compiled else ""), operation, device

    @staticmethod
    def panels(measurements: Sequence[Measurement]) -> list[tuple[str, str]]:
        """
        Comparisons of one figure, in the order of the run.

        :param measurements: Measurements of one figure.
        :return: One ``(update, operation)`` pair per panel (``update`` is empty for a full-Pfaffian run).
        :rtype: list[tuple[str, str]]
        """
        ratios = any(item.update for item in measurements)
        panels: dict[tuple[str, str], None] = {}
        for item in measurements:
            _, operation, _ = ResultsPlot.describe(item.method)
            if ratios and (operation == NEW_PFAFFIAN or item.method.startswith("pfapack")):
                continue  # the baselines are drawn in every panel of their update
            panels[(item.dtype if not ratios else item.update, operation)] = None
        return list(panels)

    @staticmethod
    def figure_groups(measurements: Iterable[Measurement]) -> dict[tuple[int, str], list[Measurement]]:
        """
        Split the measurements into figures: one per batch size, and per dtype for a ``--ratios`` run.

        :param measurements: All the measurements of a run.
        :return: The measurements of each figure, keyed by ``(batch, dtype)`` (``dtype`` is empty for a
            full-Pfaffian run, whose dtypes are panels).
        :rtype: dict[tuple[int, str], list[Measurement]]
        """
        groups: dict[tuple[int, str], list[Measurement]] = {}
        for item in measurements:
            groups.setdefault((item.batch, item.dtype if item.update else ""), []).append(item)
        return groups

    @staticmethod
    def format_seconds(seconds: float, _position: object = None) -> str:
        """
        Tick label of a time axis.

        :param seconds: Time in seconds.
        :param _position: Tick position (unused, part of matplotlib's formatter signature).
        :return: The time in ns, µs, ms or s.
        :rtype: str
        """
        for unit, scale in (("s", 1.0), ("ms", 1e-3), ("µs", 1e-6)):
            if seconds >= scale:
                return f"{seconds / scale:g} {unit}"
        return f"{seconds / 1e-9:g} ns"

    @staticmethod
    def format_bytes(size: float, _position: object = None) -> str:
        """
        Tick label of a memory axis.

        :param size: Memory in bytes.
        :param _position: Tick position (unused, part of matplotlib's formatter signature).
        :return: The memory in B, kB, MB or GB.
        :rtype: str
        """
        for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
            if size >= scale:
                return f"{size / scale:g} {unit}"
        return f"{size:g} B"

    @staticmethod
    def plot_file(path: Path, output_directory: Path | None = None) -> list[Path]:
        """
        Plot one results file.

        :param path: A JSON file written by ``library_comparison.py``.
        :param output_directory: Where to save the figures; next to ``path`` by default.
        :return: The saved figures.
        :rtype: list[Path]
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        header, measurements = load_results(path)
        directory = path.parent if output_directory is None else output_directory
        directory.mkdir(parents=True, exist_ok=True)
        saved = []
        for (batch, dtype), group in ResultsPlot.figure_groups(measurements).items():
            figure = ResultsPlot.figure(group, header, f"{path.stem}, batch {batch}" + (f", {dtype}" if dtype else ""))
            output = directory / f"{path.stem}-batch{batch}{'-' + dtype if dtype else ''}.png"
            figure.savefig(output, dpi=150, bbox_inches="tight")
            plt.close(figure)
            saved.append(output)
        return saved

    @staticmethod
    def figure(measurements: Sequence[Measurement], header: str, title: str):
        """
        One figure: a time row and a memory row per row of panels.

        :param measurements: Measurements of one batch size (and one dtype for a ``--ratios`` run).
        :param header: The run's description of the libraries and the machine, printed under the figure.
        :param title: Figure title.
        :return: The figure.
        :rtype: matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid", context="notebook", rc={"grid.color": "#e4e3df", "axes.edgecolor": "#c3c2b7"})
        frame = ResultsPlot._frame(measurements)
        panels = ResultsPlot.panels(measurements)
        columns = min(PANELS_PER_ROW, len(panels))
        panel_rows = -(-len(panels) // columns)
        figure, axes = plt.subplots(
            2 * panel_rows,
            columns,
            figsize=(PANEL_SIZE[0] * columns, PANEL_SIZE[1] * 2 * panel_rows),
            squeeze=False,
            sharex=True,
        )
        ratios = bool(frame["update"].any())
        for index, (group, operation) in enumerate(panels):
            panel_row, column = divmod(index, columns)
            in_group = frame["update" if ratios else "dtype"] == group
            shown = in_group & (
                (frame["operation"] == operation)
                | (ratios & ((frame["operation"] == NEW_PFAFFIAN) | frame["method"].str.startswith("pfapack")))
            )
            for metric_index, (metric, label) in enumerate(METRICS):
                ResultsPlot._panel(
                    axes[2 * panel_row + metric_index][column],
                    frame[shown],
                    metric,
                    f"{group}: {operation}" if ratios else f"{operation}, {group}",
                    label if column == 0 else "",
                )
        for index in range(len(panels), panel_rows * columns):
            panel_row, column = divmod(index, columns)
            for metric_index in range(len(METRICS)):
                axes[2 * panel_row + metric_index][column].set_visible(False)
        ResultsPlot._legend(figure, frame)
        figure.suptitle(title, y=1.0, fontsize="large", fontweight="bold")
        figure.text(0.0, -0.01, "\n".join(textwrap.wrap(header, 180)), fontsize="x-small", color="#52514e", va="top")
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        return figure

    @staticmethod
    def overview(measurements: Sequence[Measurement], header: str):
        """
        One figure with every measurement: time and peak memory per call against the input size.

        :param measurements: Measurements of any number of runs, full-Pfaffian and ``--ratios`` alike.
        :param header: Description of the libraries and the machine, printed under the figure.
        :return: The figure.
        :rtype: matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import seaborn as sns
        from matplotlib.lines import Line2D

        sns.set_theme(style="whitegrid", context="notebook", rc={"grid.color": "#e4e3df", "axes.edgecolor": "#c3c2b7"})
        frame = ResultsPlot._frame(measurements)
        frame["package"] = frame["library"].map(ResultsPlot.package)
        frame["input_bytes"] = [item.dimension**2 * item.batch * ITEM_SIZES.get(item.dtype, 8) for item in measurements]
        frame["call_seconds"] = [item.seconds for item in measurements]
        packages = [name for name in PACKAGE_MARKERS if name in set(frame["package"])]
        packages += sorted(set(frame["package"]) - set(packages))
        figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        for axis, (metric, label) in zip(axes, OVERVIEW_METRICS):
            for (package, device), points in frame.dropna(subset=[metric]).groupby(["package", "device"]):
                color = LIBRARY_COLORS.get(package, "#898781")
                axis.scatter(
                    points["input_bytes"],
                    points[metric],
                    s=36,
                    marker=PACKAGE_MARKERS.get(package, "o"),
                    facecolors=color if device == "CPU" else "white",
                    edgecolors=color if device == "CUDA" else "white",
                    linewidths=1.4 if device == "CUDA" else 0.6,
                    alpha=0.85,
                    zorder=3,
                )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.xaxis.set_major_formatter(ticker.FuncFormatter(ResultsPlot.format_bytes))
            formatter = ResultsPlot.format_seconds if metric == "call_seconds" else ResultsPlot.format_bytes
            axis.yaxis.set_major_formatter(ticker.FuncFormatter(formatter))
            axis.set_xlabel("input size (matrices of the batch)")
            axis.set_ylabel(label)
            axis.set_title(label.capitalize(), fontsize="medium")
            axis.tick_params(labelsize="small")
        sizes = frame["input_bytes"]
        guide = [sizes.min(), sizes.max()]
        axes[1].plot(guide, guide, color="#898781", linestyle=":", linewidth=1.5, zorder=2)
        axes[1].annotate(
            "memory = input size",
            (guide[0], guide[0]),
            xytext=(8, -4),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize="small",
            color="#52514e",
        )
        handles = [
            Line2D(
                [],
                [],
                linestyle="",
                marker=PACKAGE_MARKERS.get(name, "o"),
                markersize=7,
                markerfacecolor=LIBRARY_COLORS.get(name, "#898781"),
                markeredgecolor="white",
                label=name,
            )
            for name in packages
        ]
        handles += [
            Line2D(
                [],
                [],
                linestyle="",
                marker="o",
                markersize=7,
                markerfacecolor="#52514e",
                markeredgecolor="white",
                label="CPU (filled)",
            ),
            Line2D(
                [],
                [],
                linestyle="",
                marker="o",
                markersize=7,
                markerfacecolor="white",
                markeredgecolor="#52514e",
                markeredgewidth=1.4,
                label="CUDA (hollow)",
            ),
        ]
        figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=len(handles), frameon=False)
        figure.suptitle("Overview: every benchmark point", y=1.0, fontsize="large", fontweight="bold")
        figure.text(0.0, -0.01, "\n".join(textwrap.wrap(header, 180)), fontsize="x-small", color="#52514e", va="top")
        figure.tight_layout(rect=(0, 0, 1, 0.9))
        return figure

    @staticmethod
    def main(arguments: Sequence[str] | None = None) -> None:
        """
        Command line: plot each results file given.

        :param arguments: Command-line arguments, ``sys.argv`` by default.
        """
        parser = argparse.ArgumentParser(description=(ResultsPlot.__doc__ or "").strip().split("\n\n")[0])
        parser.add_argument("results", type=Path, nargs="+", help="JSON files written by library_comparison.py")
        parser.add_argument("--output-directory", type=Path, default=None, help="next to each results file by default")
        parser.add_argument("--overview", action="store_true", help="also plot every point of every file in one figure")
        parsed = parser.parse_args(arguments)
        for path in parsed.results:
            for output in ResultsPlot.plot_file(path, parsed.output_directory):
                print(output)
        if parsed.overview:
            import matplotlib.pyplot as plt

            runs = [load_results(path) for path in parsed.results]
            headers = list(dict.fromkeys(header for header, _ in runs))
            figure = ResultsPlot.overview([item for _, items in runs for item in items], "; ".join(headers))
            directory = parsed.results[0].parent if parsed.output_directory is None else parsed.output_directory
            output = directory / "overview.png"
            figure.savefig(output, dpi=150, bbox_inches="tight")
            plt.close(figure)
            print(output)

    @staticmethod
    def _frame(measurements: Sequence[Measurement]):
        import pandas as pd

        rows = []
        for item in measurements:
            library, operation, device = ResultsPlot.describe(item.method)
            memory = item.peak_memory_bytes
            if memory == memory:  # not nan
                memory = max(memory, MEMORY_FLOOR_BYTES)
            rows.append(
                {
                    "dimension": item.dimension,
                    "dtype": item.dtype,
                    "update": item.update,
                    "method": item.method,
                    "library": library,
                    "operation": operation,
                    "device": device,
                    "seconds": item.seconds / item.batch,
                    "peak_memory_bytes": memory,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _panel(axis, frame, metric: str, title: str, ylabel: str) -> None:
        import matplotlib.ticker as ticker
        import seaborn as sns

        data = frame.dropna(subset=[metric])
        # One line per method: a ratio panel can hold two methods of one library and device (the baselines).
        for _, method_data in data.groupby("method", sort=False):
            library, device = method_data["library"].iloc[0], method_data["device"].iloc[0]
            sns.lineplot(
                data=method_data.sort_values("dimension"),
                x="dimension",
                y=metric,
                ax=axis,
                color=LIBRARY_COLORS.get(library, "#898781"),
                dashes=DEVICE_DASHES[device],
                marker=DEVICE_MARKERS[device],
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=1,
                linewidth=2,
                errorbar=None,
                legend=False,
            )
        if data.empty:  # memory of a run saved before it was recorded
            axis.text(0.5, 0.5, "not recorded", transform=axis.transAxes, ha="center", va="center", color="#898781")
        dimensions = sorted(frame["dimension"].unique())
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.xaxis.set_major_locator(ticker.FixedLocator(dimensions))
        axis.xaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g}"))
        axis.xaxis.set_minor_locator(ticker.NullLocator())
        formatter = ResultsPlot.format_seconds if metric == "seconds" else ResultsPlot.format_bytes
        axis.yaxis.set_major_formatter(ticker.FuncFormatter(formatter))
        axis.set_title(title, fontsize="medium")
        axis.set_xlabel("matrix size")
        axis.set_ylabel(ylabel)
        axis.tick_params(labelsize="small", labelbottom=True)

    @staticmethod
    def _legend(figure, frame) -> None:
        from matplotlib.lines import Line2D

        libraries = [name for name in LIBRARY_COLORS if name in set(frame["library"])]
        libraries += sorted(set(frame["library"]) - set(libraries))
        handles = [
            Line2D([], [], color=LIBRARY_COLORS.get(name, "#898781"), linewidth=2, label=name) for name in libraries
        ]
        for device in DEVICE_DASHES:
            if device in set(frame["device"]):
                handles.append(
                    Line2D(
                        [],
                        [],
                        color="#52514e",
                        linewidth=2,
                        linestyle=(0, DEVICE_DASHES[device]) if DEVICE_DASHES[device] else "-",
                        marker=DEVICE_MARKERS[device],
                        markersize=6,
                        label=device,
                    )
                )
        figure.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=len(handles), frameon=False
        )


if __name__ == "__main__":
    ResultsPlot.main()
