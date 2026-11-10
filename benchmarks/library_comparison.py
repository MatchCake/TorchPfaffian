"""
Compare TorchPfaffian with pfapack, torchpfaff and lrux, on speed and agreement, on CPU and on GPU if available.

pfapack (https://github.com/basnijholt/pfapack) computes one NumPy matrix at a time, without gradients or GPU
support; its C implementation is the fast one and serves as the reference. torchpfaff
(https://github.com/Nuclear-Physics-with-Machine-Learning/PyPfaffian) is a differentiable PyTorch Pfaffian
with an optional Triton kernel on CUDA. lrux (https://github.com/ChenAo-Phys/lrux) is a JAX library of
low-rank Pfaffian updates; its full Pfaffian (from its dependency fermix) is timed here, jitted, in double
precision. fermix's fast kernels need a CUDA GPU: on CPU it falls back to a generic jax.numpy implementation.
This script times every library on the same random skew-symmetric matrices (the ``general`` family of
``cases.py``, scaled so that ``|Pf|`` stays near 1), on CPU and, when CUDA is available, on the GPU, and
checks that they agree with pfapack. None of these libraries is a dependency of the project, so run the script
from a separate environment, on a Python version for which pfapack ships wheels (3.12 or 3.13 at the time of
writing, since torchpfaff needs 3.12 or newer). A library that is not installed is not timed:

    uv venv --python 3.12 ~/.venvs/pfaffian-libraries
    # CPU-only torch; drop the index URL for the CUDA build, to time the GPU too
    uv pip install --python ~/.venvs/pfaffian-libraries torch --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python ~/.venvs/pfaffian-libraries pfapack . "torchpfaff @ git+https://github.com/Nuclear-Physics-with-Machine-Learning/PyPfaffian"
    uv pip install --python ~/.venvs/pfaffian-libraries lrux  # "jax[cuda13]" too, to time lrux on the GPU
    ~/.venvs/pfaffian-libraries/bin/python benchmarks/library_comparison.py --threads 1

A method that fails on an input (for example torchpfaff on complex matrices, which it does not support) is
skipped, with a note on the first failure of each method and dtype. ``--markdown`` prints a Markdown table
instead of the terminal one. The results are saved after every input; ``--render <file>`` prints the table of a
saved (possibly interrupted) run. A method whose single call takes more than ``--max-call-seconds`` (10 s) is
timed from that call and skipped on larger inputs. torchpfaff's Triton kernel compiles for minutes from m = 64
(cached on disk afterwards), so its CUDA path is skipped above ``--torchpfaff-cuda-max-size`` (64).

Threads are set through ``OMP_NUM_THREADS``, ``MKL_NUM_THREADS`` and ``RAYON_NUM_THREADS`` before torch is
imported (never ``torch.set_num_threads``, which corrupts batched LU on some MKL builds; issue #23).
"""

import argparse
import json
import os
import platform
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS")
DEFAULT_SIZES = (8, 16, 32, 64, 128, 256)
DEFAULT_BATCHES = (1, 256)
DEFAULT_DTYPES = ("float64", "complex128")
# The pure-Python pfapack loops over rows in Python, so it is only timed on small inputs.
PYTHON_PFAPACK_MAX_WORK = 64 * 64 * 64 * 16  # m^3 * batch
# Each method is repeated until this many seconds have passed (at least once, at most MAX_REPEATS times).
TIME_BUDGET_SECONDS = 1.0
MAX_REPEATS = 20
RESULTS_DIRECTORY = Path(__file__).resolve().parent / "results"
REFERENCE_METHOD = "pfapack (C)"
TABLE_HEADER = ("m", "batch", "dtype", "method", "time per matrix", "speed-up over pfapack (C)", "max rel. difference")
# Columns right-aligned in the terminal table (numbers).
RIGHT_ALIGNED_COLUMNS = (0, 1, 4, 5, 6)
FASTEST_MARKER = " ★"
# Failures already reported, as (method, dtype), so that each one is printed once and not for every size.
REPORTED_FAILURES: set[tuple[str, str]] = set()
# A method whose single call takes longer than this is timed from that call only, and skipped on every later
# input with at least as much work (m^3 * batch), so that one slow library cannot stall the whole run.
MAX_CALL_SECONDS = 10.0
# Work (m^3 * batch) at which each method went over MAX_CALL_SECONDS.
SLOW_METHODS: dict[str, int] = {}
# torchpfaff's Triton kernel takes minutes to compile from m = 64 (about 100 s on an RTX 3070) and did not
# finish within several minutes at m = 128, so its CUDA path is only timed up to this size by default.
TORCHPFAFF_CUDA_MAX_DIMENSION = 64


@dataclass
class Measurement:
    """
    Timing and agreement of one method on one input.

    ``seconds`` is the best time over the repeats for the whole batch. ``max_relative_difference`` compares the
    method's Pfaffians with those of pfapack's C implementation (``nan`` when pfapack is not available).
    """

    dimension: int
    batch: int
    dtype: str
    method: str
    seconds: float
    repeats: int
    max_relative_difference: float


def set_threads(threads: int | None) -> None:
    """
    Set the thread-count environment variables, before torch or NumPy is imported.

    :param threads: Number of threads, or ``None`` to keep the environment as it is.
    """
    if threads is None:
        return
    for variable in THREAD_VARIABLES:
        os.environ[variable] = str(threads)


def best_time(function: Callable[[], object], budget: float = TIME_BUDGET_SECONDS) -> tuple[float, int]:
    """
    Best wall time of ``function`` after one warm-up call, repeated within a time budget.

    :param function: Callable to time.
    :param budget: Seconds after which no new repeat starts.
    :return: The best time in seconds and the number of timed repeats.
    :rtype: tuple[float, int]
    """
    function()  # warm-up: lazy initialization, caches, kernel compilation
    best = float("inf")
    repeats = 0
    start = time.perf_counter()
    while repeats < MAX_REPEATS and (repeats == 0 or time.perf_counter() - start < budget):
        begin = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - begin)
        repeats += 1
    return best, repeats


def max_relative_difference(values, reference) -> float:
    """
    Largest ``|value - reference| / |reference|`` over a batch.

    :param values: Pfaffians to check, as a NumPy array of shape ``(batch,)``.
    :param reference: Reference Pfaffians, of the same shape.
    :return: The largest relative difference.
    :rtype: float
    """
    import numpy as np

    values = np.asarray(values)
    reference = np.asarray(reference)
    scale = np.maximum(np.abs(reference), np.finfo(np.float64).tiny)
    return float(np.max(np.abs(values - reference) / scale))


def table_rows(measurements: list[Measurement]) -> list[tuple[str, ...]]:
    """
    Formatted cells of the measurements, one row per measurement, with the speed-up over pfapack's C
    implementation.

    :param measurements: Measurements of every method on every input.
    :return: One tuple of cells per measurement, in the order of ``TABLE_HEADER``.
    :rtype: list[tuple[str, ...]]
    """
    reference = {(m.dimension, m.batch, m.dtype): m.seconds for m in measurements if m.method == REFERENCE_METHOD}
    rows = []
    for item in measurements:
        per_matrix = item.seconds / item.batch
        base = reference.get((item.dimension, item.batch, item.dtype))
        speedup = f"{base / item.seconds:.2f}x" if base else "n/a"
        difference = (
            "n/a"
            if item.max_relative_difference != item.max_relative_difference
            else (f"{item.max_relative_difference:.1e}")
        )
        rows.append(
            (
                str(item.dimension),
                str(item.batch),
                item.dtype,
                item.method,
                _format_seconds(per_matrix),
                speedup,
                difference,
            )
        )
    return rows


def render_table(measurements: list[Measurement]) -> str:
    """
    Markdown table of the measurements, with the speed-up over pfapack's C implementation.

    :param measurements: Measurements of every method on every input.
    :return: The table.
    :rtype: str
    """
    lines = ["| " + " | ".join(TABLE_HEADER) + " |", "|" + "---|" * len(TABLE_HEADER)]
    lines.extend("| " + " | ".join(row) + " |" for row in table_rows(measurements))
    return "\n".join(lines)


def render_terminal_table(measurements: list[Measurement]) -> str:
    """
    Box-drawn table of the measurements for the terminal, one block of rows per input, with the fastest method
    of each input marked with a star.

    :param measurements: Measurements of every method on every input.
    :return: The table.
    :rtype: str
    """
    rows = table_rows(measurements)
    groups: dict[tuple[int, int, str], list[int]] = {}
    for index, item in enumerate(measurements):
        groups.setdefault((item.dimension, item.batch, item.dtype), []).append(index)
    for indices in groups.values():
        fastest = min(indices, key=lambda index: measurements[index].seconds)
        cells = list(rows[fastest])
        cells[3] += FASTEST_MARKER
        rows[fastest] = tuple(cells)
    widths = [max(len(row[column]) for row in [TABLE_HEADER, *rows]) for column in range(len(TABLE_HEADER))]

    def border(left: str, middle: str, right: str) -> str:
        return left + middle.join("─" * (width + 2) for width in widths) + right

    lines = [border("┌", "┬", "┐"), _terminal_line(TABLE_HEADER, widths, ()), border("╞", "╪", "╡")]
    for group_index, indices in enumerate(groups.values()):
        if group_index:
            lines.append(border("├", "┼", "┤"))
        lines.extend(_terminal_line(rows[index], widths, RIGHT_ALIGNED_COLUMNS) for index in indices)
    lines.append(border("└", "┴", "┘"))
    return "\n".join(lines)


def _terminal_line(cells: Sequence[str], widths: Sequence[int], right_aligned: Sequence[int]) -> str:
    padded = [
        cell.rjust(width) if column in right_aligned else cell.ljust(width)
        for column, (cell, width) in enumerate(zip(cells, widths))
    ]
    return "│ " + " │ ".join(padded) + " │"


def _format_seconds(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.2f} s"


def _methods(matrices, pfapack_c, pfapack_python, torchpfaff=None, lrux=None) -> dict[str, Callable[[], object]]:
    import numpy as np
    import torch

    import torch_pfaffian

    arrays = [np.ascontiguousarray(matrix) for matrix in matrices.numpy()]
    methods: dict[str, Callable[[], object]] = {}
    if pfapack_c is not None:
        methods[REFERENCE_METHOD] = lambda: np.array([pfapack_c.pfaffian(array) for array in arrays])
    work = matrices.shape[-1] ** 3 * matrices.shape[0]
    if pfapack_python is not None and work <= PYTHON_PFAPACK_MAX_WORK:
        methods["pfapack (Python)"] = lambda: np.array([pfapack_python.pfaffian(array) for array in arrays])
    libraries: dict[str, Callable] = {"torchpfaffian": torch_pfaffian.pfaffian}
    if torchpfaff is not None:
        libraries["torchpfaff"] = lambda tensor: _torchpfaff_pfaffian(torchpfaff, tensor)
    devices = [("", matrices)]
    if torch.cuda.is_available():  # pragma: no cover - needs a GPU
        devices.append((" (CUDA)", matrices.cuda()))
    for suffix, on_device in devices:
        for name, function in libraries.items():
            if name == "torchpfaff" and suffix and matrices.shape[-1] > TORCHPFAFF_CUDA_MAX_DIMENSION:
                _report_once(
                    f"torchpfaff{suffix}", "", f"skipped above m={TORCHPFAFF_CUDA_MAX_DIMENSION} (Triton compile time)"
                )
                continue
            methods[f"{name}{suffix}"] = _forward_method(function, on_device)
            methods[f"{name} + gradient{suffix}"] = _gradient_method(function, on_device)
    if lrux is not None:
        methods.update(_lrux_methods(lrux, matrices.numpy()))
    return methods


def _lrux_methods(lrux, matrices) -> dict[str, Callable[[], object]]:
    # lrux computes full Pfaffians with fermix, jitted here; JAX's own CPU thread pool is not pinned by --threads.
    jax, pfaffian_function = lrux

    def loss(array):
        values = pfaffian_function(array)
        return values.real.sum(), values

    forward = jax.jit(pfaffian_function)
    gradient = jax.jit(jax.grad(loss, has_aux=True))
    devices = [("", jax.devices("cpu")[0])]
    try:
        devices.append((" (CUDA)", jax.devices("gpu")[0]))  # pragma: no cover - needs a GPU
    except RuntimeError:
        pass
    methods: dict[str, Callable[[], object]] = {}
    for suffix, device in devices:
        on_device = jax.device_put(matrices, device)
        methods[f"lrux{suffix}"] = _jax_forward_method(forward, on_device)
        methods[f"lrux + gradient{suffix}"] = _jax_gradient_method(gradient, on_device)
    return methods


def _jax_forward_method(forward: Callable, matrices) -> Callable[[], object]:
    import numpy as np

    return lambda: np.asarray(forward(matrices))  # np.asarray waits for the device


def _jax_gradient_method(gradient: Callable, matrices) -> Callable[[], object]:
    import numpy as np

    def with_gradient():
        matrix_gradient, values = gradient(matrices)
        matrix_gradient.block_until_ready()
        return np.asarray(values)

    return with_gradient


def _torchpfaff_pfaffian(torchpfaff, matrices):
    sign, log_magnitude = torchpfaff.log_pfaffian(matrices)
    return sign * log_magnitude.exp()


def _forward_method(function: Callable, matrices) -> Callable[[], object]:
    # .cpu() waits for the device, so CUDA times include the whole computation.
    return lambda: function(matrices).cpu().numpy()


def _gradient_method(function: Callable, matrices) -> Callable[[], object]:
    def with_gradient():
        leaf = matrices.clone().requires_grad_(True)
        result = function(leaf)
        result.real.sum().backward()
        leaf.grad.cpu()  # waits for the backward pass on CUDA
        return result.detach().cpu().numpy()

    return with_gradient


def measure(dimension: int, batch: int, dtype_name: str) -> list[Measurement]:
    """
    Time every available method (on CPU, and on CUDA when available) on one random batch and compare it with
    pfapack's C implementation.

    :param dimension: Matrix dimension ``m``.
    :param batch: Number of matrices.
    :param dtype_name: ``float64`` or ``complex128`` (pfapack computes in double precision).
    :return: One measurement per method that supports the input.
    :rtype: list[Measurement]
    """
    import torch
    from cases import general

    pfapack_c, pfapack_python = _import_pfapack()
    generator = torch.Generator().manual_seed(dimension * 1000 + batch)
    matrices = general(dimension, batch, getattr(torch, dtype_name), generator)
    methods = _methods(matrices, pfapack_c, pfapack_python, _import_torchpfaff(), _import_lrux())
    reference = methods[REFERENCE_METHOD]() if REFERENCE_METHOD in methods else None
    work = dimension**3 * batch
    measurements = []
    for name, function in methods.items():
        if name in SLOW_METHODS and work >= SLOW_METHODS[name]:
            _report_once(
                name, "", f"skipped from m^3 * batch = {SLOW_METHODS[name]} (over {MAX_CALL_SECONDS:g} s a call)"
            )
            continue
        print(f"  m={dimension} batch={batch} {dtype_name}: {name}", file=sys.stderr, flush=True)
        try:
            begin = time.perf_counter()
            values = function()
            first_call = time.perf_counter() - begin
        except Exception as error:  # a library that does not support this input is skipped, not fatal
            message = (str(error).strip().splitlines() or [""])[0]
            _report_once(name, dtype_name, f"{type(error).__name__}: {message}")
            continue
        difference = float("nan") if reference is None else max_relative_difference(values, reference)
        if first_call > MAX_CALL_SECONDS:
            SLOW_METHODS[name] = work
            seconds, repeats = first_call, 1  # includes any one-time compilation
        else:
            seconds, repeats = best_time(function)
        measurements.append(Measurement(dimension, batch, dtype_name, name, seconds, repeats, difference))
    return measurements


def _report_once(name: str, dtype_name: str, reason: str) -> None:
    if (name, dtype_name) in REPORTED_FAILURES:
        return
    REPORTED_FAILURES.add((name, dtype_name))
    where = f" on {dtype_name}" if dtype_name else ""
    print(f"skipped {name}{where}: {reason}", file=sys.stderr)


def _import_pfapack():
    try:
        import pfapack.ctypes as pfapack_c
        import pfapack.pfaffian as pfapack_python
    except ImportError:
        return None, None
    return pfapack_c, pfapack_python


def _import_torchpfaff():
    try:
        import torchpfaff
    except ImportError:
        return None
    return torchpfaff


def _import_lrux():
    # JAX would otherwise reserve 75% of the GPU memory, leaving little to torch.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    try:
        import jax
        import lrux  # noqa: F401
        from fermix import pf
    except ImportError:
        return None
    import warnings

    from fermix import FermixFallbackWarning

    jax.config.update("jax_enable_x64", True)  # pfapack computes in double precision
    # On CPU, fermix always falls back from its CUDA kernels to plain jax.numpy; the docstring says so once.
    warnings.filterwarnings("ignore", category=FermixFallbackWarning)
    return jax, pf


def save_results(path: Path, header: str, measurements: list[Measurement]) -> None:
    """
    Write the header and the measurements to a JSON file.

    :param path: Output file.
    :param header: Description of the libraries and the machine.
    :param measurements: Measurements so far.
    """
    path.write_text(json.dumps({"header": header, "measurements": [asdict(m) for m in measurements]}, indent=2))


def load_results(path: Path) -> tuple[str, list[Measurement]]:
    """
    Read a JSON file written by :func:`save_results`.

    :param path: Results file.
    :return: The header and the measurements.
    :rtype: tuple[str, list[Measurement]]
    """
    content = json.loads(path.read_text())
    return content["header"], [Measurement(**item) for item in content["measurements"]]


def _version(distribution: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not installed"


def _header(threads: str) -> str:
    import numpy as np
    import torch

    import torch_pfaffian

    gpu = "no GPU"
    if torch.cuda.is_available():  # pragma: no cover - needs a GPU
        gpu = f"GPU {torch.cuda.get_device_name()}, CUDA {torch.version.cuda}, triton {_version('triton')}"
    return (
        f"torchpfaffian {torch_pfaffian.__version__}, torch {torch.__version__}, numpy {np.__version__}, "
        f"pfapack {_version('pfapack')}, torchpfaff {_version('torchpfaff')}, lrux {_version('lrux')} "
        f"(fermix {_version('fermix')}, jax {_version('jax')}); Python {platform.python_version()} "
        f"on {platform.platform()}, {platform.processor() or platform.machine()}, {os.cpu_count()} cores, "
        f"threads: {threads}; {gpu}"
    )


def main() -> None:
    global MAX_CALL_SECONDS, TORCHPFAFF_CUDA_MAX_DIMENSION
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES), help="matrix dimensions m")
    parser.add_argument("--batches", type=int, nargs="+", default=list(DEFAULT_BATCHES), help="batch sizes")
    parser.add_argument("--dtypes", nargs="+", default=list(DEFAULT_DTYPES), choices=list(DEFAULT_DTYPES))
    parser.add_argument("--threads", type=int, default=None, help="threads of torch, MKL and the Rust kernel")
    parser.add_argument("--markdown", action="store_true", help="print a Markdown table instead of a terminal one")
    parser.add_argument("--render", type=Path, default=None, help="print the table of a saved results file and exit")
    parser.add_argument(
        "--max-call-seconds",
        type=float,
        default=MAX_CALL_SECONDS,
        help="a method slower than this on one call is skipped on larger inputs",
    )
    parser.add_argument(
        "--torchpfaff-cuda-max-size",
        type=int,
        default=TORCHPFAFF_CUDA_MAX_DIMENSION,
        help="largest m for torchpfaff on CUDA, whose Triton kernel compiles for minutes at large m",
    )
    arguments = parser.parse_args()
    MAX_CALL_SECONDS = arguments.max_call_seconds
    TORCHPFAFF_CUDA_MAX_DIMENSION = arguments.torchpfaff_cuda_max_size
    if arguments.render is not None:
        header, measurements = load_results(arguments.render)
        print(header)
        print(render_table(measurements) if arguments.markdown else render_terminal_table(measurements))
        return
    set_threads(arguments.threads)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    threads = "default" if arguments.threads is None else str(arguments.threads)
    header = _header(threads)
    if _import_pfapack()[0] is None:
        print("pfapack is not installed: no reference to compare with. See the module docstring.", file=sys.stderr)
    if _import_torchpfaff() is None:
        print("torchpfaff is not installed: it is not timed. See the module docstring.", file=sys.stderr)
    if _import_lrux() is None:
        print("lrux is not installed: it is not timed. See the module docstring.", file=sys.stderr)
    print(header, flush=True)
    RESULTS_DIRECTORY.mkdir(exist_ok=True)
    output = RESULTS_DIRECTORY / f"libraries-{time.strftime('%Y%m%d-%H%M%S')}.json"
    # Saved after every input, so that an interrupted run keeps its results (see --render).
    print(f"Raw results: {output}", file=sys.stderr)
    measurements: list[Measurement] = []
    for dtype_name in arguments.dtypes:
        for dimension in arguments.sizes:
            for batch in arguments.batches:
                measurements.extend(measure(dimension, batch, dtype_name))
                save_results(output, header, measurements)
                print(f"done m={dimension} batch={batch} {dtype_name}", file=sys.stderr, flush=True)
    print(render_table(measurements) if arguments.markdown else render_terminal_table(measurements))


if __name__ == "__main__":
    main()
