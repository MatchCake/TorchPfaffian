"""
Compare TorchPfaffian with pfapack, the reference Pfaffian library, on speed and agreement.

pfapack (https://github.com/basnijholt/pfapack) computes one NumPy matrix at a time, without gradients or GPU
support; its C implementation is the fast one. This script times both on the same random skew-symmetric
matrices (the ``general`` family of ``cases.py``, scaled so that ``|Pf|`` stays near 1) and checks that they
agree. pfapack is not a dependency of the project, so run the script from a separate environment, on a Python
version for which pfapack ships wheels (3.10 to 3.13 at the time of writing):

    uv venv --python 3.12 ~/.venvs/pfapack                       # pfapack has wheels up to Python 3.13
    uv pip install --python ~/.venvs/pfapack torch --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python ~/.venvs/pfapack pfapack .            # builds this package, Rust kernel included
    ~/.venvs/pfapack/bin/python benchmarks/pfapack_comparison.py --threads 1

Threads are set through ``OMP_NUM_THREADS``, ``MKL_NUM_THREADS`` and ``RAYON_NUM_THREADS`` before torch is
imported (never ``torch.set_num_threads``, which corrupts batched LU on some MKL builds; issue #23).
"""

import argparse
import json
import os
import platform
import sys
import time
from collections.abc import Callable
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


def render_table(measurements: list[Measurement]) -> str:
    """
    Markdown table of the measurements, with the speed-up over pfapack's C implementation.

    :param measurements: Measurements of every method on every input.
    :return: The table.
    :rtype: str
    """
    reference = {(m.dimension, m.batch, m.dtype): m.seconds for m in measurements if m.method == "pfapack (C)"}
    lines = [
        "| m | batch | dtype | method | time per matrix | speed-up over pfapack (C) | max rel. difference |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in measurements:
        per_matrix = item.seconds / item.batch
        base = reference.get((item.dimension, item.batch, item.dtype))
        speedup = f"{base / item.seconds:.2f}x" if base else "n/a"
        difference = (
            "n/a"
            if item.max_relative_difference != item.max_relative_difference
            else (f"{item.max_relative_difference:.1e}")
        )
        lines.append(
            f"| {item.dimension} | {item.batch} | {item.dtype} | {item.method} | {_format_seconds(per_matrix)} "
            f"| {speedup} | {difference} |"
        )
    return "\n".join(lines)


def _format_seconds(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.2f} s"


def _methods(matrices, pfapack_c, pfapack_python) -> dict[str, Callable[[], object]]:
    import numpy as np
    import torch

    import torch_pfaffian

    arrays = [np.ascontiguousarray(matrix) for matrix in matrices.numpy()]
    methods: dict[str, Callable[[], object]] = {}
    if pfapack_c is not None:
        methods["pfapack (C)"] = lambda: np.array([pfapack_c.pfaffian(array) for array in arrays])
    work = matrices.shape[-1] ** 3 * matrices.shape[0]
    if pfapack_python is not None and work <= PYTHON_PFAPACK_MAX_WORK:
        methods["pfapack (Python)"] = lambda: np.array([pfapack_python.pfaffian(array) for array in arrays])
    methods["torchpfaffian"] = lambda: torch_pfaffian.pfaffian(matrices).numpy()

    def with_gradient():
        leaf = matrices.clone().requires_grad_(True)
        result = torch_pfaffian.pfaffian(leaf)
        result.real.sum().backward()
        return result.detach().numpy()

    methods["torchpfaffian + gradient"] = with_gradient
    if torch.cuda.is_available():  # pragma: no cover - needs a GPU
        on_device = matrices.cuda()

        def on_cuda():
            result = torch_pfaffian.pfaffian(on_device)
            torch.cuda.synchronize()
            return result.cpu().numpy()

        methods["torchpfaffian (CUDA)"] = on_cuda
    return methods


def measure(dimension: int, batch: int, dtype_name: str) -> list[Measurement]:
    """
    Time every available method on one random batch and compare it with pfapack's C implementation.

    :param dimension: Matrix dimension ``m``.
    :param batch: Number of matrices.
    :param dtype_name: ``float64`` or ``complex128`` (pfapack computes in double precision).
    :return: One measurement per method.
    :rtype: list[Measurement]
    """
    import torch
    from cases import general

    pfapack_c, pfapack_python = _import_pfapack()
    generator = torch.Generator().manual_seed(dimension * 1000 + batch)
    matrices = general(dimension, batch, getattr(torch, dtype_name), generator)
    methods = _methods(matrices, pfapack_c, pfapack_python)
    reference = methods["pfapack (C)"]() if "pfapack (C)" in methods else None
    measurements = []
    for name, function in methods.items():
        values = function()
        difference = float("nan") if reference is None else max_relative_difference(values, reference)
        seconds, repeats = best_time(function)
        measurements.append(Measurement(dimension, batch, dtype_name, name, seconds, repeats, difference))
    return measurements


def _import_pfapack():
    try:
        import pfapack.ctypes as pfapack_c
        import pfapack.pfaffian as pfapack_python
    except ImportError:
        return None, None
    return pfapack_c, pfapack_python


def _header(threads: str) -> str:
    import numpy as np
    import torch

    import torch_pfaffian

    pfapack_c, _ = _import_pfapack()
    pfapack_version = "not installed"
    if pfapack_c is not None:
        from importlib.metadata import version

        pfapack_version = version("pfapack")
    return (
        f"torchpfaffian {torch_pfaffian.__version__}, torch {torch.__version__}, numpy {np.__version__}, "
        f"pfapack {pfapack_version}; Python {platform.python_version()} on {platform.platform()}, "
        f"{platform.processor() or platform.machine()}, {os.cpu_count()} cores, threads: {threads}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES), help="matrix dimensions m")
    parser.add_argument("--batches", type=int, nargs="+", default=list(DEFAULT_BATCHES), help="batch sizes")
    parser.add_argument("--dtypes", nargs="+", default=list(DEFAULT_DTYPES), choices=list(DEFAULT_DTYPES))
    parser.add_argument("--threads", type=int, default=None, help="threads of torch, MKL and the Rust kernel")
    arguments = parser.parse_args()
    set_threads(arguments.threads)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    threads = "default" if arguments.threads is None else str(arguments.threads)
    header = _header(threads)
    if _import_pfapack()[0] is None:
        print("pfapack is not installed: only TorchPfaffian is timed. See the module docstring.", file=sys.stderr)
    print(header, flush=True)
    measurements: list[Measurement] = []
    for dtype_name in arguments.dtypes:
        for dimension in arguments.sizes:
            for batch in arguments.batches:
                measurements.extend(measure(dimension, batch, dtype_name))
                print(f"done m={dimension} batch={batch} {dtype_name}", file=sys.stderr, flush=True)
    table = render_table(measurements)
    print(table)
    RESULTS_DIRECTORY.mkdir(exist_ok=True)
    output = RESULTS_DIRECTORY / f"pfapack-{time.strftime('%Y%m%d-%H%M%S')}.json"
    output.write_text(json.dumps({"header": header, "measurements": [asdict(m) for m in measurements]}, indent=2))
    print(f"\nRaw results: {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
