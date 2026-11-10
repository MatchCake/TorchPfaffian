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
saved (possibly interrupted) run. ``--ratios`` times Pfaffian ratios under low-rank updates instead (see
:func:`measure_ratios`). A method whose single call takes more than ``--max-call-seconds`` (10 s) is
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
# --ratios: number of successive accepted updates in the "row x16" case, and the delay of the delayed updates
# (lrux's tuned value for Pfaffians, arXiv:2602.05255).
STEPS = 16
DELAY = 4
UPDATES = ("rank 2", "rank 4", "row", f"row x{STEPS}")
DEFAULT_RATIO_SIZES = (32, 128, 512)
DEFAULT_RATIO_BATCHES = (1, 64)


@dataclass
class Measurement:
    """
    Timing and agreement of one method on one input.

    ``seconds`` is the best time over the repeats for the whole batch (for one update of the batch with
    ``--ratios``). ``max_relative_difference`` compares the method's Pfaffians (or ratios) with those of
    pfapack's C implementation (``nan`` when pfapack is not available). ``update`` names the low-rank update of
    a ``--ratios`` measurement, and is empty for a full Pfaffian.
    """

    dimension: int
    batch: int
    dtype: str
    method: str
    seconds: float
    repeats: int
    max_relative_difference: float
    update: str = ""


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
    :return: One tuple of cells per measurement, in the order of :func:`table_header`.
    :rtype: list[tuple[str, ...]]
    """
    with_update = _has_updates(measurements)
    reference = {_group(m): m.seconds for m in measurements if m.method == REFERENCE_METHOD}
    rows = []
    for item in measurements:
        per_matrix = item.seconds / item.batch
        base = reference.get(_group(item))
        speedup = f"{base / item.seconds:.2f}x" if base else "n/a"
        difference = (
            "n/a"
            if item.max_relative_difference != item.max_relative_difference
            else (f"{item.max_relative_difference:.1e}")
        )
        update = (item.update,) if with_update else ()
        rows.append(
            (
                str(item.dimension),
                str(item.batch),
                item.dtype,
                *update,
                item.method,
                _format_seconds(per_matrix),
                speedup,
                difference,
            )
        )
    return rows


def table_header(measurements: list[Measurement]) -> tuple[str, ...]:
    """
    Column names of :func:`table_rows`: ``TABLE_HEADER``, with an ``update`` column for ``--ratios`` runs.

    :param measurements: Measurements of every method on every input.
    :return: The column names.
    :rtype: tuple[str, ...]
    """
    if not _has_updates(measurements):
        return TABLE_HEADER
    return (*TABLE_HEADER[:3], "update", *TABLE_HEADER[3:])


def _has_updates(measurements: list[Measurement]) -> bool:
    return any(item.update for item in measurements)


def _group(item: Measurement) -> tuple[int, int, str, str]:
    return item.dimension, item.batch, item.dtype, item.update


def render_table(measurements: list[Measurement]) -> str:
    """
    Markdown table of the measurements, with the speed-up over pfapack's C implementation.

    :param measurements: Measurements of every method on every input.
    :return: The table.
    :rtype: str
    """
    header = table_header(measurements)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
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
    header = table_header(measurements)
    shift = len(header) - len(TABLE_HEADER)  # 1 with the update column
    method_column = 3 + shift
    right_aligned = tuple(column + shift if column > 2 else column for column in RIGHT_ALIGNED_COLUMNS)
    rows = table_rows(measurements)
    groups: dict[tuple[int, int, str, str], list[int]] = {}
    for index, item in enumerate(measurements):
        groups.setdefault(_group(item), []).append(index)
    for indices in groups.values():
        fastest = min(indices, key=lambda index: measurements[index].seconds)
        cells = list(rows[fastest])
        cells[method_column] += FASTEST_MARKER
        rows[fastest] = tuple(cells)
    widths = [max(len(row[column]) for row in [header, *rows]) for column in range(len(header))]

    def border(left: str, middle: str, right: str) -> str:
        return left + middle.join("─" * (width + 2) for width in widths) + right

    lines = [border("┌", "┬", "┐"), _terminal_line(header, widths, ()), border("╞", "╪", "╡")]
    for group_index, indices in enumerate(groups.values()):
        if group_index:
            lines.append(border("├", "┼", "┤"))
        lines.extend(_terminal_line(rows[index], widths, right_aligned) for index in indices)
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
    return _time_methods(methods, reference, dimension, batch, dtype_name)


def measure_ratios(dimension: int, batch: int, dtype_name: str, update: str) -> list[Measurement]:
    """
    Time Pfaffian ratios under one kind of low-rank update (``--ratios``), and compare them with pfapack.

    Every method starts from the same random matrices ``A`` (the ``general`` family) and the inverse ``A^{-1}``,
    which a Monte Carlo loop keeps from one step to the next:

    - ``"rank 2"`` and ``"rank 4"``: one dense update ``A - u J u^T`` with ``u`` of shape ``(n, 2)`` or
      ``(n, 4)``;
    - ``"row"``: one row and column replaced (the typical Monte Carlo move);
    - ``f"row x{STEPS}"``: ``STEPS`` successive row replacements, each accepted, with the immediate update of
      the inverse and with delayed updates (``max_delay = DELAY``). Times are per update.

    The reference, ``pfapack (C)``, forms each updated matrix and computes its Pfaffian, divided by the cached
    Pfaffian of ``A``: what a caller without low-rank updates does.

    :param dimension: Matrix dimension ``n``.
    :param batch: Number of matrices.
    :param dtype_name: ``float64`` or ``complex128``.
    :param update: One of ``UPDATES``.
    :return: One measurement per method that supports the input.
    :rtype: list[Measurement]
    """
    import torch
    from cases import general

    pfapack_c, _ = _import_pfapack()
    generator = torch.Generator().manual_seed(dimension * 1000 + batch)
    matrices = general(dimension, batch, getattr(torch, dtype_name), generator)
    problem = _ratio_problem(matrices, update, generator)
    methods = _ratio_methods(problem, pfapack_c, _import_lrux())
    reference = methods[REFERENCE_METHOD]() if REFERENCE_METHOD in methods else None
    divisor = STEPS if problem["steps"] else 1
    return _time_methods(methods, reference, dimension, batch, dtype_name, update, divisor)


def _time_methods(
    methods: dict[str, Callable[[], object]],
    reference,
    dimension: int,
    batch: int,
    dtype_name: str,
    update: str = "",
    divisor: int = 1,
) -> list[Measurement]:
    work = dimension**3 * batch
    label = f"m={dimension} batch={batch} {dtype_name}" + (f" {update}" if update else "")
    measurements = []
    for name, function in methods.items():
        key = f"{name} [{update}]" if update else name
        if key in SLOW_METHODS and work >= SLOW_METHODS[key]:
            _report_once(
                key, "", f"skipped from m^3 * batch = {SLOW_METHODS[key]} (over {MAX_CALL_SECONDS:g} s a call)"
            )
            continue
        print(f"  {label}: {name}", file=sys.stderr, flush=True)
        try:
            begin = time.perf_counter()
            values = function()
            first_call = time.perf_counter() - begin
        except Exception as error:  # a library that does not support this input is skipped, not fatal
            message = (str(error).strip().splitlines() or [""])[0]
            _report_once(key, dtype_name, f"{type(error).__name__}: {message}")
            continue
        difference = float("nan") if reference is None else max_relative_difference(values, reference)
        if first_call > MAX_CALL_SECONDS:
            SLOW_METHODS[key] = work
            seconds, repeats = first_call, 1  # includes any one-time compilation
        else:
            seconds, repeats = best_time(function)
        measurements.append(
            Measurement(dimension, batch, dtype_name, name, seconds / divisor, repeats, difference, update)
        )
    return measurements


def _ratio_problem(matrices, update: str, generator) -> dict:
    # The inputs of one --ratios case, on CPU: the inverse, the updates and the updated matrices.
    import torch
    from cases import _general_scale, _random

    from torch_pfaffian import pfaffian, skew_identity

    batch, dimension = matrices.shape[0], matrices.shape[-1]
    scale = _general_scale(dimension)
    inverse = torch.linalg.inv(matrices)
    inverse = 0.5 * (inverse - inverse.mT)
    problem = {"matrices": matrices, "inverse": inverse, "initial": pfaffian(matrices), "steps": False}
    if update.startswith("rank"):
        rank = int(update.split()[1])
        vectors = _random((batch, dimension, rank), matrices.dtype, generator, scale**0.5)
        identity = skew_identity(rank // 2, dtype=matrices.dtype)
        problem.update(kind="dense", update=vectors, updated=matrices - vectors @ identity @ vectors.mT)
        return problem
    count = STEPS if update == f"row x{STEPS}" else 1
    rows = torch.arange(batch)
    indices, changes = [], []
    for _ in range(count):
        index = torch.randint(0, dimension, (batch,), generator=generator)
        change = _random((batch, dimension), matrices.dtype, generator, scale)
        change[rows, index] = 0
        indices.append(index)
        changes.append(change)
    problem.update(kind="row", indices=indices, changes=changes, steps=count > 1)
    return problem


def _replace_rows(matrices, index, change, rows) -> None:
    # In place: row index of each matrix gains change, and its column loses it (a NumPy or torch batch).
    matrices[rows, index, :] += change
    matrices[rows, :, index] -= change


def _ratio_methods(problem: dict, pfapack_c, lrux) -> dict[str, Callable[[], object]]:
    import numpy as np
    import torch

    methods: dict[str, Callable[[], object]] = {}
    if pfapack_c is not None:
        matrices = problem["matrices"].numpy()
        initial = np.array([pfapack_c.pfaffian(np.ascontiguousarray(array)) for array in matrices])
        if problem["kind"] == "dense":
            updated = [np.ascontiguousarray(array) for array in problem["updated"].numpy()]

            def new_pfaffians():
                return np.array([pfapack_c.pfaffian(array) for array in updated]) / initial

        else:
            steps = [(index.numpy(), change.numpy()) for index, change in zip(problem["indices"], problem["changes"])]
            rows = np.arange(matrices.shape[0])

            def new_pfaffians():
                # Form each updated matrix and compute its Pfaffian; the last over the first is the product of
                # the ratios.
                current = matrices.copy()
                for index, change in steps:
                    _replace_rows(current, index, change, rows)
                    last = np.array([pfapack_c.pfaffian(np.ascontiguousarray(array)) for array in current])
                return last / initial

        methods[REFERENCE_METHOD] = new_pfaffians
    devices = ["cpu"]
    if torch.cuda.is_available():  # pragma: no cover - needs a GPU
        devices.append("cuda")
    for device in devices:
        suffix = " (CUDA)" if device == "cuda" else ""
        methods.update({f"{name}{suffix}": method for name, method in _torch_ratio_methods(problem, device).items()})
    if lrux is not None:
        methods.update(_lrux_ratio_methods(problem, lrux))
    return methods


def _torch_ratio_methods(problem: dict, device: str) -> dict[str, Callable[[], object]]:
    import torch

    from torch_pfaffian import (
        PfaffianUpdater,
        pfaffian,
        pfaffian_ratio,
        pfaffian_ratio_and_inverse,
        pfaffian_ratio_replace,
        pfaffian_ratio_replace_and_inverse,
    )

    inverse = problem["inverse"].to(device)
    initial = problem["initial"].to(device)
    methods: dict[str, Callable[[], object]] = {}
    if problem["kind"] == "dense":
        updated = problem["updated"].to(device)
        methods["torchpfaffian, new Pfaffian"] = lambda: (pfaffian(updated) / initial).cpu().numpy()
        vectors = problem["update"].to(device)
        methods["torchpfaffian ratio"] = lambda: pfaffian_ratio(inverse, vectors).cpu().numpy()
        methods["torchpfaffian ratio + inverse"] = lambda: pfaffian_ratio_and_inverse(inverse, vectors)[0].cpu().numpy()
        return methods
    indices = [index.to(device) for index in problem["indices"]]
    changes = [change.to(device) for change in problem["changes"]]
    matrices = problem["matrices"].to(device)
    rows = torch.arange(matrices.shape[0], device=device)

    def new_pfaffians():
        current = matrices.clone()
        for index, change in zip(indices, changes):
            _replace_rows(current, index, change, rows)
            last = pfaffian(current)
        return (last / initial).cpu().numpy()

    methods["torchpfaffian, new Pfaffian"] = new_pfaffians
    if not problem["steps"]:
        methods["torchpfaffian ratio"] = lambda: pfaffian_ratio_replace(inverse, indices[0], changes[0]).cpu().numpy()
        methods["torchpfaffian ratio + inverse"] = lambda: (
            pfaffian_ratio_replace_and_inverse(inverse, indices[0], changes[0])[0].cpu().numpy()
        )
        return methods

    def immediate():
        current, product = inverse, None
        for index, change in zip(indices, changes):
            ratio, current = pfaffian_ratio_replace_and_inverse(current, index, change)
            product = ratio if product is None else product * ratio
        return product.cpu().numpy()

    def delayed():
        updater = PfaffianUpdater.from_inverse(inverse, max_delay=DELAY)
        for index, change in zip(indices, changes):
            updater.accept_replace(index, change)
        return updater.pfaffian.cpu().numpy()

    methods["torchpfaffian, immediate updates"] = immediate
    methods[f"torchpfaffian, delayed updates (T={DELAY})"] = delayed
    return methods


def _lrux_ratio_methods(problem: dict, lrux) -> dict[str, Callable[[], object]]:
    from lrux import PfCarrier, merge_pf_delays, pf_lru, pf_lru_delayed

    jax, _ = lrux
    numpy = jax.numpy
    devices = [("", jax.devices("cpu")[0])]
    try:
        devices.append((" (CUDA)", jax.devices("gpu")[0]))  # pragma: no cover - needs a GPU
    except RuntimeError:
        pass
    methods: dict[str, Callable[[], object]] = {}
    for suffix, device in devices:

        def put(tensor, device=device):
            return jax.device_put(tensor.numpy(), device)

        inverse = put(problem["inverse"])
        if problem["kind"] == "dense":
            vectors = put(problem["update"])
            ratio = jax.jit(jax.vmap(pf_lru))
            with_inverse = jax.jit(jax.vmap(lambda a, u: pf_lru(a, u, True)))
            methods[f"lrux ratio{suffix}"] = _jax_call(lambda r=ratio, a=inverse, u=vectors: r(a, u))
            methods[f"lrux ratio + inverse{suffix}"] = _jax_call(
                lambda f=with_inverse, a=inverse, u=vectors: f(a, u)[0]
            )
            continue
        indices = [put(index) for index in problem["indices"]]
        changes = [put(change) for change in problem["changes"]]
        ratio = jax.jit(jax.vmap(lambda a, x, e: pf_lru(a, (x, e))))
        with_inverse = jax.jit(jax.vmap(lambda a, x, e: pf_lru(a, (x, e), True)))
        if not problem["steps"]:
            methods[f"lrux ratio{suffix}"] = _jax_call(
                lambda r=ratio, a=inverse, x=changes[0], e=indices[0]: r(a, x, e)
            )
            methods[f"lrux ratio + inverse{suffix}"] = _jax_call(
                lambda f=with_inverse, a=inverse, x=changes[0], e=indices[0]: f(a, x, e)[0]
            )
            continue

        def immediate(f=with_inverse, a=inverse, xs=changes, es=indices):
            product = None
            for x, e in zip(xs, es):
                ratio_value, a = f(a, x, e)
                product = ratio_value if product is None else product * ratio_value
            return product

        steps = {
            delay: jax.jit(
                jax.vmap(lambda c, x, e, t=delay: pf_lru_delayed(c, (x, e), return_update=True, current_delay=t))
            )
            for delay in range(DELAY)
        }
        merge = jax.jit(jax.vmap(merge_pf_delays))
        batch, dimension = problem["inverse"].shape[0], problem["inverse"].shape[-1]

        def delayed(steps=steps, merge=merge, a=inverse, xs=changes, es=indices, device=device):
            zeros = numpy.zeros((batch, DELAY, dimension, 2), a.dtype, device=device)
            carrier = PfCarrier(a, zeros, numpy.zeros((batch, DELAY, 2, 2), a.dtype, device=device))
            product = None
            for step, (x, e) in enumerate(zip(xs, es)):
                ratio_value, carrier = steps[step % DELAY](carrier, x, e)
                if step % DELAY == DELAY - 1:
                    carrier = merge(carrier)
                product = ratio_value if product is None else product * ratio_value
            return product

        methods[f"lrux, immediate updates{suffix}"] = _jax_call(immediate)
        methods[f"lrux, delayed updates (T={DELAY}){suffix}"] = _jax_call(delayed)
    return methods


def _jax_call(function: Callable[[], object]) -> Callable[[], object]:
    import numpy as np

    return lambda: np.asarray(function())  # np.asarray waits for the device


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
    parser.add_argument("--sizes", type=int, nargs="+", default=None, help="matrix dimensions m")
    parser.add_argument("--batches", type=int, nargs="+", default=None, help="batch sizes")
    parser.add_argument("--ratios", action="store_true", help="time Pfaffian ratios under low-rank updates")
    parser.add_argument("--updates", nargs="+", default=list(UPDATES), choices=list(UPDATES), help="with --ratios")
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
    sizes = arguments.sizes or list(DEFAULT_RATIO_SIZES if arguments.ratios else DEFAULT_SIZES)
    batches = arguments.batches or list(DEFAULT_RATIO_BATCHES if arguments.ratios else DEFAULT_BATCHES)
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
    prefix = "ratios" if arguments.ratios else "libraries"
    output = RESULTS_DIRECTORY / f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    # Saved after every input, so that an interrupted run keeps its results (see --render).
    print(f"Raw results: {output}", file=sys.stderr)
    measurements: list[Measurement] = []
    updates = arguments.updates if arguments.ratios else [""]
    for dtype_name in arguments.dtypes:
        for update in updates:
            for dimension in sizes:
                for batch in batches:
                    if update:
                        measurements.extend(measure_ratios(dimension, batch, dtype_name, update))
                    else:
                        measurements.extend(measure(dimension, batch, dtype_name))
                    save_results(output, header, measurements)
                    print(f"done m={dimension} batch={batch} {dtype_name} {update}", file=sys.stderr, flush=True)
    print(render_table(measurements) if arguments.markdown else render_terminal_table(measurements))


if __name__ == "__main__":
    main()
