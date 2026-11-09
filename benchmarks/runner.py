import argparse
import ctypes
import json
import signal
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch
from cases import BenchmarkCase, build_cases, make_matrix

# Per-mode timing budget: minimum total run time per case (s), probe time above which a case is timed once
# only (s), and the time limit of each phase of a case (s): the warm-up and timing, then the memory run.
TIMING_BUDGET = {
    "small": (0.05, 2.0, 30),
    "medium": (0.2, 5.0, 120),
    "big": (0.5, 20.0, 600),
}
MIN_REPEATS = 5
# Status of the marker line written before each case runs.
STARTED = "started"
MAX_REPEATS = 50
# Number of fixed random projections of the gradient kept for the agreement check.
GRADIENT_PROJECTIONS = 3


class CaseTimeout(Exception):
    """
    Raised by the alarm handler when a case exceeds its hard time limit.
    """


def _on_alarm(signum: int, frame: object) -> None:
    raise CaseTimeout


def _set_alarm(seconds: int) -> None:
    # SIGALRM only exists on POSIX; elsewhere compare.py's watchdog alone bounds a case. The alarm is also
    # deferred while a single C call runs, which the watchdog covers too.
    if hasattr(signal, "SIGALRM"):
        signal.alarm(seconds)


def _synchronize(device: str) -> None:
    if device == "cuda":  # pragma: no cover
        torch.cuda.synchronize()


def _operation(case: BenchmarkCase, matrix: torch.Tensor) -> Callable[[], torch.Tensor | None]:
    import torch_pfaffian

    keyword_arguments = {"sign": case.sign}
    if case.masked:
        keyword_arguments["mask_roundoff_gradient"] = True

    def run() -> torch.Tensor | None:
        leaf = matrix.detach().requires_grad_(case.backward)
        result = torch_pfaffian.pfaffian(leaf, **keyword_arguments)
        if not case.backward:
            return result
        (gradient,) = torch.autograd.grad(result.real.sum(), leaf)
        return gradient

    return run


def _time(run: Callable[[], object], device: str, bench: str) -> tuple[float, float, int, bool]:
    min_run_time, single_run_above, _ = TIMING_BUDGET[bench]
    start = time.perf_counter()
    run()
    _synchronize(device)
    probe = time.perf_counter() - start
    if probe > single_run_above:
        return probe, probe, 1, True
    samples = []
    while (sum(samples) < min_run_time or len(samples) < MIN_REPEATS) and len(samples) < MAX_REPEATS:
        start = time.perf_counter()
        run()
        _synchronize(device)
        samples.append(time.perf_counter() - start)
    # The minimum is the least disturbed run: interference from other processes only ever adds time. The
    # median is kept too, so a change is reported only when both move.
    return min(samples), statistics.median(samples), len(samples), False


def _read_status(key: str) -> int | None:
    try:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith(key):
                    return int(line.split()[1]) * 1024
    except OSError:  # pragma: no cover
        return None
    return None  # pragma: no cover


def _reset_peak_memory() -> bool:
    # Linux only: release freed heap pages, then reset the peak resident set size (VmHWM).
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        with open("/proc/self/clear_refs", "w") as clear_refs:
            clear_refs.write("5")
        return True
    except OSError:  # pragma: no cover
        return False


def _peak_memory(run: Callable[[], object], device: str) -> int | None:
    if device == "cuda":  # pragma: no cover
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.memory_allocated()
        run()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - start
    if not _reset_peak_memory():
        return None  # pragma: no cover
    start = _read_status("VmRSS:")
    run()
    peak = _read_status("VmHWM:")
    if start is None or peak is None:
        return None  # pragma: no cover
    return max(peak - start, 0)


def _summary(output: torch.Tensor, case: BenchmarkCase) -> dict[str, list[float]]:
    # Values are kept whole (they are per-batch scalars); gradients are reduced to a few fixed random
    # projections, which detect any change of the gradient while keeping the result file small.
    output = output.detach().to("cpu", torch.complex128)
    if not case.backward:
        return {"real": output.real.flatten().tolist(), "imag": output.imag.flatten().tolist()}
    generator = torch.Generator().manual_seed(case.seed)
    projections = [
        complex((output * torch.randn(output.shape, dtype=torch.float64, generator=generator)).sum())
        for _ in range(GRADIENT_PROJECTIONS)
    ]
    norm = float(torch.linalg.vector_norm(output))  # scaled internally, so it does not overflow
    return {"real": [value.real for value in projections] + [norm], "imag": [value.imag for value in projections] + [0]}


def run_case(case: BenchmarkCase, bench: str) -> dict[str, object]:
    """
    Time a case, measure its peak memory and summarize its output.

    :param case: The benchmark case.
    :param bench: Benchmark mode, which sets the timing budget.
    :return: A JSON-serializable record with ``status`` (``ok``, ``error``, ``timeout`` or
        ``unsupported``), and for ``ok`` the minimum ``seconds`` and ``median_seconds`` over the repeats,
        ``repeats``, ``single_run``, ``peak_bytes`` and the output ``summary``.
    :rtype: dict[str, object]
    """
    record: dict[str, object] = {"identifier": case.identifier}
    matrix = make_matrix(case)
    run = _operation(case, matrix)
    phase_limit = TIMING_BUDGET[bench][2]
    try:
        _set_alarm(phase_limit)
        output = run()
        _synchronize(case.device)
        seconds, median_seconds, repeats, single_run = _time(run, case.device, bench)
    except CaseTimeout:
        record["status"] = "timeout"
        return record
    except TypeError as error:
        # An older version without the keyword (e.g. mask_roundoff_gradient) cannot run the case.
        record["status"] = "unsupported" if "unexpected keyword argument" in str(error) else "error"
        record["message"] = f"TypeError: {str(error)[:200]}"
        return record
    except Exception as error:
        record["status"] = "error"
        record["message"] = f"{type(error).__name__}: {str(error)[:200]}"
        return record
    finally:
        _set_alarm(0)
    try:
        # The memory run has its own limit; a timeout there keeps the timing and only loses the memory.
        _set_alarm(phase_limit)
        peak_bytes = _peak_memory(run, case.device)
    except CaseTimeout:
        peak_bytes = None
    finally:
        _set_alarm(0)
    record.update(status="ok", seconds=seconds, median_seconds=median_seconds, repeats=repeats)
    record.update(single_run=single_run, peak_bytes=peak_bytes)
    record["summary"] = _summary(output, case) if output is not None else None
    return record


def main() -> None:
    """
    Run the benchmark cases of one version, appending one JSON record per line as each case finishes.

    The version under test is whichever ``torch_pfaffian`` is first on ``sys.path``; ``--expect-root``
    guards against silently benchmarking another installed copy. Writing per case lets ``compare.py``
    keep the finished cases and resume after the next one when a case crashes the process.
    """
    parser = argparse.ArgumentParser(description="Benchmark one torch_pfaffian version (internal to compare.py).")
    parser.add_argument("--bench", choices=list(TIMING_BUDGET), required=True)
    parser.add_argument("--devices", nargs="+", required=True)
    parser.add_argument("--families", nargs="+", required=True)
    parser.add_argument("--expect-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", type=Path, default=None, help="file listing the case identifiers to run")
    arguments = parser.parse_args()

    import torch_pfaffian

    package_root = Path(torch_pfaffian.__file__).resolve().parent.parent
    if package_root != arguments.expect_root.resolve():
        sys.exit(f"benchmarking the wrong torch_pfaffian: {torch_pfaffian.__file__}")
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _on_alarm)
    cases = build_cases(arguments.bench, arguments.devices, arguments.families)
    if arguments.only is not None:
        selected = set(arguments.only.read_text().split())
        cases = [case for case in cases if case.identifier in selected]
    with open(arguments.output, "a") as output:
        for case in cases:
            # The "started" marker lets compare.py name the case that crashed or hung the process.
            output.write(json.dumps({"identifier": case.identifier, "status": STARTED}) + "\n")
            output.flush()
            output.write(json.dumps(run_case(case, arguments.bench)) + "\n")
            output.flush()


if __name__ == "__main__":
    main()
