import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch
from cases import FAMILIES, build_cases
from runner import STARTED, TIMING_BUDGET

BENCHMARK_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = BENCHMARK_DIRECTORY.parent
RESULTS_DIRECTORY = BENCHMARK_DIRECTORY / "results"
CACHE_DIRECTORY = BENCHMARK_DIRECTORY / ".cache"
# A ratio is reported once it exceeds 1 + threshold; tiny absolute differences are timing noise.
DEFAULT_THRESHOLD = 0.25
# Below about 100 µs the best time of identical processes still varies by up to 3x on a busy machine.
TIME_NOISE_FLOOR_SECONDS = 100e-6
MEMORY_NOISE_FLOOR_BYTES = 2 << 20  # allocator caching moves the peak RSS by about 1 MiB
# Relative tolerance of the before/now agreement check, by real precision.
AGREEMENT_RTOL = {"float32": 1e-3, "complex64": 1e-3, "float64": 1e-8, "complex128": 1e-8}
DEFAULT_AGREEMENT_RTOL = 1e-3
# In these families the Pfaffian is zero up to round-off, so values and magnitude gradients are noise that
# legitimately changes with the order of operations; only the exact signed gradient is compared.
ROUNDOFF_FAMILIES = ("general_singular", "block_singular")
ROUNDOFF_COMPARED_OPERATIONS = ("signed+grad",)
FAILING_STATUSES = ("error", "timeout", "crashed", "missing")
FLAG_ORDER = ["error", "disagree", "slower", "heavier", "fixed", "faster", "lighter", "failing"]
REGRESSION_FLAGS = {"error", "disagree", "slower", "heavier"}
# A hung child gets this much time beyond the per-phase limits of its current case before it is killed.
WATCHDOG_GRACE_SECONDS = 60
LOCAL_COMMAND = "python benchmarks/compare.py --baseline {baseline}"


@dataclass
class ComparisonRow:
    """
    Before/now comparison of one benchmark case.

    ``flags`` lists what changed: ``error`` (the current version fails, crashes, times out or lacks a case the
    baseline runs), ``disagree`` (values or gradients differ beyond :data:`AGREEMENT_RTOL`), ``slower`` /
    ``faster`` (time ratio past the threshold), ``heavier`` / ``lighter`` (peak-memory ratio past the
    threshold), ``fixed`` (the baseline fails and the current version works) and ``failing`` (both fail).
    Only :data:`REGRESSION_FLAGS` count as regressions.
    """

    identifier: str
    before: dict[str, object] | None
    now: dict[str, object] | None
    time_ratio: float | None
    memory_ratio: float | None
    flags: list[str]


def _ratio(before: float | None, now: float | None) -> float | None:
    if before is None or now is None:
        return None
    if before == 0:
        return 1.0 if now == 0 else float("inf")
    return now / before


def _change_flags(
    before: float | None, now: float | None, noise_floor: float, threshold: float, names: tuple[str, str]
) -> list[str]:
    # Flags (worse, better) when the ratio passes the threshold and the absolute change is not noise.
    ratio = _ratio(before, now)
    if ratio is None or before is None or now is None or abs(now - before) < noise_floor:
        return []
    if ratio > 1 + threshold:
        return [names[0]]
    if ratio < 1 / (1 + threshold):
        return [names[1]]
    return []


def _summary_values(summary: dict[str, list[float]]) -> torch.Tensor:
    real = torch.tensor(summary["real"], dtype=torch.float64)
    imag = torch.tensor(summary["imag"], dtype=torch.float64)
    return torch.complex(real, imag)


def _non_finite_pattern(values: torch.Tensor) -> torch.Tensor:
    # NaN and +-inf flags of the real and imaginary parts, so inf and -inf or NaN and inf do not match.
    parts = (values.real, values.imag)
    return torch.stack([flag for part in parts for flag in (part.isnan(), part == torch.inf, part == -torch.inf)])


def _agree(before: dict[str, list[float]] | None, now: dict[str, list[float]] | None, rtol: float) -> bool:
    # Compared in float64: the non-finite entries must match exactly (same positions, same inf), and the
    # finite ones must agree relative to the largest finite magnitude of either version.
    if before is None or now is None:
        return before is None and now is None
    before_values, now_values = _summary_values(before), _summary_values(now)
    if before_values.shape != now_values.shape:
        return False
    before_finite, now_finite = torch.isfinite(before_values), torch.isfinite(now_values)
    if not torch.equal(before_finite, now_finite):
        return False
    if not torch.equal(_non_finite_pattern(before_values), _non_finite_pattern(now_values)):
        return False
    finite_before, finite_now = before_values[before_finite], now_values[before_finite]
    if finite_before.numel() == 0:
        return True
    scale = torch.maximum(finite_before.abs().amax(), finite_now.abs().amax())
    if scale == 0:
        return True  # both all zero
    return bool(torch.allclose(finite_now / scale, finite_before / scale, rtol=rtol, atol=rtol))


def _identifier_fields(identifier: str) -> tuple[str, str, str]:
    # (family, dtype name, operation) of "family/m=../b=../dtype/device/operation"; tolerant of odd strings.
    parts = identifier.split("/")
    family = parts[0]
    dtype_name = parts[3] if len(parts) > 3 else ""
    operation = parts[-1] if len(parts) > 1 else ""
    return family, dtype_name, operation


def _agreement_is_checked(identifier: str) -> bool:
    family, _, operation = _identifier_fields(identifier)
    return family not in ROUNDOFF_FAMILIES or operation in ROUNDOFF_COMPARED_OPERATIONS


def compare_records(
    before: list[dict[str, object]], now: list[dict[str, object]], threshold: float = DEFAULT_THRESHOLD
) -> list[ComparisonRow]:
    """
    Join the baseline and current records on their identifier and flag the changes.

    :param before: Records of the baseline version, as written by ``runner.py``.
    :param now: Records of the current version.
    :param threshold: Relative change of time or peak memory that is reported (``0.25`` reports ratios
        above ``1.25`` or below ``1 / 1.25``).
    :return: One row per case present in either version, in the order of ``now`` then ``before``.
    :rtype: list[ComparisonRow]
    """
    before_by_identifier = {str(record["identifier"]): record for record in before}
    now_by_identifier = {str(record["identifier"]): record for record in now}
    identifiers = list(now_by_identifier) + [key for key in before_by_identifier if key not in now_by_identifier]
    rows = []
    for identifier in identifiers:
        old, new = before_by_identifier.get(identifier), now_by_identifier.get(identifier)
        old_status = old["status"] if old else "missing"
        new_status = new["status"] if new else "missing"
        flags: list[str] = []
        time_ratio = memory_ratio = None
        if old_status == "unsupported":
            # A feature the baseline lacks (e.g. a new keyword) is not a regression, unless it fails now.
            flags = [] if new_status in ("ok", "unsupported") else ["error"]
        elif new_status == "unsupported":
            flags = ["error"]  # the current version dropped a feature the baseline supports
        elif old_status in FAILING_STATUSES and new_status in FAILING_STATUSES:
            flags = ["failing"]
        elif new_status in FAILING_STATUSES:
            flags = ["error"]
        elif old_status in FAILING_STATUSES:
            flags = ["fixed"]
        else:
            assert old is not None and new is not None
            old_seconds, new_seconds = old["seconds"], new["seconds"]
            old_median, new_median = old.get("median_seconds", old_seconds), new.get("median_seconds", new_seconds)
            old_bytes, new_bytes = old.get("peak_bytes"), new.get("peak_bytes")
            time_ratio = _ratio(old_seconds, new_seconds)  # type: ignore[arg-type]
            memory_ratio = _ratio(old_bytes, new_bytes)  # type: ignore[arg-type]
            rtol = AGREEMENT_RTOL.get(_identifier_fields(identifier)[1], DEFAULT_AGREEMENT_RTOL)
            if _agreement_is_checked(identifier) and not _agree(old.get("summary"), new.get("summary"), rtol):  # type: ignore[arg-type]
                flags.append("disagree")
            # A time change is reported only when the minimum and the median move the same way.
            minimum_flags = _change_flags(
                old_seconds, new_seconds, TIME_NOISE_FLOOR_SECONDS, threshold, ("slower", "faster")
            )  # type: ignore[arg-type]
            median_flags = _change_flags(
                old_median, new_median, TIME_NOISE_FLOOR_SECONDS, threshold, ("slower", "faster")
            )  # type: ignore[arg-type]
            flags += [flag for flag in minimum_flags if flag in median_flags]
            flags += _change_flags(old_bytes, new_bytes, MEMORY_NOISE_FLOOR_BYTES, threshold, ("heavier", "lighter"))  # type: ignore[arg-type]
        rows.append(ComparisonRow(identifier, old, new, time_ratio, memory_ratio, flags))
    return rows


def merge_rounds(rounds: list[list[dict[str, object]]]) -> list[dict[str, object]]:
    """
    Merge the records of several rounds of one version, keeping each case's least disturbed timing.

    A case is ``ok`` only if it is ``ok`` in every round (otherwise the first failing record is kept);
    its ``seconds``, ``median_seconds`` and ``peak_bytes`` are the minima over the rounds, and its output
    summary comes from the first round.

    :param rounds: Records per round, each in the order written by ``runner.py``.
    :return: One merged record per case.
    :rtype: list[dict[str, object]]
    """
    merged: dict[str, dict[str, object]] = {}
    for records in rounds:
        for record in records:
            identifier = str(record["identifier"])
            current = merged.get(identifier)
            if current is None:
                merged[identifier] = dict(record)
            elif current["status"] == "ok" and record["status"] != "ok":
                merged[identifier] = dict(record)
            elif current["status"] == "ok":
                for key in ("seconds", "median_seconds"):
                    current[key] = min(float(current[key]), float(record[key]))  # type: ignore[arg-type]
                peaks = [value for value in (current.get("peak_bytes"), record.get("peak_bytes")) if value is not None]
                current["peak_bytes"] = min(peaks) if peaks else None  # type: ignore[type-var]
    return list(merged.values())


def _format_seconds(record: dict[str, object] | None) -> str:
    if record is None:
        return "missing"
    if record["status"] != "ok":
        return str(record["status"])
    seconds = float(record["seconds"])  # type: ignore[arg-type]
    marker = "*" if record.get("single_run") else ""
    if seconds >= 1:
        return f"{seconds:.2f} s{marker}"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.2f} ms{marker}"
    return f"{seconds * 1e6:.0f} µs{marker}"


def _format_bytes(record: dict[str, object] | None) -> str:
    if record is None or record["status"] != "ok" or record.get("peak_bytes") is None:
        return "n/a"
    return f"{int(record['peak_bytes']) / 2**20:.1f} MiB"  # type: ignore[call-overload]


def _format_ratio(ratio: float | None) -> str:
    return "—" if ratio is None else f"{ratio:.2f}×"


def render_table(rows: list[ComparisonRow]) -> str:
    """
    Render comparison rows as a markdown table.

    :param rows: The rows to render.
    :return: A markdown table with time and peak memory before and now, their ratios and the flags.
    :rtype: str
    """
    lines = [
        "| case | time before | time now | ratio | memory before | memory now | ratio | flags |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.identifier}` | {_format_seconds(row.before)} | {_format_seconds(row.now)} | "
            f"{_format_ratio(row.time_ratio)} | {_format_bytes(row.before)} | {_format_bytes(row.now)} | "
            f"{_format_ratio(row.memory_ratio)} | {', '.join(row.flags)} |"
        )
    return "\n".join(lines)


def render_summary(rows: list[ComparisonRow], header: str) -> str:
    """
    Render the summary: counts per flag, then the flagged rows grouped by flag, regressions first.

    :param rows: All comparison rows.
    :param header: Text shown above the summary (versions, mode, machine).
    :return: Markdown text.
    :rtype: str
    """
    counts = {flag: sum(flag in row.flags for row in rows) for flag in FLAG_ORDER}
    parts = [header, "", f"{len(rows)} cases. " + ", ".join(f"{flag}: {counts[flag]}" for flag in FLAG_ORDER), ""]
    for flag in FLAG_ORDER:
        flagged = [row for row in rows if flag in row.flags]
        if not flagged:
            continue
        key = (lambda row: -(row.time_ratio or 0)) if flag == "slower" else (lambda row: row.time_ratio or 0)
        if flag in ("heavier", "lighter"):
            key = (lambda row: -(row.memory_ratio or 0)) if flag == "heavier" else (lambda row: row.memory_ratio or 0)
        parts += [f"### {flag} ({len(flagged)})", "", render_table(sorted(flagged, key=key)), ""]
    if not any(row.flags for row in rows):
        parts.append("No case changed beyond the thresholds.")
    parts.append("`*` marks a time from a single run (the case is slower than the repeat budget).")
    return "\n".join(parts)


def _run(command: list[str], cwd: Path, environment: dict[str, str] | None = None) -> None:
    result = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"command failed: {' '.join(command)}\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")


def _maturin() -> list[str]:
    executable = shutil.which("maturin") or str(Path(sys.executable).with_name("maturin"))
    return [executable] if Path(executable).exists() else [sys.executable, "-m", "maturin"]


def build_version(source: Path, name: str, workspace: Path) -> Path:
    """
    Build a release wheel of the package in ``source`` and unpack it, Rust kernel included.

    :param source: Repository checkout to build (the working tree, or a worktree of the baseline).
    :param name: Short name of the version, used for the cache and output folders.
    :param workspace: Temporary folder receiving the wheel and the unpacked package.
    :return: Folder to put first on ``PYTHONPATH`` to import this version.
    :rtype: Path
    """
    distribution = workspace / f"{name}-dist"
    site = workspace / f"{name}-site"
    environment = dict(os.environ, CARGO_TARGET_DIR=str(CACHE_DIRECTORY / f"target-{name}"))
    _run(
        [*_maturin(), "build", "--release", "-m", str(source / "rust" / "Cargo.toml"), "--out", str(distribution)],
        source,
        environment,
    )
    (wheel,) = distribution.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(site)
    return site


def _read_progress(output: Path) -> tuple[list[dict[str, object]], str | None]:
    # Records written so far, and the identifier of the case started but not finished (if any).
    records: list[dict[str, object]] = []
    in_progress = None
    if not output.exists():
        return records, in_progress
    for line in output.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry["status"] == STARTED:
            in_progress = str(entry["identifier"])
        else:
            records.append(entry)
            in_progress = None
    return records, in_progress


def run_version(
    site: Path,
    bench: str,
    devices: list[str],
    families: list[str],
    threads: int,
    output: Path,
    only: list[str] | None = None,
) -> list[dict[str, object]]:
    """
    Run ``runner.py`` for one version in fresh processes and collect one record per case.

    The runner announces each case before running it and writes its record when done. When a case crashes
    the process (segfault, abort, out-of-memory kill), or hangs past its time limits so the watchdog kills
    it, that case is recorded as ``crashed`` or ``timeout`` and a new process resumes with the next case.

    :param site: Folder holding the unpacked version, put first on ``PYTHONPATH``.
    :param bench: Benchmark mode.
    :param devices: Devices to benchmark on.
    :param families: Matrix families to run.
    :param threads: Intra-op thread count, set through ``OMP_NUM_THREADS``, ``MKL_NUM_THREADS`` and
        ``RAYON_NUM_THREADS`` (the Rust kernel), never ``torch.set_num_threads``, which corrupts batched LU
        on some MKL builds (issue #23).
    :param output: File receiving the records (overwritten).
    :param only: Case identifiers to run; ``None`` (default) runs every case of ``families``.
    :return: The records, one per case, in the order of :func:`cases.build_cases`.
    :rtype: list[dict[str, object]]
    """
    environment = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(site), os.environ.get("PYTHONPATH", "")]),
        OMP_NUM_THREADS=str(threads),
        MKL_NUM_THREADS=str(threads),
        RAYON_NUM_THREADS=str(threads),
    )
    identifiers = [case.identifier for case in build_cases(bench, devices, families)]
    if only is not None:
        selected = set(only)
        identifiers = [identifier for identifier in identifiers if identifier in selected]
    watchdog_seconds = 2 * TIMING_BUDGET[bench][2] + WATCHDOG_GRACE_SECONDS  # two phases per case, plus grace
    records: list[dict[str, object]] = []
    remaining = identifiers
    while remaining:
        selection = output.with_suffix(".only")
        selection.write_text("\n".join(remaining))
        output.unlink(missing_ok=True)
        log = output.with_suffix(".log")
        command = [sys.executable, str(BENCHMARK_DIRECTORY / "runner.py"), "--bench", bench, "--expect-root", str(site)]
        command += ["--devices", *devices, "--families", *families, "--output", str(output), "--only", str(selection)]
        with open(log, "w") as log_file:
            process = subprocess.Popen(
                command, cwd=BENCHMARK_DIRECTORY, env=environment, stdout=log_file, stderr=log_file
            )
            killed = False
            try:
                last_progress, last_size = time.monotonic(), -1
                while process.poll() is None:
                    time.sleep(0.2)
                    size = output.stat().st_size if output.exists() else 0
                    if size != last_size:
                        last_progress, last_size = time.monotonic(), size
                    elif time.monotonic() - last_progress > watchdog_seconds:
                        process.kill()
                        process.wait()
                        killed = True
            finally:
                if process.poll() is None:
                    process.kill()  # interrupted: never leave the child running
                    process.wait()
        finished, in_progress = _read_progress(output)
        records += finished
        done = {str(record["identifier"]) for record in finished}
        if process.returncode == 0 and not killed:
            break
        if in_progress is None:
            # The process failed before starting a case (import error, wrong torch_pfaffian): not recoverable.
            sys.exit(f"runner failed for {site}:\n{log.read_text()[-3000:]}")
        status = "timeout" if killed else "crashed"
        message = "killed by the watchdog" if killed else f"process exited with code {process.returncode}"
        records.append({"identifier": in_progress, "status": status, "message": message})
        done.add(in_progress)
        remaining = [identifier for identifier in remaining if identifier not in done]
    return records


def _devices(cuda: str) -> list[str]:
    if cuda == "true" and not torch.cuda.is_available():
        sys.exit("--cuda true but CUDA is not available")
    use_cuda = cuda == "true" or (cuda == "auto" and torch.cuda.is_available())
    return ["cpu", "cuda"] if use_cuda else ["cpu"]


def _load_average() -> float:
    return os.getloadavg()[0] if hasattr(os, "getloadavg") else float("nan")


def _header(baseline: str, baseline_commit: str, bench: str, devices: list[str], threads: int, load: float) -> str:
    cores = os.cpu_count() or 1
    lines = [
        f"## Benchmark: `{baseline}` ({baseline_commit[:10]}) vs working tree",
        "",
        f"mode `{bench}`, devices {', '.join(devices)}, torch {torch.__version__}, {threads} threads, "
        f"load average {load:.1f} on {cores} cores before the run",
    ]
    if load > 0.5 * cores:  # load read before the run, so the benchmark's own threads are not counted
        lines.append(f"\n> **Warning:** the machine is busy (load {load:.1f}); timings are noisy, re-run when idle.")
    return "\n".join(lines)


def _positive_integer(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected an integer >= 1, got {value}")
    return value


def _results_directory() -> Path:
    # Created only once the run has finished, with a suffix when two runs end within the same second.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    candidate, index = RESULTS_DIRECTORY / stamp, 1
    while candidate.exists():
        index += 1
        candidate = RESULTS_DIRECTORY / f"{stamp}-{index}"
    candidate.mkdir(parents=True)
    return candidate


def _interrupt(signum: int, frame: object) -> None:
    # Turn SIGTERM into KeyboardInterrupt: subprocess.run then kills the running child, and the temporary
    # folder and the baseline worktree are removed by their context manager and finally block.
    raise KeyboardInterrupt


def main() -> None:
    """
    Compare speed and peak memory of the working tree against a baseline revision.

    Builds both versions as release wheels, runs them family by family in alternating fresh processes,
    prints the summary of changed cases, and writes the full table and raw records to
    ``benchmarks/results/<timestamp>/``.
    """
    parser = argparse.ArgumentParser(description="Compare speed and memory of the working tree against a baseline.")
    parser.add_argument("--baseline", default="origin/dev", help="git revision to compare against (default origin/dev)")
    parser.add_argument("--bench", choices=["small", "medium", "big"], default="medium")
    parser.add_argument("--cuda", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="relative change reported (0.25)")
    parser.add_argument("--families", nargs="+", choices=list(FAMILIES), default=list(FAMILIES))
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="intra-op threads of the benchmarked processes (default: half the cores)",
    )
    parser.add_argument(
        "--rounds", type=_positive_integer, default=2, help="alternating rounds, best time kept (default 2)"
    )
    parser.add_argument(
        "--confirm-rounds", type=int, default=3, help="extra rounds re-running only changed cases (default 3)"
    )
    parser.add_argument("--github", action="store_true", help="write the GitHub job summary and warnings")
    arguments = parser.parse_args()
    signal.signal(signal.SIGTERM, _interrupt)

    commit = subprocess.run(
        ["git", "rev-parse", "--verify", f"{arguments.baseline}^{{commit}}"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    if commit.returncode != 0:
        sys.exit(f"unknown baseline revision {arguments.baseline!r}; run `git fetch` first")
    baseline_commit = commit.stdout.strip()
    devices = _devices(arguments.cuda)
    load = _load_average()  # read before the benchmark adds its own threads

    with tempfile.TemporaryDirectory(prefix="torch-pfaffian-bench-") as temporary:
        workspace = Path(temporary)
        worktree = workspace / "baseline-source"
        _run(["git", "worktree", "add", "--detach", str(worktree), baseline_commit], REPOSITORY_ROOT)
        try:
            print(f"building {arguments.baseline} ({baseline_commit[:10]}) and the working tree...", flush=True)
            baseline_site = build_version(worktree, "baseline", workspace)
            current_site = build_version(REPOSITORY_ROOT, "current", workspace)
        finally:
            _run(["git", "worktree", "remove", "--force", str(worktree)], REPOSITORY_ROOT)
        rounds: dict[str, list[list[dict[str, object]]]] = {"before": [], "now": []}
        versions = [("before", baseline_site), ("now", current_site)]
        for round_index in range(arguments.rounds):
            # Alternate the versions per family, and their order per round, so drifting machine load and
            # any first-or-second bias affect both versions alike.
            order = versions if round_index % 2 == 0 else versions[::-1]
            for name, _ in versions:
                rounds[name].append([])
            for family in arguments.families:
                print(f"round {round_index + 1}/{arguments.rounds}: {family}...", flush=True)
                for name, site in order:
                    output = workspace / f"{name}.json"
                    records = run_version(site, arguments.bench, devices, [family], arguments.threads, output)
                    rounds[name][-1] += records
        before = merge_rounds(rounds["before"])
        now = merge_rounds(rounds["now"])
        # Confirmation: re-run the cases whose time changed. The best time can only come down towards the
        # true one, so a real regression survives these rounds while a noisy one disappears.
        for round_index in range(arguments.confirm_rounds):
            rows = compare_records(before, now, arguments.threshold)
            changed = [row.identifier for row in rows if {"slower", "faster", "heavier", "lighter"} & set(row.flags)]
            if not changed:
                break
            print(
                f"confirming {len(changed)} case(s), round {round_index + 1}/{arguments.confirm_rounds}...", flush=True
            )
            order = versions if round_index % 2 == 0 else versions[::-1]
            for name, site in order:
                output = workspace / f"{name}-confirm.json"
                families = arguments.families
                rounds[name].append(
                    run_version(site, arguments.bench, devices, families, arguments.threads, output, changed)
                )
            before = merge_rounds(rounds["before"])
            now = merge_rounds(rounds["now"])

    rows = compare_records(before, now, arguments.threshold)
    if not rows:
        sys.exit("no benchmark case ran; check --families and --bench")
    results = _results_directory()
    header = _header(arguments.baseline, baseline_commit, arguments.bench, devices, arguments.threads, load)
    summary = render_summary(rows, header)
    (results / "summary.md").write_text(summary)
    (results / "full.md").write_text(header + "\n\n" + render_table(rows))
    (results / "records.json").write_text(json.dumps({"before": before, "now": now}))
    print(summary)
    print(f"\nfull table: {results / 'full.md'}")

    if arguments.github:
        regressions = [row for row in rows if REGRESSION_FLAGS & set(row.flags)]
        command = LOCAL_COMMAND.format(baseline=arguments.baseline)
        advice = (
            f"\n\n**{len(regressions)} case(s) regressed.** Shared CI runners are noisy: confirm on "
            f"your machine with `{command}` (default mode `medium`)."
            if regressions
            else ""
        )
        if "GITHUB_STEP_SUMMARY" in os.environ:
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as job_summary:
                job_summary.write(summary + advice + "\n")
        if regressions:
            print(
                f"::warning title=Benchmark regression::{len(regressions)} case(s) slower, heavier, failing or "
                f"disagreeing with {arguments.baseline}. Run `{command}` locally to confirm."
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("interrupted; the temporary build and the baseline worktree were removed")
