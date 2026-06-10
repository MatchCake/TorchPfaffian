import time
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from torch_pfaffian.strategies.strategy import PfaffianStrategy

Strategy = type[PfaffianStrategy]
BenchmarkResults = dict[str, dict[str, list[float]]]

METRICS = ("forward_time", "backward_time", "forward_memory", "forward_backward_memory")


def make_block_antidiagonal(
    n: int, batch_size: int, dtype: torch.dtype = torch.float64, device: str = "cpu"
) -> torch.Tensor:
    """
    Build a batch of block-antidiagonal skew matrices ``[[0, B], [-B^T, 0]]``.

    :param n: Half the matrix dimension; the result has shape ``(batch_size, 2n, 2n)``.
    :param batch_size: Number of matrices in the batch.
    :param dtype: Floating dtype of the matrices.
    :param device: Device on which to allocate the matrices.
    :return: A tensor of shape ``(batch_size, 2n, 2n)``.
    :rtype: torch.Tensor
    """
    block = torch.randn(batch_size, n, n, dtype=dtype, device=device)
    zero = torch.zeros_like(block)
    top = torch.cat([zero, block], dim=-1)
    bottom = torch.cat([-block.transpose(-1, -2), zero], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def _synchronize(is_cuda: bool) -> None:
    if is_cuda:
        torch.cuda.synchronize()


def _median_time(fn: Callable[[], object], is_cuda: bool, n_repeats: int = 3) -> float:
    fn()  # warm-up
    _synchronize(is_cuda)
    timings = []
    for _ in range(n_repeats):
        _synchronize(is_cuda)
        start = time.perf_counter()
        fn()
        _synchronize(is_cuda)
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def _peak_memory(fn: Callable[[], object], is_cuda: bool) -> float:
    # On CUDA this is the true peak; on CPU it is the profiler-reported allocated bytes.
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
        fn()
        torch.cuda.synchronize()
        return float(torch.cuda.max_memory_allocated())
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        fn()
    return float(sum(e.cpu_memory_usage for e in prof.key_averages() if e.cpu_memory_usage > 0))


def _forward_call(strategy: Strategy, matrix: torch.Tensor) -> torch.Tensor:
    return strategy.apply(matrix)


def _forward_backward_call(strategy: Strategy, matrix: torch.Tensor) -> torch.Tensor:
    grad_matrix = matrix.detach().clone().requires_grad_(True)
    strategy.apply(grad_matrix).sum().backward()
    return grad_matrix


def _confidence_interval(values: np.ndarray, confidence_z: float = 1.96) -> float:
    # Half-width of the normal-approximation confidence interval for the mean.
    if values.size < 2:
        return 0.0
    return float(confidence_z * values.std(ddof=1) / np.sqrt(values.size))


def benchmark_strategies(
    strategies: list[Strategy],
    sizes_n: list[int],
    batch_size: int,
    device: str = "cpu",
    n_seeds: int = 20,
    n_repeats: int = 3,
) -> BenchmarkResults:
    """
    Benchmark forward/backward time and memory of each strategy over several random seeds.

    For each matrix dimension and strategy the metric is measured once per seed; the returned
    values are the across-seed mean and the half-width of the 95% confidence interval.

    :param strategies: The strategy classes to benchmark (each exposes ``NAME`` and ``apply``).
    :param sizes_n: Half-dimensions to sweep; each yields matrices of dimension ``2n``.
    :param batch_size: Number of matrices per benchmarked batch.
    :param device: Device on which to run the benchmark.
    :param n_seeds: Number of random seeds used to build the confidence intervals.
    :param n_repeats: Number of timed repeats per seed (the median is kept).
    :return: A mapping ``strategy_name -> {"dimension", "<metric>_mean", "<metric>_ci"}``.
    :rtype: dict
    """
    is_cuda = device == "cuda"
    results: BenchmarkResults = {
        strategy.NAME: {
            "dimension": [],
            **{f"{metric}_mean": [] for metric in METRICS},
            **{f"{metric}_ci": [] for metric in METRICS},
        }
        for strategy in strategies
    }

    for n in sizes_n:
        per_seed: dict[str, dict[str, list[float]]] = {
            strategy.NAME: {metric: [] for metric in METRICS} for strategy in strategies
        }
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            matrix = make_block_antidiagonal(n, batch_size, dtype=torch.float64, device=device)
            for strategy in strategies:
                forward_time = _median_time(lambda: _forward_call(strategy, matrix), is_cuda, n_repeats)
                full_time = _median_time(lambda: _forward_backward_call(strategy, matrix), is_cuda, n_repeats)
                record = per_seed[strategy.NAME]
                record["forward_time"].append(forward_time)
                record["backward_time"].append(max(full_time - forward_time, 0.0))
                record["forward_memory"].append(_peak_memory(lambda: _forward_call(strategy, matrix), is_cuda))
                record["forward_backward_memory"].append(
                    _peak_memory(lambda: _forward_backward_call(strategy, matrix), is_cuda)
                )

        for strategy in strategies:
            results[strategy.NAME]["dimension"].append(2 * n)
            for metric in METRICS:
                values = np.asarray(per_seed[strategy.NAME][metric], dtype=float)
                results[strategy.NAME][f"{metric}_mean"].append(float(values.mean()))
                results[strategy.NAME][f"{metric}_ci"].append(_confidence_interval(values))
    return results


def plot_results(
    results: BenchmarkResults, strategies: list[Strategy], batch_size: int, device: str = "cpu"
) -> plt.Figure:
    """
    Plot the benchmark results as a 2x2 grid of time and memory panels with 95% CI bands.

    :param results: The mapping returned by :func:`benchmark_strategies`.
    :param strategies: The strategy classes that were benchmarked.
    :param batch_size: Batch size used for the benchmark (shown in the title).
    :param device: Device used for the benchmark (shown in the title and memory label).
    :return: The matplotlib figure.
    :rtype: matplotlib.figure.Figure
    """
    memory_label = "peak memory (bytes, CUDA)" if device == "cuda" else "allocated bytes (CPU)"
    panels = [
        ("forward_time", "Forward time", "time [s]"),
        ("backward_time", "Backward time", "time [s]"),
        ("forward_memory", "Forward memory", memory_label),
        ("forward_backward_memory", "Forward + backward memory", memory_label),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (key, title, ylabel) in zip(axes.flat, panels):
        for strategy in strategies:
            record = results[strategy.NAME]
            dimension = np.asarray(record["dimension"], dtype=float)
            mean = np.asarray(record[f"{key}_mean"], dtype=float)
            ci = np.asarray(record[f"{key}_ci"], dtype=float)
            line = ax.plot(dimension, mean, marker="o", label=strategy.NAME)[0]
            lower = np.clip(mean - ci, np.finfo(float).tiny, None)
            ax.fill_between(dimension, lower, mean + ci, alpha=0.2, color=line.get_color())
        ax.set_title(title)
        ax.set_xlabel("matrix dimension (2n)")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend()

    fig.suptitle(f"Pfaffian strategies benchmark (batch_size={batch_size}, device={device}, 95% CI over seeds)")
    fig.tight_layout()
    return fig
