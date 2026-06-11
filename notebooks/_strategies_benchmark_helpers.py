import time
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.profiler import ProfilerActivity, profile
from tqdm.auto import tqdm

from torch_pfaffian.strategies.pfaffian_block_det import PfaffianBlockDet
from torch_pfaffian.strategies.strategy import PfaffianStrategy

Strategy = type[PfaffianStrategy]
BenchmarkResults = pd.DataFrame

METRICS = ("forward_time", "backward_time", "forward_memory", "forward_backward_memory")


def make_block_antidiagonal(
    n: int, batch_size: int, dtype: torch.dtype = torch.float32, device: str = "cpu"
) -> torch.Tensor:
    """
    Build a batch of block-antidiagonal skew matrices ``[[0, B], [-B^T, 0]]`` of dimension ``n``.

    :param n: Matrix dimension (even); the result has shape ``(batch_size, n, n)``.
    :param batch_size: Number of matrices in the batch.
    :param dtype: Floating dtype of the matrices.
    :param device: Device on which to allocate the matrices.
    :return: A tensor of shape ``(batch_size, n, n)``.
    :rtype: torch.Tensor
    """
    half = n // 2
    block = torch.randn(batch_size, half, half, dtype=dtype, device=device)
    zero = torch.zeros_like(block)
    top = torch.cat([zero, block], dim=-1)
    bottom = torch.cat([-block.transpose(-1, -2), zero], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def make_random_skew(n: int, batch_size: int, dtype: torch.dtype = torch.float32, device: str = "cpu") -> torch.Tensor:
    """
    Build a batch of general (dense) random skew-symmetric matrices of dimension ``n``.

    Unlike :func:`make_block_antidiagonal`, the result has no block structure, so it is a
    representative input for the strategies that accept any skew-symmetric matrix (every strategy
    except ``PfaffianBlockDet``, which only computes the true Pfaffian on block-antidiagonal inputs).

    :param n: Matrix dimension (even); the result has shape ``(batch_size, n, n)``.
    :param batch_size: Number of matrices in the batch.
    :param dtype: Floating dtype of the matrices.
    :param device: Device on which to allocate the matrices.
    :return: A tensor of shape ``(batch_size, n, n)``.
    :rtype: torch.Tensor
    """
    full = torch.randn(batch_size, n, n, dtype=dtype, device=device)
    return full - full.transpose(-1, -2)


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


def benchmark_strategies(
    strategies: list[Strategy],
    sizes_n: list[int],
    batch_size: int,
    device: str = "cpu",
    n_seeds: int = 20,
    n_repeats: int = 3,
) -> pd.DataFrame:
    """
    Benchmark forward/backward time and memory of each strategy over several random seeds.

    Returns a tidy DataFrame with one row per (strategy, dimension, seed) combination and one
    column per metric.  Pass it directly to :func:`plot_results` or use pandas/seaborn to
    build custom views.

    :param strategies: The strategy classes to benchmark (each exposes ``NAME`` and ``apply``).
    :param sizes_n: Matrix dimensions to sweep (even integers); each yields ``n x n`` matrices.
    :param batch_size: Number of matrices per benchmarked batch.
    :param device: Device on which to run the benchmark.
    :param n_seeds: Number of random seeds used to build the confidence intervals.
    :param n_repeats: Number of timed repeats per seed (the median is kept).
    :return: A DataFrame with columns ``strategy``, ``dimension``, ``seed``, and one column per
        metric in :data:`METRICS`.
    :rtype: pandas.DataFrame
    """
    is_cuda = device == "cuda"
    records = []
    p_bar = tqdm(total=len(sizes_n) * len(strategies) * n_seeds, desc="Benchmarking strategies")
    for n in sizes_n:
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            # PfaffianBlockDet only returns the true Pfaffian on block-antidiagonal inputs; every
            # other strategy is valid on any skew matrix.
            block_matrix = make_block_antidiagonal(n, batch_size, dtype=torch.float32, device=device)
            random_matrix = make_random_skew(n, batch_size, dtype=torch.float32, device=device)
            for strategy in strategies:
                matrix = block_matrix if strategy is PfaffianBlockDet else random_matrix
                p_bar.set_description(f"Benchmarking {strategy.NAME:^30} (n: {n:^5} | seed: {seed:^5})")
                forward_time = _median_time(lambda: _forward_call(strategy, matrix), is_cuda, n_repeats)
                full_time = _median_time(lambda: _forward_backward_call(strategy, matrix), is_cuda, n_repeats)
                records.append(
                    {
                        "strategy": strategy.NAME,
                        "dimension": n,
                        "seed": seed,
                        "forward_time": forward_time,
                        "backward_time": max(full_time - forward_time, 0.0),
                        "forward_memory": _peak_memory(lambda: _forward_call(strategy, matrix), is_cuda),
                        "forward_backward_memory": _peak_memory(
                            lambda: _forward_backward_call(strategy, matrix), is_cuda
                        ),
                    }
                )
                p_bar.update(1)
    p_bar.close()
    return pd.DataFrame(records)


def plot_results(results: pd.DataFrame, strategies: list[Strategy], batch_size: int, device: str = "cpu") -> plt.Figure:
    """
    Plot the benchmark results as a 2x2 grid of time and memory panels with 95% CI bands.

    :param results: The DataFrame returned by :func:`benchmark_strategies`.
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

    df_long = results.melt(
        id_vars=["strategy", "dimension", "seed"],
        value_vars=list(METRICS),
        var_name="metric",
        value_name="value",
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (metric_key, title, ylabel) in zip(axes.flat, panels):
        subset = df_long[df_long["metric"] == metric_key]
        sns.lineplot(data=subset, x="dimension", y="value", hue="strategy", marker="o", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("matrix dimension")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

    fig.suptitle(f"Pfaffian strategies benchmark (batch_size={batch_size}, device={device}, 95% CI over seeds)")
    fig.tight_layout()
    return fig
