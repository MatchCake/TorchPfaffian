# TorchPfaffian: PyTorch-Based Pfaffian Computation

[![Star on GitHub](https://img.shields.io/github/stars/MatchCake/TorchPfaffian.svg?style=social)](https://github.com/MatchCake/TorchPfaffian/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/MatchCake/TorchPfaffian?style=social)](https://github.com/MatchCake/TorchPfaffian/network/members)
[![Python](https://img.shields.io/pypi/pyversions/torchpfaffian)](https://pypi.org/project/torchpfaffian)
[![downloads](https://img.shields.io/pypi/dm/torchpfaffian)](https://pypi.org/project/torchpfaffian)
[![PyPI version](https://img.shields.io/pypi/v/torchpfaffian)](https://pypi.org/project/torchpfaffian)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/MatchCake/TorchPfaffian/blob/dev/LICENSE)
[![DOI](https://zenodo.org/badge/862986060.svg)](https://doi.org/10.5281/zenodo.24085345)

![Tests Workflow](https://github.com/MatchCake/TorchPfaffian/actions/workflows/tests.yml/badge.svg)
![Dist Workflow](https://github.com/MatchCake/TorchPfaffian/actions/workflows/build_dist.yml/badge.svg)
![Doc Workflow](https://github.com/MatchCake/TorchPfaffian/actions/workflows/docs.yml/badge.svg)
![Coverage Badge Workflow](https://github.com/MatchCake/TorchPfaffian/actions/workflows/coverage_badge.yml/badge.svg)
[![codecov](https://codecov.io/gh/MatchCake/TorchPfaffian/graph/badge.svg?token=Z1TK10CJUC)](https://codecov.io/gh/MatchCake/TorchPfaffian)


# Description

TorchPfaffian computes the Pfaffian of batches of skew-symmetric matrices in PyTorch, with exact gradients
on CPU and GPU, in float32, float64, complex64 and complex128 (float16, bfloat16 and complex32 inputs are
computed in float32 or complex64).

## Statement of need

Pfaffians appear wherever fermions are paired: the overlaps and amplitudes of fermionic Gaussian states,
the simulation of matchgate (free-fermion) quantum circuits, BCS and Hartree–Fock–Bogoliubov wave
functions, the partition functions of planar Ising models and perfect-matching counts. In these settings
the Pfaffian is usually one step of a larger differentiable model that runs on batches of matrices,
often on a GPU, and is optimized by gradient descent.

The reference implementation, [pfapack](https://github.com/basnijholt/pfapack), works on single NumPy
arrays, without automatic differentiation or GPU support. The determinant `torch.linalg.det` gives only
`Pf(A)^2 = det(A)` and loses the sign, which carries the physics (interference between amplitudes).
[torchpfaff](https://github.com/Nuclear-Physics-with-Machine-Learning/PyPfaffian) brings a differentiable
Parlett–Reid Pfaffian to PyTorch, with a Triton kernel on CUDA, for real matrices. TorchPfaffian fills the
remaining gap: a signed, batched and differentiable Pfaffian that is a drop-in PyTorch operation, for real
and complex matrices, with exact gradients at singular matrices. It is the Pfaffian backend of the
[MatchCake](https://github.com/MatchCake/MatchCake) matchgate simulator.

## Features

- **One entry point, automatic routing.** `pfaffian()` picks the fastest correct strategy for the input
  (size, dtype, device, block structure, whether gradients and the sign are needed) from a registry of
  strategies: an exact unrolled kernel for small matrices, Parlett–Reid tridiagonalization in Rust (CPU)
  and in PyTorch (any device), determinant-based magnitudes, and a block-determinant path for matrices of
  the form `[[0, B], [-B^T, 0]]`.
- **Exact gradients, including at singular matrices.** The backward of `Pf(A)` is
  `Pf(A) A^{-T} / 2` when `A` is invertible. At singular `A`, where the inverse does not exist but the
  gradient does, it is computed exactly from the Pfaffian adjugate. Complex inputs follow PyTorch's
  Wirtinger convention.
- **Overflow-safe.** Results that overflow the dtype in the linear domain are recomputed in the log
  domain; `slog_pfaffian()` returns `(phase, log|Pf|)` directly.
- **Round-off zeros.** `mask_roundoff_gradient=True` gives a zero gradient to Pfaffians that vanish up to
  round-off, such as parity-forbidden amplitudes.
- **Batched and device-agnostic.** Any leading batch shape, on CPU or CUDA, with a native Rust kernel on CPU.
  On CPU it is faster than pfapack's C implementation on batches of small matrices, and on batches of large
  ones with several threads; on a single matrix, pfapack is faster. See the
  [performance](https://matchcake.github.io/TorchPfaffian/performance.html) page.
  `benchmarks/library_comparison.py` times TorchPfaffian, pfapack, torchpfaff and lrux side by side, on CPU
  and on GPU when available.

## Requirements

- Python 3.10 to 3.14 and PyTorch 2.1 or newer.
- Prebuilt wheels with the Rust kernel are published for Linux x86_64, Windows x86_64 and macOS arm64. On
  other platforms pip builds the package from source, which needs a Rust toolchain
  (<https://rustup.rs>).

## Installation

```bash
pip install torchpfaffian
```

On Linux, pip installs the CUDA build of PyTorch by default. For a CPU-only environment, install PyTorch
first from its CPU index:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torchpfaffian
```

For development, this project uses [uv](https://docs.astral.sh/uv/). From a clone of the repository:
```bash
uv sync --dev --extra cpu
uv run maturin develop --release -m rust/Cargo.toml
```
The `cpu`, `cu128` and `cu130` extras select the PyTorch build; they only take effect with `uv` from a
clone. `--release` matters: `maturin develop` builds the Rust kernel in debug mode by default, which makes
it much slower. See [CONTRIBUTING.md](https://github.com/MatchCake/TorchPfaffian/blob/dev/CONTRIBUTING.md) for the full development workflow.

## Usage

```python
import torch

from torch_pfaffian import pfaffian, slog_pfaffian

# Any skew-symmetric matrix of shape (..., 2n, 2n).
matrix = torch.tensor([[0.0, -3.0], [3.0, 0.0]])

pf = pfaffian(matrix)                 # signed Pfaffian (default)
magnitude = pfaffian(matrix, sign=False)  # |pf|, through the faster determinant-based path
```

`pfaffian()` chooses the strategy from the input; the full routing table is in its
[documentation](https://matchcake.github.io/TorchPfaffian/). To force a strategy, use
`get_pfaffian_function(name)`, for example `get_pfaffian_function("PfaffianParlettReid")(matrix)`.

Gradients flow through every path:

```python
matrix = torch.randn(32, 10, 10, dtype=torch.float64)
matrix = (matrix - matrix.transpose(-1, -2)).requires_grad_()

pfaffian(matrix).sum().backward()     # matrix.grad = Pf(A) A^{-T} / 2 per batch element
phase, log_abs = slog_pfaffian(matrix.detach())  # log domain, never overflows
```

### Pfaffian ratios under low-rank updates

Monte Carlo moves and neural quantum states often only need `Pf(A') / Pf(A)` for a matrix `A'` that differs
from `A` by a few rows and columns. Given `A^{-1}`, `LowRankPfaffian` computes that ratio in `O(n^2 k)`
(`O(n)` for one replaced row) instead of the `O(n^3)` of a new Pfaffian, together with the updated inverse.
It follows the convention of [lrux](https://github.com/ChenAo-Phys/lrux): `A' = A - u J u^T` with `u` of shape
`(n, 2k)` and `J = [[0, I_k], [-I_k, 0]]`.

```python
from torch_pfaffian import PfaffianUpdater, pfaffian_ratio, pfaffian_ratio_replace_and_inverse

inverse = torch.linalg.inv(matrix.detach())                 # (32, 10, 10)
update = torch.randn(32, 10, 2, dtype=torch.float64)       # rank-2 update u = (x, y)
ratio = pfaffian_ratio(inverse, update)                     # Pf(A - u J u^T) / Pf(A)
change = torch.randn(32, 10, dtype=torch.float64)
ratio, new_inverse = pfaffian_ratio_replace_and_inverse(inverse, 3, change)  # row and column 3 change

updater = PfaffianUpdater(matrix.detach(), max_delay=4)    # delayed inverse updates for long chains
accepted = torch.rand(32) < updater.ratio(update).abs() ** 2
updater.accept(update, mask=accepted)                       # only the accepted walkers move
```

For a matrix that grows block by block, as in qubit-by-qubit sampling of matchgate circuits,
`pfaffian_extension_ratio` gives `Pf([[A, B], [-B^T, C]]) / Pf(A) = Pf(C + B^T A^{-1} B)` from `A^{-1}`, and
`PfaffianChain` keeps the inverse from one qubit to the next, so that a sample of `n` qubits costs `O(n^3)`
instead of `O(n^4)`:

```python
from torch_pfaffian import PfaffianChain

chain = PfaffianChain((shots,))                              # starts from the empty matrix
for qubit in range(n_qubits):
    modes = slice(2 * qubit, 2 * qubit + 2)
    border = covariance[..., : 2 * qubit, modes]             # (shots, 2 qubit, 2)
    corners = covariance[..., modes, modes] + outcome_blocks # (shots, 2, 2, 2): the blocks for y = 0 and 1
    probabilities = chain.ratios(border, corners).abs() / 2  # conditional probabilities, (shots, 2)
    outcome = torch.multinomial(probabilities, 1)[..., 0]
    chain.extend(border, corners[torch.arange(shots), outcome])
```

Everything is batched, differentiable, and runs on CPU and CUDA. Gradients are exact for any skew-symmetric
construction of the inputs, such as a covariance matrix computed from circuit parameters; the gradient with
respect to a raw `inverse` leaf follows the skew-split convention of `pfaffian()`, while those with respect to
blocks sliced from a leaf matrix (`border`, `corner`) are only meaningful in their skew-symmetric part.
Functions that also return the updated inverse have their own name (`..._and_inverse`) rather than a flag.
`pfaffian_ratio`, `pfaffian_ratio_and_inverse`, `pfaffian_ratio_replace`,
`pfaffian_ratio_replace_and_inverse`, `pfaffian_extension_ratio`, `pfaffian_extension_ratio_and_inverse` and
`skew_identity` are shortcuts to the static methods of `LowRankPfaffian`.


# Important Links
- Documentation: [https://matchcake.github.io/TorchPfaffian/](https://matchcake.github.io/TorchPfaffian/).
- Source: [https://github.com/MatchCake/TorchPfaffian/](https://github.com/MatchCake/TorchPfaffian/).
- Tutorials: [getting started](https://matchcake.github.io/TorchPfaffian/notebooks/tutorial.html) and
  [domino tilings and dimer probabilities](https://matchcake.github.io/TorchPfaffian/notebooks/domino_tilings.html).
- Performance against pfapack: [performance](https://matchcake.github.io/TorchPfaffian/performance.html).
- Known limitations (`torch.func`, thread count, half precision): [limitations](https://matchcake.github.io/TorchPfaffian/limitations.html).
- Changes between versions: [CHANGELOG.md](https://github.com/MatchCake/TorchPfaffian/blob/dev/CHANGELOG.md).

# Found a bug, have a question or a feature request?
- [Open an issue](https://github.com/MatchCake/TorchPfaffian/issues/new/choose); questions are welcome there.
- Contributions are welcome: see [CONTRIBUTING.md](https://github.com/MatchCake/TorchPfaffian/blob/dev/CONTRIBUTING.md).

## License
[Apache License 2.0](https://github.com/MatchCake/TorchPfaffian/blob/dev/LICENSE)

## Citation
If you use TorchPfaffian in your research, please cite it:
```
@software{gince_torchpfaffian,
  author    = {Jérémie Gince},
  title     = {{TorchPfaffian}},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.24085345},
  url       = {https://doi.org/10.5281/zenodo.24085345},
}
```
This DOI always resolves to the latest version. To cite the exact version you used, take its DOI and BibTeX
entry from the [Zenodo record](https://doi.org/10.5281/zenodo.24085345).

## Related software
TorchPfaffian is compared against [pfapack](https://github.com/basnijholt/pfapack),
[torchpfaff](https://github.com/Nuclear-Physics-with-Machine-Learning/PyPfaffian) and
[lrux](https://github.com/ChenAo-Phys/lrux) in `benchmarks/library_comparison.py`. lrux, in JAX, computes
determinant and Pfaffian ratios under low-rank updates, the inner loop of fermionic Monte Carlo;
TorchPfaffian's `LowRankPfaffian` and `PfaffianUpdater` implement the same Pfaffian updates in PyTorch
([#33](https://github.com/MatchCake/TorchPfaffian/issues/33)), and `--ratios` compares them with lrux.
If you use torchpfaff or lrux, please cite them as well:
```
@software{adams_torchpfaff,
  author  = {Corey Adams},
  title   = {{torchpfaff}: A differentiable Pfaffian for {PyTorch}, with {GPU} support via {Triton}},
  year    = {2026},
  version = {0.2.0},
  url     = {https://github.com/Nuclear-Physics-with-Machine-Learning/PyPfaffian},
}

@misc{chen_lrux,
  author        = {Ao Chen and Christopher Roth},
  title         = {{lrux}: Fast low-rank updates of determinants and {Pfaffians} in {JAX}},
  year          = {2026},
  eprint        = {2602.05255},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.str-el},
  doi           = {10.48550/arXiv.2602.05255},
  url           = {https://arxiv.org/abs/2602.05255},
}
```
