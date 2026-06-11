# TorchPfaffian: PyTorch-Based Pfaffian Computation

[![Star on GitHub](https://img.shields.io/github/stars/MatchCake/TorchPfaffian.svg?style=social)](https://github.com/MatchCake/TorchPfaffian/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/MatchCake/TorchPfaffian?style=social)](https://github.com/MatchCake/TorchPfaffian/network/members)
[![Python 3.6](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![downloads](https://img.shields.io/pypi/dm/TorchPfaffian)](https://pypi.org/project/TorchPfaffian)
[![PyPI version](https://img.shields.io/pypi/v/TorchPfaffian)](https://pypi.org/project/TorchPfaffian)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

![Tests Workflow](https://github.com/MatchCake/TorchPfaffian/actions/workflows/tests.yml/badge.svg)
![Dist Workflow](https://github.com/MatchCake/TorchPfaffian/actions/workflows/build_dist.yml/badge.svg)
![Doc Workflow](https://github.com/MatchCake/TorchPfaffian/actions/workflows/docs.yml/badge.svg)
![Coverage Badge Workflow](https://github.com/MatchCake/TorchPfaffian/actions/workflows/coverage_badge.yml/badge.svg)


# Description

TorchPfaffian is a Python package for efficiently computing the Pfaffian of skew-symmetric matrices using PyTorch.
Designed as a PyTorch-based alternative to [pfapack](https://github.com/basnijholt/pfapack), it enables GPU
acceleration and supports automatic differentiation, making it particularly useful in physics, quantum computing,
and machine learning applications.

## Features:
- Efficient Pfaffian computation for skew-symmetric matrices
- GPU acceleration via PyTorch
- Support for automatic differentiation
- Seamless integration with PyTorch tensors




## Installation

With `python` and `pip` installed, run the following commands to install TorchPfaffian:
```bash
pip install torchpfaffian
```

For development, this project uses [uv](https://docs.astral.sh/uv/). Clone the repository and set up the
environment with:
```bash
uv sync --dev --extra cpu
```
Use the `cu128` or `cu130` extra instead of `cpu` to install a CUDA-enabled build of PyTorch. See
[.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) for the full contribution workflow.

### Native acceleration (optional)

A Rust-accelerated signed-Pfaffian strategy (`RustPfaffianParlettReid`) is available when the
package is built with its native extension. Building from source requires a Rust toolchain
(<https://rustup.rs>); the project builds with [maturin](https://www.maturin.rs):

```bash
uv run maturin develop --release -m rust/Cargo.toml
```

Use `--release` for an optimized build: `maturin develop` compiles in debug mode by default,
which makes the Rust kernel much slower. Installing a prebuilt wheel (or `maturin build`) is already
optimized, so this only matters for local development builds.

If the native extension is not present, the package still works using the pure-Python strategies.


## Usage

```python
import torch

from torch_pfaffian import pfaffian

# Any skew-symmetric matrix of shape (..., 2n, 2n).
matrix = torch.tensor([[0.0, -3.0], [3.0, 0.0]])

pf = pfaffian(matrix)                 # signed Pfaffian (default)
magnitude = pfaffian(matrix, sign=False)  # |pf|, using the faster det-based path
```

`pfaffian()` selects a strategy from the input: `sign=True` (the default) returns the
**signed** Pfaffian, using the native `RustPfaffianParlettReid` when the extension is built and
falling back to the pure-Python `PfaffianParlettReid` otherwise; `sign=False` returns the magnitude
using a determinant-based strategy (`PfaffianFDBPf` when gradients are needed, otherwise
`PfaffianDet`). For explicit strategy selection, use `get_pfaffian_function(name)`.


# Important Links
- Documentation at [https://MatchCake.github.io/TorchPfaffian/](https://MatchCake.github.io/TorchPfaffian/).
- Github at [https://github.com/MatchCake/TorchPfaffian/](https://github.com/MatchCake/TorchPfaffian/).




# Found a bug or have a feature request?
- [Click here to create a new issue.](https://github.com/MatchCake/TorchPfaffian/issues/new)


## License
[Apache License 2.0](LICENSE)

## Acknowledgements


## Citation
Repository:
```
@misc{torchpfaffian_Gince2025,
  title={Torch Pfaffian},
  author={Jérémie Gince},
  year={2025},
  publisher={Université de Sherbrooke},
  url={https://github.com/MatchCake/TorchPfaffian},
}
```
