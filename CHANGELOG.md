# Changelog

All notable changes to TorchPfaffian are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/). Add an entry under "Unreleased" in every pull request that
changes the package's behaviour, API, documentation or packaging.

## [Unreleased]

## [0.0.6] - 2026-11-09

### Added

- `pfaffian(..., sign=False, mask_roundoff_gradient=True)`: an opt-in gradient mask that gives a zero
  gradient to Pfaffians that vanish up to round-off, using a 1-norm and a spectral condition test
  (#19, #25, #26).
- `PfaffianBlockDet` validates the block form `[[0, B], [-B^T, 0]]`, and `pfaffian()` routes large
  block-form inputs to it (#18, #24).
- `CHANGELOG.md`, and project URLs, keywords and classifiers for every supported Python version and
  platform.
- Type stubs for the compiled Rust module (`torch_pfaffian/_rust.pyi`).
- `complex32` inputs, computed in `complex64` with a `UserWarning` (they failed on almost every path).
- Documentation: a limitations page, a performance page comparing with pfapack, and a tutorial on
  domino tilings and dimer probabilities.
- `benchmarks/pfapack_comparison.py`, which times TorchPfaffian against pfapack and checks that they agree.

### Changed

- `pfaffian()` and `slog_pfaffian()` compute `float16` and `bfloat16` inputs in `float32` and cast the
  result back, with a `UserWarning`: PyTorch's linear algebra does not support those dtypes. Before, every
  backward, the magnitude path and the `bfloat16` signed path failed for `m > 6`, and the `slog_pfaffian`
  backward failed at every size. The `float16` signed forward for `m > 6`, which ran natively, is now
  computed in `float32` too, which is more accurate. Inputs with `m <= 6` still run natively, unless the
  result is not finite or `mask_roundoff_gradient=True`.
- `pfaffian()` and `slog_pfaffian()` raise a `TypeError` for unsupported dtypes (integers, booleans,
  `float8`). Before, integer inputs returned `float32` for `m <= 6` and failed with unrelated errors
  otherwise.
- CI: macOS arm64 tests (Python 3.10 and 3.14), mypy is blocking, the notebooks run on every pull
  request, the documentation is built with warnings as errors on every pull request, and the GitHub
  Actions run on Node 24.

### Fixed

- `import torch_pfaffian` failed on a fresh install with Python 3.12 or newer: the undeclared
  `importlib_metadata` backport is replaced by the standard library `importlib.metadata`.
- `PfaffianBlockDet` returned wrong values for general skew-symmetric matrices (#18, #24).
- `pfaffian(sign=False)` crashed in the backward on complex input (#17, #22).
- The signed backward was 6 to 300 times slower on numerically singular matrices (#20, #26).
- The singular-matrix detection in the backward no longer depends on the scale of the input (#24).
- The documentation index no longer includes the README twice.
- The routing table in the `pfaffian()` docstring did not render.
- The release workflow built the macOS wheel on the retired `macos-14` runner. It now uses
  `macos-latest` and targets macOS 11.0 or newer.
- `pfaffian()` crashed on an empty batch when `m` was 0, 1, 3 or 5, or when `m` was odd and `epsilon` was
  given.
- The `slog_pfaffian()` backward raised on a matrix with a non-finite entry; it now returns a non-finite
  gradient for that element only.
- `PfaffianBlockDet` with `EXACT_SINGULAR_GRAD = False` gave a round-off determinant times a huge inverse
  as the gradient of a numerically singular block; it now gives zero, like the general strategies. As there,
  an invertible block with an LU pivot ratio below `n * eps` also gets zero in this opt-in mode.
- The code used `Tensor.all` over several dimensions, which needs PyTorch 2.2, although `torch>=2.1` is
  declared, and the unrolled kernel used a half-precision dot product, which PyTorch 2.1 lacks on CPU. The
  test suite now passes with PyTorch 2.1.2.

## [0.0.5] - 2026-07-13

### Fixed

- The Pfaffian backward for numerically singular inputs (#15).

## [0.0.4] - 2026-06-22

## [0.0.3] - 2026-06-15

## [0.0.2] - 2026-06-15

## [0.0.1] - 2026-06-11

First release on PyPI.

[Unreleased]: https://github.com/MatchCake/TorchPfaffian/compare/0.0.6...HEAD
[0.0.6]: https://github.com/MatchCake/TorchPfaffian/compare/0.0.5...0.0.6
[0.0.5]: https://github.com/MatchCake/TorchPfaffian/compare/0.0.4...0.0.5
[0.0.4]: https://github.com/MatchCake/TorchPfaffian/compare/0.0.3...0.0.4
[0.0.3]: https://github.com/MatchCake/TorchPfaffian/compare/0.0.2...0.0.3
[0.0.2]: https://github.com/MatchCake/TorchPfaffian/compare/0.0.1...0.0.2
[0.0.1]: https://github.com/MatchCake/TorchPfaffian/releases/tag/0.0.1
