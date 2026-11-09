# Changelog

All notable changes to TorchPfaffian are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/). Add an entry under "Unreleased" in every pull request that
changes the package's behaviour, API, documentation or packaging.

## [Unreleased]

### Added

- `pfaffian(..., sign=False, mask_roundoff_gradient=True)`: an opt-in gradient mask that gives a zero
  gradient to Pfaffians that vanish up to round-off, using a 1-norm and a spectral condition test
  (#19, #25, #26).
- `PfaffianBlockDet` validates the block form `[[0, B], [-B^T, 0]]`, and `pfaffian()` routes large
  block-form inputs to it (#18, #24).
- `CHANGELOG.md`, and project URLs, keywords and classifiers for every supported Python version and
  platform.
- Type stubs for the compiled Rust module (`torch_pfaffian/_rust.pyi`).
- A limitations page in the documentation.

### Changed

- CI: macOS arm64 tests (Python 3.10 and 3.14), mypy is blocking, the tutorial notebook runs on every pull request, the documentation is built
  with warnings as errors on every pull request, and the GitHub Actions run on Node 24.

### Fixed

- `import torch_pfaffian` failed on a fresh install with Python 3.12 or newer: the undeclared
  `importlib_metadata` backport is replaced by the standard library `importlib.metadata`.
- `PfaffianBlockDet` returned wrong values for general skew-symmetric matrices (#18, #24).
- `pfaffian(sign=False)` crashed in the backward on complex input (#17, #22).
- The signed backward was 6 to 300 times slower on numerically singular matrices (#20, #26).
- The singular-matrix detection in the backward no longer depends on the scale of the input (#24).
- The documentation index no longer includes the README twice.
- The routing table in the `pfaffian()` docstring did not render.

## [0.0.5] - 2026-07-13

### Fixed

- The Pfaffian backward for numerically singular inputs (#15).

## [0.0.4] - 2026-06-22

## [0.0.3] - 2026-06-15

## [0.0.2] - 2026-06-15

## [0.0.1] - 2026-06-11

First release on PyPI.

[Unreleased]: https://github.com/MatchCake/TorchPfaffian/compare/0.0.5...HEAD
[0.0.5]: https://github.com/MatchCake/TorchPfaffian/compare/0.0.4...0.0.5
[0.0.4]: https://github.com/MatchCake/TorchPfaffian/compare/0.0.3...0.0.4
[0.0.3]: https://github.com/MatchCake/TorchPfaffian/compare/0.0.2...0.0.3
[0.0.2]: https://github.com/MatchCake/TorchPfaffian/compare/0.0.1...0.0.2
[0.0.1]: https://github.com/MatchCake/TorchPfaffian/releases/tag/0.0.1
