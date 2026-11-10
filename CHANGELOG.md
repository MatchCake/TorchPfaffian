# Changelog

All notable changes to TorchPfaffian are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/). Add an entry under "Unreleased" in every pull request that
changes the package's behaviour, API, documentation or packaging.

## [Unreleased]

### Added

- `LowRankPfaffian`: Pfaffian ratios `Pf(A - u J u^T) / Pf(A)` and updated inverses from `A^{-1}`, in
  `O(n^2 k)`, and the `O(n)` ratio of a row and column replacement (`ratio_replace`), as in
  [lrux](https://github.com/ChenAo-Phys/lrux), plus the ratio of a matrix grown by `2k` rows and columns,
  `Pf(C + B^T A^{-1} B)` (`ratio_extend`). Each has a `..._and_inverse` variant that also returns the
  updated inverse. Batched, differentiable, CPU and CUDA, without host synchronization; shortcuts
  `pfaffian_ratio`, `pfaffian_ratio_replace`, `pfaffian_extension_ratio` (the growth ratio), their
  `_and_inverse` variants, and `skew_identity` (#33). Gradients with respect to a raw `inverse` follow the
  skew-split convention of `pfaffian()` (`SkewGradient`).
- `PfaffianChain`: the Pfaffians of a matrix grown block by block, with the ratios of candidate blocks, for
  qubit-by-qubit sampling of matchgate circuits in `O(n^3)` per sample (MatchCake).
- `PfaffianUpdater`: chains of accepted low-rank updates with delayed inverse updates, per-walker
  acceptance masks, and a running Pfaffian (#33).
- `benchmarks/library_comparison.py --ratios`: Pfaffian ratios against lrux and full recomputation (#33).
- `CompiledLowRankPfaffian`, and `compiled=True` for `PfaffianChain` and `PfaffianUpdater`: the low-rank
  updates run through `torch.compile` (validation stays eager), which halves the inverse updates on small
  inputs on CPU and CUDA after a one-time compilation (#35).
- `benchmarks/library_comparison.py` records the peak memory of each call next to its time, and
  `benchmarks/plot_results.py` plots both against the matrix size, one panel per comparison, and every point
  against the input size in one overview (`--overview`).

### Changed

- `benchmarks/pfapack_comparison.py` is renamed `benchmarks/library_comparison.py`. It also times
  [torchpfaff](https://github.com/Nuclear-Physics-with-Machine-Learning/PyPfaffian) and
  [lrux](https://github.com/ChenAo-Phys/lrux), runs them and TorchPfaffian on the GPU when CUDA is
  available, prints a terminal table (`--markdown` for the old one), saves its results after every input
  and re-renders a saved run with `--render`. The README mentions and cites torchpfaff and lrux.
- Lower per-call overhead of the low-rank updates on small inputs (#35): fewer operators per call (direct
  `bmm`/`mm`, closed forms for rank 2, the small Pfaffian kernel called directly, no dtype conversion when
  none is needed), and `PfaffianUpdater` keeps its delayed updates as two tensors instead of lists. On one
  CPU thread, a rank-2 ratio of an `n = 32` matrix takes about 25 µs instead of 45 µs, a rank-4 ratio 36 µs
  instead of 91 µs.
- The unrolled small Pfaffian kernel (`m <= 10`) caches its index tables per device instead of rebuilding
  them, with host-to-device copies on CUDA, on every call. This also speeds up `pfaffian()` for `m <= 6`: a
  single `6 x 6` matrix takes 29 µs instead of 47 µs on CPU, 91 µs instead of 150 µs on CUDA.

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
