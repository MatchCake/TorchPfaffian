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

- The Rust kernels run AVX2/FMA code when the CPU has it, chosen at run time (the wheels still target
  baseline x86-64); the run-time dispatch of #44 generated none. The rank-2 inverse updates make one fused pass
  over each matrix, write into NumPy arrays (transparent huge pages: 65 page faults instead of 32769 for a
  134 MB batch), and `PfaffianUpdater` updates its inverse in place (copying first an inverse it does not
  own): on one thread, a rank-2 update with its inverse of 64 matrices with `n = 512` takes 0.35 ms per
  matrix instead of 1.32 ms, a row move of 64 with `n = 128` 13.7 us instead of 23.8 us, and an updater move
  of 64 with `n = 128` 6.4 us instead of 35 us (#45).
- The Rust inverse of the backward is a Gauss-Jordan elimination (the same pivots and flags), 2.6x faster
  at `m = 32`, and is used up to `m = 96` on one matrix and `m = 128` on parallel batches.
- `RustPfaffianWithInverse`: with a graph on CPU (and for the small CUDA inputs computed on the host), one
  native call gives the Pfaffian and the inverse of its backward, and the backward only forms the gradient:
  `pfaffian()` with its gradient takes 137 us instead of 204 us for one `8 x 8` matrix, 26 us instead of 64 us
  per matrix for 256 with `m = 32`, and 273 us instead of 485 us for one `8 x 8` matrix on CUDA (#45). For a
  device input, the forward sends the Pfaffians and the gradient template `0.5 * conj(pf(A) A^{-1})^T` in one
  transfer, so the backward is one multiplication on the device, and `pfaffian()`'s finiteness check reads the
  host values instead of copying the result back.
- The host route of CUDA inputs copies them into NumPy memory instead of `tensor.cpu()`, whose pages the
  driver may leave unmapped: their first read took a page fault per 4 KiB, which made the Rust kernel 3 to 7x
  slower in some process states (#45). With the Triton kernel, a cost model of both routes, fitted on the
  benchmark machine and accounting for the host threads, chooses between them; without it the bound is
  unchanged.
- `ParlettReidTriton` delays the rank-2 updates of a panel of 4 or 8 steps and applies them to the trailing
  block in one pass: float64 batches of 256 take 4.3 ms instead of 17 ms at `m = 128` (17 us per matrix) and
  27 ms instead of 91 ms at `m = 256`; float32 is up to 2.3x faster (#45). It is used for every size up to
  `m = 1024`.
- `LowRankPfaffian.ratio`, `ratio_and_inverse`, `ratio_replace` and `ratio_replace_and_inverse` run in one
  Rust call on CPU without a graph to build: on one thread and one `n = 32` matrix, a rank-2 ratio and
  inverse take 18 us instead of 77 us and a row replacement with its inverse 23 us instead of 119 us,
  within 2x of lrux for every single-matrix case. Wide updates of large batches keep the PyTorch path.
- `PfaffianUpdater` applies accepted moves at once with those kernels when nothing is delayed and the
  inverse has at most 2^18 entries (CPU, no graph), which beats delaying there: 16 row moves of one `n = 32`
  matrix take 25 us per move instead of 157 us (lrux's delayed updates: 85 us), `n = 512` 230 us instead of
  363 us. Large batches keep the delayed updates.
- Signed `pfaffian()` on CUDA is up to 67x faster (#34). Inputs with `batch * m^2 <= 2^20`
  (`AUTO_HOST_MAX_BATCH_SQUARE`) are computed by the Rust kernel on the host, with the backward on the device,
  since the elimination on the device is bound by its kernel launches there: `m = 32` takes 0.18 ms instead of
  6.7 ms, `m = 128` 0.40 ms instead of 27 ms on one matrix on an RTX 3070. `PfaffianParlettReid`, which larger
  inputs still use, runs about a dozen operators per elimination step instead of about thirty (one gather and
  one scatter per swap, two in-place multiply-adds, no zeroing), which makes it 1.1 to 1.7x faster on CUDA
  (`m = 256` with a batch of 256: 77 ms instead of 108 ms).
- `ParlettReidTriton`: on CUDA, with Triton installed, `PfaffianParlettReid` eliminates real `float32` and
  `float64` batches with one Triton kernel launch (one program per matrix) instead of a dozen launches per
  step, up to `m = 1024` (and below 2^22 entries in `float64`, where the loop is as fast). On an RTX 3070,
  `float64` batches of 256 take 0.60 ms instead of 4.0 ms at `m = 32` and 2.9 ms instead of 8.0 ms at
  `m = 64`; `float32` is 2 to 35x faster. Complex inputs keep the PyTorch loop (#34).
- `slog_pfaffian()` uses the same Triton kernel on CUDA, still without host synchronization: one `32 x 32`
  matrix takes 0.33 ms instead of 3.95 ms, a batch of 256 0.64 ms instead of 4.0 ms (#34).
- The backward of small CUDA inputs routed to the host (at most 2^13 entries,
  `RustPfaffianParlettReid.HOST_BACKWARD_MAX_NUMEL`) runs on the host too, about 30% faster for a single
  `m = 32` or `m = 64` matrix; a double backward stays on the device (#34).
- `slog_pfaffian()`'s elimination runs about a dozen operators per step, with the same in-place swaps and
  updates as `PfaffianParlettReid`, which makes it 1.4 to 2x faster on CUDA and keeps it free of host
  synchronization (#34).
- A Rust kernel (`flagged_inverse_*`) computes the inverse and singularity flags of the backward on CPU (and
  for small CUDA inputs, on the host), with the adjugate and the gradient formed in NumPy: the signed
  gradient of 256 matrices with `m = 32` takes 19 us per matrix instead of 62 us, and with `m = 64` 93 us
  instead of 296 us. It is used up to `m = 40`, and up to `m = 96` for batches of 8 or more, where it beats
  LAPACK.
- `pfaffian()` calls the Rust strategy's forward directly when no graph is built, skipping autograd's
  `Function.apply`, and tests the finiteness of CPU results with NumPy: a single `8 x 8` Pfaffian takes
  18 us instead of 55 us.
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

### Fixed

- `pfaffian()` gave a wrong gradient to every element of a complex batch in which one element overflowed in
  the linear domain: the recovery recomputed the whole batch as `phase * exp(log|Pf|)`, whose gradient
  dropped the phase's derivative. It now recovers only the non-finite elements, with the full derivative
  `0.5 * Pf * A^{-T}` (`SlogSignedPfaffian`), and the other elements keep the gradient of the linear domain,
  which also keeps the exact adjugate gradient of singular elements in such a batch, real or complex (#28).

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
