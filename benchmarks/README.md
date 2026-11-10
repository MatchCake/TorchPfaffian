# Benchmarks

`compare.py` compares the speed and peak memory of your working tree against a baseline revision (default
`origin/dev`) and prints the cases that changed. Run it before opening a pull request:

```bash
git fetch origin
python benchmarks/compare.py                      # medium mode (about 5 minutes), CUDA if available
python benchmarks/compare.py --bench small        # about 2 minutes; the mode CI runs on every pull request
python benchmarks/compare.py --bench big --cuda true
python benchmarks/compare.py --baseline v0.0.5 --families block block_singular
```

Accuracy is the job of the unit tests. The benchmark only checks that both versions return the same values
and gradients (`disagree` flag), which covers sizes the tests cannot afford. In the singular families
(`general_singular`, `block_singular`) the Pfaffian is zero up to round-off, so only their exact signed
gradient is compared: their values legitimately change with the order of operations.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--baseline` | `origin/dev` | git revision to compare against |
| `--bench` | `medium` | `small`, `medium` or `big`: sizes, dtypes and time budget |
| `--cuda` | `auto` | `auto` benchmarks CUDA too when available, `true` requires it, `false` skips it |
| `--threads` | half the cores | threads of the benchmarked processes (torch, MKL and the Rust kernel) |
| `--threshold` | `0.25` | relative change reported (ratios above 1.25 or below 1/1.25) |
| `--rounds` | `2` | alternating rounds over every case (at least 1), best time kept |
| `--confirm-rounds` | `3` | extra rounds re-running only the cases whose time changed |
| `--families` | all | matrix families to run (see `cases.py`) |

## How it works

- Both versions are built as release wheels with `maturin`, Rust kernel included: the baseline from a
  temporary `git worktree`, the current version from the working tree (uncommitted changes included). The
  wheels are unpacked and imported through `PYTHONPATH` by a fresh process per version, with the same
  Python and torch.
- The versions run family by family in alternating processes, and the order flips every round, so drifting
  machine load affects both alike. Each case keeps its fastest time over the rounds. A time change is
  reported only when the minimum and the median both move past the threshold and the absolute change
  exceeds 100 µs (below that, identical processes still differ by up to 3x on a busy machine). Cases whose
  time or memory changed are then re-run (`--confirm-rounds`): the best value can only come down towards
  the true one, so a real regression stays flagged and noise does not.
- Peak memory is the peak resident set size of the case on Linux CPU (reset through `/proc/self/clear_refs`),
  and `torch.cuda.max_memory_allocated` on CUDA. Changes below 2 MiB are ignored.
- Threads are set through `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `RAYON_NUM_THREADS` (the Rust kernel),
  never `torch.set_num_threads`, which corrupts batched LU on some MKL builds (issue #23).
- Inputs are scaled so that `|Pf|` stays near 1 at every size: float32 cases neither overflow nor underflow,
  so they time the regular path rather than the overflow recovery.
- Each case is announced before it runs and recorded when done. A Python exception, a time limit, a crash
  of the process (segfault, abort, out-of-memory kill) or a hang (a watchdog kills the process) marks only
  that case, and a new process resumes with the next one. Cases that need a feature the baseline lacks are
  `unsupported` in the baseline, which is not a regression.

The summary groups the cases by flag. Regressions: `error` (the current version fails, crashes, times out
or drops a case), `disagree`, `slower`, `heavier`. Others: `fixed` (the baseline fails and the current
version works), `faster`, `lighter`, `failing` (both fail). The full table and raw records are written to
`benchmarks/results/<timestamp>/` (ignored by git). Timings on a busy
machine are noisy; the header warns when the load average is high, so re-run on an idle machine before
trusting a small change.

## Continuous integration

`.github/workflows/benchmark.yml` runs `--bench small` on every pull request against its base branch and
writes the summary to the job page. A regression only raises a warning, with the command to confirm it
locally, because shared runners are noisy; so does a run that could not complete.

The tests of the tool itself run with `python -m pytest -o addopts="" benchmarks/tests`.

## Comparison with pfapack, torchpfaff and lrux

`library_comparison.py` times TorchPfaffian against three other libraries and checks that they agree with
pfapack:

- [pfapack](https://github.com/basnijholt/pfapack), the reference Pfaffian library (NumPy, CPU, one matrix at
  a time);
- [torchpfaff](https://github.com/Nuclear-Physics-with-Machine-Learning/PyPfaffian), a differentiable PyTorch
  Pfaffian with a Triton kernel on CUDA;
- [lrux](https://github.com/ChenAo-Phys/lrux), a JAX library of low-rank Pfaffian updates, whose full Pfaffian
  (from its dependency [fermix](https://github.com/ChenAo-Phys/fermix), batched determinants and Pfaffians
  with JAX/Pallas kernels on GPU) is timed jitted, in double precision.

It uses random matrices from the `general` family of `cases.py`, in float64 and complex128. The PyTorch and JAX
libraries run on CPU and, when CUDA is available, on the GPU too, so the table shows whether the GPU beats
pfapack on CPU. None of these libraries is a dependency of the project: pfapack ships wheels only for some
Python versions, and torchpfaff needs Python 3.12 or newer and is installed from GitHub. Run the script from
a separate environment, outside the repository; a library that is not installed is not timed:

```bash
uv venv --python 3.12 ~/.venvs/pfaffian-libraries
uv pip install --python ~/.venvs/pfaffian-libraries torch --index-url https://download.pytorch.org/whl/cpu  # CPU only
# or, for the GPU runs: uv pip install --python ~/.venvs/pfaffian-libraries torch   (the CUDA build, with triton)
uv pip install --python ~/.venvs/pfaffian-libraries pfapack . \
    "torchpfaff @ git+https://github.com/Nuclear-Physics-with-Machine-Learning/PyPfaffian"
uv pip install --python ~/.venvs/pfaffian-libraries lrux   # add "jax[cuda13]" to time lrux on the GPU
~/.venvs/pfaffian-libraries/bin/python benchmarks/library_comparison.py --threads 1
~/.venvs/pfaffian-libraries/bin/python benchmarks/library_comparison.py --threads 8 --sizes 32 128
```

It prints a terminal table, one block per input, with the time per matrix, the speed-up over pfapack's C
implementation, the largest relative difference from it, and a star on the fastest method of each input.
`--markdown` prints a Markdown table instead. Methods ending in `(CUDA)` run on the GPU, with the time of the
transfer of the result back to the host included. A method that fails on an input is skipped with a note
(torchpfaff does not support complex matrices). fermix's fast kernels need CUDA, so on CPU lrux runs
fermix's generic `jax.numpy` fallback, and JAX's CPU threads are not pinned by `--threads`. JAX is told not
to preallocate GPU memory, so torch and JAX share the GPU. The raw records go to
`benchmarks/results/libraries-<timestamp>.json`, saved after every input; `--render <file>` prints the table
of a saved run, including an interrupted one. A method whose single call takes more than
`--max-call-seconds` (10 s) is timed from that call and skipped on larger inputs, and torchpfaff's CUDA path
is skipped above `--torchpfaff-cuda-max-size` (64): its Triton kernel takes about 100 s to compile at m = 64
and several minutes or more beyond (compiled kernels are then cached on disk). Progress is printed per method. The documentation's performance page shows a run of it.
Without pfapack, nothing is compared, but the other libraries are still timed.

### Pfaffian ratios (`--ratios`)

`--ratios` times Pfaffian ratios under low-rank updates instead of full Pfaffians: TorchPfaffian's
`LowRankPfaffian` and `PfaffianUpdater` against lrux's `pf_lru` and `pf_lru_delayed` (jitted and vmapped),
and against forming each updated matrix and computing its Pfaffian (pfapack, the reference, and
TorchPfaffian). Every method starts from the same matrices and their inverse. The `update` column is:

| update | move |
|---|---|
| `rank 2`, `rank 4` | one dense update `A - u J u^T`, with `u` of shape `(n, 2)` or `(n, 4)` |
| `row` | one row and column replaced, the usual Monte Carlo move |
| `row x16` | 16 successive accepted row replacements, with immediate and delayed (`T = 4`) inverse updates; times are per update |

`ratio` rows compute the ratio only; `ratio + inverse` rows also update the inverse, which the next move needs.
Rows ending in `, compiled` use `CompiledLowRankPfaffian` (and `PfaffianUpdater(..., compiled=True)`); their
first call, which compiles, is the agreement check and is not timed.
The defaults are `--sizes 32 128 512 --batches 1 64`, and `--updates` selects the moves:

```bash
~/.venvs/pfaffian-libraries/bin/python benchmarks/library_comparison.py --ratios --threads 1
~/.venvs/pfaffian-libraries/bin/python benchmarks/library_comparison.py --ratios --updates row "row x16" --sizes 512
```

The raw records go to `benchmarks/results/ratios-<timestamp>.json`.

### Time and memory plots

Every measurement also records the peak memory of one call of the whole batch: on CPU, the peak resident memory
that the call adds to the process (Linux only, at a resolution of one 4 KiB page, so that it covers NumPy, torch
and JAX alike); on CUDA, the peak of torch's allocator, which does not see JAX, so lrux on CUDA has no memory
record. `plot_results.py` (seaborn and matplotlib, in the `dev` group) plots the time per matrix and the peak
memory against the matrix size, with one figure per batch size (and per dtype for `--ratios`) and one panel per
comparison: the Pfaffian and the Pfaffian with its gradient per dtype, or the ratio and the ratio with the updated
inverse per low-rank update, with the Pfaffian of the updated matrix as the baseline in every panel. The color is
the library, and the line style the device. The figures are saved as PNGs next to the results file:

```bash
~/.venvs/pfaffian-libraries/bin/python benchmarks/plot_results.py benchmarks/results/libraries-<timestamp>.json
~/.venvs/pfaffian-libraries/bin/python benchmarks/plot_results.py benchmarks/results/ratios-*.json --output-directory plots
```

`--overview` also saves `overview.png`, one figure that compares the packages over every point of every file
given, with one row per device (CPU, then CUDA) and the time and the peak memory side by side. On each input and
device, each package is taken at its best method, and its time and peak memory are divided by those of the best
package there: 1x is the best, 3x three times slower or heavier. The x-axis is the input size in bytes, which
means the same for every operation, dtype and batch size, and each package has its own color and marker shape.
torch's CUDA allocator counts cuBLAS's workspace (about 8 MiB) in every call that multiplies matrices, so small
CUDA inputs read about 8 MB.

## Adding cases

Add a matrix family to `FAMILIES` in `cases.py` (for example the input of a bug you just fixed), or a size to
`SIZES`. Case identifiers are stable, so results stay comparable across runs.
