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

## Adding cases

Add a matrix family to `FAMILIES` in `cases.py` (for example the input of a bug you just fixed), or a size to
`SIZES`. Case identifiers are stable, so results stay comparable across runs.
