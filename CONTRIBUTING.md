# Contributing to TorchPfaffian

Thank you for your interest in TorchPfaffian. Bug reports, questions, documentation fixes and code are all
welcome.

## Reporting a bug or asking for a feature

Open an [issue](https://github.com/MatchCake/TorchPfaffian/issues/new/choose) using the bug or feature
template. For a bug, include a minimal script that reproduces it, the full error, and the output of:

```bash
python -c "import torch, torch_pfaffian; print(torch_pfaffian.__version__, torch.__version__, torch.version.cuda)"
```

## Getting help

Ask usage questions in an [issue](https://github.com/MatchCake/TorchPfaffian/issues/new/choose) as well;
they are answered there and help the next person with the same question. You can also write to the
maintainer at jeremie.gince@usherbrooke.ca.

## Setting up a development environment

The project uses [uv](https://docs.astral.sh/uv/) and a Rust toolchain (<https://rustup.rs>) for the native
kernel.

1. Fork and clone the repository:
   ```bash
   gh repo fork MatchCake/TorchPfaffian --clone
   cd TorchPfaffian
   ```
2. Create the environment. The extra selects the PyTorch build: `cpu`, `cu128` or `cu130`.
   ```bash
   uv sync --dev --extra cpu
   ```
3. Build the Rust kernel in release mode (a debug build makes the kernel much slower):
   ```bash
   uv run maturin develop --release -m rust/Cargo.toml
   ```
4. Install the pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```

`requirements.txt` is generated from `uv.lock` by the pre-commit hooks; do not edit it by hand. The package's
own dependencies are the ones in `pyproject.toml`.

## Making a change

1. Create a branch from `dev`.
2. Follow the conventions in [AGENTS.md](AGENTS.md): code style, docstrings, type hints and test layout.
3. Add tests for every change. New code must be covered by the tests (the CI requires more than 98%
   coverage; the project is at 100% line and branch coverage). Use the tolerance constants of
   `tests/configs.py` instead of hard-coded tolerances.
4. Update the docstrings, the README and the documentation pages when the public API, the strategy routing,
   an option or a limitation changes.
5. Add an entry under "Unreleased" in [CHANGELOG.md](CHANGELOG.md).
6. Run the checks below, then open a pull request against `dev` and fill in the template.

## Checks

```bash
uv run ruff format src tests                  # formatting
uv run ruff check src tests                   # linting
uv run mypy src tests                         # typing
uv run pytest --session-timeout=600           # tests and coverage
uv run pytest -o addopts="" --nbmake notebooks  # tutorial notebook
uvx pre-commit run --all-files                # every pre-commit hook
```

For a change that can affect speed or memory, compare it against `dev` with the benchmark tool (see
[benchmarks/README.md](benchmarks/README.md)):

```bash
git fetch origin
uv run python benchmarks/compare.py --bench small
```

To build the documentation locally:

```bash
uv sync --dev --group docs --extra cpu
uv run sphinx-apidoc -f -o ./sphinx/source ./src/torch_pfaffian
uv run make -C sphinx html                    # output in sphinx/build/html
```

Do not call `torch.set_num_threads` in tests or benchmarks: on some MKL builds it corrupts batched LU
factorizations (issue #23). Set `OMP_NUM_THREADS` and `MKL_NUM_THREADS` instead.

## Review

The maintainer reviews every pull request and may ask for changes. Once approved, it is merged into `dev`
and released with the next version.

## Code of Conduct

Everyone taking part in the project is expected to follow the [Code of Conduct](.github/CODE_OF_CONDUCT.md).
