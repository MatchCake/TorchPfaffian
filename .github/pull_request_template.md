# Description

Please provide a brief description of the changes you are making in this pull request.

------------------------------------------------------------------------------------------------------------

# Checklist

Please complete the following checklist when submitting a PR. The PR will not be reviewed until all items are checked.

- [ ] All new features include a unit test.
      Make sure that the tests passed and the coverage is
      sufficient by running
      `uv run pytest --session-timeout=600`.
- [ ] All new functions and code are clearly documented, and the README and documentation pages are
      updated when the public API, the strategy routing, an option or a limitation changes.
- [ ] An entry is added under "Unreleased" in `CHANGELOG.md`.
- [ ] The code passes all pre-commit hooks.
      You can do this by running `uvx pre-commit run --all-files`.
- [ ] The code is type-checked using Mypy.
      You can do this by running `uv run mypy src tests`.
- [ ] For a change that can affect speed or memory, the benchmark shows no regression:
      `uv run python benchmarks/compare.py --bench small`.

See [CONTRIBUTING.md](https://github.com/MatchCake/TorchPfaffian/blob/dev/CONTRIBUTING.md) for the full workflow.
