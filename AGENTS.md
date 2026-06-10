# Code style

## Files
- No file-level docstrings. Move any documentation into the class docstring instead.
- For files with only module-level functions, omit the module docstring entirely.

## Comments
- No separator comments between methods or functions (e.g. no `# --- Public interface ---`).
- Inline comments inside function bodies are fine when they clarify non-obvious things (e.g. tensor shape annotations).
- Use em dashes (—) sparingly in docstrings and comments. Prefer plain sentence structure over dashes.

## Naming conventions
- Variable names must be lowercase with underscores (`snake_case`). No single-letter or abbreviated names unless the meaning is unambiguous in context (e.g. `n` for the matrix dimension, `k` for a loop index).
- Choose descriptive names that reflect the role of the variable (e.g. `block_size` instead of `m`, `n_terms` instead of `K`).
- Module-level constants must be `UPPER_SNAKE_CASE`.

## Type hints and docstrings
- All public functions and methods must have full type annotations on every parameter and the return type.
- Docstrings use Sphinx style: `:param name: description`, `:return: description`, `:rtype: type`.

## Tensors and backends
- Any function that accepts a tensor argument must return a tensor with the same backend, dtype, and device as its primary input.
- Tensor functions must support arbitrary leading batch dimensions (expressed as `...` in shape comments, e.g. `(..., 2n, 2n)`).
- Hardware-specific branches (CUDA, platform-only code) that cannot run in CI must be marked `# pragma: no cover`.

## Testing
- Every new class or module-level function must have an associated test class added in the same PR/commit.
- Tests must achieve >98% code coverage for the new code. Numerical comparisons use tolerance constants from `tests/configs.py` (`ATOL_SCALAR_COMPARISON`, `ATOL_MATRIX_COMPARISON`, etc.) — never hardcode tolerance values.
- The `tests/` folder must mirror the package structure (replacing `src/torch_pfaffian/` with `tests/` and prefixing each path segment with `test_`). For example, `src/torch_pfaffian/utils.py` → `tests/test_utils.py`, and `src/torch_pfaffian/strategies/pfaffian_block_det.py` → `tests/test_strategies/test_pfaffian_block_det.py`.
- One test class per file. Each test file contains exactly one `Test*` class.
- All tests must be methods of a `Test*` class. No module-level `test_*` functions.
- Tests must be compatible with pytest-xdist parallel execution: no shared mutable state between tests, no dependency on execution order, and no hardcoded resources (ports, file paths) that would conflict across workers.
- Prefer `np.testing` and `torch.testing` assertion helpers over bare `assert` statements for numerical comparisons. Use `torch.autograd.gradcheck` when testing gradient correctness.

## Class method ordering
Methods within a class must appear in this order:
1. `@staticmethod`
2. `@classmethod`
3. Constructor (`__init__`)
4. Operator / dunder methods (`__call__`, `__repr__`, …)
5. Public methods
6. Protected methods (`_method`)
7. Private methods (`__method`)
8. Properties (`@property`)
