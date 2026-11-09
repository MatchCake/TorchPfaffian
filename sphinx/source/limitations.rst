Limitations
===========

This page lists what TorchPfaffian does not do, or does only partly. Each item says what happens and what
to use instead. Open issues are linked; please comment there if a limitation blocks you.

Input assumptions
-----------------

The strategies assume that the input is skew-symmetric (``A == -A^T``) and do not check it, so a
non-skew input gives a meaningless result without an error. Pass ``check_input=True`` to
:func:`torch_pfaffian.pfaffian` to validate the input first (an ``O(n^2)`` comparison).

Supported dtypes and half precision
-----------------------------------

:func:`~torch_pfaffian.pfaffian` and :func:`~torch_pfaffian.slog_pfaffian` accept ``float16``,
``bfloat16``, ``float32``, ``float64``, ``complex32``, ``complex64`` and ``complex128`` inputs, and raise a
``TypeError`` for any other dtype (integers, booleans, ``float8``): convert those first, for example with
``.to(torch.float64)``.

PyTorch has no LU factorization in ``float16``, ``bfloat16`` and ``complex32``, and every path beyond the
unrolled kernel for ``m <= 6`` needs one; ``complex32`` also lacks most arithmetic kernels. These inputs are
therefore computed in ``float32`` (``complex64`` for ``complex32``) with a ``UserWarning``, and the result is
cast back: at every size for ``complex32`` and in :func:`~torch_pfaffian.slog_pfaffian`, and for
``float16`` and ``bfloat16`` when ``m > 6``, with ``mask_roundoff_gradient=True``, or when the unrolled
kernel's result is not finite. Gradients flow back through the cast. Two consequences:

- A Pfaffian that is finite after the upcast but beyond the range of the input dtype (above 65504 in
  ``float16``) becomes ``inf``, with a ``RuntimeWarning`` when ``check_finite=True``. Use
  :func:`~torch_pfaffian.slog_pfaffian` for such inputs.
- With ``mask_roundoff_gradient=True``, the round-off criterion uses the precision of the upcast dtype. A
  Pfaffian that is zero only up to the round-off of an earlier half-precision computation is not masked.

Pass ``float32`` or ``complex64`` inputs to avoid the warning. The strategies called directly (for example
``RustPfaffianParlettReid.apply``) do not upcast. PyTorch itself marks ``complex32`` as experimental.

``torch.func`` transforms
-------------------------

``grad`` and ``jacrev`` work everywhere. The other transforms need ``check_finite=False``, since the
default ``check_finite=True`` branches on the result. With it, ``vmap``, ``jvp`` (forward-mode AD) and
``vmap(grad)`` all work for ``m <= 6``. For larger matrices, the signed path supports none of them, and
the magnitude path without ``epsilon`` supports ``vmap`` and ``jvp`` but not ``vmap(grad)``. See issue
`#21 <https://github.com/MatchCake/TorchPfaffian/issues/21>`_. Until then, pass batches directly: every
function already takes any leading batch shape.

Thread count and MKL
--------------------

On some MKL builds, any call to ``torch.set_num_threads`` (including ``torch.utils.benchmark.Timer``,
which calls it) makes PyTorch's batched LU return invalid pivots for large batched matrices
(``m >= 256``). Determinants are then wrong without an error, and inverses raise
``RuntimeError: Pivots given to lu_solve must all be greater or equal to 1``. The bug is in PyTorch or
MKL and affects every determinant- and inverse-based path. Set the thread count with the
``OMP_NUM_THREADS`` and ``MKL_NUM_THREADS`` environment variables before starting Python instead. See
issue `#23 <https://github.com/MatchCake/TorchPfaffian/issues/23>`_.

Derivatives at singular matrices
--------------------------------

The first derivative of the signed Pfaffian is exact at invertible and singular matrices. Second
derivatives are exact at invertible matrices only: at a singular matrix (``Pf(A) = 0``), double
backward through ``pfaffian(A)`` is not the true second derivative.

The magnitude ``|Pf(A)|`` is not differentiable where ``Pf(A) = 0``. Its gradient there comes from
round-off, unless ``mask_roundoff_gradient=True`` is passed, which sets it to zero. That mask uses a
numerical criterion: a genuine Pfaffian of round-off size (about ``1e-16`` times the scale of the matrix
in ``float64``) is indistinguishable from an exact zero and is masked too. See
:class:`~torch_pfaffian.strategies.pfaffian_masked_magnitude.PfaffianMaskedMagnitude` for the details.

Tiny pivots
-----------

For ``m > 6`` the linear-domain signed path treats a pivot below ``PfaffianStrategy.EPSILON``
(``1e-30``) as an exact zero, so a Pfaffian that hinges on such a pivot is returned as ``0``.
:func:`torch_pfaffian.slog_pfaffian` uses an exact-zero pivot test and computes these inputs correctly.

Devices and platforms
---------------------

The Rust kernel runs on CPU only. CUDA inputs use the PyTorch implementation of the same algorithm on
the device. Prebuilt wheels with the Rust kernel exist for Linux x86_64, Windows x86_64 and macOS arm64;
on other platforms installing from source needs a Rust toolchain.
