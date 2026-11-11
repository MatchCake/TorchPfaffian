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

On PyTorch's MKL builds, a call to ``torch.set_num_threads(n)`` with ``n >= 2`` (including through
``torch.utils.benchmark.Timer``, which calls it) corrupts the batched LU factorizations on CPU for the rest
of the process: from about ``m = 150``, every matrix of a batch of two or more that needs row swaps comes out
wrong. PyTorch splits the batch over its threads, each calling MKL's ``getrf``, which then starts its own
team of ``n`` threads, and the nested teams conflict. Determinants and inverses are mostly wrong without an
error, and sometimes inverses raise ``RuntimeError: Pivots given to lu_solve must all be greater or equal to
1``. The bug is in PyTorch and MKL (`pytorch/pytorch#142815 <https://github.com/pytorch/pytorch/issues/142815>`_,
`#141358 <https://github.com/pytorch/pytorch/issues/141358>`_).

TorchPfaffian's determinant- and inverse-based paths detect it and raise a ``RuntimeError`` instead of
returning wrong values: for a batched LU of at least two CPU matrices of size ``m >= 96`` with more than one
thread, :class:`~torch_pfaffian.kernels.checked_linalg.CheckedLinalg` factors a small probe the same way and
verifies it. The verdict is cached per probe size (``min(m, 256)``), dtype and thread count, so the probe runs
once for each (about 1 ms) and later calls only look the verdict up. A later ``torch.set_num_threads`` with
another count is detected, but one that sets the current count again, which also corrupts the factorizations,
is not seen once a verdict is cached. Set the thread count with the ``OMP_NUM_THREADS`` environment variable
before starting Python instead of calling ``torch.set_num_threads``, or call ``torch.set_num_threads(1)``. See
issue `#23 <https://github.com/MatchCake/TorchPfaffian/issues/23>`_; the check is to be removed once PyTorch
fixes the bug (`#48 <https://github.com/MatchCake/TorchPfaffian/issues/48>`_).

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

The Rust kernel runs on CPU only. A small signed CUDA input (``batch * m^2`` up to ``2^20``; for real inputs, when a cost model of both routes,
which accounts for the host threads, estimates the host faster) is copied to the host,
computed by the Rust kernel and its Pfaffian copied back, since the elimination on the device is bound by its
kernel launches at those sizes; this synchronizes with the device, and the backward runs on the device. Larger
CUDA inputs, and inputs during a CUDA graph capture, use the PyTorch implementation of the same algorithm on the
device; in ``float32`` and ``float64`` its elimination runs as one Triton kernel when Triton is installed (it is,
with PyTorch's CUDA wheels), and complex inputs use the PyTorch loop. :func:`torch_pfaffian.slog_pfaffian` always runs on the device. Prebuilt wheels with the Rust kernel exist for Linux x86_64, Windows x86_64 and macOS arm64;
on other platforms installing from source needs a Rust toolchain.
