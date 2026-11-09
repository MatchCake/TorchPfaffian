Limitations
===========

This page lists what TorchPfaffian does not do, or does only partly. Each item says what happens and what
to use instead. Open issues are linked; please comment there if a limitation blocks you.

Input assumptions
-----------------

The strategies assume that the input is skew-symmetric (``A == -A^T``) and do not check it, so a
non-skew input gives a meaningless result without an error. Pass ``check_input=True`` to
:func:`torch_pfaffian.pfaffian` to validate the input first (an ``O(n^2)`` comparison).

Half precision (``float16``, ``bfloat16``)
------------------------------------------

Matrices with ``m <= 6`` (``AUTO_SMALL_MAX``) use an exact unrolled kernel that supports every dtype,
gradients included. For larger matrices:

==================================  =================================  ==================================
Call                                ``float16``                        ``bfloat16``
==================================  =================================  ==================================
``pfaffian(A)``                     works (Rust float16 kernel)        ``TypeError`` (no Rust kernel)
``pfaffian(A)`` backward            ``NotImplementedError`` (no LU)    ``TypeError``
``pfaffian(A, sign=False)``         ``NotImplementedError`` (no LU)    ``NotImplementedError`` (no LU)
``slog_pfaffian(A)``                works                              works
``mask_roundoff_gradient=True``     ``TypeError``                      ``TypeError``
==================================  =================================  ==================================

PyTorch has no LU factorization in half precision on CPU, and the determinant- and inverse-based paths
need one. Cast to ``float32`` before the call, and back afterwards if needed.

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
