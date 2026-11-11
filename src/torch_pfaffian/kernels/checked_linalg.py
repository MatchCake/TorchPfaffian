import functools

import torch


class CheckedLinalg:
    """
    The LU-based ``torch.linalg`` functions TorchPfaffian uses, raising instead of returning corrupted results.

    PyTorch factors a batch of matrices on CPU by splitting the batch over its threads, each calling MKL's
    ``getrf``, which above a size threshold starts its own team of threads. After ``torch.set_num_threads(n)``
    with ``n >= 2``, MKL runs ``n`` threads inside each of PyTorch's, and the nested teams corrupt the
    factorizations of the matrices that need row swaps: from about ``m = 150``, every such matrix of a batch of
    two or more comes out wrong (pytorch/pytorch#142815, #141358). ``inv`` and ``solve`` then sometimes raise on
    invalid pivots, but mostly the pivots look valid and the results are wrong without an error. A single matrix,
    a single thread and a thread count set at launch (``OMP_NUM_THREADS``) are not affected, nor is ``lu_solve``
    given correct factors.

    Each function here first checks the batched factorization, for the inputs that can be affected: a CPU batch
    of at least two matrices of size at least ``MIN_SIZE``, with more than one thread. The check factors a probe,
    a batch of two Gaussian matrices of the input's dtype and of size ``min(m, PROBE_MAX_SIZE)``, the same way, and
    verifies the factorization ``P L U`` of the second on a vector ``v`` against the componentwise backward-error
    bound of LU with partial pivoting, ``|A v - P L U v| <= c m eps (|P| |L| |U| + |A|) |v|``. A failure raises a
    :class:`RuntimeError` that explains the cause and the workaround. The verdict is cached per probe size, dtype
    and thread count, so the probe runs once for each and later calls cost a dictionary lookup; a call to
    ``torch.set_num_threads`` with another count runs it again, but one with the current count, which also
    corrupts the factorizations, is not seen after the first verdict. Other inputs go straight to PyTorch.

    This is a workaround to remove once PyTorch fixes the issues
    (https://github.com/MatchCake/TorchPfaffian/issues/48).
    """

    # Smallest size checked. The corruption starts at about m = 150, where MKL's getrf starts a team of threads
    # (measured on a Ryzen 9 3900X with MKL 2024.2, and m = 151 in pytorch/pytorch#142815).
    MIN_SIZE = 96
    # Largest probe: larger inputs are judged from a probe of this size, which bounds the check's cost; it must stay
    # above the size where the corruption starts.
    PROBE_MAX_SIZE = 256
    # The residual must stay below this many times m * eps times its componentwise bound; a corrupted
    # factorization misses it by more than ten orders of magnitude.
    RESIDUAL_FACTOR = 64
    ISSUE_URLS = (
        "https://github.com/pytorch/pytorch/issues/142815",
        "https://github.com/pytorch/pytorch/issues/141358",
    )

    @staticmethod
    def check(matrix: torch.Tensor) -> None:
        """
        Raise if the batched LU factorization of ``matrix`` would be corrupted, judged from that of a probe.

        :param matrix: Matrices of shape ``(..., m, m)``.
        :raises RuntimeError: If ``matrix`` is a CPU batch of at least two matrices of size at least ``MIN_SIZE``
            and the batched factorization of a probe of its dtype is corrupted with the current thread count.
        """
        size = matrix.shape[-1]
        if matrix.device.type != "cpu" or matrix.dim() < 3 or size < CheckedLinalg.MIN_SIZE:
            return
        if matrix.numel() // (size * size) < 2:
            return
        threads = torch.get_num_threads()
        if threads < 2:
            return
        if CheckedLinalg._is_corrupted(min(size, CheckedLinalg.PROBE_MAX_SIZE), matrix.dtype, threads):
            raise RuntimeError(
                f"Batched LU factorizations of {matrix.dtype} matrices on CPU are corrupted in this process, so "
                "det, slogdet, inv and solve would return wrong results, mostly without an error. PyTorch's MKL "
                "backend corrupts them after torch.set_num_threads(n) with n >= 2 "
                f"({', '.join(CheckedLinalg.ISSUE_URLS)}). Set the thread count at launch with the OMP_NUM_THREADS "
                "environment variable instead of calling torch.set_num_threads, or call torch.set_num_threads(1)."
            )

    @staticmethod
    def det(matrix: torch.Tensor) -> torch.Tensor:
        """
        :func:`torch.linalg.det`, after :meth:`check`.

        :param matrix: Matrices of shape ``(..., m, m)``.
        :return: The determinants, of shape ``(...,)``.
        :rtype: torch.Tensor
        :raises RuntimeError: See :meth:`check`.
        """
        CheckedLinalg.check(matrix)
        return torch.linalg.det(matrix)

    @staticmethod
    def slogdet(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :func:`torch.linalg.slogdet`, after :meth:`check`.

        :param matrix: Matrices of shape ``(..., m, m)``.
        :return: The signs (phases for complex inputs) and the logarithms of the absolute determinants, each of
            shape ``(...,)``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        :raises RuntimeError: See :meth:`check`.
        """
        CheckedLinalg.check(matrix)
        sign, log_abs = torch.linalg.slogdet(matrix)
        return sign, log_abs

    @staticmethod
    def inv(matrix: torch.Tensor) -> torch.Tensor:
        """
        :func:`torch.linalg.inv`, after :meth:`check`.

        :param matrix: Invertible matrices of shape ``(..., m, m)``.
        :return: The inverses, of shape ``(..., m, m)``.
        :rtype: torch.Tensor
        :raises RuntimeError: See :meth:`check`, or as :func:`torch.linalg.inv` on a singular input.
        """
        CheckedLinalg.check(matrix)
        return torch.linalg.inv(matrix)

    @staticmethod
    def inv_ex(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :func:`torch.linalg.inv_ex`, after :meth:`check`.

        :param matrix: Matrices of shape ``(..., m, m)``.
        :return: The inverses, of shape ``(..., m, m)``, and the LAPACK status of each matrix, of shape ``(...,)``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        :raises RuntimeError: See :meth:`check`.
        """
        CheckedLinalg.check(matrix)
        inverse, info = torch.linalg.inv_ex(matrix)
        return inverse, info

    @staticmethod
    def lu_factor_ex(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :func:`torch.linalg.lu_factor_ex`, after :meth:`check`.

        :param matrix: Matrices of shape ``(..., m, m)``.
        :return: The packed factors, of shape ``(..., m, m)``, the pivots, of shape ``(..., m)``, and the LAPACK
            status of each matrix, of shape ``(...,)``.
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        :raises RuntimeError: See :meth:`check`.
        """
        CheckedLinalg.check(matrix)
        factors, pivots, info = torch.linalg.lu_factor_ex(matrix)
        return factors, pivots, info

    @staticmethod
    @functools.lru_cache(maxsize=1024)
    def _is_corrupted(size: int, dtype: torch.dtype, threads: int) -> bool:
        # The verdict of the probe, cached: the corruption depends on the process's threading state, not on the
        # input, and ``threads`` keys the cache so that a new thread count runs the probe again.
        return CheckedLinalg._probe_is_corrupted(size, dtype)

    @staticmethod
    def _probe_is_corrupted(size: int, dtype: torch.dtype) -> bool:
        # Factor two Gaussian matrices (their factorization swaps rows, and their condition number, about m, keeps
        # it accurate) as a batch and verify the second factorization, P L U = A, on a vector v.
        generator = torch.Generator().manual_seed(size)
        probe = torch.randn(2, size, size, dtype=torch.float64, generator=generator).to(dtype)
        vector = torch.randn(size, 1, dtype=torch.float64, generator=generator).to(dtype)
        factors, pivots, _ = torch.linalg.lu_factor_ex(probe)
        if not bool(((pivots >= 1) & (pivots <= size)).all()):
            return True
        permutation, lower, upper = torch.lu_unpack(factors[-1], pivots[-1])
        residual = (probe[-1] @ vector - permutation @ (lower @ (upper @ vector))).abs()
        # P |L| |U| |v| bounds the backward error of the factorization (|P| = P), |A| |v| the rounding of A v.
        bound = permutation.real @ (lower.abs() @ (upper.abs() @ vector.abs())) + probe[-1].abs() @ vector.abs()
        tolerance = CheckedLinalg.RESIDUAL_FACTOR * size * torch.finfo(dtype).eps
        return not bool((residual <= tolerance * bound).all())
