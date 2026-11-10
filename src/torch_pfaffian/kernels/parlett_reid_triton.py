import functools
from typing import Any

import torch


class ParlettReidTriton:
    """
    Parlett-Reid elimination of a batch of real skew-symmetric matrices in one Triton kernel launch.

    :class:`~torch_pfaffian.strategies.PfaffianParlettReid` on CUDA launches about a dozen small kernels per
    elimination step, so its run time is set by the ``m / 2`` steps' launches rather than by the arithmetic.
    Here one Triton program eliminates one matrix: it finds the pivot, swaps its row and column into place and
    updates the trailing block (in tiles of rows), step after step, in place in global memory, with a barrier
    between the steps' reads and writes. The batch is the grid, so the whole batch is one launch. The pivoting,
    the update order and the left-over entries match the PyTorch loop, whose post-processing (sign, product,
    zero-pivot test) is shared.

    Triton has no complex arithmetic, so only ``float32`` and ``float64`` are supported, on CUDA, up to
    ``MAX_DIMENSION``; :meth:`supports` tells whether an input qualifies, and Triton is imported only then.
    """

    MAX_DIMENSION = 1024
    # In float64 the kernel is bound by memory bandwidth and GeForce float64 arithmetic: from this many entries
    # (batch * m^2) the PyTorch loop's batched kernels are as fast (RTX 3070: m = 128 with a batch of 256, 15 ms
    # both; m = 256 with 256, 91 ms against 77 ms). float32 is faster with the kernel at every size (2 to 35x).
    FLOAT64_MAX_NUMEL = 2**22
    DTYPES = (torch.float32, torch.float64)
    # Entries of the trailing block one program updates at once (a tile of rows times the padded row length).
    TILE_ENTRIES = 4096

    @staticmethod
    def supports(matrix: torch.Tensor) -> bool:
        """
        Whether :meth:`eliminate` handles the input.

        :param matrix: Matrices of shape ``(..., m, m)``.
        :return: ``True`` for a CUDA ``float32`` or ``float64`` input with even ``2 < m <= MAX_DIMENSION`` (and fewer
            than ``FLOAT64_MAX_NUMEL`` entries in ``float64``), when Triton is installed.
        :rtype: bool
        """
        dimension = matrix.shape[-1]
        if matrix.dtype not in ParlettReidTriton.DTYPES:
            return False
        if dimension % 2 or not 2 < dimension <= ParlettReidTriton.MAX_DIMENSION:
            return False
        if matrix.dtype == torch.float64 and matrix.numel() >= ParlettReidTriton.FLOAT64_MAX_NUMEL:
            return False
        if matrix.device.type != "cuda":
            return False
        return ParlettReidTriton._kernel() is not None  # pragma: no cover - needs a GPU

    @staticmethod
    def eliminate(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover - needs a GPU
        """
        Eliminate every matrix of the batch, as the loop of :class:`~torch_pfaffian.strategies.PfaffianParlettReid`.

        :param matrix: Skew-symmetric matrices of shape ``(..., m, m)`` that :meth:`supports`.
        :return: The eliminated matrices, of shape ``(batch, m, m)``, whose super-diagonal entries ``(2i, 2i + 1)``
            are the factors of the Pfaffian and whose entries ``(2i + 1, 2i)`` are the pivots, and the number of row
            swaps of each matrix, of shape ``(batch,)``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        import triton  # type: ignore[import-not-found, unused-ignore]

        dimension = matrix.shape[-1]
        working = matrix.reshape(-1, dimension, dimension).contiguous().clone()  # eliminated in place
        batch = working.shape[0]
        swaps = torch.zeros(batch, dtype=torch.int32, device=matrix.device)
        block = triton.next_power_of_2(dimension)
        tile = max(1, min(block, ParlettReidTriton.TILE_ENTRIES // block))
        kernel = ParlettReidTriton._kernel()
        kernel[(batch,)](
            working, swaps, dimension, BLOCK=block, TILE=tile, num_warps=4 if block <= 128 else 8, enable_fp_fusion=True
        )
        return working, swaps

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _kernel() -> Any:  # pragma: no cover - needs a GPU
        # The jitted kernel, built on first use; None without Triton.
        try:
            import triton  # type: ignore[import-not-found, unused-ignore]
            import triton.language as tl  # type: ignore[import-not-found, unused-ignore]
        except ImportError:
            return None

        @triton.jit
        def parlett_reid(work_ptr, swaps_ptr, size, BLOCK: tl.constexpr, TILE: tl.constexpr):
            matrix = work_ptr + tl.program_id(0).to(tl.int64) * size * size
            index = tl.arange(0, BLOCK)
            inside = index < size
            swaps = 0
            for column in range(0, size - 2, 2):
                # Pivot: the largest |A[p, column]| for p > column, the first one on ties.
                below = inside & (index > column)
                values = tl.load(matrix + index * size + column, mask=below, other=0.0)
                magnitude = tl.where(below, tl.abs(values), -1.0)
                pivot_row = tl.argmax(magnitude, axis=0, tie_break_left=True)
                swaps += (pivot_row != column + 1).to(tl.int32)
                # Swap rows column + 1 and pivot_row, then the same columns (both writes agree when equal).
                first = tl.load(matrix + (column + 1) * size + index, mask=inside)
                second = tl.load(matrix + pivot_row * size + index, mask=inside)
                tl.debug_barrier()
                tl.store(matrix + (column + 1) * size + index, second, mask=inside)
                tl.store(matrix + pivot_row * size + index, first, mask=inside)
                tl.debug_barrier()
                first = tl.load(matrix + index * size + column + 1, mask=inside)
                second = tl.load(matrix + index * size + pivot_row, mask=inside)
                tl.debug_barrier()
                tl.store(matrix + index * size + column + 1, second, mask=inside)
                tl.store(matrix + index * size + pivot_row, first, mask=inside)
                tl.debug_barrier()
                # Schur complement of the pivot block onto the trailing block: A[i, j] += tau_i x_j - x_i tau_j,
                # tau = A[:, column] / A[column + 1, column], x = A[:, column + 1].
                pivot = tl.load(matrix + (column + 1) * size + column)
                trailing = inside & (index >= column + 2)
                tau = tl.load(matrix + index * size + column, mask=trailing, other=0.0) / pivot
                following = tl.load(matrix + index * size + column + 1, mask=trailing, other=0.0)
                for start in range(column + 2, size, TILE):
                    rows = start + tl.arange(0, TILE)
                    in_rows = rows < size
                    tau_rows = tl.load(matrix + rows * size + column, mask=in_rows, other=0.0) / pivot
                    following_rows = tl.load(matrix + rows * size + column + 1, mask=in_rows, other=0.0)
                    pointers = matrix + rows[:, None] * size + index[None, :]
                    mask = in_rows[:, None] & trailing[None, :]
                    block = tl.load(pointers, mask=mask, other=0.0)
                    block += tau_rows[:, None] * following[None, :] - following_rows[:, None] * tau[None, :]
                    tl.store(pointers, block, mask=mask)
                tl.debug_barrier()
            tl.store(swaps_ptr + tl.program_id(0), swaps)

        return parlett_reid
