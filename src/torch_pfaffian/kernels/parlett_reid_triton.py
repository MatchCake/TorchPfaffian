import functools
from typing import Any

import torch


class ParlettReidTriton:
    """
    Parlett-Reid elimination of a batch of real skew-symmetric matrices in one Triton kernel launch.

    :class:`~torch_pfaffian.strategies.PfaffianParlettReid` on CUDA launches about a dozen small kernels per
    elimination step, so its run time is set by the ``m / 2`` steps' launches rather than by the arithmetic.
    Here one Triton program eliminates one matrix, and the batch is the grid, so the whole batch is one launch.

    The elimination is blocked: the rank-2 updates of a panel of ``PANEL`` steps are delayed and applied to the
    trailing block in one pass at the end of the panel, which divides the traffic of the trailing block through
    memory by the panel size. Within a panel, each step forms the two columns it needs (the pivot column and the
    next) from the matrix as it was at the start of the panel, read as rows (the matrix is skew-symmetric), and
    the panel's delayed updates; it swaps the pivot's row and column into place, in the matrix and in the delayed
    update vectors, and stores its factor and pivot. The pivoting (the largest entry below the diagonal, the first
    on ties) matches the PyTorch loop, whose post-processing (sign, product, zero-pivot test) is shared; the
    results agree with it to round-off.

    Triton has no complex arithmetic, so only ``float32`` and ``float64`` are supported, on CUDA, up to
    ``MAX_DIMENSION``; :meth:`supports` tells whether an input qualifies, and Triton is imported only then.
    """

    MAX_DIMENSION = 1024
    DTYPES = (torch.float32, torch.float64)
    # Entries of the trailing block one program updates at once (a tile of rows times the padded row length).
    TILE_ENTRIES = 4096
    # Steps whose updates are delayed and applied together, for padded sizes up to 64 and above (measured on an
    # RTX 3070: float64 batches of 256 with m = 64 take 1.2 ms with 4 and 1.4 ms with 8, with m = 128 4.3 ms both,
    # 17 ms with the unblocked elimination; float32 m = 256, 7.7 ms with 8 and 8.7 ms with 4).
    PANEL_SMALL = 4
    PANEL = 8

    @staticmethod
    def supports(matrix: torch.Tensor) -> bool:
        """
        Whether :meth:`eliminate` handles the input.

        :param matrix: Matrices of shape ``(..., m, m)``.
        :return: ``True`` for a CUDA ``float32`` or ``float64`` input with even ``2 < m <= MAX_DIMENSION``, when
            Triton is installed.
        :rtype: bool
        """
        dimension = matrix.shape[-1]
        if matrix.dtype not in ParlettReidTriton.DTYPES:
            return False
        if dimension % 2 or not 2 < dimension <= ParlettReidTriton.MAX_DIMENSION:
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
        block = triton.next_power_of_2(dimension)
        panel = ParlettReidTriton.PANEL_SMALL if block <= 64 else ParlettReidTriton.PANEL
        # The delayed update vectors tau_t and x_t of the current panel, per matrix.
        taus = torch.empty(batch, panel, dimension, dtype=matrix.dtype, device=matrix.device)
        nexts = torch.empty_like(taus)
        swaps = torch.zeros(batch, dtype=torch.int32, device=matrix.device)
        tile = max(1, min(block, ParlettReidTriton.TILE_ENTRIES // block))
        kernel = ParlettReidTriton._kernel()
        kernel[(batch,)](
            working,
            taus,
            nexts,
            swaps,
            dimension,
            BLOCK=block,
            TILE=tile,
            PANEL=panel,
            num_warps=4 if block <= 128 else 8,
            enable_fp_fusion=True,
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
        def parlett_reid(
            work_ptr, tau_ptr, next_ptr, swaps_ptr, size, BLOCK: tl.constexpr, TILE: tl.constexpr, PANEL: tl.constexpr
        ):
            program = tl.program_id(0).to(tl.int64)
            matrix = work_ptr + program * size * size
            taus = tau_ptr + program * PANEL * size  # (PANEL, size): tau_t of the panel's steps
            nexts = next_ptr + program * PANEL * size  # (PANEL, size): x_t
            index = tl.arange(0, BLOCK)
            inside = index < size
            swaps = 0
            for panel_start in range(0, size - 2, 2 * PANEL):
                steps = tl.minimum(PANEL, (size - 2 - panel_start + 1) // 2)
                for step in range(0, steps):
                    column = panel_start + 2 * step
                    # Current column `column`: -A0[column, :] (A0 skew-symmetric) plus the delayed updates.
                    current = -tl.load(matrix + column * size + index, mask=inside, other=0.0)
                    for t in range(0, step):
                        tau_t = tl.load(taus + t * size + index, mask=inside, other=0.0)
                        next_t = tl.load(nexts + t * size + index, mask=inside, other=0.0)
                        next_at, tau_at = tl.load(nexts + t * size + column), tl.load(taus + t * size + column)
                        current += tau_t * next_at - next_t * tau_at
                    below = inside & (index > column)
                    pivot_row = tl.argmax(tl.where(below, tl.abs(current), -1.0), axis=0, tie_break_left=True)
                    swaps += (pivot_row != column + 1).to(tl.int32)
                    # Swap rows, then columns, column + 1 and pivot_row of A0 (both writes agree when equal), and
                    # those entries of the delayed update vectors.
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
                    for t in range(0, step):
                        tau_first = tl.load(taus + t * size + column + 1)
                        tau_second = tl.load(taus + t * size + pivot_row)
                        next_first = tl.load(nexts + t * size + column + 1)
                        next_second = tl.load(nexts + t * size + pivot_row)
                        tl.store(taus + t * size + column + 1, tau_second)
                        tl.store(taus + t * size + pivot_row, tau_first)
                        tl.store(nexts + t * size + column + 1, next_second)
                        tl.store(nexts + t * size + pivot_row, next_first)
                    tl.debug_barrier()
                    # Current columns `column` and `column + 1` after the swap.
                    current = -tl.load(matrix + column * size + index, mask=inside, other=0.0)
                    following = -tl.load(matrix + (column + 1) * size + index, mask=inside, other=0.0)
                    for t in range(0, step):
                        tau_t = tl.load(taus + t * size + index, mask=inside, other=0.0)
                        next_t = tl.load(nexts + t * size + index, mask=inside, other=0.0)
                        next_at, tau_at = tl.load(nexts + t * size + column), tl.load(taus + t * size + column)
                        current += tau_t * next_at - next_t * tau_at
                        following += tau_t * tl.load(nexts + t * size + column + 1) - next_t * tl.load(
                            taus + t * size + column + 1
                        )
                    pivot = tl.sum(tl.where(index == column + 1, current, 0.0), axis=0)  # A[column + 1, column]
                    # The step's pivot and factor, read by the post-processing.
                    tl.store(matrix + (column + 1) * size + column, pivot)
                    tl.store(matrix + column * size + column + 1, -pivot)
                    trailing = inside & (index >= column + 2)
                    tl.store(taus + step * size + index, tl.where(trailing, current / pivot, 0.0), mask=inside)
                    tl.store(nexts + step * size + index, tl.where(trailing, following, 0.0), mask=inside)
                    tl.debug_barrier()
                # The panel's delayed updates on the trailing block, in one pass: A += sum_t tau_t x_t^T - x_t tau_t^T.
                start = panel_start + 2 * steps
                trailing_columns = inside & (index >= start)
                for row_start in range(start, size, TILE):
                    rows = row_start + tl.arange(0, TILE)
                    in_rows = rows < size
                    pointers = matrix + rows[:, None] * size + index[None, :]
                    mask = in_rows[:, None] & trailing_columns[None, :]
                    block = tl.load(pointers, mask=mask, other=0.0)
                    for t in range(0, steps):
                        tau_rows = tl.load(taus + t * size + rows, mask=in_rows, other=0.0)
                        next_rows = tl.load(nexts + t * size + rows, mask=in_rows, other=0.0)
                        tau_t = tl.load(taus + t * size + index, mask=inside, other=0.0)
                        next_t = tl.load(nexts + t * size + index, mask=inside, other=0.0)
                        block += tau_rows[:, None] * next_t[None, :] - next_rows[:, None] * tau_t[None, :]
                    tl.store(pointers, block, mask=mask)
                tl.debug_barrier()
            tl.store(swaps_ptr + program, swaps)

        return parlett_reid
