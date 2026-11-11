"""
Type stubs of the compiled Rust kernel ``torch_pfaffian._rust`` (built from ``rust/src/lib.rs``).

Every function takes a C-contiguous batch of skew-symmetric matrices of shape ``(batch, n, n)``. The
``signed_pfaffian_*`` functions return the signed Pfaffians, of shape ``(batch,)``; the
``signed_slog_pfaffian_*`` functions return ``(phase, log|Pf|)``, each of shape ``(batch,)``. The
``flagged_inverse_*`` functions take any square matrices and return ``(inverse, rank_flag, residual_flag)``
(see ``PfaffianStrategy._flagged_inverse``), and ``signed_pfaffian_and_inverse_*`` both. The ``low_rank_*`` and
``replace_*`` functions are the kernels of ``LowRankPfaffian`` (``transposed``: the inverse buffer holds its
transpose); the ``..._in_place_*`` ones update the inverse buffer.
"""

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "low_rank_ratio_and_inverse_c128",
    "low_rank_ratio_and_inverse_c64",
    "low_rank_ratio_and_inverse_f32",
    "low_rank_ratio_and_inverse_f64",
    "low_rank_ratio_and_inverse_in_place_c128",
    "low_rank_ratio_and_inverse_in_place_c64",
    "low_rank_ratio_and_inverse_in_place_f32",
    "low_rank_ratio_and_inverse_in_place_f64",
    "low_rank_ratio_c128",
    "low_rank_ratio_c64",
    "low_rank_ratio_f32",
    "low_rank_ratio_f64",
    "replace_ratio_and_inverse_c128",
    "replace_ratio_and_inverse_c64",
    "replace_ratio_and_inverse_f32",
    "replace_ratio_and_inverse_f64",
    "replace_ratio_and_inverse_in_place_c128",
    "replace_ratio_and_inverse_in_place_c64",
    "replace_ratio_and_inverse_in_place_f32",
    "replace_ratio_and_inverse_in_place_f64",
    "replace_ratio_c128",
    "replace_ratio_c64",
    "replace_ratio_f32",
    "replace_ratio_f64",
    "flagged_inverse_c128",
    "flagged_inverse_c64",
    "flagged_inverse_f32",
    "flagged_inverse_f64",
    "signed_pfaffian_and_inverse_c128",
    "signed_pfaffian_and_inverse_c64",
    "signed_pfaffian_and_inverse_f32",
    "signed_pfaffian_and_inverse_f64",
    "signed_pfaffian_c64",
    "signed_pfaffian_c128",
    "signed_pfaffian_f16",
    "signed_pfaffian_f32",
    "signed_pfaffian_f64",
    "signed_slog_pfaffian_c64",
    "signed_slog_pfaffian_c128",
    "signed_slog_pfaffian_f32",
    "signed_slog_pfaffian_f64",
]

def signed_pfaffian_f64(matrix: NDArray[np.float64]) -> NDArray[np.float64]: ...
def signed_pfaffian_f32(matrix: NDArray[np.float32]) -> NDArray[np.float32]: ...
def signed_pfaffian_f16(matrix: NDArray[np.float16]) -> NDArray[np.float16]: ...
def signed_pfaffian_c128(matrix: NDArray[np.complex128]) -> NDArray[np.complex128]: ...
def signed_pfaffian_c64(matrix: NDArray[np.complex64]) -> NDArray[np.complex64]: ...
def signed_slog_pfaffian_f64(matrix: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def signed_slog_pfaffian_f32(matrix: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...
def signed_slog_pfaffian_c128(
    matrix: NDArray[np.complex128],
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]: ...
def signed_slog_pfaffian_c64(matrix: NDArray[np.complex64]) -> tuple[NDArray[np.complex64], NDArray[np.float32]]: ...
def flagged_inverse_f64(
    matrix: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]: ...
def flagged_inverse_f32(
    matrix: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.bool_]]: ...
def flagged_inverse_c128(
    matrix: NDArray[np.complex128],
) -> tuple[NDArray[np.complex128], NDArray[np.bool_], NDArray[np.bool_]]: ...
def flagged_inverse_c64(
    matrix: NDArray[np.complex64],
) -> tuple[NDArray[np.complex64], NDArray[np.bool_], NDArray[np.bool_]]: ...
def signed_pfaffian_and_inverse_f64(
    matrix: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]: ...
def signed_pfaffian_and_inverse_f32(
    matrix: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], NDArray[np.bool_]]: ...
def signed_pfaffian_and_inverse_c128(
    matrix: NDArray[np.complex128],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.bool_], NDArray[np.bool_]]: ...
def signed_pfaffian_and_inverse_c64(
    matrix: NDArray[np.complex64],
) -> tuple[NDArray[np.complex64], NDArray[np.complex64], NDArray[np.bool_], NDArray[np.bool_]]: ...
def low_rank_ratio_f64(
    inverse: NDArray[np.float64], transposed: bool, update: NDArray[np.float64]
) -> NDArray[np.float64]: ...
def low_rank_ratio_and_inverse_f64(
    inverse: NDArray[np.float64], transposed: bool, update: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def replace_ratio_f64(
    inverse: NDArray[np.float64], transposed: bool, index: NDArray[np.int64], change: NDArray[np.float64]
) -> NDArray[np.float64]: ...
def replace_ratio_and_inverse_f64(
    inverse: NDArray[np.float64], transposed: bool, index: NDArray[np.int64], change: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def low_rank_ratio_and_inverse_in_place_f64(
    inverse: NDArray[np.float64], transposed: bool, update: NDArray[np.float64]
) -> NDArray[np.float64]: ...
def replace_ratio_and_inverse_in_place_f64(
    inverse: NDArray[np.float64], transposed: bool, index: NDArray[np.int64], change: NDArray[np.float64]
) -> NDArray[np.float64]: ...
def low_rank_ratio_f32(
    inverse: NDArray[np.float32], transposed: bool, update: NDArray[np.float32]
) -> NDArray[np.float32]: ...
def low_rank_ratio_and_inverse_f32(
    inverse: NDArray[np.float32], transposed: bool, update: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...
def replace_ratio_f32(
    inverse: NDArray[np.float32], transposed: bool, index: NDArray[np.int64], change: NDArray[np.float32]
) -> NDArray[np.float32]: ...
def replace_ratio_and_inverse_f32(
    inverse: NDArray[np.float32], transposed: bool, index: NDArray[np.int64], change: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...
def low_rank_ratio_and_inverse_in_place_f32(
    inverse: NDArray[np.float32], transposed: bool, update: NDArray[np.float32]
) -> NDArray[np.float32]: ...
def replace_ratio_and_inverse_in_place_f32(
    inverse: NDArray[np.float32], transposed: bool, index: NDArray[np.int64], change: NDArray[np.float32]
) -> NDArray[np.float32]: ...
def low_rank_ratio_c128(
    inverse: NDArray[np.complex128], transposed: bool, update: NDArray[np.complex128]
) -> NDArray[np.complex128]: ...
def low_rank_ratio_and_inverse_c128(
    inverse: NDArray[np.complex128], transposed: bool, update: NDArray[np.complex128]
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]: ...
def replace_ratio_c128(
    inverse: NDArray[np.complex128], transposed: bool, index: NDArray[np.int64], change: NDArray[np.complex128]
) -> NDArray[np.complex128]: ...
def replace_ratio_and_inverse_c128(
    inverse: NDArray[np.complex128], transposed: bool, index: NDArray[np.int64], change: NDArray[np.complex128]
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]: ...
def low_rank_ratio_and_inverse_in_place_c128(
    inverse: NDArray[np.complex128], transposed: bool, update: NDArray[np.complex128]
) -> NDArray[np.complex128]: ...
def replace_ratio_and_inverse_in_place_c128(
    inverse: NDArray[np.complex128], transposed: bool, index: NDArray[np.int64], change: NDArray[np.complex128]
) -> NDArray[np.complex128]: ...
def low_rank_ratio_c64(
    inverse: NDArray[np.complex64], transposed: bool, update: NDArray[np.complex64]
) -> NDArray[np.complex64]: ...
def low_rank_ratio_and_inverse_c64(
    inverse: NDArray[np.complex64], transposed: bool, update: NDArray[np.complex64]
) -> tuple[NDArray[np.complex64], NDArray[np.complex64]]: ...
def replace_ratio_c64(
    inverse: NDArray[np.complex64], transposed: bool, index: NDArray[np.int64], change: NDArray[np.complex64]
) -> NDArray[np.complex64]: ...
def replace_ratio_and_inverse_c64(
    inverse: NDArray[np.complex64], transposed: bool, index: NDArray[np.int64], change: NDArray[np.complex64]
) -> tuple[NDArray[np.complex64], NDArray[np.complex64]]: ...
def low_rank_ratio_and_inverse_in_place_c64(
    inverse: NDArray[np.complex64], transposed: bool, update: NDArray[np.complex64]
) -> NDArray[np.complex64]: ...
def replace_ratio_and_inverse_in_place_c64(
    inverse: NDArray[np.complex64], transposed: bool, index: NDArray[np.int64], change: NDArray[np.complex64]
) -> NDArray[np.complex64]: ...
