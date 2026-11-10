"""
Type stubs of the compiled Rust kernel ``torch_pfaffian._rust`` (built from ``rust/src/lib.rs``).

Every function takes a C-contiguous batch of skew-symmetric matrices of shape ``(batch, n, n)``. The
``signed_pfaffian_*`` functions return the signed Pfaffians, of shape ``(batch,)``; the
``signed_slog_pfaffian_*`` functions return ``(phase, log|Pf|)``, each of shape ``(batch,)``. The
``flagged_inverse_*`` functions take any square matrices and return ``(inverse, rank_flag, residual_flag)``
(see ``PfaffianStrategy._flagged_inverse``).
"""

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "flagged_inverse_c128",
    "flagged_inverse_c64",
    "flagged_inverse_f32",
    "flagged_inverse_f64",
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
