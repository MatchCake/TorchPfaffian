import math
import zlib
from collections.abc import Callable
from dataclasses import dataclass

import torch

# Matrix sizes (m, batch) per benchmark mode: small runs in about 2 minutes (CI) and medium in about 10 on
# a 16-core machine; big is for occasional, thorough runs.
SIZES = {
    "small": [(8, 64), (32, 16), (64, 4)],
    "medium": [(8, 1), (8, 256), (32, 1), (32, 64), (64, 16), (128, 4), (256, 2)],
    "big": [(8, 1), (8, 256), (32, 1), (32, 256), (64, 64), (128, 16), (128, 64), (256, 4), (256, 16), (512, 2)],
}
DTYPES = {
    "small": [torch.float64, torch.complex128],
    "medium": [torch.float32, torch.float64, torch.complex128],
    "big": [torch.float32, torch.float64, torch.complex64, torch.complex128],
}
# The signed backward of these families takes the exact minor-based adjugate, O(m^5) per matrix (about 5 s
# at m = 128), so it is only benchmarked up to this dimension; their other operations run at every size.
EXACT_BACKWARD_FAMILIES = ("general_singular",)
EXACT_BACKWARD_MAX_DIMENSION = {"small": 32, "medium": 32, "big": 128}
# Operations: (name, sign, backward, masked). "masked" uses mask_roundoff_gradient=True.
OPERATIONS = [
    ("signed", True, False, False),
    ("signed+grad", True, True, False),
    ("magnitude", False, False, False),
    ("magnitude+grad", False, True, False),
    ("masked+grad", False, True, True),
]


@dataclass(frozen=True)
class BenchmarkCase:
    """
    One benchmark measurement: a matrix family at a size, dtype and device, through one operation.

    The case identifier is stable across processes and versions, so the baseline and the current results
    can be joined on it, and it seeds the random matrix so both versions see the same input.
    """

    family: str
    dimension: int
    batch: int
    dtype: torch.dtype
    device: str
    operation: str
    sign: bool
    backward: bool
    masked: bool

    @property
    def identifier(self) -> str:
        """
        Stable, human-readable identifier of the case.

        :return: The identifier, e.g. ``general/m=32/b=16/float64/cpu/signed+grad``.
        :rtype: str
        """
        dtype_name = str(self.dtype).removeprefix("torch.")
        return f"{self.family}/m={self.dimension}/b={self.batch}/{dtype_name}/{self.device}/{self.operation}"

    @property
    def seed(self) -> int:
        """
        Seed of the random input, derived from the identifier without the operation.

        :return: A 32-bit seed, identical for every operation on the same matrix.
        :rtype: int
        """
        matrix_identifier = self.identifier.rsplit("/", 1)[0]
        return zlib.crc32(matrix_identifier.encode())


def _random(shape: tuple[int, ...], dtype: torch.dtype, generator: torch.Generator, scale: float) -> torch.Tensor:
    # Complex entries split the variance between the real and imaginary parts, so |entry| has the same scale.
    real_dtype = torch.float64
    entries = torch.randn(shape, dtype=real_dtype, generator=generator)
    if dtype.is_complex:
        entries = (entries + 1j * torch.randn(shape, dtype=real_dtype, generator=generator)) / math.sqrt(2)
    return (scale * entries).to(dtype)


def _general_scale(dimension: int) -> float:
    # Entries of standard deviation sqrt(e / 2m) keep log|Pf| near 0 at every m, so float32 inputs neither
    # overflow (|Pf| ~ 1e145 at m = 256 for unit entries) nor underflow.
    return math.sqrt(math.e / (2 * dimension))


def _block_scale(size: int) -> float:
    # Entries of standard deviation sqrt(e / n) keep log|det(B)| near 0 for an n x n block.
    return math.sqrt(math.e / size)


def _block_form(block: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(block)
    top = torch.cat([zero, block], dim=-1)
    bottom = torch.cat([-block.transpose(-1, -2), zero], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def general(dimension: int, batch: int, dtype: torch.dtype, generator: torch.Generator) -> torch.Tensor:
    """
    Random skew-symmetric matrices.

    :param dimension: Matrix dimension ``m``.
    :param batch: Number of matrices.
    :param dtype: Dtype of the matrices.
    :param generator: Random generator.
    :return: Matrices of shape ``(batch, m, m)``.
    :rtype: torch.Tensor
    """
    entries = _random((batch, dimension, dimension), dtype, generator, _general_scale(dimension))
    return entries - entries.transpose(-1, -2)


def general_singular(dimension: int, batch: int, dtype: torch.dtype, generator: torch.Generator) -> torch.Tensor:
    """
    Random skew-symmetric matrices of corank 2 (row and column 0 duplicate row and column 1).

    :param dimension: Matrix dimension ``m``.
    :param batch: Number of matrices.
    :param dtype: Dtype of the matrices.
    :param generator: Random generator.
    :return: Matrices of shape ``(batch, m, m)``.
    :rtype: torch.Tensor
    """
    matrix = general(dimension, batch, dtype, generator)
    matrix[..., 0, :] = matrix[..., 1, :]
    matrix[..., :, 0] = matrix[..., :, 1]
    matrix[..., 0, 0] = 0.0
    matrix[..., 0, 1] = 0.0
    matrix[..., 1, 0] = 0.0
    return matrix


def block(dimension: int, batch: int, dtype: torch.dtype, generator: torch.Generator) -> torch.Tensor:
    """
    Block-form skew-symmetric matrices ``[[0, B], [-B^T, 0]]`` with a random ``B``.

    :param dimension: Matrix dimension ``m``.
    :param batch: Number of matrices.
    :param dtype: Dtype of the matrices.
    :param generator: Random generator.
    :return: Matrices of shape ``(batch, m, m)``.
    :rtype: torch.Tensor
    """
    size = dimension // 2
    return _block_form(_random((batch, size, size), dtype, generator, _block_scale(size)))


def block_singular(dimension: int, batch: int, dtype: torch.dtype, generator: torch.Generator) -> torch.Tensor:
    """
    Block-form skew-symmetric matrices whose block ``B`` has two equal rows, so ``Pf = 0``.

    :param dimension: Matrix dimension ``m``.
    :param batch: Number of matrices.
    :param dtype: Dtype of the matrices.
    :param generator: Random generator.
    :return: Matrices of shape ``(batch, m, m)``.
    :rtype: torch.Tensor
    """
    size = dimension // 2
    off_diagonal = _random((batch, size, size), dtype, generator, _block_scale(size))
    off_diagonal[..., 0, :] = off_diagonal[..., 1, :]
    return _block_form(off_diagonal)


def block_diagonal(dimension: int, batch: int, dtype: torch.dtype, generator: torch.Generator) -> torch.Tensor:
    """
    Block-diagonal skew-symmetric matrices of ``2 x 2`` blocks ``[[0, c], [-c, 0]]``, well conditioned.

    The coefficients have a log-uniform magnitude ``e^U(-1, 1)`` and a random sign (or phase), so
    ``log|Pf| = sum log|c|`` stays near 0 at every size.

    :param dimension: Matrix dimension ``m``.
    :param batch: Number of matrices.
    :param dtype: Dtype of the matrices.
    :param generator: Random generator.
    :return: Matrices of shape ``(batch, m, m)``.
    :rtype: torch.Tensor
    """
    matrix = torch.zeros(batch, dimension, dimension, dtype=dtype)
    magnitude = torch.exp(2 * torch.rand(batch, dimension // 2, dtype=torch.float64, generator=generator) - 1)
    direction = _random((batch, dimension // 2), dtype, generator, 1.0)
    coefficients = (magnitude * direction / direction.abs()).to(dtype)
    even = torch.arange(0, dimension, 2)
    matrix[:, even, even + 1] = coefficients
    matrix[:, even + 1, even] = -coefficients
    return matrix


FAMILIES: dict[str, Callable[[int, int, torch.dtype, torch.Generator], torch.Tensor]] = {
    "general": general,
    "general_singular": general_singular,
    "block": block,
    "block_singular": block_singular,
    "block_diagonal": block_diagonal,
}


def build_cases(bench: str, devices: list[str], families: list[str] | None = None) -> list[BenchmarkCase]:
    """
    Enumerate the benchmark cases of a mode.

    :param bench: Benchmark mode, one of ``small``, ``medium`` or ``big``.
    :param devices: Devices to benchmark on, e.g. ``["cpu"]`` or ``["cpu", "cuda"]``.
    :param families: Matrix families to include; ``None`` (default) includes all of :data:`FAMILIES`.
    :return: The cases, in a deterministic order. The signed backward of :data:`EXACT_BACKWARD_FAMILIES`
        is skipped above :data:`EXACT_BACKWARD_MAX_DIMENSION`.
    :rtype: list[BenchmarkCase]
    """
    selected = list(FAMILIES) if families is None else families
    return [
        BenchmarkCase(family, dimension, batch, dtype, device, operation, sign, backward, masked)
        for family in selected
        for device in devices
        for dtype in DTYPES[bench]
        for dimension, batch in SIZES[bench]
        for operation, sign, backward, masked in OPERATIONS
        if not _is_capped(bench, family, dimension, sign, backward)
    ]


def _is_capped(bench: str, family: str, dimension: int, sign: bool, backward: bool) -> bool:
    exact_backward = family in EXACT_BACKWARD_FAMILIES and sign and backward
    return exact_backward and dimension > EXACT_BACKWARD_MAX_DIMENSION[bench]


def make_matrix(case: BenchmarkCase) -> torch.Tensor:
    """
    Build the deterministic input matrix of a case on its device.

    :param case: The benchmark case.
    :return: Matrices of shape ``(batch, m, m)`` with the case's dtype and device.
    :rtype: torch.Tensor
    """
    generator = torch.Generator().manual_seed(case.seed)
    return FAMILIES[case.family](case.dimension, case.batch, case.dtype, generator).to(case.device)
