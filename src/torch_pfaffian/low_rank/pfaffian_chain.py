import torch

from .low_rank_pfaffian import LowRankPfaffian


class PfaffianChain:
    r"""
    Pfaffian of a skew-symmetric matrix that grows block by block, with the ratio of each growth step.

    The state is the inverse of the current ``m x m`` matrix ``A`` and its Pfaffian. A step appends ``2k`` rows
    and columns, the coupling ``B`` (``m x 2k``) and the block ``C`` (``2k x 2k``):

    .. math::
        \begin{pmatrix} A & B \\ -B^T & C \end{pmatrix}, \qquad
        \text{ratio} = \mathrm{Pf}\left(C + B^T A^{-1} B\right),

    in ``O(m^2 k)`` (see :meth:`~torch_pfaffian.low_rank.LowRankPfaffian.ratio_extend`), so the Pfaffians of
    all the leading blocks of an ``n x n`` matrix cost ``O(n^3)`` in total, against ``O(n^4)`` when each is
    computed anew.

    This is the loop of qubit-by-qubit sampling of a matchgate circuit (as in MatchCake). The probability of
    the outcomes ``y_0 ... y_j`` is ``2^{-(j+1)} |Pf(M_j)|``, with ``M_j`` the principal submatrix of
    ``Lambda + Lambda_y`` on the Majorana modes of qubits ``0 ... j``. Qubit ``j`` appends its two modes: ``B``
    is the covariance between the modes of the qubits sampled so far and its own, and ``C`` its ``2 x 2``
    block, which depends on the outcome ``y_j``. :meth:`ratios` gives the ratio for every candidate block (the
    conditional probabilities are ``|ratio| / 2``), and :meth:`extend` appends the sampled one::

        chain = PfaffianChain((shots,), dtype=covariance.dtype)
        for qubit in range(n_qubits):
            modes = slice(2 * qubit, 2 * qubit + 2)
            border = covariance[..., : 2 * qubit, modes]                   # (shots, 2 qubit, 2)
            corners = covariance[..., modes, modes] + outcome_blocks       # (shots, 2, 2, 2), y = 0 and 1
            probabilities = chain.ratios(border, corners).abs() / 2        # (shots, 2)
            outcome = torch.multinomial(probabilities, 1)[..., 0]          # (shots,)
            chain.extend(border, corners[torch.arange(shots), outcome])

    Every operation is a batched tensor operation on any device, without host synchronization, and is
    differentiable: gradients are exact for any skew-symmetric construction of the matrix (e.g. ``Lambda(t)`` from
    circuit parameters). The chain reads the blocks ``B`` and ``C`` and never ``-B^T``, so when they are slices of a
    skew-symmetric leaf matrix, its gradient sits on those blocks and only its skew-symmetric part, ``g - g^T``,
    matches that of :func:`~torch_pfaffian.pfaffian`. A zero ratio (an outcome of probability zero) makes the grown
    matrix singular: such a block must not be appended.

    The batch dimensions of the borders and corners broadcast with ``batch_shape``. The dtype is float32,
    float64, complex64 or complex128.
    """

    _inverse: torch.Tensor  # (..., m, m)
    _pfaffian: torch.Tensor  # (...)

    @classmethod
    def from_matrix(cls, matrix: torch.Tensor) -> "PfaffianChain":
        """
        Start from a non-empty skew-symmetric matrix, computing its inverse and Pfaffian.

        :param matrix: Invertible skew-symmetric matrix of shape ``(..., m, m)``, in float32, float64, complex64
            or complex128.
        :return: The chain, whose current matrix is ``matrix``.
        :rtype: PfaffianChain
        :raises ValueError: If ``matrix`` is not square.
        :raises TypeError: If the dtype of ``matrix`` is not supported.
        """
        from .. import pfaffian

        if matrix.dim() < 2 or matrix.shape[-1] != matrix.shape[-2]:
            raise ValueError(f"matrix must have shape (..., m, m); got {tuple(matrix.shape)}.")
        chain = cls(matrix.shape[:-2], dtype=matrix.dtype, device=matrix.device)
        inverse = torch.linalg.inv_ex(matrix)[0]  # no host synchronization; non-finite if matrix is singular
        chain._inverse = 0.5 * (inverse - inverse.mT)
        chain._pfaffian = pfaffian(matrix)
        return chain

    def __init__(
        self,
        batch_shape: tuple[int, ...] | torch.Size = (),
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        """
        Start from the empty matrix (``m = 0``, Pfaffian 1).

        :param batch_shape: Batch shape of the chain, e.g. ``(shots,)``; the first step may broadcast it.
        :param dtype: Dtype of the matrices: float32, float64, complex64 or complex128.
        :param device: Device of the matrices.
        :raises TypeError: If ``dtype`` is not supported.
        """
        from .. import LU_DTYPES

        if dtype not in LU_DTYPES:
            names = ", ".join(str(supported).removeprefix("torch.") for supported in LU_DTYPES)
            raise TypeError(f"PfaffianChain supports {names}; got {dtype}.")
        self._inverse = torch.zeros(*batch_shape, 0, 0, dtype=dtype, device=device)
        self._pfaffian = torch.ones(batch_shape, dtype=dtype, device=device)

    def ratios(self, border: torch.Tensor, corners: torch.Tensor) -> torch.Tensor:
        """
        Ratios ``Pf(M') / Pf(M)`` for several candidate blocks, without changing the state.

        :param border: Coupling ``B`` of the current rows to the new ones, of shape ``(..., m, 2k)``.
        :param corners: Candidate blocks ``C``, of shape ``(..., c, 2k, 2k)``.
        :return: The ratio for every candidate, of shape ``(..., c)``.
        :rtype: torch.Tensor
        :raises ValueError: If the shapes do not match.
        :raises TypeError: If the dtypes or devices differ from those of the chain.
        """
        if corners.dim() < 3:
            raise ValueError(f"corners must have shape (..., c, 2k, 2k); got {tuple(corners.shape)}.")
        # Candidates first, so that A^{-1} B is computed once for all of them.
        ratios = LowRankPfaffian.ratio_extend(self._inverse, border, corners.movedim(-3, 0))
        return ratios.movedim(0, -1)

    def ratio(self, border: torch.Tensor, corner: torch.Tensor) -> torch.Tensor:
        """
        Ratio ``Pf(M') / Pf(M)`` for one block, without changing the state.

        :param border: Coupling ``B`` of the current rows to the new ones, of shape ``(..., m, 2k)``.
        :param corner: Block ``C`` of the new rows and columns, of shape ``(..., 2k, 2k)``.
        :return: The ratio, of shape ``(...)``.
        :rtype: torch.Tensor
        :raises ValueError: If the shapes do not match.
        :raises TypeError: If the dtypes or devices differ from those of the chain.
        """
        return LowRankPfaffian.ratio_extend(self._inverse, border, corner)

    def extend(self, border: torch.Tensor, corner: torch.Tensor) -> torch.Tensor:
        """
        Append the rows and columns ``(B, C)`` to the current matrix.

        :param border: Coupling ``B`` of the current rows to the new ones, of shape ``(..., m, 2k)``.
        :param corner: Block ``C`` of the new rows and columns, of shape ``(..., 2k, 2k)``.
        :return: The ratio of the step, ``Pf(M') / Pf(M)``, of shape ``(...)``.
        :rtype: torch.Tensor
        :raises ValueError: If the shapes do not match.
        :raises TypeError: If the dtypes or devices differ from those of the chain.
        """
        ratio, self._inverse = LowRankPfaffian.ratio_extend_and_inverse(self._inverse, border, corner)
        self._pfaffian = self._pfaffian * ratio
        return ratio

    @property
    def dimension(self) -> int:
        """
        Size ``m`` of the current matrix.

        :return: The number of rows of the current matrix.
        :rtype: int
        """
        return self._inverse.shape[-1]

    @property
    def inverse(self) -> torch.Tensor:
        """
        Inverse of the current matrix.

        :return: The inverse, of shape ``(..., m, m)``.
        :rtype: torch.Tensor
        """
        return self._inverse

    @property
    def pfaffian(self) -> torch.Tensor:
        """
        Pfaffian of the current matrix, the product of the ratios of the steps (times that of the initial matrix
        for :meth:`from_matrix`).

        :return: The Pfaffian, of shape ``(...)``.
        :rtype: torch.Tensor
        """
        return self._pfaffian
