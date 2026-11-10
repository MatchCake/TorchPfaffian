import itertools

import pytest
import torch

from tests.configs import (
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
    TEST_SEED,
)
from torch_pfaffian import pfaffian
from torch_pfaffian.low_rank import PfaffianChain

N_QUBITS = 4
SHOTS = 64


class TestPfaffianChain:
    @staticmethod
    def _skew(*shape: int, dtype: torch.dtype = torch.float64, seed: int = TEST_SEED) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        entries = torch.randn(*shape, generator=generator, dtype=torch.float64)
        if dtype.is_complex:
            entries = torch.complex(entries, torch.randn(*shape, generator=generator, dtype=torch.float64))
        entries = entries.to(dtype)
        return entries - entries.mT

    @staticmethod
    def _covariance(n_qubits: int, seed: int = TEST_SEED) -> torch.Tensor:
        # Majorana covariance of a pure fermionic Gaussian state, Q^T Lambda_0 Q with Q random orthogonal, as
        # produced by a matchgate circuit acting on |0...0>.
        generator = torch.Generator().manual_seed(seed)
        orthogonal, _ = torch.linalg.qr(
            torch.randn(2 * n_qubits, 2 * n_qubits, generator=generator, dtype=torch.float64)
        )
        block = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64)
        initial = torch.block_diag(*[block] * n_qubits)
        return orthogonal.mT @ initial @ orthogonal

    @staticmethod
    def _outcome_block(outcome: torch.Tensor) -> torch.Tensor:
        # (Lambda_y) for one qubit: [[0, -(-1)^y], [(-1)^y, 0]], of shape (..., 2, 2).
        entry = -((-1.0) ** outcome.double())
        zero = torch.zeros_like(entry)
        return torch.stack([torch.stack([zero, entry], -1), torch.stack([-entry, zero], -1)], -2)

    @staticmethod
    def _direct_probability(covariance: torch.Tensor, outcomes: torch.Tensor) -> torch.Tensor:
        # MatchCake's ProductState formula: p(y_W) = 2^{-|W|} |Pf(Lambda|_W + Lambda_y)| on the first |W| qubits.
        n_measured = outcomes.shape[-1]
        size = 2 * n_measured
        blocks = TestPfaffianChain._outcome_block(outcomes)  # (..., n_measured, 2, 2)
        outcome_matrix = torch.zeros(*outcomes.shape[:-1], size, size, dtype=torch.float64)
        for qubit in range(n_measured):
            outcome_matrix[..., 2 * qubit : 2 * qubit + 2, 2 * qubit : 2 * qubit + 2] = blocks[..., qubit, :, :]
        return pfaffian(covariance[:size, :size] + outcome_matrix).abs() / 2**n_measured

    def test_chain_matches_the_pfaffians_of_the_leading_blocks(self):
        matrix = self._skew(3, 10, 10)
        chain = PfaffianChain((3,))
        assert chain.dimension == 0
        torch.testing.assert_close(chain.pfaffian, torch.ones(3, dtype=torch.float64))
        for size in range(2, 11, 2):
            previous = size - 2
            border, corner = matrix[..., :previous, previous:size], matrix[..., previous:size, previous:size]
            proposed = chain.ratio(border, corner)
            ratio = chain.extend(border, corner)
            torch.testing.assert_close(ratio, proposed, atol=ATOL_SCALAR_COMPARISON, rtol=RTOL_SCALAR_COMPARISON)
            torch.testing.assert_close(
                chain.pfaffian,
                pfaffian(matrix[..., :size, :size]),
                atol=ATOL_SCALAR_COMPARISON,
                rtol=RTOL_SCALAR_COMPARISON,
            )
            assert chain.dimension == size
        torch.testing.assert_close(
            chain.inverse, torch.linalg.inv(matrix), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.complex128])
    def test_blocks_of_four_and_other_dtypes(self, dtype):
        matrix = self._skew(2, 12, 12, dtype=dtype)
        chain = PfaffianChain.from_matrix(matrix[..., :4, :4])
        for start in (4, 8):
            chain.extend(matrix[..., :start, start : start + 4], matrix[..., start : start + 4, start : start + 4])
        reference = pfaffian(matrix.to(torch.complex128 if dtype.is_complex else torch.float64))
        assert chain.pfaffian.dtype == dtype
        tolerance = 1e-3 if dtype == torch.float32 else RTOL_SCALAR_COMPARISON
        torch.testing.assert_close(chain.pfaffian.to(reference.dtype), reference, atol=tolerance, rtol=tolerance)

    def test_qubit_by_qubit_sampling_reproduces_matchcake_probabilities(self):
        covariance = self._covariance(N_QUBITS)
        generator = torch.Generator().manual_seed(TEST_SEED + 3)
        chain = PfaffianChain((SHOTS,))
        outcomes = torch.zeros(SHOTS, 0, dtype=torch.long)
        candidates = torch.tensor([0, 1])
        log_probability = torch.zeros(SHOTS, dtype=torch.float64)
        for qubit in range(N_QUBITS):
            modes = slice(2 * qubit, 2 * qubit + 2)
            border = covariance[: 2 * qubit, modes].expand(SHOTS, 2 * qubit, 2)
            corners = covariance[modes, modes] + self._outcome_block(candidates)  # (2, 2, 2), y = 0 and 1
            probabilities = chain.ratios(border, corners.expand(SHOTS, 2, 2, 2)).abs() / 2  # (SHOTS, 2)
            torch.testing.assert_close(
                probabilities.sum(-1), torch.ones(SHOTS, dtype=torch.float64), atol=ATOL_SCALAR_COMPARISON, rtol=0
            )
            outcome = torch.multinomial(probabilities, 1, generator=generator)[:, 0]
            log_probability = log_probability + probabilities[torch.arange(SHOTS), outcome].log()
            chain.extend(border, corners[outcome])
            outcomes = torch.cat([outcomes, outcome[:, None]], dim=-1)
            torch.testing.assert_close(
                log_probability.exp(),
                self._direct_probability(covariance, outcomes),
                atol=ATOL_SCALAR_COMPARISON,
                rtol=RTOL_SCALAR_COMPARISON,
            )

    def test_probabilities_of_all_outcomes_sum_to_one(self):
        covariance = self._covariance(N_QUBITS, seed=TEST_SEED + 7)
        outcomes = torch.tensor(list(itertools.product([0, 1], repeat=N_QUBITS)))
        torch.testing.assert_close(
            self._direct_probability(covariance, outcomes).sum(),
            torch.tensor(1.0, dtype=torch.float64),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=0,
        )

    def test_gradient_flows_through_the_chain(self):
        matrix = self._skew(2, 6, 6).requires_grad_(True)
        chain = PfaffianChain((2,))
        for size in (2, 4, 6):
            chain.extend(matrix[..., : size - 2, size - 2 : size], matrix[..., size - 2 : size, size - 2 : size])
        chain.pfaffian.sum().backward()
        reference = matrix.detach().clone().requires_grad_(True)
        pfaffian(reference).sum().backward()
        # The chain reads the blocks B and C, not -B^T, so its gradient sits on the upper blocks while pfaffian()
        # splits it between both triangles: the skew-symmetric parts, the derivative along skew directions, agree.
        torch.testing.assert_close(
            matrix.grad - matrix.grad.mT,
            reference.grad - reference.grad.mT,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_invalid_inputs_raise(self):
        with pytest.raises(TypeError, match="supports"):
            PfaffianChain(dtype=torch.float16)
        with pytest.raises(ValueError, match="matrix"):
            PfaffianChain.from_matrix(torch.zeros(3, 4, dtype=torch.float64))
        chain = PfaffianChain()
        with pytest.raises(ValueError, match="corners"):
            chain.ratios(torch.zeros(0, 2, dtype=torch.float64), torch.zeros(2, 2, dtype=torch.float64))
        with pytest.raises(ValueError, match="border"):
            chain.extend(torch.zeros(1, 2, dtype=torch.float64), torch.zeros(2, 2, dtype=torch.float64))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    def test_cuda_matches_cpu(self):  # pragma: no cover - needs a GPU
        matrix = self._skew(3, 8, 8)
        on_cpu, on_cuda = PfaffianChain((3,)), PfaffianChain((3,), device="cuda")
        for size in (2, 4, 6, 8):
            border, corner = matrix[..., : size - 2, size - 2 : size], matrix[..., size - 2 : size, size - 2 : size]
            torch.testing.assert_close(
                on_cuda.extend(border.cuda(), corner.cuda()).cpu(),
                on_cpu.extend(border, corner),
                atol=ATOL_SCALAR_COMPARISON,
                rtol=0,
            )
        torch.testing.assert_close(on_cuda.inverse.cpu(), on_cpu.inverse, atol=ATOL_MATRIX_COMPARISON, rtol=0)
