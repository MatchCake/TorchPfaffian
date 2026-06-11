import torch

from tests.configs import ATOL_MATRIX_COMPARISON, RTOL_MATRIX_COMPARISON
from torch_pfaffian.strategies.pfaffian_parlett_reid import PfaffianParlettReid
from torch_pfaffian.strategies.strategy import PfaffianStrategy


def _random_skew(dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    full = torch.randn(dimension, dimension, dtype=torch.float64, generator=generator)
    return full - full.transpose(-1, -2)


class _RecordingContext:
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


class TestPfaffianStrategy:
    def test_class_attributes(self):
        assert PfaffianStrategy.NAME == "PfaffianStrategy"
        assert PfaffianStrategy.EPSILON == 1e-30

    def test_setup_context_saves_inputs_and_output(self):
        matrix = torch.eye(2, dtype=torch.float64)
        pfaffian = torch.tensor(1.0, dtype=torch.float64)
        context = _RecordingContext()
        PfaffianStrategy.setup_context(context, (matrix,), pfaffian)
        assert context.saved_tensors[0] is matrix
        assert context.saved_tensors[1] is pfaffian

    def test_base_forward_is_not_implemented(self):
        assert PfaffianStrategy.forward(torch.eye(2, dtype=torch.float64)) is None

    def test_base_backward_is_not_implemented(self):
        assert PfaffianStrategy.backward(None, None) is None

    def test_grad_matrix_matches_inverse_form_on_invertible(self):
        # For invertible inputs the adjugate-based gradient equals the inverse-based closed form.
        matrix = _random_skew(6, seed=0)
        pfaffian = PfaffianParlettReid.forward(matrix)
        grad_output = torch.tensor(1.7, dtype=torch.float64)
        expected = torch.einsum("...,...ij->...ji", 0.5 * grad_output * pfaffian, torch.linalg.inv(matrix))
        result = PfaffianParlettReid.pfaffian_grad_matrix(matrix, pfaffian, grad_output)
        torch.testing.assert_close(result, expected, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_grad_matrix_uses_adjugate_on_singular(self):
        # Singular batch element (pf=0): the gradient is the exact Pfaffian-adjugate form, not zero.
        singular = torch.zeros(4, 4, dtype=torch.float64)
        singular[2, 3] = 1.0
        singular[3, 2] = -1.0
        invertible = _random_skew(4, seed=1)
        matrix = torch.stack([singular, invertible])
        pfaffian = PfaffianParlettReid.forward(matrix)
        assert pfaffian[0] == 0.0  # the singular element really has Pfaffian 0
        grad_output = torch.ones(2, dtype=torch.float64)
        result = PfaffianParlettReid.pfaffian_grad_matrix(matrix, pfaffian, grad_output)
        # d pf / d A_{01} = a_{23} = 1, so the gradient of the singular element is nonzero.
        assert torch.isfinite(result).all()
        assert result[0].abs().sum() > 0
