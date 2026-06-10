from unittest import mock

import pytest
import torch

import torch_pfaffian
from torch_pfaffian import get_pfaffian_function, pfaffian, pfaffian_strategy_map
from torch_pfaffian.strategies import PfaffianDet, PfaffianFDBPf, PfaffianParlettReid


class TestTorchPfaffian:
    def test_pfaffian_strategy_map_contains_registered_strategies(self):
        assert PfaffianFDBPf.NAME.lower().strip() in pfaffian_strategy_map
        assert PfaffianDet.NAME.lower().strip() in pfaffian_strategy_map
        assert all(issubclass(cls, PfaffianFDBPf.__bases__[0]) for cls in pfaffian_strategy_map.values())

    def test_get_pfaffian_function_returns_apply_of_registered_strategy(self):
        function = get_pfaffian_function(PfaffianFDBPf.NAME)
        assert function == PfaffianFDBPf.apply

    def test_get_pfaffian_function_dispatches_to_pfaffian_det(self):
        assert get_pfaffian_function(PfaffianDet.NAME) == PfaffianDet.apply

    def test_get_pfaffian_function_default_name_is_callable(self):
        function = get_pfaffian_function()
        matrix = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64)
        torch.testing.assert_close(function(matrix), matrix[..., 0, 1])

    def test_get_pfaffian_function_is_case_and_whitespace_insensitive(self):
        function = get_pfaffian_function(f"  {PfaffianFDBPf.NAME.upper()}  ")
        assert function == PfaffianFDBPf.apply

    def test_get_pfaffian_function_unknown_name_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown strategy name"):
            get_pfaffian_function("not_a_real_strategy")

    def test_parlett_reid_is_registered(self):
        assert PfaffianParlettReid.NAME.lower().strip() in pfaffian_strategy_map
        assert get_pfaffian_function(PfaffianParlettReid.NAME) == PfaffianParlettReid.apply

    @staticmethod
    def _skew_matrix() -> torch.Tensor:
        return torch.tensor([[0.0, -3.0], [3.0, 0.0]], dtype=torch.float64)

    def test_pfaffian_signed_by_default_matches_parlett_reid(self):
        matrix = self._skew_matrix()
        torch.testing.assert_close(pfaffian(matrix), PfaffianParlettReid.apply(matrix))
        assert pfaffian(matrix).item() < 0  # signed, not magnitude

    def test_pfaffian_sign_false_returns_magnitude(self):
        matrix = self._skew_matrix()
        torch.testing.assert_close(pfaffian(matrix, sign=False), PfaffianDet.apply(matrix))
        assert pfaffian(matrix, sign=False).item() > 0

    def test_pfaffian_invalid_optimize_raises_value_error(self):
        with pytest.raises(ValueError, match="optimize"):
            pfaffian(self._skew_matrix(), optimize="fastest")

    def test_pfaffian_routes_sign_true_to_parlett_reid(self):
        with mock.patch.object(torch_pfaffian, "PfaffianParlettReid") as fake:
            fake.apply.return_value = torch.zeros(())
            pfaffian(self._skew_matrix(), sign=True)
            fake.apply.assert_called_once()

    def test_pfaffian_routes_magnitude_no_grad_to_det(self):
        with mock.patch.object(torch_pfaffian, "PfaffianDet") as fake:
            fake.apply.return_value = torch.zeros(())
            pfaffian(self._skew_matrix(), sign=False)
            fake.apply.assert_called_once()

    def test_pfaffian_routes_magnitude_with_grad_to_fdbpf(self):
        matrix = self._skew_matrix().requires_grad_(True)
        with mock.patch.object(torch_pfaffian, "PfaffianFDBPf") as fake:
            fake.apply.return_value = torch.zeros((), requires_grad=True)
            pfaffian(matrix, sign=False)
            fake.apply.assert_called_once()

    def test_pfaffian_signed_backward_flows(self):
        matrix = self._skew_matrix().requires_grad_(True)
        pfaffian(matrix).backward()
        assert matrix.grad is not None
        assert matrix.grad.shape == matrix.shape
