import pytest
import torch

from torch_pfaffian import get_pfaffian_function, pfaffian_strategy_map
from torch_pfaffian.strategies import PfaffianDet, PfaffianFDBPf


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
