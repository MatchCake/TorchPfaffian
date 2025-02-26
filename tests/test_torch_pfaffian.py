import numpy as np
import pytest

from .configs import (
    N_RANDOM_TESTS_PER_CASE,
    TEST_SEED,
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    set_seed,
)
import torch
from torch_pfaffian import pfaffian_strategy_map, get_pfaffian_function
from torch.autograd import gradcheck

set_seed(TEST_SEED)


@pytest.mark.parametrize(
    "matrix, function_name",
    [
        (np.random.rand(*shape), function_name)
        for shape in [
            (8, 8),
            (16, 6, 6),
            (18, 10, 10),
        ]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for function_name in pfaffian_strategy_map.keys()
    ]
)
def test_torch_pfaffian_gradcheck(matrix, function_name):
    func = get_pfaffian_function(function_name)
    skew_matrix = matrix - np.einsum("...ij->...ji", matrix)
    skew_tensor = torch.tensor(skew_matrix, requires_grad=True)
    assert gradcheck(
        func, (skew_tensor,),
        eps=1e-3,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON
    )


@pytest.mark.parametrize(
    "matrix, function_name",
    [
        (np.random.rand(*shape), function_name)
        for shape in [
            (8, 8),
            (16, 6, 6),
            (18, 10, 10),
        ]
        for _ in range(N_RANDOM_TESTS_PER_CASE)
        for function_name in pfaffian_strategy_map.keys()
    ]
)
def test_torch_pfaffian_forward_against_det(matrix, function_name):
    func = get_pfaffian_function(function_name)
    skew_matrix = matrix - np.einsum("...ij->...ji", matrix)
    skew_tensor = torch.tensor(skew_matrix, requires_grad=True)

    pf = func(skew_tensor)
    det = torch.linalg.det(skew_tensor)

    assert torch.allclose(
        pf ** 2, det,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON
    )

