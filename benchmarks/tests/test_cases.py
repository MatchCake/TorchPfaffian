import pytest
import torch
from cases import (
    DTYPES,
    EXACT_BACKWARD_FAMILIES,
    EXACT_BACKWARD_MAX_DIMENSION,
    FAMILIES,
    OPERATIONS,
    SIZES,
    BenchmarkCase,
    build_cases,
    make_matrix,
)


def _case(family: str = "general", dtype: torch.dtype = torch.float64, operation: str = "signed") -> BenchmarkCase:
    return BenchmarkCase(family, 8, 3, dtype, "cpu", operation, True, False, False)


class TestCases:
    @pytest.mark.parametrize("family", list(FAMILIES))
    @pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
    def test_families_are_skew_symmetric_with_requested_shape(self, family, dtype):
        matrix = make_matrix(_case(family, dtype))
        assert matrix.shape == (3, 8, 8)
        assert matrix.dtype == dtype
        torch.testing.assert_close(matrix, -matrix.transpose(-1, -2))

    @pytest.mark.parametrize("family", ["general_singular", "block_singular"])
    def test_singular_families_have_zero_pfaffian(self, family):
        determinant = torch.linalg.det(make_matrix(_case(family)))
        torch.testing.assert_close(determinant.abs(), torch.zeros(3, dtype=torch.float64))

    def test_block_family_has_zero_diagonal_blocks(self):
        matrix = make_matrix(_case("block"))
        assert torch.all(matrix[:, :4, :4] == 0) and torch.all(matrix[:, 4:, 4:] == 0)

    def test_matrix_is_deterministic_and_shared_across_operations(self):
        torch.testing.assert_close(make_matrix(_case(operation="signed")), make_matrix(_case(operation="magnitude")))

    def test_identifier_is_stable_and_readable(self):
        assert _case().identifier == "general/m=8/b=3/float64/cpu/signed"

    @pytest.mark.parametrize("bench", list(SIZES))
    def test_build_cases_covers_every_combination_but_capped_exact_backward(self, bench):
        cases = build_cases(bench, ["cpu"])
        capped = sum(dimension > EXACT_BACKWARD_MAX_DIMENSION[bench] for dimension, _ in SIZES[bench])
        signed_backward = sum(sign and backward for _, sign, backward, _ in OPERATIONS)
        skipped = len(EXACT_BACKWARD_FAMILIES) * len(DTYPES[bench]) * capped * signed_backward
        expected = len(FAMILIES) * len(DTYPES[bench]) * len(SIZES[bench]) * len(OPERATIONS) - skipped
        assert len(cases) == expected
        assert len({case.identifier for case in cases}) == expected

    def test_exact_backward_is_capped_only_for_signed_backward(self):
        cases = build_cases("medium", ["cpu"], ["general_singular"])
        large = [case for case in cases if case.dimension > EXACT_BACKWARD_MAX_DIMENSION["medium"]]
        assert large and not any(case.sign and case.backward for case in large)
        assert any(case.operation == "magnitude+grad" for case in large)

    def test_build_cases_filters_families_and_devices(self):
        cases = build_cases("small", ["cpu", "cuda"], ["block"])
        assert {case.family for case in cases} == {"block"}
        assert {case.device for case in cases} == {"cpu", "cuda"}
