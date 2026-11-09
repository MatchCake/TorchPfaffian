import compare
import pytest
from cases import build_cases
from compare import REGRESSION_FLAGS, ComparisonRow, compare_records, merge_rounds, render_summary, render_table


def _record(identifier: str = "general/m=8/b=1/float64/cpu/signed", **fields: object) -> dict[str, object]:
    record: dict[str, object] = {
        "identifier": identifier,
        "status": "ok",
        "seconds": 1e-3,
        "median_seconds": 1e-3,
        "repeats": 5,
        "single_run": False,
        "peak_bytes": 10 << 20,
        "summary": {"real": [1.0, 2.0], "imag": [0.0, 0.0]},
    }
    record.update(fields)
    return record


class TestCompare:
    def test_identical_records_have_no_flags(self):
        (row,) = compare_records([_record()], [_record()])
        assert row.flags == []
        assert row.time_ratio == pytest.approx(1.0)
        assert row.memory_ratio == pytest.approx(1.0)

    def test_slower_needs_both_minimum_and_median(self):
        slower = _record(seconds=2e-3, median_seconds=2e-3)
        only_minimum = _record(seconds=2e-3, median_seconds=1e-3)
        only_median = _record(seconds=1e-3, median_seconds=2e-3)
        assert compare_records([_record()], [slower])[0].flags == ["slower"]
        assert compare_records([_record()], [only_minimum])[0].flags == []
        assert compare_records([_record()], [only_median])[0].flags == []

    def test_faster_is_reported(self):
        faster = _record(seconds=0.5e-3, median_seconds=0.5e-3)
        assert compare_records([_record()], [faster])[0].flags == ["faster"]

    def test_change_below_threshold_is_not_reported(self):
        slightly = _record(seconds=1.2e-3, median_seconds=1.2e-3)
        assert compare_records([_record()], [slightly], threshold=0.25)[0].flags == []

    def test_absolute_time_change_below_noise_floor_is_not_reported(self):
        tiny_before = _record(seconds=30e-6, median_seconds=30e-6)
        tiny_now = _record(seconds=90e-6, median_seconds=90e-6)  # 3x, but only 60 µs
        (row,) = compare_records([tiny_before], [tiny_now])
        assert row.flags == []
        assert row.time_ratio == pytest.approx(3.0)  # the ratio is still shown

    def test_memory_changes_are_reported(self):
        heavier = _record(peak_bytes=40 << 20)
        lighter = _record(peak_bytes=2 << 20)
        assert compare_records([_record()], [heavier])[0].flags == ["heavier"]
        assert compare_records([_record()], [lighter])[0].flags == ["lighter"]

    def test_memory_change_below_noise_floor_is_not_reported(self):
        assert compare_records([_record(peak_bytes=100)], [_record(peak_bytes=2_000_000)])[0].flags == []

    def test_disagreement_uses_dtype_tolerance(self):
        perturbed = _record(summary={"real": [1.0, 2.0 + 1e-5], "imag": [0.0, 0.0]})
        float32_identifier = "general/m=8/b=1/float32/cpu/signed"
        assert compare_records([_record()], [perturbed])[0].flags == ["disagree"]
        assert (
            compare_records([_record(float32_identifier)], [_record(float32_identifier, summary=perturbed["summary"])])[
                0
            ].flags
            == []
        )

    def test_nan_summaries_agree(self):
        nan_summary = {"real": [float("nan")], "imag": [0.0]}
        assert compare_records([_record(summary=nan_summary)], [_record(summary=nan_summary)])[0].flags == []

    @pytest.mark.parametrize("status", ["error", "timeout", "crashed"])
    def test_failure_transitions_are_classified(self, status):
        failing = _record(status=status)
        assert compare_records([_record()], [failing])[0].flags == ["error"]
        assert compare_records([failing], [_record()])[0].flags == ["fixed"]  # a crash fix is not a regression
        assert compare_records([failing], [_record(status="error")])[0].flags == ["failing"]

    def test_unsupported_in_baseline_is_not_a_regression(self):
        unsupported = _record(status="unsupported")
        assert compare_records([unsupported], [_record()])[0].flags == []
        assert compare_records([unsupported], [_record(status="error")])[0].flags == ["error"]

    def test_feature_dropped_by_current_version_is_flagged(self):
        assert compare_records([_record()], [_record(status="unsupported")])[0].flags == ["error"]

    def test_case_missing_from_one_version_is_flagged(self):
        rows = compare_records([_record("a/m=8/b=1/float64/cpu/x")], [_record("b/m=8/b=1/float64/cpu/x")])
        assert {row.identifier: row.flags for row in rows} == {
            "a/m=8/b=1/float64/cpu/x": ["error"],  # dropped by the current version
            "b/m=8/b=1/float64/cpu/x": ["fixed"],  # new in the current version
        }

    def test_merge_rounds_keeps_minimum_times(self):
        merged = merge_rounds(
            [[_record(seconds=3e-3, median_seconds=4e-3)], [_record(seconds=2e-3, median_seconds=5e-3)]]
        )
        assert merged[0]["seconds"] == pytest.approx(2e-3)
        assert merged[0]["median_seconds"] == pytest.approx(4e-3)

    def test_merge_rounds_keeps_failure(self):
        merged = merge_rounds([[_record()], [_record(status="timeout")]])
        assert merged[0]["status"] == "timeout"

    def test_merge_rounds_appends_cases_from_later_rounds(self):
        merged = merge_rounds([[_record("a/m=8/b=1/float64/cpu/x")], [_record("b/m=8/b=1/float64/cpu/x")]])
        assert [record["identifier"] for record in merged] == ["a/m=8/b=1/float64/cpu/x", "b/m=8/b=1/float64/cpu/x"]

    def test_render_table_formats_times_and_flags(self):
        row = ComparisonRow("case", _record(seconds=2.5), _record(seconds=5e-6, single_run=True), 2.0, 1.0, ["slower"])
        table = render_table([row])
        assert "2.50 s" in table
        assert "5 µs*" in table
        assert "2.00×" in table
        assert "slower" in table

    def test_render_summary_groups_regressions_first(self):
        rows = compare_records(
            [_record("a/m=8/b=1/float64/cpu/x"), _record("b/m=8/b=1/float64/cpu/x")],
            [
                _record("a/m=8/b=1/float64/cpu/x", seconds=0.1e-3, median_seconds=0.1e-3),
                _record("b/m=8/b=1/float64/cpu/x", seconds=9e-3, median_seconds=9e-3),
            ],
        )
        summary = render_summary(rows, "header")
        assert summary.index("### slower") < summary.index("### faster")
        assert "slower: 1" in summary and "faster: 1" in summary

    def test_render_summary_without_changes(self):
        assert "No case changed" in render_summary(compare_records([_record()], [_record()]), "header")

    def test_threshold_boundary_is_not_reported(self):
        at_threshold = _record(seconds=1.25e-3, median_seconds=1.25e-3)
        just_past = _record(seconds=1.26e-3, median_seconds=1.26e-3)
        assert compare_records([_record()], [at_threshold], threshold=0.25)[0].flags == []
        assert compare_records([_record()], [just_past], threshold=0.25)[0].flags == ["slower"]

    @pytest.mark.parametrize(
        ("before", "now", "agree"),
        [
            ([1e300, 2e300], [1e300, 2e300], True),
            ([1e300, 2e300], [1e300, 4e300], False),  # outside the float32 range
            ([0.0, 0.0], [1e5, 0.0], False),  # zero baseline
            ([0.0, 0.0], [0.0, 0.0], True),
            ([1e-40, 3e-40], [1e-40, 9e-40], False),  # tiny values
            ([1.0, 2.0], [1.0, 2.0 + 3e-9], True),  # below the float64 tolerance
            ([1.0, 2.0], [1.0, 2.0 + 1e-6], False),
            ([1.0, float("nan")], [1.0, float("nan")], True),
            ([1.0, 2.0], [float("nan"), float("nan")], False),
            ([1.0, float("inf")], [1.0, float("-inf")], False),
            ([1.0, float("inf")], [1.0, float("nan")], False),
            ([], [], True),
        ],
    )
    def test_agreement_is_exact_in_float64(self, before, now, agree):
        before_summary = {"real": before, "imag": [0.0] * len(before)}
        now_summary = {"real": now, "imag": [0.0] * len(now)}
        flags = compare_records([_record(summary=before_summary)], [_record(summary=now_summary)])[0].flags
        assert ("disagree" not in flags) == agree

    def test_agreement_compares_imaginary_parts(self):
        before = {"real": [1.0], "imag": [1.0]}
        now = {"real": [1.0], "imag": [-1.0]}
        assert compare_records([_record(summary=before)], [_record(summary=now)])[0].flags == ["disagree"]

    @pytest.mark.parametrize(
        ("before", "now"),
        [
            (None, {"real": [1.0], "imag": [0.0]}),
            ({"real": [1.0], "imag": [0.0]}, {"real": [1.0, 2.0], "imag": [0.0, 0.0]}),
        ],
    )
    def test_missing_or_reshaped_summary_disagrees(self, before, now):
        assert compare_records([_record(summary=before)], [_record(summary=now)])[0].flags == ["disagree"]

    @pytest.mark.parametrize("family", ["general_singular", "block_singular"])
    def test_round_off_families_only_compare_the_signed_gradient(self, family):
        noise_before = {"real": [1e-17], "imag": [0.0]}
        noise_now = {"real": [3e-17], "imag": [0.0]}
        for operation, expected in (("signed", []), ("magnitude+grad", []), ("signed+grad", ["disagree"])):
            identifier = f"{family}/m=8/b=1/float64/cpu/{operation}"
            rows = compare_records(
                [_record(identifier, summary=noise_before)], [_record(identifier, summary=noise_now)]
            )
            assert rows[0].flags == expected

    def test_unknown_dtype_and_short_identifier_use_default_tolerance(self):
        for identifier in ("general/m=8/b=1/float16/cpu/signed", "odd"):
            assert compare_records([_record(identifier)], [_record(identifier)])[0].flags == []

    def test_merge_rounds_keeps_minimum_peak_memory(self):
        merged = merge_rounds(
            [[_record(peak_bytes=30 << 20)], [_record(peak_bytes=10 << 20)], [_record(peak_bytes=None)]]
        )
        assert merged[0]["peak_bytes"] == 10 << 20

    def test_regression_flags_cover_failures_disagreements_and_slowdowns(self):
        assert REGRESSION_FLAGS == {"error", "disagree", "slower", "heavier"}

    def test_run_version_recovers_from_a_crash_and_a_hang(self, tmp_path, monkeypatch):
        # A fake torch_pfaffian aborts the process on m=32 float64 signed, and hangs on m=64 complex128
        # signed: both cases are recorded and every other case still runs.
        site = tmp_path / "site"
        (site / "torch_pfaffian").mkdir(parents=True)
        (site / "torch_pfaffian" / "__init__.py").write_text(
            "import os, time, torch\n"
            "def pfaffian(matrix, *, sign=True, mask_roundoff_gradient=False):\n"
            "    plain_signed = sign and not matrix.requires_grad\n"
            "    if plain_signed and matrix.shape[-1] == 32 and matrix.dtype == torch.float64:\n"
            "        os.abort()\n"
            "    if plain_signed and matrix.shape[-1] == 64 and matrix.dtype == torch.complex128:\n"
            "        time.sleep(1000)\n"
            "    return matrix[..., 0, 1] * 1.0\n"
        )
        monkeypatch.setattr(compare, "WATCHDOG_GRACE_SECONDS", 2)
        monkeypatch.setitem(compare.TIMING_BUDGET, "small", (0.01, 2.0, 0))
        records = compare.run_version(site, "small", ["cpu"], ["general"], 1, tmp_path / "out.json")
        by_identifier = {record["identifier"]: record["status"] for record in records}
        assert by_identifier["general/m=32/b=16/float64/cpu/signed"] == "crashed"
        assert by_identifier["general/m=64/b=4/complex128/cpu/signed"] == "timeout"
        assert len(records) == len(build_cases("small", ["cpu"], ["general"]))
        assert sum(status == "ok" for status in by_identifier.values()) == len(records) - 2

    def test_run_version_stops_when_the_runner_cannot_start(self, tmp_path):
        with pytest.raises(SystemExit, match="runner failed"):
            compare.run_version(tmp_path / "no-package", "small", ["cpu"], ["general"], 1, tmp_path / "out.json")
