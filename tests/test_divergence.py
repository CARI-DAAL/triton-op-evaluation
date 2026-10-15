import math
import random

import pytest

from triton_evaluation.engine._divergence import (
    _finite_upper_median_below,
    classify_divergence,
    worst_sample,
)

TOLS = dict(
    atol=1e-3,
    rtol=2**-13,
    small_value_threshold=2**-14,
    small_value_error=2**-30,
    rel_threshold=2**-13,
)


def _result(reference, candidate, shape, **extra):
    base = dict(small_count=0, normal_count=len(reference))
    base.update(extra)
    return classify_divergence(reference, candidate, shape, **TOLS, **base)


def test_shape_mismatch_when_lengths_differ():
    # shape says 4 but candidate has 3 elements → structural mismatch
    out = classify_divergence([1, 2, 3, 4], [1, 2, 3], (4,), **TOLS, small_count=0, normal_count=4)
    assert out["kind"] == "shape_mismatch"
    assert out["sample"] == []


def test_sign_flip_constant_negation():
    reference = [1.0, 2.0, -3.0, 4.0, -5.0, 6.0, 7.0, -8.0]
    candidate = [-x for x in reference]
    out = _result(reference, candidate, (8,))
    assert out["kind"] == "sign_flip"


def test_value_clamp_candidate_saturated_at_zero():
    # golden ranges widely; candidate is clamped at 0 for the negative half
    reference = [-10.0, -8.0, -6.0, -4.0, 3.0, 5.0, 7.0, 9.0]
    candidate = [0.0, 0.0, 0.0, 0.0, 3.0, 5.0, 7.0, 9.0]
    out = _result(reference, candidate, (8,))
    assert out["kind"] == "value_clamp"


def test_constant_offset_low_variance_offset():
    reference = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    candidate = [x + 5.0 for x in reference]  # uniform +5 offset, zero variance in diff
    out = _result(reference, candidate, (8,))
    assert out["kind"] == "constant_offset"


def test_per_axis_error_concentrated_along_one_axis():
    # shape (4, 4): error only in column 0 → concentrated along axis 1
    reference = [float(i) for i in range(16)]
    candidate = list(reference)
    for row in range(4):
        candidate[row * 4 + 0] += 100.0  # corrupt entire column 0
    out = _result(reference, candidate, (4, 4))
    assert out["kind"] == "per_axis"


def test_precision_loss_small_ulp_band_errors_small_value_dominant():
    # tiny perturbations well inside the fp32 ULP band, all on small-magnitude values
    reference = [1e-8, 2e-8, 1.5e-8, 0.5e-8, 1.2e-8, 0.8e-8, 1.1e-8, 0.9e-8]
    candidate = [v + 1e-12 for v in reference]
    out = _result(reference, candidate, (8,), small_count=8, normal_count=0)
    assert out["kind"] == "precision_loss"


def test_scattered_when_no_pattern_matches():
    # large, irregular, non-axis-aligned, non-constant errors → scattered
    reference = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0]
    candidate = [5.0, -3.0, 2.0, -7.0, 1.0, 9.0, -4.0, 0.0]
    out = _result(reference, candidate, (8,))
    assert out["kind"] == "scattered"


def test_worst_sample_top_k_by_abs_diff_descending():
    reference = [0.0, 0.0, 0.0, 0.0, 0.0]
    candidate = [1.0, 5.0, -2.0, 9.0, 3.0]
    diff = [c - r for r, c in zip(reference, candidate, strict=True)]
    sample = worst_sample(reference, candidate, diff, (5,), k=3)
    assert len(sample) == 3
    # descending by |diff|
    diffs = [abs(s["diff"]) for s in sample]
    assert diffs == sorted(diffs, reverse=True)
    assert sample[0]["diff"] == 9.0
    # idx is the multi-index for the flat position
    assert sample[0]["idx"] == [3]


def test_worst_sample_non_finite_become_string_sentinels():
    reference = [1.0, math.inf, 2.0]
    candidate = [1.0, -math.inf, math.nan]
    abs_diff = [0.0, math.inf, math.nan]
    sample = worst_sample(reference, candidate, abs_diff, (3,), k=3)
    assert len(sample) == 3
    # k == n, so every element is present; locate by flat index to stay order-independent
    # (the worst-by-|diff| ranking is exercised by test_worst_sample_ranks_non_finite_above_finite).
    by_idx = {tuple(s["idx"]): s for s in sample}
    assert by_idx[(1,)]["candidate"] == "-inf"
    assert by_idx[(1,)]["reference"] == "inf"
    assert by_idx[(2,)]["candidate"] == "nan"


def test_kind_always_returned_even_when_sample_empty():
    out = classify_divergence([1, 2], [1, 2], (2,), **TOLS, small_count=0, normal_count=2)
    assert out["kind"] in ("constant_offset", "precision_loss", "scattered")
    assert isinstance(out["sample"], list)


def test_worst_sample_ranks_non_finite_above_finite():
    # inf |diff| is the worst divergence — it must surface FIRST even when k < n
    reference = [1.0, 2.0, 3.0, 4.0]
    candidate = [1.0, 2.0, 3.0, math.inf]
    abs_diff = [0.0, 0.5, 1.0, math.inf]
    sample = worst_sample(reference, candidate, abs_diff, (4,), k=2)
    assert len(sample) == 2
    assert sample[0]["idx"] == [3]  # inf ranks above every finite diff
    assert sample[0]["diff"] == "inf"


def test_matching_non_finite_pairs_report_the_zero_diff_they_rank_by():
    # A matched NaN or same-sign inf pair is equal, ranked as |diff| 0; its
    # reported diff must agree, not read NaN from inf - inf or nan - nan.
    reference = [math.nan, 1.0, math.inf, 3.0]
    candidate = [math.nan, 1.0, math.inf, 7.0]

    sample = _result(reference, candidate, (4,))["sample"]

    assert [point["diff"] for point in sample] == [4.0, 0.0, 0.0, 0.0]
    assert [point["idx"] for point in sample] == [[3], [0], [1], [2]]
    assert sample[1]["candidate"] == sample[1]["reference"] == "nan"
    assert sample[3]["candidate"] == sample[3]["reference"] == "inf"


def test_matching_non_finite_pairs_do_not_poison_structural_features():
    reference = [math.inf, math.nan, 1.0, 2.0, 3.0, 4.0]
    candidate = [math.inf, math.nan, 6.0, 7.0, 8.0, 9.0]

    out = _result(reference, candidate, (6,))

    assert out["kind"] == "constant_offset"


@pytest.mark.parametrize(
    ("values", "threshold", "expected"),
    [
        ([0.0, 1.0], 1.0, False),  # upper median, not the average of middle values
        ([0.0, 0.0, 1.0, 1.0], 1.0, False),
        ([0.0, 0.0, 0.0, 1.0], 1.0, True),
        ([0.0, 1.0, 2.0], 1.0, False),  # equality is outside the strict band
        ([0.0, 1.0, 2.0], math.nextafter(1.0, math.inf), True),
        ([math.nan, math.inf, -math.inf, 0.0], 1.0, True),
        ([math.nan, math.inf, -math.inf], 1.0, True),  # historical empty median = 0
        ([], 0.0, False),
        ([], math.nan, False),
    ],
)
def test_finite_upper_median_threshold_boundaries(values, threshold, expected):
    assert _finite_upper_median_below(values, threshold) is expected


def test_finite_upper_median_threshold_matches_sorted_oracle():
    rng = random.Random(31415)
    values = [-1.0, 0.0, 1.0, math.nextafter(1.0, math.inf), math.inf, -math.inf, math.nan]
    for length in (0, 1, 2, 3, 8, 99, 100, 257):
        for _ in range(20):
            diffs = rng.choices(values, k=length)
            ordered = sorted(value for value in diffs if math.isfinite(value))
            median = ordered[len(ordered) // 2] if ordered else 0.0
            for threshold in (-math.inf, -1.0, 0.0, 1.0, 2.0, math.inf, math.nan):
                assert _finite_upper_median_below(diffs, threshold) == (median < threshold)


@pytest.mark.parametrize("shape", [(0,), (1,), (7,), (3, 4), ()])
@pytest.mark.parametrize("k", [-1, 0, 1, 3, 8, 20])
def test_streamed_worst_sample_matches_stable_sorted_oracle(shape, k):
    n = math.prod(shape)
    # Repeated errors and mixed non-finite values exercise stable row-major ties.
    reference = ([1.0, math.inf, 0.0, 2.0] * 3)[:n]
    candidate = ([math.nan, -math.inf, 3.0, -1.0] * 3)[:n]
    diffs = [abs(c - r) for r, c in zip(reference, candidate, strict=True)]
    order = sorted(
        range(n), key=lambda i: diffs[i] if math.isfinite(diffs[i]) else math.inf,
        reverse=True,
    )[:max(k, 0)]

    def token(value):
        return float(value) if math.isfinite(value) else str(value)

    expected = []
    for i in order:
        indices = [(i // math.prod(shape[d + 1:])) % dim for d, dim in enumerate(shape)]
        expected.append({
            "idx": indices,
            "candidate": token(candidate[i]),
            "reference": token(reference[i]),
            "diff": token(candidate[i] - reference[i]),
        })
    assert worst_sample(reference, candidate, diffs, shape, k=k) == expected


def test_streamed_worst_sample_respects_shortest_input():
    assert worst_sample([0.0, 0.0], [1.0, 99.0, 100.0], [1.0], (3,), k=8) == [
        {"idx": [0], "candidate": 1.0, "reference": 0.0, "diff": 1.0},
    ]
