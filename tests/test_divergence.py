import math

from triton_evaluation.engine._divergence import classify_divergence, worst_sample


TOLS = dict(atol=1e-3, rtol=2 ** -13, small_value_threshold=2 ** -14,
            small_value_error=2 ** -30, rel_threshold=2 ** -13)


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
    sample = worst_sample(reference, candidate, [c - r for r, c in zip(reference, candidate)], (5,), k=3)
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
