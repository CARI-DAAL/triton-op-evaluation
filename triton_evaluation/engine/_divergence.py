"""Divergence diagnosis for correctness failures (pure stdlib).

Classifies the structure of an output mismatch into one of seven mutually
exclusive ``kind`` values and extracts the worst-``k`` offending elements as a
``sample``. Operates on flat Python lists + the original output ``shape`` so it
is importable by the vendored engine subprocess AND unit-testable in a
torch-free CI (engine tests fake ``torch`` via ``ModuleType``).

Thresholds below are defensible defaults; they MUST be calibrated against real
failing cases (see plan §Calibration). The decision order is fixed and
documented; ``classify`` returns the first matching kind.
"""
import heapq
import math
from collections import defaultdict

DIVERGENCE_KINDS = (
    "sign_flip",
    "value_clamp",
    "constant_offset",
    "per_axis",
    "precision_loss",
    "scattered",
)

# --- Thresholds (calibrate against real failures; see plan §Calibration) -----
SIGN_FLIP_CORR = -0.5            # corr(reference, candidate) below this -> sign_flip
CLAMP_SATURATION_RATIO = 0.1     # >10% of candidate pinned at its min/max -> value_clamp
CLAMP_RANGE_RATIO = 0.5          # ...and candidate range < 50% of reference range
CONSTANT_OFFSET_VAR_RATIO = 0.1  # std(diff)/|mean(diff)| below this -> constant_offset
PER_AXIS_PEAK_RATIO = 3.0        # per-axis |diff| profile max/mean above this -> per_axis
PER_AXIS_MAX_N = 200_000         # skip per-axis scan above this many elements (perf)
PRECISION_LOSS_SMALL_RATIO = 0.5  # small_count/(small+normal) above this ...
PRECISION_LOSS_DIFF_BAND = 10.0   # ...and median |diff| within 10x small_value_error
SAMPLE_DEFAULT_K = 8


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs, m=None):
    if not xs:
        return 0.0
    m = m if m is not None else _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _correlation(a, b):
    ma, mb = _mean(a), _mean(b)
    da = [x - ma for x in a]
    db = [y - mb for y in b]
    num = sum(x * y for x, y in zip(da, db))
    den = math.sqrt(sum(x * x for x in da) * sum(y * y for y in db))
    return num / den if den else 0.0


def _axis_peak_ratios(abs_diff, shape):
    """For each axis, the max/mean of the |diff| profile collapsed along that axis."""
    if len(shape) <= 1:
        return []
    # strides for row-major flat index -> coordinate along each axis
    strides = [1] * len(shape)
    for d in range(len(shape) - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]
    peaks = []
    for d, (stride, size) in enumerate(zip(strides, shape)):
        if size <= 1:
            continue
        profile = [0.0] * size
        for flat, val in enumerate(abs_diff):
            profile[(flat // stride) % size] += val
        mean_prof = _mean(profile)
        if mean_prof <= 0:
            continue
        peaks.append(max(profile) / mean_prof)
    return peaks


def extract_features(reference, candidate, shape, *, atol, rtol,
                     small_value_threshold, small_value_error, rel_threshold,
                     small_count, normal_count):
    n = min(len(reference), len(candidate))
    if n == 0:
        return {"n": 0}
    diff = [candidate[i] - reference[i] for i in range(n)]
    abs_diff = [abs(d) for d in diff]
    mean_diff = _mean(diff)
    std_diff = _std(diff, mean_diff)
    mean_abs_ref = _mean([abs(r) for r in reference[:n]])
    corr = _correlation(reference[:n], candidate[:n])

    cand_min = min(candidate[:n])
    cand_max = max(candidate[:n])
    sat = sum(1 for c in candidate[:n] if c == cand_min or c == cand_max)
    ref_range = (max(reference[:n]) - min(reference[:n])) or 1e-30
    cand_range = cand_max - cand_min

    finite_diffs = sorted(d for d in abs_diff if math.isfinite(d))
    median_abs_diff = finite_diffs[len(finite_diffs) // 2] if finite_diffs else 0.0

    total = (small_count + normal_count) or 1
    features = {
        "n": n,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "mean_abs_ref": mean_abs_ref,
        "corr": corr,
        "clamp_saturation_ratio": sat / n,
        "clamp_range_ratio": cand_range / ref_range,
        "median_abs_diff": median_abs_diff,
        "small_count": small_count,
        "normal_count": normal_count,
        "small_ratio": small_count / total,
        "small_value_error": small_value_error,
        "atol": atol,
        "rtol": rtol,
        "abs_diff": abs_diff if n <= PER_AXIS_MAX_N else None,
        "shape": shape,
    }
    return features


def classify(features):
    if features.get("n", 0) == 0:
        return "scattered"

    # 1. sign_flip — global sign error is the most distinct signature.
    if features["corr"] < SIGN_FLIP_CORR:
        return "sign_flip"

    # 2. value_clamp — candidate saturated at a bound while reference spans wider.
    if (features["clamp_saturation_ratio"] > CLAMP_SATURATION_RATIO
            and features["clamp_range_ratio"] < CLAMP_RANGE_RATIO):
        return "value_clamp"

    # 3. constant_offset — diff has low variance but a significant mean.
    mean_diff = features["mean_diff"]
    if (mean_diff != 0.0
            and features["std_diff"] / abs(mean_diff) < CONSTANT_OFFSET_VAR_RATIO
            and abs(mean_diff) > features["atol"] + features["rtol"] * features["mean_abs_ref"]):
        return "constant_offset"

    # 4. per_axis — error energy concentrated along a single axis.
    if features["abs_diff"] is not None:
        peaks = _axis_peak_ratios(features["abs_diff"], features["shape"])
        if peaks and max(peaks) > PER_AXIS_PEAK_RATIO:
            return "per_axis"

    # 5. precision_loss — small ULP-band errors dominated by small-magnitude values.
    if (features["small_ratio"] > PRECISION_LOSS_SMALL_RATIO
            and features["median_abs_diff"] < PRECISION_LOSS_DIFF_BAND * features["small_value_error"]):
        return "precision_loss"

    # 6. scattered — none of the structural patterns matched.
    return "scattered"


def _finite_token(x):
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return float(x)


def worst_sample(reference, candidate, abs_diff, shape, *, k=SAMPLE_DEFAULT_K):
    n = min(len(reference), len(candidate), len(abs_diff))
    if n == 0:
        return []
    # top-k by |diff|; heapq.nlargest is O(n log k). Non-finite |diff| sorts largest.
    triples = [((0.0 if not math.isfinite(abs_diff[i]) else abs_diff[i]), i)
               for i in range(n)]
    top = heapq.nlargest(min(k, n), triples, key=lambda t: t[0])
    # row-major flat index -> multi-index
    strides = [1] * len(shape)
    for d in range(len(shape) - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]
    out = []
    for _, i in sorted(top, key=lambda t: t[0], reverse=True):  # descending |diff|
        idx = [(i // strides[d]) % shape[d] for d in range(len(shape))]
        out.append({
            "idx": idx,
            "candidate": _finite_token(candidate[i]),
            "reference": _finite_token(reference[i]),
            "diff": _finite_token(candidate[i] - reference[i]),
        })
    return out


def classify_divergence(reference, candidate, shape, *, atol, rtol,
                        small_value_threshold, small_value_error, rel_threshold,
                        small_count, normal_count, sample_k=SAMPLE_DEFAULT_K):
    if len(reference) != len(candidate) or (shape and math.prod(shape) != len(reference)):
        return {"kind": "shape_mismatch", "sample": []}
    features = extract_features(
        reference, candidate, shape, atol=atol, rtol=rtol,
        small_value_threshold=small_value_threshold, small_value_error=small_value_error,
        rel_threshold=rel_threshold, small_count=small_count, normal_count=normal_count,
    )
    kind = classify(features)
    diff = [candidate[i] - reference[i] for i in range(len(reference))]
    abs_diff = [abs(d) for d in diff]
    sample = worst_sample(reference, candidate, abs_diff, shape, k=sample_k)
    return {"kind": kind, "sample": sample}
