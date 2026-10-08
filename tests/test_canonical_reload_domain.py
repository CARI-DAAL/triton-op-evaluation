"""Persisted canonical bytes must reload with the serializer's number rule.

rfc8785 writes doubles below 1e21 in plain digit form -- including values
beyond the 2^53 integer safe domain. A plain json.loads reads those literals
back as ints, and the next canonical write of the same structure raises
IntegerDomainError. Observed live: the first check outcome saved fine, the
second check's save re-serialized the reloaded first outcome and failed the
whole evaluation with EVALUATOR_EXECUTION_FAILED (LV3 GroupedMatmul, whose
garbage outputs reach magnitudes of 1e18).
"""

import json

import pytest
import rfc8785

from triton_evaluation.serialization import (
    canonical_json_bytes,
    load_canonical,
)


def test_digit_form_double_reloads_as_double():
    original = {"max_abs_diff": 1.8517161196398838e18}
    blob = canonical_json_bytes(original)
    # The serializer writes plain digits (no exponent, no dot).
    assert b"1851716119639883800" in blob
    reloaded = load_canonical(blob)
    assert isinstance(reloaded["max_abs_diff"], float)
    assert reloaded["max_abs_diff"] == original["max_abs_diff"]


def test_reloaded_structure_round_trips_canonically():
    """The regression that failed the check: reload then re-serialize."""
    outcomes = {
        "correctness": {"data": {"failures": [{"sample": [{"diff": -1.8517161196398838e18}]}]}},
    }
    stored = canonical_json_bytes(outcomes)
    outcomes["performance"] = {"status": "succeeded"}
    assert canonical_json_bytes(load_canonical(stored) | {"performance": outcomes["performance"]})


def test_plain_json_loads_would_break_the_round_trip():
    """Guards the reason load_canonical exists (documents the divergence)."""
    blob = canonical_json_bytes({"v": 1.8517161196398838e18})
    assert isinstance(json.loads(blob)["v"], int)
    with pytest.raises(rfc8785.CanonicalizationError):
        canonical_json_bytes(json.loads(blob))


def test_safe_domain_integers_stay_exact():
    blob = canonical_json_bytes({"n": 9007199254740991, "small": 42})
    reloaded = load_canonical(blob)
    assert reloaded == {"n": 9007199254740991, "small": 42}
    assert isinstance(reloaded["n"], int)


def test_negative_digit_form_double_reloads_as_double():
    reloaded = load_canonical(canonical_json_bytes({"v": -1.8517161196398838e18}))
    assert isinstance(reloaded["v"], float)
