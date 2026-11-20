"""Deterministic red/green and adversarial controls for the logit margin bound."""
from random import Random

import pytest

from szl_quant_bench.harness import run_curve
from szl_quant_bench.robustness import top1_margin_certificate
from szl_quant_bench.receipts import ReceiptChain


def certificate(a, b):
    return top1_margin_certificate(a, b)


def test_large_margin_certifies_winner_despite_nonzero_error():
    observed = certificate([[10.0, 0.0]], [[6.0, 4.0]])
    assert observed["status"] == "MEASURED"
    assert observed["margin_certified_rows"] == 1
    assert observed["eligible_rows"] == 1
    assert observed["margin_certified_fraction"] == 1.0
    assert observed["quantized_model_qualified"] is False


def test_boundary_equal_two_epsilon_does_not_certify():
    # Strictly greater, not >=, prevents falsely certifying a boundary tie.
    observed = certificate([[2.0, 0.0]], [[1.0, 1.0]])
    assert observed["margin_certified_rows"] == 0
    assert observed["eligible_not_certified_rows"] == 1


def test_unchanged_winner_below_bound_is_unproven_not_wrong():
    observed = certificate([[0.1, 0.0]], [[0.03, 0.01]])
    assert observed["margin_certified_rows"] == 0
    assert observed["eligible_rows"] == 1
    assert observed["margin_certified_fraction"] == 0.0


def test_observed_top1_flip_is_never_mislabelled_as_certified():
    observed = certificate([[1.0, 0.0]], [[0.0, 0.5]])
    assert observed["margin_certified_rows"] == 0
    assert observed["eligible_not_certified_rows"] == 1


def test_true_ties_and_single_class_rows_have_no_certificate():
    observed = certificate([[1.0, 1.0], [7.0]], [[1.0, 0.0], [6.0]])
    assert observed["tied_top1_rows"] == 1
    assert observed["single_class_rows"] == 1
    assert observed["eligible_rows"] == 0
    assert observed["status"] == "UNAVAILABLE_NO_ELIGIBLE_ROWS"
    assert observed["margin_certified_fraction"] is None


def test_nonfinite_intermediate_margin_stays_unknown():
    observed = certificate(
        [[1e308, -1e308], [2.0, 1.0]],
        [[1e308, -1e308], [2.0, 1.0]],
    )
    assert observed["numeric_unknown_rows"] == 1
    assert observed["eligible_rows"] == 1
    assert observed["margin_certified_rows"] == 1


def test_nonfinite_intermediate_epsilon_stays_unknown():
    observed = certificate([[1e308, 0.0]], [[-1e308, 0.0]])
    assert observed["numeric_unknown_rows"] == 1
    assert observed["margin_certified_fraction"] is None


def test_nonfinite_raw_logits_unknown_not_certified():
    observed = certificate([[float("nan"), 2.0]], [[0.0, 1.0]])
    assert observed["numeric_unknown_rows"] == 1
    assert observed["status"] == "UNAVAILABLE_NO_ELIGIBLE_ROWS"


@pytest.mark.parametrize(
    "a,b",
    [
        ([], []),
        ([[1.0, 2.0]], []),
        ([[1.0, 2.0]], [[1.0]]),
        ([[1.0, 2.0]], [[1.0, True]]),
        ([[1.0, 2.0]], [["untrusted", 1.0]]),
        ([[1.0], []], [[1.0], []]),
    ],
)
def test_malformed_matrices_or_nonnumeric_inputs_fail_closed(a, b):
    with pytest.raises(ValueError):
        certificate(a, b)


def test_random_finite_certificate_never_contradicts_actual_argmax():
    rng = Random(20261120)
    before = [[rng.uniform(-4, 4) for _ in range(9)] for _ in range(140)]
    after = [
        [value + rng.uniform(-0.2, 0.2) for value in row]
        for row in before
    ]
    observed = certificate(before, after)
    assert observed["eligible_rows"] == 140
    assert observed["margin_certified_rows"] > 0
    assert observed["margin_certified_rows"] <= sum(
        a.index(max(a)) == b.index(max(b)) for a, b in zip(before, after)
    )


def test_run_curve_receipt_binds_margin_counts_and_rejects_tampering():
    chain = ReceiptChain()
    result = run_curve(
        [[10.0, 0.0], [0.1, 0.0]],
        bits=[8, 2],
        chain=chain,
        provenance={"source_kind": "SYNTHETIC"},
    )
    assert result["state"] == "MEASURED"
    assert len(result["curve"]) == 2
    assert chain.verify()
    for row in result["curve"]:
        proof = row["top1_margin_certificate"]
        assert proof["total_rows"] == 2
        assert proof["eligible_rows"] <= 2
        assert proof["margin_certified_rows"] <= proof["eligible_rows"]
        assert proof["export_authorized"] is False
    assert (
        chain.chain[0]["run"]["curve"][0]["top1_margin_certificate"]
        == result["curve"][0]["top1_margin_certificate"]
    )
    chain.chain[0]["run"]["curve"][0]["top1_margin_certificate"][
        "margin_certified_rows"
    ] += 1
    assert not chain.verify()


def test_single_class_curve_does_not_invent_certified_fraction():
    result = run_curve(
        [[2.0], [-1.0]],
        bits=[8],
        provenance={"source_kind": "SYNTHETIC"},
    )
    assert result["state"] == "MEASURED"
    metric = result["curve"][0]["top1_margin_certificate"]
    assert metric["eligible_rows"] == 0
    assert metric["single_class_rows"] == 2
    assert metric["margin_certified_fraction"] is None
