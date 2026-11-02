"""Cosine is invariant to the magnitude of either nonzero vector."""
import math

import pytest

from szl_quant_bench.harness import run_curve
from szl_quant_bench.metrics import cosine, kl_divergence
from szl_quant_bench.receipts import ReceiptChain


@pytest.mark.parametrize("scale", [1e-300, 1e-200, 1.0, 1e200, 1e300])
def test_cosine_preserves_direction_at_extreme_finite_magnitudes(scale):
    vector = [3 * scale, 4 * scale]
    assert cosine(vector, vector) == pytest.approx(1.0)
    assert cosine(vector, [-x for x in vector]) == pytest.approx(-1.0)
    assert cosine(vector, [4 * scale, -3 * scale]) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("left_scale,right_scale", [(1e-300, 1e300), (1e300, 1e-300)])
def test_cosine_scales_each_vector_independently(left_scale, right_scale):
    result = cosine([3 * left_scale, 4 * left_scale], [0.0, right_scale])
    assert math.isfinite(result)
    assert result == pytest.approx(0.8)


def test_zero_vector_keeps_existing_zero_similarity_convention():
    assert cosine([0.0, 0.0], [1e-300, 2e-300]) == 0.0
    assert cosine([1e300, 2e300], [0.0, 0.0]) == 0.0
    assert cosine([], []) == 0.0


def test_curve_receipts_do_not_report_direction_loss_from_underflow():
    chain = ReceiptChain()
    result = run_curve([[1e-200]], bits=[8], chain=chain)
    assert result["state"] == "MEASURED"
    assert result["curve"][0]["mse"] == 0.0
    assert result["curve"][0]["cosine"] == 1.0
    assert chain.chain[0]["run"]["curve"][0]["cosine"] == 1.0
    assert chain.verify()


def test_unrepresentable_log_probabilities_stay_invalid():
    # Stabilizing cosine must not let the existing fail-closed numeric boundary
    # depend on an unrelated metric accidentally returning NaN.
    row = [1e308, -1e308]
    with pytest.raises(ValueError, match="numeric range"):
        kl_divergence(row, row)
