import math

import pytest

from szl_quant_bench.harness import run_curve
from szl_quant_bench.metrics import cosine, kl_divergence, top1_agreement
from szl_quant_bench.quant import dequantize, quantize
from szl_quant_bench.receipts import ReceiptChain
from random import Random


def logits_fixture():
    rng = Random(42)
    return [[rng.gauss(0, 1) for _ in range(64)] for _ in range(16)]


def test_roundtrip_16bit_near_exact():
    vals = [0.1, -0.5, 3.7, 0.0, -2.2]
    ints, scale = quantize(vals, 16)
    deq = dequantize(ints, scale)
    for a, b in zip(vals, deq):
        assert abs(a - b) < 1e-3


def test_2bit_coarser_than_8bit():
    vals = [i * 0.01 for i in range(-50, 50)]
    ints2, s2 = quantize(vals, 2)
    ints8, s8 = quantize(vals, 8)
    assert len(set(ints2)) <= 4
    assert len(set(ints8)) > 50


def test_quantize_rejects_bad_bits():
    with pytest.raises(ValueError):
        quantize([1.0], 1)


def test_kl_nonnegative_and_zero_when_identical():
    row = [0.1, 0.9, -0.4, 2.0]
    assert kl_divergence(row, row) == pytest.approx(0.0, abs=1e-9)
    assert kl_divergence(row, [2.0, -0.4, 0.9, 0.1]) > 0


def test_top1_agreement_perfect_when_identical():
    rows = [[1, 2, 3], [3, 2, 1]]
    assert top1_agreement(rows, rows) == 1.0


def test_curve_measured_and_quality_drops_at_low_bits():
    out = run_curve(logits_fixture())
    assert out["state"] == "MEASURED"
    curve = {r["bits"]: r for r in out["curve"]}
    assert curve[16]["cosine"] > 0.999
    assert curve[16]["top1_agreement"] == 1.0
    assert curve[2]["cosine"] < curve[8]["cosine"]
    assert curve[2]["kl_mean"] > curve[8]["kl_mean"]
    assert curve[4]["compression_vs_fp32"] == 8.0


def test_curve_blocked_on_empty():
    assert run_curve([])["state"] == "BLOCKED"
    assert run_curve([[]])["state"] == "BLOCKED"


def test_curve_receipt_chain_verifies():
    chain = ReceiptChain()
    out = run_curve(logits_fixture(), chain=chain)
    assert "receipt" in out and chain.verify()
    chain.chain[0]["run"]["curve"][0]["cosine"] = 0.0
    assert not chain.verify()
