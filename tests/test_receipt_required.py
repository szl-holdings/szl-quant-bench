from szl_quant_bench.harness import run_curve
from szl_quant_bench.receipts import ReceiptChain


def test_every_measured_api_result_carries_a_verifiable_receipt():
    out = run_curve([[1.0, 2.0]], provenance={"source_kind": "SYNTHETIC"})
    assert out["state"] == "MEASURED"
    receipt = out["receipt"]
    assert receipt["prev_hash"] == "0" * 64
    chain = ReceiptChain()
    chain.chain = [receipt]
    assert chain.verify()


def test_caller_chain_still_receives_the_measured_record():
    chain = ReceiptChain()
    out = run_curve([[1.0, 2.0]], chain=chain,
                    provenance={"source_kind": "SYNTHETIC"})
    assert out["receipt"] is chain.chain[-1]
    assert chain.verify()
