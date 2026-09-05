import hashlib
import json
import subprocess
import sys

import pytest

from szl_quant_bench.harness import run_curve
from szl_quant_bench.metrics import kl_divergence
from szl_quant_bench.quant import quantize
from szl_quant_bench.receipts import ReceiptChain


@pytest.mark.parametrize("logits", ["bad", [1], [[1], [2, 3]],
                                  [[True]], [[float("nan")]],
                                  [[float("inf")]], [["1"]],
                                  [[1e308, -1e308]]])
def test_invalid_matrix_never_emits_receipt(logits):
    chain = ReceiptChain()
    assert run_curve(logits, chain=chain)["state"] == "INVALID"
    assert chain.chain == []


@pytest.mark.parametrize("logits", [None, [], [[]], [[], []], ((), ())])
def test_missing_or_wholly_empty_logits_are_blocked(logits):
    chain = ReceiptChain()
    assert run_curve(logits, chain=chain)["state"] == "BLOCKED"
    assert chain.chain == []


@pytest.mark.parametrize("bits", [[], [True], [4.0], [1], [17], [4, 4], "4"])
def test_invalid_bits_never_emit_receipt(bits):
    chain = ReceiptChain()
    assert run_curve([[1.0, 2.0]], bits=bits, chain=chain)["state"] == "INVALID"
    assert chain.chain == []


def test_provenance_and_input_are_bound_to_receipt():
    provenance = {
        "source_kind": "REAL_MODEL_LOGITS", "model_id": "owner/model",
        "model_revision": "a" * 40, "prompt_set_sha256": "b" * 64,
        "runtime": "test-runtime", "hardware": "test-host",
    }
    logits = [[1.0, 2.0], [3.0, 4.0]]
    chain = ReceiptChain()
    out = run_curve(logits, chain=chain, provenance=provenance)
    assert out["state"] == "MEASURED"
    canonical = json.dumps(logits, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False)
    assert out["input_sha256"] == hashlib.sha256(canonical.encode()).hexdigest()
    assert out["provenance_authority"] == "CALLER_DECLARED"
    assert out["weight_quantization_measured"] is False
    assert out["gguf_export_authorized"] is False
    provenance["model_id"] = "mutated"
    out["curve"][0]["mse"] = 999
    out["provenance"]["model_id"] = "mutated-too"
    assert chain.verify()
    chain.chain[0]["run"]["input_sha256"] = "0" * 64
    assert not chain.verify()


def test_missing_real_provenance_is_rejected():
    out = run_curve([[1.0]], provenance={"source_kind": "REAL_MODEL_LOGITS"})
    assert out["state"] == "INVALID"


def test_unverified_caller_is_not_labeled_real():
    assert run_curve([[1.0]])["provenance"]["source_kind"] == "UNVERIFIED_CALLER_INPUT"


def test_kl_does_not_clip_large_divergence():
    assert kl_divergence([1000, 0], [0, 1000]) == pytest.approx(1000)
    with pytest.raises(ValueError):
        kl_divergence([1], [1, 2])


@pytest.mark.parametrize("bits", [True, 4.0, "4"])
def test_quantize_rejects_noninteger_bits(bits):
    with pytest.raises(ValueError):
        quantize([1.0], bits)


def test_empty_and_malformed_chains_fail_closed():
    chain = ReceiptChain()
    assert not chain.verify()
    for malformed in ([{}], [None], [{"prev_hash": "0" * 64}]):
        chain.chain = malformed
        assert not chain.verify()


def test_cli_labels_fixture_and_does_not_overwrite(tmp_path):
    destination = tmp_path / "receipt.json"
    command = [sys.executable, "-B", "-m", "szl_quant_bench.harness",
               "--output", str(destination)]
    first = subprocess.run(command, capture_output=True, text=True, check=True)
    result = json.loads(first.stdout)
    assert result["chain_valid"] is True
    assert result["provenance"]["source_kind"] == "SYNTHETIC"
    before = destination.read_bytes()
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0
    assert destination.read_bytes() == before


def test_cli_invalid_input_has_nonzero_exit_and_no_receipt(tmp_path):
    source = tmp_path / "input.json"
    source.write_text(json.dumps({"logits": [[True]],
                                  "provenance": {"source_kind": "SYNTHETIC"}}))
    result = subprocess.run([sys.executable, "-B", "-m", "szl_quant_bench.harness",
                             "--input", str(source)], capture_output=True, text=True)
    assert result.returncode == 2
    body = json.loads(result.stdout)
    assert body["state"] == "INVALID"
    assert body["chain"] == []
    assert body["chain_valid"] is False
