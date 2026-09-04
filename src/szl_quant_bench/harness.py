"""Quantization quality-curve harness. MEASURED only from real math; else BLOCKED.

Run states: MEASURED | BLOCKED | INVALID | FAILED. Never fabricate a number.
"""
import json

from .metrics import cosine, kl_divergence, mse, top1_agreement
from .quant import dequantize, quantize
from .receipts import ReceiptChain

STATES = ("MEASURED", "BLOCKED", "INVALID", "FAILED")
DEFAULT_BITS = (16, 8, 4, 3, 2)


def run_curve(logits, bits=DEFAULT_BITS, chain=None):
    """logits: list of rows (list of float), e.g. output logits for a token batch.

    Returns per-bit quality records computed from actual quantize/dequantize.
    Synthetic-fixture results are labeled as such by the caller; nothing here
    claims to measure a real model unless real logits were passed in.
    """
    if not logits or not logits[0]:
        return {"state": "BLOCKED", "reason": "empty logits matrix"}
    flat = [x for row in logits for x in row]
    n_cols = len(logits[0])
    records = []
    for b in bits:
        ints, scale = quantize(flat, b)
        deq = dequantize(ints, scale)
        rows_dq = [deq[i * n_cols:(i + 1) * n_cols] for i in range(len(logits))]
        records.append({
            "bits": b,
            "mse": round(mse(flat, deq), 8),
            "cosine": round(sum(cosine(a, d) for a, d in zip(logits, rows_dq)) / len(logits), 6),
            "kl_mean": round(sum(kl_divergence(a, d) for a, d in zip(logits, rows_dq)) / len(logits), 8),
            "top1_agreement": round(top1_agreement(logits, rows_dq), 6),
            "compression_vs_fp32": round(32.0 / b, 2),
        })
    result = {"state": "MEASURED", "lane": "quant-curve", "rows": len(logits),
              "cols": n_cols, "curve": records}
    if chain is not None:
        result["receipt"] = chain.emit({"type": "quant_curve",
                                        "rows": len(logits), "cols": n_cols,
                                        "curve": [dict(r) for r in records]})
    return result


def main():
    from random import Random
    rng = Random(42)  # deterministic synthetic fixture, honestly labeled
    logits = [[rng.gauss(0, 1) for _ in range(64)] for _ in range(16)]
    chain = ReceiptChain()
    out = run_curve(logits, chain=chain)
    out["fixture"] = "synthetic gaussian logits, seed 42 — NOT a real model"
    out["chain_valid"] = chain.verify()
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
