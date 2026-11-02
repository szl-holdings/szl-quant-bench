"""Quantization quality-curve harness. MEASURED only from real math; else BLOCKED.

Run states: MEASURED | BLOCKED | INVALID | FAILED. Never fabricate a number.
"""
import argparse
import hashlib
import json
import math
import re
import stat
from pathlib import Path

from .metrics import cosine, kl_divergence, mse, top1_agreement
from .quant import dequantize, quantize
from .receipts import ReceiptChain

STATES = ("MEASURED", "BLOCKED", "INVALID", "FAILED")
DEFAULT_BITS = (16, 8, 4, 3, 2)
MAX_INPUT_BYTES = 128 * 1024 * 1024
MAX_INPUT_DEPTH = 64


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _finite_json_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON numbers must be finite")
    return number


def _reject_json_constant(value):
    raise ValueError("JSON numbers must be finite")


def _check_input_depth(value):
    # Keep one iterator per container, rather than a second copy of wide input.
    stack = [(iter((value,)), 0)]
    while stack:
        items, depth = stack[-1]
        try:
            current = next(items)
        except StopIteration:
            stack.pop()
            continue
        if isinstance(current, (dict, list)):
            if depth >= MAX_INPUT_DEPTH:
                raise ValueError("input JSON exceeds 64 nested containers")
            children = current.values() if isinstance(current, dict) else current
            stack.append((iter(children), depth + 1))


def _load_input(path):
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("input must be a regular file")
    if metadata.st_size > MAX_INPUT_BYTES:
        raise ValueError("input exceeds 128 MiB")
    # The file may grow after stat; bound the bytes actually consumed as well.
    with path.open("rb") as source:
        encoded = source.read(MAX_INPUT_BYTES + 1)
    if len(encoded) > MAX_INPUT_BYTES:
        raise ValueError("input exceeds 128 MiB")
    try:
        value = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_json_object,
                           parse_float=_finite_json_float, parse_constant=_reject_json_constant)
    except RecursionError:
        raise ValueError("input JSON nesting exceeds the decoder limit") from None
    _check_input_depth(value)
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _provenance(value):
    if value is None:
        return {"source_kind": "UNVERIFIED_CALLER_INPUT"}
    if not isinstance(value, dict):
        raise ValueError("provenance must be an object")
    result = json.loads(_canonical(value))
    if result.get("source_kind") not in ("SYNTHETIC", "REAL_MODEL_LOGITS"):
        raise ValueError("source_kind must be SYNTHETIC or REAL_MODEL_LOGITS")
    if result["source_kind"] == "REAL_MODEL_LOGITS":
        for key in ("model_id", "runtime", "hardware"):
            if not isinstance(result.get(key), str) or not result[key].strip():
                raise ValueError(f"real-model provenance requires {key}")
        if not re.fullmatch(r"[0-9a-f]{40}", result.get("model_revision", "")):
            raise ValueError("model_revision must be an exact 40-character commit")
        if not re.fullmatch(r"[0-9a-f]{64}", result.get("prompt_set_sha256", "")):
            raise ValueError("prompt_set_sha256 must be a full SHA-256")
    return result


def run_curve(logits, bits=DEFAULT_BITS, chain=None, *, provenance=None):
    """logits: list of rows (list of float), e.g. output logits for a token batch.

    Returns per-bit quality records computed from actual quantize/dequantize.
    Synthetic-fixture results are labeled as such by the caller; nothing here
    claims to measure a real model unless real logits were passed in.
    """
    if logits is None or (isinstance(logits, (list, tuple)) and
                          (not logits or all(isinstance(row, (list, tuple)) and not row
                                             for row in logits))):
        return {"state": "BLOCKED", "reason": "empty logits matrix"}
    try:
        if not isinstance(logits, (list, tuple)) or not logits:
            raise ValueError("logits must be a nonempty rectangular matrix")
        if any(not isinstance(row, (list, tuple)) or not row for row in logits):
            raise ValueError("each logits row must be a nonempty array")
        n_cols = len(logits[0])
        if any(len(row) != n_cols for row in logits):
            raise ValueError("logits rows must have equal length")
        if any(isinstance(x, bool) or not isinstance(x, (int, float))
               or not math.isfinite(float(x)) for row in logits for x in row):
            raise ValueError("logits must contain only finite real numbers")
        if not isinstance(bits, (list, tuple)) or not bits:
            raise ValueError("bits must be a nonempty array")
        if any(type(b) is not int or not 2 <= b <= 16 for b in bits):
            raise ValueError("each bit width must be an integer in 2..16")
        if len(set(bits)) != len(bits):
            raise ValueError("duplicate bit widths are not allowed")
        origin = _provenance(provenance)
        logits = [[float(x) for x in row] for row in logits]
        input_hash = hashlib.sha256(_canonical(logits).encode()).hexdigest()
        flat = [x for row in logits for x in row]
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
        _canonical(records)  # reject overflow or any non-finite computed metric
    except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
        return {"state": "INVALID", "reason": str(exc)}
    result = {"state": "MEASURED", "lane": "quant-curve", "rows": len(logits),
              "cols": n_cols, "curve": records, "input_sha256": input_hash,
              "provenance": origin, "provenance_authority": "CALLER_DECLARED",
              "measurement": "UNIFORM_LOGIT_QUANTIZATION",
              "weight_quantization_measured": False,
              "gguf_export_authorized": False}
    receipt_chain = chain if chain is not None else ReceiptChain()
    result["receipt"] = receipt_chain.emit({"type": "quant_curve", **result})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="JSON object with logits and provenance")
    parser.add_argument("--output", type=Path, help="new receipt JSON file; never overwritten")
    args = parser.parse_args()
    from random import Random
    rng = Random(42)  # deterministic synthetic fixture, honestly labeled
    logits = [[rng.gauss(0, 1) for _ in range(64)] for _ in range(16)]
    origin = {"source_kind": "SYNTHETIC", "fixture": "gaussian seed 42"}
    chain = ReceiptChain()
    try:
        if args.input:
            envelope = _load_input(args.input)
            if not isinstance(envelope, dict) or "provenance" not in envelope:
                raise ValueError("input requires logits and provenance")
            logits, origin = envelope.get("logits"), envelope["provenance"]
        out = run_curve(logits, chain=chain, provenance=origin)
    except (OSError, ValueError, TypeError) as exc:
        out = {"state": "INVALID", "reason": str(exc)}
    out["chain"] = chain.chain
    out["chain_valid"] = chain.verify()
    encoded = json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        try:
            with args.output.open("x", encoding="utf-8") as destination:
                destination.write(encoded)
        except OSError as exc:
            parser.error(str(exc))
    print(encoded, end="")
    return 0 if out["state"] == "MEASURED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
