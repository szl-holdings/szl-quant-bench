# szl-quant-bench

Honest quantization quality-curve bench for the SZL estate. Real uniform
absmax quantization (round-to-nearest, signed ints, 2–16 bits) with quality
metrics — cosine similarity, KL divergence over softmaxed rows, top-1
agreement, MSE — on the supplied logits. This measures logit quantization,
not weight quantization, GGUF behavior, inference speed, or model quality
after a weight conversion. Stdlib only; no downloads, no network.

Doctrine: empty or missing inputs return `BLOCKED` with a reason. The demo
fixture is synthetic and labeled as such. Nothing here claims to measure a
real model unless you pass real logits in. Run states:
`MEASURED | BLOCKED | INVALID | FAILED`.

## Install and verify

```
pip install -e . pytest
python -m pytest tests/ -q
python -m szl_quant_bench.harness       # deterministic fixture curve + receipt
```

## Measured fixture curve (synthetic gaussian logits, seed 42 — NOT a real model)

| bits | cosine | KL (mean) | top-1 agreement | vs fp32 |
|---|---|---|---|---|
| 16 | 1.0000 | 0.00000 | 100.0% | 2.0x |
| 8 | 1.0000 | 0.00003 | 100.0% | 4.0x |
| 4 | 0.9882 | 0.01211 | 68.8% | 8.0x |
| 3 | 0.9420 | 0.06556 | 56.2% | 10.67x |
| 2 | 0.5715 | 0.57019 | 50.0% | 16.0x |

These numbers came out of `run_curve` in this repository, executed before the
initial push. Your model's curve will differ — run it on real logits.

## Measuring a real model

Capture a logit batch from the exact model revision and fixed evaluation
prompt set. Store a JSON object with `logits` (a rectangular finite numeric
matrix) and `provenance` containing:

- `source_kind`: `REAL_MODEL_LOGITS`
- `model_id`, `runtime`, `hardware`: nonempty identifiers
- `model_revision`: the exact 40-character lowercase commit
- `prompt_set_sha256`: the full lowercase SHA-256 of the saved prompt artifact

```sh
python -m szl_quant_bench.harness --input model-logits.json --output curve-receipt.json
```

The output file must not already exist. Missing provenance, malformed matrices,
non-finite values and invalid bit widths fail closed with a nonzero exit.
With no input the CLI runs only the explicitly `SYNTHETIC` fixture.
The Python API accepts `run_curve(logits, chain=chain, provenance=provenance)`;
omitting provenance labels the source `UNVERIFIED_CALLER_INPUT`.

Receipts bind the normalized float input matrix hash, complete provenance,
all quality records and the explicit measurement boundary. Provenance remains
`CALLER_DECLARED`: a hash cannot establish that the caller actually ran a model.
Preserve the raw logits and independent model execution evidence to support
that claim. Empty or malformed chains do not verify. KL is evaluated in log
space without silently clipping small probabilities.

Compare bit widths only on the same inputs, hardware and revision. A measured
logit curve does **not** authorize a GGUF export decision: actual quantized
model outputs and runtime evaluation are separately required.

Apache-2.0 · Doctrine v11 · SZL Holdings
