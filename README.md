# szl-quant-bench

Honest quantization quality-curve bench for the SZL estate. Real uniform
absmax quantization (round-to-nearest, signed ints, 2–16 bits) with quality
metrics — cosine similarity, KL divergence over softmaxed rows, top-1
agreement, MSE — so a GGUF/4-bit decision is made from a measured curve,
not vibes. Stdlib only; no downloads, no network.

Doctrine: empty or missing inputs return `BLOCKED` with a reason. The demo
fixture is synthetic and labeled as such. Nothing here claims to measure a
real model unless you pass real logits in. Run states:
`MEASURED | BLOCKED | INVALID | FAILED`.

## Install and verify

```
pip install -e . pytest
python -m pytest tests/ -q              # 8 tests
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

Capture a logit batch (e.g. the final-layer logits for a fixed eval prompt set
from `SZLHOLDINGS/SZL-Khipu-1.5B`), pass it to `run_curve(logits, chain=chain)`,
and attach the emitted receipt to your decision record. Compare 4-bit vs 8-bit
only on the same prompt set, same hardware, same revision — same fairness
doctrine as szl-retrieval-bench and szl-engine-bench.

Apache-2.0 · Doctrine v11 · SZL Holdings
