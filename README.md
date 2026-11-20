# szl-quant-bench

<!-- szl:header v1 -->
<!-- Native identity: szl-brand@4309f6fce89179331b0b3ed8c99494e76c5be256; Orbit v2 icon. -->
<p>
  <a href="https://github.com/szl-holdings">
    <img src="https://raw.githubusercontent.com/szl-holdings/szl-brand/4309f6fce89179331b0b3ed8c99494e76c5be256/kit/logos/orbit-v2/suite/szl-icon-32.svg" alt="SZL Holdings — Orbit v2" width="32" height="32" />
  </a>
</p>

**Control before action. Evidence after.**

[Product](https://a-11-oy.com) · [Proof](https://a11oy.net) · [GitHub organization](https://github.com/szl-holdings)
<!-- /szl:header -->

[![PyPI](https://img.shields.io/pypi/v/szl-quant-bench)](https://pypi.org/project/szl-quant-bench/) [![Python](https://img.shields.io/pypi/pyversions/szl-quant-bench)](https://pypi.org/project/szl-quant-bench/) [![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/szl-holdings/szl-quant-bench/badge)](https://scorecard.dev/viewer/?uri=github.com/szl-holdings/szl-quant-bench)

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

## Frontier research: mathematically bounded logit decisions

**Research question (not a measured improvement):** can a quantization quality
curve distinguish an unchanged argmax from an argmax whose stability has a
conservative mathematical certificate? Average cosine, KL and top-1 agreement
alone do not expose this distinction.

For each row of original logits `z` and the **actually dequantized** logits
`q`, let `a` be the unique original top-1 index, `m = z[a] -
max(z[j], j != a)` its original runner-up margin, and
`epsilon = max_i |z[i] - q[i]|` the observed L-infinity perturbation.

When `m > 2 * epsilon`, then for every competing index `j`:
`q[a] - q[j] >= m - 2*epsilon > 0`.
Consequently the row's top-1 must be preserved. Implementation
conservatively rounds the computed margin *down* and the computed error bound
*up* to neighboring binary64 floats before applying the strict inequality,
to avoid accidentally accepting a floating-point boundary equality. This is a
**sufficient, not necessary**, source-math result for these supplied and
dequantized logits.
It does not prove anything about weight quantization, unseen prompts,
inference speed, statistical generalization, GPU kernels or model safety.

Each `curve` record now includes `top1_margin_certificate` with
`eligible_rows`, `margin_certified_rows`,
`eligible_not_certified_rows`, `tied_top1_rows`,
`single_class_rows`, `numeric_unknown_rows`, and
`margin_certified_fraction`. An empty eligible denominator is JSON
`null`, **never 0 or 1**. Ambiguous ties, singleton classes and non-finite
intermediate arithmetic are never certified. Certification is independent of
the repository's unsigned receipt chain; hashes establish byte consistency,
not independent origin or signer identity.

**External methods studied; no borrowed code or new dependency:**

- [SmoothQuant (Guangxuan Xiao, Ji Lin et al., ICML 2023)](https://arxiv.org/abs/2211.10438),
  [official code](https://github.com/mit-han-lab/smoothquant):
  weight/activation transformation, *not* a logit decision certificate.
- [AWQ (Ji Lin et al., MLSys 2024)](https://arxiv.org/abs/2306.00978),
  [official code](https://github.com/mit-han-lab/llm-awq):
  activation-aware weight scaling, *not* measured here.
- [GPTQ (Elias Frantar et al., 2022)](https://arxiv.org/abs/2210.17323),
  [official code](https://github.com/IST-DASLab/gptq):
  second-order weight quantization, *not* measured here.

**Falsification gate:** negative tests must reject equality at
`m == 2*epsilon`, tied top-1, singleton rows, finite-input subtraction
overflow, false source values and receipt tampering. Randomized deterministic
fixtures must never report a certificate for a flipped top-1. Compare only
the *same* immutable model/logits/holdout data across candidate bit widths.
A real GGUF, model or production decision additionally requires independently
pinned source/model/dataset revisions, correct rights, actual runtime outputs,
held-out quality measures, memory/latency benchmarks, signed deployment
evidence and an owner-controlled release decision.

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
CLI input must be a regular UTF-8 JSON file with unique object keys and finite
numbers throughout the envelope. The 128 MiB limit applies to bytes consumed,
including growth after the initial size check. More than 64 nested containers
or a decoder nesting failure returns `INVALID` without a receipt. These checks
do not provide filesystem isolation against concurrent input-path replacement.
With no input the CLI runs only the explicitly `SYNTHETIC` fixture.
The Python API accepts `run_curve(logits, chain=chain, provenance=provenance)`;
omitting provenance labels the source `UNVERIFIED_CALLER_INPUT`. Every
`MEASURED` API result carries a hash-chained receipt record. If no chain is
supplied, the API creates a fresh one-record chain for that measurement; pass
a caller-owned `ReceiptChain` to append the record to an existing chain.

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

Estate on Hugging Face: models, datasets and Spaces are published under the [SZLHOLDINGS](https://huggingface.co/SZLHOLDINGS) org.

## License

Apache-2.0 — see [LICENSE](LICENSE).
