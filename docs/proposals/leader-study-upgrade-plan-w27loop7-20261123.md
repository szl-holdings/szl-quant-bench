# Leader-Study Upgrade Plan - W27 loop-7 - 2026-11-23

Status: PROPOSE-ONLY DRAFT - DO NOT MERGE. Owner-gated: merging requires governing approval and the estate signature-wall owner unlock.

Repo: szl-quant-bench - Honest quantization quality-curve bench: uniform absmax quantization, cosine/KL/top-1 degradation vs bit-width, hash-chained receipts. Real logits in, measured curves out — BLOCKED, never fabricated. Stdlib only. Apache-2.0.

Frame: OUROBOROS loop-7 leader study. Governing seat: Kimi kimi-k3.1-pro (json_ok, think off). Advisory seats: Astra/Sol/Claude community rails (provenance unverified, advisory only). Leader evidence web-verified 2026-11-23.

## Leader moats to absorb
- GPTQ: Post-training weight quantization using approximate second-order information to reduce quantization error. Moat: Established accuracy-oriented weight-quantization method and a strong comparison point for low-bit model compression.
- AWQ: Activation-aware weight quantization that protects salient weights using activation information. Moat: Strong low-bit accuracy and broad practical adoption; the supplied research notes report roughly 0.5-1.0 percent perplexity improvement over GPTQ at the same bit-width on Llama 3/Qwen 2-class models.
- SmoothQuant: Smooths activation outliers by migrating quantization difficulty between activations and weights. Moat: Addresses activation quantization, which is outside this repository's stated logit-quantization scope.
- llama.cpp GGUF k-quants: Deployment-oriented quantized model formats with multiple bit-width and block-quantization choices. Moat: Real inference integration, memory savings, and an established accuracy-versus-memory operating curve; the supplied notes identify Q4 as a practical sweet spot.
- MLX 4-bit on Apple Silicon: Four-bit model execution integrated with Apple's MLX platform. Moat: Hardware- and runtime-specific deployment value that a stdlib-only logit bench does not provide.
- Wikitext-2, GSM8K, and HumanEval evaluation doctrine: Perplexity and task-level evaluation for each quantization variant. Moat: Measures downstream model behavior rather than relying only on aggregate logit similarity.

## Honest gaps (grounded in repo facts)
- The README's frontier-research paragraph is truncated after defining the original top-1 index and beginning the margin definition, so the proposed certificate is not fully specified in the supplied facts.
- The supplied facts do not establish that a decision-stability certificate is implemented, tested, or included in receipts; it remains a research question rather than a measured improvement.
- The measured table is explicitly based on synthetic Gaussian logits with seed 42, not a real model, so it cannot establish GPTQ, AWQ, GGUF, MLX, perplexity, GSM8K, or HumanEval performance.
- The bench measures logit quantization only and explicitly does not measure weight quantization, GGUF behavior, inference speed, or model quality after weight conversion.
- The stated 'vs fp32' ratios are not accompanied in the supplied facts by a memory-accounting definition, so they should not be treated as measured end-to-end model compression or runtime savings.
- The README says the fixture numbers were produced before the initial push, but the supplied metadata has pushed_at set to null; the facts therefore do not independently verify a pushed implementation or its current contents.
- No real-logit evaluation results, external model artifacts, task-level benchmarks, or runtime measurements are supplied.
- The leader performance figures are supplied as web-verified research notes, but no citations or repository-local reproduction evidence are included in the facts.

## Dosed upgrade plan
- First complete the mathematical specification without changing existing quality metrics: define unique-argmax handling, ties, non-finite values, empty inputs, and the exact conservative error bound.
- Propose a row-wise certificate using e = q - z and m > 2 * max_j(abs(e[j])); prove that for every j != a, q[a] - q[j] >= m - abs(e[a]) - abs(e[j]) >= m - 2 * max_j(abs(e[j])).
- Add focused tests for certified unchanged decisions, unchanged but uncertified decisions, changed decisions, exact-boundary cases, tied original leaders, zero rows, and non-finite inputs. Use hand-computable fixtures and keep synthetic data labeled.
- Expose aggregate certificate categories separately from top-1 agreement and cosine/KL/MSE. Do not claim that certification improves quantization accuracy; it only characterizes stability under the measured dequantization error.
- Extend the receipt schema additively with a versioned certificate definition and counts while retaining the existing hash chain and run-state doctrine. Specify behavior for malformed or incomplete receipts rather than silently certifying them.
- Run the existing deterministic fixture and tests, then publish only newly executed outputs. Keep the existing table unchanged unless it is regenerated and clearly identified.
- Only after real logits are supplied, evaluate whether the certificate adds information beyond top-1 agreement. Report dataset provenance and avoid claims about weight quantization or downstream model quality.
- Treat GPTQ, AWQ, SmoothQuant, GGUF, MLX, perplexity, GSM8K, HumanEval, memory, and latency integrations as separate future proposals requiring their own dependencies, artifacts, measurements, and scope changes.
- Risk note: The main risk is overstating a conservative logit-margin certificate as a guarantee of model quality or as evidence of weight-quantization performance. Numerical edge cases, ties, non-finite values, ambiguous compression accounting, and receipt-schema changes can also create false assurance. The certificate is only as strong as its measured dequantized logits and explicit assumptions; uncertified does not mean incorrect, and certified argmax stability does not certify probabilities, rankings beyond the leader, downstream task accuracy, or inference performance.
- Propose-only note: This is a proposal, not a claim that the certificate or integrations already exist. No repository files were available beyond the supplied README excerpt and metadata, and no code was executed. Preserve the stdlib-only, no-network, BLOCKED-on-missing-input doctrine. Any implementation should be reviewed and tested before README claims or receipts are updated.

## Advisory notes (unverified rails, non-binding)
- astra: szl-quant-bench: torchao szl-block-kv: vLLM szl-energy-mcp: Home Assistant
- sol: szl-quant-bench: Hardware-specific quantization kernels for diverse accelerators szl-block-kv: Zero-copy memory mapping for heterogeneous GPU clusters szl-energy-mcp: Verified hardware-attested energy telemetry standards
- claude: szl-quant-bench: Hardware-specific quantization kernels (Triton/CUDA) for zero-shot accuracy validation. szl-block-kv: Dynamic memory fragmentation management for multi-tenant long-context serving. szl-energy-mcp: Hardware-level power rail telemetry integration for verifiable energy auditing.

## DO NOT MERGE
This plan is a proposal artifact of the 10-pass OUROBOROS loop (loop-7 of 10). No code, no CI, no behavior changes in this file. A reviewer with authority may convert to a work order later.