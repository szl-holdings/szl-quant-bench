"""Conservative top-1 logit-margin certificate for one quantization curve.

For original logits z and measured dequantized logits q in a row, let
margin = z[top1] - max(z[other]) and epsilon = max_i |z[i] - q[i]|.
If margin > 2 * epsilon, the same index must win after dequantization.

This is a sufficient, *not necessary*, post-hoc logit stability bound.
A certified row is NOT a qualified model, weight quantization, cryptographic
signature, or permission to export or serve a model. Ties, singleton classes,
and non-finite *intermediate* arithmetic cannot receive a certificate.
"""

from __future__ import annotations

import math


def top1_margin_certificate(original_rows, dequantized_rows):
    """Return counted, conservative row evidence without hiding denominators."""
    if not isinstance(original_rows, (list, tuple)) or not original_rows:
        raise ValueError("certificate requires a nonempty row matrix")
    if not isinstance(dequantized_rows, (list, tuple)) or (
        len(original_rows) != len(dequantized_rows)
    ):
        raise ValueError("certificate row count mismatch")

    eligible = certified = ties = singletons = numeric_unknown = 0
    for original, dequantized in zip(original_rows, dequantized_rows):
        if (
            not isinstance(original, (list, tuple))
            or not isinstance(dequantized, (list, tuple))
            or not original
            or len(original) != len(dequantized)
        ):
            raise ValueError("certificate requires equal nonempty logit rows")
        for value in (*original, *dequantized):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("certificate requires numeric nonboolean logits")
        # A valid Python int may exceed binary64 range. Treat its
        # conversion overflow as numerically unknown, never as a crash or
        # evidence of a certified model-quality claim.
        try:
            finite_logits = all(
                math.isfinite(float(value)) for value in (*original, *dequantized)
            )
        except OverflowError:
            finite_logits = False
        if not finite_logits:
            numeric_unknown += 1
            continue
        if len(original) == 1:
            singletons += 1
            continue
        top_value = max(original)
        winner = original.index(top_value)
        if sum(value == top_value for value in original) != 1:
            ties += 1
            continue
        runner_up = max(value for idx, value in enumerate(original) if idx != winner)
        margin = top_value - runner_up
        epsilon = max(
            abs(before - after) for before, after in zip(original, dequantized)
        )
        # Floating subtraction can round the strict boundary in either
        # direction. Shift the margin DOWN and the observed error UP by
        # one representable float step before comparing. Doubling the
        # upper error is exact in binary if finite; shift up once more to
        # avoid inadvertently accepting a rounded boundary equality.
        if not math.isfinite(margin) or not math.isfinite(epsilon):
            numeric_unknown += 1
            continue
        lower_margin = math.nextafter(margin, -math.inf)
        upper_epsilon = math.nextafter(epsilon, math.inf)
        upper_twice_epsilon = math.nextafter(2 * upper_epsilon, math.inf)
        if not math.isfinite(upper_twice_epsilon):
            numeric_unknown += 1
            continue
        eligible += 1
        if lower_margin > upper_twice_epsilon:
            # Independent observed-rank negative check for numeric regressions:
            # a sufficient margin certificate cannot coexist with a flip.
            if dequantized.index(max(dequantized)) != winner:
                raise ValueError("margin certificate contradicts observed top1")
            certified += 1

    return {
        "method": "ORIGINAL_TOP1_MARGIN_GT_2X_OBSERVED_LINF_ERROR",
        "scope": "POSTHOC_LOGIT_RANK_ONLY_NOT_WEIGHT_OR_MODEL_QUALITY",
        "status": "MEASURED" if eligible else "UNAVAILABLE_NO_ELIGIBLE_ROWS",
        "total_rows": len(original_rows),
        "eligible_rows": eligible,
        "margin_certified_rows": certified,
        "eligible_not_certified_rows": eligible - certified,
        "tied_top1_rows": ties,
        "single_class_rows": singletons,
        "numeric_unknown_rows": numeric_unknown,
        "margin_certified_fraction": (
            round(certified / eligible, 6) if eligible else None
        ),
        "quantized_model_qualified": False,
        "export_authorized": False,
    }
