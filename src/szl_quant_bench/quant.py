"""Uniform symmetric quantization: quantize/dequantize with absmax scales.

Real integer quantization math, stdlib only. Per-tensor absmax scale,
round-to-nearest, signed integers at the requested bit width.
"""


def quantize(values, bits):
    """values: flat list of floats. Returns (ints, scale). bits in 2..16."""
    if type(bits) is not int or not 2 <= bits <= 16:
        raise ValueError("bits must be in 2..16")
    if not values:
        return [], 1.0
    qmax = (1 << (bits - 1)) - 1
    absmax = max(abs(v) for v in values) or 1.0
    scale = absmax / qmax
    return [int(round(v / scale)) for v in values], scale


def dequantize(ints, scale):
    return [i * scale for i in ints]
