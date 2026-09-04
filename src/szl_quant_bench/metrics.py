"""Quality metrics between original and dequantized vectors/matrices."""
import math


def mse(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) / max(1, len(a))


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(y * y for y in b)) or 1e-12
    return dot / (na * nb)


def _softmax(row):
    m = max(row)
    e = [math.exp(x - m) for x in row]
    s = sum(e)
    return [x / s for x in e]


def kl_divergence(p_row, q_row):
    """KL(P || Q) over softmaxed rows. >= 0; 0 means identical distributions."""
    p, q = _softmax(p_row), _softmax(q_row)
    return sum(pi * math.log(pi / max(qi, 1e-12)) for pi, qi in zip(p, q) if pi > 0)


def top1_agreement(rows_a, rows_b):
    agree = sum(1 for a, b in zip(rows_a, rows_b)
                if a.index(max(a)) == b.index(max(b)))
    return agree / max(1, len(rows_a))
