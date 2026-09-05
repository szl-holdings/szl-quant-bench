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
    if len(p_row) != len(q_row) or not p_row:
        raise ValueError("KL rows must have equal nonzero length")
    # Work in log space: clipping Q would silently cap large divergences.
    pm, qm = max(p_row), max(q_row)
    pz = math.log(sum(math.exp(x - pm) for x in p_row))
    qz = math.log(sum(math.exp(x - qm) for x in q_row))
    log_p = [x - pm - pz for x in p_row]
    log_q = [x - qm - qz for x in q_row]
    return max(0.0, sum(math.exp(p) * (p - q)
                        for p, q in zip(log_p, log_q) if math.exp(p) > 0))


def top1_agreement(rows_a, rows_b):
    agree = sum(1 for a, b in zip(rows_a, rows_b)
                if a.index(max(a)) == b.index(max(b)))
    return agree / max(1, len(rows_a))
