"""Hash-chained run receipts. UNSIGNED_HONEST: proves integrity + order, not identity."""
import hashlib
import json
import time

GENESIS = "0" * 64


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


class ReceiptChain:
    def __init__(self):
        self.chain = []

    def emit(self, record):
        prev = self.chain[-1]["self_hash"] if self.chain else GENESIS
        body = {"prev_hash": prev, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "signature": "UNSIGNED_HONEST", "run": json.loads(_canonical(record))}
        body["self_hash"] = hashlib.sha256(_canonical(body).encode()).hexdigest()
        self.chain.append(body)
        return body

    def verify(self):
        prev = GENESIS
        try:
            for r in self.chain:
                if not isinstance(r, dict) or r["prev_hash"] != prev:
                    return False
                check = dict(r)
                h = check.pop("self_hash")
                if hashlib.sha256(_canonical(check).encode()).hexdigest() != h:
                    return False
                prev = h
        except (KeyError, TypeError, ValueError):
            return False
        return bool(self.chain)
