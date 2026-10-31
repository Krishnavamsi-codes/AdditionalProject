"""Matching: mutual nearest neighbour (the RIPE / glue-factory protocol for sparse methods),
optional Lowe ratio test. Exactly the same matcher is used for baseline and enhanced variants."""
import numpy as np


def _dist(d0, d1, binary):
    if binary:  # Hamming distance via bit-unpacking + matmul
        a = np.unpackbits(d0, axis=1).astype(np.float32)
        b = np.unpackbits(d1, axis=1).astype(np.float32)
        return a.sum(1)[:, None] + b.sum(1)[None] - 2 * a @ b.T
    a = d0.astype(np.float32); b = d1.astype(np.float32)
    d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T
    return np.sqrt(np.maximum(d2, 0))


def mnn(d0, d1, binary=False, ratio=None, return_dist=False, chunk=1024):
    """Return int array [M,2] of mutual nearest neighbours (+ optional ratio test).
    Row-chunked so an 8K x 8K pool never materialises the full distance matrix (identical result)."""
    if len(d0) < 2 or len(d1) < 2:
        z = np.zeros((0, 2), int)
        return (z, np.zeros(0)) if return_dist else z
    n0, n1 = len(d0), len(d1)
    nn01 = np.empty(n0, int); best01 = np.empty(n0, np.float32); second = np.empty(n0, np.float32)
    best10 = np.full(n1, np.inf, np.float32); nn10 = np.zeros(n1, int)
    for s in range(0, n0, chunk):
        D = _dist(d0[s:s + chunk], d1, binary)
        a = D.argmin(1)
        nn01[s:s + chunk] = a
        best01[s:s + chunk] = D[np.arange(len(a)), a]
        if ratio:
            second[s:s + chunk] = np.partition(D, 1, axis=1)[:, 1]
        c = D.argmin(0); cv = D[c, np.arange(n1)]
        upd = cv < best10                      # strict: keeps the first (lowest-index) minimum like argmin
        best10[upd] = cv[upd]; nn10[upd] = c[upd] + s
    i0 = np.arange(n0)
    keep = nn10[nn01] == i0
    if ratio:
        keep &= best01 <= ratio * second
    m = np.stack([i0[keep], nn01[keep]], 1)
    if return_dist:
        return m, best01[keep]
    return m
