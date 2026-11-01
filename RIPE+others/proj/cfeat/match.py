"""Matching: mutual nearest neighbour (the RIPE / glue-factory protocol for sparse methods),
optional Lowe ratio test. Exactly the same matcher is used for baseline and enhanced variants.

Distances run on the GPU (torch CUDA, TF32 off) when available; set RIPE_GPU=0 to force the numpy path.
Both paths use the same float32 formula and first-index argmin, so the matches agree (binary: exactly)."""
import os

import numpy as np

_DEV = None


def _gpu():
    """torch CUDA device, or None (no CUDA torch, or RIPE_GPU=0). Initialised lazily per process."""
    global _DEV
    if _DEV is None:
        _DEV = False
        if os.environ.get("RIPE_GPU", "1") != "0":
            try:
                import torch
                if torch.cuda.is_available():
                    torch.backends.cuda.matmul.allow_tf32 = False
                    _DEV = torch.device("cuda")
            except ImportError:
                pass
    return _DEV or None


def _prep(d, binary):
    return np.unpackbits(d, axis=1).astype(np.float32) if binary else d.astype(np.float32)


def _dist_t(a, b, binary):
    """Torch version of _dist on prepared float32 tensors."""
    d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T
    return d if binary else d.clamp_min(0).sqrt()


def _dist(d0, d1, binary):
    dev = _gpu()
    if dev is not None:
        import torch
        a = torch.from_numpy(_prep(d0, binary)).to(dev)
        b = torch.from_numpy(_prep(d1, binary)).to(dev)
        return _dist_t(a, b, binary).cpu().numpy()
    if binary:  # Hamming distance via bit-unpacking + matmul
        a = np.unpackbits(d0, axis=1).astype(np.float32)
        b = np.unpackbits(d1, axis=1).astype(np.float32)
        return a.sum(1)[:, None] + b.sum(1)[None] - 2 * a @ b.T
    a = d0.astype(np.float32); b = d1.astype(np.float32)
    d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T
    return np.sqrt(np.maximum(d2, 0))


def _mnn_gpu(d0, d1, binary, ratio, chunk, dev):
    import torch
    n0, n1 = len(d0), len(d1)
    b = torch.from_numpy(_prep(d1, binary)).to(dev)
    nn01 = torch.empty(n0, dtype=torch.long, device=dev); best01 = torch.empty(n0, device=dev)
    second = torch.empty(n0, device=dev)
    best10 = torch.full((n1,), float("inf"), device=dev); nn10 = torch.zeros(n1, dtype=torch.long, device=dev)
    for s in range(0, n0, chunk):
        D = _dist_t(torch.from_numpy(_prep(d0[s:s + chunk], binary)).to(dev), b, binary)
        v, a = D.min(1)
        nn01[s:s + chunk] = a; best01[s:s + chunk] = v
        if ratio:
            second[s:s + chunk] = D.topk(2, dim=1, largest=False).values[:, 1]
        cv, c = D.min(0)
        upd = cv < best10
        best10[upd] = cv[upd]; nn10[upd] = c[upd] + s
    return nn01.cpu().numpy(), best01.cpu().numpy(), second.cpu().numpy(), nn10.cpu().numpy()


def mnn(d0, d1, binary=False, ratio=None, return_dist=False, chunk=1024):
    """Return int array [M,2] of mutual nearest neighbours (+ optional ratio test).
    Row-chunked so an 8K x 8K pool never materialises the full distance matrix (identical result)."""
    if len(d0) < 2 or len(d1) < 2:
        z = np.zeros((0, 2), int)
        return (z, np.zeros(0)) if return_dist else z
    n0, n1 = len(d0), len(d1)
    dev = _gpu()
    if dev is not None:
        nn01, best01, second, nn10 = _mnn_gpu(d0, d1, binary, ratio, chunk, dev)
    else:
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
