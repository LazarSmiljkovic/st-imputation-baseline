"""
bench/protocols.py -- ground-truth protocols.

P1  mask_entries     hide a fraction of observed non-zeros (set to zero)
P2  downsample       binomially thin every count with probability p
P3  gene_folds       split shared genes into k folds for leave-genes-out prediction

All return the masked/thinned matrix plus what is needed to score it; the
truth never leaves this module except through the returned arrays, so a
method cannot see it.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def gene_panel(st_csr: sp.csr_matrix, mode: str, seed: int = 0):
    """Column indices to use. ``all``, ``topN`` (N most detected genes) or
    ``stratN`` (N genes sampled across detection quartiles)."""
    nnz = np.asarray((st_csr > 0).sum(axis=0)).ravel()
    if mode == "all":
        return np.arange(st_csr.shape[1])
    if mode.startswith("top"):
        n = int(mode[3:])
        return np.sort(np.argsort(-nnz)[:n])
    if mode.startswith("strat"):
        n = int(mode[5:])
        rng = np.random.default_rng(seed)
        ok = np.where(nnz >= 20)[0]                      # need some signal to hide
        q = np.quantile(nnz[ok], [0.25, 0.5, 0.75])
        bins = np.digitize(nnz[ok], q)
        per = n // 4
        pick = []
        for b in range(4):
            cand = ok[bins == b]
            pick.append(rng.choice(cand, size=min(per, cand.size), replace=False))
        return np.sort(np.concatenate(pick))
    raise ValueError(mode)


def mask_entries(st_csr: sp.csr_matrix, frac: float, seed: int):
    """Hide ``frac`` of observed non-zeros.  Returns (masked_csr, rows, cols, truth)."""
    coo = st_csr.tocoo()
    rng = np.random.default_rng(seed)
    n = coo.nnz
    sel = rng.choice(n, size=int(round(frac * n)), replace=False)
    rows, cols, truth = coo.row[sel], coo.col[sel], coo.data[sel].astype(np.float64)
    data = coo.data.copy()
    data[sel] = 0.0
    masked = sp.csr_matrix((data, (coo.row, coo.col)), shape=st_csr.shape)
    masked.eliminate_zeros()
    return masked, rows, cols, truth


def downsample(st_csr: sp.csr_matrix, p: float, seed: int):
    """Binomial thinning of every count; returns (thinned_csr, original_csr)."""
    rng = np.random.default_rng(seed)
    thin = st_csr.tocsr(copy=True)
    thin.data = rng.binomial(thin.data.astype(np.int64), p).astype(np.float32)
    thin.eliminate_zeros()
    return thin, st_csr


def gene_folds(genes, n_folds: int, seed: int):
    """Shuffle shared genes and split into folds; yields (held_out_names, kept_names)."""
    rng = np.random.default_rng(seed)
    genes = np.asarray(list(genes))
    perm = rng.permutation(len(genes))
    folds = np.array_split(perm, n_folds)
    for f in folds:
        held = genes[np.sort(f)]
        kept = genes[np.sort(np.setdiff1d(perm, f))]
        yield held, kept


def zero_sample(st_csr_masked: sp.csr_matrix, rows_masked, cols_masked, n: int, seed: int):
    """A fixed random sample of observed-zero positions (never non-zero in the
    masked matrix and not masked), used to score zero-selection and densification."""
    rng = np.random.default_rng(seed + 7919)
    n_r, n_c = st_csr_masked.shape
    masked_set = set(zip(rows_masked.tolist(), cols_masked.tolist()))
    out_r, out_c = [], []
    csr = st_csr_masked.tocsr()
    while len(out_r) < n:
        r = rng.integers(0, n_r, size=n)
        c = rng.integers(0, n_c, size=n)
        vals = np.asarray(csr[r, c]).ravel()
        for ri, ci, v in zip(r, c, vals):
            if v == 0 and (ri, ci) not in masked_set:
                out_r.append(ri); out_c.append(ci)
                if len(out_r) >= n:
                    break
    return np.asarray(out_r), np.asarray(out_c)
