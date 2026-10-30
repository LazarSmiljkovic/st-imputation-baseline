"""Shared helpers for the external-method shims (run inside the benchmark venv)."""
import argparse
import os
from pathlib import Path

try:  # honour the thread cap set by the runner inside torch as well
    import torch
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "4"))))
except Exception:
    pass

import numpy as np
import scipy.sparse as sp


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--task", choices=["denoise", "predict"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--max-ref-cells", type=int, default=0)
    ap.add_argument("--max-ref-genes", type=int, default=0,
                    help="keep only this many highly variable reference genes (plus every measured and held gene)")
    ap.add_argument("--mode", default="")
    return ap.parse_args()


def load(work):
    import anndata as ad
    work = Path(work)
    st = ad.read_h5ad(work / "st.h5ad")
    sc = ad.read_h5ad(work / "sc.h5ad")
    held = None
    if (work / "held.txt").exists():
        held = [g for g in (work / "held.txt").read_text().split("\n") if g]
    return st, sc, held


def ref_linear(sc):
    """Return the reference as linear counts (expm1 if stored log1p)."""
    X = sp.csr_matrix(sc.X).astype(np.float32)
    if str(sc.uns.get("space", "linear")) == "log1p":
        X = X.copy(); X.data = np.expm1(X.data)
    return X


def save(work, pred, scale="counts"):
    """scale: 'counts' (count-like values at each spot's depth), 'freq' (per-spot
    frequencies summing to 1), or 'log' (log1p of library-normalised expression)."""
    pred = np.asarray(pred, dtype=np.float32)
    np.savez_compressed(Path(work) / "pred.npz", pred=pred, scale=np.array(scale))


def subsample_by_label(labels, max_n, seed):
    """Stratified subsample of reference cells (keeps every label)."""
    labels = np.asarray(labels).astype(str)
    if max_n <= 0 or labels.size <= max_n:
        return np.arange(labels.size)
    rng = np.random.default_rng(seed)
    out = []
    uniq, counts = np.unique(labels, return_counts=True)
    frac = max_n / labels.size
    for u, c in zip(uniq, counts):
        idx = np.where(labels == u)[0]
        k = max(1, int(round(c * frac)))
        out.append(rng.choice(idx, size=min(k, idx.size), replace=False))
    return np.sort(np.concatenate(out))
