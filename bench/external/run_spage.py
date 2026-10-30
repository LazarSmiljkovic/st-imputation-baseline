"""SpaGE (Abdelaal et al. 2020) from the pinned checkout in third_party/SpaGE.
Both inputs are normalised as in the SpaGE tutorial (counts per million, log1p).
predict: genes in held.txt are predicted for every spot.
denoise: every measured gene is predicted in 5 folds and the predictions replace the zeros."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from common import parse, load, save, ref_linear

args = parse()
st, sc, held = load(args.work)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party" / "SpaGE"))
from SpaGE.main import SpaGE  # noqa: E402


def norm_log(X, target=None):
    """SpaGE tutorial normalisation: reference to 1e6 per cell, spatial to its median library size."""
    X = sp.csr_matrix(X).astype(np.float64)
    lib = np.asarray(X.sum(axis=1)).ravel(); lib[lib == 0] = 1.0
    target = float(np.median(lib)) if target is None else target
    X = sp.diags(target / lib) @ X
    X.data = np.log1p(X.data)
    return X.toarray()


def drop_constant(df):
    keep = df.columns[df.values.std(axis=0) > 0]
    return df[keep]


RNA = drop_constant(pd.DataFrame(norm_log(ref_linear(sc), 1e6), index=sc.obs_names, columns=sc.var_names))
SP = drop_constant(pd.DataFrame(norm_log(st.X), index=st.obs_names, columns=st.var_names))
n_pv = int(min(30, max(5, SP.shape[1] // 2)))

if args.task == "predict":
    targets = [g for g in held if g in RNA.columns]
    pred = SpaGE(SP, RNA, n_pv=n_pv, genes_to_predict=targets)
    out = np.zeros((st.n_obs, len(held)), np.float32)
    for j, g in enumerate(held):
        if g in pred.columns:
            out[:, j] = pred[g].values
    save(args.work, out, scale="log")
else:
    rng = np.random.default_rng(args.seed)
    genes = np.asarray(st.var_names)
    perm = rng.permutation(len(genes))
    folds = np.array_split(perm, 5)
    out = np.asarray(st.X.todense() if sp.issparse(st.X) else st.X, dtype=np.float32).copy()
    for f in folds:
        held_g = [g for g in genes[np.sort(f)] if g in RNA.columns]
        kept_g = [g for g in genes[np.sort(np.setdiff1d(perm, f))] if g in SP.columns]
        pred = SpaGE(SP[kept_g], RNA, n_pv=int(min(30, max(5, len(kept_g) // 2))), genes_to_predict=held_g)
        P = pred[held_g].values
        # predictions are on the log(median-library) scale; map back to counts via each spot's library size
        lib = np.asarray(sp.csr_matrix(st.X).sum(axis=1)).ravel()
        P = np.expm1(np.maximum(P, 0.0)) * (lib / float(np.median(lib[lib > 0])))[:, None]
        cols = st.var_names.get_indexer(held_g)
        block = out[:, cols]
        block = np.where(block == 0, P.astype(np.float32), block)
        out[:, cols] = block
    save(args.work, out)
