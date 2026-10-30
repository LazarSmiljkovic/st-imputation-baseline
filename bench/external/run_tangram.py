"""Tangram (Biancalani et al. 2021) via tangram-sc.
--mode cells|clusters (default cells with a stratified reference subsample of
--max-ref-cells, default 8000, to fit a 12 GB GPU). Projected genes give the
prediction (predict) or replace zeros of measured genes (denoise)."""
import numpy as np
import scipy.sparse as sp
import torch
from common import parse, load, save, ref_linear, subsample_by_label

args = parse()
st, sc, held = load(args.work)
if args.task != "predict":
    raise SystemExit("Tangram is run in the gene-prediction protocol only (its projections are not on the count scale)")
import tangram as tg
import scanpy as scp
import pandas as pd

# tangram 1.0.4 calls torch.tensor on a pandas Series (the density prior), which newer
# torch rejects; accept Series/DataFrames transparently without touching the package.
_torch_tensor = torch.tensor
def _tensor(x, *a, **k):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        x = x.values
    return _torch_tensor(x, *a, **k)
torch.tensor = _tensor

sc.X = ref_linear(sc)
st.X = sp.csr_matrix(st.X).astype(np.float32)
mode = args.mode or "cells"
max_cells = args.max_ref_cells or 8000
if mode == "cells":
    keep = subsample_by_label(sc.obs["label"].values, max_cells, args.seed)
    sc = sc[keep].copy()
raw_lib = np.asarray(sp.csr_matrix(st.X).sum(axis=1)).ravel().astype(np.float64)
scp.pp.normalize_total(sc); scp.pp.log1p(sc)
scp.pp.normalize_total(st); scp.pp.log1p(st)
training_genes = [g for g in st.var_names if g in set(sc.var_names)]
tg.pp_adatas(sc, st, genes=training_genes)
# the RNA-count density prior must come from the raw counts, not from the log-normalised matrix
st.obs["rna_count_based_density"] = raw_lib / raw_lib.sum()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
epochs = args.epochs or 500
if mode == "clusters":
    ad_map = tg.map_cells_to_space(sc, st, mode="clusters", cluster_label="label", device=device, num_epochs=epochs,
                                   random_state=args.seed + 1, verbose=False)   # tangram seeds only when random_state is truthy, so 0 must not be passed
    ad_ge = tg.project_genes(ad_map, sc, cluster_label="label")
else:
    ad_map = tg.map_cells_to_space(sc, st, mode="cells", device=device, num_epochs=epochs, random_state=args.seed + 1,
                                   verbose=False)
    ad_ge = tg.project_genes(ad_map, sc)
# tangram lower-cases gene names; projections of a log-normalised reference are on the log scale
ge_names = {g.lower(): i for i, g in enumerate(ad_ge.var_names)}
target = list(held)
idx = [ge_names.get(g.lower(), -1) for g in target]
X = ad_ge.X.toarray() if sp.issparse(ad_ge.X) else np.asarray(ad_ge.X)
pred = np.zeros((st.n_obs, len(target)), np.float32)
for j, i in enumerate(idx):
    if i >= 0:
        pred[:, j] = X[:, i]
save(args.work, pred, scale="log")
