"""gimVI (Lopez et al. 2019) via scvi-tools' GIMVI. Needs count data on both sides.
Options: --epochs, --max-ref-cells (stratified subsample), --max-ref-genes (highly
variable reference genes kept besides the measured and held genes; the model is
trained on the union of both gene sets, and tens of thousands of sparse
reference-only genes make the CPU run very slow and the training unstable)."""
import numpy as np
import scipy.sparse as sp
import torch
from common import parse, load, save, ref_linear, subsample_by_label

args = parse()
st, sc, held = load(args.work)
if args.task != "predict":
    raise SystemExit("gimVI is run in the gene-prediction protocol only")
import scanpy as scp
import scvi
from scvi.external import GIMVI

scvi.settings.seed = args.seed
sc.X = sp.csr_matrix(np.round(ref_linear(sc).toarray())).astype(np.float32)   # gimVI needs integer-like counts
st.X = sp.csr_matrix(st.X).astype(np.float32)
if args.max_ref_genes and sc.n_vars > args.max_ref_genes:
    a = sc.copy(); scp.pp.normalize_total(a, target_sum=1e4); scp.pp.log1p(a)
    scp.pp.highly_variable_genes(a, n_top_genes=min(args.max_ref_genes, a.n_vars - 1))
    keep = set(a.var_names[a.var["highly_variable"].values]) | set(st.var_names) | set(held)
    sc = sc[:, [g for g in sc.var_names if g in keep]].copy()
# genes without counts in the reference and cells/spots without counts have no library size
sc = sc[:, np.asarray(sc.X.sum(axis=0)).ravel() > 0].copy()
if args.max_ref_cells:
    keep = subsample_by_label(sc.obs["label"].values, args.max_ref_cells, args.seed)
    sc = sc[keep].copy()
sc = sc[np.asarray(sc.X.sum(axis=1)).ravel() > 0].copy()
st_genes = [g for g in st.var_names if g in set(sc.var_names)]
st_ok = np.asarray(st[:, st_genes].X.sum(axis=1)).ravel() > 0
st_train = st[st_ok, st_genes].copy()
sc.obs["batch"] = "sc"; st_train.obs["batch"] = "st"
GIMVI.setup_anndata(sc, labels_key="label")
GIMVI.setup_anndata(st_train)
model = GIMVI(sc, st_train)
epochs = args.epochs or 200
# a last mini-batch holding one sample is squeezed to 1-D and rejected by batch normalisation ("expected 2D or
# 3D input"); this bites in get_imputed_values (default batch 128) as well as in training, so one safe size is used for both
batch = next(b for b in (128, 127, 126, 125, 124, 120, 112, 100, 96) if sc.n_obs % b != 1 and st_train.n_obs % b != 1)
model.train(max_epochs=epochs, batch_size=batch, accelerator="gpu" if torch.cuda.is_available() else "cpu",
            enable_progress_bar=False)
_, imputed = model.get_imputed_values(normalized=True, batch_size=batch)   # spots x sc genes; px_scale frequencies (sum to 1 per spot)
imputed = np.asarray(imputed, dtype=np.float32)
names = {g: i for i, g in enumerate(sc.var_names)}
pred = np.zeros((st.n_obs, len(held)), np.float32)
for j, g in enumerate(held):
    i = names.get(g, -1)
    if i >= 0:
        pred[st_ok, j] = imputed[:, i]
# spots without counts over the measured genes cannot be placed by the model: give them the mean prediction
if (~st_ok).any() and st_ok.any():
    pred[~st_ok] = pred[st_ok].mean(axis=0)
if not np.isfinite(pred).all():
    raise SystemExit("gimVI produced non-finite predictions (training diverged)")
save(args.work, pred, scale="freq")
