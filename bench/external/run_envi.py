"""ENVI (Haviv et al. 2025) via the scenvi package, run from its own environment
(~/envs/zoneimpute-envi; JAX).  Package defaults are kept; --epochs overrides the
number of training steps (default 10000).  The held-out genes are passed as genes to
keep on the reference side, both tables go in as dense counts as the package expects,
and the imputed values are moved to each spot's own depth with the genes that were
measured (the imputation covers them too), so the output is on the count footing."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")     # share the GPU with other jobs

import numpy as np
import scipy.sparse as sp
from common import parse, load, save, ref_linear

args = parse()
st, sc, held = load(args.work)
if args.task != "predict":
    raise SystemExit("ENVI is run in the gene-prediction protocol only")
import jax
import scenvi

sc.X = np.asarray(ref_linear(sc).todense(), dtype=np.float32)
st.X = np.asarray(sp.csr_matrix(st.X).todense(), dtype=np.float32)
targets = [g for g in held if g in set(sc.var_names)]
model = scenvi.ENVI(spatial_data=st, sc_data=sc, sc_genes=targets)
kw = {"training_steps": int(args.epochs)} if args.epochs else {}
model.train(key=jax.random.key(args.seed), **kw)
model.impute_genes()
imp = model.spatial_data.obsm["imputation"]
measured = [g for g in model.spatial_data.var_names if g in imp.columns]
obs_lib = np.asarray(model.spatial_data[:, measured].X.sum(axis=1)).ravel().astype(np.float64)
imp_lib = imp[measured].values.sum(axis=1).astype(np.float64)
factor = np.where(imp_lib > 0, obs_lib / np.maximum(imp_lib, 1e-12), 0.0)
pred = np.zeros((st.n_obs, len(held)), np.float32)
for j, g in enumerate(held):
    if g in imp.columns:
        pred[:, j] = imp[g].values * factor
if not np.isfinite(pred).all():
    raise SystemExit("ENVI produced non-finite predictions")
save(args.work, pred, scale="counts")
