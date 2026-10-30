"""SpaMean-Impute (Hafiz, Azim, Shatabda 2026) via the SpaMeanImpute package.

The package's ``run`` helper also clusters and writes the unimputed copy, so the
imputation itself is called through its building blocks: zero_indices ->
knn_neighbors_dict -> impute_zeros_with_neighbors.  Options via --mode:
``k<K>_t<threshold>`` (defaults k=9, threshold=0.1 as in the package CLI;
the paper's example threshold is 0.5)."""
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from common import parse, load, save

args = parse()
work = Path(args.work)
st, sc, held = load(args.work)
if args.task != "denoise":
    raise SystemExit("SpaMean-Impute only denoises measured genes")
from spa_mean_impute.imputer import zero_indices, knn_neighbors_dict, impute_zeros_with_neighbors

k, thr = 9, 0.1
if args.mode:
    m = re.match(r"k(\d+)_t([0-9.]+)", args.mode)
    if m:
        k, thr = int(m.group(1)), float(m.group(2))
st.X = sp.csr_matrix(st.X).astype(np.float32)
zero_dict_spot, adata_filtered = zero_indices(st)          # drops all-zero genes
neighbors = knn_neighbors_dict(adata_filtered, k=k)
imputed = impute_zeros_with_neighbors(zero_dict_spot, neighbors, adata_filtered, drop_threshold=thr)
Y = imputed.X.toarray() if sp.issparse(imputed.X) else np.asarray(imputed.X)
# restore the all-zero genes that zero_indices removed, in the original order
out = np.zeros((st.n_obs, st.n_vars), dtype=np.float32)
cols = st.var_names.get_indexer(imputed.var_names)
out[:, cols] = Y.astype(np.float32)
save(args.work, out)
