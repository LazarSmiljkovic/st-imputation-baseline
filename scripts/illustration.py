"""Spatial illustration (Figure 4): a kidney section, two marker genes, observed counts, the proposed fill at
threshold 0.1 and the probability of a molecule; writes results/analysis/illustration_kidney.npz and the figure."""
import sys, numpy as np, scipy.sparse as sp
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from bench import datasets, protocols
from zone_impute import Config, impute_depth_type

GENES = ["Umod", "Slc12a1", "Nphs2", "Aqp2"]
pair = datasets.load("kidney")
st_all = sp.csr_matrix(pair.st.X); panel = protocols.gene_panel(st_all, "top1500", seed=0)
X = st_all[:, panel].tocsr(); genes = np.asarray(pair.st.var_names)[panel]
cfg = Config(rate_shrink=20, within_label_pcs=5, within_label_xfit=5, spatial_blend="auto", quota_mode="expected", fill_threshold=0.1, seed=42)
out, det, expected, prob = impute_depth_type(X, np.asarray(pair.st_label), cfg, coords=pair.coords, gene_names=list(genes), return_details=True)
idx = [int(np.where(genes == g)[0][0]) for g in GENES if g in genes]
np.savez_compressed(ROOT / "results/analysis/illustration_kidney.npz", coords=pair.coords, labels=np.asarray(pair.st_label).astype(str),
                    genes=genes[idx], observed=X[:, idx].toarray(), filled=out[:, idx], expected=expected[:, idx], prob=prob[:, idx],
                    spatial_w=det.get("spatial_w", np.nan), lib=np.asarray(X.sum(1)).ravel())
print("saved", genes[idx], "spatial_w", det.get("spatial_w"))
