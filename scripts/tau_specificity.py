"""Trade-off of the threshold tau on the kidney: share of zeros filled against the share of fills that fall in
the labels where the gene is enriched (rate at least twice its section mean).  Uses the full observed section
(no masking), as the illustration does; writes results/analysis/tau_specificity.csv."""
import sys, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from bench import datasets, protocols
from zone_impute import Config, impute_depth_type

TAUS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
pair = datasets.load("kidney")
st_all = sp.csr_matrix(pair.st.X); panel = protocols.gene_panel(st_all, "top1500", seed=0)
X = st_all[:, panel].tocsr(); genes = np.asarray(pair.st.var_names)[panel]
labels = np.asarray(pair.st_label).astype(str)
cfg = Config(rate_shrink=20, within_label_pcs=5, within_label_xfit=5, spatial_blend="auto", quota_mode="expected", fill_threshold=0.1,
             relabel="kmeans", seed=42, n_jobs=6)
_, det, expected, prob = impute_depth_type(X, labels, cfg, coords=pair.coords, gene_names=list(genes), return_details=True)
Xd = X.toarray() > 0
lib = np.asarray(X.sum(1)).ravel(); sf = lib / np.median(lib)
# enrichment by the stored annotation, which here only describes the tissue and plays no part in the fill
types = np.unique(labels)
rate = np.stack([X[labels == t].sum(0).A1 / sf[labels == t].sum() for t in types])          # types x genes
enriched = rate >= 2.0 * (X.sum(0).A1 / sf.sum())[None, :]
in_struct = enriched[np.searchsorted(types, labels)]                                          # spots x genes
zero = ~Xd
rows = []
for tau in TAUS:
    fill = zero & (prob >= tau)
    for name, g in [("all genes", slice(None))] + [(n, int(np.where(genes == n)[0][0])) for n in ["Umod", "Aqp2"] if n in genes]:
        f = fill[:, g]; s = in_struct[:, g]; z = zero[:, g]
        ok = enriched[:, g].any(0) if name == "all genes" else np.array(True)
        if name == "all genes":
            f, s, z = f[:, ok], s[:, ok], z[:, ok]
        rows.append(dict(tau=tau, genes=name, zeros_filled=float(f.sum() / z.sum()), fills_in_enriched=float((f & s).sum() / max(f.sum(), 1)),
                         zeros_in_enriched=float((z & s).sum() / z.sum())))
out = pd.DataFrame(rows); out.to_csv(ROOT / "results/analysis/tau_specificity.csv", index=False)
print(out.round(3).to_string())
