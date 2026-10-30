"""Share of spots whose k-means cluster (computed on the masked input, as relabel="kmeans") has at least
max(200, 40 K) spots, i.e. is large enough for the within-label correction; mask seed 42."""
import sys, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from bench import datasets
from bench.runner import _RecoveryContext
from zone_impute import _normalise_log
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
rows = []
for ds in sys.argv[1:]:
    pair = datasets.load(ds); ctx = _RecoveryContext(pair, "top1500" if "tissue:" not in ds else "all")
    X = ctx.hide("p1", 42, mask=0.2)[0]
    Z = TruncatedSVD(n_components=min(30, X.shape[1] - 1), random_state=42).fit_transform(_normalise_log(sp.csr_matrix(X)))
    k_ann = np.unique(np.asarray(pair.st_label).astype(str)).size
    for k in sorted({k_ann, 7, 30, 60}):
        lab = MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=4096, random_state=42).fit_predict(Z)
        sizes = np.bincount(lab); ok = sizes >= 200
        rows.append(dict(dataset=ds, k=k, annotation_k=k == k_ann, clusters_large=int(ok.sum()), share_spots_corrected=float(sizes[ok].sum() / sizes.sum())))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results/analysis/cluster_coverage.csv", index=False)
