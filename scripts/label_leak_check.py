"""Separate label leakage from label quality: the final configuration with (a) the stored labels, (b) k-means
labels computed on the full, unmasked section, (c) the same k-means computed on the masked section; same masks."""
import sys, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from bench import datasets, protocols, metrics
from zone_impute import impute_depth_type, Config, _normalise_log
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
def km(X, k, seed):
    Z = TruncatedSVD(n_components=min(30, X.shape[1] - 1), random_state=seed).fit_transform(_normalise_log(sp.csr_matrix(X)))
    return np.array([f"k{c}" for c in MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=4096, random_state=seed).fit_predict(Z)])
cfg = Config(rate_shrink=20, within_label_pcs=5, within_label_xfit=5, spatial_blend="auto", quota_mode="expected", fill_threshold=0.1, n_jobs=6)
rows = []
for name, genes in [("tissue:DataUploadFinal/Dataset17", "all"), ("tissue:DataUploadFinal/Dataset15", "all"), ("tissue:DataUploadFinal/Dataset9", "all"),
                    ("tissue:DataUploadFinal/Dataset13", "all"), ("adultB5", "top1500"), ("kidney", "top1500")]:
    pair = datasets.load(name); st = sp.csr_matrix(pair.st.X); panel = protocols.gene_panel(st, genes, seed=0); X = st[:, panel].tocsr()
    lab = np.asarray(pair.st_label).astype(str); k = np.unique(lab).size
    for seed in (42, 43):
        masked, r, c, truth = protocols.mask_entries(X, 0.2, seed)
        for src, L in (("stored", lab), ("kmeans full", km(X, k, seed)), ("kmeans masked", km(masked, k, seed))):
            out = impute_depth_type(masked, L, cfg.with_(seed=seed), coords=pair.coords)
            m, _ = metrics.masked_metrics(truth, out[r, c], c)
            rows.append(dict(dataset=name, seed=seed, labels=src, k=k, wmean=m["pergene_spearman_wmean"]))
            print(name, seed, src, round(m["pergene_spearman_wmean"], 4), flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results/analysis/label_leak_check.csv", index=False)
