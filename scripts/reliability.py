"""Calibration and decision quality of the model-implied probability of a molecule under binomial thinning.

At every entry that is zero after thinning the event is 'at least one removed molecule' (y_rem > 0); the
prediction mu of a method is on the scale of the removed half (keep probability 0.5).  Hidden positives are
stored for every such entry; true zeros as a random sample reweighted to their number.

Probabilities: Poisson p = 1 - exp(-mu); negative binomial p = 1 - (1 + phi_g mu)^(-1/phi_g), with a per-gene
dispersion phi_g estimated by moments on the thinned input only (the kept molecules; binomial thinning keeps the
NB dispersion), around the depth x cluster expectation with k-means clusters of that input.  Nothing the scores
are computed on enters the dispersion.
Writes results/analysis/reliability.csv (bins), reliability_summary.csv (Brier, log-loss, ECE, average
precision) and fill_decision.csv (precision and recall of filling at thresholds tau)."""
import os, sys, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path
from sklearn.metrics import average_precision_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from bench import datasets, protocols
from bench.runner import _RecoveryContext
BENCH = Path(os.environ.get("ZONEIMPUTE_RESULTS", ROOT / "results/bench"))      # a rerun: results/rerun

D = "ours:depth_type;rate_shrink=20"
M = {f"{D};within_label_pcs=5;within_label_xfit=5;spatial_blend=auto;relabel=kmeans": "proposed", f"{D};within_label_pcs=5;within_label_xfit=5;spatial_blend=auto": "proposed, stored labels", D: "label x depth", "scvi": "scVI", "magic": "MAGIC", "gene_depth": "depth only"}
DS = ["kidney", "kidney_puck:GSM5713381", "adultB5", "embryo_raw", "dlpfc:151673", "visium10x"]
EDGES = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001])
TAUS = [0.05, 0.1, 0.2, 0.3, 0.5]


def kmeans_labels(X, k, seed):
    lib = np.asarray(X.sum(1)).ravel().astype(np.float64); lib[lib == 0] = 1.0
    L = sp.diags((1e4 / lib).astype(np.float32)) @ X.astype(np.float32); L.data = np.log1p(L.data)
    Z = TruncatedSVD(n_components=min(30, X.shape[1] - 1), random_state=seed).fit_transform(L)
    return MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=4096, random_state=seed).fit_predict(Z)


def dispersion(X, labels):
    lib = np.asarray(X.sum(1)).ravel(); sf = lib / np.median(lib[lib > 0]); sf[sf == 0] = 1e-3
    labels = np.asarray(labels).astype(str); Xd = X.toarray().astype(np.float64)
    mu = np.zeros_like(Xd)
    for t in np.unique(labels):
        r = labels == t
        mu[r] = sf[r, None] * (Xd[r].sum(0) / sf[r].sum())[None, :]
    num = ((Xd - mu) ** 2 - mu).sum(0); den = (mu ** 2).sum(0)
    return np.clip(num / np.maximum(den, 1e-12), 1e-4, 50.0)


p2 = pd.read_csv(BENCH / "p2_results.csv")
p2["m"] = p2.method.str.replace(r";n_jobs=\d+", "", regex=True).map(M)
bins, summ, dec = [], [], []
for ds in DS:
    pair = datasets.load(ds); st = sp.csr_matrix(pair.st.X); panel = protocols.gene_panel(st, "top1500", seed=0)
    X = st[:, panel].tocsr(); n_true_zero = float(np.prod(X.shape) - X.nnz)
    ctx = _RecoveryContext(pair, "top1500")
    k_types = int(np.unique(np.asarray(pair.st_label).astype(str)).size); phis = {}
    for _, r in p2[(p2.dataset == ds) & p2.m.notna()].iterrows():
        if r.seed not in phis:
            Xt = ctx.hide("p2", int(r.seed), thin=float(r.thin))[0]      # exactly the input the methods received
            phis[r.seed] = dispersion(Xt, kmeans_labels(Xt, k_types, int(r.seed)))
        phi = phis[r.seed]
        f = BENCH / "runs" / r.run_id / "hidden_pred.npz"
        if not f.exists():
            continue
        z = np.load(f)
        mu = np.concatenate([np.maximum(z["pred"].astype(np.float64), 0), np.maximum(z["zero_pred"].astype(np.float64), 0)])
        g = np.concatenate([z["cols"], z["zero_cols"]]).astype(int)
        y = np.concatenate([np.ones(len(z["pred"])), np.zeros(len(z["zero_pred"]))])
        w = np.concatenate([np.ones(len(z["pred"])), np.full(len(z["zero_pred"]), n_true_zero / max(len(z["zero_pred"]), 1))])
        for model, p in (("Poisson", 1.0 - np.exp(-mu)), ("NB", 1.0 - (1.0 + phi[g] * mu) ** (-1.0 / phi[g]))):
            pc = np.clip(p, 1e-6, 1 - 1e-6)
            brier = float(np.sum(w * (p - y) ** 2) / w.sum())
            logloss = float(-np.sum(w * (y * np.log(pc) + (1 - y) * np.log(1 - pc))) / w.sum())
            k = np.digitize(p, EDGES) - 1; ece = 0.0
            for b in range(len(EDGES) - 1):
                m_ = k == b
                if not m_.any():
                    continue
                wb = w[m_].sum(); pm = float(np.sum(w[m_] * p[m_]) / wb); om = float(np.sum(w[m_] * y[m_]) / wb)
                ece += wb / w.sum() * abs(pm - om)
                bins.append(dict(dataset=ds, method=r.m, prob=model, seed=r.seed, bin=b, mean_pred=pm, observed=om, weight=wb / w.sum()))
            base = float(np.sum(w * y) / w.sum())
            hi = p >= 0.1; ece_fill = 0.0
            for b in range(len(EDGES) - 1):
                m_ = hi & (k == b)
                if m_.any():
                    wb = w[m_].sum(); ece_fill += wb / w[hi].sum() * abs(float(np.sum(w[m_] * p[m_]) / wb) - float(np.sum(w[m_] * y[m_]) / wb))
            ap = float(average_precision_score(y, p, sample_weight=w))
            summ.append(dict(dataset=ds, method=r.m, prob=model, seed=r.seed, brier=brier, brier_reference=base * (1 - base), logloss=logloss, ece=ece, ece_filled=ece_fill if hi.any() else np.nan,
                             mean_pred=float(np.sum(w * p) / w.sum()), observed_rate=base, average_precision=ap))
            if True:
                for tau in TAUS:
                    fill = p >= tau
                    tp = float(np.sum(w * (fill & (y == 1)))); fp = float(np.sum(w * (fill & (y == 0))))
                    dec.append(dict(dataset=ds, method=r.m if model == "Poisson" else f"{r.m}, NB", seed=r.seed, tau=tau, fill_share=float(np.sum(w * fill) / w.sum()),
                                    precision=tp / max(tp + fp, 1e-12), recall=tp / max(float(np.sum(w * y)), 1e-12)))
            print(ds, r.m, model, r.seed, f"brier {brier:.5f} ece {ece:.4f} ap {ap:.3f} pred {summ[-1]['mean_pred']:.4f} obs {base:.4f}", flush=True)
pd.DataFrame(bins).to_csv(ROOT / "results/analysis/reliability.csv", index=False)
pd.DataFrame(summ).to_csv(ROOT / "results/analysis/reliability_summary.csv", index=False)
pd.DataFrame(dec).to_csv(ROOT / "results/analysis/fill_decision.csv", index=False)
