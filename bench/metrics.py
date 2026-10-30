"""
bench/metrics.py -- scoring.

Masked-entry recovery (protocol P1 / P2):
    pooled Spearman and log-Pearson at hidden positions; per-gene Spearman and
    Pearson medians (genes with >= 30 hidden entries); fill fraction of hidden
    entries; Spearman on filled entries only; zero-selection AUROC / AUPRC
    (imputed value as a score for "hidden non-zero" vs "observed zero");
    densification at a fixed sample of observed zeros; per-zone Spearman.

Leave-genes-out prediction (protocol P3), per gene across spots, following
Li et al. 2022 as adopted by stDiff / SpaIM:
    PCC, Spearman, SSIM (global form on max-scaled vectors), RMSE on z-scores,
    Jensen-Shannon divergence on the vectors normalised to distributions.
    accuracy_score() turns per-gene metrics of several methods into the
    average normalised rank.

Downstream clustering (SpaMean-Impute / stDiff): ARI, NMI, AMI, homogeneity
of Leiden clusters against labels, with scanpy's default pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# ----------------------------------------------------------------------------- P1 / P2

def _safe_spearman(a, b):
    if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    return float(spearmanr(a, b).correlation)


def _safe_pearson(a, b):
    if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    return float(pearsonr(a, b)[0])


def masked_metrics(truth, pred, cols, zones=None, zero_scores=None, min_per_gene=30, strata=None, min_per_stratum=20):
    """truth/pred: values at hidden positions; cols: gene index of each hidden
    position; zones: per-position zone label (optional); zero_scores: imputed
    values at a sample of observed zeros (optional); strata: per-position depth
    group of the spot (optional), for the depth-stratified per-gene Spearman."""
    truth = np.asarray(truth, float); pred = np.asarray(pred, float)
    fin = np.isfinite(pred)
    truth, pred, cols = truth[fin], pred[fin], np.asarray(cols)[fin]
    if zones is not None:
        zones = np.asarray(zones)[fin]
    if strata is not None:
        strata = np.asarray(strata)[fin]
    out = {}
    out["n_hidden"] = int(truth.size)
    # pooled over all hidden entries and genes: dominated by between-gene abundance and by which
    # entries were filled (unfilled = 0); kept for comparability with the conference paper
    out["spearman_all"] = _safe_spearman(truth, pred)
    out["pearson_log_all"] = _safe_pearson(np.log1p(truth), np.log1p(pred))
    out["rmse_log_all"] = float(np.sqrt(np.mean((np.log1p(truth) - np.log1p(pred)) ** 2)))
    filled = pred > 0
    out["hidden_fill_frac"] = float(filled.mean())
    out["spearman_filled_only"] = _safe_spearman(truth[filled], pred[filled]) if filled.sum() > 30 else np.nan
    # per gene
    df = pd.DataFrame({"g": cols, "t": truth, "p": pred})
    per = []
    for g, d in df.groupby("g"):
        if len(d) >= min_per_gene:
            per.append((g, _safe_spearman(d.t.values, d.p.values), _safe_pearson(np.log1p(d.t.values), np.log1p(d.p.values)),
                        float((d.p.values > 0).mean()), int(len(d))))
    per = pd.DataFrame(per, columns=["gene", "spearman", "pearson_log", "fill_frac", "n_hidden"])
    out["n_genes_scored"] = int(len(per))
    out["n_genes_with_fill"] = int(per.spearman.notna().sum()) if len(per) else 0
    out["pergene_spearman_median"] = float(np.nanmedian(per.spearman)) if len(per) and per.spearman.notna().any() else np.nan
    out["pergene_spearman_mean"] = float(np.nanmean(per.spearman)) if len(per) and per.spearman.notna().any() else np.nan
    # a gene the method never fills contributes 0 (no recovery) instead of being dropped
    out["pergene_spearman_median_zerofill"] = float(np.median(per.spearman.fillna(0.0))) if len(per) else np.nan
    # within-gene ordering pooled over entries: per-gene Spearman weighted by the gene's hidden entries, unfilled
    # genes as 0.  (A pooled Spearman of within-gene ranks is not usable: count ties give every gene a
    # size-dependent rank structure, so an all-zero prediction scored 0.35 on the kidney.)
    if len(per):
        w = per.n_hidden.values.astype(float)
        out["pergene_spearman_wmean"] = float(np.sum(w * per.spearman.fillna(0.0).values) / w.sum())
    else:
        out["pergene_spearman_wmean"] = np.nan
    out["pergene_pearson_log_median"] = float(np.nanmedian(per.pearson_log)) if len(per) and per.pearson_log.notna().any() else np.nan
    # the same within groups of spots of similar depth: Spearman per (gene, depth group), weighted by
    # entries.  Depth alone can no longer order the entries, so this is what a method adds beyond depth.
    if strata is not None:
        key = np.asarray(cols).astype(np.int64) * 64 + strata.astype(np.int64)
        num = den = 0.0
        for _, g in pd.DataFrame({"k": key, "t": truth, "p": pred}).groupby("k", sort=False):
            n = len(g)
            if n < min_per_stratum:
                continue
            r = _safe_spearman(g.t.values, g.p.values)
            num += n * (r if np.isfinite(r) else 0.0); den += n
        out["pergene_spearman_depthstrat"] = float(num / den) if den else np.nan
        out["depthstrat_entry_share"] = float(den / max(truth.size, 1))
    # zone stratification
    if zones is not None:
        for z in np.unique(zones):
            m = zones == z
            out[f"spearman_zone_{z}"] = _safe_spearman(truth[m], pred[m]) if m.sum() > 50 else np.nan
            out[f"n_zone_{z}"] = int(m.sum())
    # zero selection as classification: hidden non-zeros (1) vs observed zeros (0)
    if zero_scores is not None:
        from sklearn.metrics import roc_auc_score, average_precision_score
        zs = np.asarray(zero_scores, float); zs = zs[np.isfinite(zs)]
        y = np.concatenate([np.ones(pred.size), np.zeros(zs.size)])
        s = np.concatenate([pred, zs])
        if zs.size > 0 and pred.size > 0:
            out["zero_select_auroc"] = float(roc_auc_score(y, s))
            out["zero_select_auprc"] = float(average_precision_score(y, s))
        out["densification_at_zeros"] = float((zs > 0).mean()) if zs.size else np.nan
    return out, per


# ----------------------------------------------------------------------------- P3

def _ssim_global(x, y, c1=0.01 ** 2, c2=0.03 ** 2):
    x = x / (x.max() if x.max() > 0 else 1.0)
    y = y / (y.max() if y.max() > 0 else 1.0)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = np.mean((x - mx) * (y - my))
    return float(((2 * mx * my + c1) * (2 * cov + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2)))


def _rmse_z(x, y):
    """RMSE between z-scored vectors; NaN when either is (numerically) constant, so a
    constant prediction cannot score better than an informative one."""
    sx, sy = x.std(), y.std()
    if sx <= 1e-8 * max(1.0, np.abs(x).max()) or sy <= 1e-8 * max(1.0, np.abs(y).max()):
        return float("nan")
    return float(np.sqrt(np.mean(((x - x.mean()) / sx - (y - y.mean()) / sy) ** 2)))


def _js(x, y, eps=1e-12):
    p = np.maximum(x, 0) + eps; q = np.maximum(y, 0) + eps
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def gene_prediction_metrics(truth: np.ndarray, pred: np.ndarray, gene_names) -> pd.DataFrame:
    """truth/pred: spots x genes (measured vs predicted). One row per gene."""
    rows = []
    for j, g in enumerate(gene_names):
        t = np.asarray(truth[:, j], float); p = np.asarray(pred[:, j], float)
        rows.append(dict(gene=g, pcc=_safe_pearson(t, p), spearman=_safe_spearman(t, p),
                         ssim=_ssim_global(t, p), rmse_z=_rmse_z(t, p), js=_js(t, p)))
    return pd.DataFrame(rows)


def cellwise_spearman(truth: np.ndarray, pred: np.ndarray, min_genes=5):
    """Spearman per spot across the predicted genes (the stPlus cell-wise measure), median and
    mean over spots whose truth is not constant; NaN when fewer than ``min_genes`` genes."""
    truth = np.asarray(truth, float); pred = np.asarray(pred, float)
    if truth.shape[1] < min_genes:
        return dict(cell_spearman_median=np.nan, cell_spearman_mean=np.nan, cell_spearman_n=0)
    from scipy.stats import rankdata
    rt = np.apply_along_axis(rankdata, 1, truth); rp = np.apply_along_axis(rankdata, 1, pred)
    rt = rt - rt.mean(axis=1, keepdims=True); rp = rp - rp.mean(axis=1, keepdims=True)
    den = np.sqrt((rt ** 2).sum(axis=1) * (rp ** 2).sum(axis=1))
    ok = den > 0
    r = np.where(ok, (rt * rp).sum(axis=1) / np.where(ok, den, 1.0), np.nan)
    return dict(cell_spearman_median=float(np.nanmedian(r)) if ok.any() else np.nan,
                cell_spearman_mean=float(np.nanmean(r)) if ok.any() else np.nan, cell_spearman_n=int(ok.sum()))


def accuracy_score(per_gene_by_method: dict[str, pd.DataFrame], min_coverage: float = 1.0):
    """Li et al. 2022 accuracy score: for every gene and metric rank the methods
    (best = 1, ties share the average rank), normalise to [0, 1] with 1 = best,
    average over metrics and genes.  Input: {method: DataFrame from
    gene_prediction_metrics}, possibly concatenated over seeds (averaged per gene
    here).  Only methods covering at least ``min_coverage`` of the union of genes
    are ranked, so a method with partial folds cannot shrink everyone's gene set;
    the returned frame records the genes used."""
    from scipy.stats import rankdata
    higher = {"pcc": True, "spearman": True, "ssim": True, "rmse_z": False, "js": False}
    tables = {m: d.groupby("gene")[list(higher)].mean() for m, d in per_gene_by_method.items()}
    all_genes = set().union(*[set(t.index) for t in tables.values()])
    full = [m for m, t in tables.items() if len(t) >= min_coverage * len(all_genes)]
    dropped = [m for m in tables if m not in full]
    genes = sorted(set.intersection(*[set(tables[m].index) for m in full])) if full else []
    rows = []
    for m in full:
        rows.append(dict(method=m))
    n = len(full)
    scores = {m: [] for m in full}
    for metric, hi in higher.items():
        block = np.column_stack([tables[m].loc[genes, metric].values for m in full]) if genes else np.zeros((0, n))
        for r in range(block.shape[0]):
            row = block[r]
            ok = np.isfinite(row)
            if ok.sum() < 2:
                continue
            ranks = rankdata(-row[ok] if hi else row[ok], method="average")       # 1 = best
            norm = 1.0 - (ranks - 1) / max(ok.sum() - 1, 1)
            k = 0
            for i, m in enumerate(full):
                if ok[i]:
                    scores[m].append(norm[k]); k += 1
    out = pd.DataFrame({"method": full,
                        "accuracy_score": [float(np.mean(scores[m])) if scores[m] else np.nan for m in full],
                        "n_genes": [len(genes)] * n,
                        "n_ranked_cells": [len(scores[m]) for m in full]}).set_index("method")
    out.attrs["dropped_partial_coverage"] = dropped
    return out


# ----------------------------------------------------------------------------- clustering

def clustering_metrics(X, labels, n_pcs=30, seed=0, max_cells=40000, exclude=("nan", "none", "unknown", "unassigned", "")):
    """Agreement between clusters of X (spots x genes, counts or imputed values) and the given
    labels: ARI, NMI, AMI and homogeneity.  Clusters are k-means on the top principal components
    of the log-normalised matrix, with k equal to the number of labels, which is deterministic
    for a seed and needs no graph library.  Spots without a usable label are left out."""
    import scanpy as sc
    import anndata as ad
    import scipy.sparse as sp
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, homogeneity_score
    labels = np.asarray(labels).astype(str)
    ok = ~np.isin(np.char.lower(labels), exclude)
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
    X, labels = X[ok], labels[ok]
    if X.shape[0] > max_cells:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(X.shape[0], max_cells, replace=False))
        X, labels = X[keep], labels[keep]
    a = ad.AnnData(X.astype(np.float32))
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=min(2000, a.n_vars - 1))
    sc.pp.pca(a, n_comps=min(n_pcs, a.n_vars - 1), mask_var="highly_variable", random_state=seed)
    k = int(len(np.unique(labels)))
    cl = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(a.obsm["X_pca"]).labels_.astype(str)
    return dict(ari=float(adjusted_rand_score(labels, cl)), nmi=float(normalized_mutual_info_score(labels, cl)),
                ami=float(adjusted_mutual_info_score(labels, cl)), homogeneity=float(homogeneity_score(labels, cl)),
                n_clusters=k, n_spots=int(labels.size), algorithm="kmeans")
