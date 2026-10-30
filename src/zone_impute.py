"""
zone_impute.py
==============

Switchable zone-aware imputation engine (version 3).

Every design decision is a field of :class:`Config`, so any step can be run
with or without a change.  Three presets reproduce earlier behaviour in
distribution:

* ``PRESETS["published"]``  -- the YuInfo 2026 notebook (exp-space KDE,
  uniform neighbour mean, discrete 0.85/0.65/0.45 weights, random zero fill,
  reference zero-rate quota).
* ``PRESETS["corrected"]``  -- ``impute_zone_improved`` from
  ``imputation_improved.py`` (log-space KDE, Gaussian neighbour weights,
  logistic weight ramp, evidence-ranked zero fill).
* ``PRESETS["v3"]``         -- all the new ideas switched on.

New switches (see :class:`Config` for defaults):

reference term
    ``ref_mode``      ``marginal`` (per-type distribution, as before) or
                      ``conditional`` (reference cells weighted by similarity
                      to the spot in a shared PCA space; the draw or mean is
                      taken over the spot's ``ref_k`` nearest reference cells of
                      its type).
    ``ref_dist``      ``kde`` (bootstrap + Gaussian jitter in log1p space, the
                      sampling form of a Gaussian KDE) or ``nb`` (negative
                      binomial fitted by moments to the expressing reference
                      cells, scaled by the spot's library size).
    ``ref_estimate``  ``draw`` (stochastic), ``mean`` (deterministic mean over
                      the expressing neighbour cells, no depth), ``posterior``
                      (Gamma-Poisson posterior mean with the reference as prior
                      and the spot's depth as exposure), or ``expected`` (the
                      spot's size factor times the neighbour mean with zeros
                      included, on the section's own rate scale for the type).
    ``rate_shrink``   with the ``expected`` estimate: pseudo-exposure (in units of
                      a median-depth spot) by which a cell type's rate is pulled
                      toward the gene's rate over the whole section, so that small
                      types do not get noisy or zero rates.
    ``ref_mode="none"``  the reference-free estimate :func:`impute_depth_type`:
                      size factor times the section's own rate for the spot's
                      type.  The reference then enters only through the labels.
                      Its own options (8 October 2026):
                      ``soft_labels`` none / kmeans (memberships to k-means
                      clusters of the section, ``soft_k``, ``soft_temp``) / given
                      (a probability matrix handed in through ``aux["probs"]``);
                      ``sf_mode`` total / nomito (library without mitochondrial,
                      ribosomal and haemoglobin genes) / stable (library over the
                      genes whose rate varies least across labels);
                      ``within_label_pcs`` > 0 multiplies the expected count by a
                      per-label Poisson regression of the gene on that many
                      within-label principal components of the spots
                      (``within_label_ridge``); ``rate_shrink_target`` gene /
                      parent (shrink toward the label's parent class, mapping in
                      ``aux["parents"]``); ``spatial_blend`` none / fixed
                      (weight ``spatial_w`` on a neighbour rate term) / auto (the
                      weight that best predicts molecules held out by thinning
                      the section itself).
    ``predict_blend`` in :func:`predict_genes`, the weight of the type mean next
                      to the similarity-weighted mean of the nearest reference
                      cells (the type mean as backbone, similarity as correction).
    ``ref_embed``     similarity space for the conditional reference: ``pca``
                      (joint truncated SVD of all shared genes, as before) or
                      ``markers`` (cosine similarity on the spot's cluster's top
                      ``marker_n`` marker genes, chosen on the reference by a
                      Welch t statistic of the cluster against all other cells).
    ``ref_space``     how the reference is stored: ``auto`` detects integer
                      counts vs log1p values, ``linear`` / ``log1p`` declare it,
                      ``as_is`` leaves it untouched (legacy ``exp`` sampling).

spatial term
    ``spatial_mode``          ``none`` / ``uniform`` / ``weighted`` (Gaussian on
                              distance, width = median neighbour distance).
    ``radius_mult``           if > 0, only neighbours within
                              ``radius_mult`` x median nearest-neighbour
                              distance count (``k`` is then a cap), which makes
                              the neighbourhood a circle in the data's own units.
    ``spatial_nonzero_only``  value = mean over neighbours that detect the gene;
                              detection fraction becomes the spatial evidence.
    ``type_weight``           ``none`` / ``soft`` (different-type neighbours get
                              weight ``type_weight_alpha``) / ``same_only``.

mixing weight
    ``weighting``  ``none`` / ``discrete`` / ``continuous`` / ``reversed`` /
                   ``table`` (``weight_table`` = (bin_edges, weights) fitted
                   elsewhere on held-out entries).

which zeros to fill
    ``zero_select``  ``random`` / ``evidence`` (spatial) / ``ref_evidence``
                     (fraction of the spot's reference neighbours expressing the
                     gene) / ``combined`` (rank average of both) / ``posterior``
                     (probability that the zero is a dropout given depth) /
                     ``detect_prob`` (probability that a non-zero would be observed
                     here, pi * (1 - P(0))) / ``expected`` (the expected count at
                     the spot's depth, size factor times the neighbour mean with
                     zeros included; the same quantity as ``ref_estimate="expected"``).
    ``quota_mode``   ``reference`` (fill until the reference zero rate is
                     reached, as before) / ``depth`` (fill the expected number
                     of dropouts under a Poisson depth model) / ``none`` (fill
                     every zero whose posterior dropout probability is at least
                     ``fill_threshold``) / ``expected`` (fill every zero where
                     the chance of at least one molecule at that depth,
                     1 - exp(-expected count), is at least ``fill_threshold``).

extras
    ``passes``       2 = recompute the spatial term from the first-pass result
                     and re-blend the filled entries, damped by ``damping``.
    ``denoise_all``  also replace observed non-zeros by their posterior mean
                     (only meaningful with ``ref_estimate="posterior"``).

All functions take a reference matrix (cells x genes), reference labels, an
ST matrix (spots x genes), ST labels and ST coordinates, with the two gene
sets already intersected and in the same order.  The output is a float32
dense matrix of the ST shape.  Observed non-zeros are untouched unless
``denoise_all`` is set.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict, replace
from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

__all__ = ["Config", "PRESETS", "impute", "predict_genes", "neighbor_homogeneity", "spatial_graph",
           "weights_from_h", "detect_ref_space", "fit_weight_table"]


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Config:
    # reference count space
    ref_space: str = "auto"              # auto | linear | log1p | as_is
    # reference term
    ref_mode: str = "marginal"           # marginal | conditional | none (reference-free, see impute_depth_type)
    ref_dist: str = "kde"                # kde | nb
    ref_estimate: str = "draw"           # draw | mean | posterior | expected
    kde_space: str = "log"               # log | exp (legacy, requires ref_space="as_is")
    ref_k: int = 50
    ref_pca: int = 50
    ref_embed: str = "pca"               # pca (joint SVD of all shared genes) | markers (the cluster's marker genes)
    marker_n: int = 50                   # markers per cluster for ref_embed="markers"
    ref_kernel: str = "gaussian"         # gaussian | rank
    min_spatial_mean_ratio: float = 0.01
    rate_shrink: float = 0.0             # pseudo-exposure pulling a type's rate toward the gene's overall rate
    # reference-free estimate (ref_mode="none") only
    rate_shrink_target: str = "gene"     # gene (the gene's section-wide rate) | parent (the rate of the label's parent class)
    sf_mode: str = "total"               # total | nomito (library without mt-/ribosomal/haemoglobin genes) | stable (library over genes with the most uniform rate across labels)
    soft_labels: str = "none"            # none | kmeans (soft memberships to k-means clusters of the section) | given (a probability matrix supplied by the caller)
    soft_k: int = 15
    soft_temp: float = 1.0
    within_label_pcs: int = 0            # >0: per-label Poisson regression of each gene on this many within-label principal components
    within_label_ridge: float = 1.0
    within_label_xfit: int = 1           # >1: cross-fitted correction; genes split into this many groups, each group's components come from the other genes
    within_label_source: str = "section" # section | reference (components and coefficients learned on the reference cells of the label) | blend (reference coefficients as the prior of the section's fit)
    n_jobs: int = 1                      # worker processes for the within-label correction (one task per label and gene group; results identical for any value)
    relabel: str = "none"                # none | kmeans: ignore the given labels and cluster the input matrix itself (k-means on 30 components of the log-normalised input), so no label can carry information about hidden entries
    tune_keep: float = 0.5               # keep probability of the thinning that chooses the neighbour weight
    relabel_k: int = 0                   # number of clusters for relabel=kmeans; 0 = as many as there are distinct given labels
    within_label_prior: float = 1.0      # blend: strength of the pull toward the reference coefficients
    within_label_candidates: str = ""    # e.g. "0,5": choose the number of components by thinning the section (needs no reference); overrides within_label_pcs
    tune_score: str = "deviance"         # criterion of the thinning step: deviance (proper score) | rank (per-gene Spearman at the thinned zeros)
    spatial_blend: str = "none"          # none | fixed (weight spatial_w on the neighbour rate term) | auto (weight chosen by thinning the section itself)
    spatial_w: float = 0.0
    predict_blend: float = 0.0           # predict_genes: weight of the type mean next to the similarity-weighted mean
    predict_backbone: str = "label"      # predict_genes: label (one type per spot, as above) | soft (per-spot composition over fine reference clusters, fitted on the shared genes; uses no spot labels)
    predict_clusters: int = 40           # predict_genes, soft backbone: number of reference clusters (0 = the reference's own labels)
    predict_within: int = 0              # predict_genes: >0 replaces the similar-cell average by a within-cluster regression of the target gene on this many components, learned on the reference
    predict_smooth: str = "zone"         # predict_genes: zone (the published homogeneity rule) | none | fixed (predict_smooth_w) | auto (weight chosen by predict_tune)
    predict_smooth_w: float = 0.0
    predict_tune: str = "none"           # predict_genes: none | genes (hold out a share of the shared genes once and choose backbone, clusters, components and the smoothing weight by their recovery)
    predict_tune_frac: float = 0.2
    predict_detail: str = "none"         # predict_genes, soft backbone: none | knn (multiply the composition's level by the spot's similar-cell deviation from its type mean, the within-type detail of the label backbone)
    # spatial neighbourhood
    k: int = 10
    spatial_mode: str = "weighted"       # none | uniform | weighted
    radius_mult: float = 0.0
    spatial_nonzero_only: bool = False
    type_weight: str = "none"            # none | soft | same_only
    type_weight_alpha: float = 0.25
    # homogeneity -> weight on the reference
    weighting: str = "continuous"        # none | discrete | continuous | reversed | table
    w_lo: float = 0.45
    w_hi: float = 0.85
    h0: float = 0.65
    tau: float = 0.12
    weight_table: Optional[tuple] = None  # (bin_edges, weights)
    # which zeros
    zero_select: str = "evidence"        # random | evidence | ref_evidence | combined | detect_prob | expected | posterior
    quota_mode: str = "reference"        # reference | depth | none | expected
    fill_threshold: float = 0.5
    prob_model: str = "poisson"          # poisson: p = 1 - exp(-mu) | nb: per-gene NB dispersion by moments on the input
    # extras
    passes: int = 1
    damping: float = 0.5
    denoise_all: bool = False
    seed: int = 42
    gene_chunk: int = 1000

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["weight_table"] is not None:
            d["weight_table"] = [list(map(float, d["weight_table"][0])),
                                 list(map(float, d["weight_table"][1]))]
        return d


PRESETS: dict[str, Config] = {
    "published": Config(ref_space="as_is", kde_space="exp", spatial_mode="uniform",
                        weighting="discrete", zero_select="random", quota_mode="reference"),
    "corrected": Config(ref_space="auto", kde_space="log", spatial_mode="weighted",
                        weighting="continuous", zero_select="evidence", quota_mode="reference"),
    "v3": Config(ref_space="auto", ref_mode="conditional", ref_dist="kde", ref_estimate="draw",
                 spatial_mode="weighted", spatial_nonzero_only=True, type_weight="soft",
                 weighting="continuous", zero_select="combined", quota_mode="depth"),
    "markers": Config(ref_space="auto", ref_mode="conditional", ref_embed="markers", ref_estimate="mean",
                      zero_select="ref_evidence", quota_mode="reference", weighting="none", spatial_mode="none"),
    "depth_type": Config(ref_mode="none", ref_estimate="expected", zero_select="expected", quota_mode="none",
                         fill_threshold=0.0, weighting="none", spatial_mode="none"),
    "v3_posterior": Config(ref_space="auto", ref_mode="conditional", ref_dist="nb",
                           ref_estimate="posterior", spatial_mode="weighted",
                           spatial_nonzero_only=True, type_weight="soft", weighting="continuous",
                           zero_select="posterior", quota_mode="depth"),
}


# --------------------------------------------------------------------------- #
#  Small helpers
# --------------------------------------------------------------------------- #

def _csr(X) -> sp.csr_matrix:
    return X.tocsr() if sp.issparse(X) else sp.csr_matrix(np.asarray(X))


def _dense(X, dtype=np.float32) -> np.ndarray:
    return (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(dtype, copy=False)


def detect_ref_space(X, sample: int = 200_000) -> str:
    """Guess whether a reference matrix holds linear counts or log1p values."""
    data = X.data if sp.issparse(X) else np.asarray(X)[np.asarray(X) != 0]
    if data.size == 0:
        return "linear"
    if data.size > sample:
        rng = np.random.default_rng(0)
        data = data[rng.choice(data.size, sample, replace=False)]
    integral = np.allclose(data, np.round(data), atol=1e-6)
    if integral:
        return "linear"
    return "log1p" if float(np.max(data)) < 30.0 else "linear"


def _untruncate_rate(m: np.ndarray, iters: int = 20) -> np.ndarray:
    """Poisson rate lam with zero-truncated mean m, i.e. lam/(1-exp(-lam)) = m (m > 1); Newton steps."""
    m = np.asarray(m, dtype=np.float64)
    lam = np.maximum(m - 1.0, 1e-6)                     # m -> lam + 1 for large lam; a safe start
    for _ in range(iters):
        e = np.exp(-lam)
        f = lam / np.maximum(1.0 - e, 1e-12) - m
        df = (1.0 - e - lam * e) / np.maximum((1.0 - e) ** 2, 1e-12)
        lam = np.maximum(lam - f / np.maximum(df, 1e-6), 1e-6)
    return np.where(m <= 1.0 + 1e-9, np.maximum(m, 1e-6), lam)


def _silverman_bw(values: np.ndarray) -> float:
    n = values.size
    if n < 2:
        return 1.0
    return max(1.06 * float(np.std(values)) * n ** (-0.2), 1e-6)


def _nb_dispersion(X: sp.csr_matrix, sf: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-gene negative binomial dispersion by moments around the size factor x label-rate expectation,
    from the input counts only; var = mu + phi mu^2, clipped to [1e-4, 50]."""
    labels = np.asarray(labels).astype(str); sf = np.asarray(sf, dtype=np.float64)
    num = np.zeros(X.shape[1]); den = np.zeros(X.shape[1])
    for t_ in np.unique(labels):
        r = np.where(labels == t_)[0]
        Xr = X[r].toarray().astype(np.float64)
        mu = sf[r, None] * (Xr.sum(0) / max(sf[r].sum(), 1e-12))[None, :]
        num += ((Xr - mu) ** 2 - mu).sum(0); den += (mu ** 2).sum(0)
    return np.clip(num / np.maximum(den, 1e-12), 1e-4, 50.0)


def _normalise_log(X: sp.csr_matrix, target: float = 1e4) -> sp.csr_matrix:
    X = X.astype(np.float32).tocsr(copy=True)
    lib = np.asarray(X.sum(axis=1)).ravel()
    lib[lib == 0] = 1.0
    X = sp.diags((target / lib).astype(np.float32)) @ X
    X.data = np.log1p(X.data)
    return X.tocsr()


# --------------------------------------------------------------------------- #
#  Spatial structure
# --------------------------------------------------------------------------- #

def knn_indices(coords: np.ndarray, k: int):
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(coords)
    dist, idx = nn.kneighbors(coords)
    return dist[:, 1:], idx[:, 1:]


def neighbor_homogeneity(labels, coords, k: int) -> np.ndarray:
    """Fraction of the k nearest spots (excluding self) sharing the spot's label."""
    _, idx = knn_indices(coords, k)
    labels = np.asarray(labels)
    return (labels[idx] == labels[:, None]).mean(axis=1)


def spatial_graph(coords: np.ndarray, labels, cfg: Config):
    """Row-normalised sparse neighbour-weight matrix A (n x n) and the homogeneity h."""
    n = coords.shape[0]
    dist, idx = knn_indices(coords, cfg.k)
    labels = np.asarray(labels)
    same = labels[idx] == labels[:, None]
    h = same.mean(axis=1)

    if cfg.spatial_mode == "none":
        return None, h
    if cfg.spatial_mode == "uniform":
        w = np.ones_like(dist)
    elif cfg.spatial_mode == "weighted":
        sigma = float(np.median(dist)) + 1e-12
        w = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
    else:
        raise ValueError(f"unknown spatial_mode {cfg.spatial_mode}")

    if cfg.radius_mult > 0:
        r = cfg.radius_mult * float(np.median(dist[:, 0]))
        inside = dist <= r
        inside[:, 0] = True                      # always keep the nearest neighbour
        w = w * inside

    if cfg.type_weight == "soft":
        w = w * np.where(same, 1.0, cfg.type_weight_alpha)
    elif cfg.type_weight == "same_only":
        w_same = w * same
        has = w_same.sum(axis=1) > 0
        w = np.where(has[:, None], w_same, w)    # no same-type neighbour: keep all
    elif cfg.type_weight != "none":
        raise ValueError(f"unknown type_weight {cfg.type_weight}")

    rs = w.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    w = w / rs
    rows = np.repeat(np.arange(n), cfg.k)
    A = sp.csr_matrix((w.ravel(), (rows, idx.ravel())), shape=(n, n))
    return A, h


def weights_from_h(h: np.ndarray, cfg: Config) -> np.ndarray:
    """Per-spot weight on the reference term."""
    if cfg.weighting == "none":
        return np.ones_like(h)
    if cfg.weighting == "discrete":
        w = np.full_like(h, 0.65)
        w[h >= 0.8] = 0.85
        w[h < 0.5] = 0.45
        return w
    if cfg.weighting in ("continuous", "reversed"):
        s = 1.0 / (1.0 + np.exp(-(h - cfg.h0) / cfg.tau))
        lo, hi = (cfg.w_lo, cfg.w_hi) if cfg.weighting == "continuous" else (cfg.w_hi, cfg.w_lo)
        return lo + (hi - lo) * s
    if cfg.weighting == "table":
        if cfg.weight_table is None:
            raise ValueError("weighting='table' needs cfg.weight_table=(bin_edges, weights)")
        edges, vals = cfg.weight_table
        b = np.clip(np.searchsorted(np.asarray(edges), h, side="right") - 1, 0, len(vals) - 1)
        return np.asarray(vals, dtype=np.float64)[b]
    raise ValueError(f"unknown weighting {cfg.weighting}")


def fit_weight_table(h: np.ndarray, err_ref: np.ndarray, err_spat: np.ndarray,
                     n_bins: int = 5, grid=np.linspace(0.0, 1.0, 21)):
    """Pick, per homogeneity bin, the weight that minimises the error of the
    blend w*ref + (1-w)*spatial against held-out truth.

    ``err_ref`` and ``err_spat`` are (value - truth) residuals of the two terms
    at held-out entries, ``h`` the homogeneity of the entry's spot.  Returns
    ``(bin_edges, weights)`` usable as ``Config.weight_table``."""
    edges = np.linspace(0.0, 1.0 + 1e-9, n_bins + 1)
    b = np.clip(np.searchsorted(edges, h, side="right") - 1, 0, n_bins - 1)
    weights = []
    for i in range(n_bins):
        m = b == i
        if m.sum() < 20:
            weights.append(0.65)
            continue
        best, best_w = np.inf, 0.65
        for w in grid:
            e = w * err_ref[m] + (1.0 - w) * err_spat[m]
            mse = float(np.mean(e * e))
            if mse < best:
                best, best_w = mse, float(w)
        weights.append(best_w)
    return edges, np.asarray(weights)


# --------------------------------------------------------------------------- #
#  Reference embedding for the conditional term
# --------------------------------------------------------------------------- #

def _joint_embedding(sc_lin: sp.csr_matrix, st_counts: sp.csr_matrix, n_comp: int, seed: int):
    """PCA (truncated SVD on log-normalised data) fitted on the reference,
    applied to both; rows scaled to unit length so Euclidean = cosine."""
    sc_n = _normalise_log(sc_lin)
    st_n = _normalise_log(st_counts)
    n_comp = int(min(n_comp, sc_n.shape[1] - 1, sc_n.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    E = svd.fit_transform(sc_n)
    F = svd.transform(st_n)

    def unit(M):
        nrm = np.linalg.norm(M, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        return M / nrm
    return unit(E).astype(np.float32), unit(F).astype(np.float32)


def _cluster_markers(sc_ln: sp.csr_matrix, labels: np.ndarray, n: int) -> dict:
    """Top-``n`` marker genes per label on the log-normalised reference: the Welch t statistic
    of the label's cells against all other cells, largest positive values first."""
    labels = np.asarray(labels).astype(str)
    u, inv = np.unique(labels, return_inverse=True)
    X = sp.csr_matrix(sc_ln); X2 = X.multiply(X)
    T = sp.csr_matrix((np.ones(inv.size), (inv, np.arange(inv.size))), shape=(u.size, inv.size))
    n_c = np.asarray(T.sum(axis=1)).ravel()
    s1 = np.asarray((T @ X).todense(), dtype=np.float64); s2 = np.asarray((T @ X2).todense(), dtype=np.float64)
    tot1, tot2, N = s1.sum(axis=0), s2.sum(axis=0), inv.size
    out = {}
    for i, c in enumerate(u):
        n_in, n_out = n_c[i], N - n_c[i]
        if n_in < 2 or n_out < 2:
            out[c] = np.argsort(-s1[i])[:n]; continue
        m_in, m_out = s1[i] / n_in, (tot1 - s1[i]) / n_out
        v_in = np.maximum(s2[i] / n_in - m_in ** 2, 0.0); v_out = np.maximum((tot2 - s2[i]) / n_out - m_out ** 2, 0.0)
        t = (m_in - m_out) / np.sqrt(v_in / n_in + v_out / n_out + 1e-12)
        out[c] = np.argsort(-t)[:n]
    return out


def _unit_rows(M: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(M, axis=1, keepdims=True); nrm[nrm == 0] = 1.0
    return (M / nrm).astype(np.float32)


def _ref_neighbours(E_c: np.ndarray, F_c: np.ndarray, cfg: Config):
    """Sparse W (n_st_c x n_ref_c): similarity weights over the ref_k nearest
    reference cells of the same type."""
    k = int(min(cfg.ref_k, E_c.shape[0]))
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(E_c)
    dist, idx = nn.kneighbors(F_c)
    if cfg.ref_kernel == "gaussian":
        # the width is the median neighbour distance; a floor keeps the kernel from collapsing when many
        # spots or cells share one profile (common in a sparse marker space), which gave zero weights
        sigma = max(float(np.median(dist)), 1e-3)
        w = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
    elif cfg.ref_kernel == "rank":
        w = 1.0 / (np.arange(1, k + 1, dtype=np.float64))[None, :].repeat(dist.shape[0], 0)
    else:
        raise ValueError(f"unknown ref_kernel {cfg.ref_kernel}")
    tot = w.sum(axis=1, keepdims=True)
    w = np.where(tot > 0, w / np.where(tot > 0, tot, 1.0), 1.0 / k)       # a spot far from every cell: uniform weights
    rows = np.repeat(np.arange(F_c.shape[0]), k)
    W = sp.csr_matrix((w.ravel(), (rows, idx.ravel())), shape=(F_c.shape[0], E_c.shape[0]))
    return W, idx, w


# --------------------------------------------------------------------------- #
#  Reference draws
# --------------------------------------------------------------------------- #

def _draw_marginal_kde(nz_vals: np.ndarray, n: int, scale: float, rng, space: str) -> np.ndarray:
    """Bootstrap + Gaussian jitter = a sample from a Gaussian KDE."""
    if n == 0:
        return np.zeros(0)
    if space == "log":
        lv = np.log1p(nz_vals)
        bw = max(1e-3, _silverman_bw(lv))
        s = np.expm1(rng.choice(lv, size=n) + rng.normal(0.0, bw, size=n))
    else:  # legacy: KDE on stored values, then exp()
        bw = max(0.01 * float(np.mean(nz_vals)), _silverman_bw(nz_vals))
        s = np.exp(rng.choice(nz_vals, size=n) + rng.normal(0.0, bw, size=n))
    return np.maximum(0.0, s) * scale


def _nb_moments(nz_vals: np.ndarray) -> tuple[float, float]:
    """Mean and dispersion r of a negative binomial fitted by moments to the
    expressing reference cells (r large = Poisson-like)."""
    m = float(np.mean(nz_vals))
    v = float(np.var(nz_vals))
    r = m * m / (v - m) if v > m + 1e-9 else 1e6
    return m, float(np.clip(r, 1e-3, 1e6))


def _draw_nb(mean_i: np.ndarray, r: float, rng) -> np.ndarray:
    """Negative-binomial draws with per-entry means (Gamma-Poisson form)."""
    mean_i = np.maximum(mean_i, 1e-9)
    lam = rng.gamma(shape=r, scale=mean_i / r)
    return rng.poisson(lam).astype(np.float64)


def _draw_conditional(nb_vals: np.ndarray, nb_w: np.ndarray, scale: float, rng,
                      jitter_lv: float) -> np.ndarray:
    """One draw per row: pick an expressing neighbour with probability
    proportional to its weight, add KDE jitter in log1p space.  Rows with no
    expressing neighbour return NaN (caller falls back to the marginal draw)."""
    n, k = nb_vals.shape
    expr = nb_vals > 0
    w = nb_w * expr
    tot = w.sum(axis=1)
    out = np.full(n, np.nan)
    ok = tot > 0
    if not ok.any():
        return out
    cw = np.cumsum(w[ok], axis=1) / tot[ok][:, None]
    u = rng.random(ok.sum())
    pick = (cw < u[:, None]).sum(axis=1)
    pick = np.minimum(pick, k - 1)
    vals = nb_vals[ok][np.arange(ok.sum()), pick]
    lv = np.log1p(vals) + rng.normal(0.0, jitter_lv, size=vals.size)
    out[ok] = np.maximum(0.0, np.expm1(lv)) * scale
    return out


# --------------------------------------------------------------------------- #
#  Main entry point
# --------------------------------------------------------------------------- #

def _soft_kmeans_memberships(st_csr: sp.csr_matrix, k: int, temp: float, seed: int):
    """Soft memberships of every spot to k-means clusters of the section: cluster the top 30
    principal components of the log-normalised matrix, then P[i, t] ~ exp(-d_it^2 / (temp * s)),
    with s the mean squared distance of spots to their nearest centroid."""
    from sklearn.cluster import MiniBatchKMeans
    X = _normalise_log(st_csr)
    Z = TruncatedSVD(n_components=min(30, X.shape[1] - 1), random_state=seed).fit_transform(X)
    km = MiniBatchKMeans(n_clusters=int(k), n_init=3, batch_size=4096, random_state=seed).fit(Z)
    d2 = ((Z[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
    s_ = max(float(np.mean(d2.min(axis=1))), 1e-9)
    P = np.exp(-(d2 - d2.min(axis=1, keepdims=True)) / (temp * s_))
    return (P / P.sum(axis=1, keepdims=True)).astype(np.float64), np.array([f"k{t}" for t in range(int(k))])


_NUISANCE = re.compile(r"^(mt-|MT-|Mt-|Rp[sl]\d|RP[SL]\d|Rpl|Rps|Hb[ab]-|HB[AB])")


def _size_factors(st: np.ndarray, use_genes=None) -> np.ndarray:
    """Library size over the chosen genes divided by its median; zero-library spots get 1e-3."""
    lib = st[:, use_genes].sum(axis=1, dtype=np.float64) if use_genes is not None else st.sum(axis=1, dtype=np.float64)
    sf = lib / max(float(np.median(lib[lib > 0])) if (lib > 0).any() else 1.0, 1e-9)
    sf[sf == 0] = 1e-3
    return sf


def _label_rates(st_csr, P, sf, k, prior):
    """Rates per label and gene from soft (or one-hot) memberships P (n x T): counts weighted by
    membership over size factors weighted by membership, shrunk by pseudo-exposure k toward ``prior``
    (T x G, or 1 x G)."""
    Pm = sp.csr_matrix(P) if not sp.issparse(P) else P
    counts_t = np.asarray((Pm.T @ st_csr).todense(), dtype=np.float64)
    sf_t = np.asarray(Pm.T @ sf).ravel()
    return (counts_t + k * prior) / np.maximum(sf_t[:, None] + k, 1e-12)


def _poisson_ridge_fit(Z, y, off, lam, steps=6, beta0=None):
    """Per gene (columns of y): Poisson regression of the counts on the standardised scores Z with
    offset ``off`` and a ridge penalty lam * ||beta - beta0||^2 (beta0 = 0 unless a prior mean is
    given).  Returns the coefficients (G x k)."""
    G = y.shape[1]; k_ = Z.shape[1]
    beta0 = np.zeros((G, k_)) if beta0 is None else np.asarray(beta0, dtype=np.float64)
    beta = beta0.copy()
    active = y.sum(axis=0) > 0
    for _ in range(steps):
        eta = np.clip(Z @ beta.T, -np.log(5.0), np.log(5.0))          # (m x G)
        mu = np.exp(off + eta)
        grad = Z.T @ (y - mu) - lam * (beta - beta0).T                # (k x G)
        H = np.einsum("mk,mg,ml->gkl", Z, mu, Z) + lam * np.eye(k_)[None]
        step = np.linalg.solve(H, grad.T[:, :, None])[:, :, 0]       # (G x k)
        beta[active] += step[active]
    return beta


def _poisson_ridge_multiplier(Z, y, off, lam, steps=6, beta0=None):
    """exp(Z beta) clipped to [0.2, 5] for the coefficients of ``_poisson_ridge_fit``."""
    beta = _poisson_ridge_fit(Z, y, off, lam, steps, beta0)
    return np.exp(np.clip(Z @ beta.T, -np.log(5.0), np.log(5.0)))


def _standardise_cols(M):
    mu = M.mean(axis=0); sd = M.std(axis=0) + 1e-9
    return (M - mu) / sd


def _within_fit(Xl, y, off, keep, gsel, K, lam, seed, source, ref_task=None, prior=1.0):
    """One task of the within-label correction: the multiplier (rows x genes of ``gsel``) for one label and one
    gene group.  ``Xl`` is the label's log-normalised profile (dense), ``y`` its counts, ``off`` the log of the
    current estimate; ``keep`` marks the genes whose components are used (cross-fitting)."""
    if ref_task is None:
        Z = TruncatedSVD(n_components=min(K, int(keep.sum()) - 1), random_state=seed).fit_transform(sp.csr_matrix(Xl[:, keep]))
        Z = _standardise_cols(Z)
        return _poisson_ridge_multiplier(Z, y[:, gsel], off[:, gsel], lam)
    Xr_std, yr, off_r, Xl_std = ref_task
    svd = TruncatedSVD(n_components=min(K, int(keep.sum()) - 1), random_state=seed).fit(Xr_std[:, keep])
    Zr = _standardise_cols(svd.transform(Xr_std[:, keep]))
    Zs = _standardise_cols(Xl_std[:, keep] @ svd.components_.T)
    beta_ref = _poisson_ridge_fit(Zr, yr[:, gsel], off_r[:, gsel], lam)
    if source == "reference":
        return np.exp(np.clip(Zs @ beta_ref.T, -np.log(5.0), np.log(5.0)))
    return _poisson_ridge_multiplier(Zs, y[:, gsel], off[:, gsel], float(prior), beta0=beta_ref)


def _within_label_correction(st_csr, expected, hard, cfg, seed, ref=None):
    """Per label: regress every gene's observed counts on K within-label principal components of the
    log-normalised spots (Poisson regression with the current expected count as offset, ridge
    ``within_label_ridge``), and multiply the expected counts by the fitted factor, clipped to
    [0.2, 5].  With ``within_label_xfit`` > 1 the genes are split into that many groups and each
    group's components are computed from the other genes only, so that a gene's own zero cannot
    pull its own factor down (cross-fitting).

    ``within_label_source``: "section" learns components and coefficients on the section's spots of
    the label; "reference" learns both on the reference cells of the same label (``ref`` = (linear
    counts on the same genes, labels)), each domain standardised gene by gene and score by score, and
    applies the coefficients to the section's scores; "blend" fits the section's coefficients with the
    reference coefficients as the prior mean (strength ``within_label_prior``).  Labels with too few
    spots, or too few reference cells, fall back to the section fit.

    The tasks (one per label and gene group) are independent; ``cfg.n_jobs`` > 1 runs them in worker
    processes with identical results."""
    K = int(cfg.within_label_pcs); lam = float(cfg.within_label_ridge)
    folds = max(1, int(cfg.within_label_xfit))
    source = cfg.within_label_source
    out = expected.copy()
    st = _dense(st_csr, np.float32)
    G = st.shape[1]
    grp = np.random.default_rng(seed).permutation(G) % folds if folds > 1 else np.zeros(G, dtype=int)
    sc_csr, sc_lab = (None, None) if ref is None else ref
    if sc_lab is not None:
        sc_lab = np.asarray(sc_lab).astype(str)
    tasks, where = [], []
    for lab in np.unique(hard):
        rows = np.where(hard == lab)[0]
        if rows.size < max(200, 40 * K):
            continue
        Xl = _normalise_log(st_csr[rows]).toarray()
        y_all = st[rows].astype(np.float64); off_all = np.log(np.maximum(expected[rows], 1e-8)).astype(np.float64)
        use_ref = source != "section" and sc_csr is not None
        ref_task = None
        if use_ref:
            rref = np.where(sc_lab == lab)[0]
            use_ref = rref.size >= max(50, 10 * K)
        if use_ref:
            Xr = _normalise_log(sc_csr[rref]).toarray()
            yr = _dense(sc_csr[rref], np.float32).astype(np.float64)
            libr = yr.sum(axis=1); sfr = libr / max(float(np.median(libr[libr > 0])) if (libr > 0).any() else 1.0, 1e-9); sfr[sfr == 0] = 1e-3
            rate_r = yr.sum(axis=0) / sfr.sum()
            off_r = np.log(np.maximum(sfr[:, None] * rate_r[None, :], 1e-8))
            ref_task = (_standardise_cols(Xr), yr, off_r, _standardise_cols(Xl))
        for f in range(folds):
            gsel = np.where(grp == f)[0]
            keep = grp != f if folds > 1 else np.ones(G, dtype=bool)
            if keep.sum() < 2:
                continue
            tasks.append((Xl, y_all, off_all, keep, gsel, K, lam, seed, source, ref_task, float(cfg.within_label_prior)))
            where.append((rows, gsel))
    n_jobs = int(cfg.n_jobs)
    if n_jobs > 1 and len(tasks) > 1:
        from joblib import Parallel, delayed
        mults = Parallel(n_jobs=min(n_jobs, len(tasks)), prefer="processes")(delayed(_within_fit)(*t_) for t_ in tasks)
    else:
        mults = [_within_fit(*t_) for t_ in tasks]
    for (rows, gsel), mult in zip(where, mults):
        out[np.ix_(rows, gsel)] = (expected[np.ix_(rows, gsel)] * mult).astype(np.float32)
    return out


def _neighbour_rate_term(st_csr, sf, coords, labels, cfg):
    """Each spot's size factor times the distance-weighted mean rate (count / size factor) of its
    k nearest spots, zeros included: a neighbour term on the same footing as the expected count."""
    A, _ = spatial_graph(np.asarray(coords, float), labels, cfg.with_(spatial_mode="weighted", type_weight="none", radius_mult=0.0))
    rates = sp.diags((1.0 / sf).astype(np.float32)) @ st_csr
    nb = np.asarray((A @ rates).todense(), dtype=np.float32)
    return (sf[:, None].astype(np.float32) * nb)


def _expected_matrix(st_csr, labels, cfg, coords=None, probs=None, parents=None, gene_names=None, seed=0, ref=None):
    """The expected count of every entry under the reference-free model, with the chosen options.
    Returns (expected, details)."""
    st = _dense(st_csr, np.float32)
    labels = np.asarray(labels).astype(str)
    details = {}
    # memberships
    if cfg.soft_labels == "none":
        names, inv = np.unique(labels, return_inverse=True)
        P = sp.csr_matrix((np.ones(labels.size), (np.arange(labels.size), inv)), shape=(labels.size, names.size))
        hard = labels
    elif cfg.soft_labels == "kmeans":
        Pd, names = _soft_kmeans_memberships(st_csr, cfg.soft_k, cfg.soft_temp, seed)
        P = sp.csr_matrix(Pd); hard = names[Pd.argmax(axis=1)]
    elif cfg.soft_labels == "given":
        if probs is None:
            raise ValueError("soft_labels='given' needs a (matrix, names) pair of label probabilities")
        Pd, names = probs; Pd = np.asarray(Pd, dtype=np.float64); names = np.asarray(names).astype(str)
        Pd = np.where(np.isfinite(Pd), Pd, 0.0); rs = Pd.sum(axis=1, keepdims=True)
        Pd = np.where(rs > 0, Pd / np.where(rs > 0, rs, 1.0), 1.0 / Pd.shape[1])
        P = sp.csr_matrix(Pd); hard = names[Pd.argmax(axis=1)]
    else:
        raise ValueError(f"unknown soft_labels {cfg.soft_labels}")
    # size factors
    if cfg.sf_mode == "nomito" and gene_names is not None:
        keep = np.array([not _NUISANCE.match(str(g)) for g in gene_names])
        details["sf_genes"] = int(keep.sum()); sf = _size_factors(st, np.where(keep)[0])
    else:
        sf = _size_factors(st)
    k = float(cfg.rate_shrink)
    gene_rate = st.sum(axis=0, dtype=np.float64) / sf.sum()
    if cfg.sf_mode == "stable":
        # genes whose rate varies least across labels, among the better-expressed half, carry the depth
        rate0 = _label_rates(st_csr, P, sf, k, gene_rate[None, :])
        w_t = np.asarray(P.sum(axis=0)).ravel(); w_t = w_t / w_t.sum()
        mean_r = (w_t[:, None] * rate0).sum(axis=0); var_r = (w_t[:, None] * (rate0 - mean_r) ** 2).sum(axis=0)
        cv = np.sqrt(var_r) / np.maximum(mean_r, 1e-12)
        ok = gene_rate >= np.median(gene_rate[gene_rate > 0])
        n_keep = max(20, int(0.3 * ok.sum()))
        stable = np.argsort(np.where(ok, cv, np.inf))[:n_keep]
        details["sf_genes"] = int(n_keep); sf = _size_factors(st, stable)
        gene_rate = st.sum(axis=0, dtype=np.float64) / sf.sum()
    # prior for the shrinkage
    if cfg.rate_shrink_target == "parent" and parents is not None:
        par = np.array([str(parents.get(str(t), str(t))) for t in names])
        pu, pinv = np.unique(par, return_inverse=True)
        Pp = P @ sp.csr_matrix((np.ones(names.size), (np.arange(names.size), pinv)), shape=(names.size, pu.size))
        prior = _label_rates(st_csr, Pp, sf, k, gene_rate[None, :])[pinv]       # one row per label: its parent's rate
    else:
        prior = gene_rate[None, :]
    rate = _label_rates(st_csr, P, sf, k, prior)
    expected = (sf[:, None] * np.asarray(P @ rate)).astype(np.float32)
    if cfg.within_label_pcs > 0:
        expected = _within_label_correction(st_csr, expected, hard, cfg, seed, ref=ref)
    details.update(n_labels=int(names.size), soft=cfg.soft_labels, sf_mode=cfg.sf_mode)
    return expected, sf, hard, details


def _tune_score_fn(kind, y, gene_of, n_genes):
    """Score of a prediction m for the held-out molecules y at thinned zeros (lower is better):
    mean Poisson deviance, or minus the per-gene Spearman correlation averaged over genes with
    weights equal to their number of held-out molecules."""
    if kind == "deviance":
        ylog = np.where(y > 0, y * np.log(np.maximum(y, 1e-12)), 0.0)
        def f(m):
            return float(np.mean(2.0 * (ylog - np.where(y > 0, y * np.log(m), 0.0) - (y - m))))
        return f
    if kind == "rank":
        from scipy.stats import rankdata
        order = np.argsort(gene_of, kind="stable"); g_sorted = gene_of[order]
        bounds = np.flatnonzero(np.diff(g_sorted)) + 1
        groups = [grp for grp in np.split(order, bounds) if (y[grp] > 0).sum() >= 5 and (y[grp] == 0).sum() >= 5]
        ry = [rankdata(y[grp]) for grp in groups]
        wts = np.array([(y[grp] > 0).sum() for grp in groups], dtype=float)
        def f(m):
            if not groups:
                return 0.0
            cors = np.empty(len(groups))
            for i, grp in enumerate(groups):
                rm = rankdata(m[grp]); a = rm - rm.mean(); b = ry[i] - ry[i].mean()
                den = np.sqrt((a * a).sum() * (b * b).sum())
                cors[i] = (a * b).sum() / den if den > 0 else 0.0
            return -float((cors * wts).sum() / wts.sum())
        return f
    raise ValueError(f"unknown tune_score {kind}")


def impute_depth_type(st_X, st_labels, cfg: Config = None, coords=None, probs=None, parents=None, gene_names=None,
                      return_details: bool = False, sc_X=None, sc_labels=None):
    """Reference-free expected counts: every zero gets the spot's size factor times the
    section's own rate for the spot's label,

        rate[t, g] = (counts of g in spots of type t + k * prior) / (size factors of type t + k),

    where the prior is the gene's rate over the whole section (or the label's parent class) and
    k = ``cfg.rate_shrink``.  Options: soft memberships instead of one label per spot
    (``soft_labels``), size factors from a stable set of genes (``sf_mode``), a within-label
    correction from the spot's own measured genes (``within_label_pcs``, cross-fitted with
    ``within_label_xfit``, or chosen by thinning with ``within_label_candidates``; its components
    and coefficients can be learned on the reference cells of the label instead of the section, or
    the reference can serve as the prior of the section's fit, ``within_label_source``), and a neighbour
    rate term blended with a fixed or self-tuned weight (``spatial_blend``); the thinning step scores
    candidates by ``tune_score``.  ``quota_mode="none"``
    fills every zero; ``quota_mode="expected"`` fills a zero only where the chance of at least one
    molecule at that depth, 1 - exp(-expected count), reaches ``cfg.fill_threshold``.
    Any labelling of the spots can be used; no reference is read."""
    cfg = cfg or PRESETS["depth_type"]
    if cfg.quota_mode not in ("none", "expected"):
        raise ValueError("the reference-free estimate supports quota_mode 'none' or 'expected'")
    st_csr = _csr(st_X).astype(np.float32)
    st = _dense(st_csr, np.float32)
    if cfg.relabel == "kmeans":
        from sklearn.cluster import MiniBatchKMeans
        k_ = int(cfg.relabel_k) or int(np.unique(np.asarray(st_labels).astype(str)).size)
        Zk = TruncatedSVD(n_components=min(30, st_csr.shape[1] - 1), random_state=cfg.seed).fit_transform(_normalise_log(st_csr))
        st_labels = np.array([f"k{c}" for c in MiniBatchKMeans(n_clusters=k_, n_init=3, batch_size=4096, random_state=cfg.seed).fit_predict(Zk)])
    elif cfg.relabel != "none":
        raise ValueError(f"unknown relabel {cfg.relabel}")
    ref = None
    if cfg.within_label_source != "section" and sc_X is not None and sc_labels is not None:
        sc_csr = _csr(sc_X).astype(np.float32)
        if sc_csr.shape[1] != st_csr.shape[1]:
            raise ValueError("within_label_source needs the reference on the same genes as the section")
        if (detect_ref_space(sc_csr) if cfg.ref_space == "auto" else cfg.ref_space) == "log1p":
            sc_csr = sc_csr.copy(); sc_csr.data = np.expm1(sc_csr.data)
        ref = (sc_csr, np.asarray(sc_labels).astype(str))
    cands = [int(x) for x in str(cfg.within_label_candidates).split(",") if x.strip() != ""] or [int(cfg.within_label_pcs)]
    if cfg.spatial_blend not in ("none", "fixed", "auto"):
        raise ValueError(f"unknown spatial_blend {cfg.spatial_blend}")
    if cfg.spatial_blend != "none" and coords is None:
        raise ValueError("spatial_blend needs coordinates")
    details = {}
    K, w = cands[0], (float(cfg.spatial_w) if cfg.spatial_blend == "fixed" else 0.0)
    if cfg.spatial_blend == "auto" or len(cands) > 1:
        # thin the section once; pick the number of within-label components and the neighbour weight
        # whose estimate best predicts the held-out molecules at the thinned zeros
        rng = np.random.default_rng(cfg.seed)
        thin = st_csr.copy(); thin.data = rng.binomial(thin.data.astype(np.int64), float(cfg.tune_keep)).astype(np.float32); thin.eliminate_zeros()
        held = st - _dense(thin, np.float32)
        zero = _dense(thin, np.float32) == 0
        idx = np.flatnonzero(zero.ravel())
        cap = 5_000_000 if cfg.tune_score == "rank" else 2_000_000
        if idx.size > cap:
            idx = np.sort(rng.choice(idx, cap, replace=False))
        y = held.ravel()[idx].astype(np.float64)
        score = _tune_score_fn(cfg.tune_score, y, idx % st.shape[1], st.shape[1])
        grid = np.linspace(0.0, 1.0, 21) if cfg.spatial_blend == "auto" else np.array([w])
        best = np.inf; b_ = None
        for Kc in cands:
            e_t, sf_t, hard_t, _ = _expected_matrix(thin, st_labels, cfg.with_(within_label_pcs=Kc), coords, probs, parents, gene_names, cfg.seed, ref=ref)
            a_ = e_t.ravel()[idx].astype(np.float64)
            if b_ is None and cfg.spatial_blend != "none":
                b_ = _neighbour_rate_term(thin, sf_t, coords, hard_t, cfg).ravel()[idx].astype(np.float64)
            for cand in grid:
                f_ = (1.0 - float(cfg.tune_keep)) / float(cfg.tune_keep)    # kept-half expectation -> held-out expectation
                m = np.maximum(f_ * ((1 - cand) * a_ + cand * b_), 1e-8) if b_ is not None else np.maximum(f_ * a_, 1e-8)
                s = score(m)
                if s < best:
                    best, K, w = s, int(Kc), float(cand)
        details["within_label_pcs_auto"] = K
        if cfg.spatial_blend == "auto":
            details["spatial_w_auto"] = w
    expected, sf, hard, det = _expected_matrix(st_csr, st_labels, cfg.with_(within_label_pcs=K), coords, probs, parents, gene_names, cfg.seed, ref=ref)
    details["within_label_source"] = cfg.within_label_source if ref is not None else "section"
    details.update(det)
    if cfg.spatial_blend != "none" and w > 0:
        nb = _neighbour_rate_term(st_csr, sf, coords, hard, cfg)
        expected = ((1.0 - w) * expected + w * nb).astype(np.float32)
    if cfg.spatial_blend != "none":
        details["spatial_w"] = w
    fill = st == 0
    if cfg.prob_model == "nb":
        phi = _nb_dispersion(st_csr, sf, hard)
        prob = 1.0 - (1.0 + phi[None, :] * expected.astype(np.float64)) ** (-1.0 / phi[None, :])
        details["nb_dispersion_median"] = float(np.median(phi))
    else:
        prob = 1.0 - np.exp(-expected.astype(np.float64))
    if cfg.quota_mode == "expected":
        fill &= prob >= cfg.fill_threshold
    out = np.where(fill, expected, st).astype(np.float32)
    if return_details:
        details.update(n_filled=int(fill.sum()), prob_molecule_mean_filled=float(prob[fill].mean()) if fill.any() else float("nan"))
        return out, details, expected, prob.astype(np.float32)
    return out


def impute(sc_X, sc_labels, st_X, st_labels, coords, cfg: Config = PRESETS["corrected"],
           return_details: bool = False, embed_X=None, aux=None):
    """Impute zeros of ``st_X`` (spots x genes) from the reference ``sc_X``
    (cells x genes) and the spatial neighbourhood.  Returns a float32 dense
    matrix; with ``return_details`` also a dict of per-spot quantities.

    ``embed_X`` = (sc_matrix, st_matrix) over a wider gene set (same rows) to
    build the conditional embedding from, so a gene panel can be scored while
    the similarity between spots and reference cells uses every gene."""
    sc_labels = np.asarray(sc_labels).astype(str)
    st_labels = np.asarray(st_labels).astype(str)
    coords = np.asarray(coords, dtype=np.float64)

    if cfg.ref_mode == "none":
        aux = aux or {}
        res = impute_depth_type(st_X, st_labels, cfg, coords=coords, probs=aux.get("probs"), parents=aux.get("parents"),
                                gene_names=aux.get("gene_names"), return_details=return_details,
                                sc_X=sc_X if cfg.within_label_source != "section" else None, sc_labels=sc_labels)
        if return_details:
            out, det, _, _ = res
            return out, dict(det, ref_space="unused")
        return res

    # ----- reference in a declared linear space -----
    sc_csr = _csr(sc_X).astype(np.float32)
    space = cfg.ref_space
    if space == "auto":
        space = detect_ref_space(sc_csr)
    if space == "log1p":
        sc_csr = sc_csr.copy()
        sc_csr.data = np.expm1(sc_csr.data)
    elif space not in ("linear", "as_is"):
        raise ValueError(f"unknown ref_space {cfg.ref_space}")
    if cfg.kde_space == "exp" and cfg.ref_space != "as_is":
        raise ValueError("kde_space='exp' is the legacy behaviour and needs ref_space='as_is'")
    if cfg.kde_space == "exp" and detect_ref_space(sc_csr) != "log1p":
        raise ValueError("kde_space='exp' (the published notebook) only makes sense on a log1p-stored reference; "
                         "this reference holds linear counts, so the legacy exp() would overflow. "
                         "Use the 'corrected' preset or supply a log1p reference.")

    # ----- ST matrix -----
    st_csr = _csr(st_X).astype(np.float32)
    st_dense = _dense(st_csr, np.float32)
    n_st, n_genes = st_dense.shape
    out = st_dense.copy()
    lib = np.asarray(st_csr.sum(axis=1)).ravel().astype(np.float64)
    size_factor = lib / max(float(np.median(lib[lib > 0])) if (lib > 0).any() else 1.0, 1e-9)
    size_factor[size_factor == 0] = 1e-3

    # ----- spatial structure (once) -----
    A, h = spatial_graph(coords, st_labels, cfg)
    w_all = weights_from_h(h, cfg)
    if A is not None:
        spatial_val = np.asarray((A @ st_csr).todense(), dtype=np.float32)
        det_csr = st_csr.copy(); det_csr.data[:] = 1.0
        spatial_det = np.asarray((A @ det_csr).todense(), dtype=np.float32)
        if cfg.spatial_nonzero_only:
            with np.errstate(divide="ignore", invalid="ignore"):
                spatial_val = np.where(spatial_det > 0, spatial_val / np.maximum(spatial_det, 1e-12), 0.0)
            spatial_val = spatial_val.astype(np.float32)
    else:
        spatial_val = None
        spatial_det = None

    # ----- conditional embedding (once) -----
    if cfg.ref_mode == "conditional":
        if embed_X is not None:
            sc_e = _csr(embed_X[0]).astype(np.float32)
            if space == "log1p":
                sc_e = sc_e.copy(); sc_e.data = np.expm1(sc_e.data)
            st_e = _csr(embed_X[1]).astype(np.float32)
        else:
            sc_e, st_e = sc_csr, st_csr
        if cfg.ref_embed == "pca":
            E, F = _joint_embedding(sc_e, st_e, cfg.ref_pca, cfg.seed)
        elif cfg.ref_embed == "markers":
            sc_ln, st_ln = _normalise_log(sc_e), _normalise_log(st_e)
            markers = _cluster_markers(sc_ln, sc_labels, cfg.marker_n)
        else:
            raise ValueError(f"unknown ref_embed {cfg.ref_embed}")
    elif cfg.ref_mode != "marginal":
        raise ValueError(f"unknown ref_mode {cfg.ref_mode}")

    def _type_embedding(ctype, sc_rows, st_rows):
        """Rows of the similarity space for one cell type: the global SVD, or the type's marker genes."""
        if cfg.ref_embed == "pca":
            return E[sc_rows], F[st_rows]
        M = markers[ctype]
        return _unit_rows(sc_ln[sc_rows][:, M].toarray()), _unit_rows(st_ln[st_rows][:, M].toarray())

    types = [t for t in np.unique(sc_labels) if (st_labels == t).any()]
    gene_rate_all = st_dense.sum(axis=0, dtype=np.float64) / max(float(size_factor.sum()), 1e-12)   # for rate_shrink
    chosen_records = []          # (row_idx, gene, ref_value, w) for a second pass
    n_filled = 0

    for ti, ctype in enumerate(types):
        sc_rows = np.where(sc_labels == ctype)[0]
        st_rows = np.where(st_labels == ctype)[0]
        R = sc_csr[sc_rows]                                  # sparse cells x genes
        R_det = R.copy(); R_det.data[:] = 1.0
        n_ref = R.shape[0]
        n_sp = st_rows.shape[0]
        cnt = np.asarray(R_det.sum(axis=0)).ravel()
        sums = np.asarray(R.sum(axis=0)).ravel()
        with np.errstate(invalid="ignore", divide="ignore"):
            sc_mean = np.where(cnt > 0, sums / np.maximum(cnt, 1), 0.0)
        sc_det_frac = cnt / n_ref                            # P(expressed | type)
        st_sub = st_dense[st_rows]
        st_nz = st_sub > 0
        st_cnt = st_nz.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            st_mean = np.where(st_cnt > 0, st_sub.sum(axis=0) / np.maximum(st_cnt, 1), 0.0)
        w_c = w_all[st_rows]
        sf_c = size_factor[st_rows]

        if cfg.ref_mode == "conditional":
            E_c, F_c = _type_embedding(ctype, sc_rows, st_rows)
            W, nb_idx, nb_w = _ref_neighbours(E_c, F_c, cfg)

        for g0 in range(0, n_genes, cfg.gene_chunk):
            g1 = min(n_genes, g0 + cfg.gene_chunk)
            rng = np.random.default_rng([cfg.seed, ti, g0])
            R_chunk = R[:, g0:g1]
            R_chunk_d = R_chunk.toarray()                    # cells x chunk (per type: small)
            if cfg.ref_mode == "conditional":
                cond_sum = np.asarray((W @ R_chunk).todense())          # weighted sum
                cond_det = np.asarray((W @ R_det[:, g0:g1]).todense())  # weighted detection
                with np.errstate(invalid="ignore", divide="ignore"):
                    cond_mean = np.where(cond_det > 0, cond_sum / np.maximum(cond_det, 1e-12), 0.0)

            for gl in range(g1 - g0):
                g = g0 + gl
                if sc_mean[g] <= 0:
                    continue
                zloc = np.where(~st_nz[:, g])[0]
                if zloc.size == 0:
                    continue
                nz_vals = R_chunk_d[:, gl][R_chunk_d[:, gl] > 0]
                if nz_vals.size == 0:
                    continue

                st_mean_adj = max(float(st_mean[g]), cfg.min_spatial_mean_ratio * float(sc_mean[g]))
                scale = st_mean_adj / float(sc_mean[g])
                r_disp = _nb_moments(nz_vals)[1] if (cfg.ref_dist == "nb" or cfg.ref_estimate == "posterior") else None

                # ---- expected expressed-level per zero spot (reference scale * scale) ----
                if cfg.ref_mode == "conditional":
                    mu = cond_mean[zloc, gl] * scale
                    pi = cond_det[zloc, gl]
                    miss = pi <= 0
                    mu = np.where(miss, st_mean_adj, mu)
                    pi = np.where(miss, sc_det_frac[g], pi)
                else:
                    mu = np.full(zloc.size, st_mean_adj)
                    pi = np.full(zloc.size, sc_det_frac[g])

                # ---- expected count per unit exposure (ref_estimate="expected") ----
                # the untruncated neighbour mean (zeros included) moved to the ST rate scale of this type;
                # multiplied by the spot's size factor below it is a plain regression estimate of the count
                rate = None
                if cfg.ref_estimate == "expected" or cfg.zero_select == "expected" or cfg.quota_mode == "expected":
                    ref_rate = float(sums[g]) / n_ref                                   # type mean incl. zeros
                    st_rate = ((float(st_sub[:, g].sum()) + cfg.rate_shrink * float(gene_rate_all[g]))
                               / max(float(sf_c.sum()) + cfg.rate_shrink, 1e-12))       # counts per unit exposure
                    if cfg.ref_mode == "conditional" and ref_rate > 0:
                        rate = np.where(miss, st_rate, cond_sum[zloc, gl] * (st_rate / ref_rate))
                    else:
                        rate = np.full(zloc.size, st_rate)

                # ---- dropout posterior under a Poisson depth model ----
                # mu is a mean over expressing (non-zero) observations, i.e. a zero-truncated mean;
                # the untruncated Poisson rate solves lam / (1 - exp(-lam)) = mu
                lam = _untruncate_rate(mu)
                p0 = np.exp(-lam * sf_c[zloc])
                post = pi * p0 / (pi * p0 + (1.0 - pi) + 1e-12)

                # ---- evidence score for ranking zeros ----
                if cfg.zero_select == "random":
                    score = rng.random(zloc.size)
                elif cfg.zero_select == "evidence":
                    if spatial_val is None:
                        score = rng.random(zloc.size)
                    else:
                        base = spatial_det if cfg.spatial_nonzero_only else spatial_val
                        score = base[st_rows[zloc], g].astype(np.float64)
                elif cfg.zero_select == "ref_evidence":
                    score = pi.astype(np.float64)
                elif cfg.zero_select == "combined":
                    s1 = pi.astype(np.float64)
                    if spatial_val is None:
                        score = s1
                    else:
                        base = spatial_det if cfg.spatial_nonzero_only else spatial_val
                        s2 = base[st_rows[zloc], g].astype(np.float64)
                        # average ranks so that tied evidence (very common for detection fractions)
                        # is not ordered by row position
                        score = rankdata(s1, method="average") + rankdata(s2, method="average")
                elif cfg.zero_select == "detect_prob":
                    # probability that a non-zero would be observed here: pi * (1 - p0)
                    score = pi * (1.0 - p0)
                elif cfg.zero_select == "expected":
                    # expected count at this spot's depth: the zeros most likely to hide a molecule come first
                    score = sf_c[zloc] * rate
                elif cfg.zero_select == "posterior":
                    score = post
                else:
                    raise ValueError(f"unknown zero_select {cfg.zero_select}")
                score = score + rng.random(zloc.size) * 1e-9      # deterministic tie-break

                # ---- how many zeros to fill ----
                if cfg.quota_mode == "reference":
                    target_zeros = int(round((1.0 - sc_det_frac[g]) * n_sp))
                    n_fill = max(0, n_sp - target_zeros - int(st_cnt[g]))
                    n_fill = min(n_fill, zloc.size)
                elif cfg.quota_mode == "depth":
                    n_fill = int(round(float(post.sum())))
                elif cfg.quota_mode == "expected":
                    # fill a zero when the chance of at least one molecule at this depth, 1 - exp(-expected count),
                    # reaches the threshold (pair with zero_select="expected" so that exactly those zeros are taken)
                    n_fill = int(((1.0 - np.exp(-sf_c[zloc] * rate)) >= cfg.fill_threshold).sum())
                elif cfg.quota_mode == "none":
                    # fill every zero whose dropout posterior reaches the threshold, ranked by the chosen score
                    n_fill = int((post >= cfg.fill_threshold).sum())
                else:
                    raise ValueError(f"unknown quota_mode {cfg.quota_mode}")
                if n_fill <= 0:
                    continue
                order = np.argsort(-score)[:n_fill]
                sel = zloc[order]                                      # local rows (type subset)

                # ---- reference value for the selected zeros ----
                mu_sel = mu[order]
                if cfg.ref_estimate == "mean":
                    ref_val = mu_sel
                elif cfg.ref_estimate == "expected":
                    ref_val = sf_c[sel] * rate[order]
                elif cfg.ref_estimate == "posterior":
                    # Gamma-Poisson posterior mean of the spot's expected count: exposure * E[rate | y=0]
                    ref_val = sf_c[sel] * mu_sel * r_disp / (r_disp + mu_sel * sf_c[sel])
                elif cfg.ref_estimate == "draw":
                    if cfg.ref_dist == "nb":
                        ref_val = _draw_nb(mu_sel * sf_c[sel], r_disp, rng)
                    elif cfg.ref_mode == "conditional":
                        lv = np.log1p(nz_vals)
                        bw = max(1e-3, _silverman_bw(lv))
                        nbv = R_chunk_d[nb_idx[sel], gl]                    # (n_sel, k)
                        ref_val = _draw_conditional(nbv, nb_w[sel], scale, rng, bw)
                        nan = np.isnan(ref_val)
                        if nan.any():
                            ref_val[nan] = _draw_marginal_kde(nz_vals, int(nan.sum()), scale, rng, "log")
                    else:
                        ref_val = _draw_marginal_kde(nz_vals, sel.size, scale, rng, cfg.kde_space)
                else:
                    raise ValueError(f"unknown ref_estimate {cfg.ref_estimate}")

                # ---- blend with the spatial term ----
                w_sel = w_c[sel]
                if spatial_val is not None:
                    spat = spatial_val[st_rows[sel], g].astype(np.float64)
                    blended = w_sel * ref_val + (1.0 - w_sel) * spat
                else:
                    blended = ref_val
                blended = np.maximum(0.0, np.nan_to_num(blended, nan=0.0, posinf=0.0, neginf=0.0))
                glob = st_rows[sel]
                with np.errstate(over="ignore"):
                    out[glob, g] = blended.astype(np.float32)   # legacy exp() may overflow to inf, zeroed at the end
                n_filled += sel.size
                if cfg.passes > 1:
                    chosen_records.append((glob, g, ref_val.astype(np.float32), w_sel.astype(np.float32)))

                # ---- optional denoising of observed non-zeros ----
                if cfg.denoise_all and cfg.ref_estimate == "posterior":
                    nzl = np.where(st_nz[:, g])[0]
                    y = st_sub[nzl, g].astype(np.float64)
                    mu_nz = np.full(nzl.size, st_mean_adj)
                    if cfg.ref_mode == "conditional":
                        cm = cond_mean[nzl, gl] * scale
                        mu_nz = np.where(cond_det[nzl, gl] > 0, cm, mu_nz)
                    s_nz = sf_c[nzl]
                    post_mean = s_nz * (r_disp + y) / (r_disp / np.maximum(mu_nz, 1e-9) + s_nz)
                    out[st_rows[nzl], g] = post_mean.astype(np.float32)

    # ----- optional second pass: re-blend filled entries with the imputed neighbourhood -----
    if cfg.passes > 1 and A is not None and chosen_records:
        out_csr = sp.csr_matrix(out)
        sv2 = np.asarray((A @ out_csr).todense(), dtype=np.float32)
        if cfg.spatial_nonzero_only:
            d2 = out_csr.copy(); d2.data[:] = 1.0
            det2 = np.asarray((A @ d2).todense(), dtype=np.float32)
            with np.errstate(divide="ignore", invalid="ignore"):
                sv2 = np.where(det2 > 0, sv2 / np.maximum(det2, 1e-12), 0.0).astype(np.float32)
        for glob, g, ref_val, w_sel in chosen_records:
            new = w_sel * ref_val + (1.0 - w_sel) * sv2[glob, g]
            new = np.maximum(0.0, new)
            out[glob, g] = (cfg.damping * new + (1.0 - cfg.damping) * out[glob, g]).astype(np.float32)

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if not return_details:
        return out
    details = dict(h=h, w=w_all, size_factor=size_factor, ref_space=space,
                   n_filled=int(n_filled), config=cfg.to_dict())
    return out, details


# --------------------------------------------------------------------------- #
#  Prediction of genes absent from the ST panel (leave-gene-out protocol)
# --------------------------------------------------------------------------- #

def _fine_clusters(sc_lin_shared: sp.csr_matrix, sc_labels, n_clusters: int, seed: int):
    """Clusters of the reference on the shared genes: k-means on the top principal components of the
    log-normalised matrix, or the reference's own labels when ``n_clusters`` is 0."""
    if n_clusters <= 0:
        lab = np.asarray(sc_labels).astype(str); names, inv = np.unique(lab, return_inverse=True)
        return inv, names.size
    from sklearn.cluster import MiniBatchKMeans
    Xn = _normalise_log(sc_lin_shared)
    Z = TruncatedSVD(n_components=min(30, Xn.shape[1] - 1), random_state=seed).fit_transform(Xn)
    k = int(min(n_clusters, Z.shape[0] // 20 if Z.shape[0] >= 40 else 2))
    return MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=4096, random_state=seed).fit_predict(Z), k


def _cluster_signatures(sc_lin: sp.csr_matrix, cl: np.ndarray, C: int, sf_ref: np.ndarray, k_shrink: float = 20.0):
    """Rates (counts per unit size factor) of every gene in every cluster, shrunk toward the global rate."""
    P = sp.csr_matrix((np.ones(cl.size), (np.arange(cl.size), cl)), shape=(cl.size, C))
    gene_rate = np.asarray(sc_lin.sum(axis=0)).ravel() / max(float(sf_ref.sum()), 1e-9)
    return np.asarray(_label_rates(sc_lin, P, sf_ref, k_shrink, gene_rate[None, :]), dtype=np.float64)   # (C x G)


def _soft_composition(st_csr: sp.csr_matrix, S_shared: np.ndarray, iters: int = 60):
    """Per-spot weights over clusters: Poisson regression of the spot's shared-gene counts on the cluster
    rates (the depth is the spot's own library size), by multiplicative updates on the simplex."""
    Y = _dense(st_csr, np.float64)                       # (n x G)
    lib = Y.sum(axis=1); sf = lib / max(float(np.median(lib[lib > 0])) if (lib > 0).any() else 1.0, 1e-9); sf[sf == 0] = 1e-3
    S = np.maximum(S_shared, 1e-9)                        # (C x G)
    C = S.shape[0]
    W = np.full((Y.shape[0], C), 1.0 / C)
    colsum = S.sum(axis=1)                                # (C,)
    for _ in range(iters):
        lam = sf[:, None] * (W @ S)                       # (n x G)
        W = W * ((Y / np.maximum(lam, 1e-12)) @ S.T) / np.maximum(sf[:, None] * colsum[None, :], 1e-12)
        W = W + 1e-4
        W /= W.sum(axis=1, keepdims=True)
    return W


def _within_cluster_multiplier(sc_ln_std_shared: np.ndarray, sc_target_counts: np.ndarray, sf_ref: np.ndarray, rate_target: np.ndarray,
                               st_ln_std_shared: np.ndarray, K: int, lam: float, seed: int):
    """Learn on the reference cells of one cluster how each target gene moves along K within-cluster
    components of the shared genes (Poisson regression with the cluster rate as offset); return the
    multiplier for every spot from the spot's own position on those components."""
    k = int(min(K, sc_ln_std_shared.shape[1] - 1, sc_ln_std_shared.shape[0] - 1))
    if k < 1:
        return np.ones((st_ln_std_shared.shape[0], sc_target_counts.shape[1]), dtype=np.float32)
    svd = TruncatedSVD(n_components=k, random_state=seed).fit(sc_ln_std_shared)
    Zr = _standardise_cols(svd.transform(sc_ln_std_shared))
    Zs = _standardise_cols(st_ln_std_shared @ svd.components_.T)
    off = np.log(np.maximum(sf_ref[:, None] * rate_target[None, :], 1e-8))
    beta = _poisson_ridge_fit(Zr, sc_target_counts, off, lam)
    return np.exp(np.clip(Zs @ beta.T, -np.log(5.0), np.log(5.0))).astype(np.float32)


def _predict_core(sc_shared, sc_target, sc_labels, st_csr, st_labels, cfg: Config, seed: int):
    """Prediction of the target genes at every spot, at a typical reference depth, before the platform
    scale and the spatial smoothing.  Label backbone: per type, the similar-cell average (or the type
    mean, or a within-type regression).  Soft backbone: per-spot composition over reference clusters
    times the cluster means, optionally times a within-cluster regression factor."""
    sc_labels = np.asarray(sc_labels).astype(str); st_labels = np.asarray(st_labels).astype(str)
    n_st = st_csr.shape[0]; G_t = sc_target.shape[1]
    pred = np.zeros((n_st, G_t), dtype=np.float32)
    lib_ref = np.asarray(sc_shared.sum(axis=1)).ravel()
    sf_ref = lib_ref / max(float(np.median(lib_ref[lib_ref > 0])) if (lib_ref > 0).any() else 1.0, 1e-9); sf_ref[sf_ref == 0] = 1e-3
    K = int(cfg.predict_within); lam = float(cfg.within_label_ridge)
    if K > 0:
        sc_ln = _normalise_log(sc_shared).toarray(); st_ln = _normalise_log(st_csr).toarray()
    if cfg.predict_backbone == "soft":
        cl, C = _fine_clusters(sc_shared, sc_labels, int(cfg.predict_clusters), seed)
        S_shared = _cluster_signatures(sc_shared, cl, C, sf_ref)
        W = _soft_composition(st_csr, S_shared)                                   # (n x C)
        T_dense = _dense(sc_target, np.float64)
        mean_t = np.stack([T_dense[cl == c].mean(axis=0) if (cl == c).any() else T_dense.mean(axis=0) for c in range(C)])   # (C x G_t)
        if K == 0:
            pred = (W @ mean_t).astype(np.float32)
            if cfg.predict_detail == "knn":
                # within-type detail from the label backbone: the similar-cell value relative to the type mean
                knn = _predict_core(sc_shared, sc_target, sc_labels, st_csr, st_labels, cfg.with_(predict_backbone="label", predict_detail="none", predict_within=0, predict_blend=0.0), seed)
                tm = _predict_core(sc_shared, sc_target, sc_labels, st_csr, st_labels, cfg.with_(predict_backbone="label", predict_detail="none", predict_within=0, ref_mode="marginal"), seed)
                ratio = np.clip(knn / np.maximum(tm, 1e-6), 0.2, 5.0)
                pred = (pred * ratio).astype(np.float32)
            return pred
        rate_t = _cluster_signatures(sc_target, cl, C, sf_ref)
        for c in range(C):
            rows = np.where(cl == c)[0]
            if rows.size < max(50, 10 * K):
                pred += (W[:, c:c + 1] * mean_t[c][None, :]).astype(np.float32); continue
            mu = sc_ln[rows].mean(axis=0); sd = sc_ln[rows].std(axis=0) + 1e-9
            mult = _within_cluster_multiplier((sc_ln[rows] - mu) / sd, T_dense[rows], sf_ref[rows], rate_t[c], (st_ln - mu) / sd, K, lam, seed)
            pred += (W[:, c:c + 1] * mean_t[c][None, :] * mult).astype(np.float32)
        return pred
    # label backbone
    if cfg.ref_mode == "conditional" and K == 0:
        E, F = _joint_embedding(sc_shared, st_csr, cfg.ref_pca, seed)
    types = [ty for ty in np.unique(sc_labels) if (st_labels == ty).any()]
    for ctype in types:
        sc_rows = np.where(sc_labels == ctype)[0]; st_rows = np.where(st_labels == ctype)[0]
        T = sc_target[sc_rows]
        marginal = np.repeat(np.asarray(T.mean(axis=0)), st_rows.size, axis=0).astype(np.float32)
        if K > 0 and sc_rows.size >= max(50, 10 * K):
            mu = sc_ln[sc_rows].mean(axis=0); sd = sc_ln[sc_rows].std(axis=0) + 1e-9
            rate_t = np.asarray(T.sum(axis=0)).ravel() / max(float(sf_ref[sc_rows].sum()), 1e-9)
            mult = _within_cluster_multiplier((sc_ln[sc_rows] - mu) / sd, _dense(T, np.float64), sf_ref[sc_rows], rate_t, (st_ln[st_rows] - mu) / sd, K, lam, seed)
            ref = marginal * mult
        elif cfg.ref_mode == "conditional" and K == 0:
            Wn, _, _ = _ref_neighbours(E[sc_rows], F[st_rows], cfg)
            ref = np.asarray((Wn @ T).todense(), dtype=np.float32)
            if cfg.predict_blend > 0:                      # type mean as backbone, similarity as a correction
                ref = (1.0 - cfg.predict_blend) * ref + cfg.predict_blend * marginal
        else:
            ref = marginal
        pred[st_rows] = ref
    missing = ~np.isin(st_labels, types)
    if missing.any():
        pred[missing] = np.asarray(sc_target.mean(axis=0)).astype(np.float32)
    return pred


def _smooth_prediction(pred, A, w_zone, cfg: Config, w_fixed=None):
    if A is None or cfg.predict_smooth == "none":
        return pred
    smooth = np.asarray((A @ sp.csr_matrix(pred)).todense(), dtype=np.float32)
    if cfg.predict_smooth == "zone":
        return (w_zone[:, None] * pred + (1.0 - w_zone[:, None]) * smooth).astype(np.float32)
    w = float(cfg.predict_smooth_w if w_fixed is None else w_fixed)
    return ((1.0 - w) * pred + w * smooth).astype(np.float32)


def _tune_prediction(sc_shared, sc_labels, st_csr, st_labels, A, w_zone, cfg: Config, seed: int):
    """Hold out a share of the shared genes once, predict them from the rest with every candidate
    setting, and return the configuration with the best mean per-gene Spearman on the held-out genes."""
    from scipy.stats import rankdata
    rng = np.random.default_rng(seed)
    G = sc_shared.shape[1]
    held = np.sort(rng.choice(G, max(3, int(round(cfg.predict_tune_frac * G))), replace=False))
    kept = np.setdiff1d(np.arange(G), held)
    truth = _normalise_log(st_csr).toarray()[:, held]
    ok = truth.std(axis=0) > 0
    if ok.sum() < 2:
        return cfg
    truth_r = np.column_stack([rankdata(truth[:, j]) for j in range(truth.shape[1])])[:, ok]
    k_grid = sorted({0, int(cfg.predict_within)})
    w_grid = list(np.linspace(0.0, 0.9, 10)) if cfg.predict_smooth == "auto" else [None]
    cands = []
    if cfg.predict_backbone == "soft":
        cands += [cfg.with_(predict_clusters=ncl, predict_within=K) for ncl in (10, 20, 40) for K in k_grid]
        cands += [cfg.with_(predict_backbone="label", predict_detail="none", predict_within=K) for K in k_grid]   # the label backbone stays a candidate
    else:
        cands += [cfg.with_(predict_within=K) for K in k_grid]
    best, choice = -np.inf, cfg
    for c in cands:
        if True:
            base = _predict_core(sc_shared[:, kept], sc_shared[:, held], sc_labels, st_csr[:, kept], st_labels, c, seed)
            for w in w_grid:
                p = _smooth_prediction(base, A, w_zone, c if w is None else c.with_(predict_smooth="fixed"), w)[:, ok]
                pr = np.column_stack([rankdata(p[:, j]) for j in range(p.shape[1])])
                a_ = pr - pr.mean(axis=0); b_ = truth_r - truth_r.mean(axis=0)
                cors = (a_ * b_).sum(axis=0) / np.maximum(np.sqrt((a_ * a_).sum(axis=0) * (b_ * b_).sum(axis=0)), 1e-12)
                s = float(np.nanmean(cors))
                if s > best:
                    best, choice = s, (c.with_(predict_smooth="fixed", predict_smooth_w=float(w)) if w is not None else c)
    return choice


def predict_genes(sc_X, sc_labels, st_X, st_labels, coords, target_genes, shared_genes,
                  cfg: Config = PRESETS["v3"], return_details: bool = False):
    """Predict expression of ``target_genes`` (columns of ``sc_X`` absent from
    the ST panel) at every spot.  ``sc_X`` holds all reference genes; its
    columns are indexed by ``shared_genes`` (names present in both, in the
    order of ``st_X``'s columns) and ``target_genes``.  Label backbone: the
    reference term is the spot's conditional neighbour mean within its type (or
    the type mean, or a within-type regression learned on the reference,
    ``predict_within``).  Soft backbone (``predict_backbone="soft"``): a per-spot
    composition over reference clusters fitted on the shared genes, times the
    cluster means, without any spot label.  The result is scaled by the platform
    factor estimated on shared genes and blended with its spatial smoothing by
    the zone weight, a fixed weight, or a weight chosen by holding out shared
    genes (``predict_smooth``, ``predict_tune``).  Returns (n_spots x n_targets)."""
    sc_labels = np.asarray(sc_labels).astype(str)
    st_labels = np.asarray(st_labels).astype(str)
    sc_all = _csr(sc_X).astype(np.float32)
    space = cfg.ref_space if cfg.ref_space != "auto" else detect_ref_space(sc_all)
    if space == "log1p":
        sc_all = sc_all.copy(); sc_all.data = np.expm1(sc_all.data)
    n_shared = len(shared_genes)
    sc_shared = sc_all[:, :n_shared]
    sc_target = sc_all[:, n_shared:]
    st_csr = _csr(st_X).astype(np.float32)
    A, h = spatial_graph(np.asarray(coords, dtype=np.float64), st_labels, cfg)
    w_zone = weights_from_h(h, cfg)
    # platform scale from shared genes: ratio of mean non-zero ST to mean non-zero reference
    st_nzmean = np.asarray(st_csr.sum(axis=0)).ravel() / np.maximum(np.asarray((st_csr > 0).sum(axis=0)).ravel(), 1)
    sc_nzmean = np.asarray(sc_shared.sum(axis=0)).ravel() / np.maximum(np.asarray((sc_shared > 0).sum(axis=0)).ravel(), 1)
    ok = (st_nzmean > 0) & (sc_nzmean > 0)
    scale = float(np.median(st_nzmean[ok] / sc_nzmean[ok])) if ok.any() else 1.0
    details = {}
    if cfg.predict_tune == "genes":
        cfg = _tune_prediction(sc_shared, sc_labels, st_csr, st_labels, A, w_zone, cfg, cfg.seed)
        details.update(backbone=cfg.predict_backbone, clusters=int(cfg.predict_clusters), within=int(cfg.predict_within), smooth=cfg.predict_smooth, smooth_w=float(cfg.predict_smooth_w))
    pred = _predict_core(sc_shared, sc_target, sc_labels, st_csr, st_labels, cfg, cfg.seed) * scale
    pred = _smooth_prediction(pred, A, w_zone, cfg)
    pred = np.maximum(pred, 0.0)
    return (pred, details) if return_details else pred
