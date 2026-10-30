#!/usr/bin/env python3
"""
weight_curve.py -- which blend weight is best at each level of neighbourhood homogeneity?

The method blends a reference term r and a spatial term s as w*r + (1-w)*s, with w
chosen from the spot's homogeneity h.  This script measures, on masked entries of a
real section, what w should have been: both terms are computed for every hidden entry
(engine run with every zero filled, once with w = 1 and once with w = 0), and for a
grid of w the blend is scored against the hidden truth inside each homogeneity bin.

    python scripts/weight_curve.py --dataset kidney --genes top1500 --seed 42 \
        --out results/analysis/weight_curve_kidney.csv

Output: one row per (homogeneity bin, w) with log-RMSE and the entry-weighted per-gene
Spearman, and a printed summary of the best w per bin next to the weight the preset uses.

--protocol p1 hides 20 % of the observed non-zeros.  Every hidden entry is then a true
non-zero, so an error measure there favours any "expressed-level" value.  --protocol p2
thins every count binomially instead and scores the blend against the held-out molecules
at *all* entries that are zero after thinning, true zeros included (a sample of them,
reweighted to their real number), with the mean Poisson deviance and the squared error:
proper scores for an expected count.  With thinning probability p the prediction is
multiplied by (1 - p) / p to sit on the scale of the held-out molecules.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("ZONEIMPUTE_THREADS", "4"))

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from bench import datasets, methods, protocols, runner  # noqa: E402
from zone_impute import PRESETS, impute, impute_depth_type, neighbor_homogeneity, weights_from_h  # noqa: E402


def two_terms(a, ctx, masked, base, embed):
    """Full matrices of the two terms with every zero filled.  With --ref-estimate depth_type the
    reference term is the reference-free depth x label estimate and the spatial term the
    distance-weighted neighbour average with zeros included; no reference is read."""
    out = {}
    if a.ref_estimate == "depth_type":
        out["ref"] = impute_depth_type(masked, ctx.st_labels, PRESETS["depth_type"].with_(rate_shrink=20.0, quota_mode="none"))
        out["spatial"] = methods.control_spatial_avg(masked, ctx.st_labels, ctx.coords)
        return out
    for name, w in (("ref", 1.0), ("spatial", 0.0)):
        cfg = base.with_(weighting="table", weight_table=(np.array([0.0, 1.0000001]), np.array([w])))
        out[name] = impute(ctx.sc_csr, ctx.sc_labels, masked, ctx.st_labels, ctx.coords, cfg, embed_X=embed)
    return out


def _pg_spearman(truth, pred, genes, min_n=30):
    """Entry-weighted mean of per-gene Spearman (genes with a constant prediction count as 0)."""
    order = np.argsort(genes, kind="stable"); g = genes[order]; t = truth[order]; p = pred[order]
    cuts = np.flatnonzero(np.diff(g)) + 1
    num = den = 0.0
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, g.size]):
        n = b - a
        if n < min_n:
            continue
        rt = rankdata(t[a:b]); rp = rankdata(p[a:b])
        st, sp_ = rt.std(), rp.std()
        rho = float(np.mean((rt - rt.mean()) * (rp - rp.mean())) / (st * sp_)) if st > 0 and sp_ > 0 else 0.0
        num += n * rho; den += n
    return num / den if den else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kidney"); ap.add_argument("--labels", default="codi")
    ap.add_argument("--reference", default="stored"); ap.add_argument("--coarse", action="store_true")
    ap.add_argument("--genes", default="top1500"); ap.add_argument("--mask", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--preset", default="v3")
    ap.add_argument("--ref-estimate", default="mean"); ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", choices=["p1", "p2"], default="p1"); ap.add_argument("--thin", type=float, default=0.5)
    ap.add_argument("--zero-sample", type=int, default=1_000_000)
    ap.add_argument("--cfg", default="", help="extra engine settings, e.g. 'spatial_nonzero_only=false;type_weight=none'")
    a = ap.parse_args()
    if a.protocol == "p2":
        return main_thinning(a)

    pair = datasets.load(a.dataset, **runner._load_kw(a))
    ctx = runner._RecoveryContext(pair, a.genes)
    masked, rows, cols, truth, st_ctx = ctx.hide("p1", a.seed, mask=a.mask)
    embed = (ctx.sc_full, st_ctx) if a.genes != "all" else None
    base = methods.parse_ours(f"ours:{a.preset};{a.cfg}").with_(seed=a.seed, quota_mode="none", fill_threshold=0.0,
                                                                    **({} if a.ref_estimate == "depth_type" else dict(ref_estimate=a.ref_estimate)))
    if ctx.sc_space in ("linear", "log1p"):
        base = base.with_(ref_space=ctx.sc_space)
    ok = ctx.imputable[rows]
    rows, cols, truth = rows[ok], cols[ok], truth[ok]
    terms = {}
    for name, full in two_terms(a, ctx, masked, base, embed).items():
        terms[name] = full[rows, cols].astype(np.float64)
        print(f"{name:8s} term: non-zero at {np.mean(terms[name] > 0):.3f} of hidden entries", flush=True)
        del full
    r, s = terms["ref"], terms["spatial"]
    h = neighbor_homogeneity(ctx.st_labels, ctx.coords, base.k)[rows]
    w_preset = weights_from_h(h, PRESETS[a.preset])
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01), (0.0, 1.01)]
    grid = np.round(np.linspace(0.0, 1.0, 21), 2)
    lt = np.log1p(truth)
    out = []
    for lo, hi in bins:
        m = (h >= lo) & (h < hi)
        if m.sum() < 1000:
            continue
        for w in grid:
            b = w * r[m] + (1.0 - w) * s[m]
            out.append(dict(h_lo=lo, h_hi=min(hi, 1.0), n=int(m.sum()), w=float(w), w_preset=float(w_preset[m].mean()),
                            rmse_log=float(np.sqrt(np.mean((np.log1p(b) - lt[m]) ** 2))),
                            pg_spearman=_pg_spearman(truth[m], b, cols[m])))
    df = pd.DataFrame(out)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\n{a.dataset} {a.genes} seed {a.seed}: best weight on the reference term per homogeneity bin")
    rows_ = []
    for (lo, hi), d in df.groupby(["h_lo", "h_hi"], sort=False):
        d = d.reset_index(drop=True)
        i_r, i_s = d.rmse_log.idxmin(), d.pg_spearman.idxmax()
        near = d.iloc[(d.w - d.w_preset.iloc[0]).abs().idxmin()]
        rows_.append(dict(h=f"{lo:.1f}-{hi:.1f}" if (lo, hi) != (0.0, 1.0) else "all", n=int(d.n.iloc[0]), w_preset=round(d.w_preset.iloc[0], 2),
                          w_best_rmse=d.w[i_r], rmse_best=round(d.rmse_log[i_r], 4), rmse_preset=round(near.rmse_log, 4),
                          rmse_ref_only=round(d.rmse_log.iloc[-1], 4), rmse_spatial_only=round(d.rmse_log.iloc[0], 4),
                          w_best_rho=d.w[i_s], rho_best=round(d.pg_spearman[i_s], 4), rho_preset=round(near.pg_spearman, 4),
                          rho_ref_only=round(d.pg_spearman.iloc[-1], 4), rho_spatial_only=round(d.pg_spearman.iloc[0], 4)))
    print(pd.DataFrame(rows_).to_string(index=False))
    print(f"wrote {a.out}")


def main_thinning(a):
    pair = datasets.load(a.dataset, **runner._load_kw(a))
    ctx = runner._RecoveryContext(pair, a.genes)
    masked, rows, cols, truth, st_ctx = ctx.hide("p2", a.seed, thin=a.thin)
    zr, zc = protocols.zero_sample(masked, rows, cols, a.zero_sample, a.seed)
    embed = (ctx.sc_full, st_ctx) if a.genes != "all" else None
    base = methods.parse_ours(f"ours:{a.preset};{a.cfg}").with_(seed=a.seed, quota_mode="none", fill_threshold=0.0,
                                                                    **({} if a.ref_estimate == "depth_type" else dict(ref_estimate=a.ref_estimate)))
    if ctx.sc_space in ("linear", "log1p"):
        base = base.with_(ref_space=ctx.sc_space)
    n_spots, n_genes = masked.shape
    # zero positions per spot that were zero before thinning as well (the sampled population)
    true_zero_per_spot = n_genes - np.diff(masked.tocsr().indptr) - np.bincount(rows, minlength=n_spots)
    ok_h, ok_z = ctx.imputable[rows], ctx.imputable[zr]
    rows, cols, truth = rows[ok_h], cols[ok_h], truth[ok_h]
    zr, zc = zr[ok_z], zc[ok_z]
    hid, zer = {}, {}
    for name, full in two_terms(a, ctx, masked, base, embed).items():
        hid[name] = full[rows, cols].astype(np.float64); zer[name] = full[zr, zc].astype(np.float64)
        print(f"{name:8s} term: mean at held-out non-zeros {hid[name].mean():.3f}, at true zeros {zer[name].mean():.3f}", flush=True)
        del full
    h_spot = neighbor_homogeneity(ctx.st_labels, ctx.coords, base.k)
    w_spot = weights_from_h(h_spot, PRESETS[a.preset])
    hh, hz = h_spot[rows], h_spot[zr]
    f = (1.0 - a.thin) / a.thin                                  # prediction scale -> held-out molecule scale
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01), (0.0, 1.01)]
    grid = np.round(np.linspace(0.0, 1.0, 21), 2)
    out = []
    for lo, hi in bins:
        mh = (hh >= lo) & (hh < hi); mz = (hz >= lo) & (hz < hi)
        spots = ctx.imputable & (h_spot >= lo) & (h_spot < hi)
        if mh.sum() < 1000 or mz.sum() < 1000:
            continue
        n_zero = float(true_zero_per_spot[spots].sum()); wz = n_zero / mz.sum()      # each sampled zero stands for wz zeros
        n_all = mh.sum() + n_zero
        t = truth[mh]
        for w in grid:
            bh = np.maximum(f * (w * hid["ref"][mh] + (1.0 - w) * hid["spatial"][mh]), 1e-8)
            bz = np.maximum(f * (w * zer["ref"][mz] + (1.0 - w) * zer["spatial"][mz]), 0.0)
            dev = (np.sum(2.0 * (t * np.log(t / bh) - (t - bh))) + wz * np.sum(2.0 * bz)) / n_all
            mse = (np.sum((bh - t) ** 2) + wz * np.sum(bz ** 2)) / n_all
            out.append(dict(h_lo=lo, h_hi=min(hi, 1.0), n_heldout_nonzero=int(mh.sum()), n_true_zero=int(n_zero), w=float(w),
                            w_preset=float(w_spot[spots].mean()), deviance=float(dev), mse=float(mse),
                            pg_spearman=_pg_spearman(t, bh, cols[mh])))
    df = pd.DataFrame(out)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\n{a.dataset} {a.genes} seed {a.seed} thinning p={a.thin}, estimate={a.ref_estimate}: best weight on the reference term per homogeneity bin")
    rows_ = []
    for (lo, hi), d in df.groupby(["h_lo", "h_hi"], sort=False):
        d = d.reset_index(drop=True)
        i_d, i_m = d.deviance.idxmin(), d.mse.idxmin()
        has_rho = d.pg_spearman.notna().any()                      # a small bin may have no gene with enough entries
        i_s = d.pg_spearman.idxmax() if has_rho else None
        near = d.iloc[(d.w - d.w_preset.iloc[0]).abs().idxmin()]
        rows_.append(dict(h=f"{lo:.1f}-{hi:.1f}" if (lo, hi) != (0.0, 1.0) else "all", w_preset=round(d.w_preset.iloc[0], 2),
                          w_best_dev=d.w[i_d], dev_best=round(d.deviance[i_d], 5), dev_preset=round(near.deviance, 5),
                          dev_ref_only=round(d.deviance.iloc[-1], 5), dev_spatial_only=round(d.deviance.iloc[0], 5),
                          w_best_mse=d.w[i_m], mse_best=round(d.mse[i_m], 5), mse_ref_only=round(d.mse.iloc[-1], 5),
                          mse_spatial_only=round(d.mse.iloc[0], 5),
                          w_best_rho=d.w[i_s] if has_rho else np.nan, rho_ref_only=round(d.pg_spearman.iloc[-1], 4), rho_spatial_only=round(d.pg_spearman.iloc[0], 4)))
    print(pd.DataFrame(rows_).to_string(index=False))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
