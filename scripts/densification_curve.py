#!/usr/bin/env python3
"""
densification_curve.py -- accuracy of selective filling as a function of how much is filled.

Every method that fills all zeros (a smoother, a type-mean fill, our fill-everything
variant) can be made selective after the fact: keep only the imputed values above a
cut.  This script takes the saved predictions of masked-entry runs, and for a grid of
target densification levels d picks the cut that fills a fraction d of the observed
zeros (measured on the run's fixed sample of observed zeros), then scores the
thresholded prediction at the hidden entries with the benchmark's own metric code.

    python scripts/densification_curve.py --dataset kidney --genes top1500 \
        --out results/analysis/densification_curve_kidney.csv

Selective methods (those that already fill only part of the zeros) are added as single
points at their own densification.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("ZONEIMPUTE_THREADS", "4"))

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from bench import datasets, metrics, runner  # noqa: E402

LEVELS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kidney"); ap.add_argument("--labels", default="codi")
    ap.add_argument("--reference", default="stored"); ap.add_argument("--coarse", action="store_true")
    ap.add_argument("--genes", default="top1500"); ap.add_argument("--protocol", default="p1")
    ap.add_argument("--runs", default=str(ROOT / "results" / "bench" / "runs")); ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # newest run per (method, seed) of this experiment
    picked = {}
    for mf in Path(a.runs).glob("*/meta.json"):
        m = json.loads(mf.read_text())
        if (m.get("protocol") != a.protocol or m.get("dataset") != a.dataset or m.get("genes") != a.genes
                or m.get("labels", "codi") != a.labels or m.get("reference", "stored") != a.reference
                or bool(m.get("coarse", False)) != a.coarse or m.get("context", "all") != "all" or m.get("status", "ok") != "ok"
                or not (mf.parent / "hidden_pred.npz").exists()):
            continue
        key = (m["method"], m["seed"])
        if key not in picked or m.get("time", "") > picked[key][1]:
            picked[key] = (mf.parent, m.get("time", ""))
    if not picked:
        raise SystemExit("no matching runs")
    pair = datasets.load(a.dataset, **runner._load_kw(a))
    ctx = runner._RecoveryContext(pair, a.genes)
    rows_out = []
    for (method, seed), (d, _) in sorted(picked.items()):
        z = np.load(d / "hidden_pred.npz")
        rows, cols, truth = z["rows"], z["cols"], z["truth"].astype(np.float64)
        pred, zs = z["pred"].astype(np.float64), z["zero_pred"].astype(np.float64)
        ok = ctx.imputable[rows]
        dens_full = float((zs > 0).mean())
        if method == "none":
            continue
        levels = [lv for lv in LEVELS if lv < dens_full - 1e-9] if dens_full > 0.9 else []
        points = [(dens_full, -np.inf)] + [(lv, float(np.quantile(zs, 1.0 - lv))) for lv in levels]
        for target, cut in points:
            p = np.where(pred > cut, pred, 0.0) if np.isfinite(cut) else pred
            zt = np.where(zs > cut, zs, 0.0) if np.isfinite(cut) else zs
            m, _ = metrics.masked_metrics(truth[ok], p[ok], cols[ok], zero_scores=zt)
            rows_out.append(dict(method=method, seed=seed, target=round(target, 4), thresholded=bool(np.isfinite(cut)),
                                 densification=m["densification_at_zeros"], hidden_fill=m["hidden_fill_frac"],
                                 pg_wmean=m["pergene_spearman_wmean"], pooled=m["spearman_all"],
                                 filled_only=m["spearman_filled_only"]))
        print(f"{method[:70]:70s} seed {seed}: {len(points)} points", flush=True)
    df = pd.DataFrame(rows_out)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    full = df[df.method.isin(df[df.thresholded].method.unique())]
    t = full.groupby(["method", "target"]).pg_wmean.mean().unstack().round(3)
    print("\nentry-weighted per-gene Spearman of fill-everything methods cut to a target densification:")
    print(t.to_string())
    sel = df[~df.method.isin(full.method.unique())].groupby("method")[["densification", "hidden_fill", "pg_wmean"]].mean().round(3)
    print("\nselective methods at their own densification:"); print(sel.sort_values("densification").to_string())
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
