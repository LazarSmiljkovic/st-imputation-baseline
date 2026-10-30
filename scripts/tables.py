"""Tables of the paper as CSV files, computed from the result files.

    python scripts/tables.py                        # reads results/bench and results/analysis, writes results/tables/*.csv
    python scripts/tables.py --bench results/rerun  # the same from a rerun (see reproduce.py)

Inputs: results/bench/p{1,2,3}_results.csv (bench.runner aggregate) and, for the calibration, threshold and
cluster tables, the outputs of scripts/reliability.py, tau_specificity.py, cluster_coverage.py and
label_leak_check.py in results/analysis.  Every table is also printed.

    recovery.csv          main sections: per-gene Spearman at hidden entries, paired differences, share filled
    heldout.csv           the ten sections not used for development
    section_tests.csv     Wilcoxon, Hodges-Lehmann and equivalence tests over the 21 sections
    label_leak.csv        the final configuration with stored labels and k-means on the full and on the masked section
    thinning.csv          held-out Poisson deviance and squared error under thinning
    calibration.csv       ECE, Brier score and skill under thinning
    fill_decision.csv     average precision, precision and recall of the fill rule at tau = 0.1
    sensitivity.csv       fixed settings on the kidney and D9
    threshold.csv         threshold against the share of fills in labels where the gene is enriched
    stereo_clusters.csv   number of k-means clusters on the Stereo-seq sections
    scanvi.csv            scANVI against scVI
    unmeasured.csv        left-out genes on the imaging panels, mean and mean rank over panels
    unmeasured_ranks.csv  the same ranks per panel
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
D = "ours:depth_type;rate_shrink=20"
W = "within_label_pcs=5;within_label_xfit=5;spatial_blend=auto"
R = ";relabel=kmeans"
PROP = f"{D};{W};quota_mode=expected;fill_threshold=0.1{R}"
PROP_ALL = f"{D};{W}{R}"
BASE = f"{D};quota_mode=expected;fill_threshold=0.1{R}"
NAMES = {"kidney": "kidney", "kidney_puck:GSM5713381": "kidney puck 2", "adultB5": "adult brain", "embryo_raw": "embryo brain",
         "dlpfc:151673": "DLPFC", "visium10x": "Visium brain", "tissue:DataUploadFinal/SVZFullFinal": "SVZ",
         "tissue:DataUploadFinal/Dataset9": "D9", "tissue:DataUploadFinal/Dataset1": "D1", "tissue:DataUploadFinal/Dataset17": "D17",
         "tissue:DataUploadFinal/Dataset15": "D15"}
HELD = {"kidney_puck:GSM5713354": "kidney puck, BTBR control", "kidney_puck:GSM5713350": "kidney puck, diabetic",
        "kidney_puck:GSM5713388": "kidney puck, UMOD knock-in", "dlpfc:151507": "DLPFC 151507", "dlpfc:151669": "DLPFC 151669",
        "dlpfc:151676": "DLPFC 151676", "tissue:DataUploadFinal/Dataset3": "Dataset3", "tissue:DataUploadFinal/Dataset11": "Dataset11",
        "tissue:DataUploadFinal/Dataset12": "Dataset12", "tissue:DataUploadFinal/Dataset13": "Dataset13"}
COMP = {"base": BASE, "scVI": "scvi", "MAGIC": "magic", "ALRA": "alra", "depth only": "gene_depth"}
MARGIN = 0.01


def load(bench, proto):
    d = pd.read_csv(Path(bench) / f"{proto}_results.csv", low_memory=False)
    d = d[d.status.fillna("ok") == "ok"].copy()          # runs scored before the status field existed have none
    d["m"] = d.method.str.replace(r";n_jobs=\d+", "", regex=True)       # worker processes give identical results
    for k, v in (("labels", "codi"), ("reference", "stored"), ("context", "all")):
        d[k] = d[k].fillna(v)
    d = d[(d.labels == "codi") & (d.reference == "stored") & (d.context == "all") & (d.coarse.astype(str) != "True")]
    return d[d.genes.isin(["top1500", "all"])] if "genes" in d and proto != "p3" else d


def ci(x):
    x = np.asarray(x, float)
    return stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size) if x.size > 1 else np.nan


def per_seed(d, ds, m, col="pergene_spearman_wmean"):
    return d[(d.dataset == ds) & (d.m == m)].groupby("seed")[col].mean()


def hodges_lehmann(x, alpha=0.05):
    x = np.sort(np.asarray(x)); n = len(x)
    w = np.sort([(x[i] + x[j]) / 2 for i in range(n) for j in range(i, n)])
    # normal approximation to the signed-rank distribution for the interval
    c = int(np.floor(n * (n + 1) / 4 - stats.norm.ppf(1 - alpha / 2) * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
    return float(np.median(w)), float(w[max(c, 0)]), float(w[len(w) - 1 - max(c, 0)])


def recovery(p1):
    rows = []
    for ds, name in NAMES.items():
        s = {k: per_seed(p1, ds, m) for k, m in {"proposed": PROP, **COMP}.items()}
        row = dict(section=name, **{k: v.mean() for k, v in s.items() if k != "depth only"})
        for other in ("base", "scVI"):
            common = s["proposed"].index.intersection(s[other].index)
            diff = s["proposed"].loc[common] - s[other].loc[common]
            row[f"minus_{other}"], row[f"minus_{other}_halfwidth"] = diff.mean(), ci(diff)
            row[f"minus_{other}_excludes_zero"] = bool(abs(diff.mean()) > ci(diff))
        row["filled"] = per_seed(p1, ds, PROP, "densification_at_zeros").mean()
        row["seeds"] = s["proposed"].size
        rows.append(row)
    return pd.DataFrame(rows)


def section_values(p1):
    v = p1.pivot_table(index="dataset", columns="m", values="pergene_spearman_wmean", aggfunc="mean")
    secs = list(NAMES) + list(HELD)
    return pd.DataFrame({"set": ["main"] * len(NAMES) + ["held-out"] * len(HELD), "proposed": v.loc[secs, PROP].values,
                         **{k: v.loc[secs, m].values for k, m in COMP.items()},
                         "seeds": p1[p1.m == PROP].groupby("dataset").seed.nunique().reindex(secs).values}, index=secs)


def section_tests(vals):
    rows = []
    for k in COMP:
        x = (vals.proposed - vals[k]).dropna().values
        hl, lo, hi = hodges_lehmann(x)
        p_up = stats.wilcoxon(x + MARGIN, alternative="greater").pvalue      # difference above -margin
        p_dn = stats.wilcoxon(x - MARGIN, alternative="less").pvalue         # difference below +margin
        rows.append(dict(comparator=k, n=len(x), wins=int((x > 0).sum()), above_margin=int((x > MARGIN).sum()),
                         below_margin=int((x < -MARGIN).sum()), median=float(np.median(x)), hl=hl, hl_lo=lo, hl_hi=hi,
                         wilcoxon_p=float(stats.wilcoxon(x).pvalue), tost_p=float(max(p_up, p_dn))))
    return pd.DataFrame(rows)


def thinning(p2):
    M = {PROP_ALL: "proposed (every zero)", "scvi": "scVI", "magic": "MAGIC", "gene_depth": "depth only"}
    secs = ["kidney", "kidney_puck:GSM5713381", "adultB5", "embryo_raw", "dlpfc:151673", "visium10x"]
    d = p2[p2.m.isin(M) & p2.dataset.isin(secs)]
    g = d.groupby(["dataset", "m"])[["heldout_deviance", "heldout_mse"]].mean().unstack("m")
    out = pd.concat({"deviance": g["heldout_deviance"], "mse": g["heldout_mse"]}, axis=1)
    out.columns = [f"{a}, {M[b]}" for a, b in out.columns]
    out["seeds"] = d.groupby("dataset").seed.nunique()
    return out.loc[secs].rename(index=NAMES)


def calibration(analysis):
    s = pd.read_csv(analysis / "reliability_summary.csv").groupby(["dataset", "method", "prob"]).mean(numeric_only=True)
    f = pd.read_csv(analysis / "fill_decision.csv").groupby(["dataset", "method", "tau"]).mean(numeric_only=True)
    a, b = [], []
    for ds in ["kidney", "kidney_puck:GSM5713381", "adultB5", "embryo_raw", "dlpfc:151673", "visium10x"]:
        g = lambda m, p="Poisson": s.loc[(ds, m, p)]
        bss = lambda m: 1 - g(m).brier / g("depth only").brier
        a.append(dict(section=NAMES[ds], observed=g("proposed").observed_rate, predicted=g("proposed").mean_pred,
                      ece=g("proposed").ece, ece_nb=g("proposed", "NB").ece, ece_scvi=g("scVI").ece, ece_magic=g("MAGIC").ece,
                      ece_filled=g("proposed").ece_filled, ece_filled_nb=g("proposed", "NB").ece_filled, brier=g("proposed").brier,
                      brier_skill=bss("proposed"), brier_skill_scvi=bss("scVI")))
        h = lambda m: f.loc[(ds, m, 0.1)]
        b.append(dict(section=NAMES[ds], ap=g("proposed").average_precision, ap_scvi=g("scVI").average_precision,
                      ap_depth=g("depth only").average_precision, filled=h("proposed").fill_share, precision=h("proposed").precision,
                      recall=h("proposed").recall, precision_scvi=h("scVI").precision, recall_scvi=h("scVI").recall,
                      filled_nb=h("proposed, NB").fill_share, precision_nb=h("proposed, NB").precision, recall_nb=h("proposed, NB").recall))
    return pd.DataFrame(a), pd.DataFrame(b)


def sensitivity(p1):
    F = f"{D};{W}"
    rows = [("default, every zero", f"{F}{R}"), ("default, tau 0.1", f"{F};quota_mode=expected;fill_threshold=0.1{R}"),
            ("shrinkage k 5", f"{D};rate_shrink=5;{W}{R}"), ("shrinkage k 80", f"{D};rate_shrink=80;{W}{R}"),
            ("K 3 components", f"{D};within_label_pcs=3;within_label_xfit=5;spatial_blend=auto{R}"),
            ("K 10 components", f"{D};within_label_pcs=10;within_label_xfit=5;spatial_blend=auto{R}"),
            ("3 cross-fitting folds", f"{D};within_label_pcs=5;within_label_xfit=3;spatial_blend=auto{R}"),
            ("10 cross-fitting folds", f"{D};within_label_pcs=5;within_label_xfit=10;spatial_blend=auto{R}"),
            ("keep probability 0.75", f"{F};tune_keep=0.75{R}"), ("8 clusters", f"{F};relabel_k=8{R}"),
            ("30 clusters", f"{F};relabel_k=30{R}"), ("tau 0.05", f"{F};quota_mode=expected;fill_threshold=0.05{R}"),
            ("tau 0.3", f"{F};quota_mode=expected;fill_threshold=0.3{R}"), ("tau 0.5", f"{F};quota_mode=expected;fill_threshold=0.5{R}")]
    d = p1[p1.seed.isin([42, 43])]
    g = d.groupby(["m", "dataset"])[["pergene_spearman_wmean", "densification_at_zeros"]].mean()
    out = []
    for name, m in rows:
        row = dict(setting=name)
        for ds, tag in (("kidney", "kidney"), ("tissue:DataUploadFinal/Dataset9", "D9")):
            r = g.loc[(m, ds)] if (m, ds) in g.index else pd.Series(dtype=float)
            row[f"{tag} spearman"], row[f"{tag} filled"] = r.get("pergene_spearman_wmean"), r.get("densification_at_zeros")
        out.append(row)
    return pd.DataFrame(out)


def stereo_clusters(p1, analysis):
    cov = pd.read_csv(analysis / "cluster_coverage.csv")
    d = p1[p1.seed.isin([42, 43, 44]) & (p1.genes == "top1500")]
    g = d.groupby(["m", "dataset"]).pergene_spearman_wmean.mean()
    ann = {"adultB5": 249, "embryo_raw": 259}
    out = []
    for name, suf in [("annotation (249, 259)", ""), ("7", ";relabel_k=7"), ("30", ";relabel_k=30"), ("60", ";relabel_k=60")]:
        row = dict(clusters=name)
        for ds in ann:
            k = ann[ds] if not suf else int(suf.split("=")[1])
            row[f"{NAMES[ds]} proposed"] = g.get((PROP + suf, ds)); row[f"{NAMES[ds]} base"] = g.get((BASE + suf, ds))
            row[f"{NAMES[ds]} spots corrected"] = cov[(cov.dataset == ds) & (cov.k == k)].share_spots_corrected.iloc[0]
        out.append(row)
    out.append({"clusters": "scVI", "adult brain proposed": g.get(("scvi", "adultB5")), "embryo brain proposed": g.get(("scvi", "embryo_raw"))})
    return pd.DataFrame(out)


def scanvi(p1):
    d = p1[p1.m.isin(["scvi", "scvi:mode=labels", "scvi:mode=labels_kmeans"]) & p1.seed.isin([42, 43, 44])]
    p = d.pivot_table(index=["dataset", "seed"], columns="m", values="pergene_spearman_wmean").dropna()
    rows = []
    for ds, name in NAMES.items():
        if ds not in p.index.get_level_values(0):
            continue
        q = p.loc[ds]
        rows.append(dict(section=name, scvi=q["scvi"].mean(), scanvi_masked_labels=q["scvi:mode=labels_kmeans"].mean(),
                         difference_masked=(q["scvi:mode=labels_kmeans"] - q["scvi"]).mean(), scanvi_stored_labels=q["scvi:mode=labels"].mean(),
                         difference_stored=(q["scvi:mode=labels"] - q["scvi"]).mean(), seeds=len(q)))
    return pd.DataFrame(rows)


def unmeasured(p3):
    N = {"tangram:mode=clusters": "Tangram (clusters)", "spage": "SpaGE", "ours:v3;predict_backbone=soft;predict_smooth=auto;predict_tune=genes": "proposed",
         "tangram:mode=cells;epochs=1000;max-ref-cells=20000": "Tangram (cells)", "stplus": "stPlus", "envi": "ENVI",
         "scrna_mean": "type mean (control)", "gimvi:epochs=200;max-ref-genes=2000": "gimVI"}
    C = {"pcc_median": False, "spearman_median": False, "ssim_median": False, "rmse_z_median": True, "js_median": True, "cell_spearman_median": False}
    d = p3[p3.method.isin(N)].copy(); d["m"] = d.method.map(N); d["panel"] = d.dataset.str.replace("tissue:DataUploadFinal/", "")
    d = d[~d.panel.isin(["Axolotl", "Dataset7"])]            # at chance for every method (see the paper)
    mean, rank, per = {}, {}, []
    for c, low in C.items():
        v = d.pivot_table(index="m", columns="panel", values=c, aggfunc="mean")
        r = v.rank(ascending=low)
        mean[c], rank[c] = v.mean(axis=1), r.mean(axis=1)
        per.append(r.stack().rename(c))
    out = pd.DataFrame({**{c: mean[c] for c in C}, **{f"{c} rank": rank[c] for c in C}}).loc[list(N.values())]
    return out, pd.concat(per, axis=1).reset_index().rename(columns={"m": "method"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=os.environ.get("ZONEIMPUTE_RESULTS", str(ROOT / "results/bench"))); ap.add_argument("--analysis", default=str(ROOT / "results/analysis"))
    ap.add_argument("--out", default=str(ROOT / "results/tables"))
    a = ap.parse_args()
    out, analysis = Path(a.out), Path(a.analysis); out.mkdir(parents=True, exist_ok=True)
    p1, p2, p3 = (load(a.bench, p) for p in ("p1", "p2", "p3"))
    vals = section_values(p1)
    tables = {"recovery": recovery(p1), "heldout": vals[vals.set == "held-out"].rename(index=HELD), "section_tests": section_tests(vals),
              "thinning": thinning(p2), "sensitivity": sensitivity(p1), "scanvi": scanvi(p1)}
    tables["unmeasured"], tables["unmeasured_ranks"] = unmeasured(p3)
    for name, fn in (("calibration", lambda: calibration(analysis)), ("stereo_clusters", lambda: stereo_clusters(p1, analysis))):
        try:
            r = fn()
            if isinstance(r, tuple):
                tables["calibration"], tables["fill_decision"] = r
            else:
                tables[name] = r
        except FileNotFoundError as e:
            print(f"skipped {name}: {e.filename} missing (run the script that writes it first)")
    if (analysis / "label_leak_check.csv").exists():
        q = pd.read_csv(analysis / "label_leak_check.csv")
        tables["label_leak"] = q.pivot_table(index="dataset", columns="labels", values="wmean", aggfunc="mean").rename(index=NAMES | HELD)
    if (analysis / "tau_specificity.csv").exists():
        tables["threshold"] = pd.read_csv(analysis / "tau_specificity.csv")
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=not isinstance(t.index, pd.RangeIndex))
        print(f"\n== {name}\n{t.round(4).to_string()}")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
