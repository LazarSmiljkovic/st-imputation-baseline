"""Figures of the paper, drawn from the result files.

    python scripts/figures.py            # writes results/figures/*.pdf

Figure 4 needs results/analysis/illustration_kidney.npz from scripts/illustration.py.
"""
import os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
BENCH = Path(os.environ.get("ZONEIMPUTE_RESULTS", ROOT / "results/bench"))      # a rerun: results/rerun
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})

D = "ours:depth_type;rate_shrink=20"
REC = f"{D};within_label_pcs=5;within_label_xfit=5;spatial_blend=auto;quota_mode=expected;fill_threshold=0.1"
REC_K = REC + ";relabel=kmeans"
PREV_K = f"{D};quota_mode=expected;fill_threshold=0.1;relabel=kmeans"
REC_FILL_K = f"{D};within_label_pcs=5;within_label_xfit=5;spatial_blend=auto;relabel=kmeans"
REC_FILL = f"{D};within_label_pcs=5;within_label_xfit=5;spatial_blend=auto"
PREV = f"{D};quota_mode=expected;fill_threshold=0.1"
M1 = {REC_K: "proposed (threshold 0.1)", PREV_K: "label x depth only", "scvi": "scVI", "magic": "MAGIC", "alra": "ALRA", "gene_depth": "depth only"}
COL = {"ALRA": "#a5d6a7", "proposed (threshold 0.1)": "#1b6ca8", "proposed (every zero)": "#1b6ca8", "label x depth only": "#7fb3d5", "scVI": "#d35400", "MAGIC": "#8e8e8e", "depth only": "#c9c9c9"}
DS = {"kidney": "kidney", "kidney_puck:GSM5713381": "kidney puck 2", "adultB5": "adult brain", "embryo_raw": "embryo", "dlpfc:151673": "DLPFC",
      "visium10x": "Visium brain", "tissue:DataUploadFinal/SVZFullFinal": "SVZ", "tissue:DataUploadFinal/Dataset9": "D9",
      "tissue:DataUploadFinal/Dataset1": "D1", "tissue:DataUploadFinal/Dataset17": "D17", "tissue:DataUploadFinal/Dataset15": "D15"}


def load(name):
    d = pd.read_csv(BENCH / name)
    d["method"] = d["method"].str.replace(r";n_jobs=\d+", "", regex=True)   # worker processes give identical results
    for k, v in (("labels", "codi"), ("context", "all"), ("coarse", False)):
        d[k] = d[k].fillna(v) if k in d else v
    return d[(d.labels == "codi") & (d.context == "all") & (d.coarse == False)]


def ci(x):
    x = np.asarray(x, float)
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def grouped(ax, d, metric, methods, ylabel, title):
    secs = list(DS.values()); w = 0.8 / len(methods); x = np.arange(len(secs))
    for i, (spec, name) in enumerate(methods.items()):
        means, errs = [], []
        for ds in DS:
            s = d[(d.dataset == ds) & (d.method == spec)].groupby("seed")[metric].mean()
            means.append(s.mean() if len(s) else np.nan); errs.append(ci(s) if len(s) > 1 else 0.0)
            if not len(s):
                print(f"{title}: no runs for {name} on {DS[ds]}")
        ax.bar(x + (i - len(methods) / 2 + 0.5) * w, means, w, yerr=errs, color=COL[name], label=name, error_kw=dict(lw=0.6, capsize=1.5))
    ax.set_xticks(x); ax.set_xticklabels(secs, rotation=35, ha="right"); ax.set_ylabel(ylabel); ax.set_title(title, loc="left")


def fig_recovery():
    p1 = load("p1_results.csv")
    p1 = p1[p1.genes.isin(["top1500", "all"])]
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2))
    grouped(axes[0], p1, "pergene_spearman_wmean", M1, "per-gene Spearman at hidden entries", "(a) masked entries")
    grouped(axes[1], p1, "pergene_spearman_depthstrat", M1, "the same within depth deciles", "(b) recovery within library-size groups")
    axes[0].legend(ncol=6, fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.02))
    fig.tight_layout(); fig.savefig(OUT / "fig_recovery.pdf"); plt.close(fig)


def fig_calibration():
    p2 = load("p2_results.csv")
    methods = {REC_FILL_K: "proposed (every zero)", "scvi": "scVI", "magic": "MAGIC", "gene_depth": "depth only"}
    secs = ["kidney", "kidney_puck:GSM5713381", "adultB5", "embryo_raw", "dlpfc:151673", "visium10x"]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    w = 0.8 / len(methods); x = np.arange(len(secs))
    for i, (spec, name) in enumerate(methods.items()):
        vals = []
        for ds in secs:
            s = p2[(p2.dataset == ds) & (p2.method == spec)].groupby("seed").heldout_deviance.mean()
            vals.append(s.mean() if len(s) else np.nan)
            if not len(s):
                print(f"calibration: no runs for {name} on {DS[ds]}")
        ax.bar(x + (i - len(methods) / 2 + 0.5) * w, vals, w, color=COL[name], label=name)
    ax.set_xticks(x); ax.set_xticklabels([DS[s] for s in secs]); ax.set_ylabel("held-out Poisson deviance (lower is better)")
    ax.legend(ncol=5, fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(OUT / "fig_calibration.pdf"); plt.close(fig)


def fig_densification():
    """Accuracy against the share of zeros filled on the kidney: curves for fill-everything methods thresholded after
    the fact (from the densification analysis), points for the selective methods (from the benchmark runs)."""
    d = pd.read_csv(ROOT / "results/analysis/densification_curve_kidney_v2.csv")
    curves = {"magic": ("MAGIC, thresholded", "#8e8e8e"), "scvi": ("scVI, thresholded", "#d35400"), "scrna_mean": ("type mean, thresholded", "#c9c9c9"),
              D: ("label x depth, thresholded", "#7fb3d5"), REC_FILL: ("proposed, thresholded", "#1b6ca8")}
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    for m, (label, col) in curves.items():
        s = d[d.method == m].groupby("densification").pg_wmean.mean().sort_index()
        ax.plot(100 * s.index, s.values, marker="o", ms=2.5, lw=1, color=col, label=label)
    for m, (label, mk) in {"spamean:mode=k9_t0.5": ("SpaMean-Impute", "s"), "spatial_matched": ("spatial average, matched fills", "^")}.items():
        s = d[d.method == m]
        if len(s):
            ax.scatter(100 * s.densification.mean(), s.pg_wmean.mean(), marker=mk, color="#555555", s=22, label=label, zorder=5)
    p1 = load("p1_results.csv"); k = p1[(p1.dataset == "kidney") & (p1.genes == "top1500")]
    s = k[k.method == REC]
    ax.scatter(100 * s.densification_at_zeros.mean(), s.pergene_spearman_wmean.mean(), marker="D", s=40, color="#1b6ca8", edgecolor="black", zorder=7,
               label="proposed, threshold 0.1 (the rule's own choice)")
    ax.set_xscale("log"); ax.set_xlabel("zeros filled (%)"); ax.set_ylabel("per-gene Spearman at hidden entries")
    ax.legend(fontsize=6.5, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5)); fig.tight_layout(); fig.savefig(OUT / "fig_densification.pdf", bbox_inches="tight"); plt.close(fig)


def fig_markers():
    d = pd.read_csv(ROOT / "results/marker/marker_recovery_v2.csv"); d = d[d.family == "kidney"].copy()
    d["method"] = d["method"].str.replace(r";n_jobs=\d+", "", regex=True)
    N = {"none": ("observed", "#c9c9c9"), "gene_depth": ("depth only", "#c9c9c9"), "spatial_avg": ("spatial average", "#9fc5e8"),
         PREV: ("label x depth, threshold 0.1", "#7fb3d5"), REC: ("proposed, threshold 0.1", "#1b6ca8"),
         REC_FILL: ("proposed, every zero", "#1b6ca8"), "magic": ("MAGIC", "#8e8e8e"), "alra": ("ALRA", "#a5d6a7"), "scvi": ("scVI", "#d35400")}
    d = d[d.method.isin(N) & ((d.labels == "kmeans") | ~d.method.str.startswith("ours"))]
    g = d.groupby("method").agg(auroc=("marker_auroc", "mean"), lift=("lift", "mean"), fill=("fill_frac", "mean"), fm=("false_marker_share", "mean"))
    order = [m for m in N if m in g.index]
    missing = [N[m][0] for m in N if m not in g.index]
    if missing:
        print("fig_markers: no runs for", missing)
    g = g.loc[order]; names = [N[m][0] for m in order]; cols = [N[m][1] for m in order]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
    y = np.arange(len(order))
    axes[0].barh(y, g.auroc - 0.5, left=0.5, color=cols); axes[0].set_yticks(y); axes[0].set_yticklabels(names); axes[0].invert_yaxis()
    axes[0].set_xlim(0.5, max(0.7, g.auroc.max() + 0.02)); axes[0].set_xlabel("marker AUROC (0.5 = chance)")
    for i, v in enumerate(g.auroc):
        axes[0].text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=6)
    axes[1].barh(y, g.lift, color=cols); axes[1].set_yticks(y); axes[1].set_yticklabels([]); axes[1].invert_yaxis()
    axes[1].set_xlim(0, g.lift.max() * 1.7); axes[1].set_xlabel("lift over non-specific genes")
    for i, (l, f) in enumerate(zip(g.lift, g.fill)):
        axes[1].text(l + 0.002, i, f"{l:.3f} ({100 * f:.0f}%)", va="center", fontsize=6)
    axes[2].barh(y, g.fm, color=cols); axes[2].set_yticks(y); axes[2].set_yticklabels([]); axes[2].invert_yaxis()
    axes[2].set_xlim(0, g.fm.max() * 1.3); axes[2].set_xlabel("share of false markers")
    for i, v in enumerate(g.fm):
        axes[2].text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=6)
    fig.tight_layout(); fig.savefig(OUT / "fig_markers.pdf"); plt.close(fig)


def fig_illustration():
    z = np.load(ROOT / "results/analysis/illustration_kidney.npz", allow_pickle=True)
    xy = z["coords"]; genes = list(z["genes"]); obs = z["observed"]; filled = z["filled"]; prob = z["prob"]
    show = [g for g in ("Umod", "Aqp2") if g in genes]
    fig, axes = plt.subplots(len(show), 3, figsize=(7.2, 2.4 * len(show)))
    axes = np.atleast_2d(axes)
    for r, g in enumerate(show):
        j = genes.index(g); o = obs[:, j]; f = filled[:, j]; p = prob[:, j]
        vmax = np.percentile(f[f > 0], 98) if (f > 0).any() else 1.0
        new = (o == 0) & (f > 0)
        for c, (vals, title) in enumerate(((o, f"{g}: observed counts"), (f, f"{g}: after the fill at threshold 0.1"), (np.where(o == 0, p, np.nan), f"{g}: probability of a molecule at the zeros"))):
            ax = axes[r, c]
            ax.scatter(xy[:, 0], xy[:, 1], s=0.3, c="#d9d9d9", linewidths=0, rasterized=True)
            if c < 2:
                m = vals > 0
                order = np.argsort(vals[m])
                sc = ax.scatter(xy[m, 0][order], xy[m, 1][order], s=0.6, c=vals[m][order], cmap="viridis", vmin=0, vmax=vmax, linewidths=0, rasterized=True)
                if c == 1:
                    ax.scatter(xy[new, 0], xy[new, 1], s=0.6, c=f[new], cmap="viridis", vmin=0, vmax=vmax, linewidths=0, rasterized=True)
                    ax.text(0.02, 0.02, f"{100 * new.sum() / (o == 0).sum():.0f}% of the zeros filled", transform=ax.transAxes, fontsize=6.5)
            else:
                m = ~np.isnan(vals) & (vals > 0.02)
                order = np.argsort(vals[m])
                sc = ax.scatter(xy[m, 0][order], xy[m, 1][order], s=0.6, c=vals[m][order], cmap="magma", vmin=0, vmax=1, linewidths=0, rasterized=True)
            ax.set_title(title, fontsize=7.5, loc="left"); ax.set_aspect("equal"); ax.axis("off")
            fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01).ax.tick_params(labelsize=6)
    fig.tight_layout(); fig.savefig(OUT / "fig_illustration.pdf", dpi=200); plt.close(fig)


def fig_reliability():
    """Reliability of the model-implied probability under thinning: mean predicted against observed rate of at
    least one removed molecule, in probability bins, for the proposed estimate (Poisson and negative binomial)
    and scVI; weights from the reweighted true zeros."""
    b = pd.read_csv(ROOT / "results/analysis/reliability.csv")
    secs = [("kidney", "kidney"), ("dlpfc:151673", "DLPFC"), ("adultB5", "adult brain"), ("visium10x", "Visium brain")]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1), sharex=True, sharey=True)
    for ax, (ds, name) in zip(axes, secs):
        for (meth, prob), (col, ls, lab) in {("proposed", "Poisson"): ("#1b6ca8", "-", "proposed, Poisson"), ("proposed", "NB"): ("#1b6ca8", "--", "proposed, NB"),
                                             ("scVI", "Poisson"): ("#d35400", "-", "scVI, Poisson")}.items():
            s = b[(b.dataset == ds) & (b.method == meth) & (b.prob == prob)]
            if not len(s):
                print("reliability: no data", ds, meth, prob); continue
            g = s.groupby("bin").apply(lambda x: pd.Series(dict(p=np.average(x.mean_pred, weights=x.weight), o=np.average(x.observed, weights=x.weight), w=x.weight.sum())))
            g = g[g.w > 1e-4]
            ax.plot(g.p, g.o, ls, color=col, marker="o", ms=2.5, lw=1, label=lab)
        ax.plot([0, 1], [0, 1], color="black", lw=0.5); ax.set_title(name, fontsize=7.5, loc="left"); ax.set_xlabel("predicted probability")
    axes[0].set_ylabel("observed rate"); axes[0].legend(fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "fig_reliability.pdf"); plt.close(fig)


def fig_unmeasured(extra=None):
    p3 = pd.read_csv(BENCH / "p3_results.csv")
    N = {"scrna_mean": "type mean", "tangram:mode=clusters": "Tangram (clusters)",
         "tangram:mode=cells;epochs=1000;max-ref-cells=20000": "Tangram (cells)", "spage": "SpaGE", "stplus": "stPlus",
         "gimvi:epochs=200;max-ref-genes=2000": "gimVI", "envi": "ENVI"}
    if extra:
        N.update(extra)
    d = p3[p3.method.isin(N)].copy(); d["m"] = d.method.map(N); d["ds"] = d.dataset.str.replace("tissue:DataUploadFinal/", "")
    d = d[~d.ds.isin(["Axolotl", "Dataset7"])]
    g = d.pivot_table(index="m", columns="ds", values="pcc_median", aggfunc="mean").rank(ascending=False).mean(axis=1)
    c = d.pivot_table(index="m", columns="ds", values="cell_spearman_median", aggfunc="mean").rank(ascending=False).mean(axis=1)
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    offsets = {"stPlus": (4, 4), "SpaGE": (4, -4), "proposed": (4, 4), "ENVI": (4, -4), "type mean": (4, -4), "Tangram (cells)": (4, 4), "Tangram (clusters)": (-4, -10), "gimVI": (4, 4)}
    for m in g.index:
        col = "#1b6ca8" if m.startswith("proposed") else "#555555"
        ax.scatter(g[m], c[m], color=col, s=22)
        dx, dy = offsets.get(m, (4, 4))
        ax.annotate(m, (g[m], c[m]), fontsize=6.5, xytext=(dx, dy), textcoords="offset points", ha="right" if dx < 0 else "left")
    ax.set_xlabel("mean rank, gene-wise correlation (1 = best)"); ax.set_ylabel("mean rank, cell-wise correlation (1 = best)")
    ax.invert_xaxis(); ax.invert_yaxis(); fig.tight_layout(); fig.savefig(OUT / "fig_unmeasured.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig_recovery(); fig_calibration(); fig_densification(); fig_markers(); fig_reliability()
    if (ROOT / "results/analysis/illustration_kidney.npz").exists():
        fig_illustration()
    else:
        print("Figure 4 skipped: run scripts/illustration.py first (needs the kidney data)")
    fig_unmeasured({"ours:v3;predict_backbone=soft;predict_smooth=auto;predict_tune=genes": "proposed"})
    print("figures written to", OUT)
