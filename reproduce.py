"""
reproduce.py -- rerun the recorded experiments and compare the new numbers with ours.

results/bench/p{1,2,3}_results.csv hold one row per experiment we ran (dataset, panel, seed, method, ...).
This script picks a group of those rows, runs each experiment again with bench.runner into a separate
folder (results/rerun by default), aggregates it and lists the differences to the recorded values.

    python reproduce.py list                                  # groups and their sizes
    python reproduce.py run quick                             # proposed, base and depth only on the kidney, one seed
    python reproduce.py run recovery --datasets kidney adultB5 --seeds 42
    python reproduce.py run recovery --dry-run                # print the bench.runner commands only
    python reproduce.py compare recovery                      # recorded vs rerun, per experiment

Groups
    quick       the kidney, mask seed 42: proposed, base variant, depth only (no external tools, a few minutes)
    recovery    masked entries on the 11 main and 10 held-out sections: proposed (threshold and every zero),
                base variant, depth only, scVI, MAGIC, ALRA
    variants    masked entries, every other configuration with labels from the masked input (settings,
                number of clusters, soft labels, reference-taught correction) and the scVI/scANVI variants
    thinning    binomial thinning on the six sequencing sections (deviance, calibration)
    unmeasured  left-out genes on the imaging panels (5 folds): the proposed predictor and the comparators
    markers     downstream marker test on the five kidney pucks (bench.runner_markers)
    all         every recorded experiment of the three protocols, including the development runs

The engine is deterministic for a given seed and the runs on CPU should match to the printed precision.
scVI, gimVI, Tangram, stPlus and ENVI on a GPU are not bit-reproducible; expect differences in the third decimal.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RECORDED = ROOT / "results" / "bench"

D = "ours:depth_type;rate_shrink=20"
PROPOSED = f"{D};within_label_pcs=5;within_label_xfit=5;spatial_blend=auto;quota_mode=expected;fill_threshold=0.1;relabel=kmeans"
PROPOSED_ALL = f"{D};within_label_pcs=5;within_label_xfit=5;spatial_blend=auto;relabel=kmeans"
BASE = f"{D};quota_mode=expected;fill_threshold=0.1;relabel=kmeans"
MAIN = ["kidney", "kidney_puck:GSM5713381", "adultB5", "embryo_raw", "dlpfc:151673", "visium10x"] + \
       [f"tissue:DataUploadFinal/{p}" for p in ["Dataset1", "Dataset9", "Dataset15", "Dataset17", "SVZFullFinal"]]
HELD_OUT = ["kidney_puck:GSM5713354", "kidney_puck:GSM5713350", "kidney_puck:GSM5713388", "dlpfc:151507", "dlpfc:151669",
            "dlpfc:151676"] + [f"tissue:DataUploadFinal/{p}" for p in ["Dataset3", "Dataset11", "Dataset12", "Dataset13"]]
THINNING = ["kidney", "kidney_puck:GSM5713381", "adultB5", "embryo_raw", "dlpfc:151673", "visium10x"]
PANELS = [f"tissue:DataUploadFinal/{p}" for p in ["Dataset1", "Dataset3", "Dataset9", "Dataset11", "Dataset12", "Dataset13",
                                                  "Dataset14", "Dataset15", "Dataset17", "SVZFullFinal"]]
UNMEASURED = ["ours:v3;predict_backbone=soft;predict_smooth=auto;predict_tune=genes", "scrna_mean", "tangram:mode=clusters",
              "tangram:mode=cells;epochs=1000;max-ref-cells=20000", "spage", "stplus", "gimvi:epochs=200;max-ref-genes=2000", "envi"]
MARKER_METHODS = ["none", "gene_depth", "spatial_avg", f"{D};quota_mode=expected;fill_threshold=0.1",
                  f"{D};within_label_pcs=5;within_label_xfit=5;spatial_blend=auto;quota_mode=expected;fill_threshold=0.1;n_jobs=6",
                  f"{D};within_label_pcs=5;within_label_xfit=5;spatial_blend=auto;n_jobs=6", "magic", "scvi", "alra"]

# headline measure of each protocol, used by compare
METRICS = {"p1": ["pergene_spearman_wmean", "pergene_spearman_depthstrat", "densification_at_zeros"],
           "p2": ["heldout_deviance", "heldout_mse"],
           "p3": ["pcc_median", "cell_spearman_median"]}
KEYS = ["protocol", "dataset", "labels", "reference", "coarse", "genes", "mask", "thin", "context", "folds", "fold",
        "max_genes", "seed", "method"]


def _plain(m):
    return m.str.replace(r";n_jobs=\d+", "", regex=True)       # worker processes give identical results


def _key(v):
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    try:
        f = float(v)
        return "" if np.isnan(f) else repr(f)
    except (TypeError, ValueError):
        return str(v)


def recorded(proto):
    d = pd.read_csv(RECORDED / f"{proto}_results.csv", low_memory=False)
    d = d[d.status.fillna("ok") == "ok"].copy()          # runs scored before the status field existed have none
    for k, v in (("labels", "codi"), ("reference", "stored"), ("context", "all"), ("coarse", False)):
        d[k] = d[k].fillna(v)
    d["coarse"] = d["coarse"].astype(str) == "True"
    d["cluster"] = d.get("cluster", pd.Series(index=d.index, dtype=object)).astype(str) == "True"
    return d


def select(group):
    """Recorded rows of a group, as {protocol: frame}."""
    if group == "markers":
        return {}
    p1, p2, p3 = recorded("p1"), recorded("p2"), recorded("p3")
    std = lambda d: d[(d.labels == "codi") & (d.reference == "stored") & (d.context == "all") & ~d.coarse]
    if group == "all":
        return {"p1": p1, "p2": p2, "p3": p3}
    if group == "quick":
        d = std(p1)
        return {"p1": d[(d.dataset == "kidney") & (d.genes == "top1500") & (d.seed == 42) &
                        _plain(d.method).isin([PROPOSED, BASE, "gene_depth"])]}
    if group == "recovery":
        d = std(p1)
        return {"p1": d[d.dataset.isin(MAIN + HELD_OUT) & d.genes.isin(["top1500", "all"]) &
                        _plain(d.method).isin([PROPOSED, PROPOSED_ALL, BASE, "gene_depth", "scvi", "magic", "alra"])]}
    if group == "variants":
        d = std(p1); m = _plain(d.method)
        core = m.isin([PROPOSED, PROPOSED_ALL, BASE, "scvi"])
        return {"p1": d[d.genes.isin(["top1500", "all"]) & ~core & (m.str.contains("relabel=kmeans") | m.str.startswith("scvi"))]}
    if group == "thinning":
        d = std(p2)
        keep = [PROPOSED_ALL, f"{D};within_label_pcs=5;within_label_xfit=5;spatial_blend=auto", D, "scvi", "magic", "gene_depth"]
        return {"p2": d[d.dataset.isin(THINNING) & d.genes.isin(["top1500", "all"]) & _plain(d.method).isin(keep)]}
    if group == "unmeasured":
        d = std(p3)
        return {"p3": d[d.dataset.isin(PANELS + ["tissue:DataUploadFinal/Axolotl", "tissue:DataUploadFinal/Dataset7"]) &
                        d.method.isin(UNMEASURED)]}
    raise SystemExit(f"unknown group {group}")


def commands(sel, out, datasets=None, seeds=None, extra=()):
    """bench.runner invocations that rerun exactly the selected rows (one call per configuration and seed)."""
    cmds = []
    for proto, d in sel.items():
        if datasets:
            d = d[d.dataset.isin(datasets)]
        if seeds:
            d = d[d.seed.isin(seeds)]
        if proto == "p3":
            cfg = ["dataset", "labels", "reference", "coarse", "folds", "max_genes", "seed"]
        else:
            cfg = ["dataset", "labels", "reference", "coarse", "genes", "context", "cluster", "zero_sample", "seed",
                   "mask" if proto == "p1" else "thin"]
        for key, g in d.groupby(cfg, sort=True):
            c = dict(zip(cfg, key))
            cmd = [sys.executable, "-m", "bench.runner", proto, "--dataset", c["dataset"], "--labels", c["labels"],
                   "--reference", c["reference"], "--seeds", str(int(c["seed"])), "--out", str(out)]
            if c["coarse"]:
                cmd.append("--coarse")
            if proto == "p3":
                cmd += ["--folds", str(int(c["folds"])), "--max-genes", str(int(c["max_genes"]))]
            else:
                cmd += ["--genes", c["genes"], "--context", c["context"], "--zero-sample", str(int(c["zero_sample"]))]
                cmd += ["--mask", str(c["mask"])] if proto == "p1" else ["--thin", str(c["thin"])]
                if c["cluster"]:
                    cmd.append("--cluster")
            cmd += ["--methods", *sorted(g.method.unique()), *extra]
            cmds.append(cmd)
    return cmds


def run(a):
    out = Path(a.out)
    if a.group == "markers":
        cmd = [sys.executable, "-m", "bench.runner_markers", "--family", "kidney", "--labels", "kmeans",
               "--methods", *MARKER_METHODS, "--out", str(out / "marker_recovery.csv")]
        cmds = [cmd]
    else:
        cmds = commands(select(a.group), out, a.datasets, a.seeds, ["--force"] if a.force else [])
        cmds.append([sys.executable, "-m", "bench.runner", "aggregate", "--out", str(out)])
    print(f"{len(cmds)} commands")
    for cmd in cmds:
        print("$", " ".join(shlex.quote(x) for x in cmd[1:]), flush=True)
        if not a.dry_run:
            r = subprocess.run(cmd, cwd=ROOT)
            if r.returncode and not a.keep_going:
                raise SystemExit(r.returncode)
    if not a.dry_run and a.group != "markers":
        compare(a)


def compare(a):
    out = Path(a.out)
    if a.group == "markers":
        old = pd.read_csv(ROOT / "results/marker/marker_recovery_v2.csv"); new = pd.read_csv(out / "marker_recovery.csv")
        k = ["section", "method", "labels"]
        m = old.merge(new, on=k, suffixes=("_recorded", "_rerun"))
        for c in ("marker_auroc", "lift"):
            m[f"{c}_diff"] = m[f"{c}_rerun"] - m[f"{c}_recorded"]
        print(m[k + [c for c in m if c.endswith("_diff")]].round(4).to_string(index=False)); return
    for proto, old in select(a.group).items():
        p = out / f"{proto}_results.csv"
        if not p.exists():
            print(f"{proto}: nothing rerun yet ({p} missing)"); continue
        new = pd.read_csv(p, low_memory=False)
        for k, v in (("labels", "codi"), ("reference", "stored"), ("context", "all"), ("coarse", False)):
            new[k] = new[k].fillna(v)
        new["coarse"] = new["coarse"].astype(str) == "True"
        keys = [k for k in KEYS if k in old and k in new]
        for k in keys:                                     # same types on both sides for the join (0 and 0.0, NaN)
            old[k], new[k] = old[k].map(_key), new[k].map(_key)
        mets = [c for c in METRICS[proto] if c in old and c in new]
        m = old[keys + mets].merge(new[keys + mets], on=keys, suffixes=("_recorded", "_rerun"))
        if m.empty:
            print(f"{proto}: no rerun matches a recorded experiment of this group"); continue
        print(f"\n{proto}: {len(m)} of {len(old)} recorded experiments rerun")
        for c in mets:
            diff = (m[f"{c}_rerun"] - m[f"{c}_recorded"]).abs()
            print(f"  {c:32s} max |difference| {np.nanmax(diff.values):.2e}, rows above 1e-4: {(diff > 1e-4).sum()}")
        show = m.assign(method=m.method.str.slice(0, 60))
        cols = ["dataset", "seed", "method"] + (["fold"] if proto == "p3" else []) + [f"{mets[0]}_recorded", f"{mets[0]}_rerun"]
        print(show[cols].round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("run", "compare"):
        p = sub.add_parser(name)
        p.add_argument("group", choices=["quick", "recovery", "variants", "thinning", "unmeasured", "markers", "all"])
        p.add_argument("--out", default=str(ROOT / "results" / "rerun"))
        if name == "run":
            p.add_argument("--datasets", nargs="+"); p.add_argument("--seeds", nargs="+", type=int)
            p.add_argument("--dry-run", action="store_true"); p.add_argument("--force", action="store_true")
            p.add_argument("--keep-going", action="store_true", help="continue after a failed command")
    a = ap.parse_args()
    if a.cmd == "list":
        for g in ("quick", "recovery", "variants", "thinning", "unmeasured", "all"):
            s = select(g)
            print(f"{g:11s} " + ", ".join(f"{p}: {len(d)} experiments on {d.dataset.nunique()} sections" for p, d in s.items()))
        print("markers     5 kidney pucks x", len(MARKER_METHODS), "methods")
    elif a.cmd == "run":
        run(a)
    else:
        compare(a)


if __name__ == "__main__":
    main()
