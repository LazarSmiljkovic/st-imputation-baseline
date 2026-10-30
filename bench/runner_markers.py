"""Downstream test with labels the method never used: marker-gene specificity after imputation.

For a family of sections that share an author label vocabulary (the GSE190094 kidney pucks with
the deposit's bead cell types; the spatialLIBD DLPFC sections with their manual layers), each
section in turn is the test section.  Marker genes of every author type are derived from the
OTHER sections' observed counts (Welch t of the type against the rest, averaged over sections,
top 30 positive genes within the test section's gene panel).  The test section is imputed with
labels the scorer never sees (k-means clusters of the section, or the working labels of the
benchmark), and for every author type the AUROC of each marker gene for "spot is of that type"
is computed on the imputed matrix.  A matched set of non-specific genes (|z| < 1, same detection
decile) gives the null AUROC.  Reported per type and averaged: marker AUROC, null AUROC, their
difference (lift), and the marker detection within and outside the type.

    python -m bench.runner_markers --family dlpfc --methods none gene_depth magic scvi "ours:..."

Rows already in the output file are skipped, so the script can be rerun with more methods.
"""
from __future__ import annotations
import argparse, os, sys, time, tempfile
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bench import datasets, methods, protocols

FAMILIES = {
    "kidney": ["kidney_puck:GSM5713381", "kidney_puck:GSM5713354", "kidney_puck:GSM5713350", "kidney_puck:GSM5713388", "kidney"],
    "dlpfc": ["dlpfc:151673", "dlpfc:151676", "dlpfc:151507", "dlpfc:151669"],
}
SKIP_TYPES = {"NA", "nan", "None", ""}
ROOT = Path(__file__).resolve().parents[1]


def _lognorm(X: sp.csr_matrix) -> sp.csr_matrix:
    lib = np.asarray(X.sum(axis=1)).ravel(); lib[lib == 0] = 1.0
    Y = sp.diags((1e4 / lib).astype(np.float32)) @ X
    Y.data = np.log1p(Y.data)
    return sp.csr_matrix(Y)


def _welch_z(Y: sp.csr_matrix, mask: np.ndarray) -> np.ndarray:
    """Welch t statistic per gene (column) of spots in mask against the rest, on log-normalised values."""
    def mom(M):
        n = M.shape[0]; m1 = np.asarray(M.mean(axis=0)).ravel(); m2 = np.asarray(M.multiply(M).mean(axis=0)).ravel()
        return n, m1, np.maximum(m2 - m1 ** 2, 0.0) * n / max(n - 1, 1)
    n1, a1, v1 = mom(Y[mask]); n0, a0, v0 = mom(Y[~mask])
    return (a1 - a0) / np.sqrt(v1 / n1 + v0 / n0 + 1e-12)


def markers_from_sections(others, genes, n_markers, min_spots=30):
    """Mean Welch z per (type, gene) over the other sections -> marker sets and the non-specific pool."""
    zsum, zcnt = {}, {}
    for pr in others:
        st = pr.st; idx = pd.Index(st.var_names).get_indexer(genes); ok = idx >= 0
        X = sp.csr_matrix(st.X)[:, idx[ok]]; Y = _lognorm(X)
        lab = np.asarray(pr.author).astype(str)
        for t in np.unique(lab):
            if t in SKIP_TYPES:
                continue
            m = lab == t
            if m.sum() < min_spots or (~m).sum() < min_spots:
                continue
            z = np.full(len(genes), np.nan); z[ok] = _welch_z(Y, m)
            zsum[t] = np.nan_to_num(z) + zsum.get(t, 0.0); zcnt[t] = (~np.isnan(z)).astype(int) + zcnt.get(t, 0)
    zmean = {t: zsum[t] / np.maximum(zcnt[t], 1) for t in zsum}
    sets = {}
    for t, z in zmean.items():
        order = np.argsort(-z); top = [g for g in order[:n_markers] if z[g] > 2.0]
        if len(top) >= 10:
            sets[t] = np.array(top)
    return sets, zmean


def kmeans_labels(X: sp.csr_matrix, k=15, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    Z = TruncatedSVD(n_components=min(30, X.shape[1] - 1), random_state=seed).fit_transform(_lognorm(X))
    return np.array([f"k{c}" for c in KMeans(n_clusters=k, n_init=3, random_state=seed).fit_predict(Z)])


def score(pred: np.ndarray, author: np.ndarray, sets: dict, nulls: dict, min_spots=50):
    """Per author type: mean AUROC of its markers and of the matched null genes, and marker detection in/out of the type."""
    keep = ~np.isin(author, list(SKIP_TYPES)); P = pred[keep]; lab = author[keep]
    R = rankdata(P, axis=0); n = P.shape[0]
    rows = []
    for t, M in sets.items():
        m = lab == t; n1 = int(m.sum()); n0 = n - n1
        if n1 < min_spots or n0 < min_spots:
            continue
        def auroc(cols):
            return float(np.mean((R[m][:, cols].sum(axis=0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)))
        rows.append(dict(type=t, n_spots=n1, n_markers=len(M), marker_auroc=auroc(M), null_auroc=auroc(nulls[t]),
                         in_type_detect=float((P[m][:, M] > 0).mean()), off_type_detect=float((P[~m][:, M] > 0).mean()),
                         in_type_null_detect=float((P[m][:, nulls[t]] > 0).mean())))
    return rows


def false_markers(pred: np.ndarray, author: np.ndarray, sets: dict, zmean: dict, top=30, min_spots=50):
    """Per author type: the top genes by Welch t on the imputed test section, and the share of them that are
    not markers in the other sections (average statistic below 1), i.e. markers created by the imputation;
    also the share of the other sections' markers recovered among them."""
    keep = ~np.isin(author, list(SKIP_TYPES)); lab = author[keep]
    Y = _lognorm(sp.csr_matrix(np.maximum(pred[keep], 0)))
    out = {}
    for t, M in sets.items():
        m = lab == t
        if m.sum() < min_spots or (~m).sum() < min_spots:
            continue
        zt = _welch_z(Y, m); zt = np.nan_to_num(zt, nan=-np.inf)
        top_g = np.argsort(-zt)[:top]
        other = np.nan_to_num(zmean[t], nan=0.0)
        out[t] = dict(false_marker_share=float(np.mean(other[top_g] < 1.0)), marker_recall=float(np.mean(np.isin(M, top_g))))
    return out


def matched_nulls(sets, zmean, detect, seed=0):
    """For every marker a non-specific gene (|z| < 1 for that type) from the same detection decile."""
    rng = np.random.default_rng(seed)
    dec = np.digitize(detect, np.quantile(detect, np.linspace(0.1, 0.9, 9)))
    nulls = {}
    all_g = np.arange(len(detect))
    for t, M in sets.items():
        z = np.nan_to_num(zmean[t], nan=0.0, posinf=99.0, neginf=-99.0)
        pool_ok = np.abs(z) < 1.0
        if pool_ok.sum() < len(M):                      # a type that separates almost every gene: take the least specific genes
            pool_ok = np.zeros_like(pool_ok); pool_ok[np.argsort(np.abs(z))[:max(3 * len(M), 60)]] = True
        chosen = []
        for g in M:
            cand = np.flatnonzero(pool_ok & (dec == dec[g]) & ~np.isin(all_g, chosen + list(M)))
            if cand.size == 0:
                cand = np.flatnonzero(pool_ok & ~np.isin(all_g, chosen + list(M)))
            if cand.size == 0:
                cand = np.flatnonzero(~np.isin(all_g, chosen + list(M)))
            chosen.append(int(rng.choice(cand)))
        nulls[t] = np.array(chosen)
    return nulls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=["kmeans"], help="labels handed to the methods: kmeans and/or working")
    ap.add_argument("--sections", nargs="*", default=None, help="test sections (default: every section of the family)")
    ap.add_argument("--markers", type=int, default=30)
    ap.add_argument("--genes", default="top1500")
    ap.add_argument("--out", default=str(ROOT / "results" / "marker" / "marker_recovery_v2.csv"))
    a = ap.parse_args()
    out = Path(a.out); out_types = out.with_name(out.stem + "_per_type.csv")
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["family", "section", "method", "labels"])
    names = FAMILIES[a.family]
    print(f"[load] {names}", flush=True)
    pairs = {n: datasets.load(n) for n in names}
    tests = a.sections or names
    for n in tests:
        pr = pairs[n]; st = pr.st
        st_all = sp.csr_matrix(st.X); panel = protocols.gene_panel(st_all, a.genes, seed=0)
        X = st_all[:, panel].tocsr(); genes = np.asarray(st.var_names)[panel]
        author = np.asarray(pr.author).astype(str)
        others = [pairs[m] for m in names if m != n]
        sets, zmean = markers_from_sections(others, genes, a.markers)
        common = [t for t in sets if (author == t).sum() >= 50]
        sets = {t: sets[t] for t in common}
        detect = np.asarray((X > 0).mean(axis=0)).ravel()
        nulls = matched_nulls(sets, zmean, detect)
        thin = {t: int((np.abs(np.nan_to_num(zmean[t])) < 1.0).sum()) for t in sets}
        print(f"[{n}] non-specific pool per type (|z|<1): {thin}", flush=True)
        print(f"[{n}] {X.shape[0]} spots x {X.shape[1]} genes; {len(sets)} author types with markers: {sorted(sets)}", flush=True)
        lab_sets = {}
        if "kmeans" in a.labels:
            lab_sets["kmeans"] = kmeans_labels(X, 15, 0)
        if "working" in a.labels:
            lab_sets["working"] = np.asarray(pr.st_label).astype(str)
        sc_csr = sp.csr_matrix(pr.sc.X)
        sc_idx = pd.Index(pr.sc.var_names).get_indexer(genes)
        sc_csr = sc_csr[:, np.where(sc_idx >= 0, sc_idx, 0)].tocsr() if (sc_idx >= 0).any() else sp.csr_matrix((pr.sc.n_obs, len(genes)), dtype=np.float32)
        sc_labels = pr.sc.obs[pr.sc_label].astype(str).values
        for spec in a.methods:
            for lname, labels in lab_sets.items():
                if spec in ("none", "gene_depth", "magic", "scvi") and lname != list(lab_sets)[0]:
                    continue  # label-free methods are scored once
                if ((done.section == n) & (done.method == spec) & (done.labels == lname)).any():
                    print(f"  [skip] {spec} ({lname})", flush=True); continue
                t0 = time.time()
                with tempfile.TemporaryDirectory(prefix="markers_") as work:
                    try:
                        pred = methods.denoise(spec, X, sc_csr, sc_labels, labels, pr.coords, pr.meta.get("sc_space", "auto"), 0, Path(work),
                                               st_genes=list(genes), sc_genes=list(genes), aux=dict(probs=None, parents=None))
                    except Exception as e:  # noqa
                        print(f"  [fail] {spec} ({lname}): {type(e).__name__}: {e}", flush=True); continue
                pred = np.asarray(pred, dtype=np.float32)
                rows = score(pred, author, sets, nulls)
                fm = false_markers(pred, author, sets, zmean)
                for r_ in rows:
                    r_.update(fm.get(r_["type"], dict(false_marker_share=np.nan, marker_recall=np.nan)))
                if not rows:
                    print(f"  [none] {spec}: no scorable type", flush=True); continue
                df = pd.DataFrame(rows); df.insert(0, "labels", lname); df.insert(0, "method", spec); df.insert(0, "section", n); df.insert(0, "family", a.family)
                df.to_csv(out_types, mode="a", header=not out_types.exists(), index=False)
                summ = dict(family=a.family, section=n, method=spec, labels=lname, n_types=len(df), n_spots=int(X.shape[0]), n_genes=int(X.shape[1]),
                            marker_auroc=df.marker_auroc.mean(), null_auroc=df.null_auroc.mean(), lift=(df.marker_auroc - df.null_auroc).mean(),
                            in_type_detect=df.in_type_detect.mean(), off_type_detect=df.off_type_detect.mean(), in_type_null_detect=df.in_type_null_detect.mean(),
                            false_marker_share=df.false_marker_share.mean(), marker_recall=df.marker_recall.mean(),
                            fill_frac=float((pred > 0).mean()), wall_s=round(time.time() - t0, 1))
                pd.DataFrame([summ]).to_csv(out, mode="a", header=not out.exists(), index=False)
                done = pd.concat([done, pd.DataFrame([summ])], ignore_index=True)
                print(f"  {spec} ({lname}): marker AUROC {summ['marker_auroc']:.3f} null {summ['null_auroc']:.3f} lift {summ['lift']:.3f} "
                      f"in-type detect {summ['in_type_detect']:.2f} off {summ['off_type_detect']:.2f} [{summ['wall_s']}s]", flush=True)


if __name__ == "__main__":
    main()
