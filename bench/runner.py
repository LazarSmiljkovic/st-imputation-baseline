"""
bench/runner.py -- run and record benchmark experiments.

    python -m bench.runner p1 --dataset kidney --genes top1500 --mask 0.2 --seeds 42 43 \
        --methods none scrna_mean spatial_avg spatial_matched ours:corrected ours:v3
    python -m bench.runner p2 --dataset kidney --genes top1500 --thin 0.5 --seeds 42 --methods ...
    python -m bench.runner p3 --dataset tissue:<folder> --folds 10 --seeds 0 --methods ours:v3 spage
    python -m bench.runner rescore                 # recompute metrics.json of every run from its saved predictions
    python -m bench.runner aggregate --out results/bench

Every run writes results/bench/runs/<run_id>/{meta.json,metrics.json,...}.  The
run id hashes the experimental payload *including a hash of the code that
produced the prediction* (engine file for ours:*, shim files for external
methods, methods.py for the controls), so a code change never silently reuses
a stale prediction and never overwrites an old one.  ``aggregate`` keeps, for
every logical experiment, only the most recent run and lists the superseded
ones; nothing is deleted.
"""
from __future__ import annotations

import os
# cap BLAS threads before numpy loads; several concurrent runs otherwise overheat the server
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("ZONEIMPUTE_THREADS", "4"))

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from bench import datasets, protocols, metrics, methods  # noqa: E402
from zone_impute import neighbor_homogeneity  # noqa: E402

# keys that define a logical experiment (everything in the payload except the code hash)
LOGICAL_KEYS = ["protocol", "dataset", "labels", "reference", "coarse", "genes", "mask", "thin", "context",
                "folds", "fold", "max_genes", "seed", "method"]
P3_TRUTH_TARGET = 1e4          # library size that the P3 truth and every harmonised prediction are scaled to


def _git_sha():
    try:
        sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "-C", str(ROOT), "diff", "--quiet"]) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _versions():
    out = {"python": platform.python_version()}
    for m in ("numpy", "scipy", "sklearn", "scanpy", "anndata", "pandas"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            pass
    return out


def _zones(h):
    z = np.full(h.shape, "transitional", dtype=object)
    z[h >= 0.8] = "central"; z[h < 0.5] = "peripheral"
    return z


def _run_id(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _now():
    """Millisecond timestamps, so that runs of the same experiment made seconds apart order correctly."""
    import datetime
    return datetime.datetime.now().isoformat(timespec="milliseconds")


def _write_meta(rdir, payload, rid, pair, spec, extra=None, status="ok"):
    """Written when a run starts (status running), fails (failed, with the error) and finishes (ok),
    so every folder, including a failed one, says what it was."""
    cfg = methods.parse_ours(spec).to_dict() if spec.startswith("ours:") else None
    device = "cpu (forced)" if os.environ.get("ZONEIMPUTE_FORCE_CPU", "0") == "1" else "gpu if the method uses one"
    (rdir / "meta.json").write_text(json.dumps(dict(payload, run_id=rid, git=_git_sha(), versions=_versions(), device=device,
                                                    dataset_meta=pair.meta, config=cfg, status=status, **(extra or {}),
                                                    time=_now()), indent=1, default=str))


# ----------------------------------------------------------------------------- P1 / P2

class _RecoveryContext:
    """Dataset-level quantities shared by every P1/P2 run (and by rescoring)."""

    def __init__(self, pair, genes_mode):
        self.pair = pair
        # both sides on the genes they share, in the section's order (a no-op for pairs intersected on load)
        st, sc = pair.st, pair.sc
        if st.n_vars != sc.n_vars or not (st.var_names == sc.var_names).all():
            shared = st.var_names[st.var_names.isin(sc.var_names)]
            st, sc = st[:, shared], sc[:, shared]
        self.all_genes = st.var_names
        self.st_all = sp.csr_matrix(st.X)
        self.panel = protocols.gene_panel(self.st_all, genes_mode, seed=0)
        self.st_csr = self.st_all[:, self.panel].tocsr()
        self.sc_full = sp.csr_matrix(sc.X)
        self.sc_csr = self.sc_full[:, self.panel].tocsr()
        self.sc_labels = pair.sc.obs[pair.sc_label].astype(str).values
        self.st_labels = pair.st_label
        self.coords = pair.coords
        self.genes = self.all_genes[self.panel]
        self.sc_space = pair.meta.get("sc_space", "auto")
        self.zones_label = _zones(neighbor_homogeneity(self.st_labels, self.coords, 10))
        self.zones_author = _zones(neighbor_homogeneity(pair.author, self.coords, 10)) if pair.author is not None else None
        self.imputable = np.isin(self.st_labels, np.unique(self.sc_labels))
        self.aux = dict(probs=pair.st_probs, parents=pair.parents)

    def hide(self, protocol, seed, mask=None, thin=None):
        """Returns masked panel matrix, hidden (rows, cols, truth) and the context matrix
        (every gene of the dataset with the hidden information removed)."""
        if protocol == "p1":
            masked, rows, cols, truth = protocols.mask_entries(self.st_csr, mask, seed)
            full = self.st_all.tocoo()
            hidden_keys = rows.astype(np.int64) * self.st_all.shape[1] + self.panel[cols].astype(np.int64)
            keys = full.row.astype(np.int64) * self.st_all.shape[1] + full.col.astype(np.int64)
            keep = ~np.isin(keys, hidden_keys)
            ctx = sp.csr_matrix((full.data[keep], (full.row[keep], full.col[keep])), shape=self.st_all.shape)
        else:
            # thin the whole matrix once, so the context genes carry the same reduced depth as the panel
            ctx, _ = protocols.downsample(self.st_all, thin, seed)
            masked = ctx[:, self.panel].tocsr()
            o = self.st_csr.tocoo(); m = masked.toarray()
            rows, cols, truth = o.row, o.col, o.data.astype(np.float64)
            keep = m[rows, cols] == 0          # entries that became zero: the dropout analogue
            rows, cols, truth = rows[keep], cols[keep], truth[keep]
        return masked, rows, cols, truth, ctx

    def heldout_scores(self, masked, rows, truth, pred, zr, zs, thin):
        """Proper scores for the thinning protocol.  The prediction, moved to the scale of the
        held-out molecules by (1 - p) / p, is compared with them at every entry that is zero
        after thinning: the thinned-away non-zeros and the true zeros (a sample, reweighted to
        their real number).  Squared error for every method; the mean Poisson deviance only
        for methods that predict a positive value everywhere, since it is infinite at a zero
        prediction for a positive count."""
        n_spots, n_genes = masked.shape
        true_zero = n_genes - np.diff(masked.tocsr().indptr) - np.bincount(rows, minlength=n_spots)
        okh, okz = self.imputable[rows], self.imputable[zr]
        if okz.sum() == 0 or okh.sum() == 0:
            return {}
        n_zero = float(true_zero[self.imputable].sum()); wz = n_zero / okz.sum()
        f = (1.0 - thin) / thin
        t = truth[okh]; bh = np.maximum(f * pred[okh], 0.0); bz = np.maximum(f * zs[okz], 0.0)
        n_all = okh.sum() + n_zero
        out = dict(heldout_mse=float((np.sum((bh - t) ** 2) + wz * np.sum(bz ** 2)) / n_all),
                   heldout_mean_truth=float(t.sum() / n_all),
                   heldout_mean_pred=float((bh.sum() + wz * bz.sum()) / n_all))
        if (bh > 0).mean() > 0.99:
            b = np.maximum(bh, 1e-8)
            out["heldout_deviance"] = float((np.sum(2.0 * (t * np.log(t / b) - (t - b))) + wz * np.sum(2.0 * bz)) / n_all)
        else:
            out["heldout_deviance"] = float("nan")
        return out

    @staticmethod
    def depth_strata(masked, n=10):
        """Depth group of every spot: deciles of the library size of the matrix a method is given."""
        from scipy.stats import rankdata
        lib = np.asarray(masked.sum(axis=1)).ravel()
        return np.minimum((rankdata(lib, method="average") / (lib.size + 1) * n).astype(int), n - 1)

    def score(self, rows, cols, truth, pred, zs, strata=None):
        ok = self.imputable[rows]
        m_all, per = metrics.masked_metrics(truth[ok], pred[ok], cols[ok], zones=self.zones_label[rows][ok], zero_scores=zs,
                                            strata=None if strata is None else strata[rows][ok])
        if self.zones_author is not None:
            m_auth, _ = metrics.masked_metrics(truth[ok], pred[ok], cols[ok], zones=self.zones_author[rows][ok])
            for k, v in m_auth.items():
                if k.startswith("spearman_zone_") or k.startswith("n_zone_"):
                    m_all[k.replace("zone_", "authorzone_")] = v
        m_all["n_hidden_total"] = int(truth.size)
        m_all["n_hidden_imputable"] = int(ok.sum())
        return m_all, per


def _payload_recovery(args, seed, spec, proto):
    payload = dict(dataset=args.dataset, labels=args.labels, reference=args.reference, genes=args.genes,
                   seed=seed, method=spec, coarse=bool(getattr(args, "coarse", False)), **proto)
    if getattr(args, "context", "all") != "all":
        payload["context"] = args.context
    if getattr(args, "cluster", False):
        payload["cluster"] = True
    code = methods.code_hash(spec)
    if code:
        payload["code"] = code
    return payload


def run_recovery(args):
    out_root = Path(args.out); runs = out_root / "runs"; runs.mkdir(parents=True, exist_ok=True)
    pair = datasets.load(args.dataset, **_load_kw(args))
    ctx = _RecoveryContext(pair, args.genes)

    for seed in args.seeds:
        if args.protocol == "p1":
            proto = dict(protocol="p1", mask=args.mask)
            masked, rows, cols, truth, st_ctx = ctx.hide("p1", seed, mask=args.mask)
        else:
            proto = dict(protocol="p2", thin=args.thin)
            masked, rows, cols, truth, st_ctx = ctx.hide("p2", seed, thin=args.thin)
        zr, zc = protocols.zero_sample(masked, rows, cols, args.zero_sample, seed)
        strata = ctx.depth_strata(masked, args.depth_groups)
        embed_X = (ctx.sc_full, st_ctx) if (args.genes != "all" and args.context == "all") else None

        for spec in args.methods:
            payload = _payload_recovery(args, seed, spec, proto)
            rid = _run_id(payload)
            rdir = runs / rid
            if (rdir / "metrics.json").exists() and not args.force:
                print(f"[skip] {spec} seed={seed} -> {rid}"); continue
            rdir.mkdir(parents=True, exist_ok=True)
            _write_meta(rdir, payload, rid, pair, spec, status="running")
            print(f"[run ] {spec} seed={seed} -> {rid}", flush=True)
            t0 = time.time(); rss0 = _peak_rss_gb()
            try:
                pred_full = methods.denoise(spec, masked, ctx.sc_csr, ctx.sc_labels, ctx.st_labels, ctx.coords, ctx.sc_space,
                                            seed, rdir / "work", st_genes=ctx.genes, sc_genes=ctx.genes, embed_X=embed_X, aux=ctx.aux)
            except Exception as e:  # keep going with the other methods, record the failure
                (rdir / "error.txt").write_text(repr(e)); print("   failed:", e)
                _write_meta(rdir, payload, rid, pair, spec, status="failed", extra=dict(error=repr(e))); continue
            wall = time.time() - t0
            pred = pred_full[rows, cols].astype(np.float64)
            zs = pred_full[zr, zc].astype(np.float64)
            m_all, per = ctx.score(rows, cols, truth, pred, zs, strata=strata)
            if args.protocol == "p2":
                m_all.update(ctx.heldout_scores(masked, rows, truth, pred, zr, zs, args.thin))
            if args.cluster and pair.author is not None:
                try:
                    m_all.update({f"cluster_{k}": v for k, v in metrics.clustering_metrics(pred_full, pair.author, seed=seed).items()})
                except Exception as e:
                    m_all["cluster_error"] = repr(e)
            elif args.cluster:
                m_all["cluster_error"] = "no author labels for this dataset"
            m_all.update(dict(wall_s=round(wall, 1), peak_rss_gb=round(max(_peak_rss_gb() - rss0, 0), 2),
                              n_spots=int(ctx.st_csr.shape[0]), n_genes=int(ctx.st_csr.shape[1])))
            (rdir / "metrics.json").write_text(json.dumps(m_all, indent=1))
            (rdir / "error.txt").unlink(missing_ok=True)
            _write_meta(rdir, payload, rid, pair, spec)
            np.savez_compressed(rdir / "hidden_pred.npz", rows=rows, cols=cols, truth=truth, pred=pred.astype(np.float32),
                                zero_rows=zr, zero_cols=zc, zero_pred=zs.astype(np.float32))
            per.to_csv(rdir / "per_gene.csv", index=False)
            g = lambda k: m_all.get(k, float("nan"))          # a progress line must never be able to stop a batch
            print(f"   spearman_all={g('spearman_all'):.3f} pergene_wmean={g('pergene_spearman_wmean'):.3f} "
                  f"pergene_med={g('pergene_spearman_median'):.3f} fill={g('hidden_fill_frac'):.3f} "
                  f"auroc={g('zero_select_auroc'):.3f} dens={g('densification_at_zeros'):.3f} [{wall:.0f}s]", flush=True)


# ----------------------------------------------------------------------------- P3

def _harmonise(pred, scale, lib, ref_lib_mean=None):
    """Bring a prediction onto the truth scale log1p(count / library * 1e4).
    counts: count-like values at each spot's own depth (divided by that spot's
    library); counts_common: count-like values at one typical depth (divided by
    the median library, so no per-spot pattern is imprinted); freq: per-spot
    frequencies (sum to 1); log: already log-normalised; ref_counts: count-like
    values at the reference depth (legacy type-mean control)."""
    pred = np.asarray(pred, np.float64)
    pred = np.where(np.isfinite(pred), pred, 0.0)       # a non-finite prediction counts as "nothing predicted"
    if scale == "log":
        return pred
    if scale == "freq":
        return np.log1p(np.maximum(pred, 0) * P3_TRUTH_TARGET)
    if scale == "counts_common":
        lib = np.asarray(lib, np.float64)
        return np.log1p(np.maximum(pred, 0) * P3_TRUTH_TARGET / max(float(np.median(lib[lib > 0])), 1.0))
    if scale == "ref_counts":
        return np.log1p(np.maximum(pred, 0) * P3_TRUTH_TARGET / max(float(ref_lib_mean or 1.0), 1.0))
    if scale == "counts":
        lib = np.asarray(lib, np.float64); lib = np.where(lib > 0, lib, 1.0)
        return np.log1p(np.maximum(pred, 0) * P3_TRUTH_TARGET / lib[:, None])
    raise ValueError(f"unknown prediction scale {scale!r}")


class _PredictionContext:
    def __init__(self, pair, max_genes):
        self.pair = pair
        self.st_all = pair.st; self.sc_all = pair.sc
        shared = self.st_all.var_names.intersection(self.sc_all.var_names)
        if max_genes and len(shared) > max_genes:
            rng = np.random.default_rng(0)
            shared = pd.Index(sorted(rng.choice(np.asarray(shared), max_genes, replace=False)))
        self.shared_all = shared
        self.sc_labels = self.sc_all.obs[pair.sc_label].astype(str).values
        self.st_labels = pair.st_label; self.coords = pair.coords
        self.sc_space = pair.meta.get("sc_space", "auto")
        import scanpy as sc
        st_norm = self.st_all[:, shared].copy()
        self.lib = np.asarray(sp.csr_matrix(st_norm.X).sum(axis=1)).ravel()
        sc.pp.normalize_total(st_norm, target_sum=P3_TRUTH_TARGET); sc.pp.log1p(st_norm)
        self.truth_all = sp.csr_matrix(st_norm.X); self.truth_genes = st_norm.var_names
        sc_lin = methods._ref_linear(sp.csr_matrix(self.sc_all[:, shared].X), self.sc_space)
        self.ref_lib_mean = float(np.asarray(sc_lin.sum(axis=1)).mean())
        self.transferred = pair.meta.get("label_source") == "reference-leiden-transfer"
        self.rest = self.sc_all.var_names.difference(shared)

    def truth(self, held):
        return self.truth_all[:, self.truth_genes.get_indexer(held)].toarray()

    def lib_kept(self, held):
        """Per-spot library over the genes the method saw (shared minus held), so that a prediction
        harmonised by each spot's library carries no information from the held-out genes."""
        held_idx = self.truth_genes.get_indexer(held)
        X = sp.csr_matrix(self.st_all[:, self.shared_all].X)
        return self.lib - np.asarray(X[:, held_idx].sum(axis=1)).ravel()

    def labels_for(self, kept, seed=0):
        """Spot labels for a fold.  Transferred labels are recomputed from the kept
        genes only, so the held-out genes never inform the conditioning labels."""
        if not self.transferred:
            return self.st_labels
        return datasets._transfer_labels(self.sc_all[:, kept], self.st_all[:, kept], self.sc_labels, seed=seed)


def _score_prediction(truth, pred_log, held):
    per = metrics.gene_prediction_metrics(truth, pred_log, held)
    summ = {}
    for c in ("pcc", "spearman", "ssim", "rmse_z", "js"):
        v = per[c].values
        summ[f"{c}_median"] = float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")
        summ[f"{c}_mean"] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
        summ[f"{c}_n"] = int(np.isfinite(v).sum())
    summ["n_genes"] = int(len(per))
    return summ, per


def run_prediction(args):
    out_root = Path(args.out); runs = out_root / "runs"; runs.mkdir(parents=True, exist_ok=True)
    pair = datasets.load(args.dataset, **_load_kw(args))
    ctx = _PredictionContext(pair, args.max_genes)

    for seed in args.seeds:
        for fi, (held, kept) in enumerate(protocols.gene_folds(ctx.shared_all, args.folds, seed)):
            if args.max_folds and fi >= args.max_folds:
                break
            st_kept = sp.csr_matrix(ctx.st_all[:, kept].X)
            sc_genes = list(kept) + list(held) + list(ctx.rest)          # shared, then targets, then the rest of the reference
            sc_ordered = sp.csr_matrix(ctx.sc_all[:, sc_genes].X)
            truth = ctx.truth(held)
            st_labels = None
            for spec in args.methods:
                payload = dict(dataset=args.dataset, labels=args.labels, reference=args.reference,
                               coarse=bool(getattr(args, "coarse", False)), protocol="p3", folds=args.folds, fold=fi,
                               seed=seed, method=spec, max_genes=args.max_genes)
                code = methods.code_hash(spec)
                if code:
                    payload["code"] = code
                rid = _run_id(payload); rdir = runs / rid
                if (rdir / "metrics.json").exists() and not args.force:
                    print(f"[skip] {spec} fold={fi} -> {rid}"); continue
                if st_labels is None:
                    st_labels = ctx.labels_for(kept, seed=0)
                rdir.mkdir(parents=True, exist_ok=True)
                _write_meta(rdir, payload, rid, pair, spec, status="running", extra=dict(labels_per_fold=ctx.transferred))
                print(f"[run ] {spec} seed={seed} fold={fi} ({len(held)} genes) -> {rid}", flush=True)
                t0 = time.time()
                try:
                    pred, scale = methods.predict(spec, st_kept, sc_ordered, ctx.sc_labels, st_labels, ctx.coords, list(kept),
                                                  list(held), ctx.sc_space, seed, rdir / "work", sc_genes=sc_genes)
                except Exception as e:
                    (rdir / "error.txt").write_text(repr(e)); print("   failed:", e)
                    _write_meta(rdir, payload, rid, pair, spec, status="failed", extra=dict(error=repr(e))); continue
                wall = time.time() - t0
                n_bad = int((~np.isfinite(pred)).sum())
                pred_log = _harmonise(pred, scale, ctx.lib_kept(held), ctx.ref_lib_mean)
                summ, per = _score_prediction(truth, pred_log, held)
                summ.update(metrics.cellwise_spearman(truth, pred_log))
                summ.update(dict(wall_s=round(wall, 1), pred_scale=scale, n_nonfinite_pred=n_bad))
                (rdir / "metrics.json").write_text(json.dumps(summ, indent=1))
                (rdir / "error.txt").unlink(missing_ok=True)
                _write_meta(rdir, payload, rid, pair, spec, extra=dict(pred_scale=scale, labels_per_fold=ctx.transferred))
                per.to_csv(rdir / "per_gene.csv", index=False)
                np.savez_compressed(rdir / "pred.npz", pred=pred.astype(np.float32), held=np.asarray(held, dtype=str),
                                    scale=np.array(scale))
                print(f"   pcc_med={summ.get('pcc_median', float('nan')):.3f} spearman_med={summ.get('spearman_median', float('nan')):.3f} "
                      f"ssim_med={summ.get('ssim_median', float('nan')):.3f} scale={scale} [{wall:.0f}s]", flush=True)


# ----------------------------------------------------------------------------- rescore

_LEGACY_SCALE = {"spage": "log", "stplus": "log", "tangram": "log", "gimvi": "freq", "scrna_mean": "ref_counts", "ours": "counts_common"}


def rescore(args):
    """Recompute metrics.json for every run from its saved predictions with the
    current metric code (datasets are reloaded once per configuration).  Values
    that need the full prediction matrix (clustering) are kept from the old file."""
    runs = Path(args.out) / "runs"
    groups = {}
    for d in sorted(runs.glob("*")):
        if not (d / "meta.json").exists() or not (d / "metrics.json").exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        key = (meta["dataset"], meta.get("labels", "codi"), meta.get("reference", "stored"), bool(meta.get("coarse", False)),
               meta["protocol"], meta.get("genes"), meta.get("max_genes", 0))
        groups.setdefault(key, []).append((d, meta))
    n_done = 0
    for key, items in groups.items():
        dataset, labels, reference, coarse, proto, genes, max_genes = key
        if args.dataset and dataset != args.dataset:
            continue
        if getattr(args, "only", "") and proto != args.only:
            continue
        print(f"[rescore] {dataset} labels={labels} ref={reference} coarse={coarse} {proto} ({len(items)} runs)", flush=True)
        ns = argparse.Namespace(dataset=dataset, labels=labels, reference=reference, coarse=coarse)
        try:
            pair = datasets.load(dataset, **_load_kw(ns))
        except Exception as e:
            print("   cannot load dataset:", e); continue
        if proto in ("p1", "p2"):
            ctx = _RecoveryContext(pair, genes)
            hidden = {}                                      # (seed, level) -> matrix the methods were given, rebuilt once
            for d, meta in items:
                if not (d / "hidden_pred.npz").exists():     # a run still being written by a concurrent runner
                    continue
                z = np.load(d / "hidden_pred.npz")
                old = json.loads((d / "metrics.json").read_text())
                key = (meta["seed"], meta.get("thin", 0.5) if proto == "p2" else meta.get("mask", 0.2))
                if key not in hidden:
                    hidden[key] = ctx.hide(proto, key[0], thin=key[1]) if proto == "p2" else ctx.hide(proto, key[0], mask=key[1])
                masked, rows_now = hidden[key][0], hidden[key][1]
                same = rows_now.size == z["rows"].size and np.array_equal(rows_now, z["rows"])   # false for runs of an older protocol
                m_all, per = ctx.score(z["rows"], z["cols"], z["truth"].astype(np.float64), z["pred"].astype(np.float64),
                                       z["zero_pred"].astype(np.float64), strata=ctx.depth_strata(masked, args.depth_groups) if same else None)
                if proto == "p2":
                    if same:
                        m_all.update(ctx.heldout_scores(masked, z["rows"], z["truth"].astype(np.float64), z["pred"].astype(np.float64),
                                                        z["zero_rows"], z["zero_pred"].astype(np.float64), key[1]))
                for k in ("wall_s", "peak_rss_gb", "n_spots", "n_genes"):
                    if k in old:
                        m_all[k] = old[k]
                m_all.update({k: v for k, v in old.items() if k.startswith("cluster_")})
                (d / "metrics.json").write_text(json.dumps(m_all, indent=1)); per.to_csv(d / "per_gene.csv", index=False)
                n_done += 1
        else:
            ctx = _PredictionContext(pair, max_genes)
            for d, meta in items:
                if not (d / "pred.npz").exists():            # a run still being written by a concurrent runner
                    continue
                z = np.load(d / "pred.npz")
                held = list(z["held"].astype(str))
                scale = str(z["scale"]) if "scale" in z.files else _LEGACY_SCALE.get(methods.method_family(meta["method"]), "counts")
                if methods.method_family(meta["method"]) == "ours" and scale == "counts":
                    scale = "counts_common"          # runs saved before the engine's footing was named correctly
                old = json.loads((d / "metrics.json").read_text())
                pred_log = _harmonise(z["pred"], scale, ctx.lib_kept(held), ctx.ref_lib_mean)
                summ, per = _score_prediction(ctx.truth(held), pred_log, held)
                summ.update(metrics.cellwise_spearman(ctx.truth(held), pred_log))
                summ.update(dict(wall_s=old.get("wall_s"), pred_scale=scale))
                (d / "metrics.json").write_text(json.dumps(summ, indent=1)); per.to_csv(d / "per_gene.csv", index=False)
                meta["pred_scale"] = scale; (d / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
                n_done += 1
    print(f"rescored {n_done} runs")


# ----------------------------------------------------------------------------- aggregate

def _logical_frame(df):
    """Fill defaults for payload keys that older runs did not record."""
    for k, v in (("labels", "codi"), ("reference", "stored"), ("coarse", False), ("context", "all"), ("max_genes", 0)):
        if k not in df:
            df[k] = v
        df[k] = df[k].fillna(v)
    return df


def aggregate(args):
    runs = Path(args.out) / "runs"
    rows = []
    for d in sorted(runs.glob("*")):
        if not (d / "metrics.json").exists():
            continue
        meta = json.loads((d / "meta.json").read_text()); met = json.loads((d / "metrics.json").read_text())
        row = {k: v for k, v in meta.items() if k not in ("versions", "config", "dataset_meta")}
        row["label_source"] = (meta.get("dataset_meta") or {}).get("label_source")
        row["n_shared"] = (meta.get("dataset_meta") or {}).get("n_shared")
        row.update(met)
        rows.append(row)
    if not rows:
        print("no runs"); return
    df = _logical_frame(pd.DataFrame(rows))
    df["time"] = df["time"].fillna("")
    keys = [k for k in LOGICAL_KEYS if k in df]
    df = df.sort_values("time")
    dup = df.duplicated(subset=keys, keep="last")
    if dup.any():
        sup = df[dup][keys + ["run_id", "code", "time"] if "code" in df else keys + ["run_id", "time"]]
        sup.to_csv(Path(args.out) / "superseded_runs.csv", index=False)
        print(f"{dup.sum()} superseded runs listed in superseded_runs.csv (kept on disk, excluded from tables)")
    df = df[~dup]
    for proto, sub in df.groupby("protocol"):
        p = Path(args.out) / f"{proto}_results.csv"
        sub = sub.sort_values(["dataset", "method", "seed"])
        sub.to_csv(p, index=False)
        print(f"wrote {p} ({len(sub)} runs)")
        key = [k for k in keys if k not in ("seed", "fold") and k in sub and sub[k].notna().any()]
        num = sub.select_dtypes("number").columns.difference(["seed", "fold"])
        summary = sub.groupby(key, dropna=False)[num].agg(["mean", "std", "count"]).round(4)
        summary.to_csv(Path(args.out) / f"{proto}_summary.csv")
        if proto == "p3":
            cfg_keys = [k for k in ("dataset", "labels", "reference", "coarse", "folds", "max_genes") if k in sub]
            for cfg, sds in sub.groupby(cfg_keys, dropna=False):
                cfg = dict(zip(cfg_keys, cfg if isinstance(cfg, tuple) else (cfg,)))
                per = {}
                for m, smd in sds.groupby("method"):
                    frames = [pd.read_csv(runs / r / "per_gene.csv") for r in smd["run_id"]]
                    per[m] = pd.concat(frames)
                if len(per) > 1:
                    acc = metrics.accuracy_score(per)
                    tag = cfg["dataset"].replace(":", "_").replace("/", "_")
                    if cfg.get("labels", "codi") != "codi": tag += f"_labels-{cfg['labels']}"
                    if cfg.get("coarse"): tag += "_coarse"
                    if cfg.get("folds", 5) not in (5, None): tag += f"_folds{int(cfg['folds'])}"
                    acc.to_csv(Path(args.out) / f"p3_accuracy_score_{tag}.csv")
                    print(f"accuracy score {tag} (dropped for partial coverage: {acc.attrs.get('dropped_partial_coverage')}):\n"
                          f"{acc.sort_values('accuracy_score', ascending=False).to_string()}")


# ----------------------------------------------------------------------------- CLI

def _load_kw(args):
    kw = {}
    if args.dataset in ("kidney", "visium", "embryo", "embryo_raw", "adultB5"):
        kw["label_source"] = args.labels
    if args.dataset == "kidney":
        kw["reference"] = args.reference
    if args.dataset in ("embryo", "embryo_raw", "adultB5") and getattr(args, "coarse", False):
        kw["coarse"] = True
    return kw


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="protocol", required=True)
    for name in ("p1", "p2"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True); p.add_argument("--labels", default="codi")
        p.add_argument("--reference", default="stored"); p.add_argument("--genes", default="top1500")
        p.add_argument("--mask", type=float, default=0.2); p.add_argument("--thin", type=float, default=0.5)
        p.add_argument("--seeds", type=int, nargs="+", default=[42]); p.add_argument("--methods", nargs="+", required=True)
        p.add_argument("--zero-sample", type=int, default=1_000_000); p.add_argument("--cluster", action="store_true")
        p.add_argument("--coarse", action="store_true", help="map Stereo-seq subclass labels to the 7 reference classes")
        p.add_argument("--depth-groups", type=int, default=10, help="depth groups of spots for the depth-stratified per-gene Spearman")
        p.add_argument("--context", choices=["all", "panel"], default="all",
                       help="genes available to the conditional embedding: every measured gene (minus hidden entries) or the panel only")
        p.add_argument("--out", default=str(ROOT / "results" / "bench")); p.add_argument("--force", action="store_true")
    p = sub.add_parser("p3")
    p.add_argument("--dataset", required=True); p.add_argument("--labels", default="codi"); p.add_argument("--reference", default="stored")
    p.add_argument("--coarse", action="store_true")
    p.add_argument("--folds", type=int, default=10); p.add_argument("--max-folds", type=int, default=0)
    p.add_argument("--max-genes", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0]); p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--out", default=str(ROOT / "results" / "bench")); p.add_argument("--force", action="store_true")
    p = sub.add_parser("rescore"); p.add_argument("--out", default=str(ROOT / "results" / "bench")); p.add_argument("--dataset", default="")
    p.add_argument("--only", default="", choices=["", "p1", "p2", "p3"], help="rescore one protocol only")
    p.add_argument("--depth-groups", type=int, default=10)
    p = sub.add_parser("aggregate"); p.add_argument("--out", default=str(ROOT / "results" / "bench"))
    args = ap.parse_args(argv)
    if args.protocol in ("p1", "p2"):
        run_recovery(args)
    elif args.protocol == "p3":
        run_prediction(args)
    elif args.protocol == "rescore":
        rescore(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
