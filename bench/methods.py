"""
bench/methods.py -- one interface for every method.

    denoise(spec, st_csr, sc_csr, sc_labels, st_labels, coords, sc_space, seed, work)
        -> dense float32 (spots x genes) with zeros filled / values denoised
    predict(spec, st_kept_csr, sc_all_csr, sc_labels, st_labels, coords, shared, held, sc_space, seed, work)
        -> (dense float32 (spots x len(held)) predictions for genes absent from ST, scale)

Method specs
    none                        the observed matrix
    scrna_mean                  zeros <- per-type reference non-zero mean, rescaled
    gene_depth                  zeros <- spot size factor x the gene's mean rate (depth only)
    spatial_avg                 zeros <- distance-weighted neighbour average
    spatial_matched             spatial_avg but only the top-q zeros by neighbour
                                evidence, q = the reference zero-rate quota
    ours:<preset>[;key=value]   zone_impute with a preset and overrides
    spamean | magic | scvi | spage | stplus | tangram | gimvi | envi
                                external packages, run in the benchmark venv
                                through bench/external/run_<name>.py

Our code never imports the external packages; each runs from its own
installation (or pinned checkout under third_party/) behind an h5ad/npz shim.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zone_impute import Config, PRESETS, impute, predict_genes, spatial_graph, detect_ref_space  # noqa: E402

BENCH_PY = Path(os.environ.get("ZONEIMPUTE_BENCH_PY", os.path.expanduser("~/envs/zoneimpute-bench/bin/python")))
EXTERNAL = {"spamean", "magic", "scvi", "spage", "stplus", "tangram", "gimvi", "envi", "alra"}
# methods whose dependencies cannot share the benchmark venv run from their own interpreter
EXTERNAL_PY = {"envi": Path(os.environ.get("ZONEIMPUTE_ENVI_PY", os.path.expanduser("~/envs/zoneimpute-envi/bin/python")))}
CONTROLS = {"scrna_mean", "spatial_avg", "spatial_matched", "gene_depth"}


def _file_hash(*paths, n=8):
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:n]


def code_hash(spec: str) -> str | None:
    """Short hash of the code that produces this method's prediction, so a run id
    changes whenever that code changes: the engine file for ours:*, the shim and
    its helpers for an external package, this module for the controls."""
    fam = method_family(spec)
    if fam == "ours":
        return _file_hash(ROOT / "src" / "zone_impute.py")
    if fam in EXTERNAL:
        return _file_hash(ROOT / "bench" / "external" / f"run_{fam}.py", ROOT / "bench" / "external" / "common.py")
    if fam in CONTROLS:
        return _file_hash(ROOT / "bench" / "methods.py")
    return None


# ----------------------------------------------------------------------------- spec parsing

def parse_ours(spec: str) -> Config:
    """'ours:v3;ref_k=30;type_weight=same_only' -> Config."""
    body = spec.split(":", 1)[1]
    parts = body.split(";")
    cfg = PRESETS[parts[0]]
    for kv in parts[1:]:
        if not kv:
            continue
        k, v = kv.split("=", 1)
        cur = getattr(cfg, k)
        if isinstance(cur, bool):
            val = v.lower() in ("1", "true", "yes")
        elif isinstance(cur, int):
            val = int(v)
        elif isinstance(cur, float):
            val = float(v)
        else:
            val = v
        cfg = cfg.with_(**{k: val})
    return cfg


def method_family(spec: str) -> str:
    return "ours" if spec.startswith("ours:") else spec.split(":", 1)[0]


def _external(spec: str):
    """'tangram:mode=clusters;epochs=300' -> ('tangram', {'mode': 'clusters', 'epochs': '300'})."""
    name, _, rest = spec.partition(":")
    extra = {}
    for kv in rest.split(";"):
        if kv:
            k, v = kv.split("=", 1)
            extra[k] = v
    return name, extra


# ----------------------------------------------------------------------------- controls

def _ref_linear(sc_csr, sc_space):
    sc = sc_csr.astype(np.float32).tocsr()
    if sc_space == "auto":
        sc_space = detect_ref_space(sc)
    if sc_space == "log1p":
        sc = sc.copy(); sc.data = np.expm1(sc.data)
    return sc


def control_scrna_mean(st_csr, sc_csr, sc_labels, st_labels, sc_space, ratio=0.01):
    sc = _ref_linear(sc_csr, sc_space)
    st = st_csr.toarray().astype(np.float32)
    out = st.copy()
    sc_labels = np.asarray(sc_labels).astype(str); st_labels = np.asarray(st_labels).astype(str)
    for t in np.unique(sc_labels):
        rows = np.where(st_labels == t)[0]
        if rows.size == 0:
            continue
        R = sc[sc_labels == t]
        cnt = np.asarray((R > 0).sum(axis=0)).ravel()
        sc_mean = np.asarray(R.sum(axis=0)).ravel() / np.maximum(cnt, 1)
        sub = st[rows]
        st_cnt = (sub > 0).sum(axis=0)
        st_mean = sub.sum(axis=0) / np.maximum(st_cnt, 1)
        fill = np.maximum(st_mean, ratio * sc_mean)
        fill[sc_mean <= 0] = 0.0
        zero = sub == 0
        sub = np.where(zero, fill[None, :], sub)
        out[rows] = sub
    return out


def control_gene_depth(st_csr):
    """Zeros <- the spot's size factor times the gene's mean rate over the section: what
    sequencing depth alone predicts, with no cell type, no reference and no neighbours."""
    st = st_csr.toarray().astype(np.float32)
    lib = st.sum(axis=1).astype(np.float64)
    sf = lib / max(float(np.median(lib[lib > 0])) if (lib > 0).any() else 1.0, 1e-9)
    sf[sf == 0] = 1e-3
    rate = st.sum(axis=0).astype(np.float64) / sf.sum()
    return np.where(st == 0, (sf[:, None] * rate[None, :]).astype(np.float32), st)


def control_spatial_avg(st_csr, st_labels, coords, k=10):
    cfg = Config(spatial_mode="weighted", k=k)
    A, _ = spatial_graph(np.asarray(coords, float), st_labels, cfg)
    sv = np.asarray((A @ st_csr).todense(), dtype=np.float32)
    st = st_csr.toarray().astype(np.float32)
    return np.where(st == 0, sv, st)


def control_spatial_matched(st_csr, sc_csr, sc_labels, st_labels, coords, sc_space, k=10):
    """Fill only as many zeros as the reference quota allows, choosing those with
    the strongest neighbour evidence, with the neighbour average as the value."""
    sc = _ref_linear(sc_csr, sc_space)
    cfg = Config(spatial_mode="weighted", k=k)
    A, _ = spatial_graph(np.asarray(coords, float), st_labels, cfg)
    sv = np.asarray((A @ st_csr).todense(), dtype=np.float32)
    st = st_csr.toarray().astype(np.float32)
    out = st.copy()
    sc_labels = np.asarray(sc_labels).astype(str); st_labels = np.asarray(st_labels).astype(str)
    for t in np.unique(sc_labels):
        rows = np.where(st_labels == t)[0]
        if rows.size == 0:
            continue
        R = sc[sc_labels == t]
        det = np.asarray((R > 0).sum(axis=0)).ravel() / R.shape[0]
        sub = st[rows]
        for g in range(st.shape[1]):
            zl = np.where(sub[:, g] == 0)[0]
            if zl.size == 0:
                continue
            target_zeros = int(round((1.0 - det[g]) * rows.size))
            n_fill = max(0, rows.size - target_zeros - int((sub[:, g] > 0).sum()))
            n_fill = min(n_fill, zl.size)
            if n_fill == 0:
                continue
            ev = sv[rows[zl], g]
            pick = zl[np.argsort(-ev)[:n_fill]]
            out[rows[pick], g] = sv[rows[pick], g]
    return out


# ----------------------------------------------------------------------------- external shim

def _write_inputs(work: Path, st_csr, sc_csr, sc_labels, st_labels, coords, st_genes, sc_genes, sc_space):
    import anndata as ad
    work.mkdir(parents=True, exist_ok=True)
    st = ad.AnnData(sp.csr_matrix(st_csr).astype(np.float32),
                    obs=pd.DataFrame({"label": np.asarray(st_labels).astype(str)}, index=[f"s{i}" for i in range(st_csr.shape[0])]),
                    var=pd.DataFrame(index=pd.Index(list(st_genes)).astype(str)))
    st.obsm["spatial"] = np.asarray(coords, float)
    sc = ad.AnnData(sp.csr_matrix(sc_csr).astype(np.float32),
                    obs=pd.DataFrame({"label": np.asarray(sc_labels).astype(str)}, index=[f"c{i}" for i in range(sc_csr.shape[0])]),
                    var=pd.DataFrame(index=pd.Index(list(sc_genes)).astype(str)))
    sc.uns["space"] = sc_space
    st.write_h5ad(work / "st.h5ad"); sc.write_h5ad(work / "sc.h5ad")


def _run_external(name: str, work: Path, task: str, seed: int, extra: dict | None = None, timeout=6 * 3600):
    script = ROOT / "bench" / "external" / f"run_{name}.py"
    cmd = [str(EXTERNAL_PY.get(name, BENCH_PY)), str(script), "--work", str(work), "--task", task, "--seed", str(seed)]
    for k, v in (extra or {}).items():
        cmd += [f"--{k}", str(v)]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "third_party"))
    if os.environ.get("ZONEIMPUTE_FORCE_CPU", "0") == "1":      # e.g. while the GPU driver is unavailable
        env["CUDA_VISIBLE_DEVICES"] = ""
    # cap CPU threads of external runs: the server overheats when several BLAS/torch jobs each take all cores
    nt = os.environ.get("ZONEIMPUTE_THREADS", "4")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TORCH_NUM_THREADS"):
        env.setdefault(var, nt)
    with open(work / f"{name}_{task}.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=timeout).returncode
    if rc != 0:
        raise RuntimeError(f"{name} failed (rc={rc}); see {work / f'{name}_{task}.log'}")
    z = np.load(work / "pred.npz")
    scale = str(z["scale"]) if "scale" in z.files else "counts"
    if scale == "linear":
        scale = "counts"
    return z["pred"].astype(np.float32), scale


# ----------------------------------------------------------------------------- public API

def denoise(spec, st_csr, sc_csr, sc_labels, st_labels, coords, sc_space, seed, work: Path,
            st_genes=None, sc_genes=None, extra=None, embed_X=None, aux=None):
    st_csr = sp.csr_matrix(st_csr); sc_csr = sp.csr_matrix(sc_csr)
    if spec == "none":
        return st_csr.toarray().astype(np.float32)
    if spec == "scrna_mean":
        return control_scrna_mean(st_csr, sc_csr, sc_labels, st_labels, sc_space)
    if spec == "gene_depth":
        return control_gene_depth(st_csr)
    if spec == "spatial_avg":
        return control_spatial_avg(st_csr, st_labels, coords)
    if spec == "spatial_matched":
        return control_spatial_matched(st_csr, sc_csr, sc_labels, st_labels, coords, sc_space)
    if spec.startswith("ours:"):
        cfg = parse_ours(spec).with_(seed=seed)
        if cfg.ref_space == "auto" and sc_space in ("linear", "log1p"):
            cfg = cfg.with_(ref_space=sc_space)
        return impute(sc_csr, sc_labels, st_csr, st_labels, coords, cfg, embed_X=embed_X,
                      aux=dict(aux or {}, gene_names=list(st_genes) if st_genes is not None else None))
    name, opts = _external(spec)
    if name in EXTERNAL:
        _write_inputs(work, st_csr, sc_csr, sc_labels, st_labels, coords, st_genes, sc_genes, sc_space)
        return _run_external(name, work, "denoise", seed, {**opts, **(extra or {})})[0]
    raise KeyError(spec)


def predict(spec, st_kept_csr, sc_all_csr, sc_labels, st_labels, coords, shared, held, sc_space, seed,
            work: Path, extra=None, sc_genes=None):
    """Predict ``held`` genes.  Returns (pred, scale) where scale names the
    footing of the values: 'counts' (count-like at each spot's own depth),
    'counts_common' (count-like at one common depth), 'freq' (per-spot
    frequencies), 'log' (log1p of library-normalised expression).
    ``sc_all_csr`` columns are ordered shared + held [+ every other reference
    gene]; ``sc_genes`` gives their names (default shared + held)."""
    st_kept_csr = sp.csr_matrix(st_kept_csr); sc_all_csr = sp.csr_matrix(sc_all_csr)
    n_sh = len(shared) + len(held)
    sc_genes = list(sc_genes) if sc_genes is not None else list(shared) + list(held)
    if spec.startswith("ours:"):
        cfg = parse_ours(spec).with_(seed=seed)
        if cfg.ref_space == "auto" and sc_space in ("linear", "log1p"):
            cfg = cfg.with_(ref_space=sc_space)
        # the engine scales the reference term by one platform factor, so its values sit at a typical ST depth,
        # not at each spot's own depth
        return predict_genes(sc_all_csr[:, :n_sh], sc_labels, st_kept_csr, st_labels, coords, held, shared, cfg), "counts_common"
    if spec == "scrna_mean":
        # per-type mean of the log-normalised reference, directly on the truth footing
        sc = _ref_linear(sc_all_csr, sc_space)
        lib = np.asarray(sc.sum(axis=1)).ravel(); lib[lib == 0] = 1.0
        sc = sp.diags(1e4 / lib) @ sc
        sc = sc[:, len(shared):n_sh].tocsr(); sc.data = np.log1p(sc.data)
        sc_labels = np.asarray(sc_labels).astype(str); st_labels = np.asarray(st_labels).astype(str)
        out = np.zeros((st_kept_csr.shape[0], len(held)), np.float32)
        glob = np.asarray(sc.mean(axis=0)).ravel()
        for t in np.unique(sc_labels):
            rows = st_labels == t
            if rows.any():
                out[rows] = np.asarray(sc[sc_labels == t].mean(axis=0)).ravel()
        out[~np.isin(st_labels, np.unique(sc_labels))] = glob
        return out, "log"
    name, opts = _external(spec)
    if name in EXTERNAL:
        _write_inputs(work, st_kept_csr, sc_all_csr, sc_labels, st_labels, coords, shared, sc_genes, sc_space)
        (work / "held.txt").write_text("\n".join(map(str, held)))
        return _run_external(name, work, "predict", seed, {**opts, **(extra or {})})
    raise KeyError(spec)
