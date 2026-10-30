"""
bench/datasets.py -- dataset registry and loaders.

Every loader returns a :class:`Pair`:
    st        : AnnData  (spots x genes, raw counts, .obsm[coords_key])
    sc        : AnnData  (cells x genes, counts in the declared space)
    sc_label  : obs column with reference cell types
    st_label  : array of per-spot labels (same vocabulary as sc_label where possible)
    coords    : (n, 2) float array
    author    : optional per-spot independent labels (for non-circular zones)
    meta      : dict (platform, source, count space, label source)

Paths default to the server layout (~/CoDi/data and ~/datasets) and can be
overridden with the ZONEIMPUTE_DATA and ZONEIMPUTE_DOWNLOADS environment
variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA = Path(os.environ.get("ZONEIMPUTE_DATA", os.path.expanduser("~/CoDi/data")))
DOWNLOADS = Path(os.environ.get("ZONEIMPUTE_DOWNLOADS", os.path.expanduser("~/datasets")))


@dataclass
class Pair:
    name: str
    st: "anndata.AnnData"
    sc: "anndata.AnnData"
    sc_label: str
    st_label: np.ndarray
    coords: np.ndarray
    author: np.ndarray | None = None
    meta: dict = field(default_factory=dict)
    st_probs: tuple | None = None        # (spots x labels probability matrix, label names), when an annotation gives them
    parents: dict | None = None          # label -> parent class, when the labels have a hierarchy

    @property
    def genes(self):
        return self.st.var_names


def _read(path):
    import anndata as ad
    a = ad.read_h5ad(path)
    a.var_names_make_unique()
    a.obs_names_make_unique()
    return a


def _label_csv(name: str, source: str, st_obs_names) -> np.ndarray:
    """Per-spot labels from CoDi or one of the other annotation tools."""
    if source == "codi":
        df = pd.read_csv(DATA / f"{name}_CoDi.csv").set_index("cell_id")
        return df["CoDi"].reindex(st_obs_names).astype(str).values
    import zipfile
    with zipfile.ZipFile(DATA / "tools_predictions.zip") as z:
        fn = f"tools_predictions/{name}_{source}.csv"
        with z.open(fn) as f:
            df = pd.read_csv(f)
    if source == "cell2location":
        df = df.set_index(df.columns[0])
        col = "cell2location"
    else:
        df = df.set_index("cell_id")
        col = source
    return df[col].reindex(st_obs_names).astype(str).values


def _codi_soft_probs(name: str, st_obs_names):
    """Soft labels from CoDi's two classifiers: each spot's distance label and contrastive label,
    weighted by their confidences (one label when both agree).  Returns (matrix, names)."""
    df = pd.read_csv(DATA / f"{name}_CoDi.csv").set_index("cell_id").reindex(st_obs_names)
    labs = sorted(set(df["CoDi_dist"].dropna().astype(str)) | set(df["CoDi_contrastive"].dropna().astype(str)))
    idx = {l: i for i, l in enumerate(labs)}
    P = np.zeros((len(df), len(labs)), dtype=np.float64)
    for col, conf in (("CoDi_dist", "confidence_dist"), ("CoDi_contrastive", "confidence_contrastive")):
        l = df[col].astype(str).values; c = pd.to_numeric(df[conf], errors="coerce").fillna(0.5).values
        for i, (li, ci) in enumerate(zip(l, c)):
            if li in idx:
                P[i, idx[li]] += max(float(ci), 1e-3)
    return P, np.array(labs)


def _intersect(st, sc):
    genes = st.var_names.intersection(sc.var_names)
    return st[:, genes].copy(), sc[:, genes].copy()


def load_kidney(label_source: str = "codi", reference: str = "stored") -> Pair:
    """Slide-seqV2 mouse kidney (Marshall et al. 2022) with the GSE157079 reference.

    reference="stored": the log-normalised reference shipped with CoDi.
    reference="raw":    raw counts converted from GEO (needs
                        ~/datasets/GSE157079_P0_adult_counts.h5ad, produced by
                        bench/external/convert_rds.py).
    reference="raw_adult": the raw reference restricted to adult cells (pheno == Adult),
                        which removes the neonatal progenitor types."""
    st = _read(DATA / "ST_Mouse_kidney.h5ad")
    if reference in ("raw", "raw_adult"):
        sc = _read(DOWNLOADS / "GSE157079_P0_adult_counts.h5ad")
        if reference == "raw_adult":
            sc = sc[sc.obs["pheno"].astype(str) == "Adult"].copy()
        sc_label = "clusters"
        space = "linear"
    else:
        sc = _read(DATA / "SC_Mouse_Kidney.h5ad")
        sc_label = "clusters"
        space = "log1p"
    st, sc = _intersect(st, sc)
    labels = _label_csv("mouse_kidney", label_source, st.obs_names)
    author = st.obs["author_cell_type"].astype(str).values if "author_cell_type" in st.obs else None
    return Pair("kidney", st, sc, sc_label, labels, np.asarray(st.obsm["X_spatial"], float), author,
                dict(platform="Slide-seqV2", source="GSE190094 / GSE157079", sc_space=space,
                     label_source=label_source),
                st_probs=_codi_soft_probs("mouse_kidney", st.obs_names) if label_source == "codi" else None)


def load_visium(label_source: str = "codi") -> Pair:
    st = _read(DATA / "ST_Mouse_Brain_Visium.h5ad")
    sc = _read(DATA / "SC_Mouse_Brain_Visium.h5ad")
    st, sc = _intersect(st, sc)
    labels = _label_csv("visium_mouse_brain", label_source, st.obs_names)
    return Pair("visium", st, sc, "annotation_1", labels, np.asarray(st.obsm["spatial"], float), None,
                dict(platform="Visium", source="cell2location tutorial sample", sc_space="linear",
                     label_source=label_source))


def _subclass_to_class(sc):
    """Majority cell_class of each cell_subclass in the L5 reference."""
    t = sc.obs.groupby("cell_subclass", observed=True)["cell_class"].agg(lambda x: x.value_counts().index[0])
    return t.to_dict()


def load_embryo(label_source: str = "codi", coarse: bool = False) -> Pair:
    st = _read(DATA / "ST_Mouse_embryo_whole_brain.h5ad")
    sc = _read(DATA / "SC_mouse_brain_L5.h5ad")
    st, sc = _intersect(st, sc)
    labels = _label_csv("whole_brain_mouse_embryo", label_source, st.obs_names)
    if coarse:
        m = _subclass_to_class(sc); labels = np.asarray([m.get(l, l) for l in labels], dtype=str)
    author = st.obs["sim anno"].astype(str).values if "sim anno" in st.obs else None
    return Pair("embryo", st, sc, "cell_class" if coarse else "cell_subclass", labels,
                np.asarray(st.obsm["spatial"], float), author,
                dict(platform="Stereo-seq", source="MOSTA E16.5 E1S3 / Zeisel 2018 L5", sc_space="linear",
                     label_source=label_source + ("-coarse" if coarse else "")),
                st_probs=_codi_soft_probs("whole_brain_mouse_embryo", st.obs_names) if label_source == "codi" and not coarse else None,
                parents=None if coarse else _subclass_to_class(sc))


def load_embryo_raw(label_source: str = "codi", coarse: bool = False) -> Pair:
    """The same embryo cells and genes with the raw counts of the original MOSTA file.

    ``ST_Mouse_embryo_whole_brain.h5ad`` holds depth-equalised values (every cell sums to
    nearly the same total), not counts.  The raw counts are the ``counts`` layer of MOSTA's
    ``E16.5_E1S3_cell_bin_whole_brain.h5ad`` (STOmicsDB STDS0000058); a copy restricted to the
    project's cells and genes is expected at ~/datasets/mosta/ST_Mouse_embryo_whole_brain_counts.h5ad."""
    pair = load_embryo(label_source=label_source, coarse=coarse)
    raw = _read(DOWNLOADS / "mosta" / "ST_Mouse_embryo_whole_brain_counts.h5ad")
    raw = raw[pair.st.obs_names, pair.st.var_names].copy()
    raw.obsm["spatial"] = pair.st.obsm["spatial"]
    meta = dict(pair.meta, st_counts="raw counts (MOSTA counts layer)")
    return Pair("embryo_raw", raw, pair.sc, pair.sc_label, pair.st_label, pair.coords, pair.author, meta,
                st_probs=pair.st_probs, parents=pair.parents)


def load_kidney_puck(gsm: str) -> Pair:
    """Another Slide-seqV2 kidney puck of GEO GSE190094 (Marshall et al. 2022), converted by
    scripts/convert_new_sections.py, with the raw adult GSE157079 reference.  Working labels are
    transferred from the reference clusters as for the imaging pairs; the deposit's own bead
    cell types are kept as author labels."""
    import json
    st = _read(DOWNLOADS / "gse190094" / "h5ad" / f"{gsm}.h5ad")
    sc = _read(DOWNLOADS / "GSE157079_P0_adult_counts.h5ad")
    sc = sc[sc.obs["pheno"].astype(str) == "Adult"].copy()
    st, sc = _intersect(st, sc)
    labels = _transfer_labels(sc, st, sc.obs["clusters"].astype(str).values, seed=0)
    titles = json.loads((DOWNLOADS / "gse190094" / "gsm_titles.json").read_text())
    author = st.obs["author_cell_type"].astype(str).values
    return Pair(f"kidney_puck:{gsm}", st, sc, "clusters", labels, np.asarray(st.obsm["spatial"], float), author,
                dict(platform="Slide-seqV2", source=f"GEO GSE190094 {gsm} ({titles.get(gsm, '?')}) / GSE157079 adult", sc_space="linear",
                     label_source="reference-cluster-transfer", condition=titles.get(gsm, "?")))


def load_visium10x(coarse: bool = True) -> Pair:
    """10x Genomics public adult mouse brain Visium section with the Zeisel L5 atlas as reference,
    working labels transferred from the reference classes (coarse) or subclasses."""
    st = _read(DOWNLOADS / "visium10x" / "V1_Adult_Mouse_Brain.h5ad")
    sc = _read(DATA / "SC_mouse_brain_L5.h5ad")
    st, sc = _intersect(st, sc)
    col = "cell_class" if coarse else "cell_subclass"
    labels = _transfer_labels(sc, st, sc.obs[col].astype(str).values, seed=0)
    return Pair("visium10x", st, sc, col, labels, np.asarray(st.obsm["spatial"], float), None,
                dict(platform="Visium", source="10x V1_Adult_Mouse_Brain / Zeisel 2018 L5", sc_space="linear",
                     label_source="reference-transfer-" + col))


def load_dlpfc(sample: str) -> Pair:
    """A spatialLIBD DLPFC Visium section (Maynard et al. 2021) with the manual cortical layer of
    every spot as both the working label and the author label.  No matched reference is bundled:
    the reference slot holds an empty matrix with one row per layer, so only reference-free
    methods (depth only, spatial average, depth x label, MAGIC, scVI) are meaningful here.
    Spots without a layer ('NA') are not scored."""
    import anndata as ad
    st = _read(DOWNLOADS / "dlpfc" / f"{sample}.h5ad")
    layers = st.obs["layer"].astype(str).values
    labs = sorted(set(layers) - {"NA"})
    sc = ad.AnnData(sp.csr_matrix((len(labs), st.n_vars), dtype=np.float32),
                    obs=pd.DataFrame({"layer": labs}, index=[f"placeholder_{l}" for l in labs]), var=pd.DataFrame(index=st.var_names))
    return Pair(f"dlpfc:{sample}", st, sc, "layer", layers, np.asarray(st.obsm["spatial"], float), layers.copy(),
                dict(platform="Visium", source=f"spatialLIBD DLPFC {sample}", sc_space="linear", label_source="author-layers",
                     reference="none: reference-free methods only"))


def load_adult_brain(label_source: str = "codi", coarse: bool = False) -> Pair:
    st = _read(DATA / "ST_Adult_Mouse_Brain.h5ad")
    sc = _read(DATA / "SC_mouse_brain_L5.h5ad")
    st, sc = _intersect(st, sc)
    labels = _label_csv("adult_mouse_brain_B5", label_source, st.obs_names)
    if coarse:
        m = _subclass_to_class(sc); labels = np.asarray([m.get(l, l) for l in labels], dtype=str)
    author = st.obs["celltype_pred"].astype(str).values if "celltype_pred" in st.obs else None
    return Pair("adultB5", st, sc, "cell_class" if coarse else "cell_subclass", labels,
                np.asarray(st.obsm["spatial"], float), author,
                dict(platform="Stereo-seq", source="Stereo-seq adult B5 / Zeisel 2018 L5", sc_space="linear",
                     label_source=label_source + ("-coarse" if coarse else "")),
                st_probs=_codi_soft_probs("adult_mouse_brain_B5", st.obs_names) if label_source == "codi" and not coarse else None,
                parents=None if coarse else _subclass_to_class(sc))


def list_tissue_pairs():
    """Folders of the TISSUE Zenodo deposit (one per paired dataset)."""
    root = DOWNLOADS / "tissue_pairs"
    if not root.exists():
        return []
    out = []
    for p in sorted(root.rglob("Spatial_count.txt")):
        out.append(p.parent)
    return out


def load_tissue_pair(folder) -> Pair:
    """A pair from the TISSUE deposit: Spatial_count.txt, scRNA_count.txt,
    Locations.txt, Metadata.txt.  Cell types for the spatial side come from
    Metadata.txt when present; otherwise a label is transferred by nearest
    reference centroid so that the per-type machinery has something to use."""
    import anndata as ad
    folder = Path(folder)
    def _table(path):
        """Count table as cells x genes.  The deposit stores the spatial table as
        cells x genes with a gene header, and the scRNA table as genes x cells
        with a cell header and gene names in the first (unnamed) column."""
        df = pd.read_csv(path, sep="\t", index_col=None)
        first = str(df.columns[0])
        if first.startswith("Unnamed") or first == "":
            df = df.set_index(df.columns[0])
            df.index = df.index.astype(str)
        return df

    spatial = _table(folder / "Spatial_count.txt")
    scrna = _table(folder / "scRNA_count.txt")
    # tab-separated in most pairs; Dataset11 writes both coordinates space-separated on one field
    loc = pd.read_csv(folder / "Locations.txt", sep=r"\s+", engine="python", index_col=None)
    sp_genes = set(map(str, spatial.columns))
    # orient the reference: genes must be on the columns
    if len(sp_genes & set(map(str, scrna.index))) > len(sp_genes & set(map(str, scrna.columns))):
        scrna = scrna.T
    if spatial.shape[0] != loc.shape[0] and spatial.shape[1] == loc.shape[0]:
        spatial = spatial.T
    st_X, st_was = _restore_counts(sp.csr_matrix(spatial.values.astype(np.float64)))
    sc_X, sc_was = _restore_counts(sp.csr_matrix(scrna.values.astype(np.float64)))
    st = ad.AnnData(st_X.astype(np.float32),
                    obs=pd.DataFrame(index=[f"s{i}" for i in range(spatial.shape[0])]),
                    var=pd.DataFrame(index=spatial.columns.astype(str)))
    sc = ad.AnnData(sc_X.astype(np.float32),
                    obs=pd.DataFrame(index=[f"c{i}" for i in range(scrna.shape[0])]),
                    var=pd.DataFrame(index=scrna.columns.astype(str)))
    st.var_names_make_unique(); sc.var_names_make_unique()
    coords = loc.values[:, :2].astype(float)
    st.obsm["spatial"] = coords
    author = None
    meta_file = folder / "Metadata.txt"
    if meta_file.exists():
        meta = pd.read_csv(meta_file, sep=None, engine="python")
        for col in meta.columns:
            if ("type" in col.lower() or "cluster" in col.lower() or "class" in col.lower()) and len(meta) == st.n_obs:
                author = meta[col].astype(str).values
                break
    # The deposit carries no reference labels, so a working label is derived once:
    # Leiden clusters of the reference (log-normalised PCA) transferred to the spots
    # by majority vote of their nearest reference cells in a joint embedding of the
    # shared genes.  This gives the per-type machinery something to condition on.
    shared = st.var_names.intersection(sc.var_names)
    sc.obs["ctype"] = _reference_clusters(sc, seed=0)
    st_label_for_engine = _transfer_labels(sc[:, shared], st[:, shared], sc.obs["ctype"].values, seed=0)
    return Pair(folder.name, st, sc, "ctype", st_label_for_engine, coords, author,
                dict(platform="imaging (TISSUE deposit)", source=str(folder), sc_space="linear",
                     label_source="reference-leiden-transfer", n_shared=int(len(shared)),
                     st_stored_as=st_was, sc_stored_as=sc_was,
                     st_binary=bool(st.X.data.size and st.X.data.max() <= 1)))


def _restore_counts(X):
    """Some deposited tables hold log1p(counts) (Axolotl: smallest value ln 2).  Detect that
    (non-integer values whose expm1 is integer) and return the counts; otherwise return X as is.
    Second return value says how the table was stored: 'counts', 'log1p_counts' or 'other'."""
    X = sp.csr_matrix(X).copy(); X.eliminate_zeros()
    d = X.data
    if d.size == 0:
        return X, "counts"
    sample = d[:: max(1, d.size // 200000)]
    if np.allclose(sample, np.round(sample), atol=1e-6):
        return X, "counts"
    e = np.expm1(sample)
    if sample.min() > 0 and np.allclose(e, np.round(e), atol=1e-3):
        X.data = np.round(np.expm1(d))
        return X, "log1p_counts"
    return X, "other"


def _reference_clusters(sc, seed=0, n_clusters=None):
    """Working cell types for a reference without labels: k-means on 30 PCs of the
    log-normalised data (no igraph dependency; deterministic for a seed)."""
    import scanpy as scp
    from sklearn.cluster import KMeans
    a = sc.copy()
    scp.pp.normalize_total(a, target_sum=1e4); scp.pp.log1p(a)
    scp.pp.highly_variable_genes(a, n_top_genes=min(2000, a.n_vars - 1))
    scp.pp.pca(a, n_comps=min(30, a.n_vars - 1), mask_var="highly_variable")
    k = n_clusters or int(np.clip(round(np.sqrt(a.n_obs / 100)), 8, 25))
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(a.obsm["X_pca"])
    return np.asarray(["ref%d" % c for c in km.labels_], dtype=str)


def _transfer_labels(sc_shared, st_shared, sc_labels, seed=0, k=15):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from zone_impute import _joint_embedding
    from sklearn.neighbors import NearestNeighbors
    E, F = _joint_embedding(sp.csr_matrix(sc_shared.X), sp.csr_matrix(st_shared.X), 30, seed)
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(E)
    _, idx = nn.kneighbors(F)
    labs = np.asarray(sc_labels).astype(str)[idx]
    out = []
    for row in labs:
        u, c = np.unique(row, return_counts=True)
        out.append(u[np.argmax(c)])
    return np.asarray(out, dtype=str)


def load_synthetic(seed: int = 0, n_types: int = 3, n_genes: int = 80, n_spots: int = 2000, n_ref: int = 250) -> Pair:
    """A small simulated pair for testing the harness end to end (no files needed): Poisson
    counts around type-specific gene means, spots laid out in three vertical bands of one
    type each, a reference with the same types plus a few genes the section does not measure."""
    import anndata as ad
    rng = np.random.default_rng(seed)
    n_extra = 20
    base = rng.gamma(2.0, 1.5, size=(n_types, n_genes + n_extra))
    ref, lab = [], []
    for t in range(n_types):
        depth = rng.gamma(4.0, 0.5, size=(n_ref, 1))
        ref.append(rng.poisson(base[t] * depth)); lab += [f"T{t}"] * n_ref
    genes = [f"g{i}" for i in range(n_genes + n_extra)]
    sc = ad.AnnData(sp.csr_matrix(np.vstack(ref).astype(np.float32)), obs=pd.DataFrame({"ctype": lab}, index=[f"c{i}" for i in range(n_types * n_ref)]),
                    var=pd.DataFrame(index=genes))
    coords = rng.uniform(0, 100, size=(n_spots, 2))
    band = np.minimum((coords[:, 0] // (100 / n_types)).astype(int), n_types - 1)
    depth = rng.gamma(3.0, 0.04, size=(n_spots, 1))          # shallow: about a quarter of the entries are non-zero
    X = rng.poisson(base[band][:, :n_genes] * depth)
    st = ad.AnnData(sp.csr_matrix(X.astype(np.float32)), obs=pd.DataFrame(index=[f"s{i}" for i in range(n_spots)]), var=pd.DataFrame(index=genes[:n_genes]))
    st.obsm["spatial"] = coords
    labels = np.asarray([f"T{b}" for b in band], dtype=str)
    return Pair("synthetic", st, sc, "ctype", labels, coords, labels.copy(),
                dict(platform="simulated", source="bench.datasets.load_synthetic", sc_space="linear", label_source="true"))


REGISTRY = {
    "synthetic": load_synthetic,
    "kidney": load_kidney,
    "visium": load_visium,
    "embryo": load_embryo,
    "embryo_raw": load_embryo_raw,
    "visium10x": load_visium10x,
    "adultB5": load_adult_brain,
}


def load(name: str, **kw) -> Pair:
    if name in REGISTRY:
        return REGISTRY[name](**kw)
    if name.startswith("tissue:"):
        return load_tissue_pair(DOWNLOADS / "tissue_pairs" / name.split(":", 1)[1])
    if name.startswith("kidney_puck:"):
        return load_kidney_puck(name.split(":", 1)[1])
    if name.startswith("dlpfc:"):
        return load_dlpfc(name.split(":", 1)[1])
    raise KeyError(f"unknown dataset {name}; known: {list(REGISTRY)}, tissue:<folder>, kidney_puck:<GSM>, dlpfc:<sample>")
