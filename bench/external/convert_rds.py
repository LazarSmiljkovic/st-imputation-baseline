"""Convert the GSE157079 kidney reference (.rds sparse count matrix + labels) to h5ad.

The deposit is an R `dgCMatrix` (S4), which pyreadr cannot read; the `rdata`
package parses the RDS container and we rebuild the CSC matrix from its
slots (i, p, x, Dim, Dimnames).  Run in the benchmark venv:
    python bench/external/convert_rds.py
"""
import gzip
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

D = Path(os.environ.get("ZONEIMPUTE_DOWNLOADS", os.path.expanduser("~/datasets")))
rds_gz = D / "GSE157079_P0_adult_counts.rds.gz"
rds = D / "GSE157079_P0_adult_counts.rds"
if not rds.exists():
    with gzip.open(rds_gz, "rb") as f, open(rds, "wb") as g:
        shutil.copyfileobj(f, g)

import rdata

parsed = rdata.parser.parse_file(str(rds))


def find_slots(obj, depth=0):
    """Walk the parsed R object tree and collect the slots of the first S4
    object that carries i/p/x/Dim."""
    from rdata.parser import RObjectType
    found = {}

    def walk(o):
        if found:
            return
        attrs = getattr(o, "attributes", None)
        if attrs is not None:
            slots = {}
            a = attrs
            while a is not None and getattr(a, "info", None) is not None and a.info.type == RObjectType.LIST:
                tag = a.tag
                name = None
                if tag is not None and getattr(tag, "value", None) is not None:
                    v = tag.value
                    name = v.value.decode() if hasattr(v, "value") and isinstance(v.value, (bytes, bytearray)) else str(getattr(v, "value", v))
                slots[name] = a.value[0]
                a = a.value[1]
            if {"i", "p", "x", "Dim"} <= set(slots):
                found.update(slots)
                return
        val = getattr(o, "value", None)
        if isinstance(val, (list, tuple)):
            for v in val:
                walk(v)
        elif val is not None and hasattr(val, "attributes"):
            walk(val)

    walk(obj)
    return found


def to_np(o):
    v = o.value
    return np.asarray(v)


slots = find_slots(parsed.object)
if not slots:
    # fall back to the generic converter and hope for a dict-like
    conv = rdata.conversion.convert(parsed)
    raise SystemExit(f"could not locate dgCMatrix slots; converter returned {type(conv)}")

i = to_np(slots["i"]).astype(np.int64)
p = to_np(slots["p"]).astype(np.int64)
x = to_np(slots["x"]).astype(np.float32)
dim = to_np(slots["Dim"]).astype(int)
names = slots.get("Dimnames")
rownames = colnames = None
if names is not None:
    parts = names.value
    def strs(o):
        return [e.value.decode() if isinstance(e.value, (bytes, bytearray)) else str(e.value) for e in o.value]
    rownames = strs(parts[0]) if parts[0].value is not None else None
    colnames = strs(parts[1]) if parts[1].value is not None else None
M = sp.csc_matrix((x, i, p), shape=(int(dim[0]), int(dim[1])))     # genes x cells in R
print("dgCMatrix", M.shape, "nnz", M.nnz, "max", x.max())
X = M.T.tocsr()                                                      # cells x genes
genes = pd.Index(rownames if rownames else [f"g{k}" for k in range(X.shape[1])]).astype(str)
cells = pd.Index(colnames if colnames else [f"c{k}" for k in range(X.shape[0])]).astype(str)

clusters = pd.read_csv(D / "GSE157079_P0_adult_clusters.txt.gz", sep="\t").set_index("barcodes")[["clusters"]]
pheno = pd.read_csv(D / "GSE157079_P0_adult_pheno.txt.gz", sep="\t").set_index("barcodes")[["pheno"]]
print("matrix cell ids look like:", list(cells[:3]), "| label ids like:", list(clusters.index[:3]))
obs = pd.DataFrame(index=cells)
obs = obs.join(clusters, how="left").join(pheno, how="left")
print("unlabelled cells:", int(obs["clusters"].isna().sum()))
if "clusters" not in obs.columns:
    cand = [c for c in obs.columns if "clust" in c.lower() or "type" in c.lower()]
    obs["clusters"] = obs[cand[0]].astype(str) if cand else "unknown"
a = ad.AnnData(X, obs=obs, var=pd.DataFrame(index=genes))
a.var_names_make_unique()
a.write_h5ad(D / "GSE157079_P0_adult_counts.h5ad")
print(a)
print(a.obs["clusters"].value_counts().head(20))
