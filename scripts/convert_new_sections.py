#!/usr/bin/env python3
"""
convert_new_sections.py -- conversion of the downloaded sections to .h5ad under $ZONEIMPUTE_DOWNLOADS (default ~/datasets).

  kidney pucks   GEO GSE190094 (Marshall et al. 2022): MappedDGEForR.csv.gz (genes x beads) and
                 BeadLocationsForR_qc.csv.gz (barcode, x, y, author cell type) -> gse190094/h5ad/<GSM>.h5ad;
                 the files are taken from GSE190094_RAW.tar, the puck titles from the series matrix
  10x Visium     V1_Adult_Mouse_Brain filtered matrix + spatial positions -> visium10x/V1_Adult_Mouse_Brain.h5ad
  DLPFC          HumanPilot filtered matrices, tissue positions, and the manual layer labels
                 (first-round merged and second-round combined csv) -> dlpfc/<sample>.h5ad
  MOSTA embryo   the counts layer of E16.5_E1S3_cell_bin_whole_brain.h5ad restricted to the cells and genes of
                 $ZONEIMPUTE_DATA/ST_Mouse_embryo_whole_brain.h5ad -> mosta/ST_Mouse_embryo_whole_brain_counts.h5ad

    python scripts/convert_new_sections.py --pucks GSM5713381 GSM5713354 GSM5713350 GSM5713388 --visium --dlpfc --mosta
"""
import argparse
import os
import gzip
import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

D = Path(os.environ.get("ZONEIMPUTE_DOWNLOADS", Path.home() / "datasets"))


def puck_titles():
    out = D / "gse190094" / "gsm_titles.json"
    if out.exists():
        return
    import json
    rows = {}
    with gzip.open(D / "gse190094" / "GSE190094_series_matrix.txt.gz", "rt") as f:
        for line in f:
            if line.startswith(("!Sample_title", "!Sample_geo_accession")):
                rows[line.split("\t")[0]] = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
    out.write_text(json.dumps(dict(zip(rows["!Sample_geo_accession"], rows["!Sample_title"])), indent=1))


def puck(gsm):
    out = D / "gse190094" / "h5ad" / f"{gsm}.h5ad"
    if out.exists():
        print("exists", out); return
    pucks = D / "gse190094" / "pucks"
    if not list(pucks.glob(f"{gsm}_*")):
        pucks.mkdir(parents=True, exist_ok=True)
        with tarfile.open(D / "gse190094" / "GSE190094_RAW.tar") as t:
            for m in t.getmembers():
                if m.name.startswith(f"{gsm}_"):
                    t.extract(m, pucks)
    files = sorted(pucks.glob(f"{gsm}_*"))
    dge = [f for f in files if "MappedDGE" in f.name][0]; loc = [f for f in files if f.name.endswith("_qc.csv.gz")][0]
    pos = pd.read_csv(loc).set_index("barcodes")
    cols = pd.read_csv(dge, nrows=0).columns
    counts = pd.read_csv(dge, index_col=0, dtype={c: np.float32 for c in cols[1:]})   # genes x beads
    common = pos.index.intersection(counts.columns)
    X = sp.csr_matrix(counts[common].values.T.astype(np.float32)); X.eliminate_zeros()
    a = ad.AnnData(X, obs=pd.DataFrame({"author_cell_type": pos.loc[common, "cell_type"].astype(str).values}, index=common),
                   var=pd.DataFrame(index=counts.index.astype(str)))
    a.var_names_make_unique()
    a.obsm["spatial"] = pos.loc[common, ["xcoord", "ycoord"]].values.astype(float)
    a.uns["source"] = f"GEO GSE190094 {gsm}: {dge.name}, qc bead locations with author cell types"
    out.parent.mkdir(parents=True, exist_ok=True); a.write_h5ad(out, compression="gzip")
    print("wrote", out, a.shape, "beads with counts", int((X.sum(axis=1) > 0).sum()))


def visium():
    import scanpy as sc
    out = D / "visium10x" / "V1_Adult_Mouse_Brain.h5ad"
    if out.exists():
        print("exists", out); return
    a = sc.read_10x_h5(D / "visium10x" / "V1_Adult_Mouse_Brain_filtered_feature_bc_matrix.h5"); a.var_names_make_unique()
    with tarfile.open(D / "visium10x" / "V1_Adult_Mouse_Brain_spatial.tar.gz") as t:
        pos = pd.read_csv(io.BytesIO(t.extractfile("spatial/tissue_positions_list.csv").read()), header=None, index_col=0)
    pos.columns = ["in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
    pos = pos.reindex(a.obs_names)
    a = a[pos.in_tissue.values == 1].copy()
    a.obsm["spatial"] = pos.loc[a.obs_names, ["pxl_col", "pxl_row"]].values.astype(float)
    a.X = sp.csr_matrix(a.X).astype(np.float32)
    a.uns["source"] = "10x Genomics public dataset V1_Adult_Mouse_Brain (Space Ranger 1.1.0)"
    a.write_h5ad(out, compression="gzip"); print("wrote", out, a.shape)


def dlpfc():
    import scanpy as sc
    lay = pd.concat([pd.read_csv(D / "dlpfc" / "layer_guesses_first_round_merged.csv"),
                     pd.read_csv(D / "dlpfc" / "layer_guesses_second_round_combined2.csv")])
    lay["sample_name"] = lay.sample_name.astype(str)
    for s in sorted(set(lay.sample_name)):
        out = D / "dlpfc" / f"{s}.h5ad"
        if out.exists():
            print("exists", out); continue
        a = sc.read_10x_h5(D / "dlpfc" / f"{s}_filtered_feature_bc_matrix.h5"); a.var_names_make_unique()
        pos = pd.read_csv(D / "dlpfc" / f"{s}_tissue_positions_list.txt", header=None, index_col=0)
        pos.columns = ["in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
        pos = pos.reindex(a.obs_names)
        a.obsm["spatial"] = pos[["pxl_col", "pxl_row"]].values.astype(float)
        lab = lay[lay.sample_name == s].drop_duplicates("spot_name").set_index("spot_name")["layer"].reindex(a.obs_names)
        a.obs["layer"] = lab.fillna("NA").astype(str).values
        a.X = sp.csr_matrix(a.X).astype(np.float32)
        a.uns["source"] = f"spatialLIBD DLPFC sample {s}: 10x filtered matrix (spatial-dlpfc S3), HumanPilot positions and manual layer labels"
        a.write_h5ad(out, compression="gzip")
        print("wrote", out, a.shape, "labelled", int((a.obs.layer != "NA").sum()), dict(a.obs.layer.value_counts().head(8)))


def mosta():
    out = D / "mosta" / "ST_Mouse_embryo_whole_brain_counts.h5ad"
    if out.exists():
        print("exists", out); return
    data = Path(os.environ.get("ZONEIMPUTE_DATA", Path.home() / "CoDi" / "data"))
    proj = ad.read_h5ad(data / "ST_Mouse_embryo_whole_brain.h5ad", backed="r")
    raw = ad.read_h5ad(D / "mosta" / "E16.5_E1S3_cell_bin_whole_brain.h5ad")
    raw = raw[proj.obs_names, proj.var_names]
    a = ad.AnnData(sp.csr_matrix(raw.layers["counts"]).astype(np.float32), obs=pd.DataFrame(index=proj.obs_names),
                   var=pd.DataFrame(index=proj.var_names))
    a.write_h5ad(out, compression="gzip"); print("wrote", out, a.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pucks", nargs="*", default=[]); ap.add_argument("--visium", action="store_true"); ap.add_argument("--dlpfc", action="store_true")
    ap.add_argument("--mosta", action="store_true")
    a = ap.parse_args()
    if a.pucks:
        puck_titles()
    for g in a.pucks:
        puck(g)
    if a.visium:
        visium()
    if a.dlpfc:
        dlpfc()
    if a.mosta:
        mosta()
