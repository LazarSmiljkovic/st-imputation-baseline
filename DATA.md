# Data

The loaders in `bench/datasets.py` read from two folders:

- `$ZONEIMPUTE_DATA` (default `~/CoDi/data`): the sections and references prepared for the CoDi
  cell-type annotation tool, with CoDi's labels, deposited at Zenodo (https://doi.org/10.5281/zenodo.23808580);
- `$ZONEIMPUTE_DOWNLOADS` (default `~/datasets`): everything else, downloaded from the original deposits by
  `scripts/download_data.sh` and converted by `bench/external/convert_rds.py` and `scripts/convert_new_sections.py`.

All inputs are public. Nothing in the repository depends on where they live, as long as the two variables point to the folders.

## Sections

| name in the benchmark | platform | spots × genes | source | reference |
|---|---|---|---|---|
| `kidney` | Slide-seqV2 | 36,299 beads | GEO GSE190094 (Marshall et al. 2022), puck GSM5713384 | GSE157079 (`SC_Mouse_Kidney.h5ad`) |
| `kidney_puck:GSM5713381`, `…54`, `…50`, `…88` | Slide-seqV2 | 25,757–34,638 beads | GEO GSE190094 | GSE157079 raw counts, adult cells |
| `adultB5` | Stereo-seq | 45,201 spots | adult mouse brain section B5 (`ST_Adult_Mouse_Brain.h5ad`) | Zeisel et al. 2018, level 5 (`SC_mouse_brain_L5.h5ad`) |
| `embryo_raw` | Stereo-seq | 59,704 cells | MOSTA E16.5 E1S3 whole brain, cell bins (STOmicsDB STDS0000058), raw counts | Zeisel et al. 2018, level 5 |
| `visium10x` | Visium | 2,702 spots | 10x Genomics V1_Adult_Mouse_Brain | Zeisel et al. 2018, level 5 |
| `dlpfc:<sample>` | Visium | 3,460–4,110 spots | spatialLIBD / HumanPilot (Maynard et al. 2021), manual layer labels | none (reference-free methods only) |
| `tissue:DataUploadFinal/<panel>` | imaging (MERFISH, seqFISH, STARmap and others) | 33–351 genes | TISSUE deposit, Zenodo 8259942 | paired scRNA-seq table of each panel |

The main results use the eleven sections `kidney`, `kidney_puck:GSM5713381`, `adultB5`, `embryo_raw`,
`dlpfc:151673`, `visium10x` and the panels Dataset1, Dataset9, Dataset15, Dataset17 and SVZFullFinal. The ten
held-out sections are the other three pucks, DLPFC 151507, 151669 and 151676, and Dataset3, 11, 12 and 13.

## Files under `$ZONEIMPUTE_DATA`

| file | contents |
|---|---|
| `ST_Mouse_kidney.h5ad` | kidney puck, raw counts, `obsm["X_spatial"]` |
| `SC_Mouse_Kidney.h5ad` | GSE157079 reference, log-normalised, `obs["clusters"]` |
| `ST_Adult_Mouse_Brain.h5ad` | Stereo-seq adult brain B5, raw counts, `obsm["spatial"]`, `obs["celltype_pred"]` |
| `ST_Mouse_embryo_whole_brain.h5ad` | MOSTA embryo cells and genes (depth-equalised values; only the cell and gene lists and the coordinates are used, the counts come from the MOSTA file) |
| `SC_mouse_brain_L5.h5ad` | Zeisel et al. 2018 level 5, raw counts, `obs["cell_class"]`, `obs["cell_subclass"]` |
| `mouse_kidney_CoDi.csv`, `adult_mouse_brain_B5_CoDi.csv`, `whole_brain_mouse_embryo_CoDi.csv` | CoDi labels per spot (`cell_id`, `CoDi`, `CoDi_dist`, `CoDi_contrastive` and confidences) |
| `ST_Mouse_Brain_Visium.h5ad`, `SC_Mouse_Brain_Visium.h5ad`, `visium_mouse_brain_CoDi.csv` | a Visium mouse brain section with its reference and labels, used in development runs only (`visium`) |
| `tools_predictions.zip` | labels of the same sections by Tangram, cell2location, Seurat and CytoSPACE (`--labels tangram` and so on) |

The stored CoDi labels are read by the loaders, but the main results do not use them: the proposed estimate and
the base variant compute their own labels by k-means on the masked input (`relabel=kmeans`), because labels computed
on the full section carry information about the hidden entries (see the label-leak results in the paper). The stored labels are
used for the label-leak comparison, scANVI with stored labels, the label transfer for unmeasured genes and the
development runs.

## Download and conversion

```bash
export ZONEIMPUTE_DOWNLOADS=~/datasets
export ZONEIMPUTE_DATA=~/CoDi/data
bash scripts/download_data.sh          # about 15 GB; needs curl, and anndata, scanpy and rdata in $PYTHON
```

The script downloads the prepared files of the table above (checked against `md5sums.txt`) and the GEO, 10x,
HumanPilot, MOSTA and TISSUE files, and then writes

- `GSE157079_P0_adult_counts.h5ad` (the kidney reference as raw counts, from the `.rds` sparse matrix),
- `gse190094/h5ad/<GSM>.h5ad` for the four pucks and `gse190094/gsm_titles.json`,
- `visium10x/V1_Adult_Mouse_Brain.h5ad`, `dlpfc/<sample>.h5ad`,
- `mosta/ST_Mouse_embryo_whole_brain_counts.h5ad` (the `counts` layer of the MOSTA file restricted to the cells and
  genes of `ST_Mouse_embryo_whole_brain.h5ad`),
- `tissue_pairs/DataUploadFinal/<panel>/` (unpacked TISSUE deposit).

Each conversion is deterministic; on our machine the converted files are identical to the ones the results were computed from.
