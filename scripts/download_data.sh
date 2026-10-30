#!/usr/bin/env bash
# Download the public inputs into $ZONEIMPUTE_DOWNLOADS (default ~/datasets) and convert them to .h5ad.
# Also fetches the prepared sections into $ZONEIMPUTE_DATA (default ~/CoDi/data).  About 15 GB.
# Files already present are resumed or skipped.  See DATA.md for what each one is used for.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
D=${ZONEIMPUTE_DOWNLOADS:-$HOME/datasets}
DATA=${ZONEIMPUTE_DATA:-$HOME/CoDi/data}
PY=${PYTHON:-python}
RECORD=23808580       # the prepared sections, references and CoDi labels
mkdir -p "$D"/{gse190094,visium10x,dlpfc,mosta,tissue_pairs} "$DATA"

get() {  # url, output file; the CNGB and GEO servers close long transfers, so resume until complete
  for i in $(seq 1 50); do
    curl -sS -L -C - --retry 3 --max-time 3600 -o "$2" "$1" && return 0
    sleep 10
  done
  echo "failed: $1" >&2; return 1
}

# prepared sections, references and labels (Zenodo) into $ZONEIMPUTE_DATA
cd "$DATA"
for f in ST_Mouse_kidney.h5ad SC_Mouse_Kidney.h5ad mouse_kidney_CoDi.csv ST_Adult_Mouse_Brain.h5ad adult_mouse_brain_B5_CoDi.csv \
         ST_Mouse_embryo_whole_brain.h5ad whole_brain_mouse_embryo_CoDi.csv SC_mouse_brain_L5.h5ad ST_Mouse_Brain_Visium.h5ad \
         SC_Mouse_Brain_Visium.h5ad visium_mouse_brain_CoDi.csv tools_predictions.zip md5sums.txt; do
  get https://zenodo.org/records/$RECORD/files/$f "$f"
done
md5sum -c --quiet md5sums.txt

cd "$D"
GEO=https://ftp.ncbi.nlm.nih.gov/geo/series
# Slide-seqV2 kidney pucks (Marshall et al. 2022) and the kidney single-cell reference (GSE157079)
get $GEO/GSE190nnn/GSE190094/suppl/GSE190094_RAW.tar gse190094/GSE190094_RAW.tar
get $GEO/GSE190nnn/GSE190094/matrix/GSE190094_series_matrix.txt.gz gse190094/GSE190094_series_matrix.txt.gz
for f in counts.rds.gz clusters.txt.gz pheno.txt.gz; do
  get $GEO/GSE157nnn/GSE157079/suppl/GSE157079_P0_adult_$f GSE157079_P0_adult_$f
done
# 10x Genomics adult mouse brain Visium section
V=https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Adult_Mouse_Brain
get $V/V1_Adult_Mouse_Brain_filtered_feature_bc_matrix.h5 visium10x/V1_Adult_Mouse_Brain_filtered_feature_bc_matrix.h5
get $V/V1_Adult_Mouse_Brain_spatial.tar.gz visium10x/V1_Adult_Mouse_Brain_spatial.tar.gz
# spatialLIBD DLPFC sections and their manual layer labels (Maynard et al. 2021)
H=https://raw.githubusercontent.com/LieberInstitute/HumanPilot/master
for s in 151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676; do
  get $H/10X/$s/filtered_feature_bc_matrix.h5 dlpfc/${s}_filtered_feature_bc_matrix.h5
  get $H/10X/$s/tissue_positions_list.txt dlpfc/${s}_tissue_positions_list.txt
done
get "$H/Analysis/Layer_Guesses/First_Round/spatialLIBD_layerGuesses_2019-12-19%2019_49_57_Merged.csv" dlpfc/layer_guesses_first_round_merged.csv
get "$H/Analysis/Layer_Guesses/Second_Round/spatialLIBD_layerGuesses_2019-12-30%2017_35_40_Combined2.csv" dlpfc/layer_guesses_second_round_combined2.csv
# MOSTA E16.5 embryo whole brain, cell bins (STOmicsDB STDS0000058)
get https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/E16.5_E1S3_cell_bin_whole_brain.h5ad mosta/E16.5_E1S3_cell_bin_whole_brain.h5ad
# imaging panels with paired single-cell references (TISSUE deposit, Zenodo 8259942)
get https://zenodo.org/records/8259942/files/DataUploadFinal.tar.gz tissue_pairs/DataUploadFinal.tar.gz
[ -d tissue_pairs/DataUploadFinal ] || tar -xzf tissue_pairs/DataUploadFinal.tar.gz -C tissue_pairs

"$PY" "$ROOT/bench/external/convert_rds.py"
args=(--pucks GSM5713381 GSM5713354 GSM5713350 GSM5713388 --visium --dlpfc)
"$PY" "$ROOT/scripts/convert_new_sections.py" "${args[@]}" --mosta
