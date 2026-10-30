# A fast, interpretable baseline for spatial transcriptomics imputation

Code and recorded results for the paper *A Fast, Interpretable Baseline for Spatial Transcriptomics Imputation:
Sequencing Depth and Cell Identity Analysis* (Smiljković and Mišić).

The estimate fills a zero of gene g in spot i with the spot's size factor times the rate of g in the spot's label,
corrects that rate with a few components of the spot's own measured genes (cross-fitted, so a gene never predicts
itself), blends in a neighbour term whose weight is chosen by thinning the input, and fills the zero only where the
probability of at least one molecule, 1 − exp(−expected count), reaches a threshold τ (0.1 by default). The labels
are k-means clusters of the input matrix itself, so no reference is needed for measured genes.

## Contents

```
src/zone_impute.py          the method: impute_depth_type() for measured genes, predict_genes() for genes the
                            panel did not measure; every option is a field of Config
bench/                      the benchmark harness
  datasets.py               loaders of the 24 sections and imaging panels (paths in DATA.md)
  protocols.py              masking of non-zeros (P1), binomial thinning (P2), left-out genes (P3)
  metrics.py                per-gene Spearman at hidden entries, deviance, P3 measures, clustering agreement
  methods.py                method specs ("ours:depth_type;rate_shrink=20;..."), controls, external shims
  runner.py                 runs, scores, rescoring and aggregation; one folder per run under results/.../runs
  runner_markers.py         downstream marker test on the kidney pucks
  runner_resources.py       wall time, CPU time and memory per method
  external/run_*.py         MAGIC, scVI/scANVI, gimVI, Tangram, SpaGE, stPlus, SpaMean-Impute, ENVI, ALRA
reproduce.py                reruns groups of the recorded experiments and compares them with our numbers
scripts/                    analyses behind the figures and tables (see below), data download and conversion
tests/                      engine invariants and an end-to-end run of the harness on synthetic data
results/bench/              p1/p2/p3_results.csv: every experiment we ran, one row each (the recorded numbers)
results/analysis/, marker/  outputs of the analysis scripts used by the tables and figures
environment/                pinned package versions of the three Python environments
third_party/README.md       comparator versions and the two checkouts (SpaGE, ALRA)
```

## Installation

The method and the harness need Python 3.12 and the packages in `environment/requirements-core.txt`
(numpy, scipy, pandas, scikit-learn, anndata, scanpy, joblib, matplotlib; the file pins every version we used).

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r environment/requirements-core.txt
.venv/bin/python tests/test_zone_impute.py        # engine, about 10 s
.venv/bin/python tests/test_bench_smoke.py        # harness on synthetic data, about 20 s
```

The comparators run in separate environments, given to the harness by `ZONEIMPUTE_BENCH_PY` (scVI, MAGIC, Tangram,
gimVI, stPlus, SpaGE, SpaMean-Impute; `environment/requirements-comparators.txt`) and `ZONEIMPUTE_ENVI_PY` (ENVI,
`environment/requirements-envi.txt`); ALRA runs through `ZONEIMPUTE_ALRA_RSCRIPT`. They are only needed to rerun
the comparators (see `third_party/README.md`).

## Using the method

```python
from dataclasses import replace
from zone_impute import PRESETS, impute_depth_type        # with src/ on the path

cfg = replace(PRESETS["depth_type"], rate_shrink=20, within_label_pcs=5, within_label_xfit=5, spatial_blend="auto",
              quota_mode="expected", fill_threshold=0.1, relabel="kmeans", n_jobs=8)
# X: spots x genes raw counts (scipy sparse or dense), labels: any labelling of the spots (with relabel="kmeans"
# only its number of distinct values is used), coords: spots x 2
filled = impute_depth_type(X, labels, cfg, coords=coords)
filled, details, expected, prob = impute_depth_type(X, labels, cfg, coords=coords, return_details=True)
```

`expected` holds the expected count of every entry and `prob` the probability of at least one molecule;
`quota_mode="none"` fills every zero. `n_jobs` sets the number of worker processes and does not change the result.
In the harness the same configuration is the method spec
`ours:depth_type;rate_shrink=20;within_label_pcs=5;within_label_xfit=5;spatial_blend=auto;quota_mode=expected;fill_threshold=0.1;relabel=kmeans`.

## Reproducing the results

1. Get the data (`DATA.md`, `scripts/download_data.sh`) and set `ZONEIMPUTE_DATA` and `ZONEIMPUTE_DOWNLOADS`.
2. Rerun experiments. `reproduce.py` reads the recorded experiments from `results/bench/*_results.csv`, reruns a
   group of them with `bench.runner` into `results/rerun/`, and prints the recorded and the new value of each:

   ```bash
   python reproduce.py list
   python reproduce.py run quick                     # kidney, one seed, three methods, a few minutes
   python reproduce.py run recovery --datasets kidney --seeds 42 43
   python reproduce.py run recovery                  # all 21 sections and 5 seeds, including scVI, MAGIC, ALRA
   python reproduce.py run thinning
   python reproduce.py run unmeasured
   python reproduce.py run markers
   python reproduce.py run recovery --dry-run        # only print the bench.runner commands
   ```

   The method is deterministic for a given seed: on our machine the rerun of the `quick` group matches the recorded
   values exactly. scVI, gimVI, Tangram, stPlus and ENVI on a GPU are not bit-reproducible and differ in the third
   decimal. The threads used by numerical libraries are set by `ZONEIMPUTE_THREADS` (default 4).
3. Analyses that are not single benchmark runs. `reliability.py` and `densification_curve.py` read the saved
   predictions of the runs, so they need a rerun of the `thinning` and `recovery` groups first; set
   `ZONEIMPUTE_RESULTS=results/rerun` (and `--runs results/rerun/runs` for the densification curve) to use it:

   | script | output | used for |
   |---|---|---|
   | `scripts/reliability.py` | `results/analysis/reliability*.csv`, `fill_decision.csv` | calibration under thinning, fill decision |
   | `scripts/densification_curve.py --dataset kidney --out results/analysis/densification_curve_kidney_v2.csv` | | correlation against the share of zeros filled |
   | `scripts/label_leak_check.py` | `results/analysis/label_leak_check.csv` | stored labels against k-means on the full and the masked section |
   | `scripts/cluster_coverage.py adultB5 embryo_raw` | `results/analysis/cluster_coverage.csv` | share of spots the within-label correction reaches |
   | `scripts/tau_specificity.py` | `results/analysis/tau_specificity.csv` | threshold against spatial specificity |
   | `scripts/illustration.py` | `results/analysis/illustration_kidney.npz` | the kidney figure |
   | `scripts/weight_curve.py` | `results/analysis/weight_curve_*.csv` | oracle weights by neighbourhood homogeneity (zone weighting) |
   | `scripts/timing.py`, `scripts/timing_comparators.py`, `python -m bench.runner_resources` | `results/analysis/timing_parallel.csv`, `resources.csv` | computational cost |

4. Tables and figures:

   ```bash
   python scripts/tables.py                          # results/tables/*.csv from the recorded results
   python scripts/tables.py --bench results/rerun    # the same from a rerun
   python scripts/figures.py                         # results/figures/*.pdf (ZONEIMPUTE_RESULTS=results/rerun for a rerun)
   ```

   `scripts/tables.py` lists which table of the paper each file corresponds to.

`python reproduce.py run all` reruns every recorded experiment, including the development runs behind the
negative results of the supplement (about 3,700 runs, which takes days on one workstation with a GPU).

## Licence

MIT (see `LICENSE`). The comparators keep their own licences and are not included.
