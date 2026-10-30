# Comparators

The published methods are only run, never modified or redistributed. Each one is called by a shim in
`bench/external/run_<name>.py`, which reads the masked matrix from an `.h5ad`/`.npz` file, calls the package and writes
the prediction back. The shims run in their own interpreter (`ZONEIMPUTE_BENCH_PY`, `ZONEIMPUTE_ENVI_PY`), so the
packages never share an environment with the method.

| method | source | licence | version used |
|---|---|---|---|
| scVI, scANVI, gimVI | scvi-tools (pip) | BSD-3 | `environment/requirements-comparators.txt` |
| MAGIC | magic-impute (pip) | GPL-2.0 | same |
| Tangram | tangram-sc (pip) | BSD-3 | same |
| stPlus | stPlus (pip) | MIT | same |
| SpaMean-Impute | SpaMeanImpute (pip) | MIT | same |
| ENVI | scenvi (pip, JAX) | MIT | `environment/requirements-envi.txt` |
| SpaGE | https://github.com/tabdelaal/SpaGE | MIT | commit `edd0fb7086eca791c6589020893eb4f034b195c7`, checked out in `third_party/SpaGE` |
| ALRA | https://github.com/KlugerLab/ALRA | MIT | commit `f34d46570b1221179047d9f99c235ed880cc3bae`, checked out in `third_party/ALRA`, run with R 4.3.3, rsvd 1.0.5, Matrix 1.6.5 (`ZONEIMPUTE_ALRA_RSCRIPT`) |

```bash
git clone https://github.com/tabdelaal/SpaGE third_party/SpaGE && git -C third_party/SpaGE checkout edd0fb7
git clone https://github.com/KlugerLab/ALRA third_party/ALRA && git -C third_party/ALRA checkout f34d465
```
