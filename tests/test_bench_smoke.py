"""End-to-end smoke test of the benchmark harness on a simulated pair.

Runs all three protocols, rescoring and aggregation into a temporary
folder and checks that every requested method produced finite scores.  It guards the
plumbing (payloads, metric names, progress lines, CSV columns), not the science.

Run with ``python tests/test_bench_smoke.py`` or ``pytest``.
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("ZONEIMPUTE_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402

from bench import runner  # noqa: E402

RECOVERY = ["none", "gene_depth", "scrna_mean", "spatial_avg", "spatial_matched", "ours:corrected", "ours:v3",
            "ours:v3;ref_estimate=expected;zero_select=ref_evidence",
            "ours:v3;ref_estimate=expected;weighting=none;quota_mode=none;fill_threshold=0.0"]
PREDICTION = ["ours:v3", "ours:v3;ref_mode=marginal", "scrna_mean"]


def test_harness_end_to_end():
    with tempfile.TemporaryDirectory() as out:
        runner.main(["p1", "--dataset", "synthetic", "--genes", "all", "--mask", "0.2", "--seeds", "1", "2", "--depth-groups", "4",
                     "--zero-sample", "5000", "--methods", *RECOVERY, "--out", out])
        runner.main(["p1", "--dataset", "synthetic", "--genes", "all", "--mask", "0.2", "--seeds", "1", "--cluster", "--depth-groups", "4",
                     "--zero-sample", "5000", "--methods", "none", "gene_depth", "ours:depth_type;rate_shrink=20", "--out", out])
        runner.main(["p1", "--dataset", "synthetic", "--genes", "top40", "--mask", "0.2", "--seeds", "1",
                     "--zero-sample", "5000", "--methods", "ours:v3", "--out", out])          # context-gene path
        runner.main(["p1", "--dataset", "synthetic", "--genes", "top40", "--mask", "0.2", "--seeds", "1", "--context", "panel",
                     "--zero-sample", "5000", "--methods", "ours:v3", "--out", out])
        runner.main(["p2", "--dataset", "synthetic", "--genes", "all", "--thin", "0.5", "--seeds", "1",
                     "--zero-sample", "5000", "--methods", *RECOVERY, "--out", out])
        runner.main(["p3", "--dataset", "synthetic", "--folds", "4", "--seeds", "0", "--methods", *PREDICTION, "--out", out])
        runner.main(["rescore", "--out", out, "--depth-groups", "4"])
        runner.main(["aggregate", "--out", out])
        o = Path(out)
        import json
        errors = [(json.loads((e.parent / "meta.json").read_text()).get("method"), e.read_text()[:300]) for e in (o / "runs").glob("*/error.txt")]
        assert not errors, f"runs failed: {errors}"
        p1 = pd.read_csv(o / "p1_results.csv"); p2 = pd.read_csv(o / "p2_results.csv"); p3 = pd.read_csv(o / "p3_results.csv")
        assert set(RECOVERY) <= set(p1.method) and set(RECOVERY) <= set(p2.method) and set(PREDICTION) <= set(p3.method)
        # the clustered runs supersede the plain runs of the same experiment; one of them is a new method
        assert len(p1) == 2 * len(RECOVERY) + 2 + 1 and len(p3) == 4 * len(PREDICTION)
        cl = p1[p1.cluster_ari.notna()]
        assert set(cl.method) == {"none", "gene_depth", "ours:depth_type;rate_shrink=20"} and cl.cluster_nmi.between(0, 1).all()
        assert "cell_spearman_median" in p3 and np.isfinite(p3.cell_spearman_median).all()
        scored = p1[p1.method != "none"]
        for col in ("spearman_all", "pergene_spearman_wmean", "hidden_fill_frac", "zero_select_auroc", "densification_at_zeros"):
            assert col in p1 and np.isfinite(scored[col]).all(), col
        # filling everything must reach every hidden entry; the selective variant must not
        full = p1[p1.method.str.contains("quota_mode=none")].hidden_fill_frac
        sel = p1[p1.method == "ours:v3;ref_estimate=expected;zero_select=ref_evidence"].hidden_fill_frac
        assert (full > 0.99).all() and sel.max() < full.min() - 0.02, (full.tolist(), sel.tolist())
        # within depth deciles a depth-only prediction must lose most of its apparent accuracy,
        # while a prediction that also knows the cell type keeps a clear margin over it
        g = p1.groupby("method")[["pergene_spearman_wmean", "pergene_spearman_depthstrat"]].mean()
        full_type = "ours:v3;ref_estimate=expected;weighting=none;quota_mode=none;fill_threshold=0.0"
        assert g.loc["gene_depth", "pergene_spearman_depthstrat"] < 0.5 * g.loc["gene_depth", "pergene_spearman_wmean"], g.to_dict()
        assert g.loc[full_type, "pergene_spearman_depthstrat"] > g.loc["gene_depth", "pergene_spearman_depthstrat"] + 0.05, g.to_dict()
        # thinning protocol: squared error for everyone, deviance only where every entry is predicted
        assert np.isfinite(p2.heldout_mse).all()
        dev = p2.set_index("method").heldout_deviance
        assert np.isfinite(dev["gene_depth"]) and np.isnan(dev["none"]) and np.isnan(dev["ours:v3"])
        # on Poisson data built from type-specific rates and depth, type x depth must beat depth alone
        assert dev["ours:v3;ref_estimate=expected;weighting=none;quota_mode=none;fill_threshold=0.0"] < dev["gene_depth"], dev.to_dict()
        for col in ("pcc_median", "spearman_median", "ssim_median"):
            assert np.isfinite(p3[col]).all(), col
        assert set(p3.pred_scale) == {"counts_common", "log"}
        # the types differ by construction, so the type-mean control must predict held-out genes above chance
        pm = p3.groupby("method").pcc_median.mean()
        assert pm["scrna_mean"] > 0.1, pm.to_dict()


if __name__ == "__main__":
    test_harness_end_to_end(); print("ok test_harness_end_to_end")
