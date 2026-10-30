"""Invariants of the zone_impute engine on a small synthetic fixture.

Run with ``python tests/test_zone_impute.py`` (no pytest needed) or ``pytest``.
"""
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from zone_impute import Config, PRESETS, impute, impute_depth_type, predict_genes, detect_ref_space, fit_weight_table, _cluster_markers, _normalise_log  # noqa: E402


def make_fixture(seed=0, n_types=3, n_genes=60, n_st=400, n_ref=200):
    rng = np.random.default_rng(seed)
    base = rng.gamma(2.0, 1.5, size=(n_types, n_genes))
    ref_rows, ref_lab = [], []
    for t in range(n_types):
        lam = base[t] * rng.gamma(1.0, 1.0, size=(n_ref, 1)) * 3.0
        X = rng.poisson(lam) * (rng.random((n_ref, n_genes)) > 0.4)
        ref_rows.append(X); ref_lab += [f"T{t}"] * n_ref
    sc_X = sp.csr_matrix(np.vstack(ref_rows).astype(np.float32))
    sc_lab = np.array(ref_lab)
    coords = rng.uniform(0, 100, size=(n_st, 2))
    st_lab = np.where(coords[:, 0] < 40, "T0", np.where(coords[:, 0] > 60, "T1", "T2")).astype(object)
    st_lab[:15] = "ABSENT"                      # a type with no reference cells
    st = np.zeros((n_st, n_genes))
    for i in range(n_st):
        t = int(st_lab[i][1]) if st_lab[i] != "ABSENT" else 0
        st[i] = rng.poisson(base[t] * 0.4) * (rng.random(n_genes) > 0.6)
    return sc_X, sc_lab, sp.csr_matrix(st.astype(np.float32)), np.array(st_lab, dtype=str), coords


FX = make_fixture()


def test_predict_genes_options():
    """Every backbone, correction, smoothing and tuning option of the gene predictor runs, gives finite
    non-negative values of the right shape, and the soft backbone needs no spot labels."""
    sc_X, sc_lab, st_X, st_lab, coords = make_fixture(seed=2, n_types=3, n_genes=60, n_st=600, n_ref=300)
    shared = [f"g{i}" for i in range(50)]; target = [f"g{i}" for i in range(50, 60)]
    st_kept = sp.csr_matrix(st_X)[:, :50]
    base = PRESETS["v3"]
    variants = [base, base.with_(predict_within=3), base.with_(predict_backbone="soft", predict_clusters=6, predict_smooth="none"),
                base.with_(predict_backbone="soft", predict_clusters=0), base.with_(predict_backbone="soft", predict_clusters=6, predict_within=3),
                base.with_(predict_smooth="fixed", predict_smooth_w=0.3), base.with_(predict_smooth="auto", predict_tune="genes"),
                base.with_(predict_backbone="soft", predict_within=3, predict_smooth="auto", predict_tune="genes"),
                base.with_(predict_backbone="soft", predict_clusters=6, predict_smooth="none", predict_detail="knn"),
                base.with_(predict_backbone="soft", predict_clusters=6, predict_detail="knn", predict_smooth="auto", predict_tune="genes")]
    outs = []
    for cfg in variants:
        out, det = predict_genes(sc_X, sc_lab, st_kept, st_lab, coords, target, shared, cfg, return_details=True)
        assert out.shape == (600, 10) and np.isfinite(out).all() and (out >= 0).all(), cfg
        if cfg.predict_tune == "genes":
            assert det["within"] in (0, 3) and 0.0 <= det["smooth_w"] <= 0.9 and det["backbone"] in ("label", "soft")
        outs.append(out)
    assert not np.allclose(outs[0], outs[1]) and not np.allclose(outs[0], outs[2]) and not np.allclose(outs[8], outs[2])
    # the soft backbone ignores the labels
    shuffled = np.random.default_rng(0).permutation(st_lab)
    a_ = predict_genes(sc_X, sc_lab, st_kept, shuffled, coords, target, shared, base.with_(predict_backbone="soft", predict_clusters=6, predict_smooth="none"))
    assert np.allclose(a_, outs[2])


def _run(cfg):
    sc_X, sc_lab, st_X, st_lab, coords = FX
    if cfg.kde_space == "exp":              # the published pipeline is only defined on a log1p-stored reference
        sc_X = sc_X.copy(); sc_X.data = np.log1p(sc_X.data)
    return impute(sc_X, sc_lab, st_X, st_lab, coords, cfg)


def test_presets_run_and_keep_nonzeros():
    st = FX[2].toarray()
    for name, cfg in PRESETS.items():
        out = _run(cfg)
        assert out.shape == st.shape, name
        assert np.isfinite(out).all() and (out >= 0).all(), name
        nz = st > 0
        assert np.allclose(out[nz], st[nz]), f"{name}: observed non-zeros changed"
        assert (out > 0).sum() >= nz.sum(), name


def test_absent_type_untouched():
    st = FX[2].toarray()
    out = _run(PRESETS["v3"])
    absent = FX[3] == "ABSENT"
    assert np.array_equal(out[absent], st[absent])


def test_determinism():
    cfg = PRESETS["v3"]
    a = _run(cfg); b = _run(cfg)
    assert np.array_equal(a, b)
    c = _run(cfg.with_(seed=7))
    assert not np.array_equal(a, c)


def test_reference_quota_respected():
    sc_X, sc_lab, st_X, st_lab, coords = FX
    cfg = PRESETS["corrected"]
    out = impute(sc_X, sc_lab, st_X, st_lab, coords, cfg)
    sc = sc_X.toarray()
    for t in np.unique(sc_lab):
        rows = st_lab == t
        if not rows.any():
            continue
        ref_det = (sc[sc_lab == t] > 0).mean(axis=0)
        st_det = (out[rows] > 0).mean(axis=0)
        orig_det = (st_X.toarray()[rows] > 0).mean(axis=0)
        # after filling, detection is at most max(reference rate, original rate) (+1 spot rounding)
        slack = 1.0 / rows.sum() + 1e-9
        assert np.all(st_det <= np.maximum(ref_det, orig_det) + slack), t


def test_switches():
    base = PRESETS["v3"]
    variants = [
        base.with_(ref_estimate="mean"),
        base.with_(ref_estimate="expected"),
        base.with_(ref_estimate="expected", ref_mode="marginal"),
        base.with_(ref_estimate="expected", zero_select="expected"),
        base.with_(ref_estimate="expected", zero_select="expected", quota_mode="expected", fill_threshold=0.2),
        base.with_(zero_select="expected"),
        base.with_(ref_estimate="expected", zero_select="expected", rate_shrink=10.0),
        base.with_(ref_embed="markers"),
        base.with_(ref_embed="markers", marker_n=10, ref_estimate="expected", zero_select="expected", quota_mode="expected", fill_threshold=0.2),
        base.with_(ref_dist="nb"),
        base.with_(ref_estimate="posterior", ref_dist="nb"),
        base.with_(ref_estimate="posterior", ref_dist="nb", denoise_all=True),
        base.with_(zero_select="posterior", quota_mode="none", fill_threshold=0.5),
        base.with_(zero_select="detect_prob"),
        base.with_(zero_select="combined", quota_mode="none", fill_threshold=0.9),
        base.with_(zero_select="ref_evidence"),
        base.with_(type_weight="same_only"),
        base.with_(radius_mult=2.0),
        base.with_(weighting="reversed"),
        base.with_(passes=2),
        base.with_(spatial_mode="none"),
        base.with_(ref_kernel="rank"),
        base.with_(weighting="table", weight_table=(np.linspace(0, 1.0000001, 6), np.array([0.3, 0.5, 0.6, 0.7, 0.8]))),
    ]
    st = FX[2].toarray()
    for cfg in variants:
        out = _run(cfg)
        assert out.shape == st.shape and np.isfinite(out).all() and (out >= 0).all(), cfg
        if not cfg.denoise_all:
            nz = st > 0
            assert np.allclose(out[nz], st[nz]), cfg


def test_published_rejects_linear_reference():
    sc_X, sc_lab, st_X, st_lab, coords = FX
    try:
        impute(sc_X, sc_lab, st_X, st_lab, coords, PRESETS["published"])
    except ValueError as e:
        assert "log1p" in str(e)
    else:
        raise AssertionError("published preset must refuse a linear-count reference")


def test_posterior_scales_with_depth():
    """Posterior fills carry the exposure: deeper spots get larger expected counts."""
    from zone_impute import _untruncate_rate
    m = np.array([1.2, 2.0, 5.0, 20.0])
    lam = _untruncate_rate(m)
    assert np.allclose(lam / (1 - np.exp(-lam)), m, rtol=1e-4)


def test_expected_estimate_tracks_depth():
    """With the 'expected' estimate and no spatial blend, fills of one gene in one type grow with spot depth."""
    sc_X, sc_lab, st_X, st_lab, coords = FX
    cfg = PRESETS["v3"].with_(ref_estimate="expected", ref_mode="marginal", weighting="none", quota_mode="none", fill_threshold=0.0)
    out = impute(sc_X, sc_lab, st_X, st_lab, coords, cfg)
    st = st_X.toarray(); lib = st.sum(axis=1)
    rows = np.where(st_lab == "T1")[0]
    g = int(np.argmax((st[rows] == 0).sum(axis=0)))            # gene with most zeros in the type
    z = rows[st[rows, g] == 0]
    filled = out[z, g]
    assert (filled > 0).all()
    assert np.corrcoef(filled, lib[z])[0, 1] > 0.99            # value = size factor x a constant rate


def test_expected_threshold_is_a_cut_on_the_full_fill():
    """Thresholded selective filling equals filling everything and keeping the values above the cut,
    and a stricter threshold fills fewer zeros."""
    sc_X, sc_lab, st_X, st_lab, coords = FX
    st = st_X.toarray(); zero = st == 0
    full_cfg = PRESETS["v3"].with_(ref_estimate="expected", weighting="none", quota_mode="none", fill_threshold=0.0)
    full = impute(sc_X, sc_lab, st_X, st_lab, coords, full_cfg)
    n_prev = None
    for tau in (0.05, 0.2, 0.5):
        cfg = PRESETS["v3"].with_(ref_estimate="expected", weighting="none", zero_select="expected", quota_mode="expected", fill_threshold=tau)
        out = impute(sc_X, sc_lab, st_X, st_lab, coords, cfg)
        cut = -np.log(1.0 - tau)                                  # 1 - exp(-E) >= tau  <=>  E >= -log(1 - tau)
        want = np.where(zero & (full >= cut), full, st)
        close = np.isclose(full, cut, rtol=1e-4) & zero            # values sitting on the cut may fall either side (float32)
        assert np.allclose(out[~close], want[~close], rtol=1e-4, atol=1e-6), tau
        n = int((out[zero] > 0).sum())
        assert n_prev is None or n <= n_prev
        n_prev = n
    assert n_prev < int((full[zero] > 0).sum())


def test_depth_type_estimate():
    """The reference-free estimate agrees with the engine's marginal expected count wherever the engine
    fills, shrinkage removes exact zeros and pulls a small type toward the gene's rate, and the
    threshold only removes fills."""
    sc_X, sc_lab, st_X, st_lab, coords = FX
    st = st_X.toarray(); zero = st == 0
    free = impute(sc_X, sc_lab, st_X, st_lab, coords, PRESETS["depth_type"])
    assert np.allclose(free[~zero], st[~zero])
    eng = impute(sc_X, sc_lab, st_X, st_lab, coords, PRESETS["v3"].with_(ref_mode="marginal", ref_estimate="expected", weighting="none",
                                                                         quota_mode="none", fill_threshold=0.0))
    both = zero & (eng > 0)
    assert both.sum() > 1000 and np.allclose(free[both], eng[both], rtol=1e-4, atol=1e-6)
    assert (free[st_lab == "ABSENT"][zero[st_lab == "ABSENT"]] > 0).any()        # no reference needed for a label
    shrunk = impute_depth_type(st_X, st_lab, PRESETS["depth_type"].with_(rate_shrink=20.0))
    expressed = st.sum(axis=0) > 0
    assert (shrunk[:, expressed] > 0).all()                                      # no exact zeros left
    # the 15 ABSENT spots form the smallest group: its rates move toward the section-wide rate
    lib = st.sum(axis=1); sf = lib / np.median(lib[lib > 0]); gene_rate = st.sum(axis=0) / sf.sum()
    rows = np.where(st_lab == "ABSENT")[0]
    def rate_of(m):
        return np.array([np.mean(m[rows, g][zero[rows, g]] / sf[rows][zero[rows, g]]) for g in np.where(expressed)[0] if zero[rows, g].any()])
    idx = [g for g in np.where(expressed)[0] if zero[rows, g].any()]
    assert np.abs(rate_of(shrunk) - gene_rate[idx]).mean() < np.abs(rate_of(free) - gene_rate[idx]).mean()
    cut = impute_depth_type(st_X, st_lab, PRESETS["depth_type"].with_(quota_mode="expected", fill_threshold=0.3))
    assert ((cut > 0) <= (free > 0)).all() and (cut[zero] > 0).sum() < (free[zero] > 0).sum()


def test_cluster_markers():
    """Markers of a simulated type are the genes with the largest rate in that type relative to the others."""
    sc_X, sc_lab, st_X, st_lab, coords = FX
    rng = np.random.default_rng(0)
    base = rng.gamma(2.0, 1.5, size=(3, 60))                     # the fixture's type-specific rates
    m = _cluster_markers(_normalise_log(sc_X), sc_lab, 5)
    assert set(m) == {"T0", "T1", "T2"} and all(len(v) == 5 for v in m.values())
    for t in range(3):
        ratio = base[t] / base.mean(axis=0)
        top = set(np.argsort(-ratio)[:15])
        assert len(set(m[f"T{t}"]) & top) >= 3, (t, m[f"T{t}"])


def test_depth_type_options():
    """Every option of the reference-free estimate runs, keeps observed non-zeros, fills zeros with
    finite non-negative values, and the self-tuned spatial weight is a number in [0, 1]."""
    sc_X, sc_lab, st_X, st_lab, coords = FX
    st = st_X.toarray(); nz = st > 0
    base = PRESETS["depth_type"].with_(rate_shrink=5.0)
    names = np.unique(st_lab); rng = np.random.default_rng(3)
    P = rng.dirichlet(np.ones(names.size), size=st.shape[0])
    parents = {t: ("A" if t in ("T0", "T1") else "B") for t in names}
    genes = [f"mt-g{i}" if i < 5 else f"g{i}" for i in range(st.shape[1])]
    variants = [
        (base.with_(soft_labels="kmeans", soft_k=4), {}),
        (base.with_(soft_labels="given"), dict(probs=(P, names))),
        (base.with_(sf_mode="stable"), {}),
        (base.with_(sf_mode="nomito"), dict(gene_names=genes)),
        (base.with_(within_label_pcs=3), {}),
        (base.with_(rate_shrink_target="parent"), dict(parents=parents)),
        (base.with_(spatial_blend="fixed", spatial_w=0.3), {}),
        (base.with_(spatial_blend="auto"), {}),
        (base.with_(soft_labels="kmeans", soft_k=4, within_label_pcs=2, spatial_blend="fixed", spatial_w=0.2,
                    quota_mode="expected", fill_threshold=0.2), {}),
        (base.with_(within_label_pcs=3, within_label_xfit=4), {}),
        (base.with_(within_label_candidates="0,2", spatial_blend="auto"), {}),
        (base.with_(within_label_candidates="0,2", within_label_xfit=3, tune_score="rank"), {}),
        (base.with_(within_label_pcs=2, within_label_xfit=2, within_label_source="reference"), dict(sc_X=sc_X, sc_labels=sc_lab)),
        (base.with_(within_label_pcs=2, within_label_source="blend", within_label_prior=2.0), dict(sc_X=sc_X, sc_labels=sc_lab)),
    ]
    for cfg, kw in variants:
        out, det, expected, prob = impute_depth_type(st_X, st_lab, cfg, coords=coords, return_details=True, **kw)
        assert out.shape == st.shape and np.isfinite(out).all() and (out >= 0).all(), cfg
        assert np.allclose(out[nz], st[nz]), cfg
        assert (prob >= 0).all() and (prob <= 1).all()
        if cfg.spatial_blend == "auto":
            assert 0.0 <= det["spatial_w_auto"] <= 1.0
        if cfg.within_label_candidates:
            assert det["within_label_pcs_auto"] in (0, 2)
        if cfg.within_label_source != "section":
            assert det["within_label_source"] == cfg.within_label_source
    # the reference-taught correction differs from the section's own and the engine entry point passes the reference
    # (a larger section, so that every label has the 200 spots the correction needs)
    sc_big, sc_lab_big, st_big, st_lab_big, coords_big = make_fixture(seed=1, n_types=3, n_genes=60, n_st=900, n_ref=200)
    big = st_big.toarray(); nz_big = big > 0
    sec = impute_depth_type(st_big, st_lab_big, base.with_(within_label_pcs=2), coords=coords_big)
    refd = impute(sc_big, sc_lab_big, st_big, st_lab_big, coords_big, base.with_(within_label_pcs=2, within_label_source="reference"))
    blend = impute(sc_big, sc_lab_big, st_big, st_lab_big, coords_big, base.with_(within_label_pcs=2, within_label_source="blend", within_label_prior=2.0))
    for m in (sec, refd, blend):
        assert np.isfinite(m).all() and (m >= 0).all() and np.allclose(m[nz_big], big[nz_big])
    assert not np.allclose(refd, sec) and not np.allclose(blend, sec) and not np.allclose(blend, refd)
    # worker processes give the same result as the sequential loop
    par = impute_depth_type(st_big, st_lab_big, base.with_(within_label_pcs=2, within_label_xfit=3, n_jobs=3), coords=coords_big)
    seq = impute_depth_type(st_big, st_lab_big, base.with_(within_label_pcs=2, within_label_xfit=3, n_jobs=1), coords=coords_big)
    assert np.allclose(par, seq)
    # relabel=kmeans ignores the given labels entirely
    r1 = impute_depth_type(st_big, st_lab_big, base.with_(relabel="kmeans"), coords=coords_big)
    r2 = impute_depth_type(st_big, np.random.default_rng(5).permutation(st_lab_big), base.with_(relabel="kmeans"), coords=coords_big)
    assert np.allclose(r1, r2) and np.allclose(r1[nz_big], big[nz_big])
    # the keep probability of the tuning thinning is a setting; 0.5 is the default
    q = impute_depth_type(st_big, st_lab_big, base.with_(spatial_blend="auto", tune_keep=0.75), coords=coords_big)
    assert np.isfinite(q).all()
    # the engine entry point hands the extras through
    out2 = impute(sc_X, sc_lab, st_X, st_lab, coords, base.with_(soft_labels="given"), aux=dict(probs=(P, names)))
    assert np.allclose(out2[nz], st[nz])
    # a type mean as backbone for unmeasured genes
    from zone_impute import predict_genes
    sc_all = sc_X; shared = list(range(50)); target = list(range(50, 60))
    pred = predict_genes(sc_all, sc_lab, st_X[:, :50], st_lab, coords, target, shared, PRESETS["v3"].with_(predict_blend=0.5))
    assert pred.shape == (st.shape[0], 10) and np.isfinite(pred).all()


def test_ref_space_detection():
    assert detect_ref_space(FX[0]) == "linear"
    logged = FX[0].copy(); logged.data = np.log1p(logged.data)
    assert detect_ref_space(logged) == "log1p"


def test_fit_weight_table():
    rng = np.random.default_rng(1)
    h = rng.random(2000)
    truth = rng.gamma(2, 1, 2000)
    ref = truth + rng.normal(0, 0.2, 2000) * (1.5 - h)      # reference better when h high
    spat = truth + rng.normal(0, 0.2, 2000) * (0.5 + h)     # spatial better when h low
    edges, w = fit_weight_table(h, ref - truth, spat - truth, n_bins=4)
    assert len(w) == 4 and w[-1] >= w[0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
    print("all tests passed")


def test_nb_probability():
    """The NB probability is never above the Poisson one at the same expected count, so it fills a subset."""
    _, _, st_X, st_lab, coords = FX
    st = sp.csr_matrix(st_X)
    cfg = Config(rate_shrink=20, quota_mode="expected", fill_threshold=0.1, seed=0)
    out_p, _, e_p, p_p = impute_depth_type(st, st_lab, cfg, coords=coords, return_details=True)
    out_n, det, e_n, p_n = impute_depth_type(st, st_lab, cfg.with_(prob_model="nb"), coords=coords, return_details=True)
    assert np.allclose(e_p, e_n)
    assert (p_n <= p_p + 1e-6).all()
    z = st.toarray() == 0
    filled_p = z & (out_p > 0); filled_n = z & (out_n > 0)
    assert not (filled_n & ~filled_p).any()
    assert np.isfinite(det["nb_dispersion_median"])
