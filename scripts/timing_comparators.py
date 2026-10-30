"""Uninterrupted wall and CPU time of MAGIC (8 threads, as the benchmark shim) and scVI (GPU, default length, as the
shim) on the three main sections, each run started on a CPU at or below 60 C; appends to
results/analysis/timing_parallel.csv under the method names 'magic' and 'scvi'."""
import os, sys, time, subprocess, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/analysis/timing_parallel.csv"
PY = os.path.expanduser("~/envs/zoneimpute-bench/bin/python")

def cpu_temp():
    try:
        for line in subprocess.run(["sensors"], capture_output=True, text=True).stdout.splitlines():
            if "Package id 0" in line:
                return float(line.split("+")[1].split("°")[0])
    except Exception:
        pass
    return None

def wait_cool(limit=60.0):
    t = cpu_temp()
    while t is not None and t > limit:
        time.sleep(10); t = cpu_temp()
    return t

CHILD = r'''
import sys, time, resource, numpy as np, scipy.sparse as sp
sys.path.insert(0, "."); sys.path.insert(0, "src")
from bench import datasets, protocols
ds, method = sys.argv[1], sys.argv[2]
pair = datasets.load(ds); st_all = sp.csr_matrix(pair.st.X); panel = protocols.gene_panel(st_all, "top1500", seed=0)
X = st_all[:, panel].tocsr().astype(np.float32)
t0 = time.time(); c0 = time.process_time()
if method == "magic":
    import magic
    Xd = X.astype(np.float64); lib = np.asarray(Xd.sum(axis=1)).ravel(); lib[lib == 0] = 1.0; med = float(np.median(lib))
    Xs = np.sqrt((sp.diags(med / lib) @ Xd).toarray())
    Y = magic.MAGIC(knn=5, t="auto", random_state=42, n_jobs=8, verbose=False).fit_transform(Xs)
    Y = np.maximum(np.asarray(Y), 0.0) ** 2 * (lib / med)[:, None]
else:
    import anndata as ad, scvi, torch
    scvi.settings.seed = 42
    a = ad.AnnData(X); a.layers["counts"] = a.X.copy()
    scvi.model.SCVI.setup_anndata(a, layer="counts")
    model = scvi.model.SCVI(a, n_latent=10, gene_likelihood="zinb")
    epochs = int(np.clip(400 * 20000 / max(a.n_obs, 1), 50, 400))
    model.train(max_epochs=epochs, accelerator="gpu" if torch.cuda.is_available() else "cpu", enable_progress_bar=False)
    Y = model.get_normalized_expression(library_size="latent", return_numpy=True)
wall = time.time() - t0; cpu = time.process_time() - c0
print(f"RESULT {wall:.2f} {cpu:.2f} {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f}")
'''
rows = []
for ds in ("kidney", "adultB5", "embryo_raw"):
    for method, threads in (("magic", 8), ("scvi", 2)):
        t = wait_cool()
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
        env.pop("ZONEIMPUTE_FORCE_CPU", None)
        p = subprocess.run([PY, "-c", CHILD, ds, method], capture_output=True, text=True, env=env, cwd=str(ROOT))
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
        if not line:
            print(ds, method, "FAILED", p.stderr[-800:], flush=True); continue
        wall, cpu, rss = map(float, line[0].split()[1:])
        rows.append(dict(dataset=ds, n_jobs=method, threads=threads, start_temp_C=t, wall_s=wall, cpu_s=cpu, cpu_self_s=cpu, cpu_workers_s=0.0, parent_maxrss_gb=rss))
        print(ds, method, f"wall {wall:.1f} s, cpu {cpu:.1f} s, start {t} C", flush=True)
with open(OUT, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writerows(rows)
print("COMPARATOR TIMING DONE")
