"""Wall and CPU time of the final configuration with 1, 4, 8 and 16 worker processes on the three main sections,
each run started only when the CPU package is cool (<= 60 C) and timed without interruption.  Writes
results/analysis/timing_parallel.csv.  Threads per worker are 16 / n_jobs; the sequential run uses 4 threads as in
bench/runner_resources.py."""
import os, sys, time, subprocess, resource, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/analysis/timing_parallel.csv"
NEW = "ours:depth_type;rate_shrink=20;within_label_pcs=5;within_label_xfit=5;spatial_blend=auto;quota_mode=expected;fill_threshold=0.1"

def cpu_temp():
    try:
        out = subprocess.run(["sensors"], capture_output=True, text=True).stdout
        for line in out.splitlines():
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
import sys, time, os, resource, numpy as np, scipy.sparse as sp
sys.path.insert(0, "."); sys.path.insert(0, "src")
from bench import datasets, protocols
from bench.methods import parse_ours
from zone_impute import impute_depth_type
ds, spec = sys.argv[1], sys.argv[2]
pair = datasets.load(ds); st_all = sp.csr_matrix(pair.st.X); panel = protocols.gene_panel(st_all, "top1500", seed=0)
X = st_all[:, panel].tocsr(); genes = list(np.asarray(pair.st.var_names)[panel]); lab = np.asarray(pair.st_label)
cfg = parse_ours(spec).with_(seed=42)
t0 = time.time(); c0 = time.process_time()
out = impute_depth_type(X, lab, cfg, coords=pair.coords, gene_names=genes)
wall = time.time() - t0; cpu_self = time.process_time() - c0
ch = resource.getrusage(resource.RUSAGE_CHILDREN); cpu_children = ch.ru_utime + ch.ru_stime
print(f"RESULT {wall:.2f} {cpu_self:.2f} {cpu_children:.2f} {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f}")
'''
rows = []
for ds in ("kidney", "adultB5", "embryo_raw"):
    for n_jobs, threads in ((1, 4), (4, 4), (8, 2), (16, 1)):
        t = wait_cool()
        env = dict(os.environ, ZONEIMPUTE_FORCE_CPU="1", OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
        spec = NEW if n_jobs == 1 else f"{NEW};n_jobs={n_jobs}"
        p = subprocess.run([sys.executable, "-c", CHILD, ds, spec], capture_output=True, text=True, env=env, cwd=str(ROOT))
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
        if not line:
            print(ds, n_jobs, "FAILED", p.stderr[-500:], flush=True); continue
        wall, cpu_self, cpu_children, rss = map(float, line[0].split()[1:])
        rows.append(dict(dataset=ds, n_jobs=n_jobs, threads=threads, start_temp_C=t, wall_s=wall, cpu_s=cpu_self + cpu_children, cpu_self_s=cpu_self, cpu_workers_s=cpu_children, parent_maxrss_gb=rss))
        print(ds, n_jobs, f"wall {wall:.1f} s, cpu {cpu_self + cpu_children:.1f} s (workers {cpu_children:.1f}), start {t} C", flush=True)
        with open(OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("TIMING DONE")
