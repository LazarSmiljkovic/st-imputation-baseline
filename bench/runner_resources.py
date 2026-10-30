"""
bench/runner_resources.py -- wall time, CPU time, peak memory and GPU memory of each method, one process per run.

The benchmark's own peak_rss_gb is not usable for comparison: it is the growth of the parent process's
high-water mark, so a method run after a larger one shows zero, and the external packages run in a
subprocess and show nothing.  Here every (section, method) pair runs in a fresh process under
``/usr/bin/time -v`` (peak resident size over the whole process tree, user and system CPU seconds,
wall clock), with GPU memory sampled from nvidia-smi once a second.  Before each run the orchestrator
waits until no other benchmark process is running and the CPU package is below --cool C, so that the
timings are not inflated by the thermal guard or by concurrent jobs.  The masking is the benchmark's
(P1, top-1500 panel, a fifth of the non-zeros hidden, seed 42) and the per-gene Spearman is recorded as
a sanity check.

    python -m bench.runner_resources --dataset kidney --methods gene_depth "ours:depth_type;rate_shrink=20" magic \
        --out results/analysis/resources.csv
    python -m bench.runner_resources --table --out results/analysis/resources.csv > results/analysis/RESOURCES.md

Columns: dataset, method, device, n_spots, n_genes, status, wall_s (whole process incl. loading),
load_s, method_s (the denoise call alone), cpu_s (user + system, whole process), peak_rss_gb (process
tree), rss_after_load_gb, gpu_peak_mb (above the idle level), pergene_spearman, started.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("ZONEIMPUTE_THREADS", "4"))

import argparse
import csv
import json
import re
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

OTHER_JOBS = re.compile(r"bench\.runner |bench/external/run_|count_gof\.py|term_ablation\.py run")


# ----------------------------------------------------------------------------- one run (child process)

def _vmrss_gb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9


class RssSampler:
    """Peak resident size of this process while the block runs (the high-water mark of getrusage is useless here:
    loading the data leaves a transient above which a cheap method never rises)."""

    def __init__(self, every=0.05):
        self.peak = _vmrss_gb(); self.every = every; self.stop = threading.Event()
        self.t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, _vmrss_gb()); time.sleep(self.every)

    def __enter__(self):
        self.t.start(); return self

    def __exit__(self, *exc):
        self.stop.set(); self.t.join(timeout=2); self.peak = max(self.peak, _vmrss_gb())


def run_one(a):
    import numpy as np
    from bench import datasets, metrics, runner, methods
    t0 = time.time()
    ns = argparse.Namespace(dataset=a.dataset, labels=a.labels, reference=a.reference, coarse=False)
    pair = datasets.load(a.dataset, **runner._load_kw(ns))
    ctx = runner._RecoveryContext(pair, a.genes)
    masked, rows, cols, truth, st_ctx = ctx.hide("p1", a.seed, mask=a.mask)
    embed_X = (ctx.sc_full, st_ctx) if a.genes != "all" else None
    load_s = time.time() - t0
    rss_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    work = Path(a.work) / re.sub(r"[^A-Za-z0-9_.=-]+", "_", f"{a.dataset}_{a.method}")
    # an external package runs in a subprocess whose peak is usually below this process's data footprint, so the
    # tree-wide maximum would hide it: launch the shim under /usr/bin/time and read its own peak from a file
    work.mkdir(parents=True, exist_ok=True)
    tfile = work / "time_v.txt"; tfile.unlink(missing_ok=True)
    real_run = subprocess.run

    def timed_run(cmd, *args, **kw):
        if isinstance(cmd, list) and cmd and str(cmd[1]).endswith(".py") and "/bench/external/" in str(cmd[1]):
            cmd = ["/usr/bin/time", "-v", "-o", str(tfile)] + list(cmd)
        return real_run(cmd, *args, **kw)

    subprocess.run = timed_run
    ru0 = resource.getrusage(resource.RUSAGE_SELF); rss_before = _vmrss_gb()
    t1 = time.time()
    try:
        with RssSampler() as rs:
            pred_full = methods.denoise(a.method, masked, ctx.sc_csr, ctx.sc_labels, ctx.st_labels, ctx.coords, ctx.sc_space, a.seed, work,
                                        st_genes=ctx.genes, sc_genes=ctx.genes, embed_X=embed_X)
    finally:
        subprocess.run = real_run
    method_s = time.time() - t1
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    method_cpu = round(ru1.ru_utime + ru1.ru_stime - ru0.ru_utime - ru0.ru_stime, 1)   # this process only
    method_rss_added = round(max(rs.peak - rss_before, 0.0), 2)                         # growth of this process during the method
    method_rss = None
    if tfile.exists():                                                                 # external package: its own process
        tv = parse_time_v(tfile.read_text())
        method_rss = tv["peak_rss_gb"]
        if tv["user_s"] is not None:
            method_cpu = round(tv["user_s"] + tv["sys_s"], 1)
    ok = ctx.imputable[rows]
    m, _ = metrics.masked_metrics(truth[ok], pred_full[rows, cols].astype(np.float64)[ok], cols[ok])
    print("RESOURCES_JSON " + json.dumps(dict(load_s=round(load_s, 1), method_s=round(method_s, 1), rss_after_load_gb=round(rss_load, 2),
                                             method_process_rss_gb=method_rss, method_rss_added_gb=method_rss_added, method_cpu_s=method_cpu,
                                             pergene_spearman=round(m["pergene_spearman_wmean"], 4), n_spots=int(masked.shape[0]),
                                             n_genes=int(masked.shape[1]))), flush=True)


# ----------------------------------------------------------------------------- orchestrator

def cpu_temp():
    try:
        out = subprocess.run(["sensors"], capture_output=True, text=True, timeout=10).stdout
        mm = re.search(r"Package id 0:\s*\+?([0-9.]+)", out)
        return float(mm.group(1)) if mm else None
    except Exception:
        return None


def other_jobs():
    out = subprocess.run(["pgrep", "-af", "python"], capture_output=True, text=True).stdout
    me = str(os.getpid())
    return [l for l in out.splitlines() if OTHER_JOBS.search(l) and not l.startswith(me + " ")]


def wait_until_quiet(cool, log, max_wait=7200):
    waited = 0
    while True:
        jobs = other_jobs(); t = cpu_temp()
        if not jobs and (t is None or t <= cool):
            return waited
        if waited >= max_wait:
            log(f"  waited {waited}s; proceeding anyway with {len(jobs)} other job(s) running, CPU {t} C")
            return waited
        if waited % 300 == 0:
            log(f"  waiting: {len(jobs)} other benchmark job(s), CPU {t} C")
        time.sleep(15); waited += 15


class GpuSampler:
    def __init__(self):
        self.base = self._used(); self.peak = self.base; self.stop = threading.Event()
        self.t = threading.Thread(target=self._loop, daemon=True)

    @staticmethod
    def _used():
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
            return float(out.strip().splitlines()[0])
        except Exception:
            return None

    def _loop(self):
        while not self.stop.is_set():
            u = self._used()
            if u is not None and self.peak is not None:
                self.peak = max(self.peak, u)
            time.sleep(1)

    def __enter__(self):
        if self.base is not None:
            self.t.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        if self.base is not None:
            self.t.join(timeout=5)

    @property
    def above_idle_mb(self):
        return None if self.base is None else round(self.peak - self.base, 0)


def parse_time_v(text):
    def grab(pat, conv=float):
        mm = re.search(pat, text)
        return conv(mm.group(1)) if mm else None
    wall = grab(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([0-9:.]+)", str)
    if wall:
        parts = [float(p) for p in wall.split(":")]
        wall = parts[0] * 3600 + parts[1] * 60 + parts[2] if len(parts) == 3 else parts[0] * 60 + parts[1]
    return dict(wall_s=wall, user_s=grab(r"User time \(seconds\): ([0-9.]+)"), sys_s=grab(r"System time \(seconds\): ([0-9.]+)"),
                peak_rss_gb=grab(r"Maximum resident set size \(kbytes\): ([0-9]+)", lambda v: round(int(v) / 1e6, 2)),
                exit_status=grab(r"Exit status: ([0-9]+)", int))


def orchestrate(a):
    import pandas as pd
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    logf = open(a.log, "a") if a.log else None

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        if logf:
            logf.write(line + "\n"); logf.flush()

    for spec in a.methods:
        gpu = a.gpu and spec.split(":")[0] in ("scvi", "gimvi", "tangram", "envi")
        env = dict(os.environ, ZONEIMPUTE_THREADS=str(a.threads), ZONEIMPUTE_FORCE_CPU="0" if gpu else "1")
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[v] = str(a.threads)
        waited = 0 if a.no_wait else wait_until_quiet(a.cool, log, a.max_wait)
        cmd = ["/usr/bin/time", "-v", sys.executable, "-m", "bench.runner_resources", "--one", "--dataset", a.dataset, "--method", spec,
               "--seed", str(a.seed), "--genes", a.genes, "--labels", a.labels, "--reference", a.reference, "--work", a.work]
        log(f"[run ] {a.dataset} {spec} ({'gpu' if gpu else 'cpu'}, waited {waited}s, CPU {cpu_temp()} C)")
        started = time.strftime("%Y-%m-%d %H:%M:%S")
        with GpuSampler() as g:
            p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(ROOT))
        tv = parse_time_v(p.stderr)
        mm = re.search(r"RESOURCES_JSON (\{.*\})", p.stdout)
        inner = json.loads(mm.group(1)) if mm else {}
        status = "ok" if p.returncode == 0 and mm else "failed"
        row = dict(dataset=a.dataset, method=spec, device="gpu" if gpu else "cpu", threads=a.threads, status=status, started=started,
                   wall_s=tv["wall_s"], load_s=inner.get("load_s"), method_s=inner.get("method_s"),
                   cpu_s=None if tv["user_s"] is None else round(tv["user_s"] + tv["sys_s"], 1), peak_rss_gb=tv["peak_rss_gb"],
                   rss_after_load_gb=inner.get("rss_after_load_gb"), method_process_rss_gb=inner.get("method_process_rss_gb"),
                   method_rss_added_gb=inner.get("method_rss_added_gb"), method_cpu_s=inner.get("method_cpu_s"),
                   gpu_peak_mb=g.above_idle_mb, pergene_spearman=inner.get("pergene_spearman"),
                   n_spots=inner.get("n_spots"), n_genes=inner.get("n_genes"), seed=a.seed, genes=a.genes)
        if status == "failed":
            err = (p.stderr.strip().splitlines() or ["?"])[-1] if not mm else "no result line"
            row["error"] = err[:300]
        append_row(out, row)
        log(f"   {status}: wall {row['wall_s']} s, method {row['method_s']} s, cpu {row['cpu_s']} s, peak {row['peak_rss_gb']} GB, "
            f"gpu +{row['gpu_peak_mb']} MB, rho {row['pergene_spearman']}" + (f", error: {row.get('error')}" if status == 'failed' else ""))
    if logf:
        logf.close()


def append_row(out: Path, row: dict):
    """Append one row, keeping the file's existing header (missing fields empty, new fields added by rewriting)."""
    import pandas as pd
    if not out.exists():
        pd.DataFrame([row]).to_csv(out, index=False); return
    with open(out, newline="") as f:
        header = next(csv.reader(f))
    if set(row) <= set(header):
        pd.DataFrame([{k: row.get(k) for k in header}]).to_csv(out, mode="a", header=False, index=False)
        return
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    rows.append({k: ("" if v is None else v) for k, v in row.items()})
    cols = header + [k for k in row if k not in header]
    pd.DataFrame(rows).reindex(columns=cols).to_csv(out, index=False)


def table(a):
    import pandas as pd
    d = pd.read_csv(a.out)
    d = d[d.status == "ok"].drop_duplicates(["dataset", "method", "device"], keep="last")
    L = ["# Time and memory per method, one process per run", "",
         "`method memory` is what the method needs: for a method running in the harness process, the growth of that process's "
         "resident size during the call (sampled every 50 ms); for an external package, the peak resident size of its own "
         "process, which holds only the 1,500-gene panel and the reference. `process peak` is the maximum over the whole tree "
         "measured with `/usr/bin/time -v`, data loading included, and `after load` the resident size once the section, reference "
         "and mask are in memory. `method CPU s` is user plus system time of the method alone, unaffected by pauses of the "
         "thermal guard, which can inflate `method s`. GPU memory is the peak above the idle level sampled once a second. Runs are started only when no other "
         "benchmark job is running and the CPU package is cool. Masking as in the benchmark (top 1,500 genes, a fifth of the "
         "non-zeros hidden, seed 42).", ""]
    for ds, g in d.groupby("dataset", sort=False):
        g = g.sort_values("wall_s")
        L += [f"## {ds} ({int(g.n_spots.iloc[0]):,} spots x {int(g.n_genes.iloc[0]):,} genes)", "",
              "| method | device | method s | method CPU s | method memory GB | process peak GB | after load GB | GPU MB | per-gene Spearman |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            ext = "method_process_rss_gb" in r and not pd.isna(r.method_process_rss_gb)
            if ext:
                mem = f"{r.method_process_rss_gb:.2f} (own process)"
            elif "method_rss_added_gb" in r and not pd.isna(r.method_rss_added_gb):
                mem = f"{r.method_rss_added_gb:.2f} (added)"
            else:
                mem = f"{max(r.peak_rss_gb - r.rss_after_load_gb, 0):.2f} (above loaded data)"
            cpu = r.method_cpu_s if "method_cpu_s" in r and not pd.isna(r.method_cpu_s) else r.cpu_s
            L.append(f"| {r.method} | {r.device} | {r.method_s:.1f} | {cpu:.1f} | {mem} | {r.peak_rss_gb:.2f} | {r.rss_after_load_gb:.2f} | "
                     f"{'' if pd.isna(r.gpu_peak_mb) else int(r.gpu_peak_mb)} | {r.pergene_spearman:.3f} |")
        L.append("")
    print("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--one", action="store_true", help="(internal) run one method in this process and print its result line")
    ap.add_argument("--table", action="store_true", help="print a markdown table from --out")
    ap.add_argument("--dataset", default="kidney"); ap.add_argument("--labels", default="codi"); ap.add_argument("--reference", default="stored")
    ap.add_argument("--genes", default="top1500"); ap.add_argument("--mask", type=float, default=0.2); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--method", default=None); ap.add_argument("--methods", nargs="+", default=[])
    ap.add_argument("--out", default="results/analysis/resources.csv"); ap.add_argument("--log", default="")
    ap.add_argument("--work", default="results/bench/resources_work")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("ZONEIMPUTE_THREADS", "4")))
    ap.add_argument("--cool", type=float, default=75.0, help="start a run only when the CPU package is at or below this temperature")
    ap.add_argument("--gpu", action="store_true", help="let scvi/gimvi/tangram/envi use the GPU (otherwise CPU is forced)")
    ap.add_argument("--no-wait", action="store_true", help="do not wait for other benchmark jobs or a cool CPU (testing only)")
    ap.add_argument("--max-wait", type=int, default=7200, help="seconds to wait for a quiet machine before running anyway")
    a = ap.parse_args(argv)
    if a.table:
        return table(a)
    if a.one:
        return run_one(a)
    return orchestrate(a)


if __name__ == "__main__":
    main()
