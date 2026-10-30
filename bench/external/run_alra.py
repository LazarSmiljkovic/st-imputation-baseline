"""ALRA (Linderman et al. 2022) on the ST matrix alone, through the authors' R code (third_party/ALRA, MIT).
The input is normalised as ALRA's normalize_data does (library to 10,000, natural log1p), spots without counts
are kept as all-zero rows; ALRA's completed matrix is mapped back to counts with expm1 and each spot's library."""
import os, subprocess, numpy as np, scipy.sparse as sp, scipy.io
from pathlib import Path
from common import parse, load, save
args = parse()
st, sc, held = load(args.work)
if args.task != "denoise":
    raise SystemExit("ALRA only denoises measured genes")
work = Path(args.work); root = Path(__file__).resolve().parents[2]
X = sp.csr_matrix(st.X).astype(np.float64)
lib = np.asarray(X.sum(axis=1)).ravel(); ok = lib > 0
Xn = sp.diags(np.where(ok, 1e4 / np.where(ok, lib, 1.0), 0.0)) @ X
Xn.data = np.log1p(Xn.data)
scipy.io.mmwrite(str(work / "alra_in.mtx"), sp.csr_matrix(Xn[ok]))
rscript = os.environ.get("ZONEIMPUTE_ALRA_RSCRIPT", os.path.expanduser("~/envs/zoneimpute-alra/bin/Rscript"))
subprocess.run([rscript, str(Path(__file__).with_suffix(".R")), str(work), str(args.seed), str(root)], check=True)
n, g, k = [int(v) for v in (work / "alra_dims.txt").read_text().split()]
out = np.fromfile(work / "alra_out.bin", dtype=np.float64).reshape(n, g)
Y = np.zeros(X.shape)
Y[ok] = np.expm1(np.maximum(out, 0.0)) * (lib[ok] / 1e4)[:, None]
print("ALRA rank k =", k)
save(args.work, Y)
