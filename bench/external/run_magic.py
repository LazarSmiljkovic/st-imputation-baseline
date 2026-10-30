"""MAGIC (van Dijk et al. 2018) on the ST matrix alone, via the magic-impute package.
Values are returned on the count scale (library normalisation and sqrt are undone)."""
import numpy as np
import scipy.sparse as sp
from common import parse, load, save

args = parse()
st, sc, held = load(args.work)
if args.task != "denoise":
    raise SystemExit("MAGIC only denoises measured genes")
import magic

X = sp.csr_matrix(st.X).astype(np.float64)
lib = np.asarray(X.sum(axis=1)).ravel(); lib[lib == 0] = 1.0
med = float(np.median(lib))
Xn = sp.diags(med / lib) @ X
Xn = Xn.toarray()
Xs = np.sqrt(Xn)
op = magic.MAGIC(knn=5, t="auto", random_state=args.seed, n_jobs=8, verbose=False)
Y = op.fit_transform(Xs)
Y = np.maximum(np.asarray(Y), 0.0) ** 2 * (lib / med)[:, None]
save(args.work, Y)
