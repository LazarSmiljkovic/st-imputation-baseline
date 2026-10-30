"""stPlus (Chen et al. 2021) via the stPlus package. predict only (held genes)."""
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from common import parse, load, save, ref_linear

args = parse()
work = Path(args.work)
st, sc, held = load(args.work)
if args.task != "predict":
    raise SystemExit("stPlus is run in the gene-prediction protocol only")
# stPlus hard-codes CUDA (default tensor type and .cuda() calls).  When no GPU is
# usable, neutralise those calls at runtime so the published code runs on the CPU;
# the package files themselves are not modified.
if not torch.cuda.is_available():
    torch.set_default_tensor_type = lambda *a, **k: None
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.nn.Module.cuda = lambda self, *a, **k: self
from stPlus import stPlus



def norm_log(X):
    """stPlus wants normalised, log-transformed cell x gene data."""
    X = sp.csr_matrix(X).astype(np.float64)
    lib = np.asarray(X.sum(axis=1)).ravel(); lib[lib == 0] = 1.0
    X = sp.diags(1e4 / lib) @ X
    X.data = np.log1p(X.data)
    return X.toarray()


RNA = pd.DataFrame(norm_log(ref_linear(sc)), index=sc.obs_names, columns=sc.var_names)
SP = pd.DataFrame(norm_log(st.X), index=st.obs_names, columns=st.var_names)
sig = inspect.signature(stPlus)
kw = dict(spatial_df=SP, scrna_df=RNA, genes_to_predict=np.asarray(list(held)),
          save_path_prefix=str(work / "stPlus"), random_seed=args.seed, verbose=False)
kw = {k: v for k, v in kw.items() if k in sig.parameters}
pred = stPlus(**kw)
pred = pred[list(held)] if isinstance(pred, pd.DataFrame) else pd.DataFrame(pred, columns=list(held))
save(args.work, pred.values, scale="log")
