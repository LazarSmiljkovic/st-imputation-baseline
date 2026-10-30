"""scVI (Lopez et al. 2018) trained on the ST counts alone; denoised values are the
model's expected counts at each spot's own library size."""
import numpy as np
import scipy.sparse as sp
import torch
from common import parse, load, save

args = parse()
st, sc, held = load(args.work)
if args.task != "denoise":
    raise SystemExit("scVI here only denoises measured genes")
import scvi

scvi.settings.seed = args.seed
st.X = sp.csr_matrix(st.X).astype(np.float32)
st.layers["counts"] = st.X.copy()
scvi.model.SCVI.setup_anndata(st, layer="counts")
model = scvi.model.SCVI(st, n_latent=10, gene_likelihood="zinb")
epochs = args.epochs or int(np.clip(400 * 20000 / max(st.n_obs, 1), 50, 400))
model.train(max_epochs=epochs, accelerator="gpu" if torch.cuda.is_available() else "cpu", enable_progress_bar=False)
if args.mode in ("labels", "labels_kmeans"):
    # scANVI: the same model given the spot labels that the label-based methods receive (every spot labelled)
    st.obs["label"] = st.obs["label"].astype(str)
    if args.mode == "labels_kmeans":
        # labels from the masked input itself, computed as the proposed estimate's relabel="kmeans" option
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.decomposition import TruncatedSVD
        X = st.layers["counts"].tocsr(copy=True); lib = np.asarray(X.sum(1)).ravel(); lib[lib == 0] = 1.0
        X = sp.diags((1e4 / lib).astype(np.float32)) @ X; X.data = np.log1p(X.data)
        Z = TruncatedSVD(n_components=min(30, X.shape[1] - 1), random_state=args.seed).fit_transform(X)
        k = st.obs["label"].nunique()
        st.obs["label"] = [f"k{c}" for c in MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=4096, random_state=args.seed).fit_predict(Z)]
    model = scvi.model.SCANVI.from_scvi_model(model, unlabeled_category="__unlabelled__", labels_key="label")
    model.train(max_epochs=max(20, epochs // 4), accelerator="gpu" if torch.cuda.is_available() else "cpu", enable_progress_bar=False)
Y = model.get_normalized_expression(library_size="latent", return_numpy=True)
save(args.work, np.asarray(Y))
