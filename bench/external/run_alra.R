# Calls the authors' ALRA (third_party/ALRA, MIT) on a log-normalised matrix written by run_alra.py.
args <- commandArgs(trailingOnly = TRUE)
work <- args[1]; seed <- as.integer(args[2]); root <- args[3]
suppressMessages({library(Matrix); library(rsvd)})
source(file.path(root, "third_party", "ALRA", "R", "alra.R"))
set.seed(seed)
A <- as.matrix(readMM(file.path(work, "alra_in.mtx")))
# the authors' defaults (K = 100 candidate components, noise estimated from component 80 on) need more genes than
# the small imaging panels have; there K is capped below the panel size with the same 80 % ratio
Kmax <- min(100, ncol(A) - 1, nrow(A) - 1)
k <- choose_k(A, K = Kmax, noise_start = floor(0.8 * Kmax))$k
res <- alra(A, k = k)
out <- res$A_norm_rank_k_cor_sc
writeBin(as.vector(t(out)), file.path(work, "alra_out.bin"), size = 8)
writeLines(as.character(c(nrow(out), ncol(out), k)), file.path(work, "alra_dims.txt"))
