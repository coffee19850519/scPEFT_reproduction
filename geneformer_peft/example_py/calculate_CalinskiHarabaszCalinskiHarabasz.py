import os
from sklearn.metrics import calinski_harabasz_score
import scanpy as sc
from pathlib import Path

dataset = "GCBC_with_ensembl_id"
adata = sc.read(f"/mnt/Data3/22frx/Geneformer_unsupervise/raw_adata_with_TFs/case_control/tonsil/{dataset}.h5ad")

del adata.obsm["X_pca"]
del adata.obsm["X_umap"]
# pca
sc.tl.pca(adata)

count = adata.obsm["X_pca"]
labels = adata.obs["annotation_20230508"]
dataset_name = f"tonsil_B_{dataset}"
adata.obs["celltype"] = adata.obs["annotation_20230508"]

CalinskiHarabaszScore = calinski_harabasz_score(X=count, labels=labels)
print(f"Raw count CalinskiHarabaszScore is {CalinskiHarabaszScore}")

sc.pp.neighbors(adata, use_rep="X_pca")
sc.tl.umap(adata)
sc.pl.umap(
    adata,
    color=["celltype"],
    frameon=False,
    show=False,
    save=f"_{dataset_name}_PCA_celltype"
)
