from geneformer.emb_extractor_h5ad import EmbExtractor
import anndata as ad
from collections import Counter
import scanpy as sc
from sklearn.metrics import calinski_harabasz_score
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
cuda_index = 1
dataset_names = ["GCBC"]
model_path = '/mnt/Data3/22frx/Geneformer_unsupervise/example_py/Geneformer_ckpts/GCBC/Gene_token_prompt/checkpoint-35'

for dataset_name in dataset_names:
    arrow_dataset_path = f'/mnt/Data3/22frx/Geneformer_unsupervise/data/case_control/tonsil/{dataset_name}.dataset'

    # load adata
    h5ad_dataset_path_train = f'/mnt/Data3/22frx/Geneformer_unsupervise/raw_adata_with_TFs/case_control/tonsil/{dataset_name}_with_ensembl_id.h5ad'
    adata = ad.read(h5ad_dataset_path_train)

    # # initiate EmbExtractor
    embex = EmbExtractor(model_type="Prompt",
                         num_classes=1,
                         filter_data=None,
                         max_ncells=100000,
                         emb_layer=0,
                         emb_label=["cell_type"],
                         labels_to_plot=["cell_type"],
                         forward_batch_size=80,
                         nproc=16)

    # extracts embedding from input data
    embs = embex.extract_embs(model_directory=model_path, input_data_file=arrow_dataset_path)
    adata.obsm['cell_embedding'] = embs.detach().cpu().numpy()

    embedding = adata.obsm['cell_embedding']
    labels = adata.obs["annotation_20230508"]

    CalinskiHarabaszScore = calinski_harabasz_score(X=embedding, labels=labels)
    print(f"Raw count CalinskiHarabaszScore is {CalinskiHarabaszScore}")

    sc.pp.neighbors(adata, use_rep="cell_embedding")
    sc.tl.umap(adata, min_dist=0.3)
    fig = sc.pl.umap(
        adata,
        color=["cell_type"],
        title=[
            f"CH Index = {CalinskiHarabaszScore:.4f}",
        ],
        frameon=False,
        show=False,
        save=f"_{dataset_name}_Unsupervised_Token_prompt_embeddings_celltype"
    )

    # save adata to ckpt path, at the same time, I will rename adata
    adata.write_h5ad(
        f"/mnt/Data1/frx/Geneformer/fine_tuned_models/geneformer_dataset{dataset_name}_CellClassifier/dataset_{dataset_name}_train_cell_embedding.h5ad")
