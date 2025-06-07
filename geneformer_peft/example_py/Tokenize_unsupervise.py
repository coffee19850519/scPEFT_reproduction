# take any anndata file
# ...
# tokenize data
from geneformer import TranscriptomeTokenizer

tk = TranscriptomeTokenizer({"cell_type": "cell_type"}, nproc=4)  # some annotation
dataset_name = "GCBC"
tk.tokenize_data(adata_scale=None,
                 data_directory=f"/media/fei/Data/frx/Geneformer_unsupervise/example_py/data/tonsil/",
                 output_directory="/media/fei/Data/frx/Geneformer_unsupervise/example_py/data/tonsil/",
                 output_prefix=f"{dataset_name}",
                 file_format="h5ad")
