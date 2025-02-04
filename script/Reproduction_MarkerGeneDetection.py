import copy
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings
import  pandas as pd
from einops import rearrange
from gears import PertData, GEARS
import torch
from anndata import AnnData
import scanpy as sc
import scvi
import numpy as np
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
import argparse
import scib
from tqdm import tqdm


sys.path.insert(0, "../")
from scgpt.trainer import SeqDataset
import scgpt as scg
from scgpt.model.model_prompt import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.preprocess import Preprocessor, TFPreprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, load_pretrained

sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default='COVID',help='dataset name ')
parser.add_argument("--model_path", type=str, default='../checkpoint/marker_gene_detection',help='the path of checkpoint')
parser.add_argument("--data_path", type=str, default='../data/',help='the Path of data')
parser.add_argument("--peft_type", type=str, default='Encoder_adapter',help=' Encoder_adapter/ Token_adapter / Prefix / LoRA / finetune')

args = parser.parse_args(args=[])
key_parameters=vars(args)
hyperparameter_defaults = dict(
    seed=0,
    dataset_name=args.dataset_name,
    load_model=args.model_path+f"/{args.dataset_name}/{args.peft_type}",
    mask_ratio=0.0,
    n_bins=51,
    MVC=False, # Masked value prediction for cell embedding
    ecs_thres=0.0, # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=0.0,
    lr=1e-4,
    layer_size=128,
    batch_size=1,
    nlayers=4,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead=4,  # number of heads in nn.MultiheadAttention
    dropout=0.2,  # dropout probability
    schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
    save_eval_interval=5,
    fast_transformer= False,
    pre_norm=False,
    amp=True,  # Automatic Mixed Precision
    include_zero_gene = True,
    freeze = False, #freeze
    DSBN = False,  # Domain-spec batchnorm
    data_path=args.data_path,
    prompt_type=args.peft_type,  # prefix_prompt/Gene_encoder_prompt/Gene_token_prompt/LoRA
    num_tokens=64,
    n_layers_conf=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # token
    mlp_adapter_conf=[0, 0, 0,0, 0, 0, 0,0,0,0,0,0],
    space_adapter_conf=[1, 1, 1, 1, 1, 1,0,0,0,0,0,0],
    input_style="binned",
    max_seq_len=2001,
    pad_token = "<pad>",
    pad_value=-2,
    input_layer_key="X_binned",
    mask_value=-1,
    use_batch_labels=False
)
peft_prompt_relationship = {
    "Encoder_adapter": "encoder-prompt",
    "Token_adapter": "head-prompt",
    "Prefix": "prefix-prompt",
    "LoRA": "LoRA",
    "finetune": "finetune"
}
hyperparameter_defaults.update(key_parameters)
config = argparse.Namespace(**hyperparameter_defaults)
config.prompt_type = peft_prompt_relationship[config.peft_type]
config.use_prompt = False if config.prompt_type == "finetune" else True
print(config)
set_seed(config.seed)
# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
# mask_ratio = config.mask_ratio
mask_value = -1
pad_value = -2
n_input_bins = config.n_bins

n_hvg = 2000  # number of highly variable genes
max_seq_len = n_hvg + 1
per_seq_batch_sample = False
DSBN = False  # Domain-spec batchnorm
explicit_zero_prob = config.include_zero_gene  # whether explicit bernoulli for zeros
include_zero_gene = config.include_zero_gene

dataset_name = config.dataset_name
logger = scg.logger
data_dir = config.data_path
if dataset_name == 'COVID':
    adata = sc.read(f"{data_dir}/{dataset_name}_test0.h5ad")
    n_hvg = False
    data_is_raw = True
    filter_gene_by_counts = False
    adata.obs["celltype"] = adata.obs["cell_type"].astype("category")
    adata.var["gene_name"] = adata.var.index.tolist()
if config.load_model is not None:
    model_dir = Path(config.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "model_fold0.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    # model
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(
        f"Resume model from {model_file}, the model args will be overriden by the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]
preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=filter_gene_by_counts,  # step 1
    filter_cell_by_counts=False,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=data_is_raw,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=config.n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)
preprocessor(adata, batch_key=None)
input_layer_key = "X_binned"
all_counts = (
    adata.layers[input_layer_key].A
    if issparse(adata.layers[input_layer_key])
    else adata.layers[input_layer_key]
)
genes = adata.var["gene_name"].tolist()
if config.load_model is None:
    vocab = Vocab(
        VocabPybind(genes + special_tokens, None)
    )  # bidirectional lookup [gene <-> int]
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)
ntokens = len(vocab)  # size of vocabulary
prompt_settings = {
    "use_prompt": config.use_prompt,
    "num_tokens": config.num_tokens,
    "prompt_type": config.prompt_type,
    "n_layers_conf": config.n_layers_conf,
    "mlp_adapter_conf": config.mlp_adapter_conf,
    "space_adapter_conf": config.space_adapter_conf
}
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
n_cls = torch.load(model_file, map_location=device)['cls_decoder.out_layer.bias'].shape[0]

model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=3,
    n_cls=n_cls,
    vocab=vocab,
    dropout=config.dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=config.MVC,
    do_dab=False,
    use_batch_labels=False,
    domain_spec_batchnorm=config.DSBN,
    n_input_bins=n_input_bins,
    cell_emb_style="cls",
    mvc_decoder_style="inner product",
    ecs_threshold=config.ecs_thres,
    explicit_zero_prob=False,
    use_fast_transformer=config.fast_transformer,
    fast_transformer_backend="flash",
    pre_norm=config.pre_norm,
    **prompt_settings
)
if config.load_model is not None:
    model.load_state_dict(torch.load(model_file, map_location=device), strict=True)
    logger.info("<All keys matched successfully>")
model.to(device)
def extract_attention(
        model: nn.Module,
        batch_size: int = 1,
        num_attn_layers: int = 11,
        return_new_adata: bool = True,
) -> Optional[AnnData]:
    """extract_attention on dataset of adata_t"""

    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        M = all_gene_ids.size(1)
        N = all_gene_ids.size(0)

        count = 0
        cell_gene_correlations = np.zeros((N, len(gene_ids)), dtype=np.float32)

        for i in tqdm(range(0, N, batch_size), desc="Embedding attention scores"):
            # Replicate the operations in model forward pass
            src_embs = model.encoder(torch.tensor(all_gene_ids[i: i + batch_size], dtype=torch.long).to(device))
            val_embs = model.value_encoder(torch.tensor(all_values[i: i + batch_size], dtype=torch.float).to(device))
            total_embs = src_embs + val_embs
            # Send total_embs to attention layers for attention operations
            # Retrieve the output from second to last layer
            for layer in model.transformer_encoder.layers[:num_attn_layers]:
                total_embs = layer(total_embs, src_key_padding_mask=src_key_padding_mask[i: i + batch_size].to(device))

            # Send total_embs to the last layer in native-attn
            attn_weight = model.transformer_encoder.layers[num_attn_layers].self_attn.in_proj_weight
            attn_bias = model.transformer_encoder.layers[num_attn_layers].self_attn.in_proj_bias
            qkv = F.linear(total_embs, attn_weight, attn_bias)

            # qkv = model.transformer_encoder.layers[num_attn_layers].self_attn.Wqkv(total_embs)
            # Retrieve q, k, and v from native-attn wrapper
            qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=8)
            q = qkv[:, :, 0, :, :]
            k = qkv[:, :, 1, :, :]
            v = qkv[:, :, 2, :, :]

            attn_scores = q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1)

            # Rank normalization by row
            attn_scores = attn_scores.reshape((-1, M))
            order = torch.argsort(attn_scores, dim=1)
            rank = torch.argsort(order, dim=1)
            attn_scores = rank.reshape((-1, 8, M, M)) / M

            # Rank normalization by column
            attn_scores = attn_scores.permute(0, 1, 3, 2).reshape((-1, M))
            order = torch.argsort(attn_scores, dim=1)
            rank = torch.argsort(order, dim=1)
            attn_scores = (rank.reshape((-1, 8, M, M)) / M).permute(0, 1, 3, 2)
            attn_scores = attn_scores.mean(1)

            # Except cls token
            outputs = attn_scores[:, 0, 1:].detach().cpu().numpy()
            cell_gene_correlations[count: count + len(outputs)] = outputs
            count += len(outputs)

    if return_new_adata:
        return sc.AnnData(X=cell_gene_correlations, obs=adata.obs, var=adata.var, dtype="float32")

    return None
input_layer_key = "X_binned"
all_counts = (
    adata.layers[input_layer_key].A
    if issparse(adata.layers[input_layer_key])
    else adata.layers[input_layer_key]
)

tokenized_all = tokenize_and_pad_batch(
    all_counts,
    gene_ids,
    max_len=len(genes) + 1,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,  # append <cls> token at the beginning
    include_zero_gene=include_zero_gene,
)

logger.info(
    f"data set number of samples: {tokenized_all['genes'].shape[0]}, "
    f"feature length: {tokenized_all['genes'].shape[1]}"
)
all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
src_key_padding_mask = all_gene_ids.eq(vocab[pad_token])

# Use this argument to specify which layer to extract the attention weights from
# Default to 11, extraction from the last (12th) layer. Note that index starts from 0
num_attn_layers = 11

attn_adata = extract_attention(
    model=model,
    batch_size=config.batch_size,
    num_attn_layers=num_attn_layers,
    return_new_adata=True
)
# Specify top k number of genes to be selected, and the selection setting
topk = 20
setting = 'difference'
assert setting in ["difference", "target"]

# Only used in 'difference'
target_celltype = ["Effector Memory CD8+ T"]
background_celltype = ["Memory CD8+ T"]
setting="difference"
if setting == 'difference':
    case = np.mean(attn_adata[attn_adata.obs["celltype"].isin(target_celltype)].X, axis=0)
    control = np.mean(attn_adata[attn_adata.obs["celltype"].isin(background_celltype)].X, axis=0)

    enrichment_scores = case - control
    gene_symbol = attn_adata.var.index.tolist()
# Sort in descending order based on enrichment score
df_gene_enrichment = pd.DataFrame({'gene_symbol': gene_symbol, 'scores': enrichment_scores})
df_gene_enrichment = df_gene_enrichment.sort_values(by='scores', ascending=False)

df_gene_enrichment.to_csv(f"enrichment_scores_{config.prompt_type}.csv", index=False)
