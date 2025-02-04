import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Union, Optional
import warnings

import torch
import numpy as np
import matplotlib
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from torch_geometric.loader import DataLoader
from gears import PertData, GEARS
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis
import argparse
import traceback

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model.generation_model_prompt import TransformerGenerator
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, map_raw_id_to_vocab_id, load_pretrained

matplotlib.rcParams["savefig.transparent"] = False
warnings.filterwarnings("ignore")
set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, help='Dataset name')
parser.add_argument("--load_model", type=str, help='Path to peft model')
parser.add_argument("--data_path", type=str, help='Path to dataset')
parser.add_argument("--peft_type", type=str, help='Encoder_adapter/Token_adapter/Prefix/LoRA/finetune')

args = parser.parse_args()
key_parameters = vars(args)

config = argparse.Namespace(**key_parameters)
peft_prompt_relationship = {
    "Encoder_adapter": "encoder-prompt",
    "Token_adapter": "head-prompt",
    "Prefix": "prefix-prompt",
    "LoRA": "LoRA",
    "finetune": "finetune"
}

config.prompt_type = peft_prompt_relationship[config.peft_type]
config.use_prompt = False if config.prompt_type == "finetune" else True

# settings for data prcocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
pad_value = 0  # for padding values
pert_pad_id = 2

n_hvg = 0  # number of highly variable genes
include_zero_gene = "all"  # include zero expr genes in training input, "all", "batch-wise", "row-wise", or False
max_seq_len = 1536

# settings for training
MLM = True  # whether to use masked language modeling, currently it is always on.
CLS = False  # celltype classification objective
CCE = False  # Contrastive cell embedding objective
MVC = False  # Masked value prediction for cell embedding
ECS = False  # Elastic cell similarity objective
cell_emb_style = "cls"
mvc_decoder_style = "inner product, detach"
amp = True

# settings for optimizer
batch_size = 20
eval_batch_size = 20

# settings for the model
embsize = 512  # embedding dimension
d_hid = 512  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 12  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8  # number of heads in nn.MultiheadAttention
n_layers_cls = 3
dropout = 0.2  # dropout probability
use_fast_transformer = False  # whether to use fast transformer

# settings for prompt model
use_prompt = config.use_prompt
prompt_type = config.prompt_type
num_tokens = 20

n_layers_conf = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # token
mlp_adapter_conf = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
space_adapter_conf = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

if config.prompt_type == "LoRA":
    space_adapter_conf = n_layers_conf

# dataset and evaluation choices
split = "simulation"
data_name = config.dataset_name

pert_data = PertData(config.data_path)
pert_data.load(data_name=data_name)
pert_data.prepare_split(split=split, seed=1)
pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)

if config.load_model is not None:
    model_dir = Path(config.load_model, config.dataset_name, config.peft_type)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    pert_data.adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
    genes = pert_data.adata.var["gene_name"].tolist()

    # model
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]
else:
    genes = pert_data.adata.var["gene_name"].tolist()
    vocab = Vocab(
        VocabPybind(genes + special_tokens, None)
    )  # bidirectional lookup [gene <-> int]
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(
    [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
)
n_genes = len(genes)

ntokens = len(vocab)  # size of vocabulary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerGenerator(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=n_layers_cls,
    n_cls=1,
    vocab=vocab,
    dropout=dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    pert_pad_id=pert_pad_id,
    do_mvc=MVC,
    cell_emb_style=cell_emb_style,
    mvc_decoder_style=mvc_decoder_style,
    use_fast_transformer=use_fast_transformer,
    # Prompt settings
    use_prompt=use_prompt,
    num_tokens=num_tokens,
    prompt_type=prompt_type,
    n_layers_conf=n_layers_conf,
    mlp_adapter_conf=mlp_adapter_conf,
    space_adapter_conf=space_adapter_conf,
)

if config.load_model is not None:
    try:
        model.load_state_dict(torch.load(model_file, map_location=device))
        print(f"Loading all model params from {model_file}")
    except Exception as e:
        use_flash_attn = getattr(model, "use_fast_transformer", True)
        pretrained_dict = torch.load(model_file, map_location='cpu')

        # if not use_flash_attn and prompt_type != "LoRA":
        #     pretrained_dict = {
        #         k.replace("Wqkv.", "in_proj_"): v for k, v in pretrained_dict.items()
        #     }

        # if not use_flash_attn and prompt_type == "LoRA":
        #     for idx, conf in enumerate(space_adapter_conf):
        #         if not conf:
        #             pretrained_dict = {
        #                 k.replace("Wqkv.", "in_proj_") if f"transformer_encoder.layers.{idx}.self_attn" in k else k: v
        #                 for k, v in pretrained_dict.items()
        #             }

        model_dict = model.state_dict()

        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Check the results, if there are some problems please contact the developer")

model.to(device)


def eval_perturb(
        loader: DataLoader, model: TransformerGenerator, device: torch.device
) -> Dict:
    """
    Run model in inference mode using a given data loader
    """

    model.eval()
    model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}
    logvar = []

    for itr, batch in enumerate(loader):
        batch.to(device)
        pert_cat.extend(batch.pert)

        with torch.no_grad():
            p = model.pred_perturb(batch, include_zero_gene, gene_ids=gene_ids)
            t = batch.y
            pred.extend(p.cpu())
            truth.extend(t.cpu())

            # Differentially expressed genes
            for itr, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[itr, de_idx])
                truth_de.append(t[itr, de_idx])

    # all genes
    results["pert_cat"] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results["pred"] = pred.detach().cpu().numpy().astype(np.float64)
    results["truth"] = truth.detach().cpu().numpy().astype(np.float64)

    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results["pred_de"] = pred_de.detach().cpu().numpy().astype(np.float64)
    results["truth_de"] = truth_de.detach().cpu().numpy().astype(np.float64)

    return results


test_loader = pert_data.dataloader["test_loader"]
test_res = eval_perturb(test_loader, model, device)
test_metrics, test_pert_res = compute_metrics(test_res)
for k, v in test_metrics.items():
    print(f"{k} -> {v}")
