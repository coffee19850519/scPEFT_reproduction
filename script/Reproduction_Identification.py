import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import List, Tuple, Dict, Union, Optional
import warnings
import pandas as pd
import torch
import scanpy as sc
import numpy as np
from scipy.sparse import issparse
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
import argparse
sys.path.insert(0, "../")
import scgpt as scg
from scgpt.trainer import (
    prepare_data,
    test
)
from scgpt.model.model_prompt import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor, TFPreprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics
from scgpt.reproduction_util import load_and_process_data, get_weighted_sampler
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default='ms',help='ms/COVID/NSCLC/MergedMonkey/mouse_115746/mouse_10x/mouse_smart/elegans')
parser.add_argument("--model_path", type=str, default='../checkpoint/celltype_identification',help='the path of checkpoint')
parser.add_argument("--data_path", type=str, default='../data/celltype_identification',help='the Path of data')
parser.add_argument("--peft_type", type=str, default='Encoder_adapter',help=' Encoder_adapter/ Token_adapter / Prefix / LoRA / finetune')

args = parser.parse_args()
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
    batch_size=20,
    layer_size=128,
    nlayers=4,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead=4,  # number of heads in nn.MultiheadAttention
    dropout=0.2,  # dropout probability
    schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
    save_eval_interval=5,
    fast_transformer= False,
    pre_norm=False,
    amp=True,  # Automatic Mixed Precision
    include_zero_gene = False,
    freeze = False, #freeze
    DSBN = False,  # Domain-spec batchnorm
    data_path=args.data_path,
    prompt_type=args.peft_type,
    num_tokens=64,
    n_layers_conf=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # token
    mlp_adapter_conf=[1, 1, 1, 1, 1, 1, 0,0,0,0,0,0],
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
    "finetune": "finetune",
    "HRA": "HRA",
    "OFT": "OFT",
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
mask_ratio = config.mask_ratio
mask_value = "auto"  # for masked values, now it should always be auto

include_zero_gene = config.include_zero_gene  # if True, include zero genes among hvgs in the training
max_seq_len = 2001
n_bins = config.n_bins

# input/output representation
input_style = "binned"  # "normed_raw", "log1p", or "binned"
output_style = "binned"  # "normed_raw", "log1p", or "binned"

# settings for training
MLM = False  # whether to use masked language modeling, currently it is always on.
CLS = True  # celltype classification objective
ADV = False  # Adversarial training for batch correction
CCE = False  # Contrastive cell embedding objective
MVC = config.MVC  # Masked value prediction for cell embedding
ECS = config.ecs_thres > 0  # Elastic cell similarity objective
DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
INPUT_BATCH_LABELS = False  # TODO: have these help MLM and MVC, while not to classifier
input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
adv_D_delay_epochs = 0
mvc_decoder_style = "inner product"
ecs_threshold = config.ecs_thres
dab_weight = config.dab_weight

explicit_zero_prob = MLM and include_zero_gene  # whether explicit bernoulli for zeros
do_sample_in_train = False and explicit_zero_prob  # sample the bernoulli in training

per_seq_batch_sample = False

# settings for optimizer
lr = config.lr  # TODO: test learning rate ratio between two tasks
lr_ADV = 1e-3  # learning rate for discriminator, used when ADV is True
batch_size = config.batch_size
eval_batch_size = config.batch_size
schedule_interval = 1

# settings for the model
fast_transformer = config.fast_transformer
fast_transformer_backend = "flash"  # "linear" or "flash"
embsize = config.layer_size  # embedding dimension
d_hid = config.layer_size  # dimension of the feedforward network in TransformerEncoder
nlayers = config.nlayers  # number of TransformerEncoderLayer in TransformerEncoder
nhead = config.nhead  # number of heads in nn.MultiheadAttention
dropout = config.dropout  # dropout probability

data_path = config.data_path
use_prompt = config.use_prompt
prompt_type = config.prompt_type
num_tokens = config.num_tokens
n_layers_conf = config.n_layers_conf
mlp_adapter_conf = config.mlp_adapter_conf
space_adapter_conf = config.space_adapter_conf

# logging
log_interval = 100  # iterations
save_eval_interval = config.save_eval_interval  # epochs
do_eval_scib_metrics = True
# %% validate settings
assert input_style in ["normed_raw", "log1p", "binned"]
assert output_style in ["normed_raw", "log1p", "binned"]
assert input_emb_style in ["category", "continuous", "scaling"]
if input_style == "binned":
    if input_emb_style == "scaling":
        raise ValueError("input_emb_style `scaling` is not supported for binned input.")
elif input_style == "log1p" or input_style == "normed_raw":
    if input_emb_style == "category":
        raise ValueError(
            "input_emb_style `category` is not supported for log1p or normed_raw input."
        )

if input_emb_style == "category":
    mask_value = n_bins + 1
    pad_value = n_bins  # for padding gene expr values
    n_input_bins = n_bins + 2
else:
    mask_value = -1
    pad_value = -2
    n_input_bins = n_bins

if ADV and DAB:
    raise ValueError("ADV and DAB cannot be both True.")
DAB_separate_optim = True if DAB > 1 else False
dataset_name = config.dataset_name
save_dir = Path(f"./save/{dataset_name}/{prompt_type}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")

data_dir = Path(data_path + dataset_name)
n_splits = 5
prompt_settings = {
    "use_prompt": config.use_prompt,
    "num_tokens": config.num_tokens,
    "prompt_type": config.prompt_type,
    "n_layers_conf": config.n_layers_conf,
    "mlp_adapter_conf": config.mlp_adapter_conf,
    "space_adapter_conf": config.space_adapter_conf
}
for i in range(n_splits):

    if config.load_model is not None:
        model_dir = Path(config.load_model)
        model_config_file = model_dir / "args.json"
        model_file = model_dir / f"model_fold{i}.pt"
        vocab_file = model_dir / "vocab.json"

        vocab = GeneVocab.from_file(vocab_file)
        # shutil.copy(vocab_file, save_dir / "vocab.json")
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)
        tokenized_data, data_global_describe = load_and_process_data(dataset_name, i, config, vocab, logger)
        # model
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        logger.info(
            f"Resume model from {model_file}, the model args will override the "
            f"config {model_config_file}."
        )
        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
        n_layers_cls = model_configs["n_layers_cls"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_cls = torch.load(model_file,map_location=device)['cls_decoder.out_layer.bias'].shape[0]
    ntokens = len(vocab)  # size of vocabulary
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
        do_mvc=MVC,
        do_dab=DAB,
        use_batch_labels=INPUT_BATCH_LABELS,
        domain_spec_batchnorm=config.DSBN,
        input_emb_style=input_emb_style,
        n_input_bins=n_input_bins,
        cell_emb_style=cell_emb_style,
        mvc_decoder_style=mvc_decoder_style,
        ecs_threshold=ecs_threshold,
        explicit_zero_prob=explicit_zero_prob,
        use_fast_transformer=fast_transformer,
        fast_transformer_backend=fast_transformer_backend,
        pre_norm=config.pre_norm,
        **prompt_settings
    )

    if config.prompt_type == "LoRA" or config.prompt_type == "HRA" or config.prompt_type == "OFT":
        model_weights = torch.load(model_file, map_location=device)
        if 'transformer_encoder.layers.0.self_attn.Wqkv.weight' in model_weights:
            for i in range(6):
                del model_weights[f'transformer_encoder.layers.{i}.self_attn.in_proj_weight']
                del model_weights[f'transformer_encoder.layers.{i}.self_attn.in_proj_bias']
            model.load_state_dict(model_weights, strict=True)
            logger.info("<All keys matched successfully>")
    else:
        model.load_state_dict(torch.load(model_file,map_location=device), strict=True)
        logger.info("<All keys matched successfully>")
    model.to(device)
    adata_test = data_global_describe["adata_test"]
    predictions, labels, results = test(
        model=model,
        adata=adata_test,
        gene_ids=data_global_describe["gene_ids"],
        vocab=vocab,
        config=config,
        device=device,
        logger=logger,
        data_global_describe=data_global_describe,
        dataset_name=dataset_name,
    )
    logger.info(f"fold{i}:"
        f"Accuracy: {results['test/accuracy']:.3f}, Precision: {results['test/precision']:.3f}, Recall: {results['test/recall']:.3f}, "
        f"Macro F1: {results['test/macro_f1']:.3f}"
    )


