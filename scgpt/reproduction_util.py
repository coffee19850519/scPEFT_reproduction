# %%
import copy
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings
import pandas as pd
# from . import asyn
import pickle
import torch
from anndata import AnnData
import scanpy as sc
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

import numpy as np
from scipy.sparse import issparse
from scgpt.model import TransformerModel, AdversarialDiscriminator
#from script.Tutorial_Integration import adata

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed
from scgpt.trainer import (
    test,
    train,
    evaluate,
    prepare_data,
    prepare_dataloader
)

def load_adata(data_dir, fold, dataset, celltype_key):
    adata = sc.read(data_dir / f"{fold}/{dataset}_train{fold}.h5ad")
    adata_val = sc.read(data_dir / f"{fold}/{dataset}_val{fold}.h5ad")
    adata_test = sc.read(data_dir / f"{fold}/{dataset}_test{fold}.h5ad")

    adata.obs["celltype"] = adata.obs[celltype_key].astype("category")
    adata_val.obs["celltype"] = adata_val.obs[celltype_key].astype("category")
    adata_test.obs["celltype"] = adata_test.obs[celltype_key].astype("category")

    adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
    adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"
    adata_val.obs["batch_id"] = adata_val.obs["str_batch"] = "2"

    if dataset=="elegans":
        adata = adata[~(adata.obs["celltype"]=="NA")]
        adata_val = adata_val[~(adata_val.obs["celltype"]=="NA")]
        adata_test = adata_test[~(adata_test.obs["celltype"]=="NA")]
        adata.var["gene_name"] =adata.var["orthogene"]
        adata_val.var["gene_name"] = adata_val.var["orthogene"]
        adata_test.var["gene_name"] =adata_test.var["orthogene"]
    if dataset!="elegans":
        adata.var.set_index(adata.var["gene_name"], inplace=True)
        adata_val.var.set_index(adata_val.var["gene_name"], inplace=True)
        adata_test.var.set_index(adata_test.var["gene_name"], inplace=True)

    adata = adata.concatenate((adata_test,adata_val), batch_key="str_batch")
    return adata


def load_and_process_data(
        dataset_name,
        fold_index,
        config,
        vocab,
        logger,
):
    if dataset_name == "ms":
        data_dir = Path(f"../data/celltype_identification/{dataset_name}")

        data_is_raw = False
        celltype_key = "celltype"
        adata = load_adata(data_dir, fold_index, dataset_name, celltype_key)
    elif dataset_name == "COVID":
        data_dir = Path(f"../data/celltype_identification/{dataset_name}")

        data_is_raw = True
        celltype_key = "cell_type"
        adata = load_adata(data_dir, fold_index, dataset_name, celltype_key)
    elif dataset_name == "NSCLC":
        data_dir = Path(f"../data/celltype_identification/{dataset_name}")

        data_is_raw = True
        celltype_key = "cell_type"
        adata = load_adata(data_dir, fold_index, dataset_name, celltype_key)
    elif dataset_name == "MergedMonkey":
        data_dir = Path(f"../data/cross_species/{dataset_name}")

        data_is_raw = True
        celltype_key = "CellType"
        adata = load_adata(data_dir, fold_index, dataset_name, celltype_key)
    elif dataset_name == "elegans":
        data_dir = Path(f"../data/cross_species/{dataset_name}")

        data_is_raw = True
        celltype_key = "tissue_name"
        adata = load_adata(data_dir, fold_index, dataset_name, celltype_key)
    elif dataset_name == "mouse_115746":
        data_dir = Path(f"../data/cross_species/{dataset_name}")

        data_is_raw = True
        celltype_key = "celltype"
        adata = load_adata(data_dir, fold_index, dataset_name, celltype_key)
    elif dataset_name == "mouse_10x":
        data_dir = Path(f"../data/cross_species/{dataset_name}")

        data_is_raw = True
        celltype_key = "celltype"
        adata = load_adata(data_dir, fold_index, dataset_name, celltype_key)
    elif dataset_name == "mouse_smart":
        data_dir = Path(f"../data/cross_species/{dataset_name}")

        data_is_raw = True
        celltype_key = "celltype"
        adata = load_adata(data_dir, fold_index, dataset_name, celltype_key)

    # make the batch category column
    batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
    adata.obs["batch_id"] = batch_id_labels
    celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
    celltypes = adata.obs["celltype"].unique()
    num_types = len(np.unique(celltype_id_labels))
    id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
    adata.obs["celltype_id"] = celltype_id_labels
    if dataset_name=="elegans":
        pass
    else:
        adata.var["gene_name"] = adata.var.index.tolist()

    #
    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    # set up the preprocessor, use the args to config the workflow
    preprocessor = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=False,  # step 1
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
    adata_test = adata[adata.obs["str_batch"] == "1"]
    adata = adata[adata.obs["str_batch"] != "1"]

    # preprocessor(adata, batch_key=None)
    # preprocessor(adata_test, batch_key=None)

    input_layer_key = {  # the values of this map coorespond to the keys in preprocessing
        "normed_raw": "X_normed",
        "log1p": "X_normed",
        "binned": "X_binned",
    }[config.input_style]

    genes = adata.var["gene_name"].tolist()
    train_celltype_labels = adata[adata.obs["str_batch"] == "0"].obs["celltype_id"].values  # make sure count from 0
    valid_celltype_labels = adata[adata.obs["str_batch"] == "1"].obs["celltype_id"].values  # make sure count from 0

    batch_ids = adata.obs["batch_id"].tolist()
    num_batch_types = len(set(batch_ids))

    train_batch_labels = adata[adata.obs["str_batch"] == "0"].obs["batch_id"].values
    valid_batch_labels = adata[adata.obs["str_batch"] == "1"].obs["batch_id"].values

    adata_val = adata[adata.obs["str_batch"] == "2"]
    adata = adata[adata.obs["str_batch"] == "0"]

    train_data = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )

    valid_data = (
        adata_val.layers[input_layer_key].A
        if issparse(adata_val.layers[input_layer_key])
        else adata_val.layers[input_layer_key]
    )

    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)

    tokenized_train = tokenize_and_pad_batch(
        train_data,
        gene_ids,
        max_len=config.max_seq_len,
        vocab=vocab,
        pad_token=config.pad_token,
        pad_value=config.pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=config.include_zero_gene,
    )
    tokenized_valid = tokenize_and_pad_batch(
        valid_data,
        gene_ids,
        max_len=config.max_seq_len,
        vocab=vocab,
        pad_token=config.pad_token,
        pad_value=config.pad_value,
        append_cls=True,
        include_zero_gene=config.include_zero_gene,
    )
    logger.info(
        f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
    )
    logger.info(
        f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
    )

    return [
        {
            "tokenized_train": tokenized_train,
            "tokenized_valid": tokenized_valid,
            "train_celltype_labels": train_celltype_labels,
            "valid_celltype_labels": valid_celltype_labels,
            "train_batch_labels": train_batch_labels,
            "valid_batch_labels": valid_batch_labels,
        },
        {
            "id2type": id2type,
            "gene_ids": gene_ids,
            "num_types": num_types,
            "celltypes": celltypes,
            "adata_test": adata_test,
            "num_batch_types": num_batch_types,
            "num_training_cells": len(train_data),
            "celltype_id_labels": celltype_id_labels,

        }
    ]


def get_weighted_sampler(train_data_pt, data_global_describe):
    class_counts = np.unique(train_data_pt['celltype_labels'], return_counts=True)[1]
    class_weights = 1.0 / class_counts[train_data_pt['celltype_labels']]
    sample_weights = class_weights / np.sum(class_weights)

    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
        sample_weights,
        data_global_describe["num_training_cells"],
        replacement=True
    )
    return train_sampler


def initialize_model(
        config,
        vocab,
        device,
        logger,
        data_global_describe
):
    num_types = data_global_describe["num_types"]
    num_batch_types = data_global_describe["num_batch_types"]
    ntokens = len(vocab)  # size of vocabulary

    model = TransformerModel(
        ntokens,
        config.embsize,
        config.nhead,
        config.d_hid,
        config.nlayers,
        nlayers_cls=3,
        n_cls=num_types if config.CLS else 1,
        vocab=vocab,
        dropout=config.dropout,
        pad_token=config.pad_token,
        pad_value=config.pad_value,
        do_mvc=config.MVC,
        do_dab=config.DAB,
        use_batch_labels=config.INPUT_BATCH_LABELS,
        num_batch_labels=num_batch_types,
        domain_spec_batchnorm=config.DSBN,
        input_emb_style=config.input_emb_style,
        n_input_bins=config.n_input_bins,
        cell_emb_style=config.cell_emb_style,
        mvc_decoder_style=config.mvc_decoder_style,
        ecs_threshold=config.ecs_threshold,
        explicit_zero_prob=config.explicit_zero_prob,
        use_fast_transformer=config.fast_transformer,
        fast_transformer_backend=config.fast_transformer_backend,
        pre_norm=config.pre_norm,
        peft_config=config.peft_config
    )
    if config.load_model is not None:
        model_file = os.path.join(config.load_model, "best_model.pt")
        load_pretrained(model, torch.load(model_file), verbose=False)

    pre_freeze_param_count = sum(
        dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

    # Freeze params
    if config.peft:
        freeze_parameters(model, DownstreamTasks.Identification)

    post_freeze_param_count = sum(
        dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

    logger.info("-" * 89)
    learnable_params = {k: v for k, v in model.named_parameters() if v.requires_grad}
    for k, v in learnable_params.items():
        logger.info(f"Learnable params {k} with shape {v.shape}")

    logger.info("Total Pre freeze Params: %.2fM" % (pre_freeze_param_count / 1e6,))
    logger.info("Total Post freeze Params: %.2fM" % (post_freeze_param_count / 1e6,))

    model.to(device)
    logger.info(model)

    return model
