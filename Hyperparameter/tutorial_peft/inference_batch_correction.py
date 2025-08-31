#!/usr/bin/env python3
import argparse
import copy
import gc
import json
import os
import pickle
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
import torch.nn.functional as F
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind
import re

# Import scgpt modules - use local modified scgpt
import sys
sys.path.insert(0, "../")

import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import (
    set_seed, eval_scib_metrics, load_pretrained, 
    PeftConfig, freeze_parameters, DownstreamTasks
)

# Set up environment
sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings("ignore", category=DeprecationWarning)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Batch Correction Inference for scgpt-based models'
    )
    
    # Main arguments
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to trained model file (e.g., ./savebatch/64_LORA_PBMC_10K-Dec30-15-30/best_model.pt)'
    )
    parser.add_argument(
        '--dataset_name', type=str, default='PBMC_10K',
        choices=['PBMC_10K', 'covid_subsampled', 'Perirhinal_Cortex'],
        help='Dataset name'
    )
    parser.add_argument(
        '--data_dir', type=str,
        default='./data/batch_correction',
        help='Directory containing batch correction data'
    )
    
    # Inference settings
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        help='Device to use (auto/cpu/cuda)'
    )
    parser.add_argument(
        '--amp', action='store_true', default=True,
        help='Use Automatic Mixed Precision'
    )
    
    # Data processing settings
    parser.add_argument(
        '--n_hvg', type=int, default=1200,
        help='Number of highly variable genes'
    )
    parser.add_argument(
        '--n_bins', type=int, default=51,
        help='Number of bins for value binning'
    )
    parser.add_argument(
        '--mask_ratio', type=float, default=0.0,
        help='Mask ratio for inference (usually 0.0 for inference)'
    )
    
    # Output settings
    parser.add_argument(
        '--save_results', action='store_true', default=True,
        help='Save inference results'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./batch_correction_inference_results',
        help='Directory to save inference results'
    )
    
    return parser.parse_args()


def auto_detect_peft_and_params(model_path):
    """
    Auto-detect PEFT strategy and parameters from model path
    Supports multiple formats:
    1. Training format: ./savebatch/{param}_{STRATEGY}_{DATASET}-{TIME}/best_model.pt
    2. New format: /{DATASET}/{STRATEGY}/{PARAM}/best_model.pt
    """
    model_path = Path(model_path)
    
    # Extract PEFT strategy from path
    peft_strategy = None
    peft_strategies = ['ENCODER', 'TOKEN', 'PREFIX', 'LORA', 'HYBRID']
    
    # Look for PEFT strategy in path components
    for part in model_path.parts:
        part_upper = part.upper()
        if part_upper in peft_strategies:
            peft_strategy = part_upper
            break
    
    if peft_strategy is None:
        raise ValueError(f"Cannot parse PEFT strategy from path: {model_path}")
    
    # Extract parameter value (hidden dimension)
    hidden_dim = None
    
    # Method 1: New format /{DATASET}/{STRATEGY}/{PARAM}/best_model.pt
    strategy_dir_found = False
    for i, part in enumerate(model_path.parts):
        if part.upper() == peft_strategy:
            strategy_dir_found = True
            # Parameter value should be the next directory
            if i + 1 < len(model_path.parts):
                param_str = model_path.parts[i + 1]
                try:
                    hidden_dim = float(param_str) if '.' in param_str else int(param_str)
                except ValueError:
                    pass
            break
    
    # Method 2: Original training format {param}_{STRATEGY}_{DATASET}-{TIME}
    if hidden_dim is None and not strategy_dir_found:
        save_dir_name = model_path.parent.name
        parts = save_dir_name.split('_')
        if len(parts) >= 3:
            try:
                param_str = parts[0]  # First part should be the parameter
                hidden_dim = float(param_str) if '.' in param_str else int(param_str)
            except ValueError:
                pass
    
    if hidden_dim is None:
        # Set default values based on PEFT strategy
        defaults = {
            'ENCODER': 0.5,
            'TOKEN': 0.015625,
            'PREFIX': 512,
            'LORA': 64,
            'HYBRID': 0.25
        }
        hidden_dim = defaults.get(peft_strategy, 0.25)
        print(f"PEFT param not found; using default {hidden_dim} for {peft_strategy}")

    print(f"PEFT detected: strategy={peft_strategy}, param={hidden_dim}")
    
    # Convert to PEFT configuration using the same logic as inference_cell_type.py
    peft_config = convert_hidden_to_peft_config(peft_strategy, hidden_dim)
    
    return peft_strategy, hidden_dim, peft_config


def convert_hidden_to_peft_config(peft_strategy, hidden_dim):
    """
    Convert hidden dimension to appropriate PEFT configuration parameters
    """
    
    if peft_strategy in ['ENCODER', 'TOKEN']:
        # For ENCODER/TOKEN, hidden_dim represents mlp_ratio or similar
        if peft_strategy == 'ENCODER':
            # ENCODER uses mlp_ratio
            peft_config = {
                'peft_type': 'ENCODER',
                'mlp_ratio': float(hidden_dim),
                'skip_connect': True,
                'adapter_layer_conf': [True]*6 + [False]*6,
                'ENCODER_FLAG': True
            }
        elif peft_strategy == 'TOKEN':
            # TOKEN uses mlp_ratio
            peft_config = {
                'peft_type': 'TOKEN', 
                'mlp_ratio': float(hidden_dim),
                'skip_connect': True,
                'adapter_layer_conf': [True]*6 + [False]*6,
                'TOKEN_FLAG': True
            }
            
    elif peft_strategy == 'PREFIX':
        # For PREFIX, hidden_dim represents token_nums
        peft_config = {
            'peft_type': 'PREFIX',
            'token_nums': int(hidden_dim),
            'adapter_layer_conf': [True]*6 + [False]*6,
            'PREFIX_FLAG': True
        }
        
    elif peft_strategy == 'LORA':
        # For LORA, hidden_dim represents r (rank)
        peft_config = {
            'peft_type': 'LORA',
            'r': int(hidden_dim),
            'lora_alpha': 1,
            'enable_lora': [True, False, True],
            'adapter_layer_conf': [True]*6 + [False]*6,
            'LoRA_FLAG': True
        }
        
    elif peft_strategy == 'HYBRID':
        # For HYBRID, hidden_dim represents mlp_ratio
        peft_config = {
            'peft_type': 'HYBRID',
            'mlp_ratio': float(hidden_dim),
            'skip_connect': True,
            'adapter_layer_conf': [True]*6 + [False]*6,
            'HYBRID_FLAG': True
        }
    else:
        raise ValueError(f"Unknown PEFT strategy: {peft_strategy}")
    
    return peft_config




def load_model_and_config(args, peft_strategy, peft_config):
    """Load model configuration and vocabulary"""
    print(f"Loading model from: {args.model_path}")
    
    model_path = Path(args.model_path)
    model_dir = model_path.parent
    
    # Try to find vocab and config files in various locations
    vocab_file = None
    config_file = None
    
    # Priority order for searching files
    # For structure: /{DATASET}/{STRATEGY}/{PARAM}/best_model.pt
    # Prefer config files in the same directory as the model
    possible_locations = [
        model_dir,  # Same directory as model: /{DATASET}/{STRATEGY}/{PARAM}/ (highest priority)
        model_dir.parent,  # STRATEGY directory: /{DATASET}/{STRATEGY}/
        model_dir.parent.parent,  # DATASET directory: /{DATASET}/
        model_dir.parent.parent.parent,  # One more level up
    ]
    
    # Look for vocab file (prioritize same directory as model)
    for i, location in enumerate(possible_locations, 1):
        potential_vocab = location / "vocab.json"
        if potential_vocab.exists():
            vocab_file = potential_vocab
            break
    
    # Look for config file (try multiple names, prioritize same directory as model)
    config_names = ["model_config.json", "args.json"]
    for i, location in enumerate(possible_locations, 1):
        for config_name in config_names:
            potential_config = location / config_name
            if potential_config.exists():
                config_file = potential_config
                break
        if config_file:
            break
    
    # Load vocabulary
    vocab = GeneVocab.from_file(vocab_file)
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    print(f"Vocabulary loaded (size={len(vocab)})")
    
    # Load model configuration
    with open(config_file, "r") as f:
        model_configs = json.load(f)
    
    # Check if config file contains PEFT configuration
    if "peft_config" in model_configs:
        peft_config = model_configs["peft_config"]
    
    return vocab, model_configs, peft_config


def create_model(vocab, model_configs, peft_config, device, dataset_name, args):
    """Create batch correction model using TransformerModel"""
    
    # Extract model architecture parameters
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs.get("n_layers_cls", 3)
    
    # Fixed parameters from training code
    pad_token = "<pad>"
    pad_value = -2
    mask_value = -1
    dropout = 0.2  # From training config
    DSBN = True  # Domain-spec batchnorm
    explicit_zero_prob = True
    use_fast_transformer = False
    pre_norm = False
    
    # Get number of batches - this will be set properly when data is loaded
    num_batch_types = 2  # Default, will be updated
    
    ntokens = len(vocab)
    
    print("Model architecture:")
    print(f"  d_model={embsize}, d_hid={d_hid}, nhead={nhead}, nlayers={nlayers}, vocab={ntokens}")
    
    # Create TransformerModel for batch correction (matching training code exactly)
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        vocab=vocab,
        dropout=dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=True,  # GEPC enabled
        do_dab=True,  # Domain adversarial training for batch correction
        use_batch_labels=True,
        num_batch_labels=num_batch_types,
        domain_spec_batchnorm=DSBN,
        n_input_bins=args.n_bins,
        ecs_threshold=0.8,  # From training config
        explicit_zero_prob=explicit_zero_prob,
        use_fast_transformer=use_fast_transformer,
        pre_norm=pre_norm,
        peft_config=peft_config
    )
    
    model.to(device)
    return model


def load_and_preprocess_data(args, vocab):
    """Load and preprocess batch correction data"""
    # Minimal dataset log
    print(f"Dataset: {args.dataset_name}")
    
    data_dir = Path(args.data_dir)
    dataset_name = args.dataset_name
    
    # Load dataset based on name (matching training code)
    if dataset_name == "PBMC_10K":
        adata = sc.read(data_dir / f"{dataset_name}.h5ad")
        ori_batch_col = "batch"
        adata.obs["celltype"] = adata.obs["str_labels"].astype("category")
        adata.var = adata.var.set_index("gene_symbols")
        data_is_raw = True
        n_hvg = args.n_hvg
        
    elif dataset_name == "covid_subsampled":
        adata = sc.read(data_dir / f"{dataset_name}.h5ad")
        ori_batch_col = "str_batch"
        data_is_raw = True
        n_hvg = None
        
    elif dataset_name == "Perirhinal_Cortex":
        adata = sc.read(data_dir / f"{dataset_name}.h5ad")
        ori_batch_col = "sample_id"
        adata.obs["celltype"] = adata.obs["cell_type"].astype("category")
        adata.var = adata.var.set_index("Gene")
        data_is_raw = True
        n_hvg = args.n_hvg
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # omit verbose data shape/debug details
    
    # Create batch labels (matching training code)
    adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
    batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
    adata.obs["batch_id"] = batch_id_labels
    adata.var["gene_name"] = adata.var.index.tolist()
    
    # omit verbose counts
    
    # Filter genes in vocabulary (matching training code)
    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    # omit gene-in-vocab count
    
    # Filter to genes in vocab
    adata = adata[:, adata.var["id_in_vocab"] >= 0]
    # omit filtered data shape
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=3,
        filter_cell_by_counts=False,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=data_is_raw,
        result_log1p_key="X_log1p",
        subset_hvg=n_hvg,
        hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        binning=args.n_bins,
        result_binned_key="X_binned",
    )
    
    preprocessor(adata, batch_key="str_batch" if dataset_name != "heart_cell" else None)
    
    return {
        'adata': adata,
        'batch_id_labels': batch_id_labels,
        'ori_batch_col': ori_batch_col,
        'dataset_name': dataset_name
    }


# Dataset class (matching training code)
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def prepare_inference_data(adata, vocab, args):
    """Prepare inference data"""
    
    input_layer_key = "X_binned"
    pad_token = "<pad>"
    pad_value = -2
    mask_value = -1
    
    # Get data
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    genes = adata.var["gene_name"].tolist()
    
    # Get labels
    celltypes_labels = adata.obs["celltype"].tolist()
    batch_ids = adata.obs["batch_id"].tolist()
    
    # omit verbose preparation logs
    
    # Create gene IDs
    gene_ids = np.array(vocab(genes), dtype=int)
    
    # Tokenize data
    max_seq_len = len(genes) + 1
    tokenized_data = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=True,
    )
    
    # Apply masking if needed (usually 0 for inference)
    if args.mask_ratio > 0:
        masked_values = random_mask_value(
            tokenized_data["values"],
            mask_ratio=args.mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )
    else:
        masked_values = tokenized_data["values"]
    
    # Prepare data tensors
    data_pt = {
        "gene_ids": tokenized_data["genes"],
        "values": masked_values,
        "target_values": tokenized_data["values"],
        "batch_labels": torch.from_numpy(np.array(batch_ids)).long(),
    }
    
    return data_pt, np.array(celltypes_labels), np.array(batch_ids)


def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """Prepare data loader"""
    dataset = SeqDataset(data_pt)
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader


def eval_batch_correction(model, data_loader, vocab, device, args):
    """Batch correction inference, generate cell embeddings"""
    print("Starting batch correction inference...")
    
    model.eval()
    cell_embeddings = []
    
    pad_token = "<pad>"
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            
            with torch.cuda.amp.autocast(enabled=args.amp):
                # Get cell embeddings using encode_batch method
                batch_embeddings = model.encode_batch(
                    input_gene_ids,
                    input_values.float(),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_size=input_gene_ids.size(0),
                    batch_labels=batch_labels,
                    time_step=0,
                    return_np=False,
                )
                
            cell_embeddings.append(batch_embeddings.cpu())
    
    # Concatenate all embeddings
    cell_embeddings = torch.cat(cell_embeddings, dim=0).numpy()
    
    # Normalize embeddings
    cell_embeddings = cell_embeddings / np.linalg.norm(
        cell_embeddings, axis=1, keepdims=True
    )
    
    print(f"Generated cell embeddings shape: {cell_embeddings.shape}")
    
    return cell_embeddings


def calculate_batch_correction_metrics(adata, cell_embeddings):
    """Calculate batch correction metrics"""
    print("Calculating batch correction metrics...")
    
    # Add embeddings to adata
    adata_copy = adata.copy()
    adata_copy.obsm["X_scGPT"] = cell_embeddings
    
    # Calculate scIB metrics
    results = {}
    try:
        results = eval_scib_metrics(adata_copy)
        print("scIB metrics calculated successfully")
    except Exception as e:
        print(f"Error calculating scIB metrics: {e}")
        
        # Fallback metrics
        try:
            # Basic metrics using scanpy
            sc.pp.neighbors(adata_copy, use_rep="X_scGPT")
            sc.tl.umap(adata_copy, min_dist=0.3)
            
            results = {
                'embedding_dim': cell_embeddings.shape[1],
                'n_cells': cell_embeddings.shape[0],
                'message': 'Basic preprocessing completed'
            }
        except Exception as e2:
            results = {
                'embedding_dim': cell_embeddings.shape[1],
                'n_cells': cell_embeddings.shape[0],
                'error': str(e2)
            }
    
    return results, adata_copy


def create_batch_correction_plots(adata_processed, results, save_dir):
    """Create batch correction visualization plots"""
    
    try:
        # Ensure neighbors and UMAP are computed
        if "X_umap" not in adata_processed.obsm:
            sc.pp.neighbors(adata_processed, use_rep="X_scGPT")
            sc.tl.umap(adata_processed, min_dist=0.3)
        
        # Batch UMAP
        fig = sc.pl.umap(
            adata_processed,
            color=["str_batch"],
            title=[f"Batch correction, avg_bio = {results.get('avg_bio', 0.0):.4f}"],
            frameon=False,
            return_fig=True,
            show=False,
        )
        fig.savefig(save_dir / "batch_umap.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        # Cell type UMAP
        fig = sc.pl.umap(
            adata_processed,
            color=["celltype"],
            title=[f"Cell types, avg_bio = {results.get('avg_bio', 0.0):.4f}"],
            frameon=False,
            return_fig=True,
            show=False,
        )
        fig.savefig(save_dir / "celltype_umap.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        
    except Exception as e:
        print(f"Visualization creation failed: {e}")


def print_detailed_results(metrics):
    """Print detailed evaluation results"""
    print("\n=== Batch Correction Results ===")

    def _fmt_num(val, default="N/A", fmt=".4f"):
        try:
            if isinstance(val, (int, float, np.floating)):
                return f"{float(val):{fmt}}"
        except Exception:
            pass
        return default

    # Print main metrics
    if 'avg_bio' in metrics:
        print(f"Average Bio Score: {_fmt_num(metrics.get('avg_bio'))}")
    if 'avg_batch' in metrics:
        print(f"Average Batch Score: {_fmt_num(metrics.get('avg_batch'))}")
    if 'overall_score' in metrics:
        print(f"Overall Score: {_fmt_num(metrics.get('overall_score'))}")
    
    # Print other available metrics
    for key, value in metrics.items():
        if key not in ['avg_bio', 'avg_batch', 'overall_score']:
            try:
                if isinstance(value, (int, float, np.floating)):
                    print(f"{key}: {float(value):.4f}")
                else:
                    print(f"{key}: {value}")
            except Exception:
                print(f"{key}: {value}")
    
    print("=== End Results ===")


def save_comprehensive_results(
    cell_embeddings, adata, metrics, results_adata, args, 
    peft_strategy=None, hidden_dim=None
):
    """Save all inference results"""
    
    # Create output directory
    if peft_strategy and hidden_dim is not None:
        method_name = f"{peft_strategy}_{hidden_dim}"
    else:
        method_name = "batch_correction"
    
    output_dir = Path(args.output_dir) / f"{method_name}_{args.dataset_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    # Prepare results dictionary
    results_dict = {
        'cell_embeddings': cell_embeddings,
        'metrics': metrics,
        'config': {
            'model_path': str(args.model_path),
            'dataset_name': args.dataset_name,
            'batch_size': args.batch_size,
            'device': str(args.device),
            'peft_strategy': peft_strategy,
            'hidden_dim': hidden_dim,
            'n_hvg': args.n_hvg,
            'n_bins': args.n_bins,
        }
    }
    
    # Save results as pickle
    with open(output_dir / 'batch_correction_results.pkl', 'wb') as f:
        pickle.dump(results_dict, f)
    
    # Save metrics as JSON
    metrics_json = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_json[key] = value.tolist()
        elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
            metrics_json[key] = float(value)
        else:
            try:
                metrics_json[key] = value
            except:
                metrics_json[key] = str(value)
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    # Save processed AnnData
    results_adata.write_h5ad(output_dir / 'processed_data.h5ad')
    
    # Create visualizations
    create_batch_correction_plots(results_adata, metrics, output_dir)
    return output_dir


def auto_detect_dataset_name(model_path):
    """Auto-detect dataset name from model path"""
    model_path = Path(model_path)
    dataset_names = ['PBMC_10K', 'covid_subsampled', 'Perirhinal_Cortex']
    
    for part in model_path.parts:
        if part in dataset_names:
            return part
    
    # Check if any part contains dataset name
    for part in model_path.parts:
        for dataset in dataset_names:
            if dataset in part:
                return dataset
    
    return None


def auto_detect_model_file(model_path_input):
    """Auto-detect model file path, add best_model.pt if input is directory"""
    model_path = Path(model_path_input)
    
    if model_path.is_dir():
        potential_model = model_path / "best_model.pt"
        if potential_model.exists():
            return str(potential_model)
        else:
            raise FileNotFoundError(f"best_model.pt not found in directory {model_path}")
    
    if model_path.is_file():
        return str(model_path)
    
    raise FileNotFoundError(f"Model file does not exist: {model_path}")


def main():
    """Main inference pipeline"""
    args = parse_arguments()
    
    set_seed(42)  # Match training seed
    
    # Auto-detect model file if directory is provided
    try:
        args.model_path = auto_detect_model_file(args.model_path)
    except FileNotFoundError as e:
    print(f"Model path error: {e}")
        return None, None
    
    # Device configuration
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    
    # Auto-detect dataset name if not explicitly provided in model path
    detected_dataset = auto_detect_dataset_name(args.model_path)
    if detected_dataset and detected_dataset != args.dataset_name:
        if args.dataset_name == "PBMC_10K":  # Only override if using default
            args.dataset_name = detected_dataset
    
    # Auto-detect PEFT strategy and parameters
    try:
        peft_strategy, hidden_dim, peft_config = auto_detect_peft_and_params(args.model_path)
    except ValueError as e:
        print(f"Warning: Cannot auto-detect PEFT config: {e}")
        peft_strategy = "LORA"
        hidden_dim = 64
        peft_config = convert_hidden_to_peft_config("LORA", 64)
    
    print("Starting batch correction inference...")
    print(f"  model={args.model_path}")
    print(f"  dataset={args.dataset_name}")
    print(f"  peft={peft_strategy}:{hidden_dim}")
    print(f"  batch_size={args.batch_size}")
    
    # Load model configuration and vocabulary
    vocab, model_configs, peft_config = load_model_and_config(args, peft_strategy, peft_config)
    
    # Load and preprocess data
    data_dict = load_and_preprocess_data(args, vocab)
    adata = data_dict['adata']
    
    # Update model with correct number of batches
    num_batch_types = len(set(data_dict['batch_id_labels']))
    # omit verbose batch count log
    
    # Create model
    model = create_model(vocab, model_configs, peft_config, device, args.dataset_name, args)
    
    # Update model's num_batch_labels if needed
    if hasattr(model, 'sim') and hasattr(model.sim, 'dab_head'):
        old_weight = model.sim.dab_head.out_proj.weight
        old_bias = model.sim.dab_head.out_proj.bias
        if old_weight.size(0) != num_batch_types:
            model.sim.dab_head.out_proj = nn.Linear(old_weight.size(1), num_batch_types).to(device)
    
    # Load model weights
    print(f"Loading weights: {args.model_path}")
    if Path(args.model_path).exists():
        try:
            checkpoint = torch.load(args.model_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            print("Weights loaded successfully")
        except Exception as e:
            print(f"Weight load failed: {e}")
            try:
                checkpoint = torch.load(args.model_path, map_location='cpu')
                model_dict = model.state_dict()
                
                # Filter compatible parameters
                compatible_dict = {
                    k: v for k, v in checkpoint.items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }
                
                model_dict.update(compatible_dict)
                model.load_state_dict(model_dict)
                print("Weights loaded with compatibility mode")
            except Exception as e2:
                print(f"Compatibility loading failed: {e2}")
    else:
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    # Apply PEFT parameter freezing
    if peft_strategy:
        freeze_parameters(model, DownstreamTasks.BatchCorrection)
    
    # Record parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Params: total={total_params/1e6:.2f}M, trainable={trainable_params/1e6:.2f}M ({trainable_params/total_params*100:.2f}%)")
    
    # Prepare data for inference
    data_pt, celltype_labels, batch_labels = prepare_inference_data(adata, vocab, args)
    data_loader = prepare_dataloader(data_pt, args.batch_size)
    
    # Run batch correction inference
    start_time = time.time()
    
    cell_embeddings = eval_batch_correction(model, data_loader, vocab, device, args)
    
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.2f}s")
    
    # Calculate metrics
    metrics, results_adata = calculate_batch_correction_metrics(adata, cell_embeddings)
    
    # Display results
    print_detailed_results(metrics)
    
    # Save results
    if args.save_results:
        output_dir = save_comprehensive_results(
            cell_embeddings, adata, metrics, results_adata, args,
            peft_strategy, hidden_dim
        )
    
    print("Batch correction inference completed.")
    
    # Display key metrics
    avg_bio = metrics.get('avg_bio', 'N/A')
    avg_batch = metrics.get('avg_batch', 'N/A')
    overall_score = metrics.get('overall_score', 'N/A')
    
    if isinstance(avg_bio, (int, float)):
        print(f"Average Bio Score: {avg_bio:.4f}")
    else:
        print(f"Average Bio Score: {avg_bio}")
        
    if isinstance(avg_batch, (int, float)):
        print(f"Average Batch Score: {avg_batch:.4f}")
    else:
        print(f"Average Batch Score: {avg_batch}")
        
    if isinstance(overall_score, (int, float)):
        print(f"Overall Score: {overall_score:.4f}")
    else:
        print(f"Overall Score: {overall_score}")
    
    return cell_embeddings, metrics


if __name__ == "__main__":
    cell_embeddings, metrics = main()
