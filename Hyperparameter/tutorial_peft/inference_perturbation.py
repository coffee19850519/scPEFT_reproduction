#!/usr/bin/env python3

import argparse
import json
import os
import pickle
import time
import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.loader import DataLoader
import re

# Import modules - align with training script
import sys
sys.path.insert(0, "../")

import scgpt as scg
from scgpt.model import TransformerGenerator
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, PeftConfig, freeze_parameters, DownstreamTasks, compute_perturbation_metrics

from gears import PertData, GEARS
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction

# Set up environment
warnings.filterwarnings('ignore')


def parse_arguments():
    """Parse command line arguments (concise)."""
    parser = argparse.ArgumentParser(description='Perturbation Prediction Inference for scgpt-based models')
    
    # Main arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Full path to model file (e.g., ./save/peft_perturb8_PREFIX_norman-Dec29-10-30/best_model.pt)')
    parser.add_argument('--data_name', type=str, default='norman',
                        help='Dataset name (norman, adamson)')
    parser.add_argument('--split', type=str, default='simulation',
                        help='Data split method')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing perturbation data')
    # Inference settings
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use Automatic Mixed Precision')
    
    # Data processing settings
    parser.add_argument('--include_zero_gene', type=str, default='all', choices=['all', 'batch-wise'],
                        help='How to handle zero genes')
    parser.add_argument('--max_seq_len', type=int, default=1536, help='Maximum sequence length')
    
    # Output settings
    parser.add_argument('--save_results', action='store_true', default=True,
                        help='Save inference results')
    parser.add_argument('--output_dir', type=str, default='./perturbation_inference_results',
                        help='Directory to save inference results')
    
    return parser.parse_args()


def auto_detect_peft_and_params(model_path):
    """
    Auto-detect PEFT strategy and parameter from model path.
    Supports save-dir and model_hidden formats.
    Returns: (peft_strategy, param_value, peft_config)
    """
    model_path = Path(model_path)
    
    # Extract PEFT strategy from directory name
    peft_strategy = None
    peft_strategies = ['ENCODER', 'TOKEN', 'PREFIX', 'LORA', 'HYBRID']
    
    # Look for PEFT strategy in the path components
    for part in model_path.parts:
        if part.upper() in peft_strategies:
            peft_strategy = part.upper()
            break
    
    if peft_strategy is None:
        # Check if save directory name contains strategy info
        save_dir_name = model_path.parent.name if model_path.parent else ""
        for strategy in peft_strategies:
            if strategy in save_dir_name.upper():
                peft_strategy = strategy
                break
    
    if peft_strategy is None:
        raise ValueError(f"Cannot parse PEFT strategy from path: {model_path}")
    
    # Extract parameter value - support two formats
    param_value = None
    
    # Format 1: peft_perturb{VALUE}_{STRATEGY} directory format
    save_dir_name = model_path.parent.name if model_path.parent else ""
    match = re.match(r'peft_perturb([^_]+)_([^_]+)_', save_dir_name)
    if match:
        param_str = match.group(1)
        param_value = float(param_str) if '.' in param_str else int(param_str)
        # silent
    else:
        # Try other perturb formats
        param_match = re.search(r'perturb([0-9.]+)', save_dir_name)
        if param_match:
            param_str = param_match.group(1)
            param_value = float(param_str) if '.' in param_str else int(param_str)
            # silent
    
    # Format 2: model_hidden{SIZE}.pt file format
    if param_value is None:
        hidden_match = re.search(r'model_hidden(\d+)\.pt$', model_path.name)
        if hidden_match:
            hidden_size = int(hidden_match.group(1))
            # Map hidden_size to corresponding parameter value
            param_value = map_hidden_size_to_param(peft_strategy, hidden_size)
            # silent
    
    if param_value is None:
        # Set default values based on PEFT strategy
        defaults = {
            'ENCODER': 0.015625,
            'TOKEN': 0.125,
            'PREFIX': 8,
            'LORA': 64,
            'HYBRID': 0.25
        }
        param_value = defaults.get(peft_strategy, 0.25)
        print(f"PEFT param not found; using default {param_value} for {peft_strategy}")
    
    print(f"PEFT detected: strategy={peft_strategy}, param={param_value}")
    
    # Create configuration based on strategy and parameter value
    peft_config = create_peft_config(peft_strategy, param_value)
    
    return peft_strategy, param_value, peft_config


def map_hidden_size_to_param(peft_strategy, hidden_size):
    """Map hidden size to the corresponding PEFT parameter value."""
    if peft_strategy in ['ENCODER', 'TOKEN', 'HYBRID']:
        # mlp_ratio = hidden_size / 512
        mlp_ratio = hidden_size / 512.0
        # silent
        return mlp_ratio
            
    elif peft_strategy == 'PREFIX':
        # token_nums = hidden_size (use directly)
        token_nums = hidden_size
        # silent
        return token_nums
        
    elif peft_strategy == 'LORA':
        # r = hidden_size (use directly)
        r = hidden_size
        # silent
        return r
        
    else:
        # Default: treat as mlp_ratio
        return hidden_size / 512.0


def create_peft_config(peft_strategy, param_value):
    """Create PEFT config from strategy and parameter value."""
    
    peft_params = {
        'peft_type': peft_strategy,
        'use_default_settings': False,
        'adapter_layer_conf': [True]*6 + [False]*6,
    }
    
    if peft_strategy == "HYBRID":
        peft_params['mlp_ratio'] = param_value if param_value is not None else 0.25
        peft_params['skip_connect'] = True
        
    elif peft_strategy == "ENCODER":
        peft_params['mlp_ratio'] = param_value if param_value is not None else 0.015625
        peft_params['skip_connect'] = True
        
    elif peft_strategy == "TOKEN":
        peft_params['mlp_ratio'] = param_value if param_value is not None else 0.125
        peft_params['skip_connect'] = True
        
    elif peft_strategy == "PREFIX":
        peft_params['token_nums'] = param_value if param_value is not None else 8
        
    elif peft_strategy == "LORA":
        peft_params['r'] = param_value if param_value is not None else 64
        peft_params['lora_alpha'] = 1
        peft_params['enable_lora'] = [True, False, True]
    
    peft_config = PeftConfig(**peft_params).to_dict()
    # silent
    
    return peft_config


def load_model_and_config(args, peft_strategy, peft_config):
    """Load model config and vocab from fine-tuned model directory."""
    print(f"Loading model from: {args.model_path}")
    
    model_path = Path(args.model_path)
    
    # Find PEFT strategy directory (same logic as replogle)
    peft_dir = None
    for part in model_path.parts:
        if part.upper() in ['ENCODER', 'TOKEN', 'PREFIX', 'LORA', 'HYBRID']:
            # Get the full path to this PEFT strategy directory
            peft_dir_index = model_path.parts.index(part)
            peft_dir = Path(*model_path.parts[:peft_dir_index+1])
            break
    
    if peft_dir is None:
        # If no PEFT directory found, try to look for config files in common locations
        print("Config directory not auto-detected; searching common locations...")
        
        # Try parent directory of model file
        potential_dirs = [
            model_path.parent,
            model_path.parent.parent,
        ]
        
        for potential_dir in potential_dirs:
            if potential_dir.exists() and (potential_dir / "vocab.json").exists():
                peft_dir = potential_dir
                # silent
                break
        
        if peft_dir is None:
            raise FileNotFoundError(f"Could not locate configuration directory for {model_path}")
    
    # silent
    
    # Load vocabulary - try multiple possible locations
    vocab_file = None
    possible_vocab_locations = [
        peft_dir / "vocab.json",
        model_path.parent / "vocab.json",
        model_path.parent.parent / "vocab.json",
    ]
    
    for possible_path in possible_vocab_locations:
        if possible_path.exists():
            vocab_file = possible_path
            # silent
            break
    
    if vocab_file is None:
        raise FileNotFoundError(f"Vocab file not found in any of these locations: {[str(p) for p in possible_vocab_locations]}")
    
    # Load vocabulary
    vocab = GeneVocab.from_file(vocab_file)
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    print(f"Vocabulary loaded (size={len(vocab)})")
    
    # Load model configuration - try multiple possible locations
    config_file = None
    possible_config_locations = [
        peft_dir / "args.json",
        model_path.parent / "args.json",
        model_path.parent.parent / "args.json",
    ]
    
    for possible_path in possible_config_locations:
        if possible_path.exists():
            config_file = possible_path
            # silent
            break
    
    if config_file is None:
        raise FileNotFoundError(f"Config file not found in any of these locations: {[str(p) for p in possible_config_locations]}")
    
    with open(config_file, "r") as f:
        model_configs = json.load(f)
    
    # silent
    
    return vocab, model_configs, peft_config


def create_model(vocab, model_configs, peft_config, device):
    """Create model (same config as training)."""
    
    # Extract architecture params
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]  
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]
    dropout = 0
    
    # Other fixed training params
    pad_token = "<pad>"
    pad_value = 0
    pert_pad_id = 0
    use_fast_transformer = False
    
    ntokens = len(vocab)
    
    print("Model architecture:")
    print(f"  d_model={embsize}, d_hid={d_hid}, nhead={nhead}, nlayers={nlayers}, n_layers_cls={n_layers_cls}, dropout={dropout}, vocab={ntokens}")
    
    # Create model with same params as perturbation.py
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
        use_fast_transformer=use_fast_transformer,
        peft_config=peft_config
    )
    
    model.to(device)
    return model


def load_and_preprocess_data(args, vocab):
    """Load and preprocess perturbation data using GEARS."""
    print(f"Dataset: {args.data_name}")
    
    # Load data via GEARS
    pert_data = PertData(args.data_dir)
    pert_data.load(data_name=args.data_name)
    pert_data.prepare_split(split=args.split, seed=1)
    pert_data.get_dataloader(batch_size=args.batch_size, test_batch_size=args.batch_size)
    
    # Filter genes in vocab
    genes = pert_data.adata.var["gene_name"].tolist()
    pert_data.adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in genes
    ]
    gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
    # omit gene-in-vocab count
    
    # Create gene ID mapping
    gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
    )
    
    n_genes = len(genes)
    
    # omit shapes and batch counts
    
    return {
        'pert_data': pert_data,
        'gene_ids': gene_ids,
        'n_genes': n_genes,
        'genes': genes
    }


def eval_perturb(loader: DataLoader, model: TransformerGenerator, gene_ids, include_zero_gene, device: torch.device) -> Dict:
    """Run perturbation inference."""
    print("Starting perturbation inference...")
    
    model.eval()
    model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}

    for itr, batch in enumerate(loader):
        # quiet batch logs
            
        batch.to(device)
        pert_cat.extend(batch.pert)

        with torch.no_grad():
            # Same call as training code
            p = model.pred_perturb(
                batch,
                include_zero_gene=include_zero_gene,
                gene_ids=gene_ids,
            )
            t = batch.y
            pred.extend(p.cpu())
            truth.extend(t.cpu())

            # Differentially expressed genes
            for itr_inner, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[itr_inner, de_idx])
                truth_de.append(t[itr_inner, de_idx])

    # All genes
    results["pert_cat"] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results["pred"] = pred.detach().cpu().numpy().astype(np.float64)
    results["truth"] = truth.detach().cpu().numpy().astype(np.float64)

    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results["pred_de"] = pred_de.detach().cpu().numpy().astype(np.float64)
    results["truth_de"] = truth_de.detach().cpu().numpy().astype(np.float64)

    # omit detailed result shapes

    return results


def calculate_perturbation_metrics(inference_results, pert_data):
    """Compute perturbation metrics."""
    print("Computing metrics...")
    
    # Try scGPT compute_perturbation_metrics first
    try:
        ctrl_adata = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
        metrics = compute_perturbation_metrics(inference_results, ctrl_adata)
        # silent
        
        # Also try GEARS compute_metrics
        try:
            _, metrics_pert = compute_metrics(inference_results)
            return metrics, metrics_pert
        except:
            return metrics, {}
            
    except Exception as e:
        print(f"scGPT metric computation failed; trying GEARS: {e}")
        
        # Fallback: GEARS compute_metrics
        try:
            metrics, metrics_pert = compute_metrics(inference_results)
            # silent
            return metrics, metrics_pert
            
        except Exception as e2:
            print(f"GEARS metric computation failed: {e2}")
            # Fallback basic metrics
            pred = inference_results['pred']
            truth = inference_results['truth']
            
            mse = np.mean((pred - truth) ** 2)
            mae = np.mean(np.abs(pred - truth))
            pearson_corr = np.corrcoef(pred.flatten(), truth.flatten())[0, 1]
            
            fallback_metrics = {
                'mse': mse,
                'mae': mae,
                'pearson': pearson_corr
            }
            
            return fallback_metrics, {}


def print_detailed_results(metrics):
    """Print detailed evaluation results"""
    print("\n=== Perturbation Results ===")

    def _fmt_num(val, default="N/A", fmt=".6f"):
        try:
            import numpy as _np
            if isinstance(val, (int, float, _np.floating)):
                return f"{float(val):{fmt}}"
        except Exception:
            pass
        return default

    print(f"MSE: {_fmt_num(metrics.get('mse'))}")
    print(f"MAE: {_fmt_num(metrics.get('mae'))}")
    print(f"Pearson: {_fmt_num(metrics.get('pearson'), fmt='.4f')}")
    
    # Print other available metrics
    for key, value in metrics.items():
        if key not in ['mse', 'mae', 'pearson']:
            try:
                import numpy as _np
                if isinstance(value, (int, float)) or isinstance(value, _np.floating):
                    print(f"{key}: {float(value):.4f}")
                else:
                    print(f"{key}: {value}")
            except Exception:
                print(f"{key}: {value}")
    
    print("=== End Results ===")


def save_comprehensive_results(inference_results, metrics, args, data_dict, peft_strategy=None, param_value=None):
    """Save all inference results (silent)."""
    
    # Create output directory
    if peft_strategy and param_value is not None:
        if isinstance(param_value, float):
            method_name = f"{peft_strategy}_perturb{param_value}"
        else:
            method_name = f"{peft_strategy}_perturb{param_value}"
    else:
        method_name = "perturbation_inference"
    
    output_dir = Path(args.output_dir) / f"{method_name}_{args.data_name}_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # silent
    
    # Prepare results dict
    results_dict = {
        'predictions': inference_results['pred'],
        'truth': inference_results['truth'],
        'pert_categories': inference_results['pert_cat'],
        'metrics': metrics,
        'config': {
            'model_path': str(args.model_path),
            'data_name': args.data_name,
            'split': args.split,
            'batch_size': args.batch_size,
            'device': str(args.device),
            'peft_strategy': peft_strategy,
            'param_value': param_value,
        }
    }
    
    if 'pred_de' in inference_results:
        results_dict['pred_de'] = inference_results['pred_de']
        results_dict['truth_de'] = inference_results['truth_de']
    
    # Save as pickle (silent)
    with open(output_dir / 'perturbation_inference_results.pkl', 'wb') as f:
        pickle.dump(results_dict, f)
    # silent
    
    # Save metrics as JSON (silent)
    metrics_json = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_json[key] = value.tolist()
        elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
            metrics_json[key] = float(value)
        else:
            metrics_json[key] = value
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)
    # silent
    
    # Create basic visualizations (silent)
    try:
        create_perturbation_plots(inference_results, output_dir)
    except Exception as e:
        print(f"Visualization creation failed: {e}")
    
    return output_dir


def create_perturbation_plots(inference_results, save_dir):
    """Create basic perturbation plots."""
    pred = inference_results['pred'].flatten()
    truth = inference_results['truth'].flatten()
    
    # Scatter: prediction vs truth
    plt.figure(figsize=(8, 8))
    plt.scatter(truth, pred, alpha=0.5, s=1)
    plt.plot([truth.min(), truth.max()], [truth.min(), truth.max()], 'r--', lw=2)
    plt.xlabel('Truth')
    plt.ylabel('Prediction')
    plt.title('Perturbation: Prediction vs Truth')
    plt.tight_layout()
    plt.savefig(save_dir / 'predictions_vs_truth.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Error distribution histogram
    errors = pred - truth
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=50, alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    plt.tight_layout()
    plt.savefig(save_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # silent


def main():
    """Main inference pipeline with auto-detection."""
    args = parse_arguments()
    
    set_seed(42)
    
    # Device config
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    
    # Auto-detect PEFT strategy and params
    print("Auto-detecting PEFT configuration...")
    try:
        peft_strategy, param_value, peft_config = auto_detect_peft_and_params(args.model_path)
    except ValueError as e:
        print(f"Warning: PEFT auto-detect failed: {e}; using defaults")
        peft_strategy = "PREFIX"
        param_value = None
        peft_config = {'peft_type': 'PREFIX', 'token_nums': 8, 'use_default_settings': False, 'adapter_layer_conf': [True]*6 + [False]*6}
    
    print("Starting perturbation inference...")
    print(f"  model={args.model_path}")
    print(f"  dataset={args.data_name}")
    print(f"  split={args.split}")
    print(f"  peft={peft_strategy}:{param_value}")
    print(f"  batch_size={args.batch_size}")
    
    # Load model config and vocab
    vocab, model_configs, peft_config = load_model_and_config(args, peft_strategy, peft_config)
    
    # Load and preprocess data
    data_dict = load_and_preprocess_data(args, vocab)
    
    # Create model
    model = create_model(vocab, model_configs, peft_config, device)
    
    # Load model weights
    print(f"Loading weights: {args.model_path}")
    if Path(args.model_path).exists():
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print("Weights loaded successfully")
        except Exception as e:
            print(f"Direct load failed: {e}; trying compatibility mode...")
            # Compatibility loading
            pretrained_dict = torch.load(args.model_path, map_location='cpu')
            model_dict = model.state_dict()

            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print("Weights loaded with compatibility mode")
    else:
        print(f"Warning: Model file not found: {args.model_path}")
    
    # Apply PEFT parameter freezing
    if peft_strategy:
        print(f"Freezing parameters for PEFT: {peft_strategy}")
        freeze_parameters(model, DownstreamTasks.Perturbation)
    
    # Parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: total={total_params/1e6:.2f}M, trainable={trainable_params/1e6:.2f}M ({trainable_params/total_params*100:.2f}%)")
    
    # Run inference
    print("Running inference...")
    start_time = time.time()
    
    test_loader = data_dict['pert_data'].dataloader['test_loader']
    test_res = eval_perturb(test_loader, model, data_dict['gene_ids'], args.include_zero_gene, device)
    
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.2f}s")
    
    # Compute metrics
    test_metrics, test_pert_res = calculate_perturbation_metrics(test_res, data_dict['pert_data'])
    # silent
    
    metrics = test_metrics
    metrics_pert = test_pert_res
    inference_results = test_res
    
    # Display results
    print_detailed_results(metrics)
    
    # Save results
    if args.save_results:
        output_dir = save_comprehensive_results(
            inference_results, 
            metrics, 
            args, 
            data_dict, 
            peft_strategy, 
            param_value
        )
    
    print("Perturbation inference completed.")
    
    # Safe formatting of metric values
    mse_val = metrics.get('mse', 'N/A')
    if isinstance(mse_val, (int, float)):
        print(f"MSE: {mse_val:.6f}")
    else:
        print(f"MSE: {mse_val}")
    
    pearson_val = metrics.get('pearson', 'N/A')
    if isinstance(pearson_val, (int, float)):
        print(f"Pearson: {pearson_val:.4f}")
    else:
        print(f"Pearson: {pearson_val}")
    
    return inference_results, metrics


if __name__ == "__main__":
    inference_results, metrics = main()
