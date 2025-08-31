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

# Import modules - align with project structure
import sys
sys.path.insert(0, "../")

# Use the same imports as Reproduction_Perturbation_Replogle.py
from gears_replogle import PertData
from gears_replogle.inference import compute_metrics, deeper_analysis, non_dropout_analysis
from gears_replogle.utils import create_cell_graph_dataset_for_prediction

# Import from backbone_perturb (matches reproduction code)
sys.path.insert(0, "../../")
from backbone_perturb.model import TransformerGenerator
from backbone_perturb.tokenizer.gene_tokenizer import GeneVocab
from backbone_perturb.utils import set_seed, PeftConfig, freeze_parameters, DownstreamTasks, load_pretrained

# Set up environment
warnings.filterwarnings('ignore')


def parse_arguments():
    """Parse command line arguments (concise)."""
    parser = argparse.ArgumentParser(description='Perturbation Prediction Inference with Auto-detection')
    
    # Main arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Full path to model file (e.g., ../replogle_k562/ENCODER/perturb0.25/best_model.pt)')
    parser.add_argument('--data_name', type=str, default='replogle_k562_essential',
                        help='Dataset name (replogle_k562_essential, replogle_rpe1_essential, norman, adamson)')
    parser.add_argument('--split', type=str, default='simulation',
                        help='Data split method')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing perturbation data')
    
    # Inference settings
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for inference')
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


def auto_detect_peft_and_hidden(model_path):
    model_path = Path(model_path)
    
    # Extract PEFT strategy from directory name
    peft_strategy = None
    peft_strategies = ['ENCODER', 'TOKEN', 'PREFIX', 'LORA', 'HYBRID']
    
    # Look for PEFT strategy in the path components
    for part in model_path.parts:
        if part.upper() in peft_strategies:
            peft_strategy = part.upper()
            break
    
    # Also check parent directory names
    if peft_strategy is None:
        for parent in [model_path.parent, model_path.parent.parent]:
            for part in parent.parts:
                if part.upper() in peft_strategies:
                    peft_strategy = part.upper()
                    break
            if peft_strategy:
                break
    
    if peft_strategy is None:
        # Check if any directory name contains peft strategy info
        path_str = str(model_path).upper()
        for strategy in peft_strategies:
            if strategy in path_str:
                peft_strategy = strategy
                break
    
    if peft_strategy is None:
        raise ValueError(f"Could not detect PEFT strategy from path: {model_path}")
    
    # Extract parameter value - support two formats
    param_value = None
    
    # Format 1: perturb{VALUE} directory format
    for parent in [model_path.parent, model_path.parent.parent, model_path]:
        param_match = re.search(r'perturb([\d.]+)', parent.name)
        if param_match:
            param_str = param_match.group(1)
            # Convert to appropriate type (float for ratios, int for dimensions)
            if '.' in param_str:
                param_value = float(param_str)
            else:
                param_value = int(param_str)
            # silent
            break
    
    # Format 2: model_hidden{SIZE}.pt file format
    if param_value is None:
        hidden_match = re.search(r'model_hidden(\d+)\.pt$', model_path.name)
        if hidden_match:
            hidden_size = int(hidden_match.group(1))
            # Map hidden_size to parameter value
            param_value = map_hidden_size_to_param(peft_strategy, hidden_size)
            # silent
    
    if param_value is None:
        # Set default values based on PEFT strategy
        defaults = {
            'ENCODER': 0.25,
            'TOKEN': 0.25,
            'PREFIX': 512,
            'LORA': 8,
            'HYBRID': 0.25
        }
        param_value = defaults.get(peft_strategy, 0.25)
        print(f"PEFT param not found; using default {param_value} for {peft_strategy}")
    
    # Convert parameter to PEFT configuration
    peft_config = convert_param_to_peft_config(peft_strategy, param_value)
    
    print(f"PEFT detected: strategy={peft_strategy}, param={param_value}")
    
    return peft_strategy, param_value, peft_config


def map_hidden_size_to_param(peft_strategy, hidden_size):
    """
    Map hidden_size to PEFT parameter value
    - ENCODER/TOKEN/HYBRID: mlp_ratio = hidden_size / 512 (base_hidden_dim=512)
    - PREFIX: token_nums = hidden_size (use directly)
    - LORA: r = hidden_size (use directly)
    """
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


def convert_param_to_peft_config(peft_strategy, param_value):
    """
    Convert parameter value to appropriate PEFT configuration.
    Based on the Reproduction_Perturbation_Replogle.py approach - use simple PeftConfig.
    
    Args:
        peft_strategy: PEFT strategy name ('ENCODER', 'TOKEN', etc.)
        param_value: Parameter value from directory name (not used in simple config)
    
    Returns:
        dict: PEFT configuration parameters
    """
    # Use the same PEFT mapping as Reproduction_Perturbation_Replogle.py
    peft_prompt_relationship = {
        "Encoder_adapter": "ENCODER",
        "Token_adapter": "TOKEN", 
        "Prefix": "PREFIX",
        "LoRA": "LORA",
        "finetune": False
    }
    
    # Map to the expected format
    mapped_type = peft_strategy if peft_strategy in {"ENCODER", "TOKEN", "PREFIX", "LORA", "HYBRID"} else peft_strategy

    # Build PeftConfig with param_value routed to the correct field per method
    cfg_kwargs = dict(peft_type=mapped_type)

    try:
        if mapped_type in {"ENCODER", "TOKEN", "HYBRID"}:
            # For adapters, param_value controls the hidden size via mlp_ratio
            cfg_kwargs["mlp_ratio"] = float(param_value)
        elif mapped_type == "PREFIX":
            # For prefix-tuning, this version of PeftConfig doesn't support token_nums parameter
            # We'll use default settings for PREFIX
            print(f"Warning: PREFIX with param_value={param_value} - using default settings (token_nums not supported)")
        elif mapped_type == "LORA":
            # For LoRA, param_value controls the rank r
            cfg_kwargs["r"] = int(param_value)
        # Other strategies fall back to defaults
    except Exception:
        # If parsing fails, just rely on defaults
        pass

    peft_config = PeftConfig(**cfg_kwargs).to_dict()
    return peft_config


def convert_hidden_to_peft_config(peft_strategy, hidden_dim):
    """
    Legacy function for backward compatibility.
    Convert hidden dimension to appropriate PEFT configuration parameters for perturbation task.
    """
    return convert_param_to_peft_config(peft_strategy, hidden_dim)


def load_model_and_config(args, peft_strategy, auto_peft_config):
    """Load model configuration and vocabulary with auto-detected parameters."""
    print(f"Loading model from: {args.model_path}")
    
    # Load model files - new structure: replogle_k562/PEFT_STRATEGY/perturb{VALUE}/best_model.pt
    model_path = Path(args.model_path)
    
    # Find PEFT strategy directory (one level up from perturb* directory)
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
            Path("/fs/ess/PAS1475/Fei/code/scPEFT-main/scgpt/save/scGPT_human")  # Default pretrained path
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
    
    vocab = GeneVocab.from_file(vocab_file)
    # Add special tokens
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
    
    # Use simple PEFT configuration like in Reproduction_Perturbation_Replogle.py
    # The reproduction code uses: PeftConfig(peft_type=peft_type).to_dict()
    peft_config = auto_peft_config  # This is already created by convert_param_to_peft_config

    # Align adapter layer configuration with the checkpoint to ensure full loading
    try:
        ckpt = torch.load(args.model_path, map_location='cpu')

        # Determine param keys irrespective of packaging
        def extract_param_keys(obj):
            if isinstance(obj, dict):
                # direct state_dict
                if any(isinstance(v, torch.Tensor) for v in obj.values()):
                    return list(obj.keys())
                # wrapped under 'state_dict'
                if 'state_dict' in obj and isinstance(obj['state_dict'], dict):
                    return list(obj['state_dict'].keys())
                # recursively collect
                keys = []
                def walk(d, prefix=''):
                    for k, v in d.items():
                        name = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
                        if isinstance(v, torch.Tensor):
                            keys.append(name)
                        elif isinstance(v, dict):
                            walk(v, name)
                walk(obj)
                return keys
            return []

        param_keys = extract_param_keys(ckpt)
        nlayers = int(model_configs.get("nlayers", 12))

        if peft_strategy in {"LORA", "ENCODER", "HYBRID", "PREFIX"}:
            layer_conf = [False] * nlayers
            if peft_strategy == "LORA":
                patt = re.compile(r"(?:^|\.)transformer_encoder\.layers\.(\d+)\.self_attn\.Wqkv\.")
            elif peft_strategy in {"ENCODER", "HYBRID"}:
                patt = re.compile(r"(?:^|\.)transformer_encoder\.layers\.(\d+)\.space_adapter\.")
            else:  # PREFIX
                patt = re.compile(r"(?:^|\.)transformer_encoder\.layers\.(\d+)\.adapter_embeddings")

            for k in param_keys:
                m = patt.search(k)
                if m:
                    idx = int(m.group(1))
                    if 0 <= idx < nlayers:
                        layer_conf[idx] = True

            # Only override when partially enabled in ckpt (some True, some False)
            if any(layer_conf) and not all(layer_conf):
                peft_config["adapter_layer_conf"] = layer_conf
                # silent
    except Exception as e:
        print(f"Could not infer {peft_strategy} layers from checkpoint: {e}")

    # silent
    
    return vocab, model_configs, peft_config


def create_model(vocab, model_configs, peft_config, device):
    """Create model with the same configuration as Reproduction_Perturbation_Replogle.py"""
    # Extract model architecture parameters (same as reproduction code)
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]  
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]
    dropout = 0  # Fixed value from reproduction code
    
    ntokens = len(vocab)
    
    print("Model architecture:")
    print(f"  d_model={embsize}, d_hid={d_hid}, nhead={nhead}, nlayers={nlayers}, n_layers_cls={n_layers_cls}, dropout={dropout}, vocab={ntokens}")
    
    # Create model with exact same parameters as Reproduction_Perturbation_Replogle.py
    model = TransformerGenerator(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=n_layers_cls,
        n_cls=1,  # Same as reproduction
        vocab=vocab,
        dropout=dropout,
        pad_token="<pad>",
        pad_value=0,  # Same as reproduction
        pert_pad_id=0,  # Same as reproduction
        use_fast_transformer=False,  # Same as reproduction
        peft_config=peft_config
    )
    
    model.to(device)
    return model


def load_and_preprocess_data(args, vocab):
    """Load and preprocess perturbation data."""
    print(f"Dataset: {args.data_name}")
    
    # Load perturbation data using GEARS
    pert_data = PertData(args.data_dir)
    pert_data.load(data_name=args.data_name)
    pert_data.prepare_split(split=args.split, seed=1)
    pert_data.get_dataloader(batch_size=args.batch_size, test_batch_size=args.batch_size)
    
    # Filter genes that are in vocabulary
    genes = pert_data.adata.var["gene_name"].tolist()
    pert_data.adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in genes
    ]
    gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
    # silent
    
    # Create gene_ids mapping
    gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
    )
    
    n_genes = len(genes)
    
    # silent
    
    return {
        'pert_data': pert_data,
        'gene_ids': gene_ids,
        'n_genes': n_genes,
        'genes': genes
    }


def eval_perturb(loader: DataLoader, model: TransformerGenerator, gene_ids, device: torch.device) -> Dict:
    """
    Run model in inference mode using a given data loader.
    Exact copy from Reproduction_Perturbation_Replogle.py
    """
    print("Starting perturbation inference...")
    
    model.eval()
    model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}
    
    include_zero_gene = "all"  # From reproduction code

    for itr, batch in enumerate(loader):
        # quiet batch logs
            
        batch.to(device)
        pert_cat.extend(batch.pert)

        with torch.no_grad():
            # Use exact same call as reproduction code
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

    # silent

    return results


def calculate_perturbation_metrics(inference_results, pert_data):
    """
    Calculate perturbation metrics using GEARS compute_metrics.
    Based on the reference implementation in Reproduction_Perturbation_Replogle.py
    """
    print("Computing metrics...")
    
    # Use scGPT's compute_perturbation_metrics which is more reliable
    try:
        # First try using the scGPT utility function
        from scgpt.utils import compute_perturbation_metrics
        ctrl_adata = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
        metrics = compute_perturbation_metrics(inference_results, ctrl_adata)
        # silent
        
        # Also try GEARS compute_metrics if available
        try:
            _, metrics_pert = compute_metrics(inference_results)
            return metrics, metrics_pert
        except:
            return metrics, {}
        
    except Exception as e:
        print(f"scGPT metric computation failed; trying GEARS: {e}")
        
        # Try GEARS compute_metrics as fallback
        try:
            metrics, metrics_pert = compute_metrics(inference_results)
            # silent
            return metrics, metrics_pert
            
        except Exception as e2:
            print(f"GEARS metric computation failed: {e2}")
            # Fallback to basic metrics
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
    """Print detailed evaluation results."""
    print("\n" + "="*60)
    print("PERTURBATION PREDICTION RESULTS")
    print("="*60)

    def _fmt_num(val, default="N/A", fmt=".6f"):
        try:
            import numpy as _np
            if isinstance(val, (int, float)) or isinstance(val, _np.floating):
                return f"{float(val):{fmt}}"
        except Exception:
            pass
        return default

    print(f"Mean Squared Error (MSE): {_fmt_num(metrics.get('mse'))}")
    print(f"Mean Absolute Error (MAE): {_fmt_num(metrics.get('mae'))}")
    print(f"Pearson Correlation: {_fmt_num(metrics.get('pearson'), fmt='.4f')}")
    
    # Print additional GEARS metrics if available
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
    
    print("="*60)


def save_comprehensive_results(inference_results, metrics, args, data_dict, peft_strategy=None, param_value=None):
    """Save all inference results and visualizations."""
    
    # Create output directory with method info
    if peft_strategy and param_value is not None:
        if isinstance(param_value, float):
            method_name = f"{peft_strategy}_perturb{param_value}"
        else:
            method_name = f"{peft_strategy}_perturb{param_value}"
    else:
        method_name = "perturbation_inference"
    
    output_dir = Path(args.output_dir) / f"{method_name}_{args.data_name}_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving comprehensive results to {output_dir}")
    
    # Prepare results dictionary
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
    
    if inference_results['pred_de'] is not None:
        results_dict['pred_de'] = inference_results['pred_de']
        results_dict['truth_de'] = inference_results['truth_de']
    
    # Save as pickle
    with open(output_dir / 'perturbation_inference_results.pkl', 'wb') as f:
        pickle.dump(results_dict, f)
    print(f"Results saved to {output_dir / 'perturbation_inference_results.pkl'}")
    
    # Save metrics as JSON
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
    print(f"Metrics saved to {output_dir / 'metrics.json'}")
    
    # Create basic visualization
    try:
        create_perturbation_plots(inference_results, output_dir)
    except Exception as e:
        print(f"Could not create plots: {e}")
    
    print(f"All comprehensive results saved to {output_dir}")
    return output_dir


def create_perturbation_plots(inference_results, save_dir):
    """Create basic perturbation prediction plots."""
    pred = inference_results['pred'].flatten()
    truth = inference_results['truth'].flatten()
    
    # Scatter plot of predictions vs truth
    plt.figure(figsize=(8, 8))
    plt.scatter(truth, pred, alpha=0.5, s=1)
    plt.plot([truth.min(), truth.max()], [truth.min(), truth.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Perturbation Predictions vs Truth')
    plt.tight_layout()
    plt.savefig(save_dir / 'predictions_vs_truth.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Histogram of errors
    errors = pred - truth
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=50, alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.tight_layout()
    plt.savefig(save_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Basic plots saved to {save_dir}")


def main():
    """Main perturbation inference pipeline with auto-detection."""
    args = parse_arguments()
    
    set_seed(42)  # Match training seed
    
    # Device configuration
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    
    # Auto-detect PEFT strategy and parameters from model path
    print("Auto-detecting PEFT configuration...")
    try:
        peft_strategy, param_value, auto_peft_config = auto_detect_peft_and_hidden(args.model_path)
    except ValueError as e:
        print(f"Warning: PEFT auto-detect failed: {e}; using defaults")
        peft_strategy = "TOKEN"  # Default for perturbation
        param_value = None
        auto_peft_config = {'mlp_ratio': 0.5, 'skip_connect': True}
    
    print("Starting perturbation inference...")
    print(f"  model={args.model_path}")
    print(f"  dataset={args.data_name}")
    print(f"  split={args.split}")
    print(f"  peft={peft_strategy}:{param_value}")
    print(f"  batch_size={args.batch_size}")
    
    # Load model configuration and vocabulary
    vocab, model_configs, peft_config = load_model_and_config(args, peft_strategy, auto_peft_config)
    
    # Load and preprocess perturbation data
    data_dict = load_and_preprocess_data(args, vocab)
    
    # Create model with auto-detected parameters
    model = create_model(vocab, model_configs, peft_config, device)
    
    # Load model weights with the same approach as Reproduction_Perturbation_Replogle.py
    print(f"Loading weights: {args.model_path}")
    if Path(args.model_path).exists():
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print("Weights loaded successfully")
        except Exception as e:
            print(f"Direct load failed: {e}; trying compatibility mode...")
            # Use the same fallback as reproduction code
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
    
    # Log parameter counts and list all learnable parameters (match training-style logging)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: total={total_params/1e6:.2f}M, trainable={trainable_params/1e6:.2f}M ({trainable_params/total_params*100:.2f}%)")
    
    # Run perturbation inference (same as reproduction code)
    print("Running inference...")
    start_time = time.time()
    
    test_loader = data_dict['pert_data'].dataloader['test_loader']
    test_res = eval_perturb(test_loader, model, data_dict['gene_ids'], device)
    
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.2f}s")
    
    # Calculate metrics using GEARS (same as reproduction code)
    test_metrics, test_pert_res = compute_metrics(test_res)
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
    print(f"MSE: {metrics.get('mse', 'N/A'):.6f}")
    print(f"Pearson: {metrics.get('pearson', 'N/A'):.4f}")
    
    return inference_results, metrics


if __name__ == "__main__":
    inference_results, metrics = main()
