#!/usr/bin/env python3

import argparse
import json
import os
import pickle
import time
import warnings
from typing import Dict
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import issparse
from sklearn.metrics import (
    balanced_accuracy_score, precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, classification_report
)
from torch.utils.data import DataLoader, Dataset
import re

# Import scGPT modules - keep scPEFT folder structure
import sys
sys.path.insert(0, "../")

from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt.utils import PeftConfig, freeze_parameters, DownstreamTasks, set_seed

# Set up environment
sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')


class SeqDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def run_inference(model, dataloader, device, vocab, pad_token, amp=True):
    """Run inference and return predictions, probabilities, labels, embeddings."""
    print("Starting inference...")
    
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_embeddings = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)
            
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            
            with torch.amp.autocast('cuda', enabled=amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                    CLS=True,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=False,
                )
            
            cls_output = output_dict["cls_output"]
            probabilities = torch.softmax(cls_output, dim=1)
            predictions = cls_output.argmax(dim=1)
            
            all_predictions.append(predictions.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_labels.append(celltype_labels.cpu().numpy())
            
            # Store embeddings if available
            if "cell_emb" in output_dict:
                all_embeddings.append(output_dict["cell_emb"].cpu().numpy())
            

    
    predictions = np.concatenate(all_predictions, axis=0)
    probabilities = np.concatenate(all_probabilities, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    if all_embeddings:
        embeddings = np.concatenate(all_embeddings, axis=0)
    else:
        embeddings = None
    
    print(f"Inference completed. samples={len(predictions)}")
    
    return {
        'predictions': predictions,
        'probabilities': probabilities,
        'labels': labels,
        'embeddings': embeddings
    }


def calculate_comprehensive_metrics(predictions, labels, probabilities=None):
    """Calculate comprehensive evaluation metrics"""
    accuracy = accuracy_score(labels, predictions)
    balanced_accuracy = balanced_accuracy_score(labels, predictions)
    macro_precision = precision_score(labels, predictions, average='macro', zero_division=0)
    macro_recall = recall_score(labels, predictions, average='macro', zero_division=0)
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    
    weighted_precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    weighted_recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    weighted_f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    per_class_precision = precision_score(labels, predictions, average=None, zero_division=0)
    per_class_recall = recall_score(labels, predictions, average=None, zero_division=0)
    per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)
    
    cm = confusion_matrix(labels, predictions)
    class_report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'confusion_matrix': cm,
        'classification_report': class_report
    }
    
    return metrics


def print_detailed_results(metrics, id2type):
    """Print detailed evaluation results"""
    print("\n" + "="*60)
    print("COMPREHENSIVE INFERENCE RESULTS")
    print("="*60)
    
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1 Score: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1 Score: {metrics['weighted_f1']:.4f}")
    
    print(f"\nMacro Metrics:")
    print(f"  Precision: {metrics['macro_precision']:.4f}")
    print(f"  Recall: {metrics['macro_recall']:.4f}")
    print(f"  F1-Score: {metrics['macro_f1']:.4f}")
    
    print(f"\nWeighted Metrics:")
    print(f"  Precision: {metrics['weighted_precision']:.4f}")
    print(f"  Recall: {metrics['weighted_recall']:.4f}")
    print(f"  F1-Score: {metrics['weighted_f1']:.4f}")
    
    print(f"\nPer-class Results:")
    for i, cell_type in id2type.items():
        if i < len(metrics['per_class_f1']):
            print(f"  {cell_type}:")
            print(f"    Precision: {metrics['per_class_precision'][i]:.4f}")
            print(f"    Recall: {metrics['per_class_recall'][i]:.4f}")
            print(f"    F1-Score: {metrics['per_class_f1'][i]:.4f}")
    
    print("="*60)


def create_confusion_matrix_plot(confusion_matrix, class_names, save_path):
    """Create and save confusion matrix visualization"""
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count'}
    )
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Cell Type', fontsize=12)
    plt.ylabel('True Cell Type', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # silent save


def create_metrics_barplot(metrics, id2type, save_path):
    """Create bar plot of per-class metrics"""
    class_names = [id2type[i] for i in range(len(metrics['per_class_f1']))]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].bar(class_names, metrics['per_class_precision'], color='skyblue', alpha=0.8)
    axes[0].set_title('Per-class Precision', fontweight='bold')
    axes[0].set_ylabel('Precision')
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].set_ylim(0, 1)
    
    axes[1].bar(class_names, metrics['per_class_recall'], color='lightcoral', alpha=0.8)
    axes[1].set_title('Per-class Recall', fontweight='bold')
    axes[1].set_ylabel('Recall')
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].set_ylim(0, 1)
    
    axes[2].bar(class_names, metrics['per_class_f1'], color='lightgreen', alpha=0.8)
    axes[2].set_title('Per-class F1-score', fontweight='bold')
    axes[2].set_ylabel('F1-score')
    axes[2].set_xticklabels(class_names, rotation=45, ha='right')
    axes[2].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # silent save


def save_comprehensive_results(inference_results, metrics, args, id2type, peft_strategy=None, hidden_dim=None):
    """Save all inference results and visualizations (silent)."""
    
    # Create output directory with method and fold info
    if peft_strategy and hidden_dim:
        method_name = f"{peft_strategy}_hidden{hidden_dim}"
    else:
        method_name = "inference"
    
    output_dir = Path(args.output_dir) / f"{method_name}_fold{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # silent save
    
    # Prepare results dictionary
    results_dict = {
        'predictions': inference_results['predictions'],
        'probabilities': inference_results['probabilities'],
        'labels': inference_results['labels'],
        'metrics': metrics,
        'id2type': id2type,
        'config': {
            'model_path': str(args.model_path),
            'data_path': str(args.data_path),
            'dataset_name': args.dataset_name,
            'fold': args.fold,
            'batch_size': args.batch_size,
            'device': str(args.device),
            'peft_strategy': peft_strategy,
            'hidden_dim': hidden_dim,
        }
    }
    
    if inference_results['embeddings'] is not None:
        results_dict['embeddings'] = inference_results['embeddings']
    
    # Save as pickle
    with open(output_dir / 'inference_results.pkl', 'wb') as f:
        pickle.dump(results_dict, f)
    # silent save
    
    # Save metrics as JSON
    metrics_json = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_json[key] = value.tolist()
        elif key == 'classification_report':
            metrics_json[key] = value
        else:
            metrics_json[key] = float(value) if isinstance(value, (np.float32, np.float64)) else value
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)
    # silent save
    
    # Create visualizations
    class_names = [id2type[i] for i in range(len(id2type))]
    
    create_confusion_matrix_plot(
        metrics['confusion_matrix'], 
        class_names, 
        output_dir / 'confusion_matrix.png'
    )
    
    create_metrics_barplot(
        metrics, 
        id2type, 
        output_dir / 'per_class_metrics.png'
    )
    
    # Save detailed results as CSV
    results_df = pd.DataFrame({
        'True_Label': inference_results['labels'],
        'Predicted_Label': inference_results['predictions'],
        'True_CellType': [id2type[label] for label in inference_results['labels']],
        'Predicted_CellType': [id2type[pred] for pred in inference_results['predictions']],
        'Max_Probability': np.max(inference_results['probabilities'], axis=1),
        'Correct': inference_results['labels'] == inference_results['predictions']
    })
    
    for i, class_name in enumerate(class_names):
        results_df[f'Prob_{class_name}'] = inference_results['probabilities'][:, i]
    
    results_df.to_csv(output_dir / 'detailed_results.csv', index=False)
    # silent save
    return output_dir


def prepare_inference_data(vocab, adata, gene_ids, pad_token, pad_value, max_seq_len, include_zero_gene, input_layer_key, mask_ratio, mask_value):
    """Prepare data for inference with tokenization"""
    
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )

    celltypes_labels = adata.obs["celltype_id"].values
    batch_ids = adata.obs["batch_id"].values

    tokenized_test = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=include_zero_gene,
    )

    input_values_test = random_mask_value(
        tokenized_test["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )

    test_data_pt = {
        "gene_ids": tokenized_test["genes"],
        "values": input_values_test,
        "target_values": tokenized_test["values"],
        "batch_labels": torch.from_numpy(batch_ids).long(),
        "celltype_labels": torch.from_numpy(celltypes_labels).long(),
    }
    
    # omit verbose dataset shapes
    
    return test_data_pt


def create_dataloader(data_pt: Dict[str, torch.Tensor], batch_size: int, num_workers: int = 0):
    """Create DataLoader for inference"""
    if num_workers == 0:
        try:
            num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)
        except:
            num_workers = 0
    
    dataset = SeqDataset(data_pt)
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    return dataloader


def test(model, adata, device, vocab, gene_ids, input_layer_key, 
         max_seq_len, pad_token, pad_value, mask_ratio, mask_value, 
         include_zero_gene, eval_batch_size):
    
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )

    celltypes_labels = adata.obs["celltype_id"].tolist()
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)

    tokenized_test = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=include_zero_gene,
    )

    input_values_test = random_mask_value(
        tokenized_test["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )

    test_data_pt = {
        "gene_ids": tokenized_test["genes"],
        "values": input_values_test,
        "target_values": tokenized_test["values"],
        "batch_labels": torch.from_numpy(batch_ids).long(),
        "celltype_labels": torch.from_numpy(celltypes_labels).long(),
    }

    test_loader = DataLoader(
        dataset=SeqDataset(test_data_pt),
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False
    )

    model.eval()
    predictions = evaluate(model, test_loader, device, vocab, pad_token)


    
    # Calculate metrics
    accuracy = balanced_accuracy_score(celltypes_labels, predictions)
    precision = precision_score(celltypes_labels, predictions, average="macro")
    recall = recall_score(celltypes_labels, predictions, average="macro")
    macro_f1 = f1_score(celltypes_labels, predictions, average="macro")

    print(f"Balanced accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, Macro F1: {macro_f1:.3f}")

    results = {
        "test/accuracy": accuracy,
        "test/precision": precision,
        "test/recall": recall,
        "test/macro_f1": macro_f1,
    }

    return predictions, celltypes_labels, results


def parse_arguments():
    """Parse command line arguments with comprehensive options."""
    parser = argparse.ArgumentParser(description='Cell Type Classification Inference with Auto-detection')
    
    # Main arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Full path to model file (e.g., ../ENCODER/model_hidden8.pt)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Base directory containing dataset folders')
    parser.add_argument('--dataset_name', type=str, required=True, choices=["COVID-19", "NSCLC"],
                        help='Dataset name')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help='Fold number to use for inference (0-4)')
    
    # Inference settings
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use Automatic Mixed Precision')
    
    # Data processing settings
    parser.add_argument('--mask_ratio', type=float, default=0.0, help='Mask ratio for inference')
    parser.add_argument('--n_bins', type=int, default=51, help='Number of bins for gene expression')
    parser.add_argument('--include_zero_gene', action='store_true', default=False, 
                        help='Include zero genes in analysis')
    
    # Output settings
    parser.add_argument('--save_results', action='store_true', default=True,
                        help='Save inference results')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Directory to save inference results')
    
    return parser.parse_args()


def auto_detect_peft_and_hidden(model_path):
    """
    Auto-detect PEFT strategy and hidden dimension from model path.
    
    Example path: /mnt/c/Users/wxy/Desktop/esmc/checkpoint/NSCLC/ENCODER/model_hidden8.pt
    Returns: ('ENCODER', 8, converted_peft_config)
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
        raise ValueError(f"Could not detect PEFT strategy from path: {model_path}")
    
    # Extract hidden dimension from filename
    filename = model_path.stem  # Get filename without extension
    hidden_match = re.search(r'hidden(\d+)', filename)
    
    if hidden_match:
        hidden_dim = int(hidden_match.group(1))
    else:
        raise ValueError(f"Could not extract hidden dimension from filename: {filename}")
    
    # Convert hidden dimension to PEFT configuration parameters
    peft_config = convert_hidden_to_peft_config(peft_strategy, hidden_dim)
    
    print(f"PEFT detected: strategy={peft_strategy}, hidden={hidden_dim}")
    
    return peft_strategy, hidden_dim, peft_config


def convert_hidden_to_peft_config(peft_strategy, hidden_dim):
    """
    Convert hidden dimension to appropriate PEFT configuration parameters.
    
    Based on the training code analysis:
    - ENCODER default: mlp_ratio = 0.25
    - TOKEN default: mlp_ratio = 1  
    - PREFIX default: token_nums = 8
    - LORA default: r = 8, lora_alpha = 1
    
    The hidden_dim likely represents different scales of these parameters.
    
    Args:
        peft_strategy: PEFT strategy name ('ENCODER', 'TOKEN', etc.)
        hidden_dim: Hidden dimension value from filename
    
    Returns:
        dict: PEFT configuration parameters
    """
    config = {}
    
    if peft_strategy in ['ENCODER', 'TOKEN']:
        # For adapter-based methods, use hidden_dim directly as the hidden dimension
        # Convert to mlp_ratio based on typical scGPT embsize=512
        base_embsize = 512
        if peft_strategy == 'ENCODER':
            # ENCODER typically uses smaller ratios, so hidden_dim might represent scaled mlp_ratio
            # If hidden_dim is like 64, 256, 512, it might be the actual hidden size
            mlp_ratio = hidden_dim / base_embsize
            config['mlp_ratio'] = mlp_ratio
        elif peft_strategy == 'TOKEN':
            # TOKEN typically uses ratio=1, so scale accordingly  
            mlp_ratio = hidden_dim / base_embsize
            config['mlp_ratio'] = mlp_ratio
        
        config['skip_connect'] = True  # Default setting
        
    elif peft_strategy == 'PREFIX':
        # For PREFIX, hidden_dim directly maps to token_nums
        # Scale from default 8: if hidden_dim=8 use 8, if hidden_dim=64 use 64, etc.
        config['token_nums'] = hidden_dim
        
    elif peft_strategy == 'LORA':
        # For LoRA, hidden_dim maps to rank (r)
        # Scale from default 8: if hidden_dim=8 use r=8, if hidden_dim=64 use r=64, etc.
        config['r'] = hidden_dim
        config['lora_alpha'] = 1  # Scale alpha based on rank
        config['enable_lora'] = [True, False, True]  # Match training default
    
    return config


def main():
    # Parse arguments and set up
    args = parse_arguments()
    
    # Set seed for reproducibility
    set_seed(0)
    
    # Device configuration
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Auto-detect PEFT strategy and parameters from model path
    print("Auto-detecting PEFT configuration from model path...")
    try:
        peft_strategy, hidden_dim, auto_peft_config = auto_detect_peft_and_hidden(args.model_path)
    except ValueError as e:
        print(f"Warning: Could not auto-detect PEFT config: {e}")
        print("PEFT auto-detect failed; inspecting checkpoint...")
        peft_strategy = None
        hidden_dim = None
        auto_peft_config = {}
    
    # Model path and configuration
    model_path = Path(args.model_path)
    model_dir = model_path.parent
    
    # Load vocabulary
    vocab_file = model_dir / "vocab.json"
    if not vocab_file.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_file}")
    
    vocab = GeneVocab.from_file(vocab_file)
    # Add special tokens
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    print(f"Loaded vocabulary with {len(vocab)} genes")
    
    # Load model configuration
    config_file = model_dir / "args.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Model config file not found: {config_file}")
    
    with open(config_file, "r") as f:
        model_config = json.load(f)
    
    # Set parameters from argument parsing and config file
    pad_token = "<pad>"
    mask_ratio = args.mask_ratio
    include_zero_gene = args.include_zero_gene
    max_seq_len = 2001
    n_bins = model_config.get("n_bins", args.n_bins)
    
    # Input embedding style settings
    input_emb_style = "continuous"
    if input_emb_style == "category":
        mask_value = n_bins + 1
        pad_value = n_bins
        n_input_bins = n_bins + 2
    else:
        mask_value = -1
        pad_value = -2
        n_input_bins = n_bins
    
    input_layer_key = "X_binned"
    
    # Dataset configuration
    if args.dataset_name == "COVID-19":
        celltype_key = "celltype"
        data_is_raw = False
    elif args.dataset_name == "NSCLC":
        celltype_key = "cell_type"
        data_is_raw = False  # NSCLC data is preprocessed, no need for log1p
    
    # Load test data
    data_dir = Path(args.data_path) / args.dataset_name / str(args.fold)
    test_file = data_dir / f"{args.dataset_name}_test{args.fold}.h5ad"
    
    print(f"Loading test data: {test_file}")
    adata_test = sc.read(test_file)
    
    # Process data (following reference code)
    adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "2"  # Consistent with reference code
    
    if "gene_name" in adata_test.var.columns:
        adata_test.var.set_index(adata_test.var["gene_name"], inplace=True)
    else:
        # Use existing index as gene names
        adata_test.var["gene_name"] = adata_test.var.index.tolist()
    
    # Filter genes matching vocabulary first, then do other processing
    adata_test.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata_test.var.index
    ]
    gene_ids_in_vocab = np.array(adata_test.var["id_in_vocab"])
    matched_genes = np.sum(gene_ids_in_vocab >= 0)
    print(f"Matched genes: {matched_genes}/{len(gene_ids_in_vocab)}")
    adata_test = adata_test[:, adata_test.var["id_in_vocab"] >= 0]
    
    # Process cell type labels
    if celltype_key != "celltype":
        adata_test.obs["celltype"] = adata_test.obs[celltype_key].astype("category")
    
    # Data preprocessing (must be done after gene filtering)
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=False,  # Already filtered, no need to filter again
        filter_cell_by_counts=False,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=data_is_raw,
        result_log1p_key="X_log1p",
        subset_hvg=False,
        hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        binning=n_bins,
        result_binned_key="X_binned",
    )
    
    preprocessor(adata_test, batch_key=None)
    
    # Create label mappings
    celltype_id_labels = adata_test.obs["celltype"].astype("category").cat.codes.values
    adata_test.obs["celltype_id"] = celltype_id_labels
    
    batch_id_labels = adata_test.obs["batch_id"].astype(int).values  # Convert directly to int
    adata_test.obs["batch_id"] = batch_id_labels
    
    celltypes = adata_test.obs["celltype"].unique()
    num_types = len(np.unique(celltype_id_labels))
    id2type = dict(enumerate(adata_test.obs["celltype"].astype("category").cat.categories))
    
    print(f"Number of cell types: {num_types}")
    print(f"Cell types: {list(id2type.values())}")
    
    # Gene ID matching
    genes = adata_test.var.index.tolist()  # Use index directly
    gene_ids = np.array(vocab(genes), dtype=int)  # Use vocab() function
    
    # Load checkpoint and detect PEFT configuration
    print("Loading checkpoint for PEFT configuration detection...")
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Check actual number of classes in model classification head
    if 'cls_decoder.out_layer.weight' in checkpoint:
        actual_n_cls = checkpoint['cls_decoder.out_layer.weight'].shape[0]
        print(f"Number of classes in model classification head: {actual_n_cls}")
        if actual_n_cls != num_types:
            print(f"Warning: Model class count ({actual_n_cls}) does not match data class count ({num_types})!")
            print(f"This means the model was trained on {actual_n_cls} classes but inferring {num_types} classes")
    else:
        print("Classification head weights not found")
    
    # Combine auto-detected PEFT config with checkpoint-based detection
    if peft_strategy and auto_peft_config:
        # Use auto-detected configuration
        print(f"Using auto-detected PEFT configuration: {peft_strategy}")
        # silent config dump
        
        # Create PEFT configuration with auto-detected parameters for scPEFT
        # Note: set use_default_settings=False to use our custom detected parameters
        peft_config_obj = PeftConfig(
            model_nlayers=model_config["nlayers"],
            use_default_settings=False,
            peft_type=peft_strategy,
            adapter_layer_conf=[True, True, True, True, True, True, False, False, False, False, False, False],  # First 6 layers
            **auto_peft_config  # Apply auto-detected parameters directly
        )
        
        peft_config = peft_config_obj.to_dict()
        print(f"Final PEFT config: {peft_config}")
    else:
        # Fall back to manual checkpoint-based detection
        print("Falling back to checkpoint-based PEFT detection...")
        
        # Auto-detect PEFT method and parameters
        peft_method = None
        peft_params = {'peft_type': False}
        
        # Check for space_adapter (ENCODER method)
        if any("space_adapter" in key for key in checkpoint.keys()):
            peft_method = "ENCODER"
            peft_strategy = "ENCODER"
            # Detect hidden size from actual model - key fix!
            for key, value in checkpoint.items():
                if "space_adapter.fc1.weight" in key:
                    adapter_hidden_size = value.shape[0]  # adapter hidden size
                    model_embsize = value.shape[1]  # model embedding size (should be 512)
                    mlp_ratio = adapter_hidden_size / model_embsize
                    print(f"Detected ENCODER method, adapter_hidden_size={adapter_hidden_size}, model_embsize={model_embsize}, mlp_ratio={mlp_ratio}")
                    break
            
            # Detect which layers have adapters
            adapter_layers = []
            for i in range(12):  # Assume maximum 12 layers
                if f"transformer_encoder.layers.{i}.space_adapter.fc1.weight" in checkpoint:
                    adapter_layers.append(True)
                else:
                    adapter_layers.append(False)
            
            # Ensure total 12 layers
            while len(adapter_layers) < 12:
                adapter_layers.append(False)
                
            print(f"Detected adapter layer configuration: {adapter_layers}")
            
            peft_params = {
                'peft_type': 'ENCODER',
                'use_default_settings': False,
                'adapter_layer_conf': adapter_layers,
                'mlp_ratio': mlp_ratio,
                'skip_connect': True
            }
        
        # Check for LoRA parameters
        elif any("lora_" in key for key in checkpoint.keys()):
            peft_method = "LORA"
            peft_strategy = "LORA"
            print("Detected LORA method")
            peft_params = {
                'peft_type': 'LORA',
                'use_default_settings': False,
                'adapter_layer_conf': [True]*6 + [False]*6,
                'r': 128,
                'lora_alpha': 1,
                'enable_lora': [True, False, True]
            }
        
        # Check for other PEFT methods
        elif any("prefix" in key.lower() for key in checkpoint.keys()):
            peft_method = "PREFIX"
            peft_strategy = "PREFIX"
            print("Detected PREFIX method")
            peft_params = {
                'peft_type': 'PREFIX',
                'use_default_settings': False,
                'adapter_layer_conf': [True]*6 + [False]*6,
                'token_nums': 128
            }
        
        else:
            print("No PEFT method detected, using standard model")
            peft_params = {'peft_type': False}
            peft_strategy = None
        
        print(f"PEFT method: {peft_method}")
        print(f"PEFT parameters: {peft_params}")
        
        # Create PeftConfig object for scPEFT
        peft_config_obj = PeftConfig(
            model_nlayers=model_config["nlayers"],
            **peft_params
        )
        peft_config = peft_config_obj.to_dict()
    
    # Validate key parameters
    # concise model param summary
    print("Model params:")
    print(f"  embsize={model_config['embsize']}, nlayers={model_config['nlayers']}, n_layers_cls={model_config.get('n_layers_cls', 3)}, dropout={model_config['dropout']}")
    print(f"  peft={peft_strategy}")
    
    # Create model
    print("Model architecture:")
    print(f"  d_model={model_config['embsize']}, d_hid={model_config['d_hid']}, nhead={model_config['nheads']}, nlayers={model_config['nlayers']}, vocab={len(vocab)}, n_cls={num_types}")
    
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_config["embsize"],
        nhead=model_config["nheads"],
        d_hid=model_config["d_hid"],
        nlayers=model_config["nlayers"],
        nlayers_cls=model_config.get("n_layers_cls", 3),
        n_cls=num_types,
        vocab=vocab,
        dropout=model_config["dropout"],
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        num_batch_labels=None,
        domain_spec_batchnorm=False,
        input_emb_style=input_emb_style,
        n_input_bins=n_input_bins,
        cell_emb_style="cls",
        mvc_decoder_style="inner product",
        ecs_threshold=0.0,
        explicit_zero_prob=False,
        use_fast_transformer=False,
        fast_transformer_backend="flash",
        pre_norm=False,
        peft_config=peft_config
    )
    
    # Load model weights - optimized loading logic
    print("Loading model weights...")
    
    # Check if checkpoint contains complete model
    checkpoint_keys = set(checkpoint.keys())
    encoder_keys = [k for k in checkpoint_keys if "transformer_encoder" in k and not "space_adapter" in k]
    
    # omit verbose checkpoint key counts
    
    # Load fine-tuned weights (PEFT-trained model) - the checkpoint should contain complete weights
    print(f"Loading weights from {model_path}")
    try:
        model.load_state_dict(checkpoint, strict=False)  # Use strict=False for PEFT compatibility
        print("Successfully loaded model weights")
    except Exception as e:
        print(f"Warning: Could not load some model weights: {e}")
        print("Some parameters may remain at random values")
    
    # Apply PEFT parameter freezing after loading weights
    if peft_strategy:
        print(f"Freezing parameters for PEFT: {peft_strategy}")
        freeze_parameters(model, DownstreamTasks.Identification)
    
    # Log parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: total={total_params/1e6:.2f}M, trainable={trainable_params/1e6:.2f}M ({trainable_params/total_params*100:.2f}%)")
    
    model.to(device)
    model.eval()
    
    # Prepare tokenized data for inference
    # omit tokenization logs
    test_data_pt = prepare_inference_data(vocab, adata_test, gene_ids, pad_token, pad_value, max_seq_len, include_zero_gene, input_layer_key, mask_ratio, mask_value)
    
    # Create DataLoader 
    test_dataloader = create_dataloader(test_data_pt, args.batch_size)
    
    # Run enhanced inference
    print(f"Running on {len(adata_test)} cells...")
    start_time = time.time()
    
    inference_results = run_inference(model, test_dataloader, device, vocab, pad_token, args.amp)
    
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.2f}s, per_cell={inference_time/len(adata_test):.4f}s")
    
    # Calculate comprehensive metrics
    print("Computing metrics...")
    comprehensive_metrics = calculate_comprehensive_metrics(
        inference_results['predictions'],
        inference_results['labels'],
        inference_results['probabilities']
    )
    
    # Display detailed results
    print_detailed_results(comprehensive_metrics, id2type)
    
    # Save comprehensive results
    if args.save_results:
        output_dir = save_comprehensive_results(
            inference_results, 
            comprehensive_metrics, 
            args, 
            id2type, 
            peft_strategy, 
            hidden_dim if 'hidden_dim' in locals() else None
        )
    
    print(f"\nInference completed successfully!")
    print(f"Classification F1 Score: {comprehensive_metrics['macro_f1']:.4f}")
    print(f"Balanced Accuracy: {comprehensive_metrics['balanced_accuracy']:.4f}")
    
    return inference_results, comprehensive_metrics


if __name__ == "__main__":
    inference_results, comprehensive_metrics = main()
