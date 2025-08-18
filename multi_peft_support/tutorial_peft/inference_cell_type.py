"""
Cell Type Classification Inference with Parameter-Efficient Fine-Tuning (PEFT)

This script performs inference using a pre-trained scGPT model fine-tuned for cell type 
classification using various PEFT strategies. It loads the trained model and performs
prediction on test datasets with comprehensive evaluation metrics.

Key Features:
- Load PEFT fine-tuned models (ENCODER, TOKEN, PREFIX, LORA)
- Cell type classification inference on NSCLC dataset
- Comprehensive evaluation metrics and visualization
- Support for multiple PEFT strategy combinations
- Confusion matrix and performance analysis

Usage:
    python inference_cell_type.py --model_path ./save_dir/model.pt --data_path ./data

The script will automatically:
1. Load the fine-tuned model with PEFT configuration
2. Process test data with same preprocessing pipeline
3. Perform inference and calculate metrics
4. Generate visualizations and save results
"""

import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import List, Tuple, Dict, Union, Optional
import warnings
import pandas as pd
import pickle
import torch
import scanpy as sc
import argparse
import seaborn as sns
import numpy as np
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

# Add parent directory to path for scGPT imports
sys.path.insert(0, "../")

# Import scGPT modules
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator  # Core transformer and discriminator models
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value  # Tokenization utilities
from scgpt.loss import (  # Loss functions
    masked_mse_loss,
    criterion_neg_log_bernoulli,
)
from scgpt.tokenizer.gene_tokenizer import GeneVocab  # Gene vocabulary for tokenization
from scgpt.preprocess import Preprocessor  # Data preprocessing pipeline
from scgpt import SubsetsBatchSampler  # Custom batch sampler for domain-specific batching
from scgpt.utils import set_seed, PeftConfig, freeze_parameters, DownstreamTasks, load_pretrained  # Utility functions

# Set up plotting and environment
sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"  # Suppress KMP warnings
warnings.filterwarnings('ignore')  # Suppress other warnings

def parse_arguments():
    """Parse command line arguments for inference configuration.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing all inference settings
    """
    parser = argparse.ArgumentParser(description='Cell Type Classification Inference with PEFT')
    
    # Model and data paths
    parser.add_argument('--model_dir', type=str, default='/mnt/c/Users/wxy/Desktop/esmc/multi_peft_support/all_result/NSCLC',
                        help='Base directory containing PEFT method directories')
    parser.add_argument('--peft_method', type=str, default='peft_NSCLC_ENCODER_TOKEN_PREFIX_LORA',
                        help='PEFT method name (directory name under model_dir)')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help='Fold number to use for inference (0-4)')
    parser.add_argument('--pretrained_model', type=str, default='./scGPT_human',
                        help='Path to pretrained scGPT model')
    parser.add_argument('--data_base_dir', type=str, 
                        default='/mnt/c/Users/wxy/Desktop/esmc/multi_peft_support',
                        help='Base directory containing fold data directories')
    
    # Inference settings
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use Automatic Mixed Precision')
    
    # Data processing settings
    parser.add_argument('--mask_ratio', type=float, default=0.0, help='Mask ratio for inference')
    parser.add_argument('--n_bins', type=int, default=51, help='Number of bins for gene expression')
    parser.add_argument('--include_zero_gene', action='store_true', default=False, 
                        help='Include zero genes in analysis')
    
    # PEFT configuration
    parser.add_argument('--peft_strategies', nargs='+', 
                        default=["ENCODER", "TOKEN", "PREFIX", "LORA"],
                        help='PEFT strategies used in training')
    
    # Output settings
    parser.add_argument('--save_results', action='store_true', default=True,
                        help='Save inference results')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Directory to save inference results')
    
    return parser.parse_args()

# Global configuration from training - must match training hyperparameters
hyperparameter_defaults = dict(
    seed=0,  # Random seed for reproducibility
    dataset_name="NSCLC",  # Dataset identifier
    mask_ratio=0.0,  # Masking ratio for inference (typically 0.0)
    n_bins=51,  # Number of bins for gene expression discretization
    dropout=0.2,  # Dropout probability (used in model architecture)
    include_zero_gene=False,  # Whether to include zero-expression genes
    fast_transformer=False,  # Whether to use fast transformer implementation
    pre_norm=False,  # Whether to use pre-normalization in transformer
    amp=True,  # Automatic Mixed Precision for faster inference
)

class InferenceConfig:
    """Configuration class for inference settings matching training setup.
    
    This class ensures that all inference settings match the training configuration
    to maintain consistency in data processing and model behavior.
    
    Args:
        args: Command line arguments from argparse
    """
    def __init__(self, args):
        # Copy from hyperparameter_defaults and update with args
        for key, value in hyperparameter_defaults.items():
            setattr(self, key, value)
        
        # Update with command line arguments
        self.base_model_dir = Path(args.model_dir)
        self.peft_method = args.peft_method
        self.fold = args.fold
        
        # Construct specific model directory and data directory based on fold
        self.model_dir = self.base_model_dir / self.peft_method
        self.model_file = self.model_dir / f"modelfold{self.fold}.pt"
        
        self.pretrained_model = args.pretrained_model
        self.data_base_dir = Path(args.data_base_dir)
        self.data_dir = self.data_base_dir / str(self.fold + 1)  # fold 0 -> directory 1, fold 1 -> directory 2, etc.
        
        self.batch_size = args.batch_size
        self.amp = args.amp
        self.mask_ratio = args.mask_ratio
        self.n_bins = args.n_bins
        self.include_zero_gene = args.include_zero_gene
        
        # Extract PEFT strategies from method name
        # e.g., "peft_NSCLC_ENCODER_TOKEN_PREFIX_LORA" -> ["ENCODER", "TOKEN", "PREFIX", "LORA"]
        method_parts = self.peft_method.split('_')[2:]  # Remove "peft_NSCLC" prefix
        self.peft_strategies = method_parts if method_parts else args.peft_strategies
        
        self.output_dir = Path(args.output_dir)
        
        # Device configuration
        if args.device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(args.device)
        
        # Create output directory with fold and method info
        self.output_dir = self.output_dir / f"{self.peft_method}_fold{self.fold}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model-specific settings (consistent with training)
        self.pad_token = "<pad>"  # Padding token for sequences
        self.special_tokens = [self.pad_token, "<cls>", "<eoc>"]  # Special tokens in vocabulary
        self.mask_value = "auto"  # Masking value (will be set based on input style)
        self.max_seq_len = 2001  # Maximum sequence length
        self.input_style = "binned"  # Input representation style: "normed_raw", "log1p", or "binned"
        self.output_style = "binned"  # Output representation style
        self.input_emb_style = "continuous"  # Input embedding style: "category", "continuous", or "scaling"
        self.cell_emb_style = "cls"  # Cell embedding style: "avg-pool", "w-pool", or "cls"
        self.fast_transformer_backend = "flash"  # Backend for fast transformer: "linear" or "flash"
        
        # Processing settings - must match training exactly
        if self.input_emb_style == "category":
            self.mask_value = self.n_bins + 1  # Mask value for categorical input
            self.pad_value = self.n_bins  # Pad value for categorical input
            self.n_input_bins = self.n_bins + 2  # Total bins including mask and pad
        else:
            self.mask_value = -1  # Mask value for continuous input
            self.pad_value = -2  # Pad value for continuous input
            self.n_input_bins = self.n_bins  # No additional bins for continuous input

def load_model_and_config(config: InferenceConfig):
    """Load trained model configuration and vocabulary with PEFT settings.
    
    Args:
        config: InferenceConfig object containing paths and settings
        
    Returns:
        tuple: (vocab, model_configs, peft_config)
            - vocab: GeneVocab object for tokenization
            - model_configs: Dictionary of model architecture parameters
            - peft_config: PEFT configuration dictionary
    """
    print(f"Loading model from {config.model_dir}")
    
    # Load model files - check fine-tuned directory first, fallback to pretrained
    model_config_file = config.model_dir / "args.json" if (config.model_dir / "args.json").exists() else Path(config.pretrained_model) / "args.json"
    model_file = config.model_file  # Use the specific fold model file
    vocab_file = config.model_dir / "vocab.json" if (config.model_dir / "vocab.json").exists() else Path(config.pretrained_model) / "vocab.json"
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not vocab_file.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_file}")
    
    # Load vocabulary and ensure special tokens are present
    vocab = GeneVocab.from_file(vocab_file)
    for s in config.special_tokens:
        if s not in vocab:
            vocab.append_token(s)  # Add missing special tokens
    
    # Load model configuration from JSON file
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    
    print(f"Model config loaded: {model_configs}")
    
    # Extract key model architecture parameters
    embsize = model_configs["embsize"]  # Embedding dimension
    nhead = model_configs["nheads"]  # Number of attention heads
    d_hid = model_configs["d_hid"]  # Hidden dimension in feedforward network
    nlayers = model_configs["nlayers"]  # Number of transformer layers
    n_layers_cls = model_configs.get("n_layers_cls", 3)  # Number of classification layers
    
    # Create PEFT configuration matching training setup
    # This must exactly match the configuration used during training
    peft_config = PeftConfig.create_multi_peft_config(
        peft_strategies=config.peft_strategies,  # List of PEFT methods used
        model_nlayers=nlayers,  # Number of transformer layers
        use_default_settings=True  # Use default PEFT hyperparameters
    )
    
    print(f"PEFT strategies: {config.peft_strategies}")
    print(f"PEFT configuration: {peft_config}")
    
    return vocab, model_configs, peft_config

def create_model(config: InferenceConfig, vocab, model_configs, peft_config, num_types):
    """Create model instance with same architecture as training.
    
    Args:
        config: InferenceConfig object with model settings
        vocab: GeneVocab object for tokenization
        model_configs: Dictionary of model architecture parameters
        peft_config: PEFT configuration dictionary
        num_types: Number of cell types for classification head
        
    Returns:
        TransformerModel: Configured model with loaded weights
    """
    # Extract model architecture parameters
    embsize = model_configs["embsize"]  # Embedding dimension
    nhead = model_configs["nheads"]  # Number of attention heads
    d_hid = model_configs["d_hid"]  # Hidden dimension in feedforward network
    nlayers = model_configs["nlayers"]  # Number of transformer layers
    n_layers_cls = model_configs.get("n_layers_cls", 3)  # Number of classification layers
    
    ntokens = len(vocab)  # Size of vocabulary
    
    # Create model with identical settings to training
    model = TransformerModel(
        ntokens,  # Vocabulary size
        embsize,  # Embedding dimension
        nhead,  # Number of attention heads
        d_hid,  # Hidden dimension
        nlayers,  # Number of transformer layers
        nlayers_cls=n_layers_cls,  # Number of classification layers (typically 3)
        n_cls=num_types,  # Number of cell types for classification head
        vocab=vocab,  # Gene vocabulary
        dropout=config.dropout,  # Dropout probability
        pad_token=config.pad_token,  # Padding token
        pad_value=config.pad_value,  # Padding value for gene expression
        do_mvc=False,  # Disable Masked Value Prediction for inference
        do_dab=False,  # Disable Domain Adaptation by reverse Backpropagation
        use_batch_labels=False,  # Don't use batch labels for inference
        num_batch_labels=None,  # No batch labels
        domain_spec_batchnorm=False,  # Disable domain-specific batch normalization
        input_emb_style=config.input_emb_style,  # Input embedding style
        n_input_bins=config.n_input_bins,  # Number of input bins
        cell_emb_style=config.cell_emb_style,  # Cell embedding style
        mvc_decoder_style="inner product",  # MVC decoder style (not used in inference)
        ecs_threshold=0.0,  # Elastic Cell Similarity threshold (disabled)
        explicit_zero_prob=False,  # Don't use explicit zero probability
        use_fast_transformer=config.fast_transformer,  # Fast transformer implementation
        fast_transformer_backend=config.fast_transformer_backend,  # Backend type
        pre_norm=config.pre_norm,  # Pre-normalization in transformer
        peft_config=peft_config  # Parameter-Efficient Fine-Tuning configuration
    )
    
    # Load pretrained weights first (base model)
    if Path(config.pretrained_model).exists():
        pretrained_file = Path(config.pretrained_model) / "best_model.pt"
        if pretrained_file.exists():
            print(f"Loading pretrained weights from {pretrained_file}")
            load_pretrained(model, torch.load(pretrained_file, map_location='cpu'), verbose=False)
    
    # Load fine-tuned weights (PEFT-trained model) for specific fold
    model_state_file = config.model_file
    if model_state_file.exists():
        print(f"Loading fine-tuned weights from {model_state_file}")
        state_dict = torch.load(model_state_file, map_location='cpu')
        model.load_state_dict(state_dict)  # Load the fine-tuned parameters
    
    # Apply PEFT parameter freezing (same as training)
    # Only trainable parameters should be those modified by PEFT
    if config.peft_strategies != [False] and config.peft_strategies:
        freeze_parameters(model, DownstreamTasks.Identification)  # Freeze non-PEFT parameters
    
    # Log parameter counts for verification
    pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
    print(f"Total trainable parameters: {pre_freeze_param_count / 1e6:.2f}M")
    
    # Log learnable parameters (should only be PEFT parameters)
    learnable_params = {k: v for k, v in model.named_parameters() if v.requires_grad}
    print("Learnable parameters:")
    for k, v in learnable_params.items():
        print(f"  {k}: {v.shape}")  # Show parameter names and shapes
    
    return model

def load_and_preprocess_data(config: InferenceConfig, vocab):
    """Load and preprocess test data matching training pipeline.
    
    Args:
        config: InferenceConfig object with data settings
        vocab: GeneVocab object for gene filtering
        
    Returns:
        dict: Dictionary containing processed data and metadata
    """
    print(f"Loading data from {config.data_dir}")
    
    # Load test data (same structure as training)
    if config.dataset_name == "NSCLC":
        # Construct test file name based on fold
        test_file = config.data_dir / f"NSCLC_test{config.fold + 1}.h5ad"
        if not test_file.exists():
            raise FileNotFoundError(f"NSCLC test file not found: {test_file}")
        
        print(f"Using test file: {test_file}")
        
        adata_test = sc.read(test_file)
        adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "2"
        
        # Set index if gene_name column exists
        if "gene_name" in adata_test.var.columns:
            adata_test.var.set_index(adata_test.var["gene_name"], inplace=True)
        else:
            # Use the existing index as gene names
            adata_test.var["gene_name"] = adata_test.var.index.tolist()
        
        # Filter genes that are in vocabulary
        adata_test.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in adata_test.var.index
        ]
        gene_ids_in_vocab = np.array(adata_test.var["id_in_vocab"])
        print(f"Matched {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocabulary")
        adata_test = adata_test[:, adata_test.var["id_in_vocab"] >= 0]
        
        # Set up preprocessing parameters (must match training exactly)
        data_is_raw = False  # Data is already preprocessed
        filter_gene_by_counts = False  # No gene filtering
        
        # Preprocess data using same pipeline as training
        preprocessor = Preprocessor(
            use_key="X",  # Use raw data from adata.X
            filter_gene_by_counts=filter_gene_by_counts,  # Gene filtering step 1
            filter_cell_by_counts=False,  # Cell filtering step 2
            normalize_total=1e4,  # Normalize to 10,000 total counts per cell
            result_normed_key="X_normed",  # Key for normalized data
            log1p=data_is_raw,  # Apply log1p transformation
            result_log1p_key="X_log1p",  # Key for log1p data
            subset_hvg=False,  # Don't subset to highly variable genes
            hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",  # HVG method
            binning=config.n_bins,  # Number of bins for discretization
            result_binned_key="X_binned",  # Key for binned data
        )
        
        preprocessor(adata_test, batch_key=None)
        
        # Extract processed data according to input style
        input_layer_key = {  # Map input style to data layer key
            "normed_raw": "X_normed",  # Normalized raw counts
            "log1p": "X_normed",  # Log1p transformed (uses normed as base)
            "binned": "X_binned",  # Discretized/binned expression values
        }[config.input_style]
        
        all_counts = (
            adata_test.layers[input_layer_key].A
            if issparse(adata_test.layers[input_layer_key])
            else adata_test.layers[input_layer_key]
        )
        
        # Prepare labels and mappings for classification
        celltype_id_labels = adata_test.obs["celltype"].astype("category").cat.codes.values  # Convert to numeric IDs
        celltypes = adata_test.obs["celltype"].unique()  # Unique cell type names
        num_types = len(np.unique(celltype_id_labels))  # Number of cell types
        id2type = dict(enumerate(adata_test.obs["celltype"].astype("category").cat.categories))  # ID to name mapping
        
        batch_ids = adata_test.obs["batch_id"].astype(int).values  # Batch identifiers
        genes = adata_test.var.index.tolist()  # List of gene names
        
        print(f"Test data shape: {all_counts.shape}")
        print(f"Number of cell types: {num_types}")
        print(f"Cell types: {list(id2type.values())}")
        
        return {
            'data': all_counts,
            'celltype_labels': celltype_id_labels,
            'batch_ids': batch_ids,
            'genes': genes,
            'num_types': num_types,
            'id2type': id2type,
            'celltypes': celltypes,
            'adata_test': adata_test
        }
    else:
        raise ValueError(f"Dataset {config.dataset_name} not supported")

def prepare_inference_data(config: InferenceConfig, vocab, data_dict):
    """Prepare data for inference with tokenization.
    
    Args:
        config: InferenceConfig object with processing settings
        vocab: GeneVocab object for tokenization
        data_dict: Dictionary containing processed data from load_and_preprocess_data
        
    Returns:
        dict: Dictionary with tokenized data ready for DataLoader
    """
    # Extract data components
    genes = data_dict['genes']  # List of gene names
    all_counts = data_dict['data']  # Expression matrix
    celltype_labels = data_dict['celltype_labels']  # True cell type labels
    batch_ids = data_dict['batch_ids']  # Batch identifiers
    
    # Get gene IDs from vocabulary (convert gene names to token IDs)
    gene_ids = np.array(vocab(genes), dtype=int)
    
    # Tokenize test data (same process as training)
    tokenized_test = tokenize_and_pad_batch(
        all_counts,  # Expression data matrix
        gene_ids,  # Gene token IDs
        max_len=config.max_seq_len,  # Maximum sequence length
        vocab=vocab,  # Gene vocabulary
        pad_token=config.pad_token,  # Padding token
        pad_value=config.pad_value,  # Padding value for expressions
        append_cls=True,  # Add CLS token at beginning
        include_zero_gene=config.include_zero_gene,  # Include zero-expression genes
    )
    
    # Apply masking (usually 0.0 for inference - no masking needed)
    input_values_test = random_mask_value(
        tokenized_test["values"],  # Tokenized expression values
        mask_ratio=config.mask_ratio,  # Masking ratio (typically 0.0 for inference)
        mask_value=config.mask_value,  # Value used for masked positions
        pad_value=config.pad_value,  # Value used for padding positions
    )
    
    # Create test dataset dictionary with all necessary tensors
    test_data_pt = {
        "gene_ids": tokenized_test["genes"],  # Gene token IDs [N, seq_len]
        "values": input_values_test,  # Expression values (possibly masked) [N, seq_len]
        "target_values": tokenized_test["values"],  # Original expression values [N, seq_len]
        "batch_labels": torch.from_numpy(batch_ids).long(),  # Batch IDs [N]
        "celltype_labels": torch.from_numpy(celltype_labels).long(),  # True cell type labels [N]
    }
    
    print(f"Test dataset prepared:")
    print(f"  Gene IDs shape: {test_data_pt['gene_ids'].shape}")
    print(f"  Values shape: {test_data_pt['values'].shape}")
    print(f"  Celltype labels shape: {test_data_pt['celltype_labels'].shape}")
    
    return test_data_pt

class SeqDataset(Dataset):
    """Dataset class for tokenized sequences.
    
    This dataset class handles the tokenized and processed single-cell data
    for inference, providing batched access to gene IDs, expression values,
    and labels.
    """
    def __init__(self, data: Dict[str, torch.Tensor]):
        """Initialize dataset with tokenized data dictionary."""
        self.data = data

    def __len__(self):
        """Return number of cells in dataset."""
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        """Get a single sample by index."""
        return {k: v[idx] for k, v in self.data.items()}  # Return all data for one cell

def create_dataloader(data_pt: Dict[str, torch.Tensor], batch_size: int, num_workers: int = 0):
    """Create DataLoader for inference.
    
    Args:
        data_pt: Dictionary with tokenized data tensors
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        
    Returns:
        DataLoader: Configured DataLoader for inference
    """
    # Auto-configure number of workers based on available CPUs
    if num_workers == 0:
        num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)
    
    dataset = SeqDataset(data_pt)  # Create dataset from tokenized data
    
    dataloader = DataLoader(
        dataset=dataset,  # Dataset containing tokenized sequences
        batch_size=batch_size,  # Batch size for inference
        shuffle=False,  # No shuffling for inference (preserve order)
        drop_last=False,  # Keep all samples, don't drop incomplete batches
        num_workers=num_workers,  # Number of data loading workers
        pin_memory=True,  # Pin memory for faster GPU transfer
    )
    
    return dataloader

def run_inference(model, dataloader, config: InferenceConfig, vocab):
    """Run inference on test data and return predictions.
    
    Args:
        model: Fine-tuned TransformerModel
        dataloader: DataLoader with test data
        config: InferenceConfig with device and settings
        vocab: GeneVocab for tokenization
        
    Returns:
        dict: Dictionary with predictions, probabilities, labels, and embeddings
    """
    print("Running inference...")
    
    model.eval()  # Set model to evaluation mode
    model.to(config.device)  # Move model to specified device (GPU/CPU)
    
    # Initialize lists to collect results from all batches
    all_predictions = []  # Predicted cell type IDs
    all_probabilities = []  # Class probabilities for each prediction
    all_labels = []  # True cell type labels
    all_embeddings = []  # Cell embeddings (if available)
    
    with torch.no_grad():  # Disable gradient computation for inference
        for batch_idx, batch_data in enumerate(dataloader):
            # Move all input data to the specified device
            input_gene_ids = batch_data["gene_ids"].to(config.device)  # Gene token IDs
            input_values = batch_data["values"].to(config.device)  # Expression values
            batch_labels = batch_data["batch_labels"].to(config.device)  # Batch identifiers
            celltype_labels = batch_data["celltype_labels"].to(config.device)  # True labels
            
            # Create padding mask for attention mechanism
            src_key_padding_mask = input_gene_ids.eq(vocab[config.pad_token])  # Mask padded positions
            
            # Forward pass with Automatic Mixed Precision if enabled
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,  # Gene token IDs [batch, seq_len]
                    input_values,  # Expression values [batch, seq_len]
                    src_key_padding_mask=src_key_padding_mask,  # Padding mask [batch, seq_len]
                    batch_labels=None,  # Don't use batch labels for inference
                    CLS=True,  # Enable classification output (essential for inference)
                    CCE=False,  # Disable contrastive cell embedding
                    MVC=False,  # Disable masked value prediction
                    ECS=False,  # Disable elastic cell similarity
                    do_sample=False,  # No sampling during inference
                )
            
            # Extract predictions and probabilities from classification output
            cls_output = output_dict["cls_output"]  # Raw logits [batch_size, num_classes]
            probabilities = torch.softmax(cls_output, dim=1)  # Convert to probabilities
            predictions = cls_output.argmax(dim=1)  # Get predicted class IDs
            
            # Store batch results (move to CPU and convert to numpy)
            all_predictions.append(predictions.cpu().numpy())  # Predicted labels
            all_probabilities.append(probabilities.cpu().numpy())  # Class probabilities
            all_labels.append(celltype_labels.cpu().numpy())  # True labels
            
            # Store cell embeddings if available (for downstream analysis)
            if "cell_emb" in output_dict:
                all_embeddings.append(output_dict["cell_emb"].cpu().numpy())  # Cell representations
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(dataloader)}")
    
    # Concatenate results from all batches
    predictions = np.concatenate(all_predictions, axis=0)  # All predicted labels
    probabilities = np.concatenate(all_probabilities, axis=0)  # All class probabilities
    labels = np.concatenate(all_labels, axis=0)  # All true labels
    
    # Concatenate embeddings if they were collected
    if all_embeddings:
        embeddings = np.concatenate(all_embeddings, axis=0)  # All cell embeddings
    else:
        embeddings = None  # No embeddings available
    
    print(f"Inference completed. Processed {len(predictions)} samples")
    
    return {
        'predictions': predictions,
        'probabilities': probabilities,
        'labels': labels,
        'embeddings': embeddings
    }

def calculate_metrics(predictions, labels, probabilities=None):
    """Calculate comprehensive evaluation metrics.
    
    Args:
        predictions: Predicted cell type IDs
        labels: True cell type labels
        probabilities: Class probabilities (optional)
        
    Returns:
        dict: Dictionary with all calculated metrics
    """
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, precision_score, 
        recall_score, f1_score, confusion_matrix, classification_report
    )
    
    # Calculate basic classification metrics
    accuracy = accuracy_score(labels, predictions)  # Overall accuracy
    balanced_accuracy = balanced_accuracy_score(labels, predictions)  # Balanced accuracy (accounts for class imbalance)
    macro_precision = precision_score(labels, predictions, average='macro', zero_division=0)  # Macro-averaged precision
    macro_recall = recall_score(labels, predictions, average='macro', zero_division=0)  # Macro-averaged recall
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)  # Macro-averaged F1-score
    
    # Calculate weighted metrics (weighted by support)
    weighted_precision = precision_score(labels, predictions, average='weighted', zero_division=0)  # Weighted precision
    weighted_recall = recall_score(labels, predictions, average='weighted', zero_division=0)  # Weighted recall
    weighted_f1 = f1_score(labels, predictions, average='weighted', zero_division=0)  # Weighted F1-score
    
    # Calculate per-class metrics (for detailed analysis)
    per_class_precision = precision_score(labels, predictions, average=None, zero_division=0)  # Precision for each class
    per_class_recall = recall_score(labels, predictions, average=None, zero_division=0)  # Recall for each class
    per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)  # F1-score for each class
    
    # Generate confusion matrix for visualization
    cm = confusion_matrix(labels, predictions)  # Confusion matrix
    
    # Generate detailed classification report
    class_report = classification_report(labels, predictions, output_dict=True, zero_division=0)  # Detailed per-class metrics
    
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

def print_results(metrics, id2type):
    """Print detailed evaluation results.
    
    Args:
        metrics: Dictionary with calculated metrics
        id2type: Mapping from class IDs to cell type names
    """
    print("\n" + "="*60)
    print("INFERENCE RESULTS")
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
    """Create and save confusion matrix visualization.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names for labels
        save_path: Path to save the plot
    """
    # Normalize confusion matrix (row-wise normalization)
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap visualization of confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized,  # Normalized confusion matrix
        annot=True,  # Show values in cells
        fmt='.2f',  # Format numbers to 2 decimal places
        cmap='Blues',  # Color scheme
        xticklabels=class_names,  # X-axis labels (predicted)
        yticklabels=class_names,  # Y-axis labels (true)
        cbar_kws={'label': 'Normalized Count'}  # Colorbar label
    )
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Cell Type', fontsize=12)
    plt.ylabel('True Cell Type', fontsize=12)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
    plt.yticks(rotation=0)  # Keep y-axis labels horizontal
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save high-resolution image
    plt.close()  # Close figure to free memory
    
    print(f"Confusion matrix saved to {save_path}")

def create_metrics_barplot(metrics, id2type, save_path):
    """Create bar plot of per-class metrics.
    
    Args:
        metrics: Dictionary with calculated metrics
        id2type: Mapping from class IDs to cell type names
        save_path: Path to save the plot
    """
    # Extract class names in order
    class_names = [id2type[i] for i in range(len(metrics['per_class_f1']))]
    
    # Create subplots for three metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Precision bar plot
    axes[0].bar(class_names, metrics['per_class_precision'], color='skyblue', alpha=0.8)
    axes[0].set_title('Per-class Precision', fontweight='bold')
    axes[0].set_ylabel('Precision')
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].set_ylim(0, 1)  # Set y-axis limits
    
    # Recall bar plot
    axes[1].bar(class_names, metrics['per_class_recall'], color='lightcoral', alpha=0.8)
    axes[1].set_title('Per-class Recall', fontweight='bold')
    axes[1].set_ylabel('Recall')
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].set_ylim(0, 1)  # Set y-axis limits
    
    # F1-score bar plot
    axes[2].bar(class_names, metrics['per_class_f1'], color='lightgreen', alpha=0.8)
    axes[2].set_title('Per-class F1-score', fontweight='bold')
    axes[2].set_ylabel('F1-score')
    axes[2].set_xticklabels(class_names, rotation=45, ha='right')
    axes[2].set_ylim(0, 1)  # Set y-axis limits
    
    plt.tight_layout()  # Adjust subplot spacing
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save high-resolution image
    plt.close()  # Close figure to free memory
    
    print(f"Metrics bar plot saved to {save_path}")

def save_results(inference_results, metrics, config: InferenceConfig, data_dict):
    """Save all inference results and visualizations.
    
    Args:
        inference_results: Dictionary with predictions and probabilities
        metrics: Dictionary with calculated metrics
        config: InferenceConfig with output settings
        data_dict: Dictionary with data and metadata
    """
    print(f"\nSaving results to {config.output_dir}")
    
    # Save predictions and metrics
    results_dict = {
        'predictions': inference_results['predictions'],
        'probabilities': inference_results['probabilities'],
        'labels': inference_results['labels'],
        'metrics': metrics,
        'id2type': data_dict['id2type'],
        'config': {
            'model_dir': str(config.model_dir),
            'data_dir': str(config.data_dir),
            'peft_strategies': config.peft_strategies,
            'batch_size': config.batch_size,
            'mask_ratio': config.mask_ratio,
            'device': str(config.device)
        }
    }
    
    # Save embeddings if available
    if inference_results['embeddings'] is not None:
        results_dict['embeddings'] = inference_results['embeddings']
    
    # Save as pickle
    with open(config.output_dir / 'inference_results.pkl', 'wb') as f:
        pickle.dump(results_dict, f)
    print(f"Results saved to {config.output_dir / 'inference_results.pkl'}")
    
    # Save metrics as JSON (convert numpy arrays to lists)
    metrics_json = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_json[key] = value.tolist()
        elif key == 'classification_report':
            metrics_json[key] = value  # Already a dict
        else:
            metrics_json[key] = float(value) if isinstance(value, (np.float32, np.float64)) else value
    
    with open(config.output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Metrics saved to {config.output_dir / 'metrics.json'}")
    
    # Create visualizations
    class_names = [data_dict['id2type'][i] for i in range(len(data_dict['id2type']))]
    
    # Confusion matrix
    create_confusion_matrix_plot(
        metrics['confusion_matrix'], 
        class_names, 
        config.output_dir / 'confusion_matrix.png'
    )
    
    # Per-class metrics
    create_metrics_barplot(
        metrics, 
        data_dict['id2type'], 
        config.output_dir / 'per_class_metrics.png'
    )
    
    # Save detailed results as CSV
    results_df = pd.DataFrame({
        'True_Label': inference_results['labels'],
        'Predicted_Label': inference_results['predictions'],
        'True_CellType': [data_dict['id2type'][label] for label in inference_results['labels']],
        'Predicted_CellType': [data_dict['id2type'][pred] for pred in inference_results['predictions']],
        'Max_Probability': np.max(inference_results['probabilities'], axis=1),
        'Correct': inference_results['labels'] == inference_results['predictions']
    })
    
    # Add probability columns for each class
    for i, class_name in enumerate(class_names):
        results_df[f'Prob_{class_name}'] = inference_results['probabilities'][:, i]
    
    results_df.to_csv(config.output_dir / 'detailed_results.csv', index=False)
    print(f"Detailed results saved to {config.output_dir / 'detailed_results.csv'}")
    
    print(f"All results saved to {config.output_dir}")

def main():
    """Main inference pipeline.
    
    This function orchestrates the entire inference process:
    1. Parse command line arguments
    2. Load model and configuration
    3. Load and preprocess data
    4. Run inference
    5. Calculate metrics and save results
    
    Returns:
        tuple: (inference_results, metrics)
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Create configuration object from parsed arguments
    config = InferenceConfig(args)
    
    # Set random seed for reproducibility
    set_seed(config.seed)  # Ensures consistent results across runs
    
    # Print configuration summary
    print(f"Starting inference with configuration:")
    print(f"  PEFT method: {config.peft_method}")  # PEFT method being used
    print(f"  Fold: {config.fold}")  # Current fold
    print(f"  Model directory: {config.model_dir}")  # Path to trained model
    print(f"  Model file: {config.model_file}")  # Specific model file
    print(f"  Data directory: {config.data_dir}")  # Path to test data
    print(f"  PEFT strategies: {config.peft_strategies}")  # PEFT methods used
    print(f"  Device: {config.device}")  # Computation device (GPU/CPU)
    print(f"  Batch size: {config.batch_size}")  # Inference batch size
    print(f"  Output directory: {config.output_dir}")  # Results save location
    
    # Load model configuration and vocabulary
    vocab, model_configs, peft_config = load_model_and_config(config)
    
    # Load and preprocess test data (same pipeline as training)
    data_dict = load_and_preprocess_data(config, vocab)
    
    # Create model with loaded weights and PEFT configuration
    model = create_model(config, vocab, model_configs, peft_config, data_dict['num_types'])
    
    # Prepare tokenized data for inference
    test_data_pt = prepare_inference_data(config, vocab, data_dict)
    test_dataloader = create_dataloader(test_data_pt, config.batch_size)
    
    # Run inference on all test samples
    inference_results = run_inference(model, test_dataloader, config, vocab)
    
    # Calculate comprehensive evaluation metrics
    metrics = calculate_metrics(
        inference_results['predictions'],  # Predicted cell type IDs
        inference_results['labels'],  # True cell type labels
        inference_results['probabilities']  # Class probabilities
    )
    
    # Display results in terminal
    print_results(metrics, data_dict['id2type'])
    
    # Save all results and visualizations if requested
    if args.save_results:
        save_results(inference_results, metrics, config, data_dict)
    
    print("\nInference completed successfully!")
    return inference_results, metrics  # Return results for further analysis

# Execute main function when script is run directly
if __name__ == "__main__":
    inference_results, metrics = main()  # Run the complete inference pipeline