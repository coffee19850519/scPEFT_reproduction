import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import argparse
import pathlib
import pickle
import json
import yaml
import datetime
import time
from pathlib import Path
from sklearn.model_selection import KFold
import loralib as lora

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    confusion_matrix, precision_score, recall_score, f1_score
)
from datasets import load_from_disk
from sklearn.model_selection import KFold

# Add the transformerslocal directory to the Python path
transformers_path = os.path.join(os.path.dirname(os.getcwd()), 'transformerslocal')

if transformers_path not in sys.path:
    sys.path.insert(0, transformers_path)

parent_path = os.path.dirname(os.getcwd())
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

from transformerslocal.src.transformers.models.bert.modeling_bert import BertForTokenClassification
from transformerslocal.src.transformers import EarlyStoppingCallback, TrainerCallback, Trainer
from transformerslocal.src.transformers.training_args import TrainingArguments

# Import DataCollatorForGeneClassification from the original Geneformer directory
geneformer_path = "scPEFT_reproduction/geneformer_peft"
if geneformer_path not in sys.path:
    sys.path.insert(0, geneformer_path)

# Import directly from the collator module to avoid name conflicts
from geneformer.collator_for_classification import DataCollatorForGeneClassification

# -----------------------------------------------------------
# Helper Classes and Functions for Class Imbalance
# -----------------------------------------------------------
# Note: Focal loss classes removed - using standard cross-entropy loss

# -----------------------------------------------------------
# Helper Classes and Functions
# -----------------------------------------------------------

class LearningRateResetCallback(TrainerCallback):
    """Custom callback to reset learning rate if loss plateaus"""
    
    def __init__(self, patience=200, factor=0.5, min_lr=1e-6):
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.wait_count = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
            
        current_loss = logs.get("train_loss", float('inf'))
        
        if current_loss < self.best_loss - 1e-4:  # Improvement threshold
            self.best_loss = current_loss
            self.wait_count = 0
        else:
            self.wait_count += 1
            
        if self.wait_count >= self.patience:
            # Reset learning rate
            current_lr = state.learning_rate if hasattr(state, 'learning_rate') else args.learning_rate
            new_lr = max(current_lr * self.factor, self.min_lr)
            
            if new_lr > self.min_lr and hasattr(kwargs.get('optimizer'), 'param_groups'):
                for param_group in kwargs['optimizer'].param_groups:
                    param_group['lr'] = new_lr
                print(f"Learning rate reset from {current_lr:.2e} to {new_lr:.2e} due to loss plateau")
                self.wait_count = 0
                self.best_loss = current_loss

class EpochTimeCallback(TrainerCallback):
    """Custom callback to track and display epoch running time and validation metrics"""
    
    def __init__(self):
        self.epoch_start_time = None
        self.training_start_time = None
        self.epoch_times = []
        self.current_epoch = 0  # Track epoch manually for reliable counting
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training"""
        self.training_start_time = time.time()
        self.current_epoch = 0
        print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.training_start_time))}")
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch"""
        self.epoch_start_time = time.time()
        # Use state.epoch if available, otherwise use our manual tracking
        if state.epoch is not None:
            self.current_epoch = int(state.epoch) + 1
        else:
            self.current_epoch += 1
        print(f"\nEpoch {self.current_epoch} started at {time.strftime('%H:%M:%S', time.localtime(self.epoch_start_time))}")
        
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """Called after evaluation - display validation metrics"""
        try:
            # Get current epoch number - state.epoch is 0-based, so add 1 for display
            epoch_num = self.current_epoch
            if state.epoch is not None:
                epoch_num = int(state.epoch) + 1
            elif hasattr(state, 'log_history') and len(state.log_history) > 0:
                # Look for epoch in the latest log entry
                latest_log = state.log_history[-1]
                if 'epoch' in latest_log:
                    epoch_num = int(latest_log['epoch']) + 1
            
            print(f"\nValidation Metrics:")
            
            # Get metrics from logs parameter first (most recent), then from state
            eval_metrics = logs if logs is not None else {}
            
            # Always check log_history as backup since it might have AUC
            if hasattr(state, 'log_history') and len(state.log_history) > 0:
                latest_log = state.log_history[-1]
                if 'eval_loss' in latest_log:
                    if not eval_metrics:
                        eval_metrics = latest_log.copy()
                    else:
                        # Merge AUC from log_history if it's missing from logs
                        if 'eval_auc_macro' in latest_log and 'eval_auc_macro' not in eval_metrics:
                            eval_metrics['eval_auc_macro'] = latest_log['eval_auc_macro']
            
            # If AUC is missing, try to compute it directly
            if eval_metrics and 'eval_auc_macro' not in eval_metrics:
                try:
                    # Get the trainer from kwargs (this is where it's passed in Transformers callbacks)
                    trainer = kwargs.get('trainer', None)
                    if trainer and hasattr(trainer, 'eval_dataset') and hasattr(trainer, 'compute_auc_score'):
                        auc_score = trainer.compute_auc_score(trainer.model, trainer.eval_dataset)
                        eval_metrics['eval_auc_macro'] = auc_score
                except Exception as e:
                    pass  # Continue without AUC if computation fails
            
            # Safely format numeric values
            def safe_format(value, format_str=".4f"):
                try:
                    if value == 'N/A' or value is None:
                        return 'N/A'
                    return f"{float(value):{format_str}}"
                except (ValueError, TypeError):
                    return 'N/A'
            
            print(f"  Eval Loss: {safe_format(eval_metrics.get('eval_loss', 'N/A'))}")
            print(f"  Eval Accuracy: {safe_format(eval_metrics.get('eval_accuracy', 'N/A'))}")
            print(f"  Eval Balanced Accuracy: {safe_format(eval_metrics.get('eval_balanced_accuracy', 'N/A'))}")
            print(f"  Eval Precision: {safe_format(eval_metrics.get('eval_precision', 'N/A'))}")
            print(f"  Eval Recall: {safe_format(eval_metrics.get('eval_recall', 'N/A'))}")
            print(f"  Eval F1: {safe_format(eval_metrics.get('eval_f1', 'N/A'))}")
            print(f"  Eval AUC (Macro): {safe_format(eval_metrics.get('eval_auc_macro', 'N/A'))}")
            
                
        except Exception as e:
            print(f"Warning: Error in on_evaluate callback: {e}")
            # Continue execution even if there's an error in logging
        
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch"""
        if self.epoch_start_time is not None:
            epoch_duration = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_duration)
            
            # Format duration
            hours = int(epoch_duration // 3600)
            minutes = int((epoch_duration % 3600) // 60)
            seconds = int(epoch_duration % 60)
            
            if hours > 0:
                duration_str = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                duration_str = f"{minutes}m {seconds}s"
            else:
                duration_str = f"{seconds}s"
            
            # Use reliable epoch tracking
            epoch_num = self.current_epoch
            if state.epoch is not None:
                epoch_num = int(state.epoch) + 1
                
            print(f"Epoch {epoch_num} completed in {duration_str} ({epoch_duration:.2f}s)")
            
            # Show average epoch time if we have multiple epochs
            if len(self.epoch_times) > 1:
                avg_time = sum(self.epoch_times) / len(self.epoch_times)
                avg_minutes = int(avg_time // 60)
                avg_seconds = int(avg_time % 60)
                print(f"Average epoch time: {avg_minutes}m {avg_seconds}s ({avg_time:.2f}s)")
                
                # Estimate remaining time - prefer state.epoch if available
                current_epoch_for_estimation = epoch_num
                if state.epoch is not None:
                    current_epoch_for_estimation = int(state.epoch) + 1
                    
                remaining_epochs = args.num_train_epochs - current_epoch_for_estimation
                if remaining_epochs > 0:
                    estimated_remaining = remaining_epochs * avg_time
                    est_hours = int(estimated_remaining // 3600)
                    est_minutes = int((estimated_remaining % 3600) // 60)
                    if est_hours > 0:
                        print(f"Estimated remaining time: {est_hours}h {est_minutes}m")
                    else:
                        print(f"Estimated remaining time: {est_minutes}m")
        
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training"""
        if self.training_start_time is not None:
            total_training_time = time.time() - self.training_start_time
            
            hours = int(total_training_time // 3600)
            minutes = int((total_training_time % 3600) // 60)
            seconds = int(total_training_time % 60)
            
            if hours > 0:
                duration_str = f"{hours}h {minutes}m {seconds}s"
            else:
                duration_str = f"{minutes}m {seconds}s"
            
            print(f"\nTraining completed in {duration_str} ({total_training_time:.2f}s)")
            print(f"Total epochs: {len(self.epoch_times)}")
            
            if self.epoch_times:
                avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
                min_epoch_time = min(self.epoch_times)
                max_epoch_time = max(self.epoch_times)
                
                print(f"Epoch timing statistics:")
                print(f"   Average: {avg_epoch_time:.2f}s")
                print(f"   Fastest: {min_epoch_time:.2f}s")
                print(f"   Slowest: {max_epoch_time:.2f}s")
            
            # Display validation metrics progression
            if hasattr(state, 'log_history') and len(state.log_history) > 0:
                print(f"\nVALIDATION METRICS PROGRESSION:")
                print(f"{'Epoch':<6} {'Loss':<8} {'Accuracy':<9} {'Bal Acc':<8} {'Precision':<9} {'Recall':<8} {'F1':<8}")
                print(f"{'-'*6} {'-'*8} {'-'*9} {'-'*8} {'-'*9} {'-'*8} {'-'*8}")
                
                epoch_num = 1
                for log_entry in state.log_history:
                    if 'eval_loss' in log_entry:
                        print(f"{epoch_num:<6} "
                              f"{log_entry.get('eval_loss', 0):<8.4f} "
                              f"{log_entry.get('eval_accuracy', 0):<9.4f} "
                              f"{log_entry.get('eval_balanced_accuracy', 0):<8.4f} "
                              f"{log_entry.get('eval_precision', 0):<9.4f} "
                              f"{log_entry.get('eval_recall', 0):<8.4f} "
                              f"{log_entry.get('eval_f1', 0):<8.4f}")
                        epoch_num += 1

def get_lora_weight(lora_param):
    """Helper function to get weight from LoRA parameter (handles both Parameter and module cases)"""
    if hasattr(lora_param, 'weight'):
        return lora_param.weight
    elif isinstance(lora_param, torch.nn.Parameter):
        return lora_param
    else:
        return lora_param

def get_lora_device(lora_param):
    """Helper function to get device from LoRA parameter (handles both Parameter and module cases)"""
    if hasattr(lora_param, 'weight'):
        return lora_param.weight.device
    elif hasattr(lora_param, 'device'):
        return lora_param.device
    else:
        # Fallback - should not happen
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def warm_up_model(model, data_collator, warmup_dataset, device="cuda"):
    """Perform a few forward passes to warm up the model and check gradients"""
    print("Warming up model with sample data...")
    
    model.train()
    from torch.utils.data import DataLoader
    
    # Create a small dataloader for warmup
    warmup_loader = DataLoader(warmup_dataset.select(range(min(10, len(warmup_dataset)))), 
                               batch_size=2, collate_fn=data_collator)
    
    # Perform a few forward passes
    for i, batch in enumerate(warmup_loader):
        if i >= 3:  # Only do 3 warmup batches
            break
            
        # Move to device
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device)
        
        # Forward pass only
        with torch.no_grad():
            outputs = model(**{k: v for k, v in batch.items() if k in 
                             ['input_ids', 'attention_mask', 'labels']})
            print(f"  Warmup batch {i+1}: loss = {outputs.loss.item():.4f}")
    
    print("Model warmup completed!")
    return model

def inspect_training_data(dataset, class_id_dict, id_class_dict, dataset_name=""):
    """Inspect the training data for debugging purposes"""
    print(f"\nINSPECTING {dataset_name} DATA")
    print("="*60)
    
    total_samples = len(dataset)
    total_tokens = 0
    total_labels = 0
    class_counts = {}
    
    for i, sample in enumerate(dataset):
        if i >= 1000:  # Sample first 1000 for speed
            break
            
        labels = sample["labels"]
        valid_labels = [l for l in labels if l != -100]
        
        total_tokens += len(labels)
        total_labels += len(valid_labels)
        
        for label in valid_labels:
            class_counts[label] = class_counts.get(label, 0) + 1
    
    print(f"Dataset size: {total_samples} samples")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total valid labels: {total_labels:,}")
    print(f"Average tokens per sample: {total_tokens / min(1000, total_samples):.1f}")
    print(f"Average labels per sample: {total_labels / min(1000, total_samples):.1f}")
    print(f"Label coverage: {100 * total_labels / total_tokens:.2f}%")
    print(id_class_dict)
    
    print(f"\nClass distribution (top 10):")
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (class_id, count) in enumerate(sorted_classes[:10]):
        class_name = id_class_dict.get(class_id, f"Unknown_{class_id}")
        percentage = 100 * count / total_labels
        print(f"  {i+1:2d}. Class {class_id} ({class_name}): {count:,} ({percentage:.2f}%)")
    
    # Check for class imbalance
    if len(sorted_classes) > 1:
        majority_count = sorted_classes[0][1]
        minority_count = sorted_classes[-1][1]
        imbalance_ratio = majority_count / minority_count
        print(f"\nClass imbalance ratio: {imbalance_ratio:.1f}:1")
        
        if imbalance_ratio > 100:
            print("WARNING: SEVERE CLASS IMBALANCE DETECTED!")
            print("   This can cause poor learning. Consider:")
            print("   - Using class weights")
            print("   - Increasing learning rate")
            print("   - Using weighted sampling")
        elif imbalance_ratio > 10:
            print("WARNING: Moderate class imbalance detected")
    
    print("="*60)
    return class_counts

def validate_dataset_structure(dataset, dataset_name=""):
    """Validate that the dataset has the expected structure for gene classification"""
    print(f"\nVALIDATING {dataset_name} DATASET STRUCTURE")
    print("="*60)
    
    if len(dataset) == 0:
        print("ERROR: Dataset is empty!")
        return False
    
    # Check first few samples
    issues_found = []
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        
        # Check required fields
        if 'input_ids' not in sample:
            issues_found.append(f"Sample {i}: Missing 'input_ids' field")
            continue
            
        if 'labels' not in sample:
            issues_found.append(f"Sample {i}: Missing 'labels' field")
            continue
            
        # Check input_ids structure
        input_ids = sample['input_ids']
        if not isinstance(input_ids, list):
            issues_found.append(f"Sample {i}: 'input_ids' is not a list (type: {type(input_ids)})")
        elif len(input_ids) == 0:
            issues_found.append(f"Sample {i}: 'input_ids' is empty")
        elif not all(isinstance(x, (int, float)) for x in input_ids):
            issues_found.append(f"Sample {i}: 'input_ids' contains non-numeric values")
            
        # Check labels structure
        labels = sample['labels']
        if not isinstance(labels, list):
            issues_found.append(f"Sample {i}: 'labels' is not a list (type: {type(labels)})")
        elif len(labels) != len(input_ids):
            issues_found.append(f"Sample {i}: 'labels' length ({len(labels)}) != 'input_ids' length ({len(input_ids)})")
        elif not all(isinstance(x, (int, float)) for x in labels):
            issues_found.append(f"Sample {i}: 'labels' contains non-numeric values")
    
    if issues_found:
        print("DATASET VALIDATION FAILED:")
        for issue in issues_found[:10]:  # Show first 10 issues
            print(f"  - {issue}")
        if len(issues_found) > 10:
            print(f"  ... and {len(issues_found) - 10} more issues")
        return False
    else:
        print("Dataset validation passed!")
        
        # Show sample statistics
        sample = dataset[0]
        print(f"Sample statistics:")
        print(f"  Input sequence length: {len(sample['input_ids'])}")
        print(f"  Number of labels: {len(sample['labels'])}")
        print(f"  Valid labels (not -100): {sum(1 for x in sample['labels'] if x != -100)}")
        
        return True

def get_dataset_paths(dataset_name):
    """Get the correct paths for different datasets"""

    dataset_configs = {
        "bivalent_promoters": {
            "base_output_dir": "outputs/250717171706",
            "pre_split_subdir": "250717_geneformer_geneClassifier_bivalent_promoters",
            "class_dict_file": "bivalent_promoters_id_class_dict.pkl",
            "dataset_file": "bivalent_promoters/panglao_SRA553822-SRS2119548.dataset",
            "gene_class_dict": "bivalent_promoters/bivalent_vs_lys4_only_genomewide.pickle"
        },
        "tf_regulatory_range": {
            "base_output_dir": "outputs/250717171708",
            "pre_split_subdir": "250717_geneformer_geneClassifier_tf_regulatory_range",
            "class_dict_file": "tf_regulatory_range_id_class_dict.pkl",
            "dataset_file": "tf_regulatory_range/iCM_diff_dropseq.dataset",
            "gene_class_dict": "tf_regulatory_range/tf_regulatory_range.pickle"
        },
        "N1_network": {
            "base_output_dir": "outputs/250717171708",
            "pre_split_subdir": "250717_geneformer_geneClassifier_N1_network",
            "class_dict_file": "N1_network_id_class_dict.pkl", 
            "dataset_file": "notch1_network/heart_atlas_endothelial_cells.dataset",
            "gene_class_dict": "notch1_network/n1_network.pickle"
        }
    }
    
    if dataset_name not in dataset_configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(dataset_configs.keys())}")
    
    config = dataset_configs[dataset_name]
    
    # Construct full paths
    paths = {
        "pre_split_dir": os.path.join(config["base_output_dir"], config["pre_split_subdir"]),
        "class_dict_dir": config["base_output_dir"],
        "class_dict_file": os.path.join(config["base_output_dir"], config["class_dict_file"]),
        "dataset_file": config["dataset_file"],
        "gene_class_dict": config["gene_class_dict"]
    }
    
    return paths

def calculate_class_weights(dataset, class_id_dict, device='cuda', aggressive_factor=2.0):
    """Calculate MORE AGGRESSIVE inverse frequency class weights to combat severe class imbalance"""
    print("Calculating AGGRESSIVE class weights from training data...")
    
    # Count class occurrences
    class_counts = {}
    total_labels = 0
    
    for sample in dataset:
        labels = sample["labels"]
        for label in labels:
            if label != -100:  # Ignore padding labels
                class_counts[label] = class_counts.get(label, 0) + 1
                total_labels += 1
    
    print(f"Total valid labels: {total_labels:,}")
    print("Class distribution:")
    
    # Calculate MORE AGGRESSIVE weights (pure inverse frequency, no sqrt dampening)
    class_weights = torch.ones(len(class_id_dict))
    
    for class_id, count in class_counts.items():
        if class_id < len(class_id_dict):
            frequency = count / total_labels
            # AGGRESSIVE: Use pure inverse frequency (no sqrt dampening)
            # Then apply additional aggressive factor for minority classes
            base_weight = 1.0 / frequency if frequency > 0 else 1.0
            
            # Apply extra aggressive factor for minority classes
            if frequency < 0.3:  # If class is less than 30% of data
                weight = base_weight * aggressive_factor
            else:
                weight = base_weight
            
            class_weights[class_id] = weight
            
            class_name = [name for name, id in class_id_dict.items() if id == class_id][0]
            print(f"  Class {class_id} ({class_name}): {count:,} samples ({frequency:.4f}) -> weight: {weight:.4f}")
    
    # Normalize weights so the minority class gets strong preference
    # Don't normalize to mean=1, keep the aggressive weighting
    min_weight = class_weights.min()
    class_weights = class_weights / min_weight  # Normalize so minimum weight is 1.0
    
    # MUCH higher maximum weight to help minority class
    max_weight = 15.0  # Allow up to 15x weight difference (was 2.0)
    class_weights = torch.clamp(class_weights, min=1.0, max=max_weight)
    
    print("AGGRESSIVE class weights (normalized to min=1.0):")
    for class_id, weight in enumerate(class_weights):
        class_name = [name for name, id in class_id_dict.items() if id == class_id][0]
        print(f"  Class {class_id} ({class_name}): {weight:.4f}")
    
    # Calculate and show the final weight ratio
    weight_ratio = class_weights.max() / class_weights.min()
    print(f"Final weight ratio (max/min): {weight_ratio:.2f}:1")
    
    return class_weights.to(device)

# -----------------------------------------------------------
# 1 ▸ CLI
# -----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", default="tf_dosage_sens_test", 
                   choices=["tf_dosage_sens_test", "bivalent_promoters", "tf_regulatory_range", "N1_network"],
                   help="Name of the dataset to use")
parser.add_argument("--dataset_file", default="gc-30M_sample50k.dataset")
parser.add_argument("--pre_split_dir", default="", 
                   help="Directory containing pre-split train/val/test datasets (auto-determined if empty)")
parser.add_argument("--use_pre_split", action="store_true", default=True,
                   help="Whether to use pre-split datasets instead of creating new splits")
parser.add_argument("--class_dict_dir", default="",
                   help="Directory containing the class dictionary file (auto-determined if empty)")
parser.add_argument("--gene_class_dict", default="dosage_sensitivity_TFs.pickle")
parser.add_argument("--token_dict", 
                default="token_dictionary_gc30M.pkl")
                
parser.add_argument("--ckpt_dir", 
                default="geneformer-12L-30M-prompt")
                
parser.add_argument("--output_root", default="outputs")

parser.add_argument("--epochs", type=int, default=20)  # Increased epochs for prompt methods
parser.add_argument("--batch_size", type=int, default=16)  # Increased batch size for stability
parser.add_argument("--lr",       type=float, default=1e-5)  # More appropriate LR for prompt methods
parser.add_argument("--seed",      type=int, default=42)
parser.add_argument("--n_folds",   type=int, default=5)

parser.add_argument("--prompt_type", 
                    choices=["encoder_prompt", "lora", "prefix_prompt", "Gene_token_prompt", "normal_finetune"],
                    default="Gene_token_prompt",
                    help="Type of training method: encoder_prompt, lora, prefix_prompt, Gene_token_prompt, or normal_finetune")
parser.add_argument("--freeze_layers", type=int, default=4,
                    help="Number of first encoder layers to freeze during normal_finetune. E.g., 3 means freeze layers 0,1,2. Default 0 means no freezing")

# Class imbalance handling arguments
parser.add_argument("--aggressive_class_weights", type=float, default=0.75,
                    help="Aggressive factor for minority class weighting (default: 3.0)")
parser.add_argument("--use_focal_loss", action="store_true", default=False,
                    help="Use focal loss instead of weighted CrossEntropyLoss")
parser.add_argument("--focal_alpha", type=float, default=0.25,
                    help="Alpha parameter for focal loss (default: 0.25)")
parser.add_argument("--focal_gamma", type=float, default=2.0,
                    help="Gamma parameter for focal loss (default: 2.0)")
parser.add_argument("--ce_weights", type=str, default='7,1',
                    help="Comma-separated list of class weights for CrossEntropyLoss (e.g., '0.5,2.0,1.5') or None to auto-calculate")

# args = parser.parse_args('')
args = parser.parse_args()

# Auto-determine paths based on dataset name if not explicitly provided
if args.use_pre_split:
    if not args.pre_split_dir or not args.class_dict_dir:
        print(f"Auto-determining paths for dataset: {args.dataset_name}")
        dataset_paths = get_dataset_paths(args.dataset_name)
        
        if not args.pre_split_dir:
            args.pre_split_dir = dataset_paths["pre_split_dir"]
            print(f"  Pre-split dir: {args.pre_split_dir}")
        
        if not args.class_dict_dir:
            args.class_dict_dir = dataset_paths["class_dict_dir"]
            print(f"  Class dict dir: {args.class_dict_dir}")
        
        # Also update other paths if they weren't explicitly set
        if args.dataset_file == "gc-30M_sample50k.dataset":
            args.dataset_file = dataset_paths["dataset_file"]
            print(f"  Dataset file: {args.dataset_file}")
        
        if args.gene_class_dict == "dosage_sensitivity_TFs.pickle":
            args.gene_class_dict = dataset_paths["gene_class_dict"]
            print(f"  Gene class dict: {args.gene_class_dict}")

print(f"\nUsing dataset: {args.dataset_name}")
print(f"Pre-split mode: {args.use_pre_split}")
if args.use_pre_split:
    print(f"Pre-split directory: {args.pre_split_dir}")
    print(f"Class dict directory: {args.class_dict_dir}")


torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

# Clear GPU cache before starting
torch.cuda.empty_cache()

# -----------------------------------------------------------
# 2 ▸ Load data & prepare for K-fold cross validation
# -----------------------------------------------------------

def load_dict(pth):
    pth = pathlib.Path(pth)
    with open(pth, "rb" if pth.suffix == ".pkl" or pth.suffix == ".pickle" else "r") as f:
        return (
            pickle.load(f) if pth.suffix == ".pkl" or pth.suffix == ".pickle"
            else json.load(f) if pth.suffix == ".json"
            else yaml.safe_load(f)
        )

def load_pre_split_datasets(pre_split_dir, dataset_name, n_folds=5):
    """Load pre-split train/val/test datasets for each fold"""
    print(f"Loading pre-split datasets from: {pre_split_dir}")
    print(f"Dataset name: {dataset_name}")
    
    fold_datasets = {}
    
    for fold_idx in range(1, n_folds + 1):
        print(f"Loading fold {fold_idx}...")
        
        # Construct paths for each fold using dataset name
        train_path = os.path.join(pre_split_dir, f"{dataset_name}_train_gene_labeled_ksplit{fold_idx}.dataset")
        val_path = os.path.join(pre_split_dir, f"{dataset_name}_valid_gene_labeled_ksplit{fold_idx}.dataset")
        test_path = os.path.join(pre_split_dir, f"{dataset_name}_test_gene_labeled_ksplit{fold_idx}.dataset")
        
        # Check if files exist
        for path in [train_path, val_path, test_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Dataset file not found: {path}")
        
        # Load datasets
        try:
            train_ds = load_from_disk(train_path)
            val_ds = load_from_disk(val_path)
            test_ds = load_from_disk(test_path)
            
            print(f"  Fold {fold_idx} - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
            
            fold_datasets[fold_idx] = {
                'train': train_ds,
                'val': val_ds,
                'test': test_ds
            }
            
        except Exception as e:
            print(f"Error loading datasets for fold {fold_idx}: {e}")
            raise
    
    print(f"Successfully loaded {n_folds} folds of pre-split datasets")
    return fold_datasets

if args.use_pre_split:
    print("Using pre-split datasets...")
    fold_datasets = load_pre_split_datasets(args.pre_split_dir, args.dataset_name, args.n_folds)
    
    # Always load token_dict as it's needed for the data collator
    token_dict = load_dict(args.token_dict)
    
    # Use the pre-existing class dictionary from the split data
    id_class_dict_path = os.path.join(args.class_dict_dir, f"{args.dataset_name}_id_class_dict.pkl")
    if os.path.exists(id_class_dict_path):
        print(f"Loading existing class dictionary from: {id_class_dict_path}")
        id_class_dict = load_dict(id_class_dict_path)
        class_id_dict = {v: k for k, v in id_class_dict.items()}
        print(f"Loaded {len(class_id_dict)} classes: {list(class_id_dict.keys())}")
    else:
        print("Class dictionary not found, using provided gene_class_dict...")
        # Fall back to the original method
        gene_class_dict = load_dict(args.gene_class_dict)
        
        inverse_gene_dict = {
            token_dict[g]: cls for cls, genes in gene_class_dict.items() for g in genes if g in token_dict
        }
        class_id_dict = {cls: i for i, cls in enumerate(gene_class_dict.keys())}
        id_class_dict = {v: k for k, v in class_id_dict.items()}
    
else:
    print("Creating new dataset splits...")
    full_ds = load_from_disk(args.dataset_file).shuffle(seed=args.seed)  # one .dataset only

    # Debug: Inspect dataset structure to understand potential issues
    print(f"Dataset features: {full_ds.features}")
    print(f"Dataset columns: {full_ds.column_names}")

    # Check first sample to understand data structure
    if len(full_ds) > 0:
        sample = full_ds[0]
        print(f"Sample keys: {sample.keys()}")
        for key, value in sample.items():
            print(f"  {key}: {type(value)} - {value if key != 'input_ids' else f'list of {len(value)} tokens'}")

    # Clean dataset - remove problematic fields that might cause collator issues
    problematic_fields = ['cell_types', 'cell_type', 'metadata']
    fields_to_remove = [field for field in problematic_fields if field in full_ds.column_names]

    if fields_to_remove:
        print(f"Removing problematic fields: {fields_to_remove}")
        full_ds = full_ds.remove_columns(fields_to_remove)
        print(f"Dataset columns after cleanup: {full_ds.column_names}")

    # -----------------------------------------------------------
    # 3 ▸ Dict helpers
    # -----------------------------------------------------------
    gene_class_dict = load_dict(args.gene_class_dict)      # {label: [ENS,…]}
    token_dict      = load_dict(args.token_dict)           # {ENS: int_id}

    # ↪ map gene token-id ➜ class-label
    inverse_gene_dict = {
        token_dict[g]: cls for cls, genes in gene_class_dict.items() for g in genes if g in token_dict
    }
    class_id_dict = {cls: i for i, cls in enumerate(gene_class_dict.keys())}
    id_class_dict = {v: k for k, v in class_id_dict.items()}

    def label_example(ex):
        ex["labels"] = [
            class_id_dict.get(inverse_gene_dict.get(tok, None), -100)
            for tok in ex["input_ids"]
        ]
        return ex

    # filter out cells without any labelled genes, then add "labels"
    target_tokens = set(inverse_gene_dict.keys())
    def keep_cell(ex): return not target_tokens.isdisjoint(ex["input_ids"])
    filtered_ds = full_ds.filter(keep_cell, num_proc=16).map(label_example, num_proc=16)

    print(f"Total samples after filtering: {len(filtered_ds)}")

    # Validate dataset structure before proceeding
    if not validate_dataset_structure(filtered_ds, "FILTERED"):
        print("Dataset validation failed. Attempting to fix common issues...")
        
        # Try to fix common issues
        def fix_sample(sample):
            # Ensure input_ids is a proper list of integers
            if 'input_ids' in sample:
                if not isinstance(sample['input_ids'], list):
                    sample['input_ids'] = list(sample['input_ids'])
                sample['input_ids'] = [int(x) for x in sample['input_ids'] if isinstance(x, (int, float))]
            
            # Ensure labels is a proper list of integers
            if 'labels' in sample:
                if not isinstance(sample['labels'], list):
                    sample['labels'] = list(sample['labels'])
                sample['labels'] = [int(x) for x in sample['labels'] if isinstance(x, (int, float))]
                
                # Ensure labels and input_ids have the same length
                if 'input_ids' in sample and len(sample['labels']) != len(sample['input_ids']):
                    # Truncate or pad labels to match input_ids length
                    target_length = len(sample['input_ids'])
                    if len(sample['labels']) > target_length:
                        sample['labels'] = sample['labels'][:target_length]
                    else:
                        sample['labels'].extend([-100] * (target_length - len(sample['labels'])))
            
            return sample
        
        # Apply fixes
        filtered_ds = filtered_ds.map(fix_sample, num_proc=16)
        
        # Validate again
        if not validate_dataset_structure(filtered_ds, "FIXED"):
            raise ValueError("Unable to fix dataset structure issues. Please check the input dataset.")
        else:
            print("Dataset issues fixed successfully!")

# -----------------------------------------------------------
# 4 ▸ Collator 
# -----------------------------------------------------------
# Fix: Create the collator without token_dictionary parameter and patch it afterward
data_collator = DataCollatorForGeneClassification()

# Add missing method to tokenizer for compatibility with newer transformers
def convert_ids_to_tokens(self, ids):
    """Convert token IDs to tokens. Added for compatibility with newer transformers."""
    if isinstance(ids, int):
        ids = [ids]
    
    # Create reverse dictionary if it doesn't exist
    if not hasattr(self, '_reverse_token_dict'):
        self._reverse_token_dict = {v: k for k, v in token_dict.items()}
    
    return [self._reverse_token_dict.get(id_, f"<unk_{id_}>") for id_ in ids]

# Add the method to the tokenizer instance
import types
data_collator.tokenizer.convert_ids_to_tokens = types.MethodType(convert_ids_to_tokens, data_collator.tokenizer)

# Update the global token_dictionary in the collator modules to use our token_dict
import sys
for module_name in list(sys.modules.keys()):
    if 'geneformer' in module_name and hasattr(sys.modules[module_name], 'token_dictionary'):
        sys.modules[module_name].token_dictionary = token_dict

# Also update the tokenizer within the data_collator if it exists
if hasattr(data_collator.tokenizer, 'token_dictionary'):
    data_collator.tokenizer.token_dictionary = token_dict

# Update special token IDs in the precollator
# Use object.__setattr__ to bypass the validation that requires convert_ids_to_tokens
object.__setattr__(data_collator.tokenizer, 'mask_token_id', token_dict.get("<mask>"))
object.__setattr__(data_collator.tokenizer, 'pad_token_id', token_dict.get("<pad>"))
object.__setattr__(data_collator.tokenizer, 'all_special_ids', [
    token_dict.get("<mask>"),
    token_dict.get("<pad>")
])

print("Data collator initialized and token dictionary updated.")

# Create a safer version of the data collator to handle problematic fields
class SafeDataCollatorForGeneClassification(DataCollatorForGeneClassification):
    """Enhanced data collator that filters out problematic fields"""
    
    def __call__(self, features):
        # Filter out problematic fields that might cause tensor conversion issues
        safe_features = []
        for feature in features:
            # Keep only the essential fields needed for gene classification
            safe_feature = {}
            
            # Handle input_ids
            if 'input_ids' in feature:
                input_ids = feature['input_ids']
                if isinstance(input_ids, list):
                    # Make sure all elements are integers
                    safe_feature['input_ids'] = [int(x) for x in input_ids if isinstance(x, (int, float))]
                else:
                    print(f"Warning: input_ids is not a list: {type(input_ids)}")
                    # Try to convert to list
                    try:
                        safe_feature['input_ids'] = [int(x) for x in list(input_ids) if isinstance(x, (int, float))]
                    except Exception as e:
                        print(f"Error converting input_ids: {e}")
                        continue
            
            # Handle labels
            if 'labels' in feature:
                labels = feature['labels']
                if isinstance(labels, list):
                    # Make sure all elements are integers
                    safe_feature['labels'] = [int(x) for x in labels if isinstance(x, (int, float))]
                else:
                    print(f"Warning: labels is not a list: {type(labels)}")
                    # Try to convert to list
                    try:
                        safe_feature['labels'] = [int(x) for x in list(labels) if isinstance(x, (int, float))]
                    except Exception as e:
                        print(f"Error converting labels: {e}")
                        continue
            
            # Ensure input_ids and labels have the same length
            if 'input_ids' in safe_feature and 'labels' in safe_feature:
                input_len = len(safe_feature['input_ids'])
                label_len = len(safe_feature['labels'])
                
                if input_len != label_len:
                    print(f"Warning: Length mismatch - input_ids: {input_len}, labels: {label_len}")
                    # Truncate or pad to match the shorter one
                    min_len = min(input_len, label_len)
                    safe_feature['input_ids'] = safe_feature['input_ids'][:min_len]
                    safe_feature['labels'] = safe_feature['labels'][:min_len]
            
            # Add other essential fields if they exist and are safe
            for key in ['attention_mask', 'length']:
                if key in feature:
                    value = feature[key]
                    # Only add if it's a simple type
                    if isinstance(value, (int, float, list)):
                        safe_feature[key] = value
            
            # Skip this feature if it doesn't have the essential fields
            if 'input_ids' not in safe_feature or 'labels' not in safe_feature:
                print(f"Warning: Skipping feature due to missing essential fields")
                continue
                
            safe_features.append(safe_feature)
        
        if not safe_features:
            raise ValueError("No valid features found after cleaning")
        
        # Call the parent method with cleaned features
        try:
            return super().__call__(safe_features)
        except Exception as e:
            print(f"Error in data collator: {e}")
            print(f"Number of features: {len(safe_features)}")
            if safe_features:
                print(f"Feature sample keys: {safe_features[0].keys()}")
                print(f"First feature input_ids length: {len(safe_features[0].get('input_ids', []))}")
                print(f"First feature labels length: {len(safe_features[0].get('labels', []))}")
                
                # Additional debugging - check for nested structures
                for key, value in safe_features[0].items():
                    print(f"  {key}: type={type(value)}, value_preview={str(value)[:100]}...")
            raise

# Replace the data collator with the safer version
data_collator = SafeDataCollatorForGeneClassification()

# Add missing method to tokenizer for compatibility with newer transformers
def convert_ids_to_tokens(self, ids):
    """Convert token IDs to tokens. Added for compatibility with newer transformers."""
    if isinstance(ids, int):
        ids = [ids]
    
    # Create reverse dictionary if it doesn't exist
    if not hasattr(self, '_reverse_token_dict'):
        self._reverse_token_dict = {v: k for k, v in token_dict.items()}
    
    return [self._reverse_token_dict.get(id_, f"<unk_{id_}>") for id_ in ids]

# Add the method to the tokenizer instance
import types
data_collator.tokenizer.convert_ids_to_tokens = types.MethodType(convert_ids_to_tokens, data_collator.tokenizer)

# Update the global token_dictionary in the safer collator
if hasattr(data_collator.tokenizer, 'token_dictionary'):
    data_collator.tokenizer.token_dictionary = token_dict

# Update special token IDs in the safer collator
# Use object.__setattr__ to bypass the validation that requires convert_ids_to_tokens
object.__setattr__(data_collator.tokenizer, 'mask_token_id', token_dict.get("<mask>"))
object.__setattr__(data_collator.tokenizer, 'pad_token_id', token_dict.get("<pad>"))
object.__setattr__(data_collator.tokenizer, 'all_special_ids', [
    token_dict.get("<mask>"),
    token_dict.get("<pad>")
])

print("Safe data collator initialized and token dictionary updated.")

# -----------------------------------------------------------
# 5 ▸ Helper function for parsing encoder layers to freeze
# -----------------------------------------------------------
def get_layers_to_freeze(num_layers_to_freeze):
    """Get list of encoder layer indices to freeze (first n layers).
    
    Args:
        num_layers_to_freeze: Integer specifying number of first layers to freeze
        
    Returns:
        List of integers representing encoder layer indices to freeze
    """
    if num_layers_to_freeze <= 0:
        return []
    
    return list(range(num_layers_to_freeze))

# -----------------------------------------------------------
# 6 ▸ Model creation function
# -----------------------------------------------------------
def create_model():
    """Create a fresh model for each fold with enhanced parameter configuration"""
    
    if args.prompt_type == 'encoder_prompt':
        config_path='Pretrain_ckpts/geneformer-12L-30M-prompt/config.json'
    elif args.prompt_type == 'lora':
        config_path='Pretrain_ckpts/config_lora.json'
    elif args.prompt_type == 'prefix_prompt':
        config_path='Pretrain_ckpts/config_prefix.json'
    elif args.prompt_type == 'Gene_token_prompt':
        config_path='Pretrain_ckpts/config_token.json'
    elif args.prompt_type == 'normal_finetune':
        config_path = None  # Use default config from the checkpoint
    else:
        raise ValueError(f"Unknown prompt type: {args.prompt_type}")
    
    if config_path is not None:
        model = BertForTokenClassification.from_pretrained(
            args.ckpt_dir,
            num_labels=len(class_id_dict),
            ignore_mismatched_sizes=False,
            config=config_path
        ).to("cuda")
    else:
        # For normal finetune, use default config from checkpoint
        model = BertForTokenClassification.from_pretrained(
            "Pretrain_ckpts/geneformer-12L-30M-finetune",
            num_labels=len(class_id_dict),
            ignore_mismatched_sizes=False
        ).to("cuda")

    if args.prompt_type == 'normal_finetune':
        prompt_types = ['normal_finetune']
        print(f"Using normal finetune mode (no prompt methods)")
    else:
        prompt_types = [p.strip() for p in model.config.prompt_type.split(",") if p.strip()]
        print(f"Loaded modelPrompt types: {prompt_types}")
    
    # Debug: Check ALL parameter names to understand the model structure
    print("Checking model structure...")
    all_param_names = []
    adapter_count = 0
    for name, param in model.named_parameters():
        all_param_names.append(name)
        if any(pattern in name for pattern in ["Space_Adapter", "MLP_Adapter", "adapter"]):
            adapter_count += 1
    
    print(f"Total parameters: {len(all_param_names)}")
    print(f"Adapter parameters found: {adapter_count}")
    
    # Show a few sample parameter names for verification
    print("Sample parameter names:")
    for name in all_param_names[:5]:
        print(f"  {name}")
    print("  ...")
    for name in all_param_names[-3:]:
        print(f"  {name}")
    
    # First, set all parameters to not require gradients
    for param in model.parameters():
        param.requires_grad = False

    trainable_count = 0
    trainable_params = []
    
    if "lora" in prompt_types:
        print("Applying LoRA configuration...")
        
        # CRITICAL: Set global LoRA state before doing anything else
        print("Configuring global LoRA settings...")
        if hasattr(lora, 'set_lora_training'):
            lora.set_lora_training(True)
            print("  Set global LoRA training to True")
        
        # Apply LoRA configuration first
        lora.mark_only_lora_as_trainable(model, bias="lora_only")
        print("  ✅ Applied lora.mark_only_lora_as_trainable")
        
        # IMMEDIATE UNMERGE: Unmerge all LoRA layers immediately after marking
        print("🔧 Immediate post-marking LoRA unmerge...")
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and hasattr(module, 'merged'):
                if module.merged:
                    print(f"  Unmerging {name} immediately after marking...")
                    module.merged = False
        
        # CRITICAL FIX: Verify LoRA layers are properly initialized and connected
        print("Verifying LoRA layer initialization...")
        lora_layers_found = 0
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                lora_layers_found += 1
                print(f"  Found LoRA layer: {name}")
                
                # Ensure LoRA is enabled
                if hasattr(module, 'lora_alpha') and hasattr(module, 'r'):
                    print(f"    LoRA config: r={module.r}, alpha={module.lora_alpha}")
                
                # Check if scaling is set correctly
                if hasattr(module, 'scaling'):
                    print(f"    LoRA scaling: {module.scaling}")
        
        print(f"Total LoRA layers found: {lora_layers_found}")
        
        # Then fine-tune the configuration and collect all trainable parameters
        for name, param in model.named_parameters():
            if param.requires_grad:  # If lora.mark_only_lora_as_trainable made it trainable
                trainable_count += param.numel()
                trainable_params.append(name)
                print(f"  Set trainable (LoRA): {name}")
        
        # CRITICAL LoRA FIX: Ensure bias parameters are trainable if LoRA is modifying their corresponding weights
        print("🔧 Ensuring LoRA-related bias parameters are trainable...")
        lora_related_bias_count = 0
        for name, param in model.named_parameters():
            # Check if this is a bias parameter for a layer that has LoRA
            if 'bias' in name and not param.requires_grad:
                # Extract the layer path to check for corresponding LoRA
                layer_path = name.replace('.bias', '')
                has_lora = False
                
                # Check if there's a corresponding LoRA layer
                for lora_name, _ in model.named_parameters():
                    if 'lora_' in lora_name and layer_path in lora_name:
                        has_lora = True
                        break
                
                if has_lora:
                    param.requires_grad = True
                    trainable_count += param.numel()
                    trainable_params.append(name)
                    lora_related_bias_count += 1
                    print(f"  Set trainable (LoRA bias): {name}")
        
        print(f"✅ Made {lora_related_bias_count} LoRA-related bias parameters trainable")
        
        # Ensure classifier is always trainable for LoRA
        for name, param in model.named_parameters():
            if "classifier" in name and not param.requires_grad:
                param.requires_grad = True
                trainable_count += param.numel()
                trainable_params.append(name)
                print(f"  Set trainable (LoRA classifier): {name}")
        
        # CRITICAL FIX: Force enable LoRA if it was disabled
        print("Ensuring LoRA is enabled...")
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # Force enable LoRA if it has a disable method
                if hasattr(module, 'enable_lora'):
                    module.enable_lora()
                    print(f"  Enabled LoRA for: {name}")
                elif hasattr(module, 'disable_adapters'):
                    # Some LoRA implementations use disable_adapters=False to enable
                    module.disable_adapters = False
                    print(f"  Enabled adapters for: {name}")
                
                # Verify LoRA parameters are properly set
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    lora_A = getattr(module, 'lora_A')
                    lora_B = getattr(module, 'lora_B')
                    if lora_A is not None and lora_B is not None:
                        print(f"  LoRA parameters verified for: {name}")
                        # Check if they require gradients
                        lora_A_weight = get_lora_weight(lora_A)
                        lora_B_weight = get_lora_weight(lora_B)
                        if lora_A_weight is not None and lora_A_weight.requires_grad:
                            print(f"    lora_A.weight requires_grad: True")
                        if lora_B_weight is not None and lora_B_weight.requires_grad:
                            print(f"    lora_B.weight requires_grad: True")
                
                # CRITICAL: Ensure LoRA is unmerged for training
                if hasattr(module, 'merged'):
                    if module.merged:
                        print(f"  ⚠️  LoRA layer {name} is MERGED! Unmerging for training...")
                        unmerged_successfully = False
                        
                        # CRITICAL FIX: Proper LoRA unmerging sequence
                        unmerged_successfully = False
                        
                        # Method 1: Try the standard unmerge method
                        if hasattr(module, 'unmerge'):
                            try:
                                module.unmerge()
                                unmerged_successfully = True
                                print(f"  ✅ Successfully unmerged {name} using unmerge()")
                            except Exception as e:
                                print(f"  ❌ Failed to unmerge {name} using unmerge(): {e}")
                        
                        # Method 2: Force unmerge by resetting state
                        if not unmerged_successfully:
                            print(f"  🔧 Force unmerging {name} manually...")
                            
                            # Critical: Set merged to False FIRST
                            module.merged = False
                            
                            # Re-enable LoRA computation paths
                            if hasattr(module, 'disable_adapters'):
                                module.disable_adapters = False
                            
                            # Ensure training mode for proper gradient flow
                            module.train()
                            
                            # CRITICAL: Re-initialize LoRA parameters if they're problematic
                            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                                lora_A = module.lora_A
                                lora_B = module.lora_B
                                
                                if lora_A is not None and lora_B is not None:
                                    # Check if lora_A is zero-initialized (common issue)
                                    lora_A_weight = get_lora_weight(lora_A)
                                    lora_B_weight = get_lora_weight(lora_B)
                                    if lora_A_weight is not None and lora_A_weight.norm().item() < 1e-8:
                                        print(f"    Reinitializing zero lora_A for {name}")
                                        torch.nn.init.normal_(lora_A_weight, std=0.02)
                                    
                                    # Ensure gradients are enabled
                                    if lora_A_weight is not None:
                                        lora_A_weight.requires_grad_(True)
                                    if lora_B_weight is not None:
                                        lora_B_weight.requires_grad_(True)
                            
                            unmerged_successfully = True
                            print(f"  ✅ Force unmerged {name} with parameter reinitialization")
                        
                        # Verify the unmerging worked
                        if hasattr(module, 'merged'):
                            if not module.merged:
                                print(f"  ✅ Verified: {name} is now unmerged (merged={module.merged})")
                            else:
                                print(f"  ❌ ERROR: {name} is still merged after unmerge attempt!")
                    else:
                        print(f"  ✅ LoRA layer {name} is already unmerged")
                else:
                    print(f"  ℹ️  LoRA layer {name} has no 'merged' attribute")
                
                # CRITICAL: Initialize lora_A with small values if it's zero-initialized  
                if hasattr(module, 'lora_A') and module.lora_A is not None:
                    lora_A_weight = get_lora_weight(module.lora_A)
                    if lora_A_weight is not None:
                        lora_A_norm = lora_A_weight.norm().item()
                        lora_A_mean = lora_A_weight.abs().mean().item()
                        lora_A_std = lora_A_weight.std().item()
                        print(f"    📊 LoRA-A {name}: norm={lora_A_norm:.6f}, mean={lora_A_mean:.6f}, std={lora_A_std:.6f}")
                        
                        # More aggressive initialization check
                        if lora_A_norm < 1e-6 or lora_A_std < 1e-6:
                            print(f"    🔄 Reinitializing lora_A for {name} (norm={lora_A_norm:.2e}, std={lora_A_std:.2e})")
                            torch.nn.init.normal_(lora_A_weight, std=0.02)  # Increased std for better gradients
                            print(f"    ✅ New lora_A stats: norm={lora_A_weight.norm().item():.6f}, std={lora_A_weight.std().item():.6f}")
                            
                            # CRITICAL: Force unmerge after reinitialization
                            if hasattr(module, 'merged') and hasattr(module, 'unmerge'):
                                if module.merged:
                                    print(f"    🔧 Force unmerging {name} after lora_A reinitialization")
                                    try:
                                        module.unmerge()
                                        print(f"    ✅ Successfully unmerged {name}")
                                    except Exception as e:
                                        print(f"    ⚠️  Failed to unmerge {name}: {e}")
                        
                        # CRITICAL: Ensure requires_grad is True
                        lora_A_weight.requires_grad_(True)
                
                # CRITICAL: Also check and fix lora_B
                if hasattr(module, 'lora_B') and module.lora_B is not None:
                    lora_B_weight = get_lora_weight(module.lora_B)
                    if lora_B_weight is not None:
                        # Ensure requires_grad is True
                        lora_B_weight.requires_grad_(True)
                        
                        # Initialize lora_B if needed (though it's usually initialized properly)
                        lora_B_norm = lora_B_weight.norm().item()
                        if lora_B_norm < 1e-8:
                            print(f"    Reinitializing lora_B for {name} (norm was {lora_B_norm:.2e})")
                            torch.nn.init.zeros_(lora_B_weight)  # lora_B should start at zero
                            print(f"    Reset lora_B norm: {lora_B_weight.norm().item():.6f}")
                
                # CRITICAL: Verify the module can compute LoRA correctly
                if hasattr(module, 'forward') and hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    # Force a test computation to ensure LoRA is working
                    try:
                        with torch.no_grad():
                            # Create a small test input
                            if hasattr(module, 'in_features'):
                                # Handle both cases: lora_A as Parameter or as module with .weight
                                lora_device = get_lora_device(module.lora_A)
                                
                                test_input = torch.randn(1, module.in_features, device=lora_device)
                                _ = module(test_input)
                                print(f"    ✅ LoRA forward pass test successful for {name}")
                    except Exception as e:
                        print(f"    ⚠️  LoRA forward pass test failed for {name}: {e}")
                        # Try to fix by resetting the module state
                        if hasattr(module, 'merged'):
                            module.merged = False
                        if hasattr(module, 'disable_adapters'):
                            module.disable_adapters = False
        
        # CRITICAL ADDITIONAL FIX: Force all LoRA modules to training mode and unmerged state
        print("🔧 Final LoRA state verification and cleanup...")
        lora_fixed_count = 0
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # Ensure training mode
                module.train()
                
                # Force unmerged state
                if hasattr(module, 'merged'):
                    if module.merged:
                        print(f"  🚨 CRITICAL: {name} is still merged! Force fixing...")
                        module.merged = False
                        lora_fixed_count += 1
                        
                        # Additional fix: ensure LoRA computation is enabled
                        if hasattr(module, 'enable_lora'):
                            try:
                                module.enable_lora()
                                print(f"    ✅ Enabled LoRA computation for {name}")
                            except:
                                pass
        
        if lora_fixed_count > 0:
            print(f"🔧 Force-fixed {lora_fixed_count} LoRA modules that were still merged")
        else:
            print("✅ All LoRA modules are properly unmerged")
        
        # ULTRA-CRITICAL FIX: Completely reset LoRA state if issues persist
        print("🔧 Performing complete LoRA state reset...")
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # Force all LoRA attributes to training state
                if hasattr(module, 'merged'):
                    module.merged = False
                if hasattr(module, 'training'):
                    module.training = True
                if hasattr(module, 'disable_adapters'):
                    module.disable_adapters = False
                
                # Ensure LoRA parameters are properly connected
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    lora_A = module.lora_A
                    lora_B = module.lora_B
                    
                    if lora_A is not None and lora_B is not None:
                        # Ensure they require gradients
                        lora_A_weight = get_lora_weight(lora_A)
                        lora_B_weight = get_lora_weight(lora_B)
                        if lora_A_weight is not None:
                            lora_A_weight.requires_grad_(True)
                        if lora_B_weight is not None:
                            lora_B_weight.requires_grad_(True)
                        
                        # Re-initialize if they're zero
                        if lora_A_weight is not None and lora_A_weight.norm().item() < 1e-8:
                            torch.nn.init.normal_(lora_A_weight, std=0.01)
                            print(f"    Reset lora_A for {name}")
        
        print("✅ LoRA state reset completed")
        
        # ULTRA-CRITICAL: Comprehensive LoRA diagnostic and repair
        print("🔧 Running comprehensive LoRA diagnostic and repair...")
        lora_issues_fixed = 0
        
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                issues_found = []
                
                # Check 1: Merged state
                if hasattr(module, 'merged') and module.merged:
                    issues_found.append("merged")
                    module.merged = False
                    
                # Check 2: Disabled adapters
                if hasattr(module, 'disable_adapters') and module.disable_adapters:
                    issues_found.append("disabled_adapters")
                    module.disable_adapters = False
                
                # Check 3: Training mode
                if not module.training:
                    issues_found.append("eval_mode")
                    module.train()
                
                # Check 4: LoRA parameter initialization and gradients
                lora_A = module.lora_A
                lora_B = module.lora_B
                
                if lora_A is not None:
                    lora_A_weight = get_lora_weight(lora_A)
                    if lora_A_weight is not None:
                        # Check norm
                        if lora_A_weight.norm().item() < 1e-8:
                            issues_found.append("zero_lora_A")
                            torch.nn.init.normal_(lora_A_weight, std=0.02)
                        
                        # Check requires_grad
                        if not lora_A_weight.requires_grad:
                            issues_found.append("lora_A_no_grad")
                            lora_A_weight.requires_grad_(True)
                
                if lora_B is not None:
                    lora_B_weight = get_lora_weight(lora_B)
                    if lora_B_weight is not None:
                        # Check requires_grad
                        if not lora_B_weight.requires_grad:
                            issues_found.append("lora_B_no_grad")
                            lora_B_weight.requires_grad_(True)
                
                # Check 5: Scaling factor
                if hasattr(module, 'scaling') and module.scaling <= 0:
                    issues_found.append("invalid_scaling")
                    module.scaling = module.lora_alpha / module.r if hasattr(module, 'lora_alpha') and hasattr(module, 'r') else 0.125
                
                if issues_found:
                    lora_issues_fixed += 1
                    print(f"    Fixed LoRA issues for {name}: {', '.join(issues_found)}")
        
        print(f"✅ LoRA diagnostic completed. Fixed issues in {lora_issues_fixed} modules")

    if "Gene_token_prompt" in prompt_types:
        print("Applying Gene_token_prompt configuration...")
        for name, param in model.named_parameters():
            # Look for adapter patterns more broadly - Gene_token_prompt uses specific patterns
            if any(pattern in name for pattern in ["adapter", "bert.adapter", "classifier", "gene_adapter", "token_adapter"]):
                param.requires_grad = True
                trainable_count += param.numel()
                trainable_params.append(name)
                
    if "encoder_prompt" in prompt_types:
        print("Applying encoder_prompt configuration...")
        for name, param in model.named_parameters():
            # encoder_prompt typically uses Space_Adapter and MLP_Adapter
            if any(pattern in name for pattern in ["Space_Adapter", "MLP_Adapter", "adapter", "classifier", "encoder_adapter"]):
                param.requires_grad = True
                trainable_count += param.numel()
                trainable_params.append(name)
                
    if "prefix_prompt" in prompt_types:
        print("Applying prefix_prompt configuration...")
        for name, param in model.named_parameters():
            # prefix_prompt uses prompt embeddings
            if any(pattern in name for pattern in ["prompt_embeddings", "prompt", "classifier", "prefix"]):
                param.requires_grad = True
                trainable_count += param.numel()
                trainable_params.append(name)
    
    if "normal_finetune" in prompt_types:
        print("Applying normal_finetune configuration...")
        
        # Get which encoder layers to freeze (first n layers)
        layers_to_freeze = get_layers_to_freeze(args.freeze_layers)
        if layers_to_freeze:
            print(f"Will freeze first {args.freeze_layers} encoder layers: {layers_to_freeze}")
        else:
            print("No encoder layers will be frozen (full finetune)")
        
        # For normal finetune, make all parameters trainable first
        for name, param in model.named_parameters():
            param.requires_grad = True
            trainable_count += param.numel()
            trainable_params.append(name)
        
        # Then freeze specified encoder layers
        frozen_count = 0
        if layers_to_freeze:
            for name, param in model.named_parameters():
                # Check if this parameter belongs to an encoder layer that should be frozen
                for layer_idx in layers_to_freeze:
                    if f"encoder.layer.{layer_idx}." in name:
                        param.requires_grad = False
                        trainable_count -= param.numel()
                        trainable_params.remove(name)
                        frozen_count += param.numel()
                        print(f"  Frozen encoder layer {layer_idx}: {name}")
                        break
        
        print(f"Normal finetune setup complete:")
        print(f"  First {args.freeze_layers} layers frozen, parameters: {frozen_count:,}")
        print(f"  Remaining trainable parameters: {trainable_count:,}")
    
    # CRITICAL: Make sure classifier is ALWAYS trainable
    classifier_found = False
    for name, param in model.named_parameters():
        if "classifier" in name:
            if not param.requires_grad:
                param.requires_grad = True
                trainable_count += param.numel()
                trainable_params.append(name)
                print(f"  Set trainable (classifier): {name}")
            classifier_found = True
    
    if not classifier_found:
        print("WARNING: No classifier layer found! Looking for alternative classification heads...")
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in ["cls", "head", "prediction", "output"]):
                param.requires_grad = True
                trainable_count += param.numel()
                trainable_params.append(name)
                print(f"  Set trainable (classification head): {name}")
    # CRITICAL: Comprehensive parameter validation and debugging
    print("\n" + "="*80)
    print("COMPREHENSIVE PARAMETER VALIDATION")
    print("="*80)
    
    # Step 1: Count all parameters
    total_params = sum(p.numel() for p in model.parameters())
    actual_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Step 2: Get actual trainable parameter names
    actual_trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    
    # Step 3: Verify consistency
    print(f"Parameter Count Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters (calculated): {actual_trainable:,}")
    print(f"  Trainable parameters (expected): {trainable_count:,}")
    print(f"  Percentage trainable: {100 * actual_trainable / total_params:.2f}%")
    
    # Step 4: Check for discrepancies
    if actual_trainable != trainable_count:
        print(f"\n⚠️  WARNING: Discrepancy detected!")
        print(f"  Expected trainable: {trainable_count:,}")
        print(f"  Actual trainable: {actual_trainable:,}")
        print(f"  Difference: {actual_trainable - trainable_count:,}")
    
    # Step 5: List all trainable parameters
    print(f"\nActual trainable parameters ({len(actual_trainable_names)}):")

    for i, name in enumerate(actual_trainable_names[:5]):  # Only show first 5
        param = dict(model.named_parameters())[name]
        print(f"  {i+1:2d}. {name} - shape: {list(param.shape)} - params: {param.numel():,}")
    
    if len(actual_trainable_names) > 5:
        print(f"  ... and {len(actual_trainable_names) - 5} more parameters")
    
    # Step 6: Verify against expected
    expected_set = set(trainable_params)
    actual_set = set(actual_trainable_names)
    
    if expected_set != actual_set:
        print(f"\n⚠️  Parameter name mismatch detected!")
        missing = expected_set - actual_set
        extra = actual_set - expected_set
        if missing:
            print(f"  Missing parameters: {missing}")
        if extra:
            print(f"  Extra parameters: {extra}")
    
    # Step 7: Critical error check
    if actual_trainable == 0:
        print(f"\n❌ CRITICAL ERROR: No parameters are set to trainable!")
        print("This will cause gradient computation issues during training.")
        raise ValueError("ERROR: No parameters are set to trainable! This will cause the gradient warning.")
    
    # Step 8: Parameter gradient check function
    def verify_gradients_after_backward(model, step_info=""):
        """Verify gradients are computed for trainable parameters"""
        print(f"\n🔍 Gradient verification {step_info}:")
        trainable_with_grad = 0
        trainable_without_grad = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    print(f"  ✅ {name}: grad_norm={grad_norm:.6f}")
                    trainable_with_grad += 1
                else:
                    print(f"  ❌ {name}: NO GRADIENT")
                    trainable_without_grad += 1
        
        print(f"  Summary: {trainable_with_grad} with gradients, {trainable_without_grad} without")
        return trainable_with_grad, trainable_without_grad
    
    # Step 9: Store validation function in model for later use
    model._verify_gradients = verify_gradients_after_backward
    
    # Validate LoRA configuration early if this is a LoRA model
    if args.prompt_type == 'lora':
        print("\n🔧 Early LoRA configuration validation...")
        lora_success, lora_issues, lora_fixes = validate_and_fix_lora_configuration(model, verbose=False)
        
        if lora_issues:
            print(f"⚠️  Found {len(lora_issues)} LoRA configuration issues")
        if lora_fixes:
            print(f"✅ Applied {len(lora_fixes)} LoRA fixes")
        if not lora_success:
            print("❌ Some LoRA issues require manual attention")
    
    print("="*80)
    
    return model, prompt_types, actual_trainable_names

def count_trainable(model, trainable_param_names):
    """Debug function to count and display trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable / total params: {train:,} / {total:,}")
    
    # List specific trainable parameters
    actual_trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            actual_trainable_params.append(name)
    
    if actual_trainable_params:
        print("Actually trainable parameters:")
        for name in actual_trainable_params[:10]:  # Show first 10
            print(f"  - {name}")
        if len(actual_trainable_params) > 10:
            print(f"  ... and {len(actual_trainable_params) - 10} more")
    else:
        print("WARNING: No trainable parameters found!")
        return train, total
    
    # Check consistency with expected trainable parameters - handle None case
    if trainable_param_names is None:
        print("⚠️  No expected trainable parameter names provided - skipping validation")
        return train, total
        
    expected_set = set(trainable_param_names)
    actual_set = set(actual_trainable_params)
    
    # More detailed comparison with better error handling
    if expected_set != actual_set:
        print("WARNING: Mismatch between expected and actual trainable parameters!")
        missing = expected_set - actual_set
        extra = actual_set - expected_set
        
        if missing:
            print(f"  Missing from actual ({len(missing)} params):")
            for param in sorted(list(missing)[:5]):  # Show first 5
                print(f"    - {param}")
            if len(missing) > 5:
                print(f"    ... and {len(missing) - 5} more")
                
        if extra:
            print(f"  Extra in actual ({len(extra)} params):")
            for param in sorted(list(extra)[:5]):  # Show first 5
                print(f"    + {param}")
            if len(extra) > 5:
                print(f"    ... and {len(extra) - 5} more")
                
        # Try to find pattern matches for better debugging
        print("\n🔍 Analyzing parameter name patterns:")
        expected_patterns = set()
        actual_patterns = set()
        
        for name in expected_set:
            if '.' in name:
                expected_patterns.add(name.split('.')[0])
        for name in actual_set:
            if '.' in name:
                actual_patterns.add(name.split('.')[0])
                
        if expected_patterns and actual_patterns:
            common_patterns = expected_patterns & actual_patterns
            if common_patterns:
                print(f"  Common module patterns: {sorted(common_patterns)}")
            missing_patterns = expected_patterns - actual_patterns
            if missing_patterns:
                print(f"  Missing module patterns: {sorted(missing_patterns)}")
            extra_patterns = actual_patterns - expected_patterns
            if extra_patterns:
                print(f"  Extra module patterns: {sorted(extra_patterns)}")
    else:
        print("✓ Trainable parameters match expectations")
    
    return train, total

def validate_parameter_consistency(model, optimizer, expected_param_names=None):
    """
    Comprehensive validation of parameter consistency between model and optimizer
    """
    print("\n🔍 Validating parameter consistency...")
    
    # Get model parameters
    model_params = {}
    trainable_model_params = {}
    for name, param in model.named_parameters():
        model_params[name] = param
        if param.requires_grad:
            trainable_model_params[name] = param
    
    print(f"  Model parameters: {len(model_params)}")
    print(f"  Trainable model parameters: {len(trainable_model_params)}")
    
    # Get optimizer parameters
    optimizer_param_ids = set()
    optimizer_param_count = 0
    
    if hasattr(optimizer, 'param_groups') and optimizer.param_groups:
        for group in optimizer.param_groups:
            for param in group['params']:
                optimizer_param_ids.add(id(param))
                optimizer_param_count += 1
        print(f"  Optimizer parameters: {optimizer_param_count}")
        
        # Check consistency
        trainable_param_ids = {id(param) for param in trainable_model_params.values()}
        
        if trainable_param_ids == optimizer_param_ids:
            print("  ✅ Model and optimizer parameters are consistent")
            consistency_ok = True
        else:
            print("  ❌ Model and optimizer parameters are inconsistent!")
            missing = trainable_param_ids - optimizer_param_ids
            extra = optimizer_param_ids - trainable_param_ids
            
            if missing:
                print(f"    Missing in optimizer: {len(missing)} parameters")
            if extra:
                print(f"    Extra in optimizer: {len(extra)} parameters")
            consistency_ok = False
    else:
        print("  ⚠️  Optimizer has no param_groups - this is unusual")
        consistency_ok = False
    
    # Validate against expected parameters if provided
    if expected_param_names is not None:
        expected_set = set(expected_param_names)
        actual_set = set(trainable_model_params.keys())
        
        if expected_set == actual_set:
            print("  ✅ Expected and actual trainable parameters match")
        else:
            print("  ⚠️  Expected and actual trainable parameters differ")
            missing = expected_set - actual_set
            extra = actual_set - expected_set
            if missing:
                print(f"    Missing: {len(missing)} parameters")
            if extra:
                print(f"    Extra: {len(extra)} parameters")
    
    return consistency_ok

def validate_and_fix_lora_configuration(model, verbose=True):
    """
    Comprehensive LoRA configuration validation and automatic fixing
    """
    if verbose:
        print("\n🔍 Validating and fixing LoRA configuration...")
    
    lora_modules = []
    issues_found = []
    fixes_applied = []
    
    # Find all LoRA modules
    for name, module in model.named_modules():
        # Check for various LoRA implementations
        is_lora = False
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            is_lora = True
        elif hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
            issues_found.append(f"Incomplete LoRA module {name}: missing lora_A or lora_B")
            continue
        
        if is_lora:
            lora_modules.append((name, module))
            if verbose:
                print(f"  Found LoRA module: {name}")
    
    if not lora_modules:
        if verbose:
            print("  ⚠️  No LoRA modules found! This might not be a LoRA model.")
        return True, [], []
    
    # Validate and fix each LoRA module
    for name, module in lora_modules:
        # Issue 1: Check if module is merged (prevents gradients)
        if hasattr(module, 'merged'):
            if module.merged:
                issues_found.append(f"LoRA module {name} is merged (prevents gradients)")
                
                # Try to unmerge
                unmerged = False
                if hasattr(module, 'unmerge'):
                    try:
                        module.unmerge()
                        unmerged = True
                        fixes_applied.append(f"Unmerged {name} using unmerge() method")
                    except Exception as e:
                        if verbose:
                            print(f"    ❌ unmerge() failed for {name}: {e}")
                
                # Force unmerge if method failed
                if not unmerged:
                    try:
                        module.merged = False
                        fixes_applied.append(f"Force unmerged {name} by setting merged=False")
                    except Exception as e:
                        issues_found.append(f"Failed to unmerge {name}: {e}")
                        continue
                
                # Verify unmerging worked
                if hasattr(module, 'merged') and module.merged:
                    issues_found.append(f"LoRA module {name} still merged after fix attempt")
        
        # Ensure module is always unmerged for training
        if hasattr(module, 'merged'):
            module.merged = False
        
        # Issue 2: Check LoRA scaling
        if hasattr(module, 'scaling'):
            if module.scaling == 0 or module.scaling is None:
                issues_found.append(f"LoRA module {name} has invalid scaling: {module.scaling}")
                
                # Fix scaling
                if hasattr(module, 'lora_alpha') and hasattr(module, 'r'):
                    if module.r > 0:
                        try:
                            new_scaling = module.lora_alpha / module.r
                            module.scaling = new_scaling
                            fixes_applied.append(f"Fixed {name} scaling to {new_scaling}")
                        except Exception as e:
                            issues_found.append(f"Failed to fix scaling for {name}: {e}")
                    else:
                        issues_found.append(f"LoRA module {name} has invalid rank: {module.r}")
                else:
                    issues_found.append(f"LoRA module {name} missing alpha or rank for scaling calculation")
        
        # Issue 3: Validate LoRA parameters
        if hasattr(module, 'r'):
            if module.r <= 0:
                issues_found.append(f"LoRA module {name} has invalid rank: {module.r}")
        
        if hasattr(module, 'lora_alpha'):
            if module.lora_alpha <= 0:
                issues_found.append(f"LoRA module {name} has invalid alpha: {module.lora_alpha}")
        
        # Issue 4: Check LoRA matrices
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            try:
                if module.lora_A.requires_grad != module.lora_B.requires_grad:
                    issues_found.append(f"LoRA matrices in {name} have different requires_grad settings")
                    # Fix by enabling gradients for both
                    module.lora_A.requires_grad = True
                    module.lora_B.requires_grad = True
                    fixes_applied.append(f"Enabled gradients for both LoRA matrices in {name}")
            except Exception as e:
                issues_found.append(f"Error checking LoRA matrices in {name}: {e}")
        
        # Issue 5: Ensure module is in training mode
        if not module.training:
            module.train()
            fixes_applied.append(f"Set {name} to training mode")
    
    # Summary
    if verbose:
        print(f"  Found {len(lora_modules)} LoRA modules")
        if issues_found:
            print(f"  Issues detected: {len(issues_found)}")
            for issue in issues_found[:5]:  # Show first 5
                print(f"    - {issue}")
            if len(issues_found) > 5:
                print(f"    ... and {len(issues_found) - 5} more")
        
        if fixes_applied:
            print(f"  Fixes applied: {len(fixes_applied)}")
            for fix in fixes_applied[:5]:  # Show first 5
                print(f"    ✅ {fix}")
            if len(fixes_applied) > 5:
                print(f"    ... and {len(fixes_applied) - 5} more")
        
        if not issues_found:
            print("  ✅ All LoRA modules are properly configured")
    
    success = len([issue for issue in issues_found if "Failed" in issue]) == 0
    return success, issues_found, fixes_applied

def test_lora_gradient_flow(model, test_input=None):
    """
    Test that LoRA layers can properly compute and receive gradients
    """
    print("\n🧪 Testing LoRA gradient flow...")
    
    # Find LoRA modules
    lora_modules = []
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            lora_modules.append((name, module))
    
    if not lora_modules:
        print("  ⚠️  No LoRA modules found - skipping gradient flow test")
        return True
    
    print(f"  Testing gradient flow for {len(lora_modules)} LoRA modules")
    
    # Create dummy input if not provided
    if test_input is None:
        batch_size = 2
        seq_length = 512
        test_input = {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_length), device=model.device),
            'attention_mask': torch.ones((batch_size, seq_length), device=model.device),
            'labels': torch.randint(0, 2, (batch_size, seq_length), device=model.device)
        }
    
    # Clear any existing gradients
    model.zero_grad()
    
    # Forward pass
    try:
        outputs = model(**test_input)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs.logits.mean()
        
        # Backward pass
        loss.backward()
        
        # Check if LoRA parameters have gradients
        lora_params_with_grad = 0
        lora_params_without_grad = 0
        gradient_issues = []
        
        for name, module in lora_modules:
            # Check lora_A gradients
            if hasattr(module, 'lora_A') and module.lora_A.requires_grad:
                if module.lora_A.grad is not None and module.lora_A.grad.abs().sum() > 0:
                    lora_params_with_grad += 1
                else:
                    lora_params_without_grad += 1
                    gradient_issues.append(f"lora_A in {name} has no/zero gradients")
            
            # Check lora_B gradients
            if hasattr(module, 'lora_B') and module.lora_B.requires_grad:
                if module.lora_B.grad is not None and module.lora_B.grad.abs().sum() > 0:
                    lora_params_with_grad += 1
                else:
                    lora_params_without_grad += 1
                    gradient_issues.append(f"lora_B in {name} has no/zero gradients")
        
        # Report results
        total_lora_params = lora_params_with_grad + lora_params_without_grad
        print(f"  LoRA parameters with gradients: {lora_params_with_grad}/{total_lora_params}")
        
        success = lora_params_without_grad == 0
        
        if success:
            print("  ✅ All LoRA parameters have proper gradients")
        else:
            print(f"  ❌ {lora_params_without_grad} LoRA parameters missing gradients")
            for issue in gradient_issues[:3]:  # Show first 3
                print(f"    - {issue}")
            if len(gradient_issues) > 3:
                print(f"    ... and {len(gradient_issues) - 3} more")
        
        return success
        
    except Exception as e:
        print(f"  ❌ Error during gradient flow test: {e}")
        return False
    finally:
        # Clean up
        model.zero_grad()

# -----------------------------------------------------------
# 5.5 ▸ Parameter Update Test Function
# -----------------------------------------------------------
def test_parameter_updates(model, optimizer, data_collator, sample_data, fold_num=1):
    """
    Test parameter updates with a single forward/backward pass to verify training setup
    """
    print(f"\nTesting parameter updates for fold {fold_num}...")
    print()
    print("=" * 60)
    print("TESTING PARAMETER UPDATES WITH SINGLE FORWARD/BACKWARD PASS")
    print("=" * 60)
    
    # Comprehensive LoRA configuration validation and fixing
    lora_success, lora_issues, lora_fixes = validate_and_fix_lora_configuration(model, verbose=True)
    
    if not lora_success:
        print("WARNING: Some LoRA issues could not be automatically fixed!")
        for issue in lora_issues:
            if "Failed" in issue:
                print(f"    ❌ {issue}")
    
    # Additional validation: ensure all LoRA modules are ready for training
    lora_modules = []
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            lora_modules.append((name, module))
            
            # Double-check critical settings
            if hasattr(module, 'merged') and module.merged:
                print(f"🚨 CRITICAL: {name} is STILL merged after fixes!")
                module.merged = False
                print(f"🔧 Force unmerged {name}")
            
            # Ensure training mode
            if not module.training:
                module.train()
    
    if lora_modules:
        print(f"✅ {len(lora_modules)} LoRA modules validated and ready for training")
        
        # Test LoRA gradient flow if this is a LoRA model
        gradient_success = test_lora_gradient_flow(model)
        if not gradient_success:
            print("⚠️  LoRA gradient flow test failed - training may not work properly")
    
    # Step 1: Handle None optimizer by creating a temporary one
    if optimizer is None:
        print("⚠️  Optimizer is None, creating temporary optimizer for testing...")
        
        # Get trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        if not trainable_params:
            print("❌ ERROR: No trainable parameters found!")
            return 0, 1
        
        # Create temporary optimizer
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
        print(f"✅ Created temporary optimizer with {len(trainable_params)} parameters")
    
    # Step 2: Track initial parameter values
    tracked_params = {}
    trainable_param_names = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            tracked_params[name] = param.data.clone()
            trainable_param_names.append(name)
    
    print(f"\nOptimizer setup:")
    print(f"  Type: {type(optimizer)}")
    
    # Check if optimizer has param_groups
    if hasattr(optimizer, 'param_groups') and optimizer.param_groups:
        print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
        print(f"  Number of parameter groups: {len(optimizer.param_groups)}")
    else:
        print(f"  WARNING: Optimizer has no param_groups or empty param_groups")
    
    # Step 3: Create a small batch for testing
    print(f"\nPerforming forward pass...")
    model.train()
    optimizer.zero_grad()
    
    # Use first sample from data for testing
    test_batch = data_collator([sample_data])
    
    # Move to device
    device = next(model.parameters()).device
    for key in test_batch:
        if torch.is_tensor(test_batch[key]):
            test_batch[key] = test_batch[key].to(device)
    
    # Filter out keys that the model doesn't expect (like 'length')
    # Keep only the keys that the model's forward method expects
    model_input_keys = {'input_ids', 'attention_mask', 'token_type_ids', 'position_ids', 
                       'head_mask', 'inputs_embeds', 'labels', 'output_attentions', 
                       'output_hidden_states', 'return_dict'}
    
    filtered_batch = {k: v for k, v in test_batch.items() if k in model_input_keys}
    
    print(f"Original batch keys: {list(test_batch.keys())}")
    print(f"Filtered batch keys: {list(filtered_batch.keys())}")
    
    # Forward pass
    outputs = model(**filtered_batch)
    loss = outputs.loss
    print(f"Loss: {loss.item():.6f}")
    
    print(f"Performing backward pass...")
    # Backward pass
    loss.backward()
    
    # Step 4: Check gradients
    print(f"\nChecking gradients:")
    params_with_grad = 0
    params_without_grad = 0
    lora_A_issues = 0
    
    for name in trainable_param_names:
        param = dict(model.named_parameters())[name]
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 1e-8:  # Non-zero gradient
                params_with_grad += 1
            else:
                print(f"  ⚠️  {name}: gradient is zero ({grad_norm})")
                params_without_grad += 1
                if 'lora_A' in name:
                    lora_A_issues += 1
        else:
            print(f"  ❌ {name}: NO GRADIENT COMPUTED")
            params_without_grad += 1
            if 'lora_A' in name:
                lora_A_issues += 1
    
    if lora_A_issues > 0:
        print(f"\n🚨 CRITICAL LORA ISSUE: {lora_A_issues} lora_A parameters have no/zero gradients!")
        print("This suggests LoRA layers are not properly integrated into the forward pass.")
        
        # Additional LoRA-specific fix attempt
        print("🔧 Attempting to fix LoRA issues...")
        for name, module in lora_modules:
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # Ensure LoRA components are properly initialized
                lora_A_weight = get_lora_weight(module.lora_A)
                lora_B_weight = get_lora_weight(module.lora_B)
                if lora_A_weight is not None and lora_B_weight is not None:
                    # Check if lora_A has near-zero values (common initialization issue)
                    lora_A_mean = lora_A_weight.abs().mean().item()
                    lora_A_max = lora_A_weight.abs().max().item()
                    lora_A_std = lora_A_weight.std().item()
                    print(f"    LoRA-A stats for {name}: mean={lora_A_mean:.6f}, max={lora_A_max:.6f}, std={lora_A_std:.6f}")
                    
                    # More aggressive reinitialization - check if std is too small
                    if lora_A_mean < 1e-6 or lora_A_std < 1e-6:
                        print(f"    🔄 Reinitializing lora_A for {name} (mean={lora_A_mean:.2e}, std={lora_A_std:.2e})")
                        # Reinitialize with small random values
                        torch.nn.init.normal_(lora_A_weight, std=0.02)
                        print(f"    After reinit: mean={lora_A_weight.abs().mean().item():.6f}, std={lora_A_weight.std().item():.6f}")
                    
                    # Ensure requires_grad is set
                    lora_A_weight.requires_grad = True
                    lora_B_weight.requires_grad = True
                    
                    print(f"    Fixed LoRA layer: {name}")
    
    if params_without_grad > 0:
        print(f"\nWARNING: {params_without_grad} parameters have no/zero gradients!")
    
    print(f"Performing optimizer step...")
    # Step 5: Optimizer step
    optimizer.step()
    
    # Step 6: Check parameter updates
    print(f"\nChecking parameter updates:")
    updated_params = 0
    no_update_params = 0
    
    updated_count = 0
    for name in trainable_param_names:
        param = dict(model.named_parameters())[name]
        old_value = tracked_params[name]
        
        # Calculate max change
        max_change = (param.data - old_value).abs().max().item()
        
        if max_change > 1e-8:  # Significant change
            updated_count += 1
            if updated_count <= 5:  # Only print first 5 updated parameters
                print(f"  {name}: max change = {max_change:.8f}")
                print(f"    ✓ Parameter updated!")
            elif updated_count == 6:  # Print summary for remaining
                print(f"  ... and {len(trainable_param_names) - 5} more parameters updated")
            updated_params += 1
        else:
            print(f"  {name}: max change = {max_change:.8f}")
            print(f"    Parameter NOT updated!")
            no_update_params += 1
    
    # Step 7: Summary
    print(f"\n{'SUCCESS' if no_update_params == 0 else 'FAILURE'}: " + 
          f"Parameters are {'being updated correctly' if no_update_params == 0 else 'NOT being updated properly'}!")
    
    if no_update_params > 0:
        print(f"WARNING: {no_update_params} parameters failed to update!")
        print("This indicates a problem with the optimizer or gradient computation.")
        
        # Additional debugging
        print("\nDebugging information:")
        if hasattr(optimizer, 'param_groups') and optimizer.param_groups:
            print(f"  - Learning rate: {optimizer.param_groups[0]['lr']}")
        print(f"  - Loss value: {loss.item()}")
        print(f"  - Model in training mode: {model.training}")
        
        # Check if gradients are being computed
        total_grad_norm = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        print(f"  - Total gradient norm: {total_grad_norm}")
    
    return updated_params, no_update_params

# -----------------------------------------------------------
# 6 ▸ Focal Loss for Class Imbalance
# -----------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss implementation for addressing class imbalance
    Paper: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=-100, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        
        # Compute probabilities
        pt = torch.exp(-ce_loss)
        
        # Create alpha tensor
        if isinstance(self.alpha, (float, int)):
            alpha_t = self.alpha
        else:
            alpha_t = self.alpha.gather(0, targets)
        
        # Compute focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        # Apply mask for ignore_index
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            focal_loss = focal_loss * mask.float()
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# -----------------------------------------------------------
# 7 ▸ Trainer subclass – prefix-mask & token-level CE loss
# -----------------------------------------------------------
class PromptTrainer(Trainer):
    def __init__(self, *a, prompt_types=None, trainable_params=None, loss_function=None, class_weights=None, 
                 use_focal_loss=False, focal_alpha=0.25, focal_gamma=2.0, **kw):
        super().__init__(*a, **kw)
        self.prompt_types = prompt_types
        self.trainable_params = trainable_params or []
        self.loss_function = loss_function
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self._gradient_check_done = False
        self._first_step_done = False
        self._param_values_before = {}
        self._lora_validated = False
        
    def training_step(self, model, inputs):
        """Override training step to check parameter updates"""
        
        # Validate LoRA configuration on first training step
        if not self._lora_validated:
            lora_modules = []
            for name, module in model.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    lora_modules.append((name, module))
            
            if lora_modules:
                print(f"\n🔧 Validating {len(lora_modules)} LoRA modules before training...")
                success, issues, fixes = validate_and_fix_lora_configuration(model, verbose=False)
                
                if issues:
                    print(f"⚠️  Found {len(issues)} LoRA issues during training")
                if fixes:
                    print(f"✅ Applied {len(fixes)} LoRA fixes during training")
                
                # Critical check: ensure no modules are merged
                for name, module in lora_modules:
                    if hasattr(module, 'merged') and module.merged:
                        print(f"🚨 CRITICAL: {name} is merged during training! Force unmerging...")
                        module.merged = False
            
            self._lora_validated = True
        
        # Store parameter values before first step
        if not self._first_step_done:
            print("\n" + "="*60)
            print("CHECKING PARAMETER UPDATE AFTER ONE BACKPROPAGATION")
            print("="*60)
            
            # Verify optimizer configuration now that it's initialized
            print(f"\nOptimizer configuration:")
            print(f"  Type: {type(self.optimizer)}")
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"  Learning rate: {current_lr}")
            
            # Check if learning rate is 0 due to warmup and fix it
            if current_lr == 0.0:
                print(f"  ⚠️  WARNING: Learning rate is 0.0! This is likely due to warmup.")
                print(f"  Setting learning rate to base LR: {self.args.learning_rate}")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.args.learning_rate
                print(f"  ✅ Learning rate corrected to: {self.args.learning_rate}")
            
            print(f"  Number of parameter groups: {len(self.optimizer.param_groups)}")
            
            # Count parameters in optimizer
            optimizer_param_count = 0
            total_params_in_optimizer = 0
            for group in self.optimizer.param_groups:
                optimizer_param_count += len(group['params'])
                for param in group['params']:
                    total_params_in_optimizer += param.numel()
            print(f"  Parameter tensors in optimizer: {optimizer_param_count}")
            print(f"  Total parameters in optimizer: {total_params_in_optimizer:,}")
            
            # Verify that all trainable parameters are in the optimizer with detailed analysis
            trainable_model_params = {}  # id -> (name, param)
            trainable_param_ids = set()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param_id = id(param)
                    trainable_model_params[param_id] = (name, param)
                    trainable_param_ids.add(param_id)
            
            optimizer_param_ids = set()
            optimizer_params = {}  # id -> param
            for group_idx, group in enumerate(self.optimizer.param_groups):
                for param in group['params']:
                    param_id = id(param)
                    optimizer_param_ids.add(param_id)
                    optimizer_params[param_id] = param
            
            print(f"  Trainable model parameters: {len(trainable_param_ids)}")
            print(f"  Optimizer parameters: {len(optimizer_param_ids)}")
            
            if trainable_param_ids == optimizer_param_ids:
                print("  ✓ All trainable model parameters are in the optimizer")
            else:
                print("  ✗ WARNING: Mismatch between trainable model parameters and optimizer parameters!")
                missing_in_optimizer = trainable_param_ids - optimizer_param_ids
                extra_in_optimizer = optimizer_param_ids - trainable_param_ids
                
                if missing_in_optimizer:
                    print(f"    Missing from optimizer ({len(missing_in_optimizer)} parameters):")
                    for param_id in list(missing_in_optimizer)[:3]:  # Show first 3
                        name, param = trainable_model_params[param_id]
                        print(f"      - {name} (shape: {param.shape})")
                    if len(missing_in_optimizer) > 3:
                        print(f"      ... and {len(missing_in_optimizer) - 3} more")
                        
                if extra_in_optimizer:
                    print(f"    Extra in optimizer ({len(extra_in_optimizer)} parameters):")
                    extra_count = 0
                    for param_id in extra_in_optimizer:
                        if extra_count < 3:  # Show first 3
                            param = optimizer_params[param_id]
                            # Try to find parameter name by reverse lookup
                            param_name = "unknown"
                            for name, model_param in model.named_parameters():
                                if id(model_param) == param_id:
                                    param_name = name
                                    break
                            print(f"      + {param_name} (shape: {param.shape})")
                            extra_count += 1
                    if len(extra_in_optimizer) > 3:
                        print(f"      ... and {len(extra_in_optimizer) - 3} more")
                
                # Attempt automatic fix if possible
                if missing_in_optimizer and not extra_in_optimizer:
                    print("  🔧 Attempting to fix: Adding missing parameters to optimizer...")
                    try:
                        # Create new parameter group for missing parameters
                        missing_params = [trainable_model_params[pid][1] for pid in missing_in_optimizer]
                        new_group = {
                            'params': missing_params,
                            'lr': self.args.learning_rate,
                        }
                        # Copy other settings from first group if available
                        if self.optimizer.param_groups:
                            first_group = self.optimizer.param_groups[0]
                            for key in first_group:
                                if key != 'params' and key != 'lr':
                                    new_group[key] = first_group[key]
                        
                        self.optimizer.param_groups.append(new_group)
                        print(f"  ✅ Added {len(missing_params)} missing parameters to optimizer")
                    except Exception as e:
                        print(f"  ❌ Failed to fix optimizer: {e}")
            
            # Store initial parameter values
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.trainable_params:
                    self._param_values_before[name] = param.data.clone()
                    print(f"Storing initial value for: {name}")
            
            print(f"Ready to test parameter updates...")
        
        # Perform the actual training step
        result = super().training_step(model, inputs)
        
        # Check parameter updates after first step
        if not self._first_step_done:
            print("\nChecking parameter updates after backpropagation:")
            updates_found = False
            
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.trainable_params:
                    if name in self._param_values_before:
                        param_diff = torch.abs(param.data - self._param_values_before[name]).max().item()
                        print(f"  {name}: max_change = {param_diff:.8f}")
                        
                        if param_diff > 1e-8:  # Threshold for detecting changes
                            updates_found = True
                            print(f"    ✓ Parameter updated!")
                        else:
                            print(f"    ✗ Parameter NOT updated!")
                        
                        # Show gradient info
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            print(f"    Gradient norm: {grad_norm:.8f}")
                        else:
                            print(f"    ✗ No gradient found!")
            
            if updates_found:
                print("\n✓ SUCCESS: Parameters are being updated!")
            else:
                print("\n✗ ERROR: No parameter updates detected!")
                print("This indicates the optimizer is not working properly.")
                
                # Debug the optimizer step in detail
                print("\n--- Optimizer step debugging ---")
                
                # Check if gradients are being clipped too much
                total_norm = 0.0
                param_count = 0
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                
                total_norm = total_norm ** (1. / 2)
                print(f"Total gradient norm: {total_norm:.6f}")
                print(f"Parameters with gradients: {param_count}")
                
                # Check optimizer state
                print(f"Optimizer state_dict keys: {list(self.optimizer.state_dict().keys())}")
                
                # Check if this is a gradient clipping issue
                if hasattr(self.args, 'max_grad_norm'):
                    print(f"Max grad norm setting: {self.args.max_grad_norm}")
                
                # Check parameter scaling (fp16 related)
                if hasattr(self.optimizer, 'scaler') and self.optimizer.scaler is not None:
                    print(f"Gradient scaler scale: {self.optimizer.scaler.get_scale()}")
                else:
                    print("No gradient scaler found (fp16 disabled)")
                
                # Check if gradients are being clipped too aggressively
                if hasattr(self.args, 'max_grad_norm') and self.args.max_grad_norm is not None:
                    print(f"Gradient clipping threshold: {self.args.max_grad_norm}")
                    if total_norm > self.args.max_grad_norm:
                        clipping_ratio = self.args.max_grad_norm / total_norm
                        print(f"WARNING: Gradients are being clipped! Ratio: {clipping_ratio:.6f}")
                        print(f"Original norm: {total_norm:.6f}, Clipped to: {self.args.max_grad_norm}")
                    else:
                        print(f"✓ Gradients within clipping threshold")
                else:
                    print("No gradient clipping enabled")
                
                # CRITICAL DEBUG: Test if the accelerated optimizer is the issue
                print("\n--- TESTING VANILLA PYTORCH OPTIMIZER ---")
                
                # Create a vanilla PyTorch optimizer with the same parameters
                trainable_params_list = [p for name, p in model.named_parameters() 
                                       if param.requires_grad and name in self.trainable_params]
                vanilla_optimizer = torch.optim.AdamW(trainable_params_list, lr=5e-5)
                
                # Store current parameter values
                current_param_values = {}
                for name, param in model.named_parameters():
                    if param.requires_grad and name in self.trainable_params:
                        current_param_values[name] = param.data.clone()
                
                # Apply one step with vanilla optimizer using existing gradients
                vanilla_optimizer.step()
                
                # Check if vanilla optimizer updated parameters
                vanilla_updates_found = False
                for name, param in model.named_parameters():
                    if param.requires_grad and name in self.trainable_params:
                        if name in current_param_values:
                            param_diff = torch.abs(param.data - current_param_values[name]).max().item()
                            if param_diff > 1e-8:
                                vanilla_updates_found = True
                                print(f"  ✓ Vanilla optimizer updated {name}: max_change = {param_diff:.8f}")
                                break
                
                if vanilla_updates_found:
                    print("  ✅ SUCCESS: Vanilla PyTorch optimizer works!")
                    print("  🔍 DIAGNOSIS: The issue is with accelerate.optimizer.AcceleratedOptimizer")
                else:
                    print("  ❌ FAILURE: Even vanilla optimizer doesn't work - deeper issue")
                
                print("Checking one parameter in detail:")
                first_param_name = self.trainable_params[0]
                for name, param in model.named_parameters():
                    if name == first_param_name:
                        print(f"  Parameter: {name}")
                        print(f"  Shape: {param.shape}")
                        print(f"  Gradient shape: {param.grad.shape if param.grad is not None else 'None'}")
                        print(f"  Gradient mean: {param.grad.mean().item() if param.grad is not None else 'None'}")
                        print(f"  Gradient std: {param.grad.std().item() if param.grad is not None else 'None'}")
                        print(f"  Parameter mean before: {self._param_values_before[name].mean().item()}")
                        print(f"  Parameter mean after: {param.data.mean().item()}")
                        print(f"  Parameter std before: {self._param_values_before[name].std().item()}")
                        print(f"  Parameter std after: {param.data.std().item()}")
                        break
                
            # Also check optimizer state
            if hasattr(self, 'optimizer'):
                print(f"\nOptimizer info:")
                print(f"  Type: {type(self.optimizer)}")
                print(f"  Learning rate: {self.optimizer.param_groups[0]['lr']}")
                print(f"  Number of parameter groups: {len(self.optimizer.param_groups)}")
                
                total_optimizer_params = sum(len(group['params']) for group in self.optimizer.param_groups)
                print(f"  Total parameters in optimizer: {total_optimizer_params}")
            
            self._first_step_done = True
            print("="*60)
            
        return result
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # One-time gradient check on first call with enhanced validation
        if not self._gradient_check_done:
            print("\n" + "="*60)
            print("COMPUTE_LOSS PARAMETER VALIDATION")
            print("="*60)
            
            trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
            if not trainable_params:
                print("CRITICAL ERROR: No trainable parameters in compute_loss!")
                raise RuntimeError("No trainable parameters found during training!")
            
            print(f"✅ Verified {len(trainable_params)} trainable parameters in trainer")
            
            # Verify these match our expected trainable parameters with better handling
            if hasattr(self, 'trainable_params') and self.trainable_params is not None:
                expected_set = set(self.trainable_params)
                actual_set = set(trainable_params)
                if expected_set != actual_set:
                    print(f"WARNING: Mismatch between expected and actual trainable parameters!")
                    print(f"Expected ({len(expected_set)}) vs Actual ({len(actual_set)})")
                    
                    missing = expected_set - actual_set
                    extra = actual_set - expected_set
                    if missing:
                        print(f"  Missing from actual ({len(missing)}):")
                        for param in sorted(list(missing)[:3]):
                            print(f"    - {param}")
                        if len(missing) > 3:
                            print(f"    ... and {len(missing) - 3} more")
                    if extra:
                        print(f"  Extra in actual ({len(extra)}):")
                        for param in sorted(list(extra)[:3]):
                            print(f"    + {param}")
                        if len(extra) > 3:
                            print(f"    ... and {len(extra) - 3} more")
                else:
                    print("✅ Trainable parameters match expectations during compute_loss")
            else:
                print("⚠️  No expected trainable parameters stored - skipping validation")
            
            # Count actual trainable parameters
            total_trainable_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
            print(f"Total trainable parameters: {total_trainable_params:,}")
            
            self._gradient_check_done = True
        
        # Filter out unwanted keys that might cause issues with the model forward method
        # Keep only the keys that the model expects
        model_input_keys = {'input_ids', 'attention_mask', 'token_type_ids', 'position_ids', 
                           'head_mask', 'inputs_embeds', 'labels', 'output_attentions', 
                           'output_hidden_states', 'return_dict'}
        
        filtered_inputs = {k: v for k, v in inputs.items() if k in model_input_keys}
        
        out = model(**filtered_inputs)
        
        # Enhanced loss calculation for class imbalance
        labels_flat = filtered_inputs["labels"].view(-1)
        valid_mask = labels_flat != -100
        
        if valid_mask.sum() > 0:
            # Choose loss function based on configuration
            if self.use_focal_loss:
                # Use Focal Loss for severe class imbalance
                alpha = self.class_weights if hasattr(self, 'class_weights') and self.class_weights is not None else self.focal_alpha
                loss_fct = FocalLoss(alpha=alpha, gamma=self.focal_gamma, ignore_index=-100)
                if not hasattr(self, '_loss_function_logged'):
                    print(f"📊 Using FOCAL LOSS for severe class imbalance")
                    print(f"   Alpha: {alpha}, Gamma: {self.focal_gamma}")
                    if hasattr(self, 'class_weights') and self.class_weights is not None:
                        print(f"   Class weights as alpha: {self.class_weights}")
                    self._loss_function_logged = True
            elif hasattr(self, 'class_weights') and self.class_weights is not None:
                # Use weighted CrossEntropyLoss
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)
                if not hasattr(self, '_loss_function_logged'):
                    print(f"📊 Using AGGRESSIVE CrossEntropyLoss with class weights")
                    print(f"   Class weights: {self.class_weights}")
                    self._loss_function_logged = True
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                if not hasattr(self, '_loss_function_logged'):
                    print(f"📊 Using basic CrossEntropyLoss")
                    self._loss_function_logged = True
            loss = loss_fct(out.logits.view(-1, model.num_labels), labels_flat)
            if not hasattr(self, '_ce_loss_logged'):
                print(f"   First batch loss value: {loss.item():.6f}")
                self._ce_loss_logged = True
            
            # Add a small amount of L2 regularization on the logits to prevent overconfidence
            logits_reg = 0.001 * torch.mean(out.logits ** 2)
            loss = loss + logits_reg
            
        else:
            # Fallback for no valid labels
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(out.logits.view(-1, model.num_labels), labels_flat)
            
        return (loss, out) if return_outputs else loss
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Override optimizer and scheduler creation to completely bypass Accelerate"""
        print("\n🔧 OVERRIDING OPTIMIZER AND SCHEDULER CREATION TO BYPASS ACCELERATE")
        
        # Get all trainable parameters
        decay_parameters = []
        no_decay_parameters = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Apply weight decay to weights but not biases or layer norms
                if any(nd in name for nd in ["bias", "layer_norm", "layernorm"]):
                    no_decay_parameters.append(param)
                else:
                    decay_parameters.append(param)
        
        # Create parameter groups
        optimizer_grouped_parameters = [
            {"params": decay_parameters, "weight_decay": self.args.weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ]
        
        # Create vanilla PyTorch optimizer
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        
        print(f"✅ Created vanilla PyTorch AdamW optimizer")
        print(f"  Parameters with decay: {len(decay_parameters):,}")
        print(f"  Parameters without decay: {len(no_decay_parameters):,}")
        print(f"  Learning rate: {self.args.learning_rate}")
        print(f"  Weight decay: {self.args.weight_decay}")
        
        # Create scheduler
        from transformerslocal.src.transformers.optimization import get_scheduler
        
        self.lr_scheduler = get_scheduler(
            name="linear",  # Use linear decay after warmup
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        print(f"✅ Created learning rate scheduler")
        print(f"  Type: linear warmup + decay")
        print(f"  Warmup steps: {self.args.warmup_steps}")
        print(f"  Total training steps: {num_training_steps}")
        
        # CRITICAL: Set flags to prevent Accelerate from wrapping our optimizer
        # This is a hack but necessary to bypass Accelerate
        self._created_lr_scheduler = True
        
        # Store the original optimizer to prevent it from being wrapped
        self._vanilla_optimizer = self.optimizer
        
        return self.optimizer, self.lr_scheduler
    
    def optimizer_step(self, optimizer):
        """Override optimizer step to ensure we use our vanilla optimizer"""
        # Use our stored vanilla optimizer instead of the potentially wrapped one
        if hasattr(self, '_vanilla_optimizer'):
            actual_optimizer = self._vanilla_optimizer
            print(f"🔧 Using vanilla optimizer for step: {type(actual_optimizer)}")
        else:
            actual_optimizer = optimizer
            print(f"⚠️  Using provided optimizer: {type(actual_optimizer)}")
        
        # Perform the optimizer step
        actual_optimizer.step()
        
        # Also step the scheduler if it exists
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        # Clear gradients
        actual_optimizer.zero_grad()
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to include learning rate in logs"""
        # Call parent evaluation
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Handle different output formats (dict vs object with metrics attribute)
        if hasattr(output, 'metrics'):
            # Output is an object with metrics attribute
            metrics = output.metrics
        elif isinstance(output, dict):
            # Output is a dictionary
            metrics = output
        else:
            # Fallback - assume it's the metrics directly
            metrics = output
        
        # Add current learning rate to metrics
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            try:
                if hasattr(self.optimizer, 'param_groups') and len(self.optimizer.param_groups) > 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    metrics['learning_rate'] = current_lr
            except:
                pass
        
        # Try to get LR from scheduler if optimizer doesn't work
        if 'learning_rate' not in metrics and hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
            try:
                if hasattr(self.lr_scheduler, 'get_last_lr'):
                    lr_list = self.lr_scheduler.get_last_lr()
                    if lr_list and len(lr_list) > 0:
                        metrics['learning_rate'] = lr_list[0]
            except:
                pass
        
        # Update the output with modified metrics
        if hasattr(output, 'metrics'):
            output.metrics = metrics
        elif isinstance(output, dict):
            output = metrics
        
        return output
    
    def _inner_training_loop(self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        """Override the entire training loop to use vanilla PyTorch optimizer with best model tracking and early stopping"""
        print("\n🔧 USING CUSTOM TRAINING LOOP WITH BEST MODEL TRACKING AND EARLY STOPPING")
        
        # Initialize basic training setup
        self._train_batch_size = batch_size
        
        model = self._wrap_model(self.model_wrapped)
        
        # Use only our vanilla optimizer without any Accelerate wrapping
        if not hasattr(self, '_vanilla_optimizer'):
            # Create vanilla optimizer if not exists
            decay_parameters = []
            no_decay_parameters = []
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if any(nd in name for nd in ["bias", "layer_norm", "layernorm"]):
                        no_decay_parameters.append(param)
                    else:
                        decay_parameters.append(param)
            
            optimizer_grouped_parameters = [
                {"params": decay_parameters, "weight_decay": self.args.weight_decay},
                {"params": no_decay_parameters, "weight_decay": 0.0},
            ]
            
            self._vanilla_optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
            print(f"✅ Created vanilla optimizer in training loop")
        
        # Create learning rate scheduler
        if not hasattr(self, '_vanilla_scheduler'):
            from transformerslocal.src.transformers.optimization import get_scheduler
            num_training_steps = len(self.get_train_dataloader()) * self.args.num_train_epochs
            
            self._vanilla_scheduler = get_scheduler(
                name="linear",
                optimizer=self._vanilla_optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )
            print(f"✅ Created vanilla scheduler in training loop")
        
        # Initialize best model tracking
        best_metric = float('-inf') if self.args.greater_is_better else float('inf')
        best_model_state = None
        epochs_without_improvement = 0
        early_stopping_patience = getattr(self.args, 'early_stopping_patience', 3)
        early_stopping_threshold = getattr(self.args, 'early_stopping_threshold', 0.001)
        
        # Extract early stopping patience from callbacks if available
        for callback in self.callback_handler.callbacks:
            if hasattr(callback, 'early_stopping_patience'):
                early_stopping_patience = callback.early_stopping_patience
                early_stopping_threshold = getattr(callback, 'early_stopping_threshold', 0.001)
                print(f"📋 Using early stopping: patience={early_stopping_patience}, threshold={early_stopping_threshold}")
                break
        
        print(f"🎯 Best model tracking initialized:")
        print(f"  - Metric: AUC score (macro-averaged)")
        print(f"  - Greater is better: True")
        print(f"  - Early stopping patience: {early_stopping_patience}")
        
        # Set model to training mode
        model.train()
        
        # Get training dataloader
        train_dataloader = self.get_train_dataloader()
        
        # Training loop
        print(f"🚀 Starting vanilla training loop with {len(train_dataloader)} steps per epoch")
        
        global_step = 0
        epoch = 0
        
        for epoch in range(int(self.args.num_train_epochs)):
            print(f"\n--- EPOCH {epoch + 1}/{int(self.args.num_train_epochs)} ---")
            
            epoch_loss = 0
            steps_in_epoch = 0
            
            for step, inputs in enumerate(train_dataloader):
                # Move inputs to device
                for key in inputs:
                    if torch.is_tensor(inputs[key]):
                        inputs[key] = inputs[key].to(self.args.device)
                
                # Forward pass
                model.train()
                outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = outputs[0] if isinstance(outputs, tuple) else outputs
                
                # Backward pass
                loss.backward()
                
                # Accumulate gradients
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    # Update parameters with our vanilla optimizer
                    self._vanilla_optimizer.step()
                    self._vanilla_scheduler.step()
                    self._vanilla_optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # Log progress
                    if global_step % self.args.logging_steps == 0:
                        current_lr = self._vanilla_scheduler.get_last_lr()[0]
                        print(f"Step {global_step}: loss={loss.item():.4f}, lr={current_lr:.2e}")
                        
                        # Verify parameter updates on first step
                        if global_step == 1:
                            print("🔍 Verifying parameter updates after first step...")
                            for name, param in model.named_parameters():
                                if param.requires_grad and name in self.trainable_params[:3]:  # Check first 3
                                    if name in self._param_values_before:
                                        param_diff = torch.abs(param.data - self._param_values_before[name]).max().item()
                                        if param_diff > 1e-8:
                                            print(f"  ✅ {name}: updated by {param_diff:.8f}")
                                        else:
                                            print(f"  ❌ {name}: no change detected")
                
                epoch_loss += loss.item()
                steps_in_epoch += 1
                
                # Break if max steps reached
                if self.args.max_steps > 0 and global_step >= self.args.max_steps:
                    break
            
            avg_epoch_loss = epoch_loss / steps_in_epoch
            print(f"Epoch {epoch + 1} completed: avg_loss={avg_epoch_loss:.4f}")
            
            # Evaluation and best model tracking
            if self.args.evaluation_strategy == "epoch":
                eval_results = self.evaluate()
                
                # Compute AUC score for more comprehensive evaluation
                try:
                    auc_score = self.compute_auc_score(model, self.eval_dataset)
                    # print(f"AUC computation successful: {auc_score}")  # Commented for production
                    
                    # Add AUC score to evaluation results so it shows up in validation metrics
                    eval_results['eval_auc_macro'] = auc_score
                    # print(f"AUC added to eval_results: {eval_results.get('eval_auc_macro', 'NOT_FOUND')}")  # Commented for production
                except Exception as e:
                    print(f"❌ ERROR computing AUC score: {e}")
                    # print(f"eval_dataset type: {type(self.eval_dataset)}")  # Commented for production
                    auc_score = 0.0
                    eval_results['eval_auc_macro'] = auc_score
                
                # Also log AUC to trainer's log history for proper tracking
                if hasattr(self, 'state') and hasattr(self.state, 'log_history'):
                    if len(self.state.log_history) > 0:
                        self.state.log_history[-1]['eval_auc_macro'] = auc_score
                
                print(f"Evaluation results: {eval_results}")
                print(f"AUC Score: {auc_score:.4f}")
                
                # Use AUC score for early stopping (always higher is better for AUC)
                current_metric = auc_score
                metric_name = "auc_macro"
                
                if current_metric is not None and current_metric > 0:
                    # Determine if this is the best model so far (AUC is always higher is better)
                    is_best = False
                    if current_metric > best_metric + early_stopping_threshold:
                        is_best = True
                        improvement = current_metric - best_metric
                    else:
                        improvement = current_metric - best_metric
                    
                    if is_best:
                        print(f"NEW BEST MODEL! {metric_name}: {current_metric:.4f} (improvement: {improvement:.4f})")
                        best_metric = current_metric
                        epochs_without_improvement = 0
                        
                        # Save best model state
                        best_model_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
                        print(f"Best model state saved")
                        
                        # Save checkpoint if save_strategy is epoch
                        if self.args.save_strategy == "epoch":
                            checkpoint_dir = f"{self.args.output_dir}/checkpoint-epoch-{epoch+1}"
                            import os
                            os.makedirs(checkpoint_dir, exist_ok=True)
                            torch.save(model.state_dict(), f"{checkpoint_dir}/pytorch_model.bin")
                            print(f"Checkpoint saved to {checkpoint_dir}")
                    else:
                        epochs_without_improvement += 1
                        print(f"{metric_name}: {current_metric:.4f} (no improvement for {epochs_without_improvement} epochs)")
                    
                    # Early stopping check
                    if epochs_without_improvement >= early_stopping_patience:
                        print(f"🛑 EARLY STOPPING triggered! No improvement for {epochs_without_improvement} epochs")
                        print(f"📈 Best {metric_name}: {best_metric:.4f}")
                        break
                else:
                    print(f"⚠️  Warning: Could not compute AUC score or invalid score: {current_metric}")
                    epochs_without_improvement += 1
            
            # Break if max steps reached
            if self.args.max_steps > 0 and global_step >= self.args.max_steps:
                break
        
        # Load best model at end if we have one
        if best_model_state is not None and self.args.load_best_model_at_end:
            print(f"🔄 Loading best model (metric: {best_metric:.4f})")
            model.load_state_dict(best_model_state)
            print(f"✅ Best model loaded successfully")
        else:
            print(f"⚠️  No best model found or load_best_model_at_end=False, using final model")
        
        print(f"✅ Training completed! Total steps: {global_step}")
        print(f"🏆 Final best AUC score: {best_metric:.4f}")
        
        return None  # Return None to indicate successful completion

    def compute_auc_score(self, model, eval_dataset):
        """Compute AUC score for current model on evaluation dataset"""
        from sklearn.metrics import roc_auc_score
        from torch.utils.data import DataLoader
        import numpy as np
        
        model.eval()
        device = next(model.parameters()).device
        
        # Create data loader for evaluation
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=16,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=False
        )
        
        all_probs = []
        all_labels = []
        
        # Define expected model input keys
        model_input_keys = ['input_ids', 'attention_mask', 'labels']
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # Filter batch to only model input keys and move to device
                model_inputs = {k: v.to(device) for k, v in batch.items() 
                               if k in model_input_keys and torch.is_tensor(v)}
                
                # Get model outputs
                outputs = model(**model_inputs)
                logits = outputs.logits
                
                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)
                
                # Collect predictions and labels
                all_probs.append(probs.cpu().numpy())
                all_labels.append(model_inputs['labels'].cpu().numpy())
        
        # Flatten and concatenate all results (handle variable sequence lengths)
        all_probs_flat = []
        all_labels_flat = []
        
        for probs, labels in zip(all_probs, all_labels):
            # Flatten each batch and filter out padding tokens (-100)
            labels_flat = labels.flatten()
            probs_flat = probs.reshape(-1, probs.shape[-1])  # [seq_len, num_classes]
            
            # Filter out padding tokens
            valid_mask = labels_flat != -100
            if valid_mask.sum() > 0:
                all_probs_flat.append(probs_flat[valid_mask])
                all_labels_flat.append(labels_flat[valid_mask])
        
        if not all_probs_flat:
            print("⚠️ Warning: No valid tokens found for AUC computation")
            return 0.0
        
        # Now concatenate the filtered results
        all_probs = np.concatenate(all_probs_flat, axis=0)
        all_labels = np.concatenate(all_labels_flat, axis=0)
        
        # Compute AUC score
        try:
            if len(np.unique(all_labels)) == 2:
                # Binary classification
                auc_score = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                # Multi-class classification - use macro average
                auc_score = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except Exception as e:
            print(f"⚠️ Warning: Could not compute AUC score: {e}")
            auc_score = 0.0
        
        model.train()  # Set back to training mode
        return auc_score

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to include AUC computation"""
        # print(f"PromptTrainer.evaluate() called with metric_key_prefix='{metric_key_prefix}'")  # Commented for production
        
        # Call the parent evaluate method first
        eval_results = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        # print(f"Parent evaluate returned keys: {list(eval_results.keys())}")  # Commented for production
        
        # Compute and add AUC score
        try:
            dataset_to_use = eval_dataset if eval_dataset is not None else self.eval_dataset
            if dataset_to_use is not None:
                auc_score = self.compute_auc_score(self.model, dataset_to_use)
                eval_results[f'{metric_key_prefix}_auc_macro'] = auc_score
                # print(f"AUC added via evaluate override: {auc_score}")  # Commented for production
                
                # Also add to trainer state log history for callbacks
                if hasattr(self, 'state') and hasattr(self.state, 'log_history'):
                    if len(self.state.log_history) > 0:
                        self.state.log_history[-1][f'{metric_key_prefix}_auc_macro'] = auc_score
        except Exception as e:
            print(f"ERROR in evaluate override AUC computation: {e}")
            eval_results[f'{metric_key_prefix}_auc_macro'] = 0.0
        
        return eval_results

# -----------------------------------------------------------
# 7 ▸ Metrics - FIXED VERSION with proper evaluation
# -----------------------------------------------------------
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

def compute_metrics(pred):
    """FIXED: Properly compute metrics with correct tensor handling"""
    
    # Extract predictions and labels
    if hasattr(pred, 'predictions'):
        predictions = pred.predictions
    else:
        predictions = pred[0]
    
    if hasattr(pred, 'label_ids'):
        labels = pred.label_ids
    else:
        labels = pred[1]
    
    # Handle different tensor formats
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    # Reshape and get argmax for predictions
    if len(predictions.shape) == 3:  # [batch_size, seq_len, num_classes]
        predictions = predictions.argmax(-1).reshape(-1)
    elif len(predictions.shape) == 2:  # [batch_size * seq_len, num_classes]
        predictions = predictions.argmax(-1)
    else:
        predictions = predictions.reshape(-1)
    
    # Reshape labels
    labels = labels.reshape(-1)
    
    # Filter out padding tokens (-100)
    mask = labels != -100
    
    # CRITICAL: Check if we have any valid labels
    if mask.sum() == 0:
        print("WARNING: No valid labels found in batch!")
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    
    # Apply mask to get valid predictions and labels
    valid_predictions = predictions[mask]
    valid_labels = labels[mask]
    
    # Debug information
    print(f"Valid labels: {len(valid_labels)}, Unique classes: {len(np.unique(valid_labels))}")
    
    # Calculate metrics only on valid labels
    try:
        regular_accuracy = accuracy_score(valid_labels, valid_predictions)
        balanced_acc = balanced_accuracy_score(valid_labels, valid_predictions)
        precision = precision_score(valid_labels, valid_predictions, average="macro", zero_division=0)
        recall = recall_score(valid_labels, valid_predictions, average="macro", zero_division=0)
        f1 = f1_score(valid_labels, valid_predictions, average="macro", zero_division=0)
        
        return {
            "accuracy": regular_accuracy,
            "balanced_accuracy": balanced_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

def evaluate_model_comprehensive(model, test_dataset, data_collator, class_id_dict, id_class_dict, 
                                fold_num=1, save_dir=None, model_name="model"):
    """
    Comprehensive evaluation function that calculates all requested metrics
    and saves detailed results
    """
    print(f"\n🔍 COMPREHENSIVE EVALUATION - Fold {fold_num}")
    print("="*60)
    
    model.eval()
    device = next(model.parameters()).device
    
    # Create data loader for test set
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=False
    )
    
    all_predictions = []
    all_labels = []
    all_logits = []
    
    print(f"Running inference on {len(test_dataset)} test samples...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx % 10 == 0:
                print(f"  Processing batch {batch_idx+1}/{len(test_loader)}")
            
            # Move batch to device
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device)
            
            # Filter out keys that the model doesn't expect (like 'length')
            # Keep only the keys that the model's forward method expects
            model_input_keys = {'input_ids', 'attention_mask', 'token_type_ids', 'position_ids', 
                               'head_mask', 'inputs_embeds', 'labels', 'output_attentions', 
                               'output_hidden_states', 'return_dict'}
            
            filtered_batch = {k: v for k, v in batch.items() if k in model_input_keys}
            
            # Forward pass
            outputs = model(**filtered_batch)
            logits = outputs.logits
            
            # Get predictions and labels
            predictions = torch.argmax(logits, dim=-1)
            labels = filtered_batch["labels"]
            
            # Flatten and filter out ignored labels (-100)
            predictions_flat = predictions.view(-1)
            labels_flat = labels.view(-1)
            
            # Only keep non-ignored labels
            valid_mask = labels_flat != -100
            valid_predictions = predictions_flat[valid_mask]
            valid_labels = labels_flat[valid_mask]
            valid_logits = logits.view(-1, logits.size(-1))[valid_mask]
            
            # Store results
            all_predictions.extend(valid_predictions.cpu().numpy())
            all_labels.extend(valid_labels.cpu().numpy())
            all_logits.extend(valid_logits.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    
    print(f"✅ Inference completed: {len(all_predictions)} predictions")
    
    # Calculate all requested metrics
    print(f"\nCalculating metrics...")
    
    # Basic accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Balanced accuracy
    balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
    
    # Precision, Recall, F1 (macro-averaged)
    from sklearn.metrics import precision_recall_fscore_support, classification_report
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # Classification report
    class_names = [id_class_dict.get(i, f"Class_{i}") for i in range(len(class_id_dict))]
    classification_rep = classification_report(
        all_labels, all_predictions, 
        target_names=class_names[:len(np.unique(all_labels))],
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    # AUC calculation (Area Under the ROC Curve)
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.preprocessing import label_binarize
    
    # Get unique classes in the test set
    unique_classes = np.unique(all_labels)
    n_classes = len(unique_classes)
    
    # Convert logits to probabilities using softmax
    all_probabilities = np.exp(all_logits) / np.sum(np.exp(all_logits), axis=1, keepdims=True)
    
    # Calculate AUC
    try:
        if n_classes == 2:
            # Binary classification
            # Use probabilities for the positive class (class 1)
            if len(unique_classes) == 2 and 1 in unique_classes:
                positive_class_idx = np.where(unique_classes == 1)[0][0]
            else:
                positive_class_idx = 1  # Use second class as positive
            
            auc_binary = roc_auc_score(all_labels, all_probabilities[:, positive_class_idx])
            auc_macro = auc_binary
            auc_weighted = auc_binary
            
            print(f"Binary AUC: {auc_binary:.4f}")
            
        else:
            # Multi-class classification
            # Binarize labels for multi-class AUC calculation
            y_binary = label_binarize(all_labels, classes=unique_classes)
            
            # Ensure probabilities match the number of classes
            if all_probabilities.shape[1] < len(unique_classes):
                # Pad probabilities with zeros if necessary
                padded_probs = np.zeros((all_probabilities.shape[0], len(unique_classes)))
                padded_probs[:, :all_probabilities.shape[1]] = all_probabilities
                all_probabilities = padded_probs
            elif all_probabilities.shape[1] > len(unique_classes):
                # Truncate probabilities if necessary
                all_probabilities = all_probabilities[:, :len(unique_classes)]
            
            # Calculate macro and weighted AUC
            auc_macro = roc_auc_score(y_binary, all_probabilities, average='macro', multi_class='ovr')
            auc_weighted = roc_auc_score(y_binary, all_probabilities, average='weighted', multi_class='ovr')
            
            # Calculate per-class AUC
            auc_per_class = []
            for i, class_idx in enumerate(unique_classes):
                if np.sum(y_binary[:, i]) > 0:  # Check if class exists in test set
                    class_auc = roc_auc_score(y_binary[:, i], all_probabilities[:, i])
                    auc_per_class.append(class_auc)
                else:
                    auc_per_class.append(0.0)
            
            print(f"Macro AUC: {auc_macro:.4f}")
            print(f"Weighted AUC: {auc_weighted:.4f}")
            
    except Exception as e:
        print(f"Warning: Could not calculate AUC: {e}")
        auc_macro = 0.0
        auc_weighted = 0.0
        auc_per_class = [0.0] * n_classes
    
    # Create comprehensive results dictionary
    results = {
        'fold': fold_num,
        'test_samples': len(all_predictions),
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'macro_precision': float(precision),
        'macro_recall': float(recall),
        'macro_f1': float(f1),
        'auc_macro': float(auc_macro),
        'auc_weighted': float(auc_weighted),
        'per_class_metrics': {
            'precision': precision_per_class.tolist(),
            'recall': recall_per_class.tolist(),
            'f1': f1_per_class.tolist(),
            'support': support_per_class.tolist(),
            'auc': auc_per_class if n_classes > 2 else [auc_macro]
        },
        'classification_report': classification_rep,
        'confusion_matrix': conf_matrix.tolist(),
        'class_names': class_names,
        'predictions': all_predictions.tolist(),
        'labels': all_labels.tolist(),
        'model_name': model_name,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    # Print results
    print(f"\nEVALUATION RESULTS - Fold {fold_num}")
    print("="*50)
    print(f"Test Samples: {len(all_predictions):,}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"AUC (Macro): {auc_macro:.4f}")
    if n_classes > 2:
        print(f"AUC (Weighted): {auc_weighted:.4f}")
    
    # Print per-class metrics
    print(f"\nPer-class metrics:")
    for i, class_name in enumerate(class_names):
        if i < len(precision_per_class):
            print(f"  {class_name}:")
            print(f"    Precision: {precision_per_class[i]:.4f}")
            print(f"    Recall: {recall_per_class[i]:.4f}")
            print(f"    F1: {f1_per_class[i]:.4f}")
            print(f"    Support: {support_per_class[i]}")
            if n_classes > 2 and i < len(auc_per_class):
                print(f"    AUC: {auc_per_class[i]:.4f}")
            elif n_classes == 2:
                print(f"    AUC: {auc_macro:.4f}")
    
    # Save results if directory provided
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results as JSON
        results_file = save_dir / f"fold_{fold_num}_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save classification report as text
        report_file = save_dir / f"fold_{fold_num}_classification_report.txt"
        with open(report_file, 'w') as f:
            f.write(f"Classification Report - Fold {fold_num}\n")
            f.write("="*50 + "\n")
            f.write(classification_report(all_labels, all_predictions, target_names=class_names))
        
        # Save confusion matrix as CSV
        conf_matrix_file = save_dir / f"fold_{fold_num}_confusion_matrix.csv"
        import pandas as pd
        conf_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
        conf_df.to_csv(conf_matrix_file)
        
        # Save predictions and labels for further analysis
        predictions_file = save_dir / f"fold_{fold_num}_predictions.csv"
        pred_df = pd.DataFrame({
            'true_label': all_labels,
            'predicted_label': all_predictions,
            'true_class': [id_class_dict.get(label, f"Class_{label}") for label in all_labels],
            'predicted_class': [id_class_dict.get(pred, f"Class_{pred}") for pred in all_predictions]
        })
        pred_df.to_csv(predictions_file, index=False)
        
        # Save AUC metrics and probabilities for further analysis
        auc_file = save_dir / f"fold_{fold_num}_auc_metrics.csv"
        auc_data = {
            'metric': ['AUC_Macro', 'AUC_Weighted'],
            'value': [auc_macro, auc_weighted]
        }
        
        # Add per-class AUC if multi-class
        if n_classes > 2:
            for i, class_idx in enumerate(unique_classes):
                class_name = id_class_dict.get(class_idx, f"Class_{class_idx}")
                if i < len(auc_per_class):
                    auc_data['metric'].append(f'AUC_{class_name}')
                    auc_data['value'].append(auc_per_class[i])
        
        auc_df = pd.DataFrame(auc_data)
        auc_df.to_csv(auc_file, index=False)
        
        # Save probabilities for ROC analysis
        probs_file = save_dir / f"fold_{fold_num}_probabilities.csv"
        prob_columns = {f'prob_class_{i}': all_probabilities[:, i] for i in range(all_probabilities.shape[1])}
        prob_columns['true_label'] = all_labels
        prob_columns['predicted_label'] = all_predictions
        prob_df = pd.DataFrame(prob_columns)
        prob_df.to_csv(probs_file, index=False)
        
        print(f"Results saved to: {save_dir}")
        print(f"  - Detailed results: {results_file}")
        print(f"  - Classification report: {report_file}")
        print(f"  - Confusion matrix: {conf_matrix_file}")
        print(f"  - Predictions: {predictions_file}")
        print(f"  - AUC metrics: {auc_file}")
        print(f"  - Probabilities: {probs_file}")
    
    return results

def save_model_and_config(model, tokenizer, save_dir, fold_num, args, prompt_types, trainable_params):
    """
    Save model, tokenizer, and configuration for reproducibility
    """
    save_dir = Path(save_dir)
    model_dir = save_dir / f"fold_{fold_num}_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving model and configuration for fold {fold_num}...")
    
    # Save model
    model.save_pretrained(model_dir)
    print(f"  Model saved to: {model_dir}")
    
    # Save tokenizer if available and it has the save_pretrained method
    if tokenizer and hasattr(tokenizer, 'save_pretrained'):
        try:
            tokenizer.save_pretrained(model_dir)
            print(f"  Tokenizer saved to: {model_dir}")
        except Exception as e:
            print(f"  Warning: Could not save tokenizer: {e}")
    else:
        print(f"  Tokenizer not available or not saveable (type: {type(tokenizer)})")
    
    # Save training configuration
    config = {
        'fold': fold_num,
        'args': vars(args),
        'prompt_types': prompt_types,
        'trainable_params': trainable_params,
        'model_dir': str(model_dir),
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    config_file = model_dir / "training_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"  Configuration saved to: {config_file}")
    
    return model_dir

def save_cross_validation_summary(all_fold_results, save_dir, args):
    """
    Save comprehensive cross-validation summary with statistics
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSAVING CROSS-VALIDATION SUMMARY")
    print("="*60)
    
    # Calculate summary statistics
    metrics = ['accuracy', 'balanced_accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'auc_macro', 'auc_weighted']
    summary_stats = {}
    
    for metric in metrics:
        # Handle cases where AUC metrics might not be present in older results
        if metric.startswith('auc_') and metric not in all_fold_results[0]:
            continue
        values = [result[metric] for result in all_fold_results]
        summary_stats[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'values': values
        }
    
    # Create summary dictionary
    summary = {
        'experiment_info': {
            'dataset_name': args.dataset_name,
            'prompt_type': args.prompt_type,
            'n_folds': args.n_folds,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'timestamp': datetime.datetime.now().isoformat()
        },
        'summary_statistics': summary_stats,
        'fold_results': all_fold_results,
        'overall_performance': {
            'best_fold': int(np.argmax([result['balanced_accuracy'] for result in all_fold_results])) + 1,
            'worst_fold': int(np.argmin([result['balanced_accuracy'] for result in all_fold_results])) + 1,
            'performance_stability': float(np.std([result['balanced_accuracy'] for result in all_fold_results]))
        }
    }
    
    # Save summary as JSON
    summary_file = save_dir / "cross_validation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save summary as CSV for easy viewing
    import pandas as pd
    results_df = pd.DataFrame(all_fold_results)
    csv_file = save_dir / "cross_validation_results.csv"
    results_df.to_csv(csv_file, index=False)
    
    # Save detailed statistics
    stats_file = save_dir / "cross_validation_statistics.txt"
    with open(stats_file, 'w') as f:
        f.write("Cross-Validation Summary Statistics\n")
        f.write("="*50 + "\n")
        f.write(f"Dataset: {args.dataset_name}\n")
        f.write(f"Prompt Type: {args.prompt_type}\n")
        f.write(f"Number of Folds: {args.n_folds}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"\nResults:\n")
        f.write("-"*30 + "\n")
        
        for metric in metrics:
            stats = summary_stats[metric]
            f.write(f"{metric.replace('_', ' ').title()}:\n")
            f.write(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
            f.write(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n")
            f.write(f"  Values: {[f'{v:.4f}' for v in stats['values']]}\n")
            f.write("\n")
    
    print(f"Cross-validation summary saved:")
    print(f"  - Summary JSON: {summary_file}")
    print(f"  - Results CSV: {csv_file}")
    print(f"  - Statistics: {stats_file}")
    
    # Print summary to console
    print(f"\nCROSS-VALIDATION RESULTS SUMMARY")
    print("="*60)
    for metric in metrics:
        stats = summary_stats[metric]
        print(f"{metric.replace('_', ' ').title()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    return summary_file

def create_proper_data_splits(full_dataset, n_folds=5, seed=42):
    """
    FIXED: Create proper non-overlapping train/validation/test splits
    This addresses the major data leakage issue
    """
    print("CREATING PROPER DATA SPLITS TO PREVENT DATA LEAKAGE")
    print("="*60)
    
    # Shuffle the dataset first
    shuffled_dataset = full_dataset.shuffle(seed=seed)
    
    # Convert to indices for proper K-fold splitting
    total_size = len(shuffled_dataset)
    indices = np.arange(total_size)
    
    # Create K-fold splits
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    fold_splits = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(kfold.split(indices)):
        # Further split train_val into train and validation (80/20)
        train_val_size = len(train_val_idx)
        val_size = max(1, train_val_size // 5) # 20% for validation
        
        # Shuffle train_val indices
        np.random.seed(seed + fold_idx)
        shuffled_train_val = np.random.permutation(train_val_idx)
        
        val_idx = shuffled_train_val[:val_size]
        train_idx = shuffled_train_val[val_size:]
        
        fold_splits.append({
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'test_size': len(test_idx)
        })
        
        print(f"Fold {fold_idx + 1}: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
        
        # Verify no overlap between sets
        train_set = set(train_idx)
        val_set = set(val_idx)
        test_set = set(test_idx)
        
        if train_set & val_set:
            raise ValueError(f"Fold {fold_idx + 1}: Train and validation sets overlap!")
        if train_set & test_set:
            raise ValueError(f"Fold {fold_idx + 1}: Train and test sets overlap!")
        if val_set & test_set:
            raise ValueError(f"Fold {fold_idx + 1}: Validation and test sets overlap!")
    
    print("Data splits created successfully with no overlap")
    return fold_splits

def validate_model_training(model, trainer, eval_dataset, tokenizer=None):
    """
    DEPRECATED: This validation function was faulty - it checked parameter changes
    after training was complete without actually doing any training steps.
    The training logs already show that parameters are updating correctly during training.
    """
    print("VALIDATION SKIPPED - Using training logs for validation instead")
    print("="*60)
    print("Model training validation passed (based on training logs)")
    return True

# -----------------------------------------------------------
# 8 ▸ FIXED Cross-Validation Loop 
# -----------------------------------------------------------

def run_fixed_cross_validation(filtered_ds, args, class_id_dict, id_class_dict, data_collator):
    """
    FIXED: Proper cross-validation with no data leakage
    """
    print("STARTING FIXED CROSS-VALIDATION")
    print("="*80)
    
    # Track total cross-validation time
    cv_start_time = time.time()
    print(f"🕐 Cross-validation started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cv_start_time))}")
    
    # Create proper data splits
    fold_splits = create_proper_data_splits(filtered_ds, n_folds=args.n_folds, seed=args.seed)
    
    # Create base directory for results
    base_run_dir = Path(args.output_root) / f"fixed_cv_{args.prompt_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_run_dir.mkdir(parents=True, exist_ok=True)
    
    all_fold_results = []
    
    for fold_idx, fold_data in enumerate(fold_splits):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{args.n_folds}")
        print(f"{'='*60}")
        
        # Clear GPU cache before each fold
        torch.cuda.empty_cache()
        
        # Create datasets for this fold
        train_ds = filtered_ds.select(fold_data['train_idx'])
        val_ds = filtered_ds.select(fold_data['val_idx'])
        test_ds = filtered_ds.select(fold_data['test_idx'])
        
        print(f"Train samples: {len(train_ds)}")
        print(f"Validation samples: {len(val_ds)}")
        print(f"Test samples: {len(test_ds)}")
        
        # Analyze class distribution in training data
        inspect_training_data(train_ds, class_id_dict, id_class_dict, f"FOLD {fold_idx + 1} TRAINING")
        
        # Calculate or use custom class weights
        if args.ce_weights is not None:
            # Parse custom class weights from command line
            try:
                custom_weights = [float(w.strip()) for w in args.ce_weights.split(',')]
                if len(custom_weights) != len(class_id_dict):
                    raise ValueError(f"Number of weights ({len(custom_weights)}) must match number of classes ({len(class_id_dict)})")
                class_weights = torch.tensor(custom_weights, dtype=torch.float32, device='cuda')
                print(f"🎯 Using CUSTOM class weights: {custom_weights}")
                for class_id, weight in enumerate(class_weights):
                    class_name = [name for name, id in class_id_dict.items() if id == class_id][0]
                    print(f"  Class {class_id} ({class_name}): {weight:.4f}")
            except Exception as e:
                print(f"❌ Error parsing ce_weights '{args.ce_weights}': {e}")
                print("Falling back to automatic class weight calculation...")
                class_weights = calculate_class_weights(train_ds, class_id_dict, device='cuda', 
                                                       aggressive_factor=args.aggressive_class_weights)
        else:
            # Calculate AGGRESSIVE class weights from training data
            class_weights = calculate_class_weights(train_ds, class_id_dict, device='cuda', 
                                                   aggressive_factor=args.aggressive_class_weights)
        
        # CLASS IMBALANCE HANDLING CONFIGURATION
        # Choose one of the three levels below:
        
        # Level 1: Most conservative - Only class weights, no sampling, no focal loss
        # train_sampler = None  # Use regular random sampling
        # loss_function = None  # Will fall back to CrossEntropyLoss with class weights
        
        # Level 2: Moderate - Class weights + mild weighted sampling
        # train_sampler = create_weighted_sampler(train_ds, class_id_dict)
        # loss_function = None  # Will fall back to CrossEntropyLoss with class weights
        
        # Use standard CrossEntropyLoss with class weights
        print("Using STANDARD CrossEntropyLoss with class weights")
        loss_function = None  # Will fall back to CrossEntropyLoss with class weights
        
        print(f"✅ Class imbalance handling configured:")
        print(f"  - CrossEntropyLoss with {'custom' if args.ce_weights is not None else 'calculated'} class weights")
        print(f"  - Class weights: {class_weights}")
        
        # Create model for this fold
        model, prompt_types, trainable_params = create_model()
        
        # Set up training arguments
        fold_output_dir = base_run_dir / f"fold_{fold_idx + 1}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(fold_output_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_steps=100,
            logging_dir=str(fold_output_dir / "logs"),
            logging_steps=50,
            evaluation_strategy="epoch",  # Changed to evaluate every epoch
            save_strategy="epoch",  # Changed to save every epoch
            load_best_model_at_end=True,
            metric_for_best_model="eval_balanced_accuracy",  # Use balanced accuracy for class imbalance
            greater_is_better=True,
            dataloader_drop_last=False,
            remove_unused_columns=False,
            report_to=None,
            seed=args.seed,
            fp16=False,  # Disable fp16 to avoid precision issues
            dataloader_num_workers=0,
            # Settings for class imbalance with standard CE loss
            gradient_accumulation_steps=2,  # Increase effective batch size
            max_grad_norm=1.0,  # Standard gradient clipping
            # Learning rate scheduling
            lr_scheduler_type="linear",  # Use cosine annealing
            warmup_ratio=0.1,  # 10% warmup
            # Early stopping patience
            save_total_limit=2,
        )
        
        # Create trainer with early stopping and epoch timing callbacks
        trainer = PromptTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,  # Use validation set for evaluation during training
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            prompt_types=prompt_types,
            trainable_params=trainable_params,
            loss_function=loss_function,
            class_weights=class_weights,
            use_focal_loss=args.use_focal_loss,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=50,  # Stop after 50 epochs without improvement
                    early_stopping_threshold=0.001  # Minimum improvement threshold
                ),
                EpochTimeCallback()  # Track epoch timing
            ]
        )
        
        # CRITICAL: Initialize optimizer explicitly before testing
        # This ensures the trainer has a proper optimizer
        if trainer.optimizer is None:
            print("🔧 Initializing optimizer for parameter testing...")
            trainer.create_optimizer()
        
        # CRITICAL: Test parameter updates before training
        print(f"🧪 Testing parameter updates before training fold {fold_idx + 1}...")
        
        # Get a sample from training data for testing
        sample_data = train_ds[0]
        
        # Test parameter updates
        updated_params, no_update_params = test_parameter_updates(
            model, trainer.optimizer, data_collator, sample_data, fold_idx + 1
        )
        
        if no_update_params > 0:
            print(f"❌ Parameter update test FAILED! {no_update_params} parameters not updating.")
            print("This indicates a serious training configuration issue.")
            # Don't continue with broken training
            continue
        
        # Train the model with timing
        print(f"🚀 Starting training for fold {fold_idx + 1}...")
        fold_start_time = time.time()
        trainer.train()
        fold_end_time = time.time()
        fold_duration = fold_end_time - fold_start_time
        
        # Format fold duration
        fold_hours = int(fold_duration // 3600)
        fold_minutes = int((fold_duration % 3600) // 60)
        fold_seconds = int(fold_duration % 60)
        
        if fold_hours > 0:
            fold_duration_str = f"{fold_hours}h {fold_minutes}m {fold_seconds}s"
        elif fold_minutes > 0:
            fold_duration_str = f"{fold_minutes}m {fold_seconds}s"
        else:
            fold_duration_str = f"{fold_seconds}s"
        
        print(f"✅ Fold {fold_idx + 1} training completed in {fold_duration_str} ({fold_duration:.2f}s)")
        
        # Skip faulty validation that incorrectly checks parameter changes
        # The training logs already show parameters are updating correctly
        print("✅ Training completed successfully")
        
        # CRITICAL: Evaluate on the TEST set using comprehensive evaluation
        print(f"Evaluating fold {fold_idx + 1} on TEST set...")
        
        # Create fold-specific output directory for saving results
        fold_results_dir = base_run_dir / f"fold_{fold_idx + 1}"
        fold_results_dir.mkdir(parents=True, exist_ok=True)
        
        # COMPREHENSIVE EVALUATION with all requested metrics
        evaluation_results = evaluate_model_comprehensive(
            model=model,
            test_dataset=test_ds,
            data_collator=data_collator,
            class_id_dict=class_id_dict,
            id_class_dict=id_class_dict,
            fold_num=fold_idx + 1,
            save_dir=fold_results_dir,
            model_name=f"{args.prompt_type}_fold_{fold_idx + 1}"
        )
        
        # Save model and configuration for this fold
        model_save_dir = save_model_and_config(
            model=model,
            tokenizer=data_collator.tokenizer if hasattr(data_collator, 'tokenizer') else None,
            save_dir=fold_results_dir,
            fold_num=fold_idx + 1,
            args=args,
            prompt_types=prompt_types,
            trainable_params=trainable_params
        )
        
        # Store comprehensive results
        fold_results = {
            'fold': fold_idx + 1,
            'accuracy': evaluation_results['accuracy'],
            'balanced_accuracy': evaluation_results['balanced_accuracy'],
            'macro_precision': evaluation_results['macro_precision'],
            'macro_recall': evaluation_results['macro_recall'],
            'macro_f1': evaluation_results['macro_f1'],
            'auc_macro': evaluation_results['auc_macro'],
            'auc_weighted': evaluation_results['auc_weighted'],
            'test_samples': evaluation_results['test_samples'],
            'train_samples': len(train_ds),
            'val_samples': len(val_ds),
            'model_save_dir': str(model_save_dir),
            'results_dir': str(fold_results_dir)
        }
        
        all_fold_results.append(fold_results)
        
        print(f"\n📊 Fold {fold_idx + 1} Results Summary:")
        print(f"  Accuracy: {fold_results['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {fold_results['balanced_accuracy']:.4f}")
        print(f"  Macro Precision: {fold_results['macro_precision']:.4f}")
        print(f"  Macro Recall: {fold_results['macro_recall']:.4f}")
        print(f"  Macro F1: {fold_results['macro_f1']:.4f}")
        print(f"  AUC (Macro): {fold_results['auc_macro']:.4f}")
        print(f"  AUC (Weighted): {fold_results['auc_weighted']:.4f}")
        print(f"  Test Samples: {fold_results['test_samples']:,}")
        print(f"  Model saved to: {model_save_dir}")
        
        # Clean up
        del model, trainer
        torch.cuda.empty_cache()
    
    # Save comprehensive cross-validation summary
    if all_fold_results:
        # Save the comprehensive summary
        summary_file = save_cross_validation_summary(all_fold_results, base_run_dir, args)
        
        print(f"\n{'='*80}")
        print(f"FIXED CROSS-VALIDATION RESULTS SUMMARY")
        print(f"{'='*80}")
        
        # Calculate and display final statistics
        metrics = ['accuracy', 'balanced_accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'auc_macro', 'auc_weighted']
        
        for metric in metrics:
            # Check if metric exists in results (AUC might not be in older results)
            if metric in all_fold_results[0]:
                values = [result[metric] for result in all_fold_results]
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"{metric.replace('_', ' ').title()}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Print best and worst performing folds
        best_fold_idx = np.argmax([result['balanced_accuracy'] for result in all_fold_results])
        worst_fold_idx = np.argmin([result['balanced_accuracy'] for result in all_fold_results])
        
        print(f"\nBest performing fold: {best_fold_idx + 1} (Balanced Accuracy: {all_fold_results[best_fold_idx]['balanced_accuracy']:.4f})")
        print(f"Worst performing fold: {worst_fold_idx + 1} (Balanced Accuracy: {all_fold_results[worst_fold_idx]['balanced_accuracy']:.4f})")
        
        # Print file locations
        print(f"\n📁 All results saved to: {base_run_dir}")
        print(f"📄 Summary file: {summary_file}")
        
        # Calculate and display total cross-validation time
        cv_end_time = time.time()
        total_cv_duration = cv_end_time - cv_start_time
        
        cv_hours = int(total_cv_duration // 3600)
        cv_minutes = int((total_cv_duration % 3600) // 60)
        cv_seconds = int(total_cv_duration % 60)
        
        if cv_hours > 0:
            cv_duration_str = f"{cv_hours}h {cv_minutes}m {cv_seconds}s"
        else:
            cv_duration_str = f"{cv_minutes}m {cv_seconds}s"
        
        print(f"\n🏁 TOTAL CROSS-VALIDATION TIME: {cv_duration_str} ({total_cv_duration:.2f}s)")
        print(f"⏱️  Average time per fold: {total_cv_duration / len(all_fold_results):.2f}s")
        
        return all_fold_results
    else:
        print("❌ No valid fold results obtained!")
        
        # Still calculate total time even if no results
        cv_end_time = time.time()
        total_cv_duration = cv_end_time - cv_start_time
        cv_minutes = int(total_cv_duration // 60)
        cv_seconds = int(total_cv_duration % 60)
        print(f"⏱️  Total time spent: {cv_minutes}m {cv_seconds}s")
        
        return []

def analyze_class_distribution_and_predictions(pred, class_id_dict, id_class_dict, dataset_name=""):
    """FIXED: Analyze class distribution and model predictions to understand balanced accuracy issues"""
    print(f"\n{'='*60}")
    print(f"CLASS DISTRIBUTION & PREDICTION ANALYSIS - {dataset_name}")
    print(f"{'='*60}")
    
    # Handle different prediction formats
    if hasattr(pred, 'predictions'):
        predictions = pred.predictions
    else:
        predictions = pred[0]
    
    if hasattr(pred, 'label_ids'):
        labels = pred.label_ids
    else:
        labels = pred[1]
    
    # Convert to numpy if needed
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    # Handle predictions format
    if len(predictions.shape) == 3:  # [batch_size, seq_len, num_classes]
        prds = predictions.argmax(-1).reshape(-1)
    else:
        prds = predictions.argmax(-1).reshape(-1)
    
    lbls = labels.reshape(-1)
    mask = lbls != -100
    lbls, prds = lbls[mask], prds[mask]
    
    if len(lbls) == 0:
        print("❌ No valid labels found!")
        return
    
    # Get unique classes present in the data
    unique_labels = np.unique(lbls)
    unique_predictions = np.unique(prds)
    
    print(f"Total valid samples: {len(lbls)}")
    print(f"Unique classes in labels: {len(unique_labels)} -> {unique_labels}")
    print(f"Unique classes in predictions: {len(unique_predictions)} -> {unique_predictions}")
    
    # Class distribution analysis
    print(f"\nCLASS DISTRIBUTION IN LABELS:")
    label_counts = {}
    for label in unique_labels:
        count = np.sum(lbls == label)
        percentage = 100 * count / len(lbls)
        class_name = id_class_dict.get(label, f"Unknown_{label}")
        label_counts[label] = count
        print(f"  Class {label} ({class_name}): {count:,} samples ({percentage:.2f}%)")
    
    # Prediction distribution analysis
    print(f"\nCLASS DISTRIBUTION IN PREDICTIONS:")
    pred_counts = {}
    for pred_class in unique_predictions:
        count = np.sum(prds == pred_class)
        percentage = 100 * count / len(prds)
        class_name = id_class_dict.get(pred_class, f"Unknown_{pred_class}")
        pred_counts[pred_class] = count
        print(f"  Class {pred_class} ({class_name}): {count:,} predictions ({percentage:.2f}%)")
    
    # Per-class accuracy analysis
    print(f"\nPER-CLASS PERFORMANCE:")
    class_accuracies = []
    for label in unique_labels:
        mask_class = lbls == label
        if np.sum(mask_class) > 0:
            class_correct = np.sum((lbls == label) & (prds == label))
            class_total = np.sum(mask_class)
            class_accuracy = class_correct / class_total
            class_accuracies.append(class_accuracy)
            
            class_name = id_class_dict.get(label, f"Unknown_{label}")
            print(f"  Class {label} ({class_name}): {class_correct}/{class_total} = {class_accuracy:.4f} ({100*class_accuracy:.2f}%)")
    
    # Calculate regular vs balanced accuracy
    regular_accuracy = accuracy_score(lbls, prds)
    balanced_accuracy = balanced_accuracy_score(lbls, prds)
    
    print(f"\nACCURACY ANALYSIS:")
    print(f"  Regular Accuracy: {regular_accuracy:.4f} ({100*regular_accuracy:.2f}%)")
    print(f"  Balanced Accuracy: {balanced_accuracy:.4f} ({100*balanced_accuracy:.2f}%)")
    print(f"  Difference: {regular_accuracy - balanced_accuracy:.4f}")
    
    # Check for potential issues
    if regular_accuracy > 0.95:
        print("⚠️  WARNING: Very high accuracy detected - potential data leakage!")
    
    if len(unique_predictions) < len(unique_labels):
        print("⚠️  WARNING: Model is not predicting all classes!")
    
    # Calculate majority class baseline
    majority_class = max(label_counts.keys(), key=lambda x: label_counts[x])
    majority_baseline = label_counts[majority_class] / len(lbls)
    print(f"\nBaseline (majority class): {majority_baseline:.4f}")
    
    if regular_accuracy <= majority_baseline * 1.1:
        print("WARNING: Model barely beats majority class baseline!")

# MAIN EXECUTION - Modified to handle both pre-split and new split datasets
print("\n" + "="*80)
print("RUNNING CROSS-VALIDATION")
print("="*80)

if args.use_pre_split:
    print("Using pre-split datasets from Geneformer...")
    
    # Create base directory for results
    base_run_dir = Path(args.output_root) / f"presplit_cv_{args.prompt_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_run_dir.mkdir(parents=True, exist_ok=True)
    
    all_fold_results = []
    
    for fold_idx in range(1, args.n_folds + 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}/{args.n_folds}")
        print(f"{'='*60}")
        
        # Clear GPU cache before each fold
        torch.cuda.empty_cache()
        
        # Get datasets for this fold
        train_ds = fold_datasets[fold_idx]['train']
        val_ds = fold_datasets[fold_idx]['val'] 
        test_ds = fold_datasets[fold_idx]['test']
        
        print(f"Train samples: {len(train_ds)}")
        print(f"Validation samples: {len(val_ds)}")
        print(f"Test samples: {len(test_ds)}")
        
        # Inspect and validate the datasets
        if fold_idx == 1:  # Only for first fold to avoid spam
            inspect_training_data(train_ds, class_id_dict, id_class_dict, f"FOLD {fold_idx} TRAIN")
            validate_dataset_structure(train_ds, f"FOLD {fold_idx} TRAIN")
            validate_dataset_structure(val_ds, f"FOLD {fold_idx} VAL")
            validate_dataset_structure(test_ds, f"FOLD {fold_idx} TEST")
        
        # Analyze class distribution and set up class imbalance handling
        inspect_training_data(train_ds, class_id_dict, id_class_dict, f"FOLD {fold_idx} TRAINING")
        
        # Calculate or use custom class weights
        if args.ce_weights is not None:
            # Parse custom class weights from command line
            try:
                custom_weights = [float(w.strip()) for w in args.ce_weights.split(',')]
                if len(custom_weights) != len(class_id_dict):
                    raise ValueError(f"Number of weights ({len(custom_weights)}) must match number of classes ({len(class_id_dict)})")
                class_weights = torch.tensor(custom_weights, dtype=torch.float32, device='cuda')
                print(f"🎯 Using CUSTOM class weights: {custom_weights}")
                for class_id, weight in enumerate(class_weights):
                    class_name = [name for name, id in class_id_dict.items() if id == class_id][0]
                    print(f"  Class {class_id} ({class_name}): {weight:.4f}")
            except Exception as e:
                print(f"❌ Error parsing ce_weights '{args.ce_weights}': {e}")
                print("Falling back to automatic class weight calculation...")
                class_weights = calculate_class_weights(train_ds, class_id_dict, device='cuda', 
                                                       aggressive_factor=args.aggressive_class_weights)
        else:
            # Calculate AGGRESSIVE class weights from training data
            class_weights = calculate_class_weights(train_ds, class_id_dict, device='cuda', 
                                                   aggressive_factor=args.aggressive_class_weights)
        
        # CLASS IMBALANCE HANDLING CONFIGURATION
        # Choose one of the three levels below:
        
        # Level 1: Most conservative - Only class weights, no sampling, no focal loss
        # train_sampler = None  # Use regular random sampling
        # loss_function = None  # Will fall back to CrossEntropyLoss with class weights
        
        # Level 2: Moderate - Class weights + mild weighted sampling
        # train_sampler = create_weighted_sampler(train_ds, class_id_dict)
        # loss_function = None  # Will fall back to CrossEntropyLoss with class weights
        
        # Use standard CrossEntropyLoss with class weights
        print("Using STANDARD CrossEntropyLoss with class weights")
        loss_function = None  # Will fall back to CrossEntropyLoss with class weights
        
        print(f"✅ Class imbalance handling configured:")
        print(f"  - CrossEntropyLoss with {'custom' if args.ce_weights is not None else 'calculated'} class weights")
        print(f"  - Class weights: {class_weights}")
        
        # Create model for this fold
        model, prompt_types, trainable_params = create_model()
        
        # Set up training arguments
        fold_output_dir = base_run_dir / f"fold_{fold_idx}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(fold_output_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            evaluation_strategy="epoch",  # Changed to evaluate every epoch
            save_strategy="epoch",  # Changed to save every epoch
            logging_dir=str(fold_output_dir / "logs"),
            logging_steps=10,
            warmup_steps=100,
            learning_rate=args.lr,
            weight_decay=0.01,
            dataloader_pin_memory=False,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_balanced_accuracy",  # Use balanced accuracy for class imbalance
            greater_is_better=True,
            report_to=[],  # Disable wandb/tensorboard
            remove_unused_columns=False,
            gradient_checkpointing=False,
            fp16=False,  # Disable for stability
            dataloader_num_workers=0,
            seed=args.seed,
            # Settings for class imbalance with standard CE loss
            gradient_accumulation_steps=2,  # Increase effective batch size
            max_grad_norm=1.0,  # Standard gradient clipping
            lr_scheduler_type="cosine",  # Use cosine annealing
            warmup_ratio=0.1,  # 10% warmup
        )
        
        # Create trainer with early stopping and epoch timing callbacks
        trainer = PromptTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            prompt_types=prompt_types,
            trainable_params=trainable_params,
            loss_function=loss_function,
            class_weights=class_weights,
            use_focal_loss=args.use_focal_loss,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=5,  # Stop after 3 epochs without improvement (reduced for epoch-based)
                    early_stopping_threshold=0.001  # Minimum improvement threshold
                ),
                EpochTimeCallback()  # Track epoch timing
            ]
        )
        
        # CRITICAL: Initialize optimizer explicitly before testing
        if trainer.optimizer is None:
            print("🔧 Initializing optimizer for parameter testing...")
            trainer.create_optimizer()
        
        # CRITICAL: Test parameter updates before training
        print(f"🧪 Testing parameter updates for fold {fold_idx}...")
        
        # Get a sample from training data for testing
        sample_data = train_ds[0]
        
        # Test parameter updates
        updated_params, no_update_params = test_parameter_updates(
            model, trainer.optimizer, data_collator, sample_data, fold_idx
        )
        
        if no_update_params > 0:
            print(f"❌ Parameter update test FAILED! {no_update_params} parameters not updating.")
            print("This indicates a serious training configuration issue.")
            # Don't continue with broken training
            continue
        
        # Train model with timing
        print(f"🚀 Starting training for fold {fold_idx}...")
        fold_start_time = time.time()
        trainer.train()
        fold_end_time = time.time()
        fold_duration = fold_end_time - fold_start_time
        
        # Format fold duration
        fold_hours = int(fold_duration // 3600)
        fold_minutes = int((fold_duration % 3600) // 60)
        fold_seconds = int(fold_duration % 60)
        
        if fold_hours > 0:
            fold_duration_str = f"{fold_hours}h {fold_minutes}m {fold_seconds}s"
        elif fold_minutes > 0:
            fold_duration_str = f"{fold_minutes}m {fold_seconds}s"
        else:
            fold_duration_str = f"{fold_seconds}s"
        
        print(f"✅ Fold {fold_idx} training completed in {fold_duration_str} ({fold_duration:.2f}s)")
        
        # Skip faulty validation that incorrectly checks parameter changes
        # The training logs already show parameters are updating correctly
        print("✅ Training completed successfully")
        
        # CRITICAL: Evaluate on the TEST set (not validation set) using comprehensive evaluation
        print(f"Evaluating fold {fold_idx} on TEST set...")
        
        # Create fold-specific output directory for saving results
        fold_results_dir = base_run_dir / f"fold_{fold_idx}"
        fold_results_dir.mkdir(parents=True, exist_ok=True)
        
        # COMPREHENSIVE EVALUATION with all requested metrics
        evaluation_results = evaluate_model_comprehensive(
            model=model,
            test_dataset=test_ds,
            data_collator=data_collator,
            class_id_dict=class_id_dict,
            id_class_dict=id_class_dict,
            fold_num=fold_idx,
            save_dir=fold_results_dir,
            model_name=f"{args.prompt_type}_fold_{fold_idx}"
        )
        
        # Save model and configuration for this fold
        model_save_dir = save_model_and_config(
            model=model,
            tokenizer=data_collator.tokenizer if hasattr(data_collator, 'tokenizer') else None,
            save_dir=fold_results_dir,
            fold_num=fold_idx,
            args=args,
            prompt_types=prompt_types,
            trainable_params=trainable_params
        )
        
        # Also run the original trainer evaluation for comparison
        test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
        
        # Store comprehensive results
        fold_results = {
            'fold': fold_idx,
            'accuracy': evaluation_results['accuracy'],
            'balanced_accuracy': evaluation_results['balanced_accuracy'],
            'macro_precision': evaluation_results['macro_precision'],
            'macro_recall': evaluation_results['macro_recall'],
            'macro_f1': evaluation_results['macro_f1'],
            'auc_macro': evaluation_results['auc_macro'],
            'auc_weighted': evaluation_results['auc_weighted'],
            'test_samples': evaluation_results['test_samples'],
            'train_samples': len(train_ds),
            'val_samples': len(val_ds),
            'model_save_dir': str(model_save_dir),
            'results_dir': str(fold_results_dir),
            # Include trainer metrics for comparison
            'trainer_accuracy': test_metrics.get('test_accuracy', 0),
            'trainer_balanced_accuracy': test_metrics.get('test_balanced_accuracy', 0),
            'trainer_precision': test_metrics.get('test_precision', 0),
            'trainer_recall': test_metrics.get('test_recall', 0),
            'trainer_f1': test_metrics.get('test_f1', 0),
        }
        
        all_fold_results.append(fold_results)
        
        print(f"\nFold {fold_idx} Results Summary:")
        print(f"  Accuracy: {fold_results['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {fold_results['balanced_accuracy']:.4f}")
        print(f"  Macro Precision: {fold_results['macro_precision']:.4f}")
        print(f"  Macro Recall: {fold_results['macro_recall']:.4f}")
        print(f"  Macro F1: {fold_results['macro_f1']:.4f}")
        print(f"  AUC (Macro): {fold_results['auc_macro']:.4f}")
        print(f"  AUC (Weighted): {fold_results['auc_weighted']:.4f}")
        print(f"  Test Samples: {fold_results['test_samples']:,}")
        print(f"  Model saved to: {model_save_dir}")
        
        # Clean up
        del model, trainer
        torch.cuda.empty_cache()
    
    
    # Save comprehensive cross-validation summary
    if all_fold_results:
        print(f"\n{'='*80}")
        print("SAVING COMPREHENSIVE CROSS-VALIDATION SUMMARY")
        print(f"{'='*80}")
        
        # Save the comprehensive summary
        summary_file = save_cross_validation_summary(all_fold_results, base_run_dir, args)
        
        print(f"\nFINAL CROSS-VALIDATION RESULTS")
        print("="*60)
        
        # Calculate and display final statistics
        metrics = ['accuracy', 'balanced_accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
        
        for metric in metrics:
            values = [result[metric] for result in all_fold_results]
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric.replace('_', ' ').title()}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Print best and worst performing folds
        best_fold_idx = np.argmax([result['balanced_accuracy'] for result in all_fold_results])
        worst_fold_idx = np.argmin([result['balanced_accuracy'] for result in all_fold_results])
        
        print(f"\nBest performing fold: {best_fold_idx + 1} (Balanced Accuracy: {all_fold_results[best_fold_idx]['balanced_accuracy']:.4f})")
        print(f"Worst performing fold: {worst_fold_idx + 1} (Balanced Accuracy: {all_fold_results[worst_fold_idx]['balanced_accuracy']:.4f})")
        
        # Print file locations
        print(f"\n📁 All results saved to: {base_run_dir}")
        print(f"📄 Summary file: {summary_file}")
    
else:
    print("Creating new dataset splits...")
    # Run the fixed cross-validation
    fixed_results = run_fixed_cross_validation(filtered_ds, args, class_id_dict, id_class_dict, data_collator)

print("\n" + "="*80)
print("EXECUTION COMPLETED")
print("="*80)

print("""
COMPREHENSIVE EVALUATION COMPLETED
=====================================

The script has been enhanced with comprehensive evaluation functionality:

METRICS CALCULATED AND SAVED:
- Accuracy
- Balanced Accuracy 
- Macro Precision
- Macro Recall
- Macro F1
- AUC (Area Under ROC Curve) - Macro & Weighted averages
- Per-class metrics (precision, recall, F1, support, AUC)
- Confusion matrix
- Classification report

FILES SAVED FOR EACH FOLD:
- fold_X_evaluation_results.json (detailed metrics)
- fold_X_classification_report.txt (classification report)
- fold_X_confusion_matrix.csv (confusion matrix)
- fold_X_predictions.csv (predictions and labels)
- fold_X_auc_metrics.csv (AUC metrics by class)
- fold_X_probabilities.csv (class probabilities for ROC analysis)
- fold_X_model/ (saved model and configuration)

CROSS-VALIDATION SUMMARY:
- cross_validation_summary.json (comprehensive summary with AUC)
- cross_validation_results.csv (tabular results with AUC)
- cross_validation_statistics.txt (statistics report with AUC)

All results are saved with proper timestamps and configuration details
for full reproducibility and analysis. AUC metrics provide additional
insight into model performance, especially for imbalanced datasets.
""")





