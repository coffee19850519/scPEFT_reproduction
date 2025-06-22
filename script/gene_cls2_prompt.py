import sys
import os

# Debug current paths
print("Current working directory:", os.getcwd())
print("Parent directory:", os.path.dirname(os.getcwd()))

# Add the transformerslocal directory to the Python path
transformers_path = os.path.join(os.path.dirname(os.getcwd()), 'transformerslocal')
print("Transformers path:", transformers_path)
print("Path exists:", os.path.exists(transformers_path))

if transformers_path not in sys.path:
    sys.path.insert(0, transformers_path)

print("Updated sys.path:")
for path in sys.path[:5]:  # Show first 5 paths
    print(f"  {path}")


parent_path = os.path.dirname(os.getcwd())
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)
print(f"Added parent path: {parent_path}")

from transformerslocal.src.transformers.models.bert.modeling_bert import BertForTokenClassification
from transformerslocal.src.transformers import EarlyStoppingCallback

from transformers import Trainer
from transformers.training_args import TrainingArguments

# Import DataCollatorForGeneClassification from the original Geneformer directory
import sys
geneformer_path = "/fs/scratch/PCON0022/ch/scPEFT_reproduction/geneformer_peft"
if geneformer_path not in sys.path:
    sys.path.insert(0, geneformer_path)

# Import directly from the collator module to avoid name conflicts
from geneformer.collator_for_classification import DataCollatorForGeneClassification

import argparse, json, yaml, pickle, random, datetime, pathlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch, torch.nn as nn
from datasets import load_from_disk

import loralib as lora
import pandas as pd
from sklearn.model_selection import KFold


print("Import successful with parent path!")

#!/usr/bin/env python
# gene_prompt_trainer.py
# -----------------------------------------------------------
# Fine-tune a prompt-enabled Geneformer model for GENE
# classification using 5-fold cross validation.
# -----------------------------------------------------------


# -----------------------------------------------------------
# 1 ▸ CLI
# -----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_file", default="/fs/scratch/PCON0022/ch/Geneformer/examples/example_input_files/gc-30M_sample50k.dataset")
parser.add_argument("--gene_class_dict", default="/fs/scratch/PCON0022/ch/Geneformer/examples/example_input_files/dosage_sensitivity_TFs.pickle")
parser.add_argument("--token_dict", 
                default="/fs/scratch/PCON0022/ch/scPEFT_reproduction/Geneformer/geneformer/token_dictionary_gc95M.pkl")
                # default="/fs/scratch/PCON0022/ch/scPEFT_reproduction/Geneformer/geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl")
parser.add_argument("--ckpt_dir", 
                default="/fs/scratch/PCON0022/ch/scPEFT_reproduction/geneformer_peft/Pretrain_ckpts/Pretrain_ckpts/geneformer-12L-30M-prompt")
                # default="/fs/scratch/PCON0022/ch/Geneformer/gf-6L-30M-i2048")
parser.add_argument("--output_root", default="/fs/scratch/PCON0022/ch/scPEFT_reproduction/geneformer_peft/example_py/outputs")

parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=8)  # Reduced from 32 to 8
parser.add_argument("--lr",        type=float, default=5e-5)
parser.add_argument("--seed",      type=int, default=42)
parser.add_argument("--n_folds",   type=int, default=5)
args = parser.parse_args('')

torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

# Clear GPU cache before starting
torch.cuda.empty_cache()

# -----------------------------------------------------------
# 2 ▸ Load data & prepare for K-fold cross validation
# -----------------------------------------------------------
full_ds = load_from_disk(args.dataset_file).shuffle(seed=args.seed)  # one .dataset only

# -----------------------------------------------------------
# 3 ▸ Dict helpers
# -----------------------------------------------------------
def load_dict(pth):
    p = pathlib.Path(pth)
    with open(pth, "rb" if p.suffix == ".pkl" or p.suffix == ".pickle" else "r") as f:
        return (
            pickle.load(f) if p.suffix == ".pkl" or p.suffix == ".pickle"
            else json.load(f) if p.suffix == ".json"
            else yaml.safe_load(f)
        )

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

# -----------------------------------------------------------
# 4 ▸ Collator 
# -----------------------------------------------------------
# Fix: Create the collator without token_dictionary parameter and patch it afterward
data_collator = DataCollatorForGeneClassification()

# Update the global token_dictionary in the collator modules to use our token_dict
import sys
for module_name in list(sys.modules.keys()):
    if 'geneformer' in module_name and hasattr(sys.modules[module_name], 'token_dictionary'):
        sys.modules[module_name].token_dictionary = token_dict

# Also update the tokenizer within the data_collator if it exists
if hasattr(data_collator.tokenizer, 'token_dictionary'):
    data_collator.tokenizer.token_dictionary = token_dict

# Update special token IDs in the precollator
data_collator.tokenizer.mask_token_id = token_dict.get("<mask>")
data_collator.tokenizer.pad_token_id = token_dict.get("<pad>")
data_collator.tokenizer.all_special_ids = [
    token_dict.get("<mask>"),
    token_dict.get("<pad>")
]

print("Data collator initialized and token dictionary updated.")

# -----------------------------------------------------------
# 5 ▸ Model creation function
# -----------------------------------------------------------
def create_model():
    """Create a fresh model for each fold"""
    model = BertForTokenClassification.from_pretrained(
        args.ckpt_dir,
        num_labels=len(class_id_dict),
        ignore_mismatched_sizes=True,
    ).to("cuda")

    prompt_types = [p.strip() for p in model.config.prompt_type.split(",") if p.strip()]
    print(f"Prompt types: {prompt_types}")

    if "lora" in prompt_types:
        lora.mark_only_lora_as_trainable(model, bias="lora_only")
        for name, para in model.named_parameters():
            if "lora_key" in name:
                para.requires_grad = False
            if "classifier" in name:
                para.requires_grad = True

    if "Gene_token_prompt" in prompt_types:
        for n,p in model.named_parameters():
            p.requires_grad = ("bert.adapter" in n) or ("classifier" in n)
    if "encoder_prompt" in prompt_types:
        for n,p in model.named_parameters():
            p.requires_grad = (("Space_Adapter" in n) or ("MLP_Adapter" in n) or ("classifier" in n))
    if "prefix_prompt" in prompt_types:
        for n,p in model.named_parameters():
            p.requires_grad = ("prompt_embeddings" in n) or ("classifier" in n)
    
    return model, prompt_types

# -----------------------------------------------------------
# 6 ▸ Trainer subclass – prefix-mask & token-level CE loss
# -----------------------------------------------------------
class PromptTrainer(Trainer):
    def __init__(self,*a,prompt_types=None,**kw):
        super().__init__(*a,**kw); self.prompt_types=prompt_types
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Filter out unwanted keys that might cause issues with the model forward method
        # Keep only the keys that the model expects
        model_input_keys = {'input_ids', 'attention_mask', 'token_type_ids', 'position_ids', 
                           'head_mask', 'inputs_embeds', 'labels', 'output_attentions', 
                           'output_hidden_states', 'return_dict'}
        
        filtered_inputs = {k: v for k, v in inputs.items() if k in model_input_keys}
        
        if "prefix_prompt" in self.prompt_types:
            bs = filtered_inputs["input_ids"].size(0)
            num_prefix_tokens = model.config.num_token
            
            # Create prefix tokens for attention mask
            pre = torch.ones(bs, num_prefix_tokens, device=filtered_inputs["input_ids"].device)
            filtered_inputs["attention_mask"] = torch.cat(
                (torch.cat((filtered_inputs["attention_mask"][:,:1], pre), dim=1),
                 filtered_inputs["attention_mask"][:,1:]), dim=1)
            
            # Also need to pad the labels with -100 for the prefix tokens
            prefix_labels = torch.full((bs, num_prefix_tokens), -100, 
                                     device=filtered_inputs["labels"].device,
                                     dtype=filtered_inputs["labels"].dtype)
            filtered_inputs["labels"] = torch.cat(
                (torch.cat((filtered_inputs["labels"][:,:1], prefix_labels), dim=1),
                 filtered_inputs["labels"][:,1:]), dim=1)
        
        out = model(**filtered_inputs)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(out.logits.view(-1, model.num_labels),
                        filtered_inputs["labels"].view(-1))
        return (loss, out) if return_outputs else loss

# -----------------------------------------------------------
# 7 ▸ Metrics - Enhanced with clear balanced accuracy reporting
# -----------------------------------------------------------
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, accuracy_score
def compute_metrics(pred):
    lbls = pred.label_ids.reshape(-1)
    prds = pred.predictions.argmax(-1).reshape(-1)
    
    # Ensure predictions and labels have the same shape
    min_len = min(len(lbls), len(prds))
    lbls = lbls[:min_len]
    prds = prds[:min_len]
    
    mask = lbls != -100
    lbls, prds = lbls[mask], prds[mask]
    
    # Check if we have any valid labels after filtering
    if len(lbls) == 0:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    
    # Calculate both regular and balanced accuracy
    regular_accuracy = accuracy_score(lbls, prds)
    balanced_acc = balanced_accuracy_score(lbls, prds)
    
    return {
        "accuracy": regular_accuracy,
        "balanced_accuracy": balanced_acc,
        "precision": precision_score(lbls, prds, average="macro", zero_division=0),
        "recall": recall_score(lbls, prds, average="macro", zero_division=0),
        "f1": f1_score(lbls, prds, average="macro", zero_division=0),
    }

# -----------------------------------------------------------
# 8 ▸ K-Fold Cross Validation
# -----------------------------------------------------------

# Create output directory
base_run_dir = Path(args.output_root) / "5fold_cv" / Path(args.dataset_file).stem / datetime.datetime.now().strftime("%y%m%d_%H%M%S")
base_run_dir.mkdir(parents=True, exist_ok=True)

# Initialize KFold
kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

# Store results for all folds
all_fold_results = []
fold_test_metrics = []

print(f"\nStarting {args.n_folds}-fold cross validation...")
print(f"Total samples: {len(filtered_ds)}")

# Convert dataset to indices for KFold
dataset_indices = list(range(len(filtered_ds)))

for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset_indices)):
    print(f"\n{'='*60}")
    print(f"FOLD {fold + 1}/{args.n_folds}")
    print(f"{'='*60}")
    print(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
    
    # Clear GPU cache before each fold
    torch.cuda.empty_cache()
    
    # Create train and test datasets for this fold
    train_ds = filtered_ds.select(train_idx)
    test_ds = filtered_ds.select(test_idx)
    
    # Split training data to get validation set (80/20 split of training data)
    train_val_split = train_ds.train_test_split(test_size=0.2, seed=args.seed + fold)
    train_fold_ds = train_val_split["train"]
    eval_fold_ds = train_val_split["test"]
    
    print(f"Final split - Train: {len(train_fold_ds)}, Val: {len(eval_fold_ds)}, Test: {len(test_ds)}")
    
    # Create fresh model for this fold
    model, prompt_types = create_model()
    def count_trainable(model):
        total = sum(p.numel() for p in model.parameters())
        train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable / total params: {train:,} / {total:,}")


    count_trainable(model)
    
    # Create fold-specific output directory
    fold_dir = base_run_dir / f"fold_{fold + 1}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(fold_dir)
    
    # Training arguments for this fold with memory optimizations
    training_args = TrainingArguments(
        output_dir=str(fold_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        warmup_steps=500,
        report_to="none",
        logging_dir=str(fold_dir / "logs"),
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        fp16=True,  # Use mixed precision training
        dataloader_num_workers=0,  # Reduce memory usage
        remove_unused_columns=False,  # Ensure labels are preserved
        dataloader_pin_memory=False,  # Reduce memory usage
    )
    
    # Create trainer for this fold
    trainer = PromptTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_fold_ds,
        eval_dataset=eval_fold_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        prompt_types=prompt_types,
    )
    
    # Train the model
    print(f"Training fold {fold + 1}...")
    trainer.train()
    
    # Evaluate on test set
    print(f"Evaluating fold {fold + 1}...")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    trainer.save_metrics("test", test_metrics)
    trainer.save_model(fold_dir)
    
    # Store results
    fold_results = {
        'fold': fold + 1,
        'test_accuracy': test_metrics.get('test_accuracy', 0),
        'test_balanced_accuracy': test_metrics.get('test_balanced_accuracy', 0),
        'test_precision': test_metrics.get('test_precision', 0),
        'test_recall': test_metrics.get('test_recall', 0),
        'test_f1': test_metrics.get('test_f1', 0),
        'train_samples': len(train_fold_ds),
        'val_samples': len(eval_fold_ds),
        'test_samples': len(test_ds)
    }
    
    all_fold_results.append(fold_results)
    fold_test_metrics.append(test_metrics)
    
    # Display fold results
    print(f"\nFOLD {fold + 1} TEST RESULTS:")
    print(f"Regular Accuracy: {test_metrics.get('test_accuracy', 'N/A'):.4f}")
    print(f"Balanced Accuracy: {test_metrics.get('test_balanced_accuracy', 'N/A'):.4f}")
    print(f"Precision (macro): {test_metrics.get('test_precision', 'N/A'):.4f}")
    print(f"Recall (macro): {test_metrics.get('test_recall', 'N/A'):.4f}")
    print(f"F1 Score (macro): {test_metrics.get('test_f1', 'N/A'):.4f}")
    
    # Clean up GPU memory
    del model, trainer
    torch.cuda.empty_cache()

# -----------------------------------------------------------
# 9 ▸ Aggregate results across all folds
# -----------------------------------------------------------
print(f"\n{'='*80}")
print("5-FOLD CROSS VALIDATION SUMMARY")
print(f"{'='*80}")

# Calculate mean and std for each metric
metrics_to_aggregate = ['test_accuracy', 'test_balanced_accuracy', 'test_precision', 'test_recall', 'test_f1']
aggregated_results = {}

for metric in metrics_to_aggregate:
    values = [fold[metric] for fold in all_fold_results]
    aggregated_results[metric] = {
        'mean': np.mean(values),
        'std': np.std(values),
        'values': values
    }

# Display aggregated results
print(f"Regular Accuracy: {aggregated_results['test_accuracy']['mean']:.4f} ± {aggregated_results['test_accuracy']['std']:.4f}")
print(f"Balanced Accuracy: {aggregated_results['test_balanced_accuracy']['mean']:.4f} ± {aggregated_results['test_balanced_accuracy']['std']:.4f}")
print(f"Precision (macro): {aggregated_results['test_precision']['mean']:.4f} ± {aggregated_results['test_precision']['std']:.4f}")
print(f"Recall (macro): {aggregated_results['test_recall']['mean']:.4f} ± {aggregated_results['test_recall']['std']:.4f}")
print(f"F1 Score (macro): {aggregated_results['test_f1']['mean']:.4f} ± {aggregated_results['test_f1']['std']:.4f}")

print(f"\nPer-fold results:")
for i, fold_result in enumerate(all_fold_results):
    print(f"Fold {i+1}: Acc={fold_result['test_accuracy']:.4f}, "
          f"Bal_Acc={fold_result['test_balanced_accuracy']:.4f}, "
          f"F1={fold_result['test_f1']:.4f}")

# Save comprehensive results
final_results = {
    'aggregated_results': aggregated_results,
    'fold_results': all_fold_results,
    'experiment_config': {
        'n_folds': args.n_folds,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'seed': args.seed,
        'dataset_file': args.dataset_file,
        'total_samples': len(filtered_ds)
    }
}

results_file = base_run_dir / "5fold_cv_results.json"
with open(results_file, 'w') as f:
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    json_results = json.loads(json.dumps(final_results, default=convert_numpy))
    json.dump(json_results, f, indent=2)

print(f"\nDetailed results saved to: {results_file}")
print(f"Individual fold results saved in: {base_run_dir}")
print(f"{'='*80}")
