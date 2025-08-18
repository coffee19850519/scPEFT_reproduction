# Multi-PEFT Support for scGPT

This repository provides comprehensive **Multi-Parameter Efficient Fine-Tuning (Multi-PEFT)** support for scGPT models, enabling simultaneous application of multiple PEFT strategies for enhanced single-cell analysis with extensive experimental validation.

## Key Features

- **Multi-PEFT Strategy Support**: Simultaneously apply multiple PEFT methods (LoRA, Adapters, Prefix tuning, etc.)
- **Complete Pipeline**: Training and inference scripts with comprehensive evaluation
- **Cross-Validation Results**: Pre-trained models across 5-fold cross-validation on NSCLC dataset
- **Memory Efficient**: Significantly reduce GPU memory usage while maintaining performance
- **Flexible Configuration**: Support 15 different PEFT strategy combinations
- **Detailed Analysis**: Comprehensive metrics, confusion matrices, and visualization tools

## Supported PEFT Methods

| Method | Description | Available Models |
|:-------|:------------|:-----------------|
| **LoRA** | Low-Rank Adaptation for attention layers | ✅ |
| **ENCODER** | Adapter modules in encoder layers | ✅ |
| **TOKEN** | Token-level adaptation | ✅ |
| **PREFIX** | Prefix tuning for generation tasks | ✅ |
| **Multi-PEFT** | All strategies combined | ✅ |

## Pre-trained Models

We provide extensive pre-trained models for cell type classification on NSCLC dataset:

### Available Model Combinations (15 variants × 5 folds = 75 models)
- Single strategies: `ENCODER`, `TOKEN`, `PREFIX`, `LORA`
- Dual combinations: `ENCODER_TOKEN`, `ENCODER_PREFIX`, `ENCODER_LORA`, `TOKEN_PREFIX`, `TOKEN_LORA`, `PREFIX_LORA`
- Triple combinations: `ENCODER_TOKEN_PREFIX`, `ENCODER_TOKEN_LORA`, `ENCODER_PREFIX_LORA`, `TOKEN_PREFIX_LORA`
- Full Multi-PEFT: `ENCODER_TOKEN_PREFIX_LORA`

All models are available in [link]([adapter_combination]([NSCLC](https://mailmissouri-my.sharepoint.com/:f:/r/personal/hefe_umsystem_edu/Documents/scPEFT_checkpoints/adapter_combination/NSCLC?csf=1&web=1&e=Mwumc0)) with cross-validation splits.

## Installation

### Download Pretrained Base Models

Download the scGPT pretrained model (e.g., [whole-human model](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y?usp=sharing)) and place it in your desired directory:

```bash
# Example directory structure
./scGPT_human/
├── args.json
├── best_model.pt
└── vocab.json
```

## Quick Start

### Training with Multi-PEFT

```bash
# Navigate to tutorial directory
cd tutorial_peft

# Train with all PEFT strategies (default)
python train_cell_type.py
```

The training script supports all combinations:

```python
# Available PEFT strategy combinations
peft_strategies = [
    ["ENCODER"],
    ["TOKEN"], 
    ["PREFIX"],
    ["LORA"],
    ["ENCODER", "TOKEN"],
    ["ENCODER", "PREFIX"],
    ["ENCODER", "LORA"],
    ["TOKEN", "PREFIX"],
    ["TOKEN", "LORA"],
    ["PREFIX", "LORA"],
    ["ENCODER", "TOKEN", "PREFIX"],
    ["ENCODER", "TOKEN", "LORA"],
    ["ENCODER", "PREFIX", "LORA"],
    ["TOKEN", "PREFIX", "LORA"],
    ["ENCODER", "TOKEN", "PREFIX", "LORA"]  # Full Multi-PEFT
]
```

### Inference with Comprehensive Analysis

```bash
# Basic inference with detailed analysis
python inference_cell_type.py \
      --model_dir /mnt/c/Users/wxy/Desktop/esmc/multi_peft_support/all_result/NSCLC \
      --data_base_dir /mnt/c/Users/wxy/Desktop/esmc/multi_peft_support \
      --peft_method peft_NSCLC_TOKEN \
      --fold 0 \
      --pretrained_model ./scGPT_human \
      --batch_size 32 \
      --output_dir ./inference_results \
      --device auto \
      --save_results
```

### Inference Output

The inference script generates comprehensive results:

- `inference_results.pkl` - Complete results with predictions and probabilities
- `metrics.json` - Detailed evaluation metrics
- `confusion_matrix.png` - Normalized confusion matrix visualization
- `per_class_metrics.png` - Per-class precision, recall, F1-score
- `detailed_results.csv` - Cell-level predictions with probabilities

## File Structure

```
multi_peft_support/
├── scgpt/                              # Core scGPT modules with PEFT support
│   ├── model/                          # Enhanced model architectures
│   │   ├── model.py                    # Main TransformerModel with PEFT
│   │   └── MultiheadAttentionLoRA.py   # LoRA attention implementation
│   ├── tokenizer/                      # Gene tokenization utilities
│   ├── utils/                          # PEFT utilities and configurations
│   └── ...
├── tutorial_peft/                      # Training and inference scripts
│   ├── train_cell_type.py             # Multi-PEFT training script
│   └── inference_cell_type.py         # Comprehensive inference script
├── all_result/                         # Pre-trained models repository
│   └── NSCLC/                         # NSCLC dataset results
│       ├── fold0/                     # Cross-validation fold 0
│       │   ├── peft_NSCLC_ENCODER/    # Single ENCODER strategy
│       │   ├── peft_NSCLC_LORA/       # Single LORA strategy
│       │   ├── ...                    # All 15 combinations
│       │   └── peft_NSCLC_ENCODER_TOKEN_PREFIX_LORA/  # Full Multi-PEFT
│       ├── fold1/                     # Cross-validation fold 1
│       ├── ...                        # Folds 2-4
│       └── fold4/                     # Cross-validation fold 4
├── dataset/                                 # Sample data (fold 1)
│   ├── NSCLC_train1.h5ad
│   ├── NSCLC_val1.h5ad
│   └── NSCLC_test1.h5ad
├── Reproduction_Identification.py      # Reproduction script
├── requirements.yaml                   # Environment dependencies
└── README.md                          # This file
```
