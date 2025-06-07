
# Geneformer-PEFT: Parameter-Efficient Fine-Tuning for Cell Type Classification

This project is built on the HuggingFace BERT architecture and the [Geneformer](https://github.com/broadinstitute/geneformer) framework, integrating various lightweight fine-tuning methods such as **Prompt Tuning**, **Adapter**, and **LoRA** to achieve efficient cell type classification on single-cell transcriptomic data.

> 🔬 Main entry file: `../geneformer_peft/example_py/cell_classification_prompt_fold.py`  
> 🎯 Purpose: Supports fine-tuning under different PEFT strategies, cross-validation, handling class imbalance, and outputting multiple metrics.

---

## 🧩 Features

The script provides the following features:

- ✅ Loads pre-split `train/val/test` datasets;
- ✅ Supports K-fold cross-validation;
- ✅ Enables various **PEFT methods** (e.g., LoRA / Prefix Prompt / Adapter) via configuration;
- ✅ Uses **class-weighted CrossEntropyLoss** to address class imbalance;
- ✅ Outputs metrics such as macro/micro accuracy, precision, recall, and F1;
- ✅ Supports combination of multiple prompt types with dynamic layer freezing;
- ✅ All classification-related model changes are implemented in `transformerslocal`.

---

## 📦 Project Dependencies & File Structure

**Make sure to extract the following directories** (already uploaded as `.zip` files):

```
.
├── transformerslocal/          # Modified Transformers source code (supports LoRA, Prompt, etc.)
├── Pretrain_ckpts/             # Pretrained weights and configuration (config.json specifies fine-tuning method)
├── example_py/data/           
└── example_py/cell_classification_prompt_fold.py  # Main script
```

---

## ⚙️ Command-Line Arguments

| Argument          | Description                                               | Default Value   |
|-------------------|-----------------------------------------------------------|-----------------|
| `--dataset_name`  | Dataset name (must match structure under `data_path`)     | NSCLC           |
| `--data_path`     | Root directory of dataset (must include split subfolders) | `/media/...`    |
| `--output_path`   | Output directory for models and predictions               | `/media/...`    |
| `--out_prefix`    | Custom output prefix (currently unused)                   | `""`            |
| `--epoch`         | Number of fine-tuning epochs                              | `1`             |
| `--use_prompt`    | Whether to use prompt tuning                              | `True`          |
| `--lr`            | Learning rate                                             | `0.0001`        |
| `--batch_size`    | Training batch size                                       | `20`            |

Example command:

```bash
python ../geneformer_peft/example_py/cell_classification_prompt_fold.py \
  --dataset_name NSCLC \
  --data_path ./example_py/data \
  --output_path ./save/ \
  --epoch 3 \
  --batch_size 32 \
  --lr 5e-5
```

---

## 🧠 PEFT Configuration

This project supports the following fine-tuning methods, which can be **used in combination**:

| `prompt_type`         | Description                                                 |
|------------------------|-------------------------------------------------------------|
| `lora`                 | Applies to Query/Value layers in Attention                  |
| `prefix_prompt`        | Inserts learnable prompt embeddings before token input      |
| `Gene_token_prompt`    | Adapter modules inserted after embeddings                   |
| `encoder_prompt`       | Adapter modules inside Transformer layers                   |

Configure via `./Pretrain_ckpts/geneformer-12L-30M-prompt/config.json`:

```json
"prompt_type": "Gene_token_prompt,lora",
"if_use_prompt": true,
"lora_r": 8,
"num_token": 64,
"mlp_adapter_conf": [1,1,1,1,1,1,1,1,1,1,1,1],
"space_adapter_conf": [1,1,1,1,1,1,1,1,1,1,1,1]
```

> - `prompt_type`: Prompt fine-tuning types (comma-separated if multiple)  
> - `mlp_adapter_conf`: Enable MLP Adapter for each layer (12 total)  
> - `space_adapter_conf`: Enable Space Adapter for each layer  

---

## 🧰 Classifier Module

The classification model used is:

```python
from geneformer import Classifier
# Actually points to:
from transformerslocal.src.transformers.models.bert.modeling_bert import BertForSequenceClassification
```

All structure definitions for classification heads, LoRA injection, prompt embedding insertion, and adapter integration are implemented in `transformerslocal/src/transformers/models/bert/`, maintaining compatibility with the official HuggingFace interface.

---

## 🔒 Layer Freezing Strategy

In `cell_classification_prompt_fold.py`, from **line 235 to 271**, dynamic settings are applied based on the selected `prompt_type`:

- Layers with `requires_grad = True` will be trainable;
- Others will be frozen;
- Fine-tuning will be limited to LoRA, prompt, or adapter layers when specified.

Example:

```python
if 'lora' in prompt_types:
    lora.mark_only_lora_as_trainable(model, bias='lora_only')
    for name, para in model.named_parameters():
        if "lora_key" in name:
            para.requires_grad = False
        if "classifier" in name:
            para.requires_grad = True

if 'Gene_token_prompt' in prompt_types:
    for name, para in model.named_parameters():
        if "bert.adapter" in name:
            para.requires_grad = True
        elif "classifier" in name:
            para.requires_grad = True
        else:
            para.requires_grad = False

if 'encoder_prompt' in prompt_types:
    for name, para in model.named_parameters():
        if "Space_Adapter" in name:
            para.requires_grad = True
        elif "MLP_Adapter" in name:
            para.requires_grad = True
        elif "classifier" in name:
            para.requires_grad = True
        else:
            para.requires_grad = False

if 'prefix_prompt' in prompt_types:
    for name, para in model.named_parameters():
        if "prompt_embeddings" in name:
            para.requires_grad = True
        elif "classifier" in name:
            para.requires_grad = True
        else:
            para.requires_grad = False
```

---

## 📈 Output

Each fold produces:

- `*.csv`: Predicted labels, confidence scores, and ground truth;
- `pytorch_model.bin`: The fine-tuned model;
- `predictions.pickle`: Prediction results;
- Evaluation metrics (macro/micro precision, recall, F1) printed to terminal;
- All results saved in `output_path/<prompt_type>/<dataset>/<fold>/`.

---

## 🔧 Integration Suggestions

Two possible approaches:

1. **Directly modify Geneformer's Classifier:**  
   Replace Geneformer's original classifier with the PEFT-compatible classification logic from this script.

2. **Adapt Geneformer’s dataset format to this script:**  
   Keep Geneformer’s model unchanged and convert your dataset format to match this script’s expected input.
