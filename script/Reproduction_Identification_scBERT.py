#!/usr/bin/env python
# coding: utf-8

# # Reproduce PEFT(Parameter-Efficient Fine-Tuning) on Pre-trained Model(scBERT) with Identification
# In this tutorial, we demonstrate how to reproduce a PEFT (Parameter-Efficient Fine-Tuning) pre-trained model(scBERT) on a specific dataset for the identification cell type task. This tutorial serves as a practical example. There are two steps that need to be executed.

# ## Step 1: Modify the parameters
# ### There are four key settings that the reader needs to modify. The available options are listed below:
# ***data_name*** : NSCLC/COVID-19/ms_scBERT, \
# ***data_path***: {data_path}/celltype_annotation, \
# ***model_path***: {checkpoint_path}, \
# ***prompt_type***: Gene_encoder_prompt/ Gene_token_prompt / prefix_prompt / LoRA, \
# ***pretrain_data_path***:{pretrain_data_path}/panglao_human.h5ad.

# In[1]:


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default='COVID-19',help='NSCLC/COVID-19/ms_scBERT')
parser.add_argument("--data_path", type=str, default='/fs/ess/PAS1475/Fei/Parallel_experiments_data/celltype_annotation', help='Path of data for finetune.')
parser.add_argument("--model_path", type=str, default='/fs/ess/PAS1475/Yang/yy/SCbert/ckpts/COVID-19_Gene_token_prompt_0_best_model.pt',
                    help='Path of pretrained model.')
parser.add_argument("--prompt_type", type=str, default='Gene_token_prompt',help='Gene_encoder_prompt/Gene_token_prompt/prefix_prompt /LoRA')
parser.add_argument("--pretrain_data_path", type=str, default='/fs/ess/PAS1475/Fei/scBERT_pretrain_data/panglao_human.h5ad', help='Path of pretrain data.')


# ## Step 2: Start the validation

# In[2]:


import os
import gc
import argparse
import copy
import pickle
import json
import random
import math
import random
from functools import reduce
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support,     classification_report, balanced_accuracy_score
import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import DataLoader, Dataset
import sys
sys.path.insert(0, "../")
from performer_pytorch.performer_pytorch_prompt import PerformerLM
import scanpy as sc
import anndata as ad
from pipline_scBERT.utils import *
import pickle as pkl
from pipline_scBERT.Benckmark_utils import scBERT_preprocess
import warnings
warnings.filterwarnings("ignore")

parser.add_argument("--batch_size", type=int, default=1, help='Number of batch size.')
parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
parser.add_argument("--learning_rate", type=float, default=5e-4, help='Learning rate.')
parser.add_argument("--ckpt_dir", type=str, default='../ckpts/', help='Directory of checkpoint to save.')
parser.add_argument("--local_rank", type=int, default=0, help='Local process rank.')
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
parser.add_argument("--tokens", type=str, default=64,help='prefix token number')
parser.add_argument("--space_conf", type=str, default=[1,1,1,1,1,1],help='encoder space adapter list')
parser.add_argument("--mlp_conf", type=str, default=[1,1,1,1,1,1],help='encoder mlp adapter list')
parser.add_argument("--start_fold", default=0, type=int, help='Current fold index.')

args, unknown = parser.parse_known_args()
prompt_type=args.prompt_type
SEED = args.seed
EPOCHS = args.epoch
gpu=args.local_rank
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
SEQ_LEN = args.gene_num + 1
VALIDATE_EVERY = args.valid_every
PATIENCE = 5
UNASSIGN_THRES = 0.0
CLASS = args.bin_num + 2
POS_EMBED_USING = args.pos_embed
ckpt_dir = args.ckpt_dir
os.makedirs(ckpt_dir, exist_ok=True)
data_name = args.data_name
data_path = args.data_path
pretrain_data = args.pretrain_data_path
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
tokens = args.tokens
space_conf = args.space_conf
mlp_conf = args.mlp_conf

fold_idx = args.start_fold


# In[3]:



class SCDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0] - 1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        seq_label = self.label[rand_start]
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]


class Identity(torch.nn.Module):
    def __init__(self, dropout=0., h_dim=100, out_dim=10):
        super(Identity, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        if prompt_type=="prefix_prompt":
            self.fc1 = nn.Linear(in_features=SEQ_LEN+64, out_features=512, bias=True)
        else:
            self.fc1 = nn.Linear(in_features=SEQ_LEN, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        x = x[:, None, :, :]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


# In[4]:



if data_name=="ms" or data_name=="zheng68k":
    data_train = sc.read_h5ad(f'{data_path}/{data_name}/{data_name}_train.h5ad')
    data_val = sc.read_h5ad(f'{data_path}/{data_name}/{data_name}_val.h5ad')
    data_test = sc.read_h5ad(f'{data_path}/{data_name}/{data_name}_test.h5ad')
else:
    data_train = sc.read_h5ad(f'{data_path}/{data_name}/{str(fold_idx)}/{data_name}_train{str(fold_idx)}.h5ad')
    data_val = sc.read_h5ad(f'{data_path}/{data_name}/{str(fold_idx)}/{data_name}_val{str(fold_idx)}.h5ad')
    data_test = sc.read_h5ad(f'{data_path}/{data_name}/{str(fold_idx)}/{data_name}_test{str(fold_idx)}.h5ad')

data_train.obs["str_batch"] = "0"
data_test.obs["str_batch"] = "1"
data_val.obs["str_batch"] = "2"
data_train = data_train.concatenate((data_test, data_val), batch_key="str_batch")

if data_name == 'ms_scBERT':
    data_is_raw = False
    celltype_key = 'celltype'
elif data_name == 'zheng68k':
    data_is_raw = False
    celltype_key = 'celltype'
elif data_name == 'COVID-19':
    data_is_raw = True
    celltype_key = 'cell_type'
elif data_name == 'NSCLC':
    data_is_raw = True
    celltype_key = 'cell_type'

panglao = sc.read_h5ad(pretrain_data)
preprocess = scBERT_preprocess(panglao, data_train)
data_train = preprocess(data_is_raw=data_is_raw)
del panglao

# celltype_key = 'Factor Value[inferred cell type - authors labels]'
label_dict, _ = np.unique(np.array(data_train.obs[celltype_key]), return_inverse=True)
data_test=data_train[data_train.obs["str_batch"] == "1"]
data_train = data_train[data_train.obs["str_batch"] != "1"]   #training and val 
_, label = np.unique(np.array(data_train.obs[celltype_key]), return_inverse=True)    # training and val label

# store the label dict and label for prediction
with open(f'{data_name}_label_dict_{prompt_type}_{str(fold_idx)}fold', 'wb') as fp:
    pkl.dump(label_dict, fp)
with open(f'{data_name}_label_{prompt_type}_{str(fold_idx)}fold', 'wb') as fp:
    pkl.dump(label, fp)
class_num = np.unique(label, return_counts=True)[1].tolist()
class_weight = torch.tensor([(1 - (x / sum(class_num))) ** 2 for x in class_num])

data_val = data_train[data_train.obs["str_batch"] == "2"]
# data_test=data_train[data_train.obs["str_batch"] == "1"]
data_train = data_train[data_train.obs["str_batch"] == "0"]
_, label_val = np.unique(np.array(data_val.obs[celltype_key]), return_inverse=True)
_, label_train = np.unique(np.array(data_train.obs[celltype_key]), return_inverse=True)
_, label_test = np.unique(np.array(data_test.obs[celltype_key]), return_inverse=True)
label_val = torch.from_numpy(label_val)
label_train = torch.from_numpy(label_train)
label_test = torch.from_numpy(label_test)
data_train = data_train.X
data_val = data_val.X
data_test = data_test.X
train_dataset = SCDataset(data_train, label_train)
val_dataset = SCDataset(data_val, label_val)
test_dataset = SCDataset(data_test, label_test)
train_num = len(train_dataset)
train_class_num = np.unique(train_dataset.label,return_counts=True)[1]
sample_weights = 1.0 / train_class_num[train_dataset.label]
sample_weights = sample_weights / np.sum(sample_weights)
train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, train_num, replacement=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# In[5]:


model = PerformerLM(
    num_tokens=CLASS,
    dim=200,
    depth=6,
    max_seq_len=SEQ_LEN,
    heads=10,
    local_attn_heads=0,
    g2v_position_emb=POS_EMBED_USING,
    prompt_type=prompt_type,
    tokens=tokens,
    space_conf=space_conf,
    mlp_conf=mlp_conf,
)

# ckpt = torch.load('/fs/ess/PAS1475/Fei/scBERT-finetune/panglao_pretrain.pth')
# model.load_state_dict(ckpt['model_state_dict'], False)
model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])
model.load_state_dict(torch.load(args.model_path))

# print(prompt_type)
# print(args.model_path)

if prompt_type=="Gene_token_prompt":
    for param in model.parameters():
        param.requires_grad = False
    for param in model.Space_Adapter.parameters():
        param.requires_grad = True
    for param in model.to_out.parameters():
        param.requires_grad = True
elif prompt_type=="Gene_encoder_prompt":
    for param in model.parameters():
        param.requires_grad = False
    for param in model.performer.Encoder_Space.parameters():
        param.requires_grad = True
    for param in model.performer.Encoder_MLP.parameters():
        param.requires_grad = True
    for param in model.to_out.parameters():
        param.requires_grad = True

elif prompt_type=="LoRA":
    for param in model.parameters():
        param.requires_grad = False
    for name, para in model.named_parameters():
        if 'to_q' in name:
            para.requires_grad = True
        if 'to_v' in name:
            para.requires_grad = True
    for param in model.to_out.parameters():
        param.requires_grad = True


elif prompt_type=="prefix_prompt":
    for param in model.parameters():
        param.requires_grad = False
    for param in model.to_out.parameters():
        param.requires_grad = True
    for name, para in model.named_parameters():
        if 'prompt_embeddings' in name:
            para.requires_grad = True




def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print()
        print('total:{}'.format(total_num))
        print('trainable:{}'.format(trainable_num))

get_parameter_number(model)
model = model.to(device)
trigger_times = 0
max_acc = 0.0
best_val_loss = float("inf")
# model.load_state_dict(torch.load(args.model_path))

def test(model: nn.Module, test_loader: DataLoader) -> float:
    model.eval()
    predictions = []
    truths = []
    for index, (data_t, labels_t) in enumerate(test_loader):
        index += 1
        data_t, labels_t = data_t.to(device), labels_t.to(device)
        logits = model(data_t)
        softmax = nn.Softmax(dim=-1)
        final_prob = softmax(logits)
        final = final_prob.argmax(dim=-1)
        final[np.amax(np.array(final_prob.detach().cpu().numpy()), axis=-1) < UNASSIGN_THRES] = -1
        predictions.append(final.detach().cpu().numpy())
        truths.append(labels_t.detach().cpu().numpy())
        if index % 100 == 0:
            tqdm.write(f'Batch {index}/{len(test_loader)}')
    predictions=np.concatenate(predictions, axis=0)
    truths=np.concatenate(truths, axis=0)
    save_dict = {
        "labels": truths,
        "results": predictions,
        "id_maps": label_dict
    }
    with open(ckpt_dir +f"{data_name}_{prompt_type}_{str(fold_idx)}_results.pkl", "wb") as f:
            pickle.dump(save_dict, f)
    return predictions,truths

predictions,celltypes_labels=test(model,test_loader)
from sklearn.metrics import  precision_score, recall_score, f1_score, balanced_accuracy_score

balanced_accuracy = balanced_accuracy_score(celltypes_labels, predictions)
f1 = f1_score(celltypes_labels, predictions, average="macro")
precision = precision_score(celltypes_labels, predictions, average="macro")
recall = recall_score(celltypes_labels, predictions, average="macro")

print(f"macro Accuracy: {balanced_accuracy:.3f}, macro Precision: {precision:.3f},macro Recall: {recall:.3f}, "f"macro F1: {f1:.3f}")
micro_f1 = f1_score(celltypes_labels, predictions, average="micro")
micro_precision = precision_score(celltypes_labels, predictions, average="micro")
micro_recall = recall_score(celltypes_labels, predictions, average="micro")
print(f"micro Accuracy: {balanced_accuracy:.3f}, micro Precision: {micro_precision:.3f},micro Recall: {micro_recall:.3f}, "f"micro F1: {micro_f1:.3f}")


# In[ ]:





# In[ ]:




