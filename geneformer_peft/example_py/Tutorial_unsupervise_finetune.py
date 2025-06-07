import datetime
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pickle
import random
import subprocess

import numpy as np
import pytz
import torch
from datasets import load_from_disk
from transformers import BertConfig, BertForMaskedLM, TrainingArguments
from geneformer import GeneformerPretrainer

seed_num = 0
random.seed(seed_num)
np.random.seed(seed_num)
seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# set local time/directories
timezone = pytz.timezone("US/Eastern")
rootdir = "./Geneformer_ckpts"

# set model parameters
# model type
model_type = "Pretrained"  # default is bert but if using unsupervised-tuning the type is Pretrained
dataset = "GCBC"

# set training parameters
# batch size for training and eval
geneformer_batch_size = 2
# max learning rate
max_lr = 1e-4
# learning schedule
lr_schedule_fn = "linear"
# warmup steps
warmup_steps = 500
# number of epochs
epochs = 20
# optimizer
optimizer = "adamw"

# output directories
training_output_dir = f"{rootdir}/{dataset}/finetune"
model_output_dir = os.path.join(training_output_dir, "models/")

# ensure not overwriting previously saved model
model_output_file = os.path.join(model_output_dir, "pytorch_model.bin")
if os.path.isfile(model_output_file) is True:
    raise Exception("Model already saved to this directory.")

# make training and model output directories
subprocess.call(f"mkdir {training_output_dir}", shell=True)
subprocess.call(f"mkdir {model_output_dir}", shell=True)

if model_type == "Pretrained":
    model_directory = "../Pretrain_ckpts/geneformer-12L-30M-finetune"
    model = BertForMaskedLM.from_pretrained(model_directory)

    model = model.train()

# define the training arguments
training_args = {
    "learning_rate": max_lr,
    "do_train": True,
    "do_eval": True,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "logging_steps": 1,
    "group_by_length": True,
    "length_column_name": "length",
    "disable_tqdm": True,
    "lr_scheduler_type": lr_schedule_fn,
    "warmup_steps": warmup_steps,
    "weight_decay": 0.001,
    "per_device_train_batch_size": geneformer_batch_size,
    "per_device_eval_batch_size": geneformer_batch_size,
    "num_train_epochs": epochs,
    "load_best_model_at_end": True,
    "output_dir": training_output_dir,
}
training_args = TrainingArguments(**training_args)

with open("../geneformer/token_dictionary.pkl", "rb") as fp:
    token_dictionary = pickle.load(fp)

# load and split dataset
train_val_dataset = load_from_disk(
    f"/media/fei/Data/frx/Geneformer_unsupervise/example_py/data/tonsil/{dataset}.dataset")
train_val_dataset = train_val_dataset.train_test_split(test_size=0.1, shuffle=True)
train_dataset = train_val_dataset['train']
val_dataset = train_val_dataset['test']

# define the trainer
trainer = GeneformerPretrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    example_lengths_file="../geneformer/genecorpus_30M_2048_lengths.pkl",
    token_dictionary=token_dictionary,
)

print("Starting training.")
# train
trainer.train()

# save model
trainer.save_model(model_output_dir)
