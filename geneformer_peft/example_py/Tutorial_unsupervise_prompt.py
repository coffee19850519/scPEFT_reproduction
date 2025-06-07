import os

# os.environ["NCCL_DEBUG"] = "INFO"
# os.environ["OMPI_MCA_opal_cuda_support"] = "true"
# os.environ["CONDA_OVERRIDE_GLIBC"] = "2.56"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import loralib as lora
import pickle
import random
import subprocess
import numpy as np
import pytz
import torch
from datasets import load_from_disk
from transformers import TrainingArguments
from transformerslocal.src.transformers.models.bert.modeling_bert import BertForMaskedLM
from sklearn.model_selection import train_test_split
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

# set model parameters and load h5ad
model_type = "Pretrained"  # default is bert but if using unsupervised-tuning the type is Pretrained
dataset = "GCBC"

# batch size for training and eval
geneformer_batch_size = 2
# max learning rate
max_lr = 1e-4
# learning schedule
lr_schedule_fn = "linear"
# warmup steps
warmup_steps = 500
# number of epochs
epochs = 45
# optimizer
optimizer = "adamw"

# load model
if model_type == "Pretrained":
    model_directory = "/media/fei/Data/frx/Geneformer_unsupervise/Pretrain_ckpts/geneformer-12L-30M-prompt"
    model = BertForMaskedLM.from_pretrained(
        model_directory,
        output_hidden_states=True
    )

    # load weights by ourself to check validity
    if model.config.name_or_path is not None:
        model_file = os.path.join(model.config.name_or_path, "pytorch_model.bin")

        try:
            model.load_state_dict(torch.load(model_file))
            print(f"Loading all model params from {model_file}")
        except:
            # only load params that are in the model and match the size
            pretrained_dict = torch.load(model_file, map_location='cpu')
            model_dict = model.state_dict()

            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                print(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

    # freeze_param
    prompt_type = model.config.prompt_type
    print(f"Prompt type is {model.config.prompt_type}")
    pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())

    for name, para in model.named_parameters():
        para.requires_grad = False
    for param in model.cls.parameters():
        param.requires_grad = True
    for name, para in model.named_parameters():
        if 'lora_' in name:
            para.requires_grad = True
        if 'prompt_embeddings' in name:
            para.requires_grad = True
        if 'adapter' in name:
            para.requires_grad = True

    model.cls.predictions.decoder.bias.requires_grad = True
    model.cls.predictions.decoder.weight.requires_grad = True

    print("-" * 89)
    learnable_params = {k: v for k, v in model.named_parameters() if v.requires_grad == True}
    for k, v in learnable_params.items():
        print(f"Learnable params {k} with shape {v.shape}")

    post_freeze_param_count = sum(
        dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
    print("Total Pre freeze Params: %.2fM" % (pre_freeze_param_count / 1e6,))
    print("Total Post freeze Params: %.2fM" % (post_freeze_param_count / 1e6,))

# make model to train
model = model.train()

# output directories
training_output_dir = f"{rootdir}/{dataset}/{prompt_type}/"
model_output_dir = os.path.join(training_output_dir, "models/")

# ensure not overwriting previously saved model
model_output_file = os.path.join(model_output_dir, "pytorch_model.bin")
if os.path.isfile(model_output_file) is True:
    raise Exception("Model already saved to this directory.")

# make training and model output directories
subprocess.call(f"mkdir {training_output_dir}", shell=True)
subprocess.call(f"mkdir {model_output_dir}", shell=True)

# # define the training arguments
# training_args = {
#     "learning_rate": max_lr,
#     "do_train": True,
#     "do_eval": False,
#     "evaluation_strategy": "no",
#     "save_strategy": "no",
#     "logging_steps": 1,
#     "group_by_length": True,
#     "length_column_name": "length",
#     "disable_tqdm": True,
#     "lr_scheduler_type": lr_schedule_fn,
#     "warmup_steps": warmup_steps,
#     "weight_decay": 0.001,
#     "per_device_train_batch_size": geneformer_batch_size,
#     "per_device_eval_batch_size": geneformer_batch_size,
#     "num_train_epochs": epochs,
#     "load_best_model_at_end": True,
#     "output_dir": training_output_dir,
# }


training_args = {
    "learning_rate": max_lr,
    "do_train": True,
    "do_eval": True,
    "group_by_length": True,
    "length_column_name": "length",
    "disable_tqdm": False,
    "lr_scheduler_type": lr_schedule_fn,
    "warmup_steps": warmup_steps,
    "weight_decay": 0.001,
    "per_device_train_batch_size": geneformer_batch_size,
    "num_train_epochs": epochs,
    "save_strategy": "steps",
    "save_steps": 2000,  # 8 saves per epoch
    "logging_steps": 1,
    "output_dir": training_output_dir,
    "logging_dir": training_output_dir,
}
training_args = TrainingArguments(**training_args)

# training_args = TrainingArguments(**training_args)

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
    train_dataset=val_dataset,
    eval_dataset=train_dataset,
    example_lengths_file="../geneformer/genecorpus_30M_2048_lengths.pkl",
    token_dictionary=token_dictionary,
)

print("Starting training.")

# train
trainer.train()

# save model
trainer.save_model(model_output_dir)
