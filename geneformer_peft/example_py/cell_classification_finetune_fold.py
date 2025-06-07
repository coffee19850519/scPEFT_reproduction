## Geneformer Fine-Tuning for Cell Annotation Application
import argparse
# %%

import os
import sys
from pathlib import Path
# GPU_NUMBER = [0]
# os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
# os.environ["NCCL_DEBUG"] = "INFO"
import pandas as pd
import torch
from transformerslocal.src.transformers import EarlyStoppingCallback
import torch.nn as nn
from collections import Counter
import datetime
import pickle
import subprocess
import seaborn as sns;

sns.set()
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score
from transformers import BertForSequenceClassification
# from transformers  import BertForSequenceClassification
from transformers import Trainer
from transformers.training_args import TrainingArguments
from geneformer import DataCollatorForCellClassification
import numpy as np
import loralib as lora
from numpy import mean

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default='NSCLC',
                    help='ms/zheng68k/COVID/NSCLC/mouse_115746/MergedHuman/MergedMonkey/elegans')
parser.add_argument("--data_path", type=str, default='/media/fei/Data/zxy/Geneformer 3090/Geneformer_data/',
                    help='/mnt/Data5/23zmr/Geneformer_data,/mnt/Data1/frx/Geneformer/example_py/data')
parser.add_argument("--output_path", type=str, default=f"/media/fei/Data/zxy/Geneformer 3090/save/",
                    help='Path of data for predicting.')
parser.add_argument("--finetune_type", type=str, default='Cls_finetune', help='Full_finetune/Cls_finetune')
parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
parser.add_argument("--lr", type=int, default=0.0001, help='')
parser.add_argument("--batch_size", type=int, default=100, help='')
args = parser.parse_args()

dataset_name = args.dataset_name
data_dir = args.data_path
finetune_type = args.finetune_type
if dataset_name == "MergedHuman":
    n_splits = 4
else:
    n_splits = 5
accuracys = []
precisions = []
recalls = []
macro_f1s = []
mic_precisions = []
mic_recalls = []
micro_f1s = []
for i in range(n_splits):

    train_data_dir = f'{data_dir}/{dataset_name}/{i}/{dataset_name}_train{i}.dataset'
    eval_data_dir = f'{data_dir}/{dataset_name}/{i}/{dataset_name}_val{i}.dataset'
    test_data_dir = f'{data_dir}/{dataset_name}/{i}/{dataset_name}_test{i}.dataset'
    train_dataset = load_from_disk(train_data_dir)
    eval_dataset = load_from_disk(eval_data_dir)
    test_dataset = load_from_disk(test_data_dir)

    dataset_list = []
    evalset_list = []
    organ_list = []
    testset_list = []
    target_dict_list = []

    trainset_organ = train_dataset
    evalset_organ = eval_dataset
    testset_organ = test_dataset
    # per scDeepsort published method, drop cell types representing <0.5% of cells
    celltype_counter = Counter(trainset_organ["cell_type"])
    total_cells = sum(celltype_counter.values())
    cells_to_keep = [k for k, v in celltype_counter.items() if v > (0.005 * total_cells)]

    trainset_organ_subset = trainset_organ
    evalset_organ_subset = evalset_organ
    testset_organ_subset = testset_organ

    # shuffle datasets and rename columns
    trainset_organ_shuffled = trainset_organ_subset.shuffle(seed=42)
    trainset_organ_shuffled = trainset_organ_shuffled.rename_column("cell_type", "label")

    evalset_organ_shuffled = evalset_organ_subset.shuffle(seed=42)
    evalset_organ_shuffled = evalset_organ_shuffled.rename_column("cell_type", "label")

    testset_organ_shuffled = testset_organ_subset.shuffle(seed=42)
    testset_organ_shuffled = testset_organ_shuffled.rename_column("cell_type", "label")

    # create dictionary of cell types : label ids
    target_names = list(Counter(
        trainset_organ_shuffled["label"] + evalset_organ_shuffled["label"] + testset_organ_shuffled["label"]).keys())
    target_name_id_dict = dict(zip(target_names, [i for i in range(len(target_names))]))
    target_dict_list += [target_name_id_dict]
    target_name_id_celltype_dict = {value: key for key, value in target_name_id_dict.items()}


    # change labels to numerical ids
    def classes_to_ids(example):
        example["label"] = target_name_id_dict[example["label"]]
        return example


    labeled_trainset = trainset_organ_shuffled.map(classes_to_ids, num_proc=16)
    labeled_evalset = evalset_organ_shuffled.map(classes_to_ids, num_proc=16)
    labeled_testset = testset_organ_shuffled.map(classes_to_ids, num_proc=16)

    labeled_train_split = labeled_trainset
    labeled_eval_split = labeled_evalset

    labeled_test_split_subset = labeled_testset
    trained_labels = list(Counter(labeled_train_split["label"]).keys())
    labeled_eval_split_subset = labeled_eval_split

    dataset_list += [labeled_train_split]
    evalset_list += [labeled_eval_split_subset]
    testset_list += [labeled_test_split_subset]
    # %%

    trainset_dict = dataset_list
    traintargetdict_dict = target_dict_list
    evalset_dict = evalset_list
    testset_dict = testset_list


    def SaveProcessResult(labels, pred_cellcodes, confidence_socre):
        d_save = {'pred_celltype': [target_name_id_celltype_dict[pred_cellcode] for pred_cellcode in pred_cellcodes],
                  'gt_celltype': [target_name_id_celltype_dict[label] for label in labels],
                  'pred_cellcode': pred_cellcodes,
                  'gt_cellcode': labels,
                  'confidence_socre': confidence_socre
                  }
        pd.DataFrame(data=d_save).to_csv(f'{output_dir}/Geneformer_dataset{dataset_name}_result.csv')


    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = balanced_accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='macro')
        recall = recall_score(labels, preds, average='macro')
        macro_f1 = f1_score(labels, preds, average='macro')

        micro_precision = precision_score(labels, preds, average='micro')
        micro_recall = recall_score(labels, preds, average='micro')
        micro_f1 = f1_score(labels, preds, average='micro')
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'macro_f1': macro_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1

        }


    max_input_size = 2 ** 11  # 2048
    max_lr = args.lr
    # how many pretrained layers to freeze
    # number gpus
    # number cpu cores
    num_proc = 16
    # batch size for training and eval
    geneformer_batch_size = args.batch_size
    # learning schedule
    lr_schedule_fn = "linear"
    # warmup steps
    warmup_steps = 500
    # number of epochs
    epochs = args.epoch
    # optimizer
    optimizer = "adamw"

    # %%
    celltype_id_labels = [target_name_id_dict[item] for item in train_dataset["cell_type"]]
    class_num = np.unique(celltype_id_labels, return_counts=True)[1].tolist()
    class_weight = torch.tensor([(1 - (x / sum(class_num))) ** 2 for x in class_num])
    # weighted_loss = nn.CrossEntropyLoss(weight=class_weight)
    weighted_loss = nn.CrossEntropyLoss()

    organ_trainset = trainset_dict[0]
    organ_evalset = evalset_dict[0]
    organ_testset = testset_dict[0]
    organ_label_dict = traintargetdict_dict[0]
    logging_steps = 1

    # reload pretrained model
    model = BertForSequenceClassification.from_pretrained(
        "../geneformer-12L-30M",
        # "/mnt/Data1/23zmr/Geneformer/fine_tuned_models/geneformer_datasetms_CellClassifier/checkpoint-12600",
        num_labels=len(organ_label_dict.keys()),
        output_attentions=False,
        output_hidden_states=False,
        ignore_mismatched_sizes=True).to("cuda")
    output_dir = args.output_path + finetune_type + '/' + dataset_name + f"/{i}"
    model.config.save_pretrained(output_dir)

    for name, para in model.named_parameters():
        para.requires_grad = True
    if finetune_type == "Cls_finetune":
        for name, para in model.named_parameters():
            if 'classifier' not in name:
                para.requires_grad = False

    else:
        for name, para in model.named_parameters():
            para.requires_grad = True


    class MyTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(**inputs)
            loss_fct = weighted_loss.to(outputs['logits'].device)
            loss = loss_fct(outputs['logits'], inputs['labels'])

            return (loss, outputs) if return_outputs else loss


    # Statistical learnable parameters
    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print()
        print('total:{}'.format(total_num))
        print('trainable:{}'.format(trainable_num))


    get_parameter_number(model)

    # define output directory path
    current_date = datetime.datetime.now()
    datestamp = f"{str(current_date.year)[-2:]}-{current_date.month:02d}-{current_date.day:02d}_{current_date.hour:02d}:{current_date.minute:02d}:{current_date.second:02d}"

    # ensure not overwriting previously saved model
    saved_model_test = os.path.join(output_dir, f"pytorch_model.bin")
    model.config.save_pretrained(output_dir)
    print(datestamp)
    if os.path.isfile(saved_model_test) == True:
        raise Exception("Model already saved to this directory.")

    # make output directory
    subprocess.call(f'mkdir {output_dir}', shell=True)
    # set training arguments
    training_args = {
        "learning_rate": max_lr,
        "do_train": True,
        "do_eval": True,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_steps": logging_steps,
        "group_by_length": True,
        "length_column_name": "length",
        "disable_tqdm": False,
        "lr_scheduler_type": lr_schedule_fn,
        "warmup_steps": warmup_steps,
        "weight_decay": 0.001,
        "per_device_train_batch_size": geneformer_batch_size,
        "per_device_eval_batch_size": geneformer_batch_size,
        "num_train_epochs": epochs,
        "load_best_model_at_end": True,
        "output_dir": output_dir,
        "save_safetensors": False,
    }

    training_args_init = TrainingArguments(**training_args)

    # create the trainer
    trainer = MyTrainer(
        model=model,
        args=training_args_init,
        data_collator=DataCollatorForCellClassification(),
        train_dataset=organ_trainset,
        eval_dataset=organ_evalset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    # train the cell type classifier
    trainer.train()
    predictions = trainer.predict(organ_testset)

    print(f"fold{i}:"
          f"macro Accuracy: {predictions.metrics['test_accuracy']:.3f}, macro Precision: {predictions.metrics['test_precision']:.3f},macro Recall: {predictions.metrics['test_recall']:.3f}, "
          f"macro F1: {predictions.metrics['test_macro_f1']:.3f}"
          )
    print(f"fold{i}:"
          f"micro Accuracy: {predictions.metrics['test_accuracy']:.3f}, micro Precision: {predictions.metrics['test_micro_precision']:.3f}, micro Recall: {predictions.metrics['test_micro_recall']:.3f}, "
          f"micro F1: {predictions.metrics['test_micro_f1']:.3f}")
    accuracys.append(predictions.metrics["test_accuracy"])
    precisions.append(predictions.metrics["test_precision"])
    recalls.append(predictions.metrics["test_recall"])
    macro_f1s.append(predictions.metrics["test_macro_f1"])
    mic_precisions.append(predictions.metrics["test_micro_precision"])
    mic_recalls.append(predictions.metrics["test_micro_recall"])
    micro_f1s.append(predictions.metrics["test_micro_f1"])

    with open(f"{output_dir}/predictions.pickle", "wb") as fp:
        pickle.dump(predictions, fp)
    trainer.save_metrics("eval", predictions.metrics)
    trainer.save_model(output_dir)
print(" aver outcome:"
      f"macro Accuracy: {mean(accuracys):.3f}, macro Precision: {mean(precisions):.3f},macro Recall: {mean(recalls):.3f}, "
      f"macro F1: {mean(macro_f1s):.3f}"
      )
print(" variance outcome:"
      f"macro Accuracy: {np.sqrt(np.var(accuracys)):.3f}, macro Precision: {np.sqrt(np.var(precisions)):.3f},macro Recall: {np.sqrt(np.var(recalls)):.3f}, "
      f"macro F1: {np.sqrt(np.var(macro_f1s)):.3f}"
      )
print(" aver outcome:"
      f"micro Accuracy: {mean(accuracys):.3f}, micro Precision: {mean(mic_precisions):.3f},micro Recall: {mean(mic_recalls):.3f}, "
      f"micro F1: {mean(micro_f1s):.3f}"
      )
print(" variance outcome:"
      f"micro Accuracy: {np.sqrt(np.var(accuracys)):.3f}, micro Precision: {np.sqrt(np.var(mic_precisions)):.3f},micro Recall: {np.sqrt(np.var(mic_recalls)):.3f}, "
      f"micro F1: {np.sqrt(np.var(micro_f1s)):.3f}"
      )
