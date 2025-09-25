from gears import PertData, GEARS
from pathlib import Path
import numpy as np
import pandas as pd
import json
import tempfile
import shutil
import argparse
import os
import session_info
import torch
import argparse
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from scipy.stats import pearsonr, spearmanr
from gears.inference import evaluate, compute_metrics, deeper_analysis, \
                  non_dropout_analysis

parser = argparse.ArgumentParser(description='Calculate metric')
parser.add_argument('--dataset_path', type=str, default= "data/gears_pert_data/", help='The path of dataset')
parser.add_argument('--dataset_name', type=str, default= "replogle_rpe1_essential", required = True, help='The name of dataset')
parser.add_argument("--resluts_path", type=str, default= "results/replogle_rpe1_essential_linear_results/all_predictions.json", required = True, help = "the path of linear model results.")
args = parser.parse_args()


# perturbation data path
pert_data_folder = Path(args.dataset_path)
pert_data = PertData(pert_data_folder)
pert_data.load(data_path = args.dataset_path+args.dataset_name)
adata = pert_data.adata
pert_data.prepare_split(split = 'simulation', seed = 1, train_gene_set_size=0.75)
pert_data.get_dataloader(batch_size =1 , test_batch_size = 1)

# linear prediction results
with open(args.resluts_path, "r") as f:
    predictions = json.load(f)

def evaluate(loader,  uncertainty):
    """
    Run model in inference mode using a given data loader
    """

    # model.eval()
    # model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}
    logvar = []
    de_index = {}
    for itr, batch in enumerate(loader):

        # batch.to(device)
        pert_cat.extend(batch.pert)
        # de_index[batch.pert] =
        with torch.no_grad():
            t = batch.y
            # pred.extend(p.cpu())
            truth.extend(t.cpu())

            # Differentially expressed genes
            for itr, de_idx in enumerate(batch.de_idx):
                # pred_de.append(p[itr, de_idx])
                truth_de.append(t[itr, de_idx])
                de_index[batch.pert[0]] = de_idx

   # all genes
    results['pert_cat'] = np.array(pert_cat)

    truth = torch.stack(truth)

    results['truth'] = truth.detach().cpu().numpy()

    truth_de = torch.stack(truth_de)

    results['truth_de'] = truth_de.detach().cpu().numpy()

    return results,de_index

def compute_metrics(results,predict_results,de_index, gene_idx=None):
    """
    Given results from a model run and the ground truth, compute metrics

    """
    metrics = {}
    metrics_pert = {}

    metric2fct = {
        'mse': mse,
        'pearson': pearsonr
    }

    for m in metric2fct.keys():
        metrics[m] = []
        metrics[m + '_de'] = []

    for pert in np.unique(results['pert_cat']):

        metrics_pert[pert] = {}
        p_idx = np.where(results['pert_cat'] == pert)[0]

        for m, fct in metric2fct.items():
            if m == 'pearson':
                val = fct(np.array(predict_results[pert.split("+")[0]]), results['truth'][p_idx].mean(0))[0]
                if np.isnan(val):
                    val = 0
            else:
                val = fct(np.array(predict_results[pert.split("+")[0]]), results['truth'][p_idx].mean(0))

            metrics_pert[pert][m] = val
            metrics[m].append(metrics_pert[pert][m])

        if pert != 'ctrl':

            for m, fct in metric2fct.items():
                if m == 'pearson':
                    val = fct(np.array(predict_results[pert.split("+")[0]])[de_index[pert]], results['truth_de'][p_idx].mean(0))[0]
                    if np.isnan(val):
                        val = 0
                else:
                    val = fct(np.array(predict_results[pert.split("+")[0]])[de_index[pert]], results['truth_de'][p_idx].mean(0))

                metrics_pert[pert][m + '_de'] = val
                metrics[m + '_de'].append(metrics_pert[pert][m + '_de'])

        else:
            for m, fct in metric2fct.items():
                metrics_pert[pert][m + '_de'] = 0

    for m in metric2fct.keys():
        metrics[m] = np.mean(metrics[m])
        metrics[m + '_de'] = np.mean(metrics[m + '_de'])

    return metrics, metrics_pert

results, de_index = evaluate(pert_data.dataloader['test_loader'],"uncertainty")

test_metrics, test_pert_res= compute_metrics(results,predictions,de_index)
print(test_metrics)

