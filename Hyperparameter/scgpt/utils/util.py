import functools
import json
import logging
import os
from pathlib import Path
import random
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from torchtext.vocab import Vocab
from anndata import AnnData
from matplotlib import pyplot as plt
from matplotlib import axes
from IPython import get_ipython
from enum import Enum, auto

from .. import logger


@dataclass
class PeftConfig:
    r"""PeftConfig is a config of Parameter-Efficient Fine-Tuning. Users can adjust the settings
    to control whether to use PEFT, and if so, which adapter to choose.

    Args:
        use_default_settings: Whether to use the default settings of scPEFT.

        model_nlayers: The number of model layers; the default is 12.

        peft_type: The type of Parameter-Efficient Fine-Tuning (required),
            default is false, which means is not using Parameter-Efficient Fine-Tuning.
            If users want to use Parameter-Efficient Fine-Tuning,you can choose HYBRID/ENCODER/TOKEN/PREFIX/LORA

        adapter_layer_conf: The layer to which you want the adapter to be applied in the configuration.

        mlp_ratio: Used to control the hidden dimension of the adapter.
            This setting is only used in TOKEN, ENCODER, or HYBRID adapters.

        skip_connect: Used to control whether the adapter uses residual connections, default is true
            This setting is only used in TOKEN/ENCODER/HYBRID adapter.

        token_nums: The number of PEFT embeddings, which is only used in the PREFIX adapter.

        r: This setting is only used in LoRA.

        lora_alpha: This setting is only used in LoRA.

        enable_lora: This setting is only used in LoRA.

    Examples::
        >>> peft_config = PeftConfig(
        >>>    peft_type="LORA",
        >>>    r=8,
        >>>    lora_alpha=1,
        >>>    adapter_layer_conf=[True for _ in range(12)]
        >>> ).to_dict()
        >>> print(peft_config)
    """

    model_nlayers: Union[int] = 12
    use_default_settings: Union[bool] = False

    peft_type: Union[str, bool] = None
    adapter_layer_conf: List[bool] = None

    # Settings for Encoder/Hybrid/Token Adapter
    mlp_ratio: Union[float] = None
    skip_connect: Union[bool] = None

    # Settings for Prefix Adapter
    token_nums: Union[int] = None

    # Settings for LoRA
    r: Union[int] = None
    lora_alpha: Union[int] = None
    enable_lora: List[bool] = None

    def __post_init__(self):
        # Default settings for adapters
        if self.use_default_settings:
            self.adapter_layer_conf = [item <= int(self.model_nlayers / 2) for item in range(1, self.model_nlayers + 1)]

            if self.peft_type == "HYBRID":
                self.mlp_ratio = 0.25
                self.skip_connect = True

            if self.peft_type == "ENCODER":
                self.mlp_ratio = 0.25
                self.skip_connect = True

            if self.peft_type == "TOKEN":
                self.mlp_ratio = 1
                self.skip_connect = True

            if self.peft_type == "PREFIX":
                self.token_nums = 8

            if self.peft_type == "LORA":
                self.r = 8
                self.lora_alpha = 1
                self.enable_lora = [True, False, True]

        # Setting flags based on peft_type
        self.set_flags()

    def set_flags(self):
        # HYBRID / ENCODER / TOKEN / PREFIX / LORA
        peft_type_flags = {
            "HYBRID": (True, True, False, False),
            "ENCODER": (True, False, False, False),
            "TOKEN": (False, True, False, False),
            "PREFIX": (False, False, True, False),
            "LORA": (False, False, False, True)
        }

        if self.peft_type in peft_type_flags:
            self.ENCODER_FLAG, self.TOKEN_FLAG, self.PREFIX_FLAG, self.LoRA_FLAG = peft_type_flags[self.peft_type]
        else:
            self.ENCODER_FLAG, self.TOKEN_FLAG, self.PREFIX_FLAG, self.LoRA_FLAG = (False, False, False, False)

    def to_dict(self):
        return self.__repr__()

    def __repr__(self):
        representation: Dict[str, Union[str, List[bool], int, float]] = {"peft_type": self.peft_type}

        # Initialize common fields
        common_fields = {
            "mlp_ratio": self.mlp_ratio,
            "skip_connect": self.skip_connect,
            "adapter_layer_conf": self.adapter_layer_conf,
        }

        # Include relevant details based on the peft_type
        if self.peft_type == "HYBRID":
            representation.update(common_fields)
            representation["ENCODER_FLAG"] = self.ENCODER_FLAG
            representation["TOKEN_FLAG"] = self.TOKEN_FLAG

        elif self.peft_type == "ENCODER":
            representation.update(common_fields)
            representation["ENCODER_FLAG"] = self.ENCODER_FLAG

        elif self.peft_type == "TOKEN":
            representation["TOKEN_FLAG"] = self.TOKEN_FLAG
            representation["mlp_ratio"] = self.mlp_ratio

        elif self.peft_type == "PREFIX":
            representation["PREFIX_FLAG"] = self.PREFIX_FLAG
            representation["adapter_layer_conf"] = self.adapter_layer_conf
            representation["token_nums"] = self.token_nums

        elif self.peft_type == "LORA":
            representation.update({
                "r": self.r,
                "LoRA_FLAG": self.LoRA_FLAG,
                "lora_alpha": self.lora_alpha,
                "adapter_layer_conf": self.adapter_layer_conf
            })

        else:
            representation["ENCODER_FLAG"] = self.ENCODER_FLAG
            representation["TOKEN_FLAG"] = self.TOKEN_FLAG
            representation["PREFIX_FLAG"] = self.PREFIX_FLAG
            representation["LoRA_FLAG"] = self.LoRA_FLAG

        return representation


def load_tfs(tfs_file_path: Path, vocab: Vocab):
    """
       Load tf genes from file.
    """

    with open(tfs_file_path, 'r') as file:
        tfs = [line.strip() for line in file.readlines()]

    # filter tfs, which is not in vocab
    tfs = [tf for tf in tfs if tf in vocab]

    return tfs


def gene_vocabulary():
    """
    Generate the gene name2id and id2name dictionaries.
    """
    pass


def set_seed(seed):
    """set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # if n_gpu > 0:
    #     torch.cuda.manual_seed_all(seed)


def add_file_handler(logger: logging.Logger, log_file_path: Path):
    """
    Add a file handler to the logger.
    """
    h = logging.FileHandler(log_file_path)

    # format showing time, name, function, and message
    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s-%(funcName)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    h.setFormatter(formatter)
    h.setLevel(logger.level)
    logger.addHandler(h)


def category_str2int(category_strs: List[str]) -> List[int]:
    set_category_strs = set(category_strs)
    name2id = {name: i for i, name in enumerate(set_category_strs)}
    return [name2id[name] for name in category_strs]


def isnotebook() -> bool:
    """check whether excuting in jupyter notebook."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return True  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def get_free_gpu():
    import subprocess
    import sys
    from io import StringIO
    import pandas as pd

    gpu_stats = subprocess.check_output(
        [
            "nvidia-smi",
            "--format=csv",
            "--query-gpu=memory.used,memory.free",
        ]
    ).decode("utf-8")
    gpu_df = pd.read_csv(
        StringIO(gpu_stats), names=["memory.used", "memory.free"], skiprows=1
    )
    # Suppress verbose GPU usage printing; return the index only
    gpu_df["memory.free"] = gpu_df["memory.free"].map(lambda x: int(x.rstrip(" [MiB]")))
    idx = gpu_df["memory.free"].idxmax()
    # Suppress verbose GPU selection message

    return idx


def get_git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


def get_git_diff():
    commit = get_git_commit()
    return subprocess.check_output(["git", "diff", commit]).decode("utf-8").strip()


def histogram(
        *data: List[np.ndarray],
        label: List[str] = ["train", "valid"],
        color: List[str] = ["blue", "red"],
        figsize: Tuple[int, int] = (9, 4),
        title: Optional[str] = None,
        show: bool = False,
        save: Optional[str] = None,
) -> axes.Axes:
    """
    Plot histogram of the data.

    Args:
        data (List[np.ndarray]): The data to plot.
        label (List[str]): The label of the data.
        color (List[str]): The color of the data.
        figsize (Tuple[int, int]): The size of the figure.
        title (Optional[str]): The title of the figure.
        show (bool): Whether to show the figure.
        save (Optional[str]): The path to save the figure.

    Returns:
        axes.Axes: The axes of the figure.
    """
    # show histogram of the clipped values
    assert len(data) == len(label), "The number of data and labels must be equal."

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=150)
    max_value = max(np.max(data) for data in data)
    ax.hist(
        [d.flatten() for d in data],
        bins=np.arange(0, max_value + 1, 1) + 0.5 if max_value < 60 else 60,
        label=label,
        density=True,
        histtype="bar",
        linewidth=2,
        rwidth=0.85,
        color=color,
    )
    ax.legend()
    ax.set_xlabel("counts")
    ax.set_ylabel("density")

    if title is not None:
        ax.set_title(title)

    if show:
        plt.show()

    if save is not None:
        fig.savefig(save, bbox_inches="tight")

    return ax


def _indicate_col_name(adata: AnnData, promt_str: str) -> Optional[str]:
    """
    Indicate the column name of the data.

    Args:
        adata (AnnData): The AnnData object.
        promt_str (str): The peft string.

    Returns:
        Optional[str]: The column name.
    """
    while True:
        col_name = input(promt_str)
        if col_name == "":
            col_name = None
            break
        elif col_name in adata.var.columns:
            break
        elif col_name in adata.obs.columns:
            break
        else:
        print(f"Column {col_name} not found. Please input again.")

    return col_name


def find_required_colums(
        adata: AnnData,
        id: str,
        configs_dir: Union[str, Path],
        update: bool = False,
) -> List[Optional[str]]:
    """
    Find the required columns in AnnData, including celltype column, str_celltype
    column, the gene name column, and the experimental batch key.

    This function asks the user to input the required column names if the first
    time loading the data. The names are saved in the config file and will be
    automatically loaded next time.

    Args:
        adata (AnnData): The AnnData object.
        id (str): The id of the AnnData object, will be used as the file name for
            saving the config file.
        configs_dir (Union[str, Path]): The directory of saved config files.
        update (bool): Whether to update the config file.

    Returns:
        List[Optional[str]]: The required columns, including celltype_col, str_celltype_col,
            gene_col, and batch_col.
    """
    if isinstance(configs_dir, str):
        configs_dir = Path(configs_dir)

    if not configs_dir.exists():
        configs_dir.mkdir()

    config_file = configs_dir / f"{id}.json"

    if not config_file.exists() or update:
        print(
            "Config file missing. Please input the required column names."
        )
        celltype_col = _indicate_col_name(
            adata,
            "Please input the celltype column name (skip if not applicable): ",
        )
        str_celltype_col = _indicate_col_name(
            adata, "Please input the str_celltype column name: "
        )
        gene_col = _indicate_col_name(adata, "Please input the gene column name: ")
        batch_col = _indicate_col_name(adata, "Please input the batch column name: ")

        config = {
            "celltype_col": celltype_col,
            "str_celltype_col": str_celltype_col,
            "gene_col": gene_col,
            "batch_col": batch_col,
        }

        with open(config_file, "w") as f:
            json.dump(config, f)

    else:
        with open(config_file, "r") as f:
            config = json.load(f)

    return [
        config["celltype_col"],
        config["str_celltype_col"],
        config["gene_col"],
        config["batch_col"],
    ]


def tensorlist2tensor(tensorlist, pad_value):
    max_len = max(len(t) for t in tensorlist)
    dtype = tensorlist[0].dtype
    device = tensorlist[0].device
    tensor = torch.zeros(len(tensorlist), max_len, dtype=dtype, device=device)
    tensor.fill_(pad_value)
    for i, t in enumerate(tensorlist):
        tensor[i, : len(t)] = t
    return tensor


def map_raw_id_to_vocab_id(
        raw_ids: Union[np.ndarray, torch.Tensor],
        gene_ids: np.ndarray,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Map some raw ids which are indices of the raw gene names to the indices of the

    Args:
        raw_ids: the raw ids to map
        gene_ids: the gene ids to map to
    """
    if isinstance(raw_ids, torch.Tensor):
        device = raw_ids.device
        dtype = raw_ids.dtype
        return_pt = True
        raw_ids = raw_ids.cpu().numpy()
    elif isinstance(raw_ids, np.ndarray):
        return_pt = False
        dtype = raw_ids.dtype
    else:
        raise ValueError(f"raw_ids must be either torch.Tensor or np.ndarray.")

    if raw_ids.ndim != 1:
        raise ValueError(f"raw_ids must be 1d, got {raw_ids.ndim}d.")

    if gene_ids.ndim != 1:
        raise ValueError(f"gene_ids must be 1d, got {gene_ids.ndim}d.")

    mapped_ids: np.ndarray = gene_ids[raw_ids]
    assert mapped_ids.shape == raw_ids.shape
    if return_pt:
        return torch.from_numpy(mapped_ids).type(dtype).to(device)
    return mapped_ids.astype(dtype)


def load_pretrained(
        model: torch.nn.Module,
        pretrained_params: Mapping[str, torch.Tensor],
        strict: bool = False,
        prefix: Optional[List[str]] = None,
        verbose: bool = True,
) -> torch.nn.Module:
    """
    Load pretrained weights to the model.

    Args:
        model (torch.nn.Module): The model to load weights to.
        pretrained_params (Mapping[str, torch.Tensor]): The pretrained parameters.
        strict (bool): Whether to strictly enforce that the keys in :attr:`pretrained_params`
            match the keys returned by this module's :meth:`Module.state_dict`. Default to False.
        prefix (List[str]): The list of prefix strings to match with the keys in
            :attr:`pretrained_params`. The matched keys will be loaded. Default to None.

    Returns:
        torch.nn.Module: The model with pretrained weights.
    """

    use_flash_attn = getattr(model, "use_fast_transformer", True)
    if not use_flash_attn:
        pretrained_params = {
            k.replace("Wqkv.", "in_proj_"): v for k, v in pretrained_params.items()
        }

    if prefix is not None and len(prefix) > 0:
        if isinstance(prefix, str):
            prefix = [prefix]
        pretrained_params = {
            k: v
            for k, v in pretrained_params.items()
            if any(k.startswith(p) for p in prefix)
        }

    model_dict = model.state_dict()
    if strict:
        if verbose:
            for k, v in pretrained_params.items():
                logger.info(f"Loading parameter {k} with shape {v.shape}")
        model_dict.update(pretrained_params)
        model.load_state_dict(model_dict)
    else:
        if verbose:
            for k, v in pretrained_params.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    logger.info(f"Loading parameter {k} with shape {v.shape}")
        pretrained_params = {
            k: v
            for k, v in pretrained_params.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_params)
        model.load_state_dict(model_dict)

    return model


class DownstreamTasks(Enum):
    Perturbation = "perturbation"
    Identification = "identification"
    BatchCorrection = "batchcorrection"
    CellPopulationDiscovery = "cellpopulationdiscovery"


def freeze_parameters(
        model: torch.nn.Module,
        task: DownstreamTasks,
):
    """
    Load pretrained weights to the model.
    Freeze the specific parameters of the model when using the Parameter-Efficient Fine-Tuning

    Args:
        model (torch.nn.Module): The model to load weights to.
        pretrained_params (DownstreamTasks): The downstream task currently being executed.

    Examples::
        >>> freeze_parameters(model, DownstreamTasks.Identification)
    """
    if any('lora' in param_key for param_key in model.state_dict().keys()):
        import loralib as lora
        lora.mark_only_lora_as_trainable(model, bias='lora_only')
    else:
        for param in model.parameters():
            param.requires_grad = False

    # Set requires_grad to True for parameters containing any of the specified peft keywords.
    keywords = ('lora', 'adapter')
    params_to_update = filter(lambda p: any(keyword in p[0] for keyword in keywords), model.named_parameters())
    for _, param in params_to_update:
        param.requires_grad = True

    modules_to_update = None
    if task == DownstreamTasks.Identification:
        modules_to_update = [
            model.cls_decoder.parameters()
        ]

    if task == DownstreamTasks.BatchCorrection:
        modules_to_update = [
            model.decoder.parameters(),
            model.mvc_decoder.parameters(),
            model.grad_reverse_discriminator.parameters(),
            model.batch_encoder.parameters(),
            model.dsbn.parameters()
        ]

    if task == DownstreamTasks.CellPopulationDiscovery:
        modules_to_update = [
            model.decoder.parameters(),
            model.mvc_decoder.parameters(),
        ]

    if task == DownstreamTasks.Perturbation:
        modules_to_update = [
            model.decoder.parameters(),
            # model.encoder.parameters(),
            model.value_encoder.parameters(),
            model.pert_encoder.parameters(),
        ]

    assert modules_to_update
    all_parameters = [param for module in modules_to_update for param in module]
    for param in all_parameters:
        param.requires_grad = True


# Wrapper for all scib metrics, we leave out some metrics like hvg_score, cell_cyvle,
# trajectory_conservation, because we only evaluate the latent embeddings here and
# these metrics are evaluating the reconstructed gene expressions or pseudotimes.
def eval_scib_metrics(
        adata: AnnData,
        batch_key: str = "str_batch",
        label_key: str = "celltype",
        notes: Optional[str] = None,
) -> Dict:
    import scib

    results = scib.metrics.metrics(
        adata,
        adata_int=adata,
        batch_key=batch_key,
        label_key=label_key,
        embed="X_scGPT",
        isolated_labels_asw_=False,
        silhouette_=True,
        hvg_score_=False,
        graph_conn_=True,
        pcr_=True,
        isolated_labels_f1_=False,
        trajectory_=False,
        nmi_=True,  # use the clustering, bias to the best matching
        ari_=True,  # use the clustering, bias to the best matching
        cell_cycle_=False,
        kBET_=False,  # kBET return nan sometimes, need to examine
        ilisi_=False,
        clisi_=False,
    )

    if notes is not None:
        logger.info(f"{notes}")

    logger.info(f"{results}")

    result_dict = results[0].to_dict()
    logger.info(
        "Biological Conservation Metrics: \n"
        f"ASW (cell-type): {result_dict['ASW_label']:.4f}, graph cLISI: {result_dict['cLISI']:.4f}, "
        f"isolated label silhouette: {result_dict['isolated_label_silhouette']:.4f}, \n"
        "Batch Effect Removal Metrics: \n"
        f"PCR_batch: {result_dict['PCR_batch']:.4f}, ASW (batch): {result_dict['ASW_label/batch']:.4f}, "
        f"graph connectivity: {result_dict['graph_conn']:.4f}, graph iLISI: {result_dict['iLISI']:.4f}"
    )

    result_dict["avg_bio"] = np.mean(
        [
            result_dict["NMI_cluster/label"],
            result_dict["ARI_cluster/label"],
            result_dict["ASW_label"],
        ]
    )

    # remove nan value in result_dict
    result_dict = {k: v for k, v in result_dict.items() if not np.isnan(v)}

    return result_dict


def compute_perturbation_metrics(
        results: Dict,
        ctrl_adata: AnnData,
        non_zero_genes: bool = False,
        return_raw: bool = False,
) -> Dict:
    """
    Given results from a model run and the ground truth, compute metrics

    Args:
        results (:obj:`Dict`): The results from a model run
        ctrl_adata (:obj:`AnnData`): The adata of the control condtion
        non_zero_genes (:obj:`bool`, optional): Whether to only consider non-zero
            genes in the ground truth when computing metrics
        return_raw (:obj:`bool`, optional): Whether to return the raw metrics or
            the mean of the metrics. Default is False.

    Returns:
        :obj:`Dict`: The metrics computed
    """
    from scipy.stats import pearsonr

    # metrics:
    #   Pearson correlation of expression on all genes, on DE genes,
    #   Pearson correlation of expression change on all genes, on DE genes,

    metrics_across_genes = {
        "pearson": [],
        "pearson_de": [],
        "pearson_delta": [],
        "pearson_de_delta": [],
    }

    metrics_across_conditions = {
        "pearson": [],
        "pearson_delta": [],
    }

    conditions = np.unique(results["pert_cat"])
    assert not "ctrl" in conditions, "ctrl should not be in test conditions"
    condition2idx = {c: np.where(results["pert_cat"] == c)[0] for c in conditions}

    mean_ctrl = np.array(ctrl_adata.X.mean(0)).flatten()  # (n_genes,)
    assert ctrl_adata.X.max() <= 1000, "gene expression should be log transformed"

    true_perturbed = results["truth"]  # (n_cells, n_genes)
    assert true_perturbed.max() <= 1000, "gene expression should be log transformed"
    true_mean_perturbed_by_condition = np.array(
        [true_perturbed[condition2idx[c]].mean(0) for c in conditions]
    )  # (n_conditions, n_genes)
    true_mean_delta_by_condition = true_mean_perturbed_by_condition - mean_ctrl
    zero_rows = np.where(np.all(true_mean_perturbed_by_condition == 0, axis=1))[
        0
    ].tolist()
    zero_cols = np.where(np.all(true_mean_perturbed_by_condition == 0, axis=0))[
        0
    ].tolist()

    pred_perturbed = results["pred"]  # (n_cells, n_genes)
    pred_mean_perturbed_by_condition = np.array(
        [pred_perturbed[condition2idx[c]].mean(0) for c in conditions]
    )  # (n_conditions, n_genes)
    pred_mean_delta_by_condition = pred_mean_perturbed_by_condition - mean_ctrl

    def corr_over_genes(x, y, conditions, res_list, skip_rows=[], non_zero_mask=None):
        """compute pearson correlation over genes for each condition"""
        for i, c in enumerate(conditions):
            if i in skip_rows:
                continue
            x_, y_ = x[i], y[i]
            if non_zero_mask is not None:
                x_ = x_[non_zero_mask[i]]
                y_ = y_[non_zero_mask[i]]
            res_list.append(pearsonr(x_, y_)[0])

    corr_over_genes(
        true_mean_perturbed_by_condition,
        pred_mean_perturbed_by_condition,
        conditions,
        metrics_across_genes["pearson"],
        zero_rows,
        non_zero_mask=true_mean_perturbed_by_condition != 0 if non_zero_genes else None,
    )
    corr_over_genes(
        true_mean_delta_by_condition,
        pred_mean_delta_by_condition,
        conditions,
        metrics_across_genes["pearson_delta"],
        zero_rows,
        non_zero_mask=true_mean_perturbed_by_condition != 0 if non_zero_genes else None,
    )

    def find_DE_genes(adata, condition, geneid2idx, non_zero_genes=False, top_n=20):
        """
        Find the DE genes for a condition
        """
        key_components = next(
            iter(adata.uns["rank_genes_groups_cov_all"].keys())
        ).split("_")
        assert len(key_components) == 3, "rank_genes_groups_cov_all key is not valid"

        condition_key = "_".join([key_components[0], condition, key_components[2]])

        de_genes = adata.uns["rank_genes_groups_cov_all"][condition_key]
        if non_zero_genes:
            de_genes = adata.uns["top_non_dropout_de_20"][condition_key]
            # de_genes = adata.uns["rank_genes_groups_cov_all"][condition_key]
            # de_genes = de_genes[adata.uns["non_zeros_gene_idx"][condition_key]]
            # assert len(de_genes) > top_n

        de_genes = de_genes[:top_n]

        de_idx = [geneid2idx[i] for i in de_genes]

        return de_idx, de_genes

    geneid2idx = dict(zip(ctrl_adata.var.index.values, range(len(ctrl_adata.var))))
    de_idx = {
        c: find_DE_genes(ctrl_adata, c, geneid2idx, non_zero_genes)[0]
        for c in conditions
    }
    mean_ctrl_de = np.array(
        [mean_ctrl[de_idx[c]] for c in conditions]
    )  # (n_conditions, n_diff_genes)

    true_mean_perturbed_by_condition_de = np.array(
        [
            true_mean_perturbed_by_condition[i, de_idx[c]]
            for i, c in enumerate(conditions)
        ]
    )  # (n_conditions, n_diff_genes)
    zero_rows_de = np.where(np.all(true_mean_perturbed_by_condition_de == 0, axis=1))[
        0
    ].tolist()
    true_mean_delta_by_condition_de = true_mean_perturbed_by_condition_de - mean_ctrl_de

    pred_mean_perturbed_by_condition_de = np.array(
        [
            pred_mean_perturbed_by_condition[i, de_idx[c]]
            for i, c in enumerate(conditions)
        ]
    )  # (n_conditions, n_diff_genes)
    pred_mean_delta_by_condition_de = pred_mean_perturbed_by_condition_de - mean_ctrl_de

    corr_over_genes(
        true_mean_perturbed_by_condition_de,
        pred_mean_perturbed_by_condition_de,
        conditions,
        metrics_across_genes["pearson_de"],
        zero_rows_de,
    )
    corr_over_genes(
        true_mean_delta_by_condition_de,
        pred_mean_delta_by_condition_de,
        conditions,
        metrics_across_genes["pearson_de_delta"],
        zero_rows_de,
    )

    if not return_raw:
        for k, v in metrics_across_genes.items():
            metrics_across_genes[k] = np.mean(v)
        for k, v in metrics_across_conditions.items():
            metrics_across_conditions[k] = np.mean(v)
    metrics = metrics_across_genes

    return metrics


# wrapper to make sure all methods are called only on the main process
def main_process_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if os.environ.get("LOCAL_RANK", "0") == "0":
            return func(*args, **kwargs)

    return wrapper


# class wrapper to make sure all methods are called only on the main process
class MainProcessOnly:
    def __init__(self, obj):
        self.obj = obj

    def __getattr__(self, name):
        attr = getattr(self.obj, name)

        if callable(attr):
            attr = main_process_only(attr)

        return attr
