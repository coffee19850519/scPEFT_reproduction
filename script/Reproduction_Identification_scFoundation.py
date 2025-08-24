import argparse
import random,os
import numpy as np
import pandas as pd
import argparse
import torch
from tqdm import tqdm
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.nn import functional as F
import sys
sys.path.insert(0, "../")
from torch.utils.data import DataLoader, Dataset
import scipy.sparse
from scipy.sparse import issparse
import scanpy as sc
from scfoundation.annotation.load import *
from torch.utils.data import DataLoader, Dataset
import copy
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, \
    classification_report, balanced_accuracy_score
import warnings
from sklearn.metrics import classification_report
import time
import pickle
warnings.filterwarnings("ignore")
####################################Settings#################################
parser = argparse.ArgumentParser(description='Drug_response_pre')

parser.add_argument('--data_path', type=str, default='./data/ms/0/ms_test0.h5ad', help='input data path')
parser.add_argument("--data_name", type=str, default='ms',help='NSCLC/COVID/ms')
parser.add_argument('--model_path',  type=str, default='None', help='pre-trained model path')
parser.add_argument('--save_path', type=str, default='./results/', help='save path')
parser.add_argument("--use_prompt", type=bool, default=True)
parser.add_argument("--peft_type", type=str, default='Encoder_adapter',help=' Encoder_adapter/ Token_adapter / Prefix / LoRA / finetune')
parser.add_argument("--fold_idx", type=str, default='1')


parser.add_argument('--task_name', type=str, default='deepcdr', help='task name')
parser.add_argument('--input_type', type=str, default='singlecell',choices=['singlecell','bulk'], help='input type; default: singlecell')
parser.add_argument('--output_type', type=str, default='gene',choices=['cell','gene','gene_batch','gene_expression'], help='cell or gene embedding; default: cell the difference between gene and gene_batch is that in gene mode the gene embedding will be processed one by one. while in gene_batch mode, the gene embedding will be processed in batch. GEARS use gene_batch mode.')
parser.add_argument('--pool_type', type=str, default='all',choices=['all','max'], help='pooling type of cell embedding; default: all only valid for output_type=cell')
parser.add_argument('--tgthighres', type=str, default='t4', help='the targeted high resolution (start with t) or the fold change of the high resolution (start with f), or the addtion (start with a) of the high resoultion. only valid for input_type=singlecell')
parser.add_argument('--pre_normalized', type=str, default='F',choices=['F','T','A'], help='if normalized before input; default: False (F). choice: True(T), Append(A) When input_type=bulk: pre_normalized=T means log10(sum of gene expression). pre_normalized=F means sum of gene expression without normalization. When input_type=singlecell: pre_normalized=T or F means gene expression is already normalized+log1p or not. pre_normalized=A means gene expression is normalized and log1p transformed. the total count is appended to the end of the gene expression matrix.')
parser.add_argument('--demo', action='store_true', default=False, help='if demo, only infer 10 samples')
parser.add_argument('--version',  type=str, default='ce', help='only valid for output_type=cell. For read depth enhancemnet, version=rde For others, version=ce')
parser.add_argument('--ckpt_name',  type=str, default='01B-resolution', help='checkpoint name')
parser.add_argument("--batch_size", type=int, default=16, help='Number of batch size.')
parser.add_argument("--epoch", type=int, default=1, help='Number of epochs.')
parser.add_argument("--learning_rate", type=float, default=5e-4, help='Learning rate.')

args = parser.parse_args()

n_layers_conf=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # token
mlp_adapter_conf=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
space_adapter_conf=[1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1]
# mlp_adapter_conf=[0,0,0,0,0,0, 1, 1, 1, 1, 1, 1]
# space_adapter_conf=[0,0,0,0,0,0, 1, 1, 1, 1, 1, 1]
peft_prompt_relationship = {
    "Encoder_adapter": "encoder-prompt",
    "Token_adapter": "head-prompt",
    "Prefix": "prefix-prompt",
    "LoRA": "LoRA",
    "finetune": "finetune"
}

prompt_type = peft_prompt_relationship[args.peft_type]
prompt_settings = {
    "use_prompt": args.use_prompt,
    "num_tokens": 64,
    "prompt_type": prompt_type,
    "n_layers_conf": n_layers_conf,
    "mlp_adapter_conf": mlp_adapter_conf,
    "space_adapter_conf": space_adapter_conf
}
if args.data_name == 'ms':
    args.pre_normalized = 'T'
    print("pre normalized",args.pre_normalized)

ckpt_dir = args.save_path
LEARNING_RATE = args.learning_rate
####################################Settings#################################

def main_gene_selection(X_df, gene_list):
    """
    Describe:
        rebuild the input adata to select target genes encode protein
    Parameters:
        adata->`~anndata.AnnData` object: adata with var index_name by gene symbol
        gene_list->list: wanted target gene
    Returns:
        adata_new->`~anndata.AnnData` object
        to_fill_columns->list: zero padding gene
    """
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    print("mapping gene num:" + str(len(gene_list) - len(to_fill_columns)))
    padding_df = pd.DataFrame(np.zeros((X_df.shape[0], len(to_fill_columns))),
                              columns=to_fill_columns,
                              index=X_df.index)
    X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1),
                        index=X_df.index,
                        columns=list(X_df.columns) + list(padding_df.columns))
    X_df = X_df[gene_list]

    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns, var


class SCDataset(Dataset):
    def __init__(self, adata, gene_list=None,label_to_int=None, transform=None):
        """
        Args:
            args: object with attributes:
                - data_path: path to the .h5ad file
                - pre_normalized: 'T' or 'F', whether the data is already normalized
            gene_list: full list of target genes (length >= 19264)
            transform: optional transform to be applied on a sample
        """
        self.transform = transform
        self.gene_list = gene_list
        self.label_to_int = label_to_int
        # Load and process data
        # adata = sc.read_h5ad(data_path)
        idx = adata.obs_names.tolist()
        # _, label = np.unique(np.array(adata.obs["celltype"]), return_inverse=True)
        label = adata.obs[celltype_key].map(label_to_int).to_numpy()

        try:
            col = adata.var.gene_name.tolist()
        except:
            col = adata.var_names.tolist()

        if issparse(adata.X):
            gexpr = adata.X.toarray()
        else:
            gexpr = adata.X

        gexpr = pd.DataFrame(gexpr, index=idx, columns=col)

        if gexpr.shape[1] < 19264:
            assert self.gene_list is not None, "gene_list must be provided when gene count < 19264"
            gexpr, _, _ = main_gene_selection(gexpr, self.gene_list)
            assert gexpr.shape[1] >= 19264

        if args.pre_normalized == 'F':
            print("preprocess data")
            adata = sc.AnnData(gexpr)
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
            gexpr = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)

        self.features = torch.tensor(gexpr.values, dtype=torch.float32)
        self.label = label  # categorical string, optionally can map to int

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        expression = self.features[idx]
        label = self.label[idx]
        if self.transform:
            expression = self.transform(expression)
        return expression, label

class Adapter(nn.Module):
    def __init__(self, D_features=768, mlp_ratio=0.25, drop_rate=0.5, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.dropout1 = nn.Dropout(drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        # dropout
        xs = self.dropout1(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + self.dropout2(xs)
        else:
            x = self.dropout2(xs)
        return x


class LinearProbingClassifier(nn.Module):

    def __init__(self, ckpt_path,prompt_settings,key,n_class,pool_type,frozenmore=True):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.frozenmore = frozenmore
        self.key = key
        self.prompt_settings = prompt_settings
        self.n_class = n_class
        self.pool_type = pool_type

    def build(self):
        model,model_config = load_model_frommmf(self.ckpt_path, self.prompt_settings, self.key)
        self.token_emb = model.token_emb
        self.pos_emb = model.pos_emb
        self.encoder = model.encoder

        if self.prompt_settings['prompt_type'] == 'head-prompt':
            self.Space_Adapter = Adapter()

        if self.pool_type == 'all':
            self.fc1 = nn.Sequential(
                nn.Linear(model_config['encoder']['hidden_dim']*4, 256),
                nn.ReLU(),
                nn.Linear(256, self.n_class)  # ['n_class']
            )
        elif self.pool_type == 'max':

            self.fc1 = nn.Sequential(
                nn.Linear(model_config['encoder']['hidden_dim'], model_config['encoder']['hidden_dim']),
                nn.LayerNorm(model_config['encoder']['hidden_dim']),
                nn.ReLU(),
                nn.Dropout(),

                nn.Linear(model_config['encoder']['hidden_dim'], model_config['encoder']['hidden_dim']),
                nn.LayerNorm(model_config['encoder']['hidden_dim']),
                nn.ReLU(),
                nn.Dropout(),

                nn.Linear(model_config['encoder']['hidden_dim'], self.n_class)  # ['n_class']
            )
        if self.pool_type == 'all':
            self.norm = torch.nn.BatchNorm1d(model_config['encoder']['hidden_dim']*4, affine=False, eps=1e-6)
            # self.norm = torch.nn.LayerNorm(model_config['encoder']['hidden_dim'] * 4)
        elif self.pool_type == 'max':
            self.norm = torch.nn.BatchNorm1d(model_config['encoder']['hidden_dim'], affine=False, eps=1e-6)
            # self.norm = torch.nn.LayerNorm(model_config['encoder']['hidden_dim'])

        self.model_config = model_config

        keywords = ('lora', 'adapter', 'Adapter','prompt_embeddings')
        for name, para in model.named_parameters():
            para.requires_grad = False
        params_to_update = filter(lambda p: any(keyword in p[0] for keyword in keywords),
                                  model.named_parameters())
        for _, param in params_to_update:
            param.requires_grad = True
        for na, param in self.fc1.named_parameters():
            param.requires_grad = True
        for na, param in self.norm.named_parameters():
            param.requires_grad = True


        
    def forward(self, data, *args, **kwargs):

        x = data # (B, L)
        # print(x.shape)
        value_labels = x > 0


        x, x_padding = gatherData(x, value_labels, self.model_config['pad_token_id'])
        data_gene_ids = torch.arange(19264, device=x.device).repeat(x.shape[0], 1)
        position_gene_ids, _ = gatherData(data_gene_ids, value_labels,
                                        self.model_config['pad_token_id'])
        # print(x.shape, flush=True)
        x = self.token_emb(torch.unsqueeze(x, 2).float(), output_weight = 0)
        position_emb = self.pos_emb(position_gene_ids)
        x += position_emb

        # print(x.shape, flush=True)
        if self.prompt_settings['prompt_type'] == 'head-prompt':
            x = self.Space_Adapter(x)

        logits = self.encoder(x,x_padding)

        # mlp
        geneemb1 = logits[:, -1, :]
        geneemb2 = logits[:, -2, :]
        geneemb3, _ = torch.max(logits[:, :-2, :], dim=1)
        geneemb4 = torch.mean(logits[:,:-2,:], dim=1)
        if self.pool_type == 'all':
            geneembmerge = torch.concat([geneemb1, geneemb2, geneemb3, geneemb4], axis=1)
        elif self.pool_type == 'max':
            geneembmerge, _ = torch.max(logits, dim=1)


        logits = self.fc1(geneembmerge)

        return logits,geneembmerge


class EarlyStopping():
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}", flush=True)
            if self.counter >= self.patience:
                print('INFO: Early stopping', flush=True)
                self.early_stop = True

def compute_class_weights(sample_counts, cap=50):
    max_count = max(sample_counts)
    B = [max(max_count / c, cap) for c in sample_counts]
    B = np.array(B)
    weights = (B / B.sum())*B
    return torch.tensor(weights, dtype=torch.float32)

def test(model: nn.Module, test_loader: DataLoader) -> float:
    # model.eval()
    predictions = []
    truths = []
    all_embeddings = []

    with torch.no_grad():
        for index, (data_t, labels_t) in enumerate(test_loader):
            data_t, labels_t = data_t.to(device), labels_t.to(device)
            logits,embeddings = model(data_t)
            all_embeddings.append(embeddings.cpu())
            softmax = nn.Softmax(dim=-1)
            final_prob = softmax(logits)
            final = final_prob.argmax(dim=-1)
            final[np.amax(np.array(final_prob.detach().cpu().numpy()), axis=-1) < UNASSIGN_THRES] = -1
            predictions.append(final.detach().cpu().numpy())
            truths.append(labels_t.detach().cpu().numpy())
            # tqdm.write(f'Batch {index + 1}/{len(test_loader)}')
    predictions = np.concatenate(predictions, axis=0)
    truths = np.concatenate(truths, axis=0)
    save_dict = {
        "labels": truths,
        "results": predictions,
        "id_maps": unique_labels
    }
    with open(ckpt_dir + f"{args.data_name}_{args.peft_type}_{args.fold_idx}_results.pkl", "wb") as f:
        pickle.dump(save_dict, f)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    np.save(ckpt_dir + f"{args.data_name}_{args.peft_type}_{args.fold_idx}_embedding.npy", all_embeddings.numpy())
    return predictions, truths

if __name__=='__main__':

    random.seed(0)
    np.random.seed(0)  # numpy random generator

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    gene_list_df = pd.read_csv('scfoundation/annotation/OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
    gene_list = list(gene_list_df['gene_name'])

    # Load data
    # load_data(args)

    # Load model
    if args.version == 'noversion':
        ckpt_path = args.model_path
        key = None
    else:
        ckpt_path = './models/models.ckpt'
        if args.output_type == 'cell':
            if args.version == 'ce':
                key = 'cell'
            elif args.version == 'rde':
                key = 'rde'
            else:
                raise ValueError('No version found')
        elif args.output_type == 'gene':
            key = 'gene'
        elif args.output_type == 'gene_batch':
            key = 'gene'
        elif args.output_type == 'gene_expression':  # Not recommended
            key = 'gene'
        else:
            raise ValueError('output_mode must be one of cell gene, gene_batch, gene_expression')
    print(key,flush=True)

    train_datapath = f'{args.data_path}/{args.data_name}/{args.fold_idx}/{args.data_name}_train{args.fold_idx}.h5ad'
    # val_datapath = f'{args.data_path}/{args.data_name}/{args.fold_idx}/{args.data_name}_train{args.fold_idx}.h5ad'
    val_datapath = f'{args.data_path}/{args.data_name}/{args.fold_idx}/{args.data_name}_val{args.fold_idx}.h5ad'
    test_datapath = f'{args.data_path}/{args.data_name}/{args.fold_idx}/{args.data_name}_test{args.fold_idx}.h5ad'
    print(test_datapath)
    if args.data_name == 'ms':
        celltype_key = 'celltype'
    elif args.data_name == 'COVID':
        celltype_key = 'cell_type'
    elif args.data_name == 'NSCLC':
        celltype_key = 'cell_type'

    train_adata = sc.read_h5ad(train_datapath)
    val_adata = sc.read_h5ad(val_datapath)
    test_adata = sc.read_h5ad(test_datapath)

    unique_labels = np.unique(train_adata.obs[celltype_key])
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

    train_dataset = SCDataset(train_adata, gene_list,label_to_int)
    val_dataset = SCDataset(val_adata, gene_list,label_to_int)
    test_dataset = SCDataset(test_adata, gene_list,label_to_int)
    print('testing data number:',len(test_dataset))

    train_num = len(train_dataset)
    train_class_num = np.unique(train_dataset.label, return_counts=True)[1]
    n_class = len(np.unique(train_dataset.label))
    print(f"n_class: {n_class}", flush=True)



    sample_weights = 1.0 / train_class_num[train_dataset.label]
    sample_weights = sample_weights / np.sum(sample_weights)


    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, train_num, replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = LinearProbingClassifier(ckpt_path=ckpt_path,prompt_settings = prompt_settings,key= key,n_class=n_class,pool_type = args.pool_type)
    model_path = ckpt_dir + args.data_name + '_' + args.peft_type + '_' + args.fold_idx + '_' + args.pre_normalized + '_' + args.output_type + '_' + f"best_model.pt"
    print(model_path)
    model.build()
    model = model.to(device)

    # for name, param in model.named_parameters():
    #     print(name, torch.mean(param.data), torch.std(param.data))
    model.load_state_dict(torch.load(model_path))
    # for name, param in model.named_parameters():
    #     print(name, torch.mean(param.data), torch.std(param.data))

    model.eval()

    print(model)

    print("-" * 89, flush=True)

    max_acc = 0.0
    best_val_loss = float("inf")
    UNASSIGN_THRES = 0.0
    softmax = nn.Softmax(dim=-1)


    predictions, celltypes_labels = test(model, test_loader)

    from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score,accuracy_score

    accuracy = accuracy_score(celltypes_labels, predictions)
    print(f"accuracy: {accuracy:.3f}",flush=True)

    balanced_accuracy = balanced_accuracy_score(celltypes_labels, predictions)
    f1 = f1_score(celltypes_labels, predictions, average="macro")
    precision = precision_score(celltypes_labels, predictions, average="macro")
    recall = recall_score(celltypes_labels, predictions, average="macro")

    print(
        f"macro Accuracy: {balanced_accuracy:.3f}, macro Precision: {precision:.3f},macro Recall: {recall:.3f}, "f"macro F1: {f1:.3f}",
        flush=True)
    micro_f1 = f1_score(celltypes_labels, predictions, average="micro")
    micro_precision = precision_score(celltypes_labels, predictions, average="micro")
    micro_recall = recall_score(celltypes_labels, predictions, average="micro")
    print(
        f"micro Accuracy: {balanced_accuracy:.3f}, micro Precision: {micro_precision:.3f},micro Recall: {micro_recall:.3f}, "f"micro F1: {micro_f1:.3f}",
        flush=True)








