import argparse
import copy
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from performer_pytorch.performer_pytorch_prompt import PerformerLM
import scanpy as sc
from utils import *
from Benckmark_utils import scBERT_preprocess
import warnings
from sklearn.metrics import f1_score, balanced_accuracy_score

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default='NSCLC', help='NSCLC/COVID/ms')
parser.add_argument("--data_path", type=str, default='../data/celltype_annoration/', help='Path of data for finetune.')
parser.add_argument("--model_path", type=str, default='../scBERT_ckpts', help='Path of pretrained model.')
parser.add_argument("--batch_size", type=int, default=1, help='Number of batch size.')
parser.add_argument("--pretrain_data_path", type=str, default='./panglao_human.h5ad', help='Path of pretrain data.')
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')  # 2000
parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
parser.add_argument("--prompt_type", type=str, default='prefix_prompt', help='prefix_prompt')
parser.add_argument("--tokens", type=str, default=64, help='prefix token number')
parser.add_argument("--space_conf", type=str, default=[1, 1, 1, 1, 1, 1],
                    help='encoder space adapter list')  # [0，0，0, 1, 1, 1]
parser.add_argument("--mlp_conf", type=str, default=[0, 0, 0, 0, 0, 0], help='encoder mlp adapter list')
parser.add_argument("--mlp_ratio", type=float, default=0.25, help='mlp_ratio.')

args = parser.parse_args()
prompt_type = args.prompt_type
SEED = args.seed
BATCH_SIZE = args.batch_size
SEQ_LEN = args.gene_num + 1

UNASSIGN_THRES = 0.0
CLASS = args.bin_num + 2
POS_EMBED_USING = args.pos_embed
data_name = args.data_name
data_path = args.data_path
pretrain_data = args.pretrain_data_path
device = torch.device(f"cuda:0")
tokens = args.tokens
space_conf = args.space_conf
mlp_conf = args.mlp_conf
mlp_ratio = args.mlp_ratio


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
        if prompt_type == "prefix_prompt":
            self.fc1 = nn.Linear(in_features=SEQ_LEN + 64, out_features=512, bias=True)
            # self.fc1 = nn.Linear(in_features=sum(gene_mask)+1 + 64, out_features=512, bias=True)

        else:
            self.fc1 = nn.Linear(in_features=SEQ_LEN, out_features=512, bias=True)
            # self.fc1 = nn.Linear(in_features=sum(gene_mask)+1, out_features=512, bias=True)

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
        # print(x.shape)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


for fold_idx in range(5):

    data_train = sc.read_h5ad(f'{data_path}/{data_name}/{str(fold_idx)}/{data_name}_train{str(fold_idx)}.h5ad')
    data_val = sc.read_h5ad(f'{data_path}/{data_name}/{str(fold_idx)}/{data_name}_val{str(fold_idx)}.h5ad')
    data_test = sc.read_h5ad(f'{data_path}/{data_name}/{str(fold_idx)}/{data_name}_test{str(fold_idx)}.h5ad')

    data_train.obs["str_batch"] = "0"
    data_test.obs["str_batch"] = "1"
    data_val.obs["str_batch"] = "2"
    data_train = data_train.concatenate((data_test, data_val), batch_key="str_batch")

    if data_name == 'ms':
        data_is_raw = False
        celltype_key = 'celltype'
        data_train.var_names = data_train.var["gene_name"]
    elif data_name == 'COVID':
        data_is_raw = True
        celltype_key = 'cell_type'
    elif data_name == 'NSCLC':
        data_is_raw = True
        celltype_key = 'cell_type'

    panglao = sc.read_h5ad(pretrain_data)
    preprocess = scBERT_preprocess(panglao, data_train)
    data_train = preprocess(data_is_raw=data_is_raw)
    gene_mask = data_train.uns["gene_mask"]
    del panglao

    label_dict, _ = np.unique(np.array(data_train.obs[celltype_key]), return_inverse=True)
    data_test = data_train[data_train.obs["str_batch"] == "1"]

    _, label_test = np.unique(np.array(data_test.obs[celltype_key]), return_inverse=True)
    label_test = torch.from_numpy(label_test)

    data_test = data_test.X
    test_dataset = SCDataset(data_test, label_test)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
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
        gene_mask=gene_mask,
        mlp_ratio=mlp_ratio,
    )

    path = args.model_path
    path = f"{path}/{data_name}/scPEFT/{fold_idx}/best_model.pt"
    ckpt = torch.load(path)
    model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])

    model.load_state_dict(ckpt, True)
    model = model.to(device)


    def test(model: nn.Module, test_loader: DataLoader) -> float:
        model.eval()
        predictions = []
        truths = []
        for index, (data_t, labels_t) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Inference",
                                              leave=True):
            index += 1
            data_t, labels_t = data_t.to(device), labels_t.to(device)
            logits = model(data_t)
            softmax = nn.Softmax(dim=-1)
            final_prob = softmax(logits)
            final = final_prob.argmax(dim=-1)
            final[np.amax(np.array(final_prob.detach().cpu().numpy()), axis=-1) < UNASSIGN_THRES] = -1
            predictions.append(final.detach().cpu().numpy())
            truths.append(labels_t.detach().cpu().numpy())
        predictions = np.concatenate(predictions, axis=0)
        truths = np.concatenate(truths, axis=0)
        return predictions, truths


    predictions, celltypes_labels = test(model, test_loader)
    balanced_accuracy = balanced_accuracy_score(celltypes_labels, predictions)
    f1 = f1_score(celltypes_labels, predictions, average="macro")

    print(f"fold_{fold_idx}: Accuracy: {balanced_accuracy:.3f}, "f"macro F1: {f1:.3f}", flush=True)
