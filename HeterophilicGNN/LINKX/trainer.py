import torch
import numpy as np
import os
import copy

# ---- Models ----
from models import LINKXC as LINKX
from models import H2GCN_Improved as H2GCN
from models import build_dense_adjacency as build_adjacency

# ---- Train functions ----
from train_functions import train_linkx, evaluate_linkx, evaluate_loss_linkx
from train_functions import train_gnn, evaluate_gnn, evaluate_loss_gnn, build_A_batch

# ---- Dataset ----
from dataset import load_nc_dataset
from data_utils import load_fixed_splits
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# ----- args --------
import argparse

# ------- Grid search ---
from itertools import product
from tqdm import tqdm
from types import SimpleNamespace

def get_args():
    parser = argparse.ArgumentParser(description="LINKX Training Config")

    parser.add_argument('--model_name', type=str, default='LINKX')
    parser.add_argument('--save_dir', type=str, default='results')

    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-3)

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=64)

    parser.add_argument('--patience', type=int, default=50)

    parser.add_argument('--ignoreA', action='store_true')
    parser.add_argument('--ignoreX', action='store_true')
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--sub_dataname', type=str, default='')

    return parser.parse_args()

class NodeDataset(Dataset):
    def __init__(self, nodes):
        self.nodes = nodes

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        return self.nodes[idx]


def nc_to_data(dataset, dataset_name, device):
    graph, label = dataset[0]

    x = graph['node_feat'].to(device)
    edge_index = graph['edge_index'].to(device)
    y = label.squeeze().to(device)

    num_nodes = graph['num_nodes']

    # split_idx = dataset.get_idx_split()
    splits_lst = load_fixed_splits("wisconsin", None)
    print("split_lst size inside nc_to_data:",len(splits_lst))
    for i, split_idx in enumerate(splits_lst):
        # print(f"\n--- Split {i} ---")

        train_idx = split_idx['train']
        val_idx   = split_idx['valid']
        test_idx  = split_idx['test']
        
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask   = torch.zeros_like(train_mask)
    test_mask  = torch.zeros_like(train_mask)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # remove invalid labels
    valid = y >= 0
    train_mask &= valid
    val_mask &= valid
    test_mask &= valid

    class Data: pass

    data = Data()
    data.x = x
    data.edge_index = edge_index
    data.y = y
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.num_nodes = num_nodes
    data.num_features = x.size(1)

    return data

def rand_train_test_idx(labels, train_prop=0.6, val_prop=0.2, seed=None):
    """
    Randomly splits nodes into train/val/test indices.
    """

    if seed is not None:
        torch.manual_seed(seed)

    n = labels.shape[0]
    perm = torch.randperm(n)

    train_end = int(train_prop * n)
    val_end   = int((train_prop + val_prop) * n)

    train_idx = perm[:train_end]
    val_idx   = perm[train_end:val_end]
    test_idx  = perm[val_end:]

    return train_idx, val_idx, test_idx

# =================================================================================================================================

def run_experiment(dataset_name, args):

    print('Cuda available:', torch.cuda.is_available())
    print(f"\n===== DATASET: {dataset_name} =====")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = load_nc_dataset(dataset_name, args.sub_dataname)
    data_ = nc_to_data(dataset, dataset_name, device)
    
    labels = data_.y
    unique_labels = torch.unique(labels)

    label_map = {old.item(): i for i, old in enumerate(unique_labels)}

    data_.y = torch.tensor([label_map[x.item()] for x in labels], device=labels.device)

    num_classes = int(data_.y.max().item()) + 1
    
    print("Num classes:", num_classes)
    print("Label min:", data_.y.min().item())
    print("Label max:", data_.y.max().item())

    # Normalize features
    data_.x = data_.x / data_.x.sum(dim=1, keepdim=True).clamp(min=1)

    num_nodes = data_.num_nodes

    os.makedirs(args.save_dir, exist_ok=True)

    all_acc = []

    print("Model Name:", args.model_name)
    print(f"lr : {args.lr} | weight_decay : {args.weight_decay} | dropout : {args.dropout} | batch_size = {args.batch_size} | hidden_dim : {args.hidden_dim}|patience: {args.patience}| Epochs: {args.epochs}")

    # ---------------- SPLIT HANDLING ----------------
    name = dataset_name
    if args.sub_dataname and args.sub_dataname != 'None':
        name += f'-{args.sub_dataname}'

    split_path = f'./data/splits/{name}-splits.npy'

    if os.path.exists(split_path):
        splits_lst = load_fixed_splits(dataset_name, args.sub_dataname)
        num_runs = len(splits_lst)
        print(f"Using FIXED splits: {num_runs}")
    else:
        splits_lst = None
        num_runs = 10  # standard for fb100
        print(f"Using RANDOM splits: {num_runs}")

    # ---------------- MAIN LOOP ----------------
    for i in range(num_runs):
        print(f"\n--- Run {i} ---")

        # ----- Get split -----
        if splits_lst is not None:
            split_idx = splits_lst[i]
            train_idx = split_idx['train']
            val_idx   = split_idx['valid']
            test_idx  = split_idx['test']
        else:
            train_idx, val_idx, test_idx = rand_train_test_idx(data_.y)

        # ----- Fix dtype + device -----
        train_idx = train_idx.long().to(device)
        val_idx   = val_idx.long().to(device)
        test_idx  = test_idx.long().to(device)

        # ----- Safety checks -----
        assert train_idx.max() < num_nodes
        assert val_idx.max() < num_nodes
        assert test_idx.max() < num_nodes

        # ----- Create masks -----
        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        val_mask   = torch.zeros_like(train_mask)
        test_mask  = torch.zeros_like(train_mask)

        train_mask[train_idx] = True
        val_mask[val_idx]     = True
        test_mask[test_idx]   = True

        # ----- DataLoader -----
        train_nodes = train_mask.nonzero(as_tuple=True)[0]

        train_loader = DataLoader(
            NodeDataset(train_nodes),
            batch_size=1024,
            shuffle=True
        )

        # ----- Model -----
        if args.model_name == 'LINKX':
            model = LINKX(
                feat_dim=data_.num_features,
                hidden_dim=args.hidden_dim,
                num_nodes=num_nodes,
                num_classes=num_classes,
                ignoreA=args.ignoreA,
                ignoreX=args.ignoreX,
                dropout=args.dropout
            ).to(device)

        elif args.model_name == 'H2GCN':
            model = H2GCN(
                in_dim=data_.num_features,
                hidden_dim=args.hidden_dim,
                num_classes=num_classes,
                num_layers=2,
                dropout=args.dropout
            ).to(device)
        else:
            raise ValueError("Unknown model")

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)

        train_losses, val_losses, val_accs = [], [], []

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        # ---------------- TRAINING ----------------
        for epoch in range(args.epochs):

            if args.model_name == 'LINKX':
                train_loss = train_linkx(model, data_, train_loader, optimizer, scheduler, device, args.debug)
                val_loss   = evaluate_loss_linkx(model, data_, val_mask, args.batch_size)
                val_acc    = evaluate_linkx(model, data_, val_mask, args.batch_size)

            else:
                train_loss = train_gnn(model, data_, train_mask, optimizer, args.debug)
                val_loss   = evaluate_loss_gnn(model, data_, val_mask)
                val_acc    = evaluate_gnn(model, data_, val_mask)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            # ----- Early stopping -----
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # ----- Load best -----
        if best_state is not None:
            model.load_state_dict(best_state)

        # ----- Test -----
        if args.model_name == 'LINKX':
            test_acc = evaluate_linkx(model, data_, test_mask, args.batch_size)
        else:
            test_acc = evaluate_gnn(model, data_, test_mask)

        print(f"Test Accuracy: {test_acc:.4f}")
        all_acc.append(test_acc)

        # ----- Save -----
        torch.save({
            "dataset": dataset_name,
            "split": i,
            "train_loss": train_losses,
            "val_loss": val_losses,
            "val_acc": val_accs,
            "test_acc": test_acc
        }, f"{args.save_dir}/{dataset_name}_{args.model_name}_split_{i}.pt")

    print(f"\n{dataset_name.upper()} → Avg Accuracy: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")

    return np.mean(all_acc)


# ---- Entry ----
if __name__ == '__main__':
    torch.manual_seed(42)

#  =========== Run by custom args ==========
    datasets = [
        # "arxiv-year",
        # "Penn94",
        'fb100',
        # "actor",
        # "texas",
        # "wisconsin",
        # "chameleon",
    ]
    args = get_args()

    for name in datasets:
        run_experiment(
            name,
            args
        )

    
# ====== Grid Search =======================

    # data_name = "arxiv-year"
    # search_space = {
    #     "lr": [5e-4, 1e-3, 2e-3],              # around your current LR
    #     "weight_decay": [5e-3, 8e-3, 1e-2],    # around your WD
    #     "hidden_dim": [64, 128],              # model capacity
    #     "dropout": [0.4, 0.5, 0.6]            # regularization tuning
    # }
    # search_space = {
    #     "lr": [5e-4, 1e-3, 2e-3],
    #     "weight_decay": [5e-3, 8e-3, 1e-2],
    #     "hidden_dim": [64, 128],
    #     "dropout": [0.4, 0.5, 0.6]
    # }

    # base_config = {
    # "model_name": "LINKX",
    # "save_dir": "results",
    # "epochs": 500,
    # "batch_size": 1024,
    # "patience": 300,
    # "ignoreA": False,
    # "ignoreX": False,
    # "debug": False
    # }

    # keys = list(search_space.keys())
    # values = list(search_space.values())
    # combinations = list(product(*values))

    # results = []

    # pbar = tqdm(combinations, desc="Grid Search")

    # for combo in pbar:
    #     config = base_config.copy()
    #     config.update(dict(zip(keys, combo)))

    #     pbar.set_postfix({
    #         "lr": config["lr"],
    #         "wd": config["weight_decay"]
    #     })

    #     args = SimpleNamespace(**config)

    #     val_acc = run_experiment(data_name, args)

    #     results.append({
    #         "config": config,
    #         "val_acc": val_acc
    #     })

# # sort
# results = sorted(results, key=lambda x: x["val_acc"], reverse=True)

# print("\nTop configs:")
# for r in results[:5]:
#     print(r)