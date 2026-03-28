import torch
import copy
import os
import numpy as np
from torch_geometric.datasets import WebKB, WikipediaNetwork, Planetoid, Actor, CoraFull
from models import MultiHopGNN, GCN
from models import H2GCN_Improved as H2GCN
from train_functions import train, evaluate, evaluate_loss


def load_dataset(name):
    name = name.lower()

    if name in ["cornell", "texas", "wisconsin"]:
        dataset = WebKB(root=f"data/{name}", name=name.capitalize())

    elif name in ["chameleon", "squirrel"]:
        dataset = WikipediaNetwork(root=f"data/{name}", name=name)

    elif name == "actor":
        dataset = Actor(root="data/actor")

    elif name in ["cora", "citeseer", "pubmed"]:
        dataset = Planetoid(root=f"data/{name}", name=name.capitalize())

    elif name == "cora_full":
        dataset = CoraFull(root="data/cora_full")

    else:
        raise ValueError(f"Unknown dataset: {name}")

    return dataset[0], dataset.num_classes


def run_experiment(dataset_name, model_name='H2GCN'):
    print(f"\n===== DATASET: {dataset_name} =====")

    data, num_classes = load_dataset(dataset_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    num_splits = data.train_mask.shape[1] if data.train_mask.dim() > 1 else 1

    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    all_acc = []

    for i in range(num_splits):
        print(f"\n--- Split {i} ---")

        # Initialize model
        if model_name == 'MultiHopGNN':
            model = MultiHopGNN(
                in_dim=data.num_features,
                hidden_dim=64,
                num_classes=num_classes,
                num_hops=2,
                dropout=0.5
            ).to(device)

        elif model_name == 'H2GCN':
            model = H2GCN(
                in_dim=data.num_features,
                hidden_dim=128,
                num_classes=num_classes,
                num_layers=2,
                dropout=0.1
            ).to(device)

        elif model_name == 'GCN':
            model = GCN(
                in_dim=data.num_features,
                hidden_dim=64,
                num_classes=num_classes,
                dropout=0.1
            ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

        train_mask = data.train_mask[:, i] if num_splits > 1 else data.train_mask
        val_mask   = data.val_mask[:, i]   if num_splits > 1 else data.val_mask
        test_mask  = data.test_mask[:, i]  if num_splits > 1 else data.test_mask

        # Logging containers
        train_losses = []
        val_losses = []
        val_accs = []

        best_val_loss = float('inf')
        patience = 100
        patience_counter = 0
        best_state = None

        for epoch in range(200):

            train_loss = train(model, data, train_mask, optimizer)
            val_loss = evaluate_loss(model, data, val_mask)
            val_acc = evaluate(model, data, val_mask)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)

        test_acc = evaluate(model, data, test_mask)
        print(f"Test Accuracy: {test_acc:.4f}")

        all_acc.append(test_acc)

        # Save results
        torch.save({
            "dataset": dataset_name,
            "split": i,
            "train_loss": train_losses,
            "val_loss": val_losses,
            "val_acc": val_accs,
            "test_acc": test_acc
        }, f"{save_dir}/{dataset_name}_split_{i}.pt")

    print(f"\n{dataset_name.upper()} → Avg Accuracy: {np.mean(all_acc):.4f}")


if __name__ == '__main__':
    torch.manual_seed(42)

    datasets = [
        # "actor",
        # "wisconsin",
        # "squirrel",
        # "chameleon",
        # "cornell",
        "cora_full",
        # "citeseer",
        # "pubmed",
        # "cora",
    ]

    for name in datasets:
        run_experiment(name, model_name='H2GCN')
