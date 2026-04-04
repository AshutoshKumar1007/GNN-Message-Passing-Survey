import torch
import numpy as np
import os
import copy

# ---- Models ----
from models import LINKXC as LINKX
from models import H2GCN_Improved as H2GCN
from models import build_sparse_adjacency as build_adjacency

# ---- Train functions ----
from train_functions import train_linkx, evaluate_linkx, evaluate_loss_linkx
from train_functions import train_gnn, evaluate_gnn, evaluate_loss_gnn

# ---- Dataset ----
from dataset import load_dataset   


def run_experiment(dataset_name, model_name='LINKX', save_dir="results"):
  print('Cuda available:', torch.cuda.is_available())
  print(f"\n===== DATASET: {dataset_name} =====")

  data_, num_classes = load_dataset(dataset_name)

  # Normalize features
  data_.x = data_.x / data_.x.sum(dim=1, keepdim=True).clamp(min=1)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  data_ = data_.to(device)

  num_nodes = data_.num_nodes

  # Build adjacency ONCE 
  if model_name == 'LINKX':
      A = build_adjacency(data_.edge_index, num_nodes, device)

  num_splits = data_.train_mask.shape[1] if data_.train_mask.dim() > 1 else 1
  os.makedirs(save_dir, exist_ok=True)

  all_acc = []

  for i in range(num_splits):
      print(f"\n--- Split {i} ---")

      # ---- Model init ----
      if model_name == 'LINKX':
          model = LINKX(
              feat_dim=data_.num_features,
              hidden_dim=64,
              num_nodes=num_nodes,
              num_classes=num_classes
          ).to(device)

      elif model_name == 'H2GCN':
          model = H2GCN(
              in_dim=data_.num_features,
              hidden_dim=64,
              num_classes=num_classes,
              num_layers=2,
              dropout=0.5
          ).to(device)

      else:
          raise ValueError("Unknown model")

      optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)

      train_mask = data_.train_mask[:, i] if num_splits > 1 else data_.train_mask
      val_mask   = data_.val_mask[:, i]   if num_splits > 1 else data_.val_mask
      test_mask  = data_.test_mask[:, i]  if num_splits > 1 else data_.test_mask

      train_losses, val_losses, val_accs = [], [], []

      best_val_loss = float('inf')
      patience = 50
      patience_counter = 0
      best_state = None

      batch_size = 1024

      # ---- Training loop ----
      for epoch in range(300):
        if model_name == 'LINKX':
            train_loss = train_linkx(model, data_, A, train_mask, optimizer, batch_size, device)
            val_loss = evaluate_loss_linkx(model, data_, A, val_mask)
            val_acc = evaluate_linkx(model, data_, A, val_mask)

        else:
            train_loss = train_gnn(model, data_, train_mask, optimizer)
            val_loss = evaluate_loss_gnn(model, data_, val_mask)
            val_acc = evaluate_gnn(model, data_, val_mask)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

      # Load best
      if best_state is not None:
          model.load_state_dict(best_state)

      # ---- Test ----
      if model_name == 'LINKX':
          test_acc = evaluate_linkx(model, data_, A, test_mask)
      else:
          test_acc = evaluate_gnn(model, data_, test_mask)

      print(f"Test Accuracy: {test_acc:.4f}")
      all_acc.append(test_acc)

      # Save
      torch.save({
          "dataset": dataset_name,
          "split": i,
          "train_loss": train_losses,
          "val_loss": val_losses,
          "val_acc": val_accs,
          "test_acc": test_acc
      }, f"{save_dir}/{dataset_name}_{model_name}_split_{i}.pt")

  print(f"\n{dataset_name.upper()} → Avg Accuracy: {np.mean(all_acc):.4f}")


# ---- Entry ----
if __name__ == '__main__':
    torch.manual_seed(42)

    datasets = [
        "actor",
        # "texas",
        # "wisconsin",
        # "chameleon",
    ]

    for name in datasets:
        run_experiment(name, model_name='LINKX')