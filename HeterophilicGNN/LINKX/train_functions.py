import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def pf_func():
    print("Tested Loading ")
    
def train_linkx(model, data, train_loader, optimizer, scheduler, device, debug = False):
    model.train()
    total_loss = 0
     
    if debug: print("Inside train_linkx, Device:",device)
    counter = 1
    for batch_nodes in train_loader:
        if debug:
            while counter > 0:
                print("Batch Nodes Dtype:",batch_nodes.dtype)
                print("Batch Nodes shaep:",batch_nodes.shape)
                counter -= 1
        batch_nodes = batch_nodes.to(device)

        A_batch = build_A_batch(
            data.edge_index,
            batch_nodes,
            data.num_nodes,
            device
        )
        
        if debug:
            print("A_batch:", A_batch.shape)
            print("X_batch:", data.x[batch_nodes].shape)
        out = model(data.x,data.edge_index, A_batch, batch_nodes, debug)

        loss = F.cross_entropy(out, data.y[batch_nodes])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

@torch.no_grad()
def evaluate_linkx(model, data, mask, batch_size=1024):
    model.eval()

    nodes = torch.where(mask)[0]
    loader = DataLoader(nodes, batch_size=batch_size)

    correct = 0
    total = 0

    for batch_nodes in loader:
        batch_nodes = batch_nodes.to(data.x.device)

        A_batch = build_A_batch(
            data.edge_index,
            batch_nodes,
            data.num_nodes,
            data.x.device
        )

        out = model(data.x, data.edge_index, A_batch, batch_nodes)
        pred = out.argmax(dim=1)

        correct += (pred == data.y[batch_nodes]).sum().item()
        total += batch_nodes.size(0)

    return correct / total

@torch.no_grad()
def evaluate_loss_linkx(model, data, mask, batch_size=1024):
    model.eval()

    nodes = torch.where(mask)[0]
    loader = DataLoader(nodes, batch_size=batch_size)

    total_loss = 0
    total = 0

    for batch_nodes in loader:
        batch_nodes = batch_nodes.to(data.x.device)

        A_batch = build_A_batch(
            data.edge_index,
            batch_nodes,
            data.num_nodes,
            data.x.device
        )

        out = model(data.x, data.edge_index, A_batch, batch_nodes)

        loss = F.cross_entropy(out, data.y[batch_nodes], reduction='sum')

        total_loss += loss.item()
        total += batch_nodes.size(0)

    return total_loss / total


def train_gnn(model, data, train_mask, optimizer, debug = False):
  model.train()
  optimizer.zero_grad()
  if debug: print("Inside forward Function: ", data.x.shape)
  out = model(data.x, data.edge_index)
  loss = F.cross_entropy(out[train_mask], data.y[train_mask])
  loss.backward()
  optimizer.step()
  return loss.item()

def evaluate_gnn(model, data, val_mask):
  model.eval()
  out = model(data.x, data.edge_index)
  pred = out.argmax(dim = 1)
  correct = (pred[val_mask] == data.y[val_mask]).sum().item()
  return correct/ val_mask.sum().item()

def evaluate_loss_gnn(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[mask], data.y[mask])
    return loss.item()


def build_A_batch(edge_index, batch_nodes, num_nodes, device):
    row, col = edge_index

    # mask edges where source is in batch
    mask = torch.isin(row, batch_nodes)

    row = row[mask]
    col = col[mask]

    # map global → batch index
    batch_map = -torch.ones(num_nodes, dtype=torch.long, device=device)
    batch_map[batch_nodes] = torch.arange(batch_nodes.size(0), device=device)

    batch_row = batch_map[row]

    A_batch = torch.zeros((batch_nodes.size(0), num_nodes), device=device)
    A_batch[batch_row, col] = 1

    return A_batch