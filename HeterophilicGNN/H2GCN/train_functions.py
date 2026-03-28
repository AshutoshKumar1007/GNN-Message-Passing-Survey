import torch
import torch.nn.functional as F

def train(model, data, train_mask, optimizer):
  model.train()
  optimizer.zero_grad()
  out = model(data.x, data.edge_index)
  loss = F.cross_entropy(out[train_mask], data.y[train_mask])
  loss.backward()
  optimizer.step()
  return loss.item()

def evaluate(model, data, val_mask):
  model.eval()
  out = model(data.x, data.edge_index)
  pred = out.argmax(dim = 1)
  correct = (pred[val_mask] == data.y[val_mask]).sum().item()
  return correct/ val_mask.sum().item()

def evaluate_loss(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[mask], data.y[mask])
    return loss.item()
