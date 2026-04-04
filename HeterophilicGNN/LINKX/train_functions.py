import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def train(model, data, A, train_mask, optimizer, batch_size, device='cuda'):
    model.train()

    # Get train node indices
    train_nodes = train_mask.nonzero(as_tuple=True)[0]

    # Sample batch from TRAIN nodes only
    batch_nodes = train_nodes[
        torch.randint(0, train_nodes.size(0), (batch_size,), device=device)
    ]

    # Forward
    out = model(data.x, A, batch_nodes)   # (B, num_classes)

    # Compute loss ONLY on batch
    loss = F.cross_entropy(out, data.y[batch_nodes])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def evaluate(model, data, A, mask):
    model.eval()

    out = model(data.x, A)   # full forward
    pred = out.argmax(dim=1)

    correct = (pred[mask] == data.y[mask]).sum().item()
    return correct / mask.sum().item()

@torch.no_grad()
def evaluate_loss(model, data, A, mask):
    model.eval()

    out = model(data.x, A)
    loss = F.cross_entropy(out[mask], data.y[mask])

    return loss.item()