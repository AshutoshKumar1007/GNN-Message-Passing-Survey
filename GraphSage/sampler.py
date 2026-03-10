import torch
import torch.nn as nn

'''
adj_info -> shape = [num_nodes, max_degree]
i.e.,
node 0 → [3,5,7,1]
node 1 → [0,2,4,3]
node 2 → [1,6,5,4]
'''

import torch
import torch.nn as nn

class UniformNeighborSampler(nn.Module):

  def __init__(self, adj_info):
    super().__init__()
    self.adj_info = adj_info

  def forward(self, ids, num_samples):

    # lookup adjacency rows
    adj_lists = self.adj_info[ids]

    # shuffle neighbors
    perm = torch.randperm(adj_lists.shape[1])
    adj_lists = adj_lists[:, perm]

    # take first k
    adj_lists = adj_lists[:, :num_samples]

    return adj_lists
  
def main():
  adj_info = torch.tensor([
    [1,2,3,4],
    [0,2,4,5],
    [0,1,3,5]
  ])
  sampler = UniformNeighborSampler(adj_info)

  nodes = torch.tensor([0,1])
  neighbors = sampler(nodes, num_samples=3)

  print(neighbors)

if __name__ == '__main__':
  main()