import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
# =========================================================
# 1. GCN (Kipf & Welling) - Baseline
# =========================================================
 
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def normalize(self, A):
        deg = A.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        return D_inv_sqrt @ A @ D_inv_sqrt
    
    def forward(self, X, edge_index):
        n = X.size(0)
        
        A = torch.zeros((n, n), device=X.device)
        A[edge_index[0], edge_index[1]] = 1
        
        # Undirected
        A = A + A.T
        A = (A > 0).float()
        
        # Add self-loops (CRITICAL for GCN)
        A = A + torch.eye(n, device=X.device)
        
        A_hat = self.normalize(A)
        
        # Layer 1
        H = A_hat @ X
        H = self.lin1(H)
        H = F.relu(H)
        H = self.dropout(H)
        
        # Layer 2
        H = A_hat @ H
        H = self.lin2(H)
        
        return H
 
 
# =========================================================
# 2. MultiHopGNN 
# =========================================================
 
class MultiHopGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_hops=2, dropout=0.5):
        super().__init__()
        
        self.num_hops = num_hops
        self.dropout = nn.Dropout(dropout)
        
        # MLP after concatenation
        self.mlp = nn.Sequential(
            nn.Linear((num_hops + 1) * in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def normalize(self, A):
        deg = A.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        return D_inv_sqrt @ A @ D_inv_sqrt
    
    def forward(self, X, edge_index):
        n = X.size(0)
        
        # Build adjacency
        A = torch.zeros((n, n), device=X.device)
        A[edge_index[0], edge_index[1]] = 1
        
        A_norm = self.normalize(A)
        
        # Compute multi-hop features
        features = [X]
        X_k = X
        
        for _ in range(self.num_hops):
            X_k = A_norm @ X_k
            features.append(X_k)
        
        # Concatenate all hops
        H = torch.cat(features, dim=1)
        
        H = self.dropout(H)
        
        out = self.mlp(H)
        
        return out
 
 
 
# =========================================================
# 3. H2GCN 
# =========================================================
 
class H2GCN_Improved(nn.Module):
    """
    Improved H2GCN implementation based on the paper:
    'Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs'
    
    Key fixes:
    1. NO self-loops in adjacency (critical for heterophily)
    2. Proper graph symmetrization (undirected)
    3. Separate embeddings for each hop combination
    4. Correct dimension tracking
    5. Optional ego/neighbor separation (D1)
    6. Row normalization instead of symmetric
    """
    
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=2, 
                 dropout=0.5, use_ego_neighbor_separation=True, debug = False):
        super().__init__()
        self.debug = debug
        self.K = num_layers
        self.use_ego = use_ego_neighbor_separation
        
        # S1: Feature embedding
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        
        # S2: Create separate embeddings for each propagation layer
        self.layer_embeddings = nn.ModuleList()
        
        for k in range(num_layers):
            if use_ego_neighbor_separation:
                # For each hop, we separate ego (self) and neighbor features
                layer_in_dim = hidden_dim * 4  # 2 hops × 2 (ego/neighbor)
            else:
                layer_in_dim = hidden_dim * 2  # Just 2 hops
            
            self.layer_embeddings.append(
                nn.Linear(layer_in_dim, hidden_dim)
            )
        
        # S3: Final classifier
        final_dim = hidden_dim * (num_layers + 1)
        self.classifier = nn.Linear(final_dim, num_classes)
    
    def normalize(self, A):
        """Row-normalization: D^{-1}A (as per paper)"""
        deg = A.sum(dim=1, keepdim=True)
        deg = torch.clamp(deg, min=1.0)
        return A / deg
    
    def compute_A2(self, A):
        """Compute clean 2-hop adjacency matrix"""
        A2 = A @ A
        
        # Remove self-loops and 1-hop neighbors
        I = torch.eye(A.shape[0], device=A.device)
        A2 = A2 - A - I
        
        # Binarize
        A2 = (A2 > 0).float()
        
        return A2
    
    def build_adjacency(self, edge_index, n, device):
        """Build adjacency matrix - NO self-loops (critical for heterophily)"""
        assert edge_index.min() >= 0, "Negative index found"
        assert edge_index.max() < n, f"Index out of bounds: max={edge_index.max()}, n={n}"
        A = torch.zeros((n, n), device=device)
        A[edge_index[0], edge_index[1]] = 1
        
        # Make undirected (symmetrize)
        A = A + A.T
        A = (A > 0).float()
        
        # DO NOT add self-loops
        return A
    
    def separate_ego_neighbor(self, A_norm, X):
        """Design D1: Separate ego and neighbor embeddings"""
        X_ego = X
        X_neighbor = A_norm @ X
        return X_ego, X_neighbor
    
    def forward(self, X, edge_index):
        n = X.size(0)
        
        # Build adjacency matrices
        if self.debug:
            print("n:", n)
            print("X shape:", X.shape)
            print("edge_index shape:", edge_index.shape)
            print("edge_index min:", edge_index.min().item())
            print("edge_index max:", edge_index.max().item())

        A = self.build_adjacency(edge_index, n, X.device)


        A1_norm = self.normalize(A)
        
        A2 = self.compute_A2(A)
        A2_norm = self.normalize(A2)
        
        # S1: Initial embedding
        r0 = F.relu(self.embed(X))
        r0 = self.dropout(r0)
        
        representations = [r0]
        r_prev = r0
        
        # S2: Multi-layer propagation
        for k in range(self.K):
            # Process 1-hop neighbors
            if self.use_ego:
                r1_ego, r1_neighbor = self.separate_ego_neighbor(A1_norm, r_prev)
            else:
                r1_neighbor = A1_norm @ r_prev
                r1_ego = r_prev
            
            # Process 2-hop neighbors
            if self.use_ego:
                r2_ego, r2_neighbor = self.separate_ego_neighbor(A2_norm, r_prev)
            else:
                r2_neighbor = A2_norm @ r_prev
                r2_ego = r_prev
            
            # Concatenate all features
            if self.use_ego:
                r_k_input = torch.cat([r1_ego, r1_neighbor, r2_ego, r2_neighbor], dim=1)
            else:
                r_k_input = torch.cat([r1_neighbor, r2_neighbor], dim=1)
            
            # Transform with layer-specific embedding
            r_k = F.relu(self.layer_embeddings[k](r_k_input))
            r_k = self.dropout(r_k)
            
            representations.append(r_k)
            r_prev = r_k
        
        # S3: Combine all representations
        r_final = torch.cat(representations, dim=1)
        r_final = self.dropout(r_final)
        
        out = self.classifier(r_final)
        
        return out
 
# =========================================================
# 4. LINKX
# =========================================================
class LINKX(nn.Module):
    def __init__(self, feat_dim, hidden_dim, adj_dim, num_classes):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.adj_dim = adj_dim
        self.num_classes = num_classes
        self.MLPX = nn.Linear(feat_dim, hidden_dim)
        self.MLPA = nn.Linear(adj_dim, hidden_dim)
        self.W = nn.Linear(2*hidden_dim, hidden_dim)
        self.out_transform = nn.Linear(hidden_dim, num_classes)
        
    def build_adjacency(self, edge_index, n, normalization = False, device = 'cuda'):
        """Build adjacency matrix - NO self-loops (critical for heterophily)"""
        assert edge_index.min() >= 0, "Negative index found"
        assert edge_index.max() < n, f"Index out of bounds: max={edge_index.max()}, n={n}"

        A = torch.zeros((n, n), device=device)
        A[edge_index[0], edge_index[1]] = 1
        
        # Make undirected (symmetrize)
        A = A + A.T
        A = (A > 0).float()

        if normalization == True:
            # Optional normalization
            deg = A.sum(dim=1, keepdim=True)
            A = A / (deg + 1e-8)
        # DO NOT add self-loops
        return A
    
    def forward(self, X, edge_index):

        A =  self.build_adjacency(edge_index, self.adj_dim,normalization=True, device= X.device)
        
        HX = self.MLPX(X)
        HA = self.MLPA(A)
        H = torch.cat((HA,HX), dim=1)
        out_emb = self.W(H) + HA + HX

        return torch.relu(self.out_transform(out_emb))
    
# =========================================================
# 5. LINKXC
# =========================================================

def build_sparse_adjacency(edge_index, num_nodes, normalization = True, device = 'cuda'):
    """
    edge_index: (2, E)
    """

    row, col = edge_index

    # Make undirected
    row = torch.cat([row, col])
    col = torch.cat([col, row])

    values = torch.ones(row.size(0), device=device)

    A = torch.sparse_coo_tensor(
        torch.stack([row, col]),
        values,
        (num_nodes, num_nodes)
    )

    # Row normalization (optional but recommended)
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1)
    deg_inv = 1.0 / deg

    # Normalize values
    row_idx = row
    values = deg_inv[row_idx]

    A = torch.sparse_coo_tensor(
        torch.stack([row, col]),
        values,
        (num_nodes, num_nodes)
    )

    return A.coalesce()

class LINKXC(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_nodes, num_classes):
        super().__init__()

        self.MLPX = nn.Linear(feat_dim, hidden_dim)
        self.MLPA = nn.Linear(num_nodes, hidden_dim)

        self.W = nn.Linear(2 * hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, X, A_sparse, batch_nodes=None):
        """
        X: (N, feat_dim)
        A_sparse: sparse adjacency (N x N)
        batch_nodes: indices of nodes to train on
        """

        if batch_nodes is None:
            # full batch
            A_batch = A_sparse.to_dense()
            X_batch = X
        else:
            # ONLY sample rows, keep full columns
            A_batch = A_sparse[batch_nodes].to_dense()   # (B, N)
            X_batch = X[batch_nodes]      # (B, feat_dim)

        # MLPs
        HX = torch.relu(self.MLPX(X_batch))
        HA = torch.relu(self.MLPA(A_batch))

        # Combine
        H = torch.cat([HA, HX], dim=1)
        H = self.W(H) + HA + HX

        return self.out(H)