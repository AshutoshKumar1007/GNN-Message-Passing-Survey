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
# 3. H2GCN - Original
# =========================================================
 
class H2GCN_Original(nn.Module):
    """Your original H2GCN implementation - kept for comparison"""
    
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers, dropout=0.5):
        super().__init__()
        
        self.K = num_layers
        
        # Feature embedding (S1)
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        
        # Final classifier
        final_dim = (2 ** (num_layers + 1) - 1) * hidden_dim
        self.classifier = nn.Linear(final_dim, num_classes)
    
    def normalize(self, A):
        """Symmetric normalization"""
        deg = A.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        return D_inv_sqrt @ A @ D_inv_sqrt
    
    def compute_A2(self, A):
        """Compute clean 2-hop adjacency"""
        A2 = A @ A
        
        # Remove self and 1-hop
        I = torch.eye(A.shape[0], device=A.device)
        A2 = A2 - A - I
        
        # Binarize
        A2 = (A2 > 0).float()
        
        # Normalize
        return self.normalize(A2)
    
    def forward(self, X, edge_index):
        n = X.size(0)
        
        # Build Adjacency for this Batch
        A = torch.zeros((n, n), device=X.device)
        A[edge_index[0], edge_index[1]] = 1
        
        # Compute A1 & A2 locally
        A1 = self.normalize(A)
        A2 = self.compute_A2(A)
        
        # S1: initial embedding
        r0 = F.relu(self.embed(X))
        r0 = self.dropout(r0)
        representations = [r0]
        r_prev = r0
        
        # S2: propagation layers
        for k in range(self.K):
            # 1-hop aggregation
            h1 = A1 @ r_prev
            
            # 2-hop aggregation
            h2 = A2 @ r_prev
            
            # Concatenate (D2)
            r_k = torch.cat([h1, h2], dim=1)
            r_k = self.dropout(r_k)
            
            representations.append(r_k)
            r_prev = r_k
        
        # S3: final combination (D3)
        r_final = torch.cat(representations, dim=1)
        r_final = self.dropout(r_final)
        
        out = self.classifier(r_final)
        
        return out
 
 
# =========================================================
# 4. H2GCN 
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
# 5. H2GCN - Simple Version (without ego-neighbor separation)
# =========================================================
 
class H2GCN_Simple(nn.Module):
    """
    Simplified H2GCN without ego-neighbor separation.
    Often performs similarly with fewer parameters.
    """
    
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5):
        super().__init__()
        
        self.K = num_layers
        
        # Feature embedding
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        
        # Layer-specific transformations
        self.layer_weights = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(num_layers)
        ])
        
        # Final classifier
        final_dim = hidden_dim * (num_layers + 1)
        self.classifier = nn.Linear(final_dim, num_classes)
    
    def normalize(self, A):
        """Row-normalization"""
        deg = A.sum(dim=1, keepdim=True)
        deg = torch.clamp(deg, min=1.0)
        return A / deg
    
    def compute_A2(self, A):
        """Clean 2-hop adjacency"""
        A2 = A @ A
        I = torch.eye(A.shape[0], device=A.device)
        A2 = A2 - A - I
        A2 = (A2 > 0).float()
        return A2
    
    def build_adjacency(self, edge_index, n, device):
        """Build undirected adjacency WITHOUT self-loops"""
        A = torch.zeros((n, n), device=device)
        A[edge_index[0], edge_index[1]] = 1
        A = A + A.T
        A = (A > 0).float()
        return A
    
    def forward(self, X, edge_index):
        n = X.size(0)
        
        # Build adjacency
        A = self.build_adjacency(edge_index, n, X.device)
        A1 = self.normalize(A)
        A2 = self.normalize(self.compute_A2(A))
        
        # Initial embedding
        r0 = F.relu(self.embed(X))
        r0 = self.dropout(r0)
        
        representations = [r0]
        r_prev = r0
        
        # Multi-layer propagation
        for k in range(self.K):
            # 1-hop and 2-hop aggregation
            h1 = A1 @ r_prev
            h2 = A2 @ r_prev
            
            # Concatenate and transform
            h = torch.cat([h1, h2], dim=1)
            r_k = F.relu(self.layer_weights[k](h))
            r_k = self.dropout(r_k)
            
            representations.append(r_k)
            r_prev = r_k
        
        # Final combination
        r_final = torch.cat(representations, dim=1)
        r_final = self.dropout(r_final)
        
        out = self.classifier(r_final)
        
        return out
 
 
# Default H2GCN points to the improved version
H2GCN = H2GCN_Simple  # Use simple by default (good balance of performance/complexity)