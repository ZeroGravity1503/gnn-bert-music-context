"""
GNN encoder for Task 2 (and reused as the graph branch in Task 3/4).

Implemented in pure PyTorch (manual scatter-mean via index_add) rather
than requiring torch_geometric, since PyG's wheel matching against a
specific torch/CUDA build is a common source of install failures in
constrained environments. Swap in `torch_geometric.nn.SAGEConv` /
`GATConv` directly if you have PyG installed and prefer it -- the
message-passing math is identical to what's below and to the project
spec:

    h_i^(l+1) = sigma( W^(l) . CONCAT( h_i^(l), MEAN_{j in N(i)} h_j^(l) ) )

Batching: instead of torch_geometric's Batch object, we use the
`batch_index` vector produced by datasets.collate_graphs (maps each node
to which graph in the mini-batch it belongs to) for the final readout.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_mean(src, index, dim_size):
    """Mean-aggregate src[i] into buckets given by index[i], for i in range(len(src))."""
    out = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    count.index_add_(0, index, torch.ones(src.shape[0], 1, device=src.device, dtype=src.dtype))
    count = count.clamp(min=1.0)
    return out / count


class SAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)

    def forward(self, h, edge_index):
        # edge_index: (2, E) with src -> dst. Aggregate messages FROM
        # neighbors (src) INTO each node (dst).
        src, dst = edge_index[0], edge_index[1]
        neighbor_msgs = h[src]                              # (E, in_dim)
        agg = scatter_mean(neighbor_msgs, dst, dim_size=h.shape[0])  # (N, in_dim)
        combined = torch.cat([h, agg], dim=1)                # (N, 2*in_dim)
        return F.relu(self.lin(combined))


class GATLayer(nn.Module):
    """Single-head GAT layer (simplified additive attention), pure PyTorch."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.attn = nn.Linear(2 * out_dim, 1)

    def forward(self, h, edge_index):
        src, dst = edge_index[0], edge_index[1]
        h_proj = self.lin(h)                                 # (N, out_dim)
        e = self.attn(torch.cat([h_proj[src], h_proj[dst]], dim=1)).squeeze(-1)
        e = F.leaky_relu(e, 0.2)
        # softmax over incoming edges per dst node
        e_exp = torch.exp(e - e.max())
        denom = torch.zeros(h.shape[0], device=h.device).index_add_(0, dst, e_exp).clamp(min=1e-8)
        alpha = e_exp / denom[dst]
        weighted = h_proj[src] * alpha.unsqueeze(-1)
        out = torch.zeros_like(h_proj).index_add_(0, dst, weighted)
        return F.elu(out)


class GNNEncoder(nn.Module):
    """Stacks L message-passing layers, returns per-graph readout via mean pool."""

    def __init__(self, in_dim, hidden_dim=128, num_layers=3, gnn_type="sage"):
        super().__init__()
        layer_cls = SAGELayer if gnn_type == "sage" else GATLayer
        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList([
            layer_cls(dims[i], dims[i + 1]) for i in range(num_layers)
        ])
        self.out_dim = hidden_dim

    def forward(self, x, edge_index, batch_index, num_graphs=None):
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
        if num_graphs is None:
            num_graphs = int(batch_index.max().item()) + 1
        g = scatter_mean(h, batch_index, dim_size=num_graphs)  # graph-level readout
        return h, g   # h = node embeddings (for cross-attention), g = graph embedding


class CNNBaseline(nn.Module):
    """
    B2 baseline (Section 8): CNN directly on the mel-spectrogram, with no
    graph structure and no text. Simple 1D-conv-over-time stack + global
    pooling, matching the "CNN mel-spec" row in Table 3.
    """

    def __init__(self, n_mels=128, num_tags=20):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, num_tags)

    def forward(self, mel):
        # mel: (B, n_mels, T)
        h = self.conv(mel).squeeze(-1)   # (B, 64)
        return self.head(h)


class GNNTagClassifier(nn.Module):
    """Task 2: segment graph -> multi-label tags, trained standalone."""

    def __init__(self, in_dim, hidden_dim=128, num_layers=3, num_tags=20, gnn_type="sage"):
        super().__init__()
        self.encoder = GNNEncoder(in_dim, hidden_dim, num_layers, gnn_type)
        self.head = nn.Linear(hidden_dim, num_tags)

    def forward(self, x, edge_index, batch_index):
        _, g = self.encoder(x, edge_index, batch_index)
        return self.head(g)
