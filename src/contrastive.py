"""
Task 4: Contrastive dual-encoder GNN-BERT for MusicCaps alignment
(Algorithm 4).

    g_i = Normalize(GNN(G_i))
    t_i = Normalize(BERT_CLS(caption_i))
    S_ij = g_i^T t_j / tau
    L_NCE = -1/N sum_i log( exp(S_ii) / sum_j exp(S_ij) )

Symmetrized here (graph->text AND text->graph), which is the standard
CLIP-style InfoNCE and matches the "Caption->Audio" / "Audio->Caption"
retrieval evaluation the spec asks for.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_encoder import BertTextEncoder
from gnn_model import GNNEncoder


class ContrastiveGNNBert(nn.Module):
    def __init__(self, bert_name="distilbert-base-uncased", gnn_in_dim=140,
                 gnn_hidden=128, gnn_layers=3, proj_dim=128, gnn_type="sage"):
        super().__init__()
        self.text_encoder = BertTextEncoder(bert_name)
        self.gnn_encoder = GNNEncoder(gnn_in_dim, gnn_hidden, gnn_layers, gnn_type)

        self.graph_proj = nn.Linear(gnn_hidden, proj_dim)
        self.text_proj = nn.Linear(self.text_encoder.hidden_size, proj_dim)

    def forward(self, input_ids, attention_mask, x, edge_index, batch_index):
        _, t = self.text_encoder(input_ids, attention_mask)
        num_graphs = input_ids.shape[0]
        _, g = self.gnn_encoder(x, edge_index, batch_index, num_graphs=num_graphs)

        g_emb = F.normalize(self.graph_proj(g), dim=-1)
        t_emb = F.normalize(self.text_proj(t), dim=-1)
        return g_emb, t_emb


def info_nce_loss(g_emb, t_emb, temperature=0.07):
    """Symmetric InfoNCE over a batch of (graph, text) pairs."""
    logits = g_emb @ t_emb.t() / temperature   # (B, B), S_ij = g_i . t_j
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_g2t = F.cross_entropy(logits, labels)          # graph -> text
    loss_t2g = F.cross_entropy(logits.t(), labels)       # text -> graph
    return (loss_g2t + loss_t2g) / 2


@torch.no_grad()
def retrieval_recall_at_k(g_emb, t_emb, k_list=(1, 5, 10)):
    """
    Computes Caption->Audio and Audio->Caption R@K on a held-out batch.
    Assumes row i in g_emb and row i in t_emb are the true pair.
    """
    sim = g_emb @ t_emb.t()   # (N, N)
    n = sim.shape[0]
    results = {}

    # audio -> caption: for each graph row, is the true text in top-k cols?
    ranks_a2c = (-sim).argsort(dim=1)
    for k in k_list:
        hits = sum(i in ranks_a2c[i, :k].tolist() for i in range(n))
        results[f"audio2caption_R@{k}"] = hits / n

    # caption -> audio: transpose
    ranks_c2a = (-sim.t()).argsort(dim=1)
    for k in k_list:
        hits = sum(i in ranks_c2a[i, :k].tolist() for i in range(n))
        results[f"caption2audio_R@{k}"] = hits / n

    return results
