"""
Task 3: GNN-BERT fusion for multi-context understanding (Algorithm 3).

    H_text = BERT(X_text), t = H_text[CLS]
    g = GNN_Readout(G)
    z = CrossAttention(g, H_text)   [or CONCAT(g, t)]
    yhat = sigma(W z + b)
    L = L_tags + alpha * L_valence + beta * L_arousal
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_encoder import BertTextEncoder
from gnn_model import GNNEncoder


class CrossAttentionFusion(nn.Module):
    """
    Single-query cross-attention: the graph embedding g attends over the
    full BERT token sequence H_text, per the spec:
        A = softmax(QK^T / sqrt(d)),  Q = g W_Q,  K = H_text W_K
        z = CONCAT(g, A H_text)
    """

    def __init__(self, gnn_dim, text_dim):
        super().__init__()
        self.d = text_dim
        self.wq = nn.Linear(gnn_dim, text_dim)
        self.wk = nn.Linear(text_dim, text_dim)
        self.wv = nn.Linear(text_dim, text_dim)
        self.out_dim = gnn_dim + text_dim

    def forward(self, g, H_text, attention_mask=None):
        # g: (B, gnn_dim), H_text: (B, L, text_dim)
        q = self.wq(g).unsqueeze(1)                       # (B, 1, text_dim)
        k = self.wk(H_text)                                # (B, L, text_dim)
        v = self.wv(H_text)                                # (B, L, text_dim)
        scores = torch.bmm(q, k.transpose(1, 2)) / (self.d ** 0.5)  # (B, 1, L)
        if attention_mask is not None:
            mask = (1.0 - attention_mask.unsqueeze(1).float()) * -1e9
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)                   # (B, 1, L)
        attended = torch.bmm(attn, v).squeeze(1)            # (B, text_dim)
        z = torch.cat([g, attended], dim=1)                 # (B, gnn_dim + text_dim)
        return z


class GNNBertFusionModel(nn.Module):
    def __init__(self, bert_name="distilbert-base-uncased", gnn_in_dim=140,
                 gnn_hidden=128, gnn_layers=3, num_tags=20, fusion_type="cross_attn",
                 gnn_type="sage"):
        super().__init__()
        self.text_encoder = BertTextEncoder(bert_name)
        self.gnn_encoder = GNNEncoder(gnn_in_dim, gnn_hidden, gnn_layers, gnn_type)
        self.fusion_type = fusion_type

        text_dim = self.text_encoder.hidden_size
        if fusion_type == "cross_attn":
            self.fusion = CrossAttentionFusion(gnn_hidden, text_dim)
            z_dim = self.fusion.out_dim
        else:  # early concat baseline (used in ablations)
            self.fusion = None
            z_dim = gnn_hidden + text_dim

        self.tag_head = nn.Linear(z_dim, num_tags)
        self.va_head = nn.Linear(z_dim, 2)   # [valence, arousal]

    def forward(self, input_ids, attention_mask, x, edge_index, batch_index):
        H_text, t = self.text_encoder(input_ids, attention_mask)
        num_graphs = input_ids.shape[0]
        _, g = self.gnn_encoder(x, edge_index, batch_index, num_graphs=num_graphs)

        if self.fusion_type == "cross_attn":
            z = self.fusion(g, H_text, attention_mask)
        else:
            z = torch.cat([g, t], dim=1)

        tag_logits = self.tag_head(z)
        va_pred = self.va_head(z)   # raw regression output (not scaled to [1,9] here;
                                     # scale/clip in evaluate.py or add a sigmoid*8+1 head)
        return tag_logits, va_pred
