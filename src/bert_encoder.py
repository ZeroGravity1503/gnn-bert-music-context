"""
Shared BERT text encoder. Used standalone in Task 1 and as the text branch
in Task 3 (fusion) and Task 4 (contrastive).

Htext = BERT(Xtext) in R^{L x d};  t = Htext[CLS]  (pooled representation).
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class BertTextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", freeze=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        H = out.last_hidden_state              # (B, L, d)  -- full sequence, for cross-attention
        # CLS-equivalent pooled vector. DistilBERT has no pooler_output,
        # so we use the first token's hidden state (works for BERT too).
        t = H[:, 0, :]                          # (B, d)
        return H, t


class BertTagClassifier(nn.Module):
    """Task 1: BERT CLS -> multi-label sigmoid head."""

    def __init__(self, model_name="distilbert-base-uncased", num_tags=20,
                 freeze_bert=False):
        super().__init__()
        self.encoder = BertTextEncoder(model_name, freeze=freeze_bert)
        self.head = nn.Linear(self.encoder.hidden_size, num_tags)

    def forward(self, input_ids, attention_mask):
        _, t = self.encoder(input_ids, attention_mask)
        logits = self.head(t)                   # (B, num_tags)
        return logits
