"""
OFFLINE SMOKE-TEST STUB -- not for real experiments / submission.

This sandbox has no network access to huggingface.co, so real pretrained
`distilbert-base-uncased` weights/tokenizer can't be downloaded here.
This module builds a tiny, randomly-initialized BERT (via `transformers`'
BertConfig/BertModel, which only needs the *architecture* code, already
installed locally -- no download) plus a minimal whitespace tokenizer with
a vocab built from whatever captions are on disk.

It exists ONLY to verify that the rest of the pipeline (datasets, graph
batching, fusion, contrastive loss, training loop, evaluation) is wired
correctly end-to-end. The numbers it produces are meaningless -- run
train.py/evaluate.py normally (with `model.bert_name: distilbert-base-uncased`
or `bert-base-uncased`) on a machine with internet access (local / Colab)
for real results.

Usage: import build_tiny_bert_and_tokenizer() and monkeypatch
AutoModel.from_pretrained / AutoTokenizer.from_pretrained, OR just run:

    python src/offline_stub.py --smoke_test_task 1
    python src/offline_stub.py --smoke_test_task 2
    python src/offline_stub.py --smoke_test_task 3
    python src/offline_stub.py --smoke_test_task 4
"""
import argparse
import json
import re

import torch
from transformers import BertConfig, BertModel


class SimpleWhitespaceTokenizer:
    """Minimal stand-in for a HF tokenizer's __call__ interface."""

    def __init__(self, vocab):
        self.vocab = vocab  # word -> id, includes [PAD]=0, [CLS]=1, [UNK]=2

    def __call__(self, text, truncation=True, padding="max_length",
                 max_length=128, return_tensors="pt"):
        words = re.findall(r"[a-z0-9']+", text.lower())
        ids = [self.vocab.get("[CLS]", 1)]
        for w in words[:max_length - 1]:
            ids.append(self.vocab.get(w, self.vocab.get("[UNK]", 2)))
        attn = [1] * len(ids)
        while len(ids) < max_length:
            ids.append(self.vocab.get("[PAD]", 0))
            attn.append(0)
        ids, attn = ids[:max_length], attn[:max_length]
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.tensor([attn]),
        }

    @classmethod
    def build_from_captions(cls, captions):
        vocab = {"[PAD]": 0, "[CLS]": 1, "[UNK]": 2}
        for c in captions:
            for w in re.findall(r"[a-z0-9']+", c.lower()):
                if w not in vocab:
                    vocab[w] = len(vocab)
        return cls(vocab)


def build_tiny_bert_and_tokenizer(vocab_size=2000, hidden_size=64, num_layers=2):
    config = BertConfig(
        vocab_size=vocab_size, hidden_size=hidden_size,
        num_hidden_layers=num_layers, num_attention_heads=4,
        intermediate_size=hidden_size * 2, max_position_embeddings=256,
    )
    model = BertModel(config)  # random init, no download needed
    return model, config


def patch_bert_encoder_for_offline_smoke_test():
    """
    Monkeypatches bert_encoder.BertTextEncoder to use the tiny random BERT
    + whitespace tokenizer instead of AutoModel/AutoTokenizer.from_pretrained.
    Call this BEFORE importing/instantiating any model that uses BertTextEncoder.
    """
    import bert_encoder
    from datasets import load_manifest

    captions = [m["caption"] for m in load_manifest("train")] + \
               [m["caption"] for m in load_manifest("val")] + \
               [m["caption"] for m in load_manifest("test")]
    tokenizer = SimpleWhitespaceTokenizer.build_from_captions(captions)
    tiny_model, config = build_tiny_bert_and_tokenizer(vocab_size=len(tokenizer.vocab))

    orig_init = bert_encoder.BertTextEncoder.__init__

    def patched_init(self, model_name="distilbert-base-uncased", freeze=False):
        import torch.nn as nn
        nn.Module.__init__(self)
        self.bert = BertModel(config)  # fresh random init per instance
        self.hidden_size = config.hidden_size
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    bert_encoder.BertTextEncoder.__init__ = patched_init

    import transformers
    _orig_tok = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = staticmethod(lambda *a, **k: tokenizer)
    return tokenizer


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke_test_task", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()

    patch_bert_encoder_for_offline_smoke_test()

    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["bert_hidden"] = 64

    import train as train_mod
    device = train_mod.get_device()
    print(f"[offline smoke test] task={args.smoke_test_task}, device={device}")

    if args.smoke_test_task == 1:
        train_mod.train_task1(cfg, args.epochs, device)
    elif args.smoke_test_task == 2:
        train_mod.train_task2(cfg, args.epochs, device)
    elif args.smoke_test_task == 3:
        train_mod.train_task3(cfg, args.epochs, device)
    elif args.smoke_test_task == 4:
        train_mod.train_task4(cfg, args.epochs, device)
    print("[offline smoke test] OK -- pipeline ran end-to-end without errors.")
