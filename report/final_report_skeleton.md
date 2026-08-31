# GNN-Based BERT for Understanding Context from Music

Course: Neural Networks (CSE425 / EEE474 / CSE715)
[Names / IDs / Group]

> **Note:** This is a structural skeleton, not a finished report. Every
> `[FILL:...]` marker needs real numbers from training on real data
> (FMA-medium / MagnaTagATune + MusicCaps/DEAM) — the toy-data smoke test
> in this repo proves the pipeline is correct, but its numbers are
> meaningless and must not be reported. Paste this into the Overleaf
> NeurIPS/IEEE template linked in the project spec for final formatting.

## Abstract
[FILL: 150-200 words summarizing motivation, method (BERT+GNN fusion),
and headline results across Tasks 1-4.]

## 1. Introduction & Motivation
- Why context understanding in music is inherently multi-modal and
  relational (see Section 1 of the project spec for the framing).
- Gap: sequence-only models (CNN/RNN on spectrograms) miss relational
  structure (chord transitions, segment repetition, cross-modal links to
  lyrics/tags).
- Contribution: hybrid BERT (text) + GNN (structure) system for
  multi-label tagging, emotion regression, and cross-modal retrieval.

## 2. Problem Definition
Restate the formalism from the spec (T = (X_audio, X_text, G, y), the
GNN message-passing update, the fusion readout, and the multi-label BCE
objective). [Reuse the LaTeX from Section 2 of the spec, in your own
words for the surrounding prose.]

## 3. Dataset & Preprocessing
- Datasets used: [FILL — e.g. FMA-medium (audio+genre/tags) + MusicCaps
  (captions) + DEAM (valence/arousal)]
- Preprocessing pipeline: resample to 22,050 Hz, log-mel (128 bins) +
  chroma (12 bins), per-track normalization, 5-10s segmentation
  (`src/audio_features.py`).
- Graph construction: segment graphs (temporal adjacency + cosine
  similarity edges) and chord-transition graphs (`src/graph_builder.py`).
- Splits: [FILL — official FMA/MagnaTagATune splits used; describe
  artist-level leakage prevention.]
- Dataset statistics table: [FILL — n_tracks train/val/test, tag
  distribution, avg segments/track.]

## 4. Model Architecture
### 4.1 Task 1 — BERT Tag Classifier
`src/bert_encoder.py` — CLS pooled representation -> linear sigmoid head.

### 4.2 Task 2 — GNN on Structure Graphs
`src/gnn_model.py` — GraphSAGE (or GAT) message passing, mean-pool
readout. Compared against CNN-on-mel baseline (`CNNBaseline`).

### 4.3 Task 3 — GNN-BERT Fusion
`src/fusion_model.py` — cross-attention fusion (graph embedding as
query over BERT token sequence) vs. early-concat ablation. Multi-task
loss: tags (BCE) + valence/arousal (MSE).

### 4.4 Task 4 — Contrastive Retrieval
`src/contrastive.py` — dual encoder, symmetric InfoNCE, projected to a
shared embedding space.

[Insert an architecture diagram here — see `results/plots/` once
generated, or draw one showing X_audio/X_text -> BERT/GNN -> fusion -> heads.]

## 5. Experimental Setup
- Hyperparameters: [FILL from `config.yaml` — batch size, LR, epochs,
  GNN layers/hidden dim, fusion type, temperature.]
- Hardware: [FILL — e.g. "trained on Colab T4 GPU" / "university GPU cluster".]
- Baselines: B1 (random), B2 (CNN on mel-spec), B3 (BERT-only = Task 1),
  B4 (optional: PCA+MLP on hand-crafted features).

## 6. Results
### 6.1 Task 1 — Tag Classification (text-only)
[FILL: Macro-F1 / Micro-F1 curves vs. epoch; final test metrics table;
5 example predictions.]

### 6.2 Task 2 — Structure-Only Classification
[FILL: GNN vs. CNN-on-mel comparison table (Macro-F1, AUC-PR).]

### 6.3 Task 3 — Fusion
[FILL: Ablation table — BERT-only vs. GNN-only vs. early-concat vs.
cross-attention. Valence/arousal MAE and R². t-SNE plot of z (colored
by genre/mood) from `results/plots/tsne_fusion_z.png`. 3 case studies
showing graph structure + caption alignment.]

### 6.4 Task 4 — Contrastive Retrieval (bonus)
[FILL: R@1/R@5/R@10 for audio->caption and caption->audio on MusicCaps
test split; 10 qualitative retrieval examples from
`results/retrieval_examples/task4_qualitative.json`; optional human
evaluation (5 listeners rating retrieved-clip/caption match 1-5).]

### 6.5 Overall Comparison Table
| Model | Macro-F1 | AUC-PR | MAE (emotion) | R@5 (retrieval) |
|---|---|---|---|---|
| B1 Random | [FILL] | [FILL] | – | [FILL] |
| B2 CNN mel-spec | [FILL] | [FILL] | [FILL] | – |
| Task 1: BERT-only | [FILL] | [FILL] | – | – |
| Task 2: GNN-only | [FILL] | [FILL] | [FILL] | – |
| Task 3: GNN-BERT fusion | [FILL] | [FILL] | [FILL] | – |
| Task 4: Contrastive | [FILL] | [FILL] | – | [FILL] |

## 7. Analysis & Discussion
[FILL: Where does fusion help most? Which tags benefit from graph
structure vs. text alone? Failure cases. Graph coherence score
(Section 6 of spec) if computed.]

## 8. Limitations & Future Work
[FILL: dataset size/coverage limits, compute constraints, what a larger
audio backbone (e.g. replacing hand-crafted chroma/mel with a
pretrained audio encoder) might add.]

## 9. Reproducibility
Code: `github.com/[FILL: your repo URL]`
All configs in `config.yaml`; toy-data smoke test in `README.md` verifies
pipeline correctness independent of real-data results.

## References
[FILL: FMA, MagnaTagATune, MusicCaps, DEAM dataset papers; BERT;
GraphSAGE; GAT papers.]
