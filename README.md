# GNN-BERT Music Context Understanding

Course: Neural Networks (CSE425 / EEE474 / CSE715)
Authors: Fardin M Rahin Hossain (23101256), Afia Masuda

A hybrid BERT + Graph Neural Network system for understanding musical
context: multi-label tag classification from text (Task 1), from audio
structure graphs (Task 2), GNN-BERT fusion (Task 3, with a full
BERT-only/GNN-only/early-concat/cross-attention ablation and a DEAM
valence/arousal regression extension), and contrastive audio-caption
retrieval (Task 4). Full results and analysis are in `report/final_report.pdf`.

## Status: complete, trained and evaluated on real data

All four tasks are trained and evaluated on real datasets:
**MagnaTagATune** (4,500 real clips, Tasks 1-3), **DEAM** (1,802 real
clips, Task 3's emotion-regression extension), and **MusicCaps**
(1,854 real clips, Task 4). See `report/final_report.pdf` for full
results, or `results/metrics.json` for the raw numbers.

## Repository structure

```text
gnn-bert-music-context/

├── README.md
├── requirements.txt
├── config.yaml, config_mtat.yaml, config_deam.yaml,
│   config_musiccaps.yaml, config_mtat_concat.yaml

├── data/
│   ├── raw/ # raw dataset downloads (not committed, see data/raw/README.md)
│   ├── processed/
│   │   └── graph_samples/ # 25 real MTAT graph samples
│   ├── packed/ # compressed graph archives (all 3 datasets)
│   └── splits/ # real train/val/test JSON (+ deam/, musiccaps/ subfolders)

├── notebooks/
│   ├── eda.ipynb # real tag frequency + valence/arousal analysis
│   └── demo_context.ipynb # real end-to-end inference example

├── src/
│   ├── audio_features.py, graph_builder.py, bert_encoder.py,
│   │   gnn_model.py, fusion_model.py, contrastive.py, train.py, evaluate.py
│   ├── datasets.py # Dataset classes
│   ├── prepare_magnatagatune.py, prepare_deam.py, prepare_musiccaps.py,
│   │   download_musiccaps.py # real data preparation (produced the submitted results)
│   ├── pack_dir.py, unpack_dir.py # graph persistence utilities
│   ├── train_task1_with_curves.py # Task 1 F1-per-epoch tracking
│   └── make_toy_dataset.py # synthetic data generator, used for pipeline testing only

├── results/
│   ├── metrics.json # consolidated real results, all 4 tasks + DEAM
│   ├── plots/
│   │   └── tsne_fusion_z.png
│   ├── retrieval_examples/
│   │   └── task4_qualitative.json
│   └── (per-task loss histories, example predictions, case studies)

└── report/
    ├── final_report.pdf
    └── final_report.tex
```

See "Structural deviations from the assignment's example" below for why
this differs from the assignment's single-dataset illustrative structure.

## Reproducing the real results

### MagnaTagATune (Tasks 1-3)
```bash
python src/prepare_magnatagatune.py --raw_dir <path_to_downloaded_mtat> \
    --feat_dir /content/mtat_features --splits_dir data/splits --max_per_split 1500
python src/graph_builder.py
python src/train.py --config config_mtat.yaml --task 1 --epochs 12
python src/train.py --config config_mtat.yaml --task 2 --epochs 12
python src/train.py --config config_mtat.yaml --task 3 --epochs 12          # cross-attention
python src/train.py --config config_mtat_concat.yaml --task 3 --epochs 12  # early-concat ablation
python src/train_task1_with_curves.py --config config_mtat.yaml --epochs 12
python src/evaluate.py --task 1 --config config_mtat.yaml
python src/evaluate.py --task 2 --config config_mtat.yaml --baselines
python src/evaluate.py --task 3 --config config_mtat.yaml
```
MagnaTagATune's own `annotations_final.csv` is parsed directly (no
external split-file dependency); see the script's docstring for the
hex-folder split logic.

### DEAM (Task 3's emotion-regression extension)
```bash
python src/prepare_deam.py --raw_dir <path_to_downloaded_deam>
python src/train.py --config config_deam.yaml --task 3 --epochs 12
python src/evaluate.py --task 3 --config config_deam.yaml
```
Trained as a fully separate experiment from MTAT (`tag_weight: 0`,
real `emotion_alpha`/`emotion_beta`) since DEAM and MTAT are different
recordings with no genuine per-track label alignment.

### MusicCaps (Task 4)
```bash
python src/download_musiccaps.py --out_dir <out_dir> --cookies <cookies.txt> \
    --checkpoint_dir data/musiccaps_checkpoint
python src/prepare_musiccaps.py --manifest <out_dir>/downloaded_manifest.csv
python src/train.py --config config_musiccaps.yaml --task 4 --epochs 12
python src/evaluate.py --task 4 --config config_musiccaps.yaml
```
Requires `cookies.txt` (exported browser cookies) to bypass YouTube's
anti-bot check on automated downloads. Expect ~30-40% download yield
due to unavailable videos and rate-limiting -- this is normal.

### Pipeline sanity check (toy data, no downloads needed)
```bash
python src/make_toy_dataset.py
python src/audio_features.py && python src/graph_builder.py
python src/train.py --task 1 --epochs 2   # repeat for --task 2/3/4
```
This uses synthetic data only to verify the pipeline runs end-to-end;
it does not produce reportable results.

## Structural deviations from the assignment's example

This repo extends the assignment's single-dataset example structure to
support three real datasets (MagnaTagATune, DEAM, MusicCaps) instead of
one. Every addition is functional, not incidental:

- `config_mtat.yaml`, `config_deam.yaml`, `config_musiccaps.yaml`,
  `config_mtat_concat.yaml`: one config per dataset/ablation variant
  (vs. a single `config.yaml`), since each needs different
  paths/hyperparameters.
- `src/datasets.py`: Dataset classes used by `train.py`/`evaluate.py`.
- `src/prepare_magnatagatune.py`, `prepare_deam.py`, `prepare_musiccaps.py`,
  `download_musiccaps.py`: the actual code that produced the real,
  submitted data/results for each dataset -- required for reproducibility.
- `src/pack_dir.py`/`unpack_dir.py`: utilities to persist/restore graph
  data given Colab's ephemeral storage and GitHub's 100MB file limit.
- `src/train_task1_with_curves.py`: tracks validation Macro-F1/Micro-F1
  per epoch for Task 1 (the assignment's own requirement), which the
  main `train.py` does not track by default (only training loss).
- `data/packed/`: compressed graph archives (safe to commit; unpacked
  locally via `unpack_dir.py`).
- `data/splits/deam/`, `data/splits/musiccaps/`: nested under the main
  `splits/` folder alongside MagnaTagATune's own split files.
- `data/raw/` is intentionally absent: raw audio (~GBs) is
  redownloadable via the `prepare_*.py`/`download_musiccaps.py` scripts
  and not committed, standard practice for datasets this size.
- `results/`: contains the required `metrics.json`, `plots/`,
  `retrieval_examples/`, plus per-task loss histories, example
  predictions, and case studies referenced directly in the report.
