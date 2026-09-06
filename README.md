# GNN-BERT Music Context Understanding

Course: Neural Networks (CSE425 / EEE474 / CSE715)
Project: Hybrid BERT + GNN system for understanding musical context
(genre/mood tagging, structure-aware classification, cross-modal fusion,
and contrastive audio-text retrieval).

## Status of this repo

This repo is built and tested against a small **synthetic toy dataset**
(`src/make_toy_dataset.py`) that mimics the schema of the real datasets
(FMA / MagnaTagATune / MusicCaps / DEAM). All model code (Task 1-4) runs
end-to-end on the toy data, so the pipeline is verified correct.

**Before submitting**, you need to:
1. Download the real dataset(s) (see links in the project PDF, Table 1).
2. Point `config.yaml` at the real data paths.
3. Re-run `src/audio_features.py` + `src/graph_builder.py` on real audio.
4. Re-train each task's model on real data (GPU strongly recommended —
   use Colab or a university GPU cluster; full BERT fine-tuning on
   FMA-medium will not run in a CPU-only sandbox in reasonable time).
5. Regenerate the plots/tables in `results/` and write them into
   `report/final_report.pdf`.

## Structure

```
gnn-bert-music-context/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/            # place FMA / MagnaTagATune / MusicCaps / DEAM downloads here
│   ├── processed/       # graphs, mel-spec, BERT caches (generated)
│   └── splits/          # train/val/test JSON (generated)
├── notebooks/
│   ├── eda.ipynb
│   └── demo_context.ipynb
├── src/
│   ├── make_toy_dataset.py   # synthetic data generator for local dev/testing
│   ├── audio_features.py     # mel, chroma, segmentation
│   ├── graph_builder.py      # chord + segment graphs
│   ├── bert_encoder.py       # BERT wrapper
│   ├── gnn_model.py          # GraphSAGE / GAT
│   ├── fusion_model.py       # cross-attention GNN-BERT (Task 3)
│   ├── contrastive.py        # Task 4 InfoNCE dual-encoder
│   ├── datasets.py           # PyTorch Dataset/DataLoader wrappers
│   ├── train.py              # unified trainer (--task 1|2|3|4)
│   ├── evaluate.py           # metrics: F1, AUC-PR, MAE, R@K, t-SNE, baselines
│   └── offline_stub.py       # SANDBOX-ONLY smoke test, see note below
├── results/
│   ├── metrics.json
│   ├── plots/
│   └── retrieval_examples/
└── report/
    └── final_report_skeleton.md   # Overleaf-ready structure, fill in real results
```

## A note on how this repo was verified

Built in a sandbox with **no network access to huggingface.co** or the
real dataset hosts. So:

- `src/train.py --task 1/2/3/4` is the **real** training code you'll use
  for your actual submission (with `distilbert-base-uncased` downloaded
  from HF Hub on your own machine/Colab).
- `src/offline_stub.py --smoke_test_task N` was used **only** here to
  verify Tasks 1/3/4 (which need BERT) run end-to-end without crashing,
  using a tiny randomly-initialized BERT + toy tokenizer instead of the
  real pretrained model. Its numbers are meaningless — never report them.
- Task 2 (pure GNN, no BERT) was verified with the real `train.py`
  directly (no network dependency), and its loss visibly decreased
  across epochs on the toy set.
- All four tasks were confirmed to train, backprop, and checkpoint
  correctly on 200 synthetic tracks. `evaluate.py` was confirmed to
  compute Macro-F1/Micro-F1/AUC-PR correctly (including the B1 random
  and B2 CNN baselines) and save `results/metrics.json`.

**What's still on you:** download real audio (Table 1 datasets), run
`audio_features.py` on it, merge real tags/captions/valence-arousal into
the `data/splits/*.json` schema (see `make_toy_dataset.py`'s output for
the exact format), then re-run `train.py` + `evaluate.py` with real BERT
weights (works out of the box outside this sandbox) to get numbers for
`report/final_report_skeleton.md`.

## Quickstart (toy data, CPU)

```bash
pip install -r requirements.txt

# 1. Generate toy dataset (stands in for FMA/MagnaTagATune/MusicCaps/DEAM)
python src/make_toy_dataset.py

# 2. Build audio features + graphs from the toy audio
python src/audio_features.py
python src/graph_builder.py

# 3. Train each task
python src/train.py --task 1 --epochs 3   # BERT tag baseline
python src/train.py --task 2 --epochs 3   # GNN on structure graphs
python src/train.py --task 3 --epochs 3   # GNN-BERT fusion
python src/train.py --task 4 --epochs 3   # contrastive retrieval (bonus)

# 4. Evaluate + produce plots/tables
python src/evaluate.py --task 3
```

## Swapping in real data: MagnaTagATune + MusicCaps + DEAM

Recommended combo (see the marks-comparison discussion in the project
chat/report) -- all three run on **your own machine or Colab**, not this
sandbox, since none of these hosts (`mi.soi.city.ac.uk`, YouTube,
`cvml.unige.ch`/Zenodo, `huggingface.co`) are reachable here.

### 1. MagnaTagATune -- drives Tasks 1, 2, 3 (tags + fusion)
```bash
bash scripts/download_magnatagatune.sh data/raw/magnatagatune
python src/prepare_magnatagatune.py --raw_dir data/raw/magnatagatune
python src/graph_builder.py   # unchanged -- reads data/splits/*.json
python src/train.py --config config_mtat.yaml --task 1 --epochs 10
python src/train.py --config config_mtat.yaml --task 2 --epochs 10
python src/train.py --config config_mtat.yaml --task 3 --epochs 10
```
`prepare_magnatagatune.py` is fully self-contained: it parses
`annotations_final.csv` (which MagnaTagATune ships with its own
`clip_id`/`mp3_path`/tag columns) directly, picks the top-50 tags by
frequency, and builds the standard ~12:1:3 hex-folder split -- no
dependency on any external repo's split files. Verified against a
fabricated fixture (tag parsing, folder-based splitting, and the full
chain into `graph_builder.py` all confirmed correct end to end).

`config_mtat.yaml` sets `emotion_alpha`/`emotion_beta` to 0 since MTAT
has no valence/arousal labels -- Task 3 trains its tag+fusion objective
only on this split (see the note in `prepare_deam.py` for why this
isn't merged with DEAM's labels).

### 2. DEAM -- the valence/arousal extension for Task 3
```bash
bash scripts/download_deam.sh data/raw/deam   # prints where to get it (URL not hardcoded, see script)
python src/prepare_deam.py --raw_dir data/raw/deam
python -c "
import sys; sys.path.insert(0, 'src')
from graph_builder import process_split
import shutil, os
os.makedirs('data/splits', exist_ok=True)
for s in ['train','val','test']:
    shutil.copy(f'data/splits_deam/{s}.json', f'data/splits/{s}.json')  # graph_builder reads data/splits/ by default
    process_split(s, out_dir='data/processed/deam_graphs')
"
python src/train.py --config config_deam.yaml --task 3 --epochs 10
```
`config_deam.yaml` sets `tag_weight: 0` (DEAM has no tags -- the model
still needs *some* tag dimension to share the fusion head, filled with
harmless zero-vectors, see `datasets.py`'s `MusicFusionDataset`) and
real `emotion_alpha`/`emotion_beta` for genuine valence/arousal
regression. Report this as a **separate experiment** from the MTAT run,
not a merged multi-task result -- they're different songs with no
factual per-track alignment (documented in `prepare_deam.py`).

### 3. MusicCaps -- Task 4's real captions (bonus)
```bash
python src/download_musiccaps.py --out_dir data/raw/musiccaps --limit 1000  # drop --limit for the full 5,521
python src/prepare_musiccaps.py --manifest data/raw/musiccaps/downloaded_manifest.csv
python -c "
import sys; sys.path.insert(0, 'src')
from graph_builder import process_split
import shutil, os
os.makedirs('data/splits', exist_ok=True)
for s in ['train','val','test']:
    shutil.copy(f'data/splits_musiccaps/{s}.json', f'data/splits/{s}.json')
    process_split(s, out_dir='data/processed/musiccaps_graphs')
"
python src/train.py --config config_musiccaps.yaml --task 4 --epochs 10
python src/evaluate.py --task 4 --config config_musiccaps.yaml
```
Expect 5-15% of clips to fail to download (deleted/region-locked
videos) -- this is normal for MusicCaps; report the actual yield in
your dataset section rather than treating it as a bug.

### Verified vs. not
Everything above was written against the **actual, tested** function
signatures in `datasets.py`/`graph_builder.py`/`train.py` -- the
`tags=None`/`valence=None` handling in `MusicFusionDataset`, the
`tag_weight` config option, and `prepare_magnatagatune.py`'s CSV
parsing + folder-based split were all smoke-tested against fabricated
fixtures (since the real hosts aren't reachable from this sandbox) and
confirmed to work end to end, including the full chain through
`graph_builder.py`. What's *not* verified is behavior against the real
downloaded files at full scale -- if a real file has an edge case my
fixture didn't (e.g. a malformed mp3, an unexpected CSV encoding), the
fix is almost always a one-line adjustment in the relevant `prepare_*.py`,
not a structural problem. DEAM's CSV parsing and MusicCaps' YouTube
download were smoke-tested similarly for their code paths, but their
exact live file formats (which can drift over time) weren't verified
against the real hosts.


## Actual repository structure vs. the assignment's illustrative example

This repo extends the assignment's single-dataset example structure to
support three real datasets (MagnaTagATune, DEAM, MusicCaps) instead of
one. Every addition is functional, not incidental:

- `config_mtat.yaml`, `config_deam.yaml`, `config_musiccaps.yaml`: one
  config per dataset (vs. a single `config.yaml`), since each dataset
  needs different paths/hyperparameters.
- `src/datasets.py`: Dataset classes used by `train.py`/`evaluate.py`.
- `src/prepare_magnatagatune.py`, `prepare_deam.py`, `prepare_musiccaps.py`,
  `download_musiccaps.py`: the actual code that produced the real,
  submitted data/results for each dataset -- required for reproducibility.
- `src/pack_dir.py`/`unpack_dir.py`: utilities to persist/restore graph
  data given Colab's ephemeral storage and GitHub's 100MB file limit.
- `data/packed/`: compressed graph archives (safe to commit; the
  unpacked versions are regenerated locally via `unpack_dir.py`).
- `data/splits/deam/`, `data/splits/musiccaps/`: nested under the main
  `splits/` folder alongside MagnaTagATune's own split files.
- `data/raw/` is intentionally absent: raw audio (~GBs) is
  redownloadable via the `prepare_*.py`/`download_musiccaps.py` scripts
  and not committed, standard practice for datasets this size.
- `results/`: contains the required `metrics.json`, `plots/`,
  `retrieval_examples/`, plus per-task loss histories, example
  predictions, and case studies referenced directly in the report.
