# Hybrid Music Clustering (Bangla + English) — Easy, Medium, Hard

This repository implements a full pipeline for hybrid music clustering using:
- Handcrafted audio features (69-dim)
- Log-mel spectrograms (for conv VAE / β-VAE)
- Baselines (PCA + K-Means)
- Latent clustering (VAE/AE + K-Means/Agglomerative/DBSCAN)
- Hard task: multimodal clustering using audio + language + genre (and optional lyrics)

> Note: Raw audio is not included in this repo. Use the dataset preparation steps below to reproduce.

---

## 1) Setup

### 1.1 Create and activate environment

py -m venv .venv
.\.venv\Scripts\Activate.ps1

### 1.2 Install dependencies

py -m pip install --upgrade pip
py -m pip install -r requirements.txt

### 2) Data layout (local machine)

Expected local paths (not pushed to GitHub):

data/raw/bangla/... (Bangla audio files)

data/raw/english_fma/... (English audio files from FMA subset)

Balanced subset folders created by script:

data/raw_balanced_500_hard/bangla

data/raw_balanced_500_hard/english

### 3) Easy Task (Audio features --> VAE --> K-Means + Visualization + Baseline)
3.1 Build dataset (audio features)
py .\scripts\build_dataset.py --sources .\sources_500.yaml --out_dir data/processed --seg_seconds 10 --num_segments 2 --seg_strategy random --max_files_total 500 --min_duration 1

## 3.2 Clean dataset
py .\scripts\clean_dataset_cli.py --in_dir data/processed

## 3.3 Make splits
py .\scripts\make_splits_cli.py --in_csv data/processed/dataset_clean.csv --out_dir data/processed/splits

## 3.4 Baseline: PCA + K-Means
py .\scripts\baseline_pca_kmeans.py


Outputs go to data/processed/results_baseline/

## 3.5 Train VAE + cluster + visualize (t-SNE)
py .\scripts\train_vae.py
py .\scripts\vae_kmeans_tsne.py


Outputs go to data/processed/results_vae/

### 4) Medium Task (Log-mels + ConvVAE → clustering evaluation; optional lyrics hybrid)
## 4.1 Build dataset with log-mels
py .\scripts\build_dataset.py --sources .\sources_500.yaml --out_dir data/processed_medium --save_logmels --seg_seconds 3 --num_segments 1 --seg_strategy random --max_files_total 500 --min_duration 1

## 4.2 Clean + split
py .\scripts\clean_dataset_cli.py --in_dir data/processed_medium
py .\scripts\make_splits_cli.py --in_csv data/processed_medium/dataset_clean.csv --out_dir data/processed_medium/splits

## 4.3 Train ConvVAE and export latent
py .\scripts\train_conv_vae.py --in_dir data/processed_medium --latent_dim 16 --epochs 60 --batch_size 32

## 4.4 Evaluate clustering (audio features vs conv latent)
py .\scripts\medium_cluster_eval.py

## 4.5  Jamendo lyrics hybrid

Build lyrics embeddings:

py .\scripts\build_lyrics_embeddings.py --dataset_dir data/processed_jamendo --lyrics_root jamendolyrics/lyrics


Build hybrid features:

py .\scripts\make_hybrid_features.py --dataset_dir data/processed_jamendo


Evaluate:

py .\scripts\medium_cluster_eval_cli.py --data_dir data/processed_jamendo --use_hybrid

### 5) Hard Task (Balanced 250+250 + genre + β-VAE + multimodal clustering)

## 5.1 Prepare English subset (FMA)
py .\scripts\prepare_fma_english_250.py --tracks_csv "data/raw/fma_downloads/fma_metadata/fma_metadata/tracks.csv" --fma_small_root "data/raw/fma_downloads/fma_small/fma_small" --out_root "data/raw/english_fma" --total 250

## 5.2 Create balanced 500 subset
py .\scripts\make_balanced_subset.py --bangla_root data/raw/bangla --english_root data/raw/english_fma --out_root data/raw_balanced_500_hard --n_bangla 250 --n_english 250 --out_sources_yaml sources_500_hard.yaml

## 5.3 Build hard dataset (log-mels + audio features)
py .\scripts\build_dataset.py --sources .\sources_500_hard.yaml --out_dir data/processed_hard --save_logmels --seg_seconds 3 --num_segments 1 --seg_strategy random --max_files_total 500 --min_duration 1

## 5.4 Clean + split
py .\scripts\clean_dataset_cli.py --in_dir data/processed_hard
py .\scripts\make_splits_cli.py --in_csv data/processed_hard/dataset_clean.csv --out_dir data/processed_hard/splits

## 5.5 Add genre column
py .\scripts\add_genre_column.py

## 5.6 Train β-VAE (log-mels)
py .\scripts\train_beta_vae.py --csv data/processed_hard/dataset_clean_with_genre.csv --splits_dir data/processed_hard/splits --out_dir data/processed_hard/results_beta_vae --latent_dim 16 --beta 4 --epochs 80 --batch_size 32 --lr 1e-3

## 5.7 Train AE baseline on audio_features (69-dim)
py .\scripts\train_audio_ae.py --data_dir data/processed_hard --splits_dir data/processed_hard/splits --out_dir data/processed_hard/results_audio_ae --latent_dim 16 --epochs 250 --batch_size 64 --lr 1e-3

## 5.8 Hard evaluation + final outputs
py .\scripts\hard_cluster_eval.py --csv data/processed_hard/dataset_clean_with_genre.csv --data_dir data/processed_hard --beta_latent data/processed_hard/results_beta_vae/latent_mu.npy --ae_latent data/processed_hard/results_audio_ae/latent.npy --out_dir data/processed_hard/results_hard_final --k_min 2 --k_max 12 --seed 42


### Final outputs are saved to:

data/processed_hard/results_hard_final/metrics_summary.csv

data/processed_hard/results_hard_final/cluster_assignments_best.csv

data/processed_hard/results_hard_final/umap_or_tsne_*.png

data/processed_hard/results_hard_final/best_run.json







