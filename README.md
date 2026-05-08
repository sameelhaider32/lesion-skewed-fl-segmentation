# Skew-Aware Federated Learning for Brain Tumour Segmentation

**Course:** CS 437 — Deep Learning  
**Authors:** Zainab Usman (27100409) · Sameel Haider (27100045)  
**Dataset:** [FeTS 2022](https://fets-ai.github.io/Challenge/) — Federated Tumour Segmentation Challenge

---

## Overview

Standard federated learning aggregation (FedAvg) weights clients by dataset size alone. In brain tumour segmentation, small tumours are rare and unevenly distributed across hospital clients, so the clients that actually have small-tumour cases get no more influence over the global model than clients that have none. This creates a **fairness gap**: the global model learns to segment large tumours reasonably well but consistently fails on small ones.

This project proposes **SCWA-LW** (Size-Composition Weighted Aggregation with Layer-Wise routing) to fix this. Clients with more small-tumour cases get a higher aggregation weight, but *only* for the decoder layers of the U-Net. The encoder (which learns general brain anatomy useful to all clients) keeps standard FedAvg. The result is a 10% relative improvement in small-tumour Dice and 8.3% improvement in fair score over the FedAvg baseline.

---

## Results Summary

| Method | Small Dice | Medium Dice | Large Dice | Fair Score |
|---|---|---|---|---|
| Phase 2 — FedProx Baseline | 0.088 | 0.326 | 0.599 | 0.190 |
| Exp A — FedAvg Baseline | 0.116 | 0.413 | 0.694 | 0.241 |
| Exp B1 — Strong FTL | 0.118 | 0.446 | 0.704 | 0.247 |
| Exp B2 — FedProx + FTL (no SCWA) | 0.107 | 0.402 | 0.681 | 0.225 |
| **Exp C — Full System (SCWA-LW)** | **0.128** | 0.435 | **0.716** | **0.261** |

> **Key finding:** B2 (FedProx without SCWA-LW) scores *below* the FedAvg baseline on small tumours, confirming that SCWA-LW alone drives the improvement. C vs B2 = +0.021 small Dice, +0.036 fair score (pure SCWA contribution).

> **Fair Score** = harmonic mean of Small / Medium / Large Dice.  
> Test set: 90 locked cases (30 small / 30 medium / 30 large), evaluated with sliding-window inference at overlap 0.75.

---

## Folder Structure

```
lesion-skewed-fl-segmentation/
│
├── README.md
├── .gitignore
│
├── data/
│   ├── README.md                              # Explains the two data files below
│   ├── final_data_partitions.json             # 5-client train/val/test splits
│   └── master_lesion_stats.csv               # Per-case lesion volumes and size groups
│
├── deliverables/
│   ├── Deliverable 2.ipynb                    # Phase 2 formal submission notebook
│   ├── phase2_fedprox_baseline/
│   │   └── fedavg + fedprox (35 rounds).ipynb    # FedAvg & FedProx experiments, local epoch tuning
│   ├── phase3_lads/
│   │   └── LADS_COMPLETE.ipynb                    # LADS experiment (see note below — failed)
│   │   └── Group20_27100045_27100409_phase3_baseline.ipynb                    # Comparable phase 3 baseline
│   └── phase4_experiments/
│       ├── Experiment_A.ipynb                     # Exp A: FedAvg + mild FTL, 50 rounds
│       ├── Experiment_B-FINAL.ipynb               # Exp B1: FedAvg + strong FTL, 30 rounds
│       ├── Experiment_B2.ipynb                    # Exp B2: FedProx + strong FTL, NO SCWA, 30 rounds
│       ├── Experiment_C-FINAL.ipynb               # Exp C: FedProx + SCWA-LW + strong FTL, 30 rounds
│       └── Report_Figures_Notebook.ipynb          # Loads saved outputs, generates all report figures
│
├── report/
│   ├── main.tex                               # ICML 2021 format LaTeX source
│   └── main.bib                               # Bibliography
│
└── outputs/                                   # Gitignored — checkpoints, history CSVs, figures
```

---

## Phase Descriptions

### Data Files (`data/`)
Two files are required before running any Phase 4 notebook:

- **`final_data_partitions.json`** — defines the 5-client federation. Contains per-client train paths, per-client validation paths (10 cases each, stratified to guarantee small-lesion coverage), and the locked 90-case test set. Generated once using Dirichlet (α=0.5) applied independently per size group.
- **`master_lesion_stats.csv`** — per-case whole-tumour volume (cm³) and assigned size group (small/medium/large) for all FeTS 2022 cases. Used during partitioning and at evaluation to assign each test case to its size group.

---

### Phase 2 — Experimentation and Baselines (`deliverables/phase2_fedprox_baseline/`)

**`fedavg + fedprox (35 rounds).ipynb`** covers two things in one notebook:

#### FedAvg Experiments
Before settling on a configuration we ran FedAvg with different numbers of local epochs per round to understand how the setup behaves. The key question was finding the right local epoch count — too few and the model doesn't learn enough per round; too many and clients diverge too far from each other (client drift), which hurts aggregation. These runs gave us the hyperparameter intuition carried forward into all Phase 4 experiments.

#### FedProx Baseline
With a stable setup confirmed, we added FedProx regularisation (μ=0.01), which penalises each client for drifting too far from the global model during local training. This used a smaller 3D U-Net (channels 16–256, ~4.7M parameters), standard Dice loss, and ran for 35 rounds. It serves as the earliest reference point and shows clearly how poorly small tumours are handled without any size-aware strategy (Small Dice = 0.088).

---

### Phase 3 — LADS (Failed Approach) (`deliverables/phase3_lads/`)

**LADS** (Lesion-Aware Dynamic Sampling) attempted to address small-tumour under-representation by oversampling small-lesion patches during local training so each client's gradient signal was biased toward small tumours regardless of how many small-tumour cases that client had.

**Why it failed:**
- Oversampling small patches at the *local* training level does improve per-client gradients, but those improvements are completely diluted at aggregation. FedAvg still averages all clients equally, so the enriched gradient signal from a small-lesion-rich client is washed out by clients that have no such cases.
- The approach treated the problem as a *local* data imbalance issue, but the real problem is at the *aggregation* level — who gets to influence the global model and by how much.
- Result: no meaningful improvement over the Phase 2 baseline on small-tumour Dice, confirming that fixing local training alone is insufficient.

This failure directly motivated SCWA-LW, which addresses the problem at the aggregation step instead.

---

### Phase 4 — Controlled Ablation (Main Contribution) (`deliverables/phase4_experiments/`)

Four experiments on the same 5-client FeTS 2022 federation (Dirichlet α=0.5, 80 cases/client):

| Experiment | Strong FTL | FedProx | SCWA-LW | Rounds |
|---|:---:|:---:|:---:|:---:|
| A: FedAvg Baseline | ✗ | ✗ | ✗ | 50 |
| B1: Strong FTL | ✓ | ✗ | ✗ | 30 |
| B2: FedProx + FTL (no SCWA) | ✓ | ✓ | ✗ | 30 |
| C: Full System | ✓ | ✓ | ✓ | 30 |

**Experiment B2 is the critical control.** It adds FedProx on top of B1's loss config but deliberately excludes SCWA-LW. B2 scores *below* Baseline A on small tumours (0.107 vs 0.116), showing that FedProx alone is actively harmful — it suppresses Client 1's beneficial divergence without fixing the aggregation bias. Comparing C to B2 gives the cleanest isolation of SCWA-LW's contribution (+0.021 small Dice, +0.036 fair score) since the only difference between them is the aggregation weights.

**Model:** 3D U-Net, channels=(32, 64, 128, 256, 320), ~12.9M parameters, patch size 96³.

**SCWA-LW** computes a weight per client:

$$\tilde{w}_i = 0.5 \cdot \frac{n_i}{N} + 0.5 \cdot \frac{s_i}{S}$$

where $n_i$ = total cases, $s_i$ = small-tumour cases. These weights are applied **only to decoder layers**; encoder layers use standard FedAvg.

**Report_Figures_Notebook.ipynb** loads the saved checkpoints and history CSVs from Drive and generates all publication figures without retraining.

---

## Setup

### Requirements
```bash
pip install -r requirements.txt
```

Key dependencies:
```
monai>=1.2
torch>=2.0
nibabel
pandas
matplotlib
```

### Dataset
Download the [FeTS 2022 Training Data](https://fets-ai.github.io/Challenge/) and place it at:
```
/content/drive/MyDrive/FeTS_unzipped/train/MICCAI_FeTS2022_TrainingData/
```
Each case folder should contain four NIfTI files: `*flair*`, `*t1*`, `*t1ce*`, `*t2*`, and `*seg*`.

### Running Experiments
All notebooks are designed to run on **Google Colab with an A100 GPU**. Mount your Google Drive and set the paths in the `PATHS` cell at the top of each notebook. Estimated runtime: ~1.5–2 hours per experiment on A100.

---

## Citation

If you use this code, please cite the FeTS 2022 benchmark:
```
@inproceedings{fets2022,
  title={The Federated Tumor Segmentation (FeTS) Challenge 2022},
  ...
}
```
```
