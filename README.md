# Skew-Aware Federated Learning for Brain Tumour Segmentation

**Course:** CS 437 — Distributed and Federated Machine Learning  
**Authors:** Zainab Usman (27100409) · Sameel Haider (27100045)  
**Dataset:** [FeTS 2022](https://fets-ai.github.io/Challenge/) — Federated Tumour Segmentation Challenge

---

## Overview

Standard federated learning aggregation (FedAvg) weights clients by dataset size alone. In brain tumour segmentation, small tumours are rare and unevenly distributed across hospital clients — so the clients that actually have small-tumour cases get no more influence over the global model than clients that have none. This creates a **fairness gap**: the global model learns to segment large tumours reasonably well but consistently fails on small ones.

This project proposes **SCWA-LW** (Size-Composition Weighted Aggregation with Layer-Wise routing) to fix this. Clients with more small-tumour cases get a higher aggregation weight, but *only* for the decoder layers of the U-Net. The encoder (which learns general brain anatomy useful to all clients) keeps standard FedAvg. The result is a 10% relative improvement in small-tumour Dice and 8.3% improvement in fair score over the FedAvg baseline.

---

## Results Summary

| Method | Small Dice | Medium Dice | Large Dice | Fair Score |
|---|---|---|---|---|
| Phase 2 — FedProx Baseline | 0.088 | 0.326 | 0.599 | 0.190 |
| Exp A — FedAvg Baseline | 0.116 | 0.413 | 0.694 | 0.241 |
| Exp B — Strong FTL | 0.118 | 0.446 | 0.704 | 0.247 |
| **Exp C — Full System (SCWA-LW)** | **0.128** | 0.435 | **0.716** | **0.261** |

> **Fair Score** = harmonic mean of Small / Medium / Large Dice.  
> Test set: 90 locked cases (30 small / 30 medium / 30 large), evaluated with sliding-window inference at overlap 0.75.

---

## Folder Structure

```
lesion-skewed-fl-segmentation/
│
├── README.md
├── requirements.txt
│
├── phase2_fedprox_baseline/
│   ├── Phase2_FedAvg_Experiments.ipynb    # FedAvg runs exploring local epoch count and hyperparams
│   └── Phase2_FedProx_Baseline.ipynb      # Final FedProx baseline, small model (4.7M), standard Dice loss
│
├── phase3_lads/
│   └── Phase3_LADS.ipynb                  # LADS experiment (see note below — failed)
│
├── phase4_experiments/
│   ├── Phase4_Baseline.ipynb              # Exp A: FedAvg + mild FTL, 50 rounds
│   ├── Experiment_B_Final.ipynb           # Exp B: FedAvg + strong FTL, 30 rounds
│   ├── Experiment_C_Final.ipynb           # Exp C: FedProx + SCWA-LW + strong FTL, 30 rounds
│   └── Report_Figures_Notebook.ipynb      # Loads saved outputs, generates all report figures
│
├── report/
│   ├── main.tex                           # ICML 2021 format LaTeX source
│   └── main.bib                           # Bibliography
│
└── outputs/                               # Gitignored — checkpoints, history CSVs, figures
    ├── fedavg_baseline/
    ├── fedavg_lft_strong/
    └── fedprox_scwa_lw/
```

---

## Phase Descriptions

### Phase 2 — Experimentation and Baselines

#### FedAvg Experiments
Before settling on a final configuration, we ran a series of FedAvg experiments to understand how the federated training setup behaves and find a good starting point. The main thing we were trying to figure out was the **optimal number of local epochs per round** — too few and the model doesn't learn enough from each client per round; too many and the clients diverge too far from each other (client drift), which hurts the global model when weights are averaged.

We tried different values and observed how training stability and validation Dice changed. These experiments gave us the intuition and hyperparameter choices (learning rate, batch size, patch size) that carried forward into all Phase 4 experiments.

#### FedProx Baseline
Once we had a stable FedAvg setup, we switched to FedProx regularisation (μ=0.01) as the formal Phase 2 submission. FedProx adds a penalty during local training that stops each client's weights from drifting too far from the global model, which helps in non-IID settings like ours.

This used a smaller 3D U-Net (channels 16–256, ~4.7M parameters), standard Dice loss, and 1 local epoch per round. It serves as the reference point that all Phase 4 experiments are compared against, and it shows clearly how poorly small tumours are handled without any size-aware strategy (Small Dice = 0.088).

---

### Phase 3 — LADS (Failed Approach)

**LADS** (Lesion-Aware Dynamic Sampling) attempted to address small-tumour under-representation by oversampling small-lesion patches during local training so each client's gradient signal was biased toward small tumours regardless of how many small-tumour cases that client had.

**Why it failed:**
- Oversampling small patches at the *local* training level does improve per-client gradients, but those improvements are completely diluted at aggregation. FedAvg still averages all clients equally, so the enriched gradient signal from a small-lesion-rich client is washed out by clients that have no such cases.
- The approach treated the problem as a *local* data imbalance issue, but the real problem is at the *aggregation* level — who gets to influence the global model and by how much.
- Result: no meaningful improvement over the Phase 2 baseline on small-tumour Dice, confirming that fixing local training alone is insufficient.

This failure directly motivated SCWA-LW, which addresses the problem at the aggregation step instead.

---

### Phase 4 — Controlled Ablation (Main Contribution)

Three experiments on the same 5-client FeTS 2022 federation (Dirichlet α=0.5, 80 cases/client):

| Experiment | Strong FTL | FedProx | SCWA-LW | Rounds |
|---|:---:|:---:|:---:|:---:|
| A: FedAvg Baseline | ✗ | ✗ | ✗ | 50 |
| B: Loss Only | ✓ | ✗ | ✗ | 30 |
| C: Full System | ✓ | ✓ | ✓ | 30 |

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
