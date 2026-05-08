# data/

This folder contains the partition and statistics files generated from the FeTS 2022 training pool.
These files are required by all Phase 4 notebooks and the Report Figures notebook.

## Files


### final_data_partitions.json
Defines the 5-client federation split. Contains:
- Per-client train case paths (80 cases each, capped via stratified sampling)
- Per-client validation case paths (10 cases each, stratified to guarantee small-lesion coverage)
- Locked test set paths (90 cases: 30 small / 30 medium / 30 large)

Generated once using Dirichlet distribution (alpha=0.5) applied independently per size group.

### master_lesion_stats.csv
Per-case lesion volume statistics for all FeTS 2022 training cases. Contains:
- Case ID / path
- Whole-tumour volume in cm3
- Assigned size group (small < 30 cm3 / medium 30-100 cm3 / large > 100 cm3)

Used to assign size groups during partitioning and evaluation.

## Note
The actual FeTS 2022 MRI data is NOT included here due to size and data use agreements.
Download it from: https://fets-ai.github.io/Challenge/
