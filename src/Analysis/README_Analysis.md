# Analysis

This directory implements **Stage 7 (Statistical Analysis & Evaluation)**
of the evaluation protocol.

It contains post-hoc analysis scripts used to aggregate results, perform
statistical significance testing, and generate figures reported in the paper.

The scripts operate exclusively on result files produced by earlier stages
(e.g., supervised classification and robustness experiments).

---

## Scope

The analysis includes, among others:
- aggregation of metrics across multiple runs
- winsorized score aggregation
- McNemar significance testing
- robustness and edge-removal analysis
- figure and table generation for reporting

All outputs are written to the corresponding `Output_*` subdirectories.

---

## Environment Notes

No dedicated Conda environment is required for this stage.

The scripts rely on standard scientific Python packages.
Support for reading Parquet files requires `fastparquet`,
which is already included in the environments of earlier stages
(e.g., supervised classification).

---

## Note

This stage is **optional** for reproducing the main experimental results,
as all reported analyses are generated from stored result artifacts.
