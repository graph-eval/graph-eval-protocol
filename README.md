# Evaluation Protocol for Graph-Derived Signals in Tabular Learning

This repository contains the official implementation of the evaluation protocol and the representative case study results as described in the paper: 
**"A Systematic Evaluation Protocol of Graph-Derived Signals for Tabular Machine Learning"**.

The framework is built on a **modular, multi-stage architecture**, ensuring full reproducibility of the experimental results obtained on the Elliptic (Bitcoin) transaction Dataset.

**This repository provides:**
- the full reference implementation of the evaluation protocol proposed in the paper,
- all experimental stages used to generate the reported results on the Elliptic dataset,
- a modular and reproducible pipeline for evaluating graph-derived signals in tabular learning.

The code is organized to mirror the methodological structure of the paper.

The following section provides a high-level overview of the evaluation pipeline.
Detailed implementation instructions and execution guidelines are documented in dedicated README files within the respective subdirectories.
For reproducibility, each stage provides a corresponding `environment.yml` specifying all required dependencies.


---

## Repository Structure and Pipeline Stages

The evaluation process is divided into seven logical steps, reflecting the rigorous methodology of our proposed protocol:

### 1. Data Foundation & Perturbation (`src/EllipticDataSet`)
This folder contains all input data required for the downstream classification task, including:

- the original Elliptic transaction graph and node features  
- label information and predefined data splits  
- scripts for generating **random edge removals**, which are used to perform robustness analyses under controlled graph perturbations  

This stage serves as the fixed input layer for all subsequent experiments.

### 2. Taxonomy-Driven Feature Generation (`src/Feature_Generation`)
This module implements the generation of **graph-derived signals** and the construction of
reproducible **feature artifacts** used in all downstream experiments.

It includes:
- scripts for generating diverse types of graph-derived signals,
  such as proximity embeddings, structural representations, and centrality-based indicators
- preprocessing utilities for **merging base features, labels, and graph-derived signals**
- controlled **dataset splitting strategies**, including:
  - random train/validation/test splits
  - temporally consistent splits for time-aware evaluation
- export routines for creating standardized **artifact bundles** (features, labels, metadata)

All generated artifacts are:
- stored persistently (e.g., Parquet format),
- annotated with configuration metadata (e.g., split mode, seeds, variants),
- and reused across classifiers and experimental runs without recomputation.

### 3. Graph Neural Networks (`src/GNN`)
Implementation of Graph Neural Network representatives (GCN, GAT) and Graph Contrastive Learning (GCL). This folder allows for both embedding generation and end-to-end GNN performance comparisons.

### 4. Artifact Management (`src/artifacts`)
A centralized storage for intermediate feature sets generated in the previous steps. These artifacts serve as the standardized input for the supervised learning pipeline, supporting both single-task and batch-mode processing.

### 5. Controlled Supervised Classification (`src/Supervised_Classification`)
To ensure statistical rigor and prevent optimization bias, this module implements a two-stage approach for six standard classifiers:
* **Stage A (Optimization):** Automated hyperparameter optimization (Bayesian search) applied to specific artifacts.
* **Stage B (Validation):** Multi-seed application of classifiers using the fixed optimal hyperparameters from Stage A to assess performance variance.

### 6. Standardized Result Logging (`src/Results_...`)
Automated export of experimental logs, structured for reproducibility and follow-up analysis. Results are distinguished by:
* Optimized vs. fixed hyperparameter configurations.
* Multi-seeded performance distributions.
* Robustness metrics under varying structural graph perturbations.
Result folders are generated automatically per experimental configuration.

### 7. Statistical Analysis & Evaluation (`src/Analysis`)
The core analytical suite of the protocol, providing scripts for:
* **Winsorized F1-Scores:** For robust mean estimation.
* **McNemar Significance Testing:** Formal hypothesis testing to validate performance gains.
* **Stability:** Stability and variance analysis across random seeds.
* **Robustness Analysis:** Quantifying the decay of signal utility under structural noise.
* **Metric Calculation:** Additional derived metrics and comparative summaries.

---

## Note on Reproducibility and Results

This repository includes result logs to allow immediate verification of the findings presented in the paper without re-running the entire computational pipeline.
All reported results are derived from the publicly available Elliptic Bitcoin dataset.
The repository does not contain any proprietary or restricted data beyond what is already publicly accessible.

---

## Technical Requirements
- **Reproducibility:** Each module contains an `environment.yml` file.
- **Modularity:** Steps can be executed independently if the required artifacts are present.
- **Hardware:** CPU-optimized for classical ML; GPU support for GNN modules (DGL/PyTorch).

---

## Contact
For scientific inquiries regarding the protocol or the dataset:
**Mario Heidrich** - [heidrichmario@gmail.com](mailto:heidrichmario@gmail.com)
