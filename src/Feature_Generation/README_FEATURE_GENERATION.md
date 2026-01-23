# Elliptic Feature Generation Pipeline – Consolidated Master README

This document is a **fully unified, consolidated, and extended README** for the entire Elliptic Feature Generation pipeline.  
It includes operational details, such as:

- Correct conda environments  
- Required working directory  
- How to run **every** compute script  
- Edge-drop behavior (new vs. old version)  
- Naming conventions  
- Preprocessing, splits, and artifact generation  
- Reproducibility design  
- Full workflow examples  

It is designed as the **single source of truth** for the entire feature extraction system.

---

# 1. Overview

The Elliptic Feature Generation pipeline creates all graph-based and structural features required for downstream ML experiments, including:

- **Graph Indicators**  
  (Degree, PageRank, Clustering, Betweenness, Louvain, Leiden, etc.)

- **Structural Embeddings**  
  (Spectral, GraphWave, Role2Vec, ffstruc2vec)

- **Proximity Embeddings**  
  (Deterministic Node2Vec variants: Balanced, BFS, DFS · DeepWalk)

- **GNN-based Embeddings**  
  (GCN, GAT, GCL)

- **Preprocessing & Export**  
  (Random split, temporal split, metadata tracking)

All generated features are stored as **Parquet files** under:

```
Feature_Generation/artifacts/
```

The pipeline supports both:
- **Full graph**, and  
- **Edge-dropped variants** using parameter `--drop_rate`.

---

# 2. Required Environment and Working Directory

## 2.1 Conda Environment

For reproducibility, environment specifications are provided as Conda YAML files.
For feature generation, method-specific environments are located under
`src/Feature_Generation/environments`.
Additional stages provide their corresponding `environment.yml` files within the respective subdirectories.

## 2.2 IMPORTANT: Correct Working Directory

All compute scripts use **relative paths**, such as:

```
../EllipticDataSet/
../artifacts/
```

Therefore, **you MUST run all compute scripts from:**

```
<project_root>\src\Feature_Generation
```

Change into this directory before running *any* script:

```bash
cd <project_root>\src\Feature_Generation
```

If you run scripts from elsewhere, **Edgelists and artifacts will not be found**.

---

# 3. Directory Structure

```
Feature_Generation/
│
├── compute_graph_indicators.py
├── compute_spectral_embeddings.py
├── compute_graphwave_embeddings.py
├── compute_role2vec_embeddings.py
├── compute_ffstruc2vec_embeddings.py
├── compute_proximity_embeddings.py
├── compute_leiden_communities.py
│
├── elliptic_data.py
├── preprocessing_export.py
├── run_export_grid.py
└── export_wrapper_random.py
```

---

# 4. Edgelist Inputs & Edge-Drop Variants

## 4.1 Full Graph Edgelist

```
EllipticDataSet/elliptic_txs_edgelist.csv
```

## 4.2 Edge-Dropped Edgelists

New naming convention (seed removed):

```
elliptic_txs_edgelist_25.csv
elliptic_txs_edgelist_50.csv
```

To generate:

```bash
python make_edge_subsample.py --drop_percent 25
```

Seed is fixed internally to 42 but **never appears in filenames**.

---

# 5. Unified Naming Rules

## 5.1 Full Graph

No suffix:

```
spectral_embeddings.parquet
role2vec_embeddings.parquet
degree_centrality.parquet
```

## 5.2 Edge-Drop Variants

Suffix = `_{drop_rate}`:

```
spectral_embeddings_25.parquet
role2vec_embeddings_25.parquet
degree_centrality_25.parquet
```

## 5.3 No Seed in Any Feature Filename

All compute scripts internally use:

```
SEED = 42
```

…but this **does NOT appear in filenames**.

This makes filenames stable and consistent.

---

## 6. CLI Parameters for All Compute Scripts

All compute scripts support exactly one of the following arguments:

--drop_rate <int>
--variant <string>

Rules:
- `--variant` and `--drop_rate` are mutually exclusive
- If `--variant` is set, it overrides `--drop_rate`
- If neither is set, the full graph is used


---

# 7. How to Run Every Compute Script

Below: always assume:

```bash
conda activate node_embedding_p
cd <project_root>\src\Feature_Generation
```

---

## 7.1 Graph Indicators

```
python compute_graph_indicators.py
python compute_graph_indicators.py --drop_rate 25
```

Outputs:

```
degree_centrality.parquet
degree_centrality_25.parquet
...
```

---

## 7.2 Spectral Embeddings

```
python compute_spectral_embeddings.py
python compute_spectral_embeddings.py --drop_rate 25
```

---

## 7.3 GraphWave Embeddings

```
python compute_graphwave_embeddings.py
python compute_graphwave_embeddings.py --drop_rate 25
```

Outputs use:

```
graphwave_embeddings.parquet
graphwave_embeddings_25.parquet
```

---

## 7.4 Role2Vec

Role2Vec = WL relabeling → structural walks → Word2Vec.

### Full Graph

```
python compute_role2vec_embeddings.py
```

### Edge-Drop Variant

```
python compute_role2vec_embeddings.py --drop_rate 25
```

Outputs:

```
role2vec_embeddings.parquet
role2vec_embeddings_25.parquet
```

---

## 7.5 ffstruc2vec

```
python compute_ffstruc2vec_embeddings.py
python compute_ffstruc2vec_embeddings.py --drop_rate 25
```

---

## 7.6 Proximity Embeddings  
(Deterministic Node2Vec & DeepWalk)

```
python compute_proximity_embeddings.py
python compute_proximity_embeddings.py --drop_rate 25
```

Outputs:

```
node2vec_balanced_embeddings.parquet
node2vec_balanced_embeddings_25.parquet
deepwalk_embeddings.parquet
deepwalk_embeddings_25.parquet
```

---

## 7.7 Community Detection (Leiden)

```
python compute_leiden_communities.py
python compute_leiden_communities.py --drop_rate 25
```

---

# 8. Loading Features in `elliptic_data.py`

The loader determines the suffix in the following order:

1. If `variant` is not None:
   suffix = f"_{variant}"
2. Else if `drop_rate` is not None:
   suffix = f"_{drop_rate}"
3. Else:
   suffix = ""


Thus it loads filenames like:

```
degree_centrality.parquet
degree_centrality_25.parquet
role2vec_embeddings_25.parquet
spectral_embeddings_50.parquet
```

Seed never influences loading.

---

# 9. Preprocessing, Splits & Export

## 9.1 Random Split

```
python export_wrapper_random.py

The export wrapper accepts an optional argument:

--seed <int>
--drop_rate <int>
--variant <str>


--drop_rate controls which feature set is loaded (full graph vs. edge-dropped variants).

Examples:

Full graph (no edge drop):
python export_wrapper_random.py --seed 42

Edge-drop 25%:
python export_wrapper_random.py --seed 42 --drop_rate 25

Edge-drop 50%:
python export_wrapper_random.py --seed 42 --drop_rate 50


If neither --drop_rate nor --variant is provided:
→ artifacts with suffix "_edges0"

If --drop_rate is provided:
→ artifacts with suffix "_edges{drop_rate}"

If --variant is provided:
→ artifacts with suffix "_edges{variant}"



All other behavior (naming conventions, suffix generation, and metadata) remains unchanged.
```

## 9.2 Temporal Split

```
python export_wrapper_temporal.py
```

Temporal split definition:

- Train: weeks 1–35  
- Validation: weeks 36–42  
- Test: weeks 43–49  

## 9.3 Final Artifact Naming

Final exported artifacts include:

- Selected features  
- Drop-rate  
- Split seed  
- Random split percentages  

Examples:

```
..._edges0_splitSeed42_random_val20_test20
..._edges25_splitSeed1337_random_val20_test20
```

Note: **Split seed is NOT used in feature generation**, only for dataset splitting.

---

# 10. Reproducibility

The pipeline ensures full reproducibility through:

### Compute scripts  
- Fixed seeds: `SEED = 42`
- Deterministic Word2Vec / Node2Vec runs
- Preserved node ordering
- PYTHONHASHSEED = 42
- Thread limiting (OMP, MKL)

### Edge-drop generation  
- Seed = 42 internally  
- Never included in filenames  

### Final artifacts  
- Store metadata in `meta.json`

---

# 11. Full End-to-End Workflow Examples

## 11.1 Full Graph Pipeline

1. Generate all features:

```
python compute_graph_indicators.py
python compute_proximity_embeddings.py
python compute_spectral_embeddings.py
python compute_graphwave_embeddings.py
python compute_role2vec_embeddings.py
python compute_ffstruc2vec_embeddings.py
```

2. Export artifacts:

```
python export_wrapper_random.py --seed 42
```

---

## 11.2 Edge-Drop Pipeline (Example: 25%)

1. Create edge-dropped Edgelist:

```
python make_edge_subsample.py --drop_percent 25
```

2. Compute features:

```
python compute_graph_indicators.py --drop_rate 25
python compute_spectral_embeddings.py --drop_rate 25
python compute_graphwave_embeddings.py --drop_rate 25
python compute_role2vec_embeddings.py --drop_rate 25
python compute_proximity_embeddings.py --drop_rate 25
python compute_ffstruc2vec_embeddings.py --drop_rate 25
```

3. Export artifacts:

```
python export_wrapper_random.py --seed 42
```

---

# 12. Best Practices

✔ Always activate the correct environment  
✔ Always run compute scripts from `Feature_Generation/`  
✔ Never mix full graph & edge-drop features  
✔ Use deterministic seeds for reproducibility  
✔ Inspect generated `meta.json` for traceability  

---

# 13. Batch-Export

In addition to manually exporting a single artifact via:

python export_wrapper_random.py --seed 42 --drop_rate 25

you can automate the generation of multiple artifacts across a grid of
seed × drop_rate combinations.

Run:

conda activate feature_generation
cd <project_root>\src\Feature_Generation

python run_export_grid.py

---

# 14. Contact

For research use, implementation questions, or reproducibility concerns:

**Mario Heidrich**  
(Feature Generation Pipeline Author)

---

This consolidated README replaces all previous versions and should be used as the authoritative documentation for the entire pipeline.
