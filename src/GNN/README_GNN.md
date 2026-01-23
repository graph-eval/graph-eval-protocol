# Graph Neural Networks (GCN, GAT, GCL)


## Project Structure (Required for Python Imports)

The project must follow the structure below:

```
src/
│
├── GNN/
│   ├── GCN.py
│   ├── GAT.py
│   ├── GCL.py
│   ├── GCN_EndToEnd.py
│   ├── GAT_EndToEnd.py
│   ├── GCL_EndToEnd.py
│
├── Utilities/
│   ├── common_utils.py
│   ├── ...
│
├── artifacts/
│   └── elliptic/
│       └── <artifact_folders>
│
└── Results_.../
```


**Important:**  
The folders `GNN/` and `Utilities/` must be located at the same directory level.  
Only then do imports such as the following work correctly:

```python
from Utilities.common_utils import ...
```

---

## Running a GCN Experiment

Because relative package imports are used, the scripts must be executed in module mode.

### Run a Single GCN Experiment

```
python -m GNN.GCN --artifact <artifact_folder>
```

Example:

```
python -m GNN.GCN --artifact run_2025_11_20_001
```

---

### Run GCN on All Artifacts in artifacts/elliptic

```
python -m GNN.GCN --folder artifacts/elliptic
```

---

## Node Embedding Export

All GNN models (GCN, GAT, GCL) automatically export node embeddings,
analogous to GraphWave or Node2Vec.

**Output location:**

```
<project_root>/src/artifacts/elliptic/gcn_embeddings.parquet
```

The file contains:

- `txId`
- `gcn_emb_0 ... gcn_emb_D`

---

## Using GCN Embeddings with Classical Classifiers

After a GCN run, embeddings can be loaded as follows:

```python
import pandas as pd

emb = pd.read_parquet(
    "<project_root>/src/artifacts/elliptic/gcn_embeddings.parquet"
)
```

They can then be merged with transaction-level features:

```python
df_merged = trx_features.merge(emb, on="txId", how="left")
```

The resulting feature matrix can be used with any classifier
(e.g., XGBoost, Random Forests, MLPs).

---

## Robustness Experiments (Edge-Drop Variants)

In addition to standard runs on the full graph, all GNN models support
robustness evaluation under perturbed graph structures.

Edge-drop experiments are enabled via the optional parameters:

--drop_rate <int>
--seed <int>

When specified, the corresponding edge-dropped edgelist is automatically
loaded following the naming convention:

elliptic_txs_edgelist_<drop_rate>_<seed>.csv

This mechanism mirrors the robustness evaluation strategy used for
feature-based and supervised classification experiments.

### Example: GCN with Edge-Drop

```bash
python -m GNN.GCN --artifact run_2025_11_20_001 --drop_rate 25 --seed 42
```
