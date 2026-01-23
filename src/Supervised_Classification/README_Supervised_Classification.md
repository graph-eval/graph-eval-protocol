# Supervised Classification

This directory implements **Stage 5 (Controlled Supervised Classification)**
of the evaluation protocol.

It contains all scripts and resources required to perform supervised
classification experiments on the Elliptic dataset. The structure is modular,
supports both single and batch executions, and stores all results and artifacts
in a fully reproducible manner.

---

## Environment Setup

All experiments are executed in a Conda environment defined by:

environment_supervised_classification.yml

---

## Artifacts (Input Data)

All classifiers operate on precomputed feature artifacts stored as Parquet files:

```
src/
└── artifacts/
    └── elliptic/
        ├── <ARTIFACT_NAME>/
        │   ├── X_train.parquet
        │   ├── X_validation.parquet
        │   ├── X_test.parquet
        │   ├── y_train.parquet
        │   ├── y_validation.parquet
        │   └── y_test.parquet
```


These artifacts must exist prior to execution.
The function `has_required_files()` automatically checks file completeness.

---

## Running an Experiment

### 1. Run a Single Artifact

Example:

```bash
python GB.py --artifact 2025_01_elliptic_window32

```

The script automatically loads:

- X_train.parquet  
- X_validation.parquet  
- X_test.parquet  
- y_train.parquet  
- y_validation.parquet  
- y_test.parquet  

and performs:

- Hyperparameter optimization (Hyperopt)
- Best-iteration selection via validation LogLoss
- Final refit on TRAIN + VALIDATION
- Feature importance computation (permutation + intrinsic)
- Threshold optimization (F1@t*)

All outputs are stored under:

```
Results_.../.../
    elliptic_XGB_<ARTIFACT_NAME>.json
    elliptic_XGB_<ARTIFACT_NAME>__perm.parquet
    elliptic_XGB_<ARTIFACT_NAME>__intr.parquet
    elliptic_XGB_<ARTIFACT_NAME>__trials.joblib
    elliptic_XGB_<ARTIFACT_NAME>__bundle.joblib
```

---

## Fixed-Hyperparameter Classification

In addition to the standard pipelines, fixed-hyperparameter variants
(*_fix_hp.py) are provided.

### Procedure
For each edge-drop artifact:

1. The corresponding full-graph model bundle is identified.
2. Stored best_params are reused.
3. No hyperparameter optimization is performed.
4. The decision threshold t* is re-estimated on the validation split
   of the edge-drop artifact.
5. The model is refit on TRAIN + VALIDATION.
6. Evaluation is performed on TEST.

### Result Storage
Results are stored separately from hyperparameter-optimized runs under:

Results_HyPa_fix/
└── Results_<drop>_<seed>/
    ├── LR/
    ├── MLP/
    ├── NB/
    ├── RF/
    ├── SVC/
    └── XGB/


---

## Loading a Trained Model Bundle

For downstream inference, a serialized bundle is generated automatically:

```
elliptic_XGB_<ARTIFACT>__bundle.joblib
```

The bundle contains:

- trained model
- optimal decision threshold t*
- feature names
- fraud-class index
- best iteration 

Example:

```python
import joblib

bundle = joblib.load("elliptic_XGB_<ARTIFACT>__bundle.joblib")
clf = bundle["model"]
t_star = bundle["threshold_t_star"]
feature_names = bundle["feature_names"]

proba = clf.predict_proba(X_new)
fraud_idx = bundle["fraud_proba_index"]
pred = (proba[:, fraud_idx] >= t_star).astype(int)
```

---

### Batch Execution over All Artifacts

```
cd <project-folder>/src
python -m Supervised_Classification.NB --folder artifacts/elliptic
```

---

## Output Files

### JSON Summary  
Metrics and run metadata.

### Parquet Files
- permutation feature importance
- intrinsic feature importance

### Joblib  
- Hyperopt trials
- serialized model bundle

---

## Common Commands (Summary)

```
# Single Run
python XGB.py --artifact <NAME>

# Batch Run
python XGB.py --folder ../artifacts/elliptic --jobs 4
```

---

## Notes

This setup is fully modular and suitable for:

- reproducible ML pipelines
- hyperparameter optimization
- feature importance analysis
- deployment via serialized model bundles

Additional models or pipelines can be added by implementing a script following the same structure.

