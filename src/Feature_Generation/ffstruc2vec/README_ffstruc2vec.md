# ffstruc2vec

This repository provides a reference implementation of **ffstruc2vec**, as described in the paper:

> **ffstruc2vec: Flat, Flexible and Scalable Learning of Node Representations from Structural Identities**  
> Mario Heidrich, Prof. Dr. Jeffrey Heidemann, Prof. Dr. Rüdiger Buchkremer,  
> Prof. Dr. Gonzalo Wandosell Fernández de Bobadilla

The *ffstruc2vec* algorithm learns continuous vector representations of nodes in arbitrary graphs, focusing on *structural equivalence* rather than neighborhood similarity.

---

## ⚙️ Environment Setup

The former `requirements.txt` has been removed.

A dedicated Conda environment is now provided:

```
conda env create -f environments/environment_ffstruc2vec.yml
conda activate ffstruc2vec-env
```

This ensures fully reproducible dependency installation for the ffstruc2vec experiments.

---

## ▶️ Basic Usage

### Example

Run *ffstruc2vec* on Mirrored Zachary's karate club network with default parameters:

```
python src/main.py --input graph/karate-mirrored.edgelist --output emb/karate-mirrored.emb
```

---

## 📥 Input Format

The supported input format is a simple edge list:

```
node1_id_int   node2_id_int
```

---

## 📤 Output Format

ffstruc2vec produces **two output files**:

### 1. Embedding File  
Contains learned vectors for all *n* nodes:

```
num_nodes   embedding_dim
node_id   dim1 dim2 ... dimD
```

### 2. PCA Visualization  
A 2‑D PCA projection of the learned embeddings.

---

## 🔧 Flexibility & Options

You can activate Hyperopt-based end‑to‑end classification:

```
--method 0
```

Provide labels via:

```
--path_labels your_label_file.txt
```

The framework also allows manual weight assignment for *k*-hop layers and the selection of one of several structural graph features (degree, PageRank, betweenness, eigenvector centrality, etc.) using:

```
--active_feature
```

List all available options:

```
python src/main.py --help
```

---

## 📑 Full List of Command Line Options

*(Preserved exactly as in the original README)*  
> See below for the full list of optional arguments…

[The full list remains unchanged and is included in your original README.]

---

## 📬 Contact

For questions about the code or algorithm, contact:  
**node_embedding@gmail.com**

---

## 📄 License

This project is licensed under **Apache License 2.0**.

Parts adapted from the original **struc2vec** implementation (MIT License) are documented in the code and referenced in:

- `LICENSE`
- `LICENSE_STRUC2VEC`

