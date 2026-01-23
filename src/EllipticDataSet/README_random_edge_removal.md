# Random Edge Removal (Elliptic Dataset)

This directory contains the script used to generate random edge-removal variants
of the Elliptic Bitcoin transaction graph, as used in the robustness experiments
reported in the accompanying paper.

The purpose of these variants is to assess the stability of graph-derived signals
and downstream classification performance under incomplete or perturbed graph
observations.

## Methodology

Edges are removed uniformly at random from the original transaction graph.
For a given removal rate, a fixed random seed is used to ensure reproducibility.
The remaining edges are written to a new edge list file, which can then be used
to recompute graph-derived signals and rerun the downstream experiments.

In the paper, the following random edge-removal rates are considered:
- **25 %**
- **50 %**

## Script

The edge-removal variants are generated using the following script:

- `make_edge_subsample.py`

This script takes an input edge list in CSV format and randomly removes a
user-specified percentage of edges.

## Usage

### Random edge removal (25%)

```bash
python make_edge_subsample.py \
  --input elliptic_txs_edgelist.csv \
  --drop_percent 25
