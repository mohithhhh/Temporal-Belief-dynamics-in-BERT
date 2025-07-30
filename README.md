# Temporal-Belief-dynamics-in-BERT

This repository contains the code and results for the research project *Temporal Belief Dynamics in BERT: A Graph-Theoretic Approach to Interpretable Representation Drift* by Mohith D K (PES University, Bangalore - 500085, India). The project analyzes how BERT's internal representations evolve during fine-tuning on the SST-2 dataset using graph-theoretic methods.

## Overview
- **Objective**: Track, measure, and interpret representation drift in BERT over 10 training epochs using cosine similarity, similarity graphs, and UMAP visualizations.
- **Dataset**: Stanford Sentiment Treebank 2 (SST-2) subset (1000 training, 200 validation samples).
- **Environment**: Developed in Kaggle Notebooks and Google Colab with NVIDIA Tesla T4 GPUs.

## Tasks and Results

### Task 1: Cosine Similarity Drift
- **Description**: Quantified representation drift by calculating cosine similarity between pre-trained and epoch-wise CLS embeddings.
- **Implementation**: Used the formula \( D^{(t)} = \frac{1}{M} \sum_{i=1}^M (1 - \cos(\mathrm{H}_i^{(0)}, \mathrm{H}_i^{(t)})) \) across 200 validation samples.
- **Results**:
  - Generated `cosine_similarity_drift.png` (Figure 1): Shows a decrease from 0.37 (Epoch 1) to 0.04 (Epoch 10).
  - Table 1: Lists drift values (e.g., Epoch 1: 0.3511, Epoch 2: 0.2741, etc.).

### Task 2: Similarity Graph Construction and Structural Drift
- **Description**: Constructed similarity graphs with nodes as CLS embeddings and edges based on cosine similarity thresholds, measuring structural changes with Graph Edit Distance (GED) and Node Centrality Drift.
- **Implementation**: Applied GED and degree centrality analysis on epoch-wise graphs.
- **Results**:
  - Generated `graph_edit_distance.png`: Displays GED values (e.g., Epoch 1-2: 305308.7670, Epoch 9-10: 19978.8).
  - Table 2: GED transitions across epochs.
  - Generated `node_centrality_drift.png` (Figure 3): Shows centrality changes aligning with GED trends.
  - Table 3: Centrality drift values (e.g., Epoch 1: 0.3231, Epoch 10: 0.2031).

### Task 3: UMAP Manifold and Cluster Evolution
- **Description**: Visualized CLS embeddings in 2D using UMAP and tracked cluster evolution with K-Means.
- **Implementation**: Projected embeddings and computed centroid shifts across epochs.
- **Results**:
  - Generated `umap_projection.png` (Figure 4): Side-by-side comparison of pre-trained vs. post-fine-tuned embeddings, showing distinct sentiment clusters.
  - Identified increasing coherence in positive and negative subgroups.

### Task 4: Edge Transition Analysis
- **Description**: Analyzed edge additions and removals in similarity graphs to map structural evolution.
- **Implementation**: Compared consecutive graphs and visualized transitions with color-coded edges (green for added, blue for removed).
- **Results**:
  - Generated `edge_transition_analysis.png` (Figure 5): Highlights high edge changes in early epochs, stabilizing later.
  - Noted matching plateau with GED trends.

### Task 5: Layer-Wise Representation Drift
- **Description**: Measured drift across BERT’s 12 layers to assess hierarchical changes.
- **Implementation**: Calculated cosine similarity changes for hidden states per layer.
- **Results**:
  - Generated `layer_wise_drift.png` (Figure 6): Shows greater drift in upper layers (e.g., >10, indicating opposing shifts).
  - Table 4: Detailed layer-wise drift values.

### Task 6: Class-Wise Representation Drift
- **Description**: Analyzed drift separately for positive and negative sentiment classes.
- **Implementation**: Computed class-specific cosine drift on validation samples.
- **Results**:
  - Generated `class_wise_drift.png` (Figure 7): Indicates higher drift for negative class, suggesting context-dependent adjustments.
  - Noted class-specific decision boundary evolution.

## Directory Structure
- `data/`: Raw SST-2 TSV files (train.tsv, test.tsv, dev.tsv).
- `notebooks/`: Jupyter notebook (e.g., bertt-2.ipynb) with implementation.
- `outputs/`: Generated plots (e.g., cosine_similarity_drift.png), metrics (e.g., tables), and `animation.gif`.
- `docs/`: Documentation (this README.md).

## Setup
- **Dependencies**: Install required packages:
  ```bash
  pip install transformers torch numpy pandas matplotlib umap-learn scipy networkx scikit-learn
