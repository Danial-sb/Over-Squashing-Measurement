# Over-Squashing in GNNs and Causal Inference of Rewiring Strategies

[![arXiv](https://img.shields.io/badge/arXiv-2508.09265v1-b31b1b.svg)](https://arxiv.org/abs/2508.09265v1)

Official implementation of the paper:

> **Over-Squashing in GNNs and Causal Inference of Rewiring Strategies**  
> > 🏆 **Accepted at CIKM 2025, Seoul, Republic of Korea**  
> Danial Saber, Amirali Salehi-Abari  
> Ontario Tech University, 2025
>
> ## 🔍 Overview

Graph Neural Networks (GNNs) achieve state-of-the-art performance across domains such as recommendation systems, material design, and drug repurposing.  
However, **message-passing GNNs** suffer from **over-squashing**—the exponential compression of long-range information from distant nodes—which limits their expressivity.

This repository provides:
- A **topology-focused, theoretically grounded metric** for measuring over-squashing based on **decay rates of node-pair sensitivities**.
- **Four graph-level statistics** for characterizing over-squashing:  
  *Prevalence, Intensity, Variability, Extremity.*
- A **causal inference framework** for evaluating the effectiveness of rewiring strategies.
- Implementations of **rewiring baselines**: FoSR, DIGL, SDRF, BORF, and GTR.
- Extensive experiments on **graph and node classification benchmarks**, enabling diagnosis of when rewiring is beneficial.

Our plug-and-play diagnostic tool lets practitioners decide—*before training*—whether rewiring is likely to pay off.

---
> ## Getting Started

- For reproducing all the results, run automated_graph_level_evaluation() or automated_node_level_evaluation() in `Treatment_Effects.py` file. 
---

> ## Citing Us/BibTex
Please cite our work if you find it useful in any way.

```
@article{saber2024scalable,
  title={Scalable Expressiveness through Preprocessed Graph Perturbations},
  author={Saber, Danial and Salehi-Abari, Amirali},
  journal={arXiv preprint arXiv:2406.11714},
  year={2024}
}
```
