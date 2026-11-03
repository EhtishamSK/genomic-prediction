# Genomic Prediction & Multi-Locus Association Pipeline

A modular, reproducible codebase for genomic selection, cross-validation data splitting. This repository integrates classical linear models, tree-based machine learning algorithms, and deep neural network architectures to evaluate predictive accuracy across complex trait datasets.

---

## Repository Structure

```text
├── data_split.R             # Proportional stratified sampling for training/validation splits
├── data_split_subpop.R      # Population structure-aware partitioning (e.g., K-means + performance tiers)
├── dnngp_ensemble.py        # Deep Neural Network Genomic Prediction (DNN-GP) ensemble pipeline
├── lgbm_ensemble.py         # LightGBM gradient-boosting machine learning prediction model
├── svr_ensemble.py          # Support Vector Regression (SVR) kernel-based model pipeline
├── deepgs_ensemble.py       # Deep Genomic Selection (DeepGS) marker-effect pipeline
├── deepgwas_ensemble.py     # Deep learning-based multi-locus GWAS architecture
├── datasplit/               # Generated partition outputs, cross-validation folds, and subset tables
└── train-test-pop/          # training and testing population 
```

---

## Pipeline Overview

### 1. Data Partitioning & Stratification (`R/`)
Ensures balanced representation of genetic diversity across evaluation folds to prevent overfitting and structure bias.
- **`data_split.R`**: Executes proportional stratified sampling to separate phenotypic observations into symmetric training and validation cohorts.
- **`data_split_subpop.R`**: Integrates population structure clustering with historical performance tiers to maintain subpopulation allele frequencies across splits.

### 2. Machine Learning & Deep Learning Architectures (`Python/`)
Implements diverse predictive algorithms designed for high-dimensional genomic data:
- **`dnngp_ensemble.py`**: Custom deep learning framework utilizing dense neural network topologies for multi-environment trait prediction.
- **`lgbm_ensemble.py`**: Fast, tree-based gradient boosting tailored for capturing non-linear marker interactions and epistatic effects.
- **`svr_ensemble.py`**: Kernel-based regression model optimized for high-dimensional marker matrices ($p > n$).
- **`deepgs_ensemble.py`**: Deep learning pipeline structured specifically for genomic selection and breeding value estimation.
- **`deepgwas_ensemble.py`**: Advanced network architecture designed to map and validate marker-trait associations alongside genomic prediction tasks.

---
